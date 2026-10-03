"""
The three Labyrinth models used in the Deep Graph InfoMax loop, built from one config.

    NodeEncoder       chess graph (64 nodes, 8 edge types)  ->  64 node embeddings          [64, node_dim]
    GlobalSummarizer  chess graph                           ->  one summary vector          [summary_dim]
    Discriminator     (node embedding, summary)             ->  score, or a full score matrix for InfoNCE

Every architectural choice lives in DGIConfig so that DEHB can search over it: how many
unpooling layers the summarizer stacks, how wide the MLPs inside each unpooling layer are,
how wide the GAT that follows each unpooling layer is, how the enlarged graph is flattened,
and how large the final summary vector is. build_architecture_configspace() describes that
search space and build_models() turns a sampled configuration into the three models.

The training loop is not here. What is here is designed so that the loop can be written
without touching these classes: the summarizer reports the log-probability and entropy of
the unpooling decisions it sampled (for a REINFORCE / PPO term, since those decisions are
discrete) and can replay a recorded set of decisions exactly (for PPO's ratio), the encoder
can process a list of positions as one disconnected graph, and the discriminator can score
every node against every summary in one call.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

import chess_graph as cg
import guo_et_al_unpooling as unpool


# ##### ##### ##### ##### #####
#       Configuration


@dataclass
class DGIConfig:
    """
    Every knob for the three models. Field names are the ConfigSpace hyperparameter names,
    so a dict sampled by DEHB can be turned into one of these with DGIConfig.from_dict.
    Fields not present in a sampled dict keep their defaults, which lets the search space
    cover a subset of the knobs.
    """
    # ----- Graph (fixed by chess_graph) -----
    node_feature_dim: int = cg.NUM_NODE_FEATURES
    num_edge_types: int = cg.NUM_EDGE_TYPES
    num_squares: int = 64

    # ----- NodeEncoder: the local encoder -----
    enc_num_hops: int = 3          # GAT layers; each hop lets a square hear from one more ring of neighbors
    enc_hidden_dim: int = 128      # width of every hop except the last
    enc_heads: int = 4             # attention heads per hop (averaged, not concatenated)
    enc_dropout: float = 0.1
    node_dim: int = 512            # width of the final node embedding, and of the discriminator's node input

    # ----- GlobalSummarizer -----
    sum_in_dim: int = 64           # node features are lifted to this width before the first unpooling layer
    sum_num_unpool: int = 2        # how many unpooling layers to stack; the graph can double in size at each
    sum_unpool_hidden: int = 128   # hidden width of the MLPs inside each unpooling layer (kv, kia, kie, kw)
    sum_edge_dim: int = 16         # width of the edge features the unpooling layers produce and consume
    sum_gat_heads: int = 4         # heads of the GAT that refines features after each unpooling layer
    sum_gat_dropout: float = 0.0
    sum_flat_dim: int = 16         # each node is projected to this width before the graph is flattened
    sum_mlp_hidden: int = 1024     # width of the MLP that reads the flattened graph
    sum_mlp_layers: int = 2        # number of hidden layers in that MLP
    summary_dim: int = 4096        # the final summary vector; meant to be much larger than node_dim

    # ----- Discriminator -----
    disc_bilinear_maps: int = 8    # number of bilinear forms combined by the scoring MLP
    disc_hidden: int = 128

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "DGIConfig":
        """Build a config from a (possibly partial) dict such as a DEHB sample; unknown keys are ignored."""
        names = {f.name for f in dataclasses.fields(cls)}
        clean = {k: v for k, v in values.items() if k in names}
        # ConfigSpace hands back numpy scalars; normalize so the fields hold plain Python numbers
        for k, v in list(clean.items()):
            if hasattr(v, "item"):
                clean[k] = v.item()
        return cls(**clean)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    # ----- Derived sizes -----
    @property
    def max_unpooled_nodes(self) -> int:
        """Every node may split in two at every layer, so this bounds the enlarged graph."""
        return self.num_squares * (2 ** self.sum_num_unpool)

    @property
    def flattened_size(self) -> int:
        """Length of the flattened graph vector that feeds the summarizer's MLP."""
        return self.max_unpooled_nodes * self.sum_flat_dim

    def validate(self) -> None:
        if self.sum_in_dim < 4:
            raise ValueError("sum_in_dim must be at least 4: the unpooling layer projects features to "
                             "floor(d/2) + floor(d/4) dimensions and needs both parts to be non-empty")
        if self.sum_num_unpool < 1:
            raise ValueError("sum_num_unpool must be at least 1")
        if self.enc_num_hops < 1:
            raise ValueError("enc_num_hops must be at least 1")


def _mlp(sizes: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
    """Linear -> LayerNorm -> LeakyReLU (-> Dropout) between every pair of sizes; the last layer is linear."""
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.LayerNorm(sizes[i + 1]), nn.LeakyReLU(0.05)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


# ##### ##### ##### ##### #####
#       Chess graph -> single graph


def chess_graph_to_single(
    x: torch.Tensor,
    edge_index_list: Sequence[torch.Tensor],
    num_edge_types: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merge the per-type edge lists from chess_graph.create_filled_chess_graphs into one
    edge_index with a one-hot edge type as the edge feature.

    x: [N, node_feature_dim]; edge_index_list: one [2, E_i] tensor per edge type.
    Returns (x, edge_index [2, E], edge_attr [E, num_edge_types]).

    A pair of squares joined by several edge types (e2-e3 is both a pawn move and a king
    move) keeps one edge per type, each with its own one-hot, so the GAT and the unpooling
    layers see one message per relation rather than a multi-hot blend.
    """
    if num_edge_types is None:
        num_edge_types = len(edge_index_list)
    if len(edge_index_list) != num_edge_types:
        raise ValueError(f"expected {num_edge_types} edge lists, got {len(edge_index_list)}")
    device, dtype = x.device, x.dtype
    parts_index: List[torch.Tensor] = []
    parts_attr: List[torch.Tensor] = []
    for i, ei in enumerate(edge_index_list):
        if ei.numel() == 0:
            continue
        ei = ei.to(device)
        parts_index.append(ei)
        parts_attr.append(
            F.one_hot(torch.full((ei.size(1),), i, dtype=torch.long, device=device), num_edge_types).to(dtype)
        )
    if not parts_index:
        return x, torch.empty(2, 0, dtype=torch.long, device=device), torch.empty(0, num_edge_types, dtype=dtype, device=device)
    return x, torch.cat(parts_index, dim=1), torch.cat(parts_attr, dim=0)


def batch_chess_graphs(
    graphs: Sequence[Tuple[Sequence[torch.Tensor], torch.Tensor]],
    num_edge_types: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stack several positions into one disconnected graph, the way PyG batches graphs.

    graphs: a sequence of (edge_index_list, x) pairs exactly as create_filled_chess_graphs
    returns them. Returns (x [B*64, F], edge_index [2, E], edge_attr [E, T], batch [B*64])
    where batch[i] is the index of the position node i belongs to.
    """
    xs, eis, eas, batch = [], [], [], []
    offset = 0
    for b, (edge_index_list, x) in enumerate(graphs):
        x, ei, ea = chess_graph_to_single(x, edge_index_list, num_edge_types)
        xs.append(x)
        eis.append(ei + offset)
        eas.append(ea)
        batch.append(torch.full((x.size(0),), b, dtype=torch.long, device=x.device))
        offset += x.size(0)
    return torch.cat(xs), torch.cat(eis, dim=1), torch.cat(eas), torch.cat(batch)


# ##### ##### ##### ##### #####
#       NodeEncoder (local encoder)


class NodeEncoder(nn.Module):
    """
    Encodes the 64 squares into node embeddings with a stack of GAT layers over the merged
    chess graph. Edge types enter as one-hot edge features, so attention can weigh a knight
    relation differently from a pawn attack. After enc_num_hops layers every square has
    heard from every other square along several paths.
    """

    def __init__(self, config: DGIConfig):
        super().__init__()
        config.validate()
        self.config = config
        widths = [config.node_feature_dim] + [config.enc_hidden_dim] * (config.enc_num_hops - 1) + [config.node_dim]
        self.layers = nn.ModuleList(
            GATConv(
                widths[i], widths[i + 1],
                heads=config.enc_heads, concat=False,
                dropout=config.enc_dropout, edge_dim=config.num_edge_types,
            )
            for i in range(config.enc_num_hops)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(widths[i + 1]) for i in range(config.enc_num_hops - 1))

    @property
    def node_dim(self) -> int:
        return self.config.node_dim

    def forward_merged(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Encode an already merged graph (possibly a batch of positions stacked with batch_chess_graphs)."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_attr)
            if i < len(self.layers) - 1:
                h = F.leaky_relu(self.norms[i](h), 0.05)
        return h

    def forward(self, x: torch.Tensor, edge_index_list: Sequence[torch.Tensor]) -> torch.Tensor:
        """x: [64, node_feature_dim], edge_index_list from create_filled_chess_graphs. Returns [64, node_dim]."""
        return self.forward_merged(*chess_graph_to_single(x, edge_index_list, self.config.num_edge_types))


# ##### ##### ##### ##### #####
#       GlobalSummarizer


@dataclass
class SummaryOutput:
    """
    What the summarizer returns. Beyond the summary itself, everything the training loop
    needs to apply a policy-gradient update to the unpooling decisions:

    summary     [summary_dim]  the position summary
    log_prob    scalar         log-probability of every unpooling decision that was sampled, summed
    entropy     scalar         entropy of those decisions, summed (for an entropy bonus)
    num_nodes   int            size of the enlarged graph before padding
    actions     list           one record per unpooling layer; pass back as actions_to_replay
                               to reproduce exactly the same enlarged graph under new parameters
    """
    summary: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    num_nodes: int
    actions: List[Dict[str, Any]]


class GlobalSummarizer(nn.Module):
    """
    Summarizes a whole position into one large vector, following the structure of the
    unpooling paper's generator:

        lift node features to sum_in_dim
        repeat sum_num_unpool times:
            unpooling layer      (each node may split in two; structure and features are learned)
            GAT                  (refines the features of the enlarged graph)
        project every node to sum_flat_dim
        zero-pad to the largest graph the stack can produce, flatten
        MLP -> summary_dim

    Flattening rather than pooling keeps every node's contribution in a fixed slot, so no
    information is averaged away; the price is that the MLP input is
    64 * 2**sum_num_unpool * sum_flat_dim wide, which the config exposes as flattened_size.

    The unpooling decisions are sampled, so the enlarged graph differs from call to call
    unless actions_to_replay is given. Their log-probability and entropy are returned so
    the training loop can update them with REINFORCE or PPO; the feature computations
    (the MLPs inside the unpooling layers, the GATs, the head) are ordinary differentiable
    modules and receive gradients directly from the loss.
    """

    def __init__(self, config: DGIConfig):
        super().__init__()
        config.validate()
        self.config = config
        c = config

        self.lift = _mlp([c.node_feature_dim, c.sum_in_dim, c.sum_in_dim])

        self.unpool_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        edge_in = c.num_edge_types  # the first layer consumes the one-hot edge types
        for _ in range(c.sum_num_unpool):
            self.unpool_layers.append(
                unpool.GuoUnpool(
                    dx=c.sum_in_dim, dw=edge_in, dy=c.sum_in_dim, du=c.sum_edge_dim,
                    kv=c.sum_unpool_hidden, kia=c.sum_unpool_hidden,
                    kie=c.sum_unpool_hidden, kw=c.sum_unpool_hidden,
                )
            )
            self.gat_layers.append(
                GATConv(
                    c.sum_in_dim, c.sum_in_dim, heads=c.sum_gat_heads, concat=False,
                    dropout=c.sum_gat_dropout, edge_dim=c.sum_edge_dim,
                )
            )
            self.gat_norms.append(nn.LayerNorm(c.sum_in_dim))
            edge_in = c.sum_edge_dim

        self.to_flat = nn.Linear(c.sum_in_dim, c.sum_flat_dim)
        self.head = _mlp([c.flattened_size] + [c.sum_mlp_hidden] * c.sum_mlp_layers + [c.summary_dim])

    @property
    def summary_dim(self) -> int:
        return self.config.summary_dim

    @property
    def max_nodes(self) -> int:
        return self.config.max_unpooled_nodes

    def forward(
        self,
        x: torch.Tensor,
        edge_index_list: Sequence[torch.Tensor],
        rng: Optional[torch.Generator] = None,
        actions_to_replay: Optional[List[Dict[str, Any]]] = None,
    ) -> SummaryOutput:
        """
        x: [64, node_feature_dim]; edge_index_list from create_filled_chess_graphs.
        rng: generator for the unpooling decisions (reproducibility).
        actions_to_replay: the `actions` of a previous SummaryOutput for this same position,
            to rebuild that exact enlarged graph (needed for PPO's probability ratio).
        """
        x, edge_index, edge_attr = chess_graph_to_single(x, edge_index_list, self.config.num_edge_types)
        h = self.lift(x)

        log_prob = h.new_zeros(())
        entropy = h.new_zeros(())
        actions: List[Dict[str, Any]] = []
        for k, (unpool_layer, gat, norm) in enumerate(zip(self.unpool_layers, self.gat_layers, self.gat_norms)):
            replay = actions_to_replay[k] if actions_to_replay is not None else None
            h, edge_index, edge_attr, lp, ent, _parent_map, _sets, recorded = unpool_layer(
                h, edge_index, edge_attr, actions_to_replay=replay, rng=rng
            )
            log_prob = log_prob + lp
            entropy = entropy + ent
            actions.append(recorded)
            h = F.leaky_relu(norm(gat(h, edge_index, edge_attr)), 0.05)

        num_nodes = h.size(0)
        if num_nodes > self.max_nodes:  # cannot happen (each layer at most doubles), guard anyway
            raise RuntimeError(f"unpooled graph has {num_nodes} nodes, above the bound {self.max_nodes}")

        flat = self.to_flat(h)                                        # [num_nodes, flat_dim]
        padded = flat.new_zeros(self.max_nodes, flat.size(1))
        padded[:num_nodes] = flat
        summary = self.head(padded.reshape(-1))                       # [summary_dim]
        return SummaryOutput(summary, log_prob, entropy, num_nodes, actions)


# ##### ##### ##### ##### #####
#       Discriminator


class Discriminator(nn.Module):
    """
    Scores how well a node embedding matches a position summary. A set of bilinear forms
    compares the two vectors and a small MLP combines those scores non-linearly. In the
    DGI loop the score is used as an InfoNCE logit: for each summary, the node embeddings
    of its own position should outscore node embeddings drawn from perturbed positions.
    """

    def __init__(self, config: DGIConfig):
        super().__init__()
        self.config = config
        self.W = nn.Parameter(torch.randn(config.disc_bilinear_maps, config.node_dim, config.summary_dim) * 0.02)
        self.mlp = _mlp([config.disc_bilinear_maps, config.disc_hidden, 1])

    def forward(self, node_embeddings: torch.Tensor, summaries: torch.Tensor) -> torch.Tensor:
        """Paired scoring. node_embeddings [M, node_dim], summaries [M, summary_dim] -> [M]."""
        bilinear = torch.einsum("md,kds,ms->mk", node_embeddings, self.W, summaries)
        return self.mlp(bilinear).squeeze(-1)

    def score_matrix(self, node_embeddings: torch.Tensor, summaries: torch.Tensor) -> torch.Tensor:
        """
        Every node against every summary, for InfoNCE.
        node_embeddings [M, node_dim], summaries [B, summary_dim] -> [M, B], where entry
        (m, b) is the score of node m against summary b. Row-wise softmax over B ranks the
        summaries for a node; column-wise over M ranks the nodes for a summary.
        """
        projected = torch.einsum("md,kds->mks", node_embeddings, self.W)          # [M, K, S]
        bilinear = torch.einsum("mks,bs->mbk", projected, summaries)              # [M, B, K]
        return self.mlp(bilinear).squeeze(-1)                                     # [M, B]


# ##### ##### ##### ##### #####
#       Building from a config, and the DEHB search space


def build_models(config: DGIConfig) -> Tuple[NodeEncoder, GlobalSummarizer, Discriminator]:
    """The three models for one configuration. node_dim and summary_dim tie them together."""
    return NodeEncoder(config), GlobalSummarizer(config), Discriminator(config)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def describe(config: DGIConfig) -> Dict[str, Any]:
    """Sizes worth looking at before committing to a configuration."""
    enc, summ, disc = build_models(config)
    return {
        "encoder_params": count_parameters(enc),
        "summarizer_params": count_parameters(summ),
        "discriminator_params": count_parameters(disc),
        "max_unpooled_nodes": config.max_unpooled_nodes,
        "flattened_size": config.flattened_size,
        "summary_dim": config.summary_dim,
    }


def build_architecture_configspace(seed: int = 0):
    """
    The DEHB search space over the architecture. Training hyperparameters (learning rate,
    batch size, the PPO knobs for the unpooling decisions) belong to the training loop and
    will be added to this space, or a companion one, when that loop exists.

    Watch flattened_size: sum_num_unpool=5 with sum_flat_dim=32 gives a 65,536 wide MLP
    input. Use DGIConfig.flattened_size or describe() in the objective to skip or penalize
    configurations that are too large for the budget.
    """
    import ConfigSpace as CS
    cs = CS.ConfigurationSpace(seed=seed)
    # NodeEncoder
    cs.add_hyperparameter(CS.UniformIntegerHyperparameter("enc_num_hops", lower=2, upper=4))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("enc_hidden_dim", choices=[64, 128, 256]))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("enc_heads", choices=[1, 2, 4, 8]))
    cs.add_hyperparameter(CS.UniformFloatHyperparameter("enc_dropout", lower=0.0, upper=0.3))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("node_dim", choices=[256, 512, 1024]))
    # GlobalSummarizer
    cs.add_hyperparameter(CS.CategoricalHyperparameter("sum_in_dim", choices=[16, 32, 64, 128]))
    cs.add_hyperparameter(CS.UniformIntegerHyperparameter("sum_num_unpool", lower=1, upper=5))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("sum_unpool_hidden", choices=[64, 128, 256, 512]))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("sum_edge_dim", choices=[8, 16, 32]))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("sum_gat_heads", choices=[1, 2, 4]))
    cs.add_hyperparameter(CS.UniformFloatHyperparameter("sum_gat_dropout", lower=0.0, upper=0.2))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("sum_flat_dim", choices=[4, 8, 16, 32]))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("sum_mlp_hidden", choices=[512, 1024, 2048]))
    cs.add_hyperparameter(CS.UniformIntegerHyperparameter("sum_mlp_layers", lower=1, upper=3))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("summary_dim", choices=[1024, 2048, 4096]))
    # Discriminator
    cs.add_hyperparameter(CS.CategoricalHyperparameter("disc_bilinear_maps", choices=[4, 8, 16]))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("disc_hidden", choices=[64, 128, 256]))
    return cs


# ----- ----- -----
# Program Body

if __name__ == "__main__":
    import json

    fen = "1q1rkr2/pp3pnp/2pn2pQ/3p4/3Pb3/2P2NP1/PP2P2P/3RKRNB b KQkq - 1 15"
    edges, x = cg.create_filled_chess_graphs(fen)

    config = DGIConfig()
    print("Default configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    print("\nSizes:")
    print(json.dumps(describe(config), indent=2))

    encoder, summarizer, discriminator = build_models(config)
    rng = torch.Generator().manual_seed(0)

    h = encoder(x, edges)
    out = summarizer(x, edges, rng=rng)
    replay = summarizer(x, edges, actions_to_replay=out.actions)
    scores = discriminator.score_matrix(h, out.summary.unsqueeze(0))

    print(f"\nnode embeddings     {tuple(h.shape)}")
    print(f"summary             {tuple(out.summary.shape)}  from {out.num_nodes} unpooled nodes "
          f"(bound {summarizer.max_nodes}), log_prob={out.log_prob.item():.2f}, entropy={out.entropy.item():.2f}")
    print(f"replay identical    {torch.allclose(replay.summary, out.summary)}")
    print(f"score matrix        {tuple(scores.shape)}")
