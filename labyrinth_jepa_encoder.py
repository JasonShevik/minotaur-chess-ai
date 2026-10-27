"""
The Labyrinth position encoder, trained as a JEPA (joint-embedding predictive architecture).

One encoder turns a chess position into a position vector z. A second copy of it, the target
encoder, is a slow moving average of the first and receives no gradients; its outputs are the
answers the predictors are graded against. Two predictors give the training signal:

    masked squares   The encoder sees the board with some squares hidden (labyrinth_data builds
                     that masked view leak-free). A small GAT on the chess graph predicts what the
                     target encoder, seeing the whole board, makes of each hidden square.
    next positions   From z of a position and a move (the from/to square tokens, the promotion
                     piece and the move flags), predict the target encoder's z of the position the
                     move leads to; then, from that prediction and the next move, the position after
                     that, and so on: a rollout in latent space, with a loss at every step. Moves are
                     always described with the STARTING position's square tokens, so the rollout never
                     sees an intermediate position, only predicts it. A curriculum lengthens it: one
                     move ahead at first, one more every rollout_stage_positions training positions, up
                     to rollout_max_steps (see rollout_schedule).

The encoder:

    squares    the 64 square features are embedded (hidden squares get a learned mask token) and
               refined by gat_layers GAT layers on the chess graph: 64 square tokens
    extras     token_variant decides what joins the square tokens:
                 "squares"    nothing
                 "registers"  num_registers learned tokens, the same for every position
                 "unpool"     the nodes of an enlarged graph: unpool_layers Guo et al. unpooling
                              layers, each followed by a GAT, grown from the square tokens
    transformer  all tokens of a position attend to each other (no positional encodings: tokens
               are an unordered set, so the unpooled nodes' arbitrary order cannot matter)
    readout    readout_queries learned queries attend over the final tokens; their outputs are
               concatenated and projected to z (z_dim)

Spreading the embeddings. Without negatives, nothing in the prediction losses makes different
positions get different vectors: the untrained encoder already maps every position to nearly the same
z, and the predictors then win by learning that shared vector. An anti-collapse term on z fixes this:
  "sigreg"  LeJEPA's SIGReg (Balestriero & LeCun, 2025), the default: random 1-D projections of the
            batch of embeddings are each tested against a standard normal (Epps-Pulley statistic), which
            pushes the embeddings towards an isotropic Gaussian, with no shared component and every
            direction used;
  "vicreg"  VICReg's variance and covariance terms: every dimension keeps a spread of at least 1 across
            the batch, and dimensions are decorrelated.

The three variants share everything else, so comparing them answers whether the unpooling layers
earn their place. The unpooling decisions are sampled everywhere, in the target encoder too; their
log-probability is returned per position and trained with REINFORCE (reward: minus that
position's prediction loss), with a learned baseline.

Everything is in JEPAConfig, and build_jepa_configspace() is the DEHB search space over it. The
DEHB objective is LiDAR (Thilak et al., ICLR 2024): the effective rank of the linear discriminant
matrix whose classes are positions and whose members are several random encodings of each
position. It weighs every direction of the embedding by how much positions differ along it relative to
how much one position's encodings differ, so directions that only carry sampling noise add almost
nothing (they inflate RankMe instead). RankMe and the self-consistency ratio are reported alongside as
diagnostics. See representation_report().
"""
from __future__ import annotations

import copy
import dataclasses
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

import chess_graph as cg
import guo_et_al_unpooling_v2 as unpool
import labyrinth_data as ld

TOKEN_VARIANTS = ("squares", "registers", "unpool")
ANTICOLLAPSE = ("sigreg", "vicreg", "none")


# ##### ##### ##### ##### #####
#       Configuration


@dataclass
class JEPAConfig:
    """
    Every knob for the encoder, the predictors and the training signal. Field names are the
    ConfigSpace hyperparameter names, so a DEHB sample becomes a config with JEPAConfig.from_dict.
    """
    # ----- Graph (fixed by chess_graph) -----
    node_feature_dim: int = cg.NUM_NODE_FEATURES
    num_edge_types: int = cg.NUM_EDGE_TYPES

    # ----- Square stage: GAT layers on the chess graph -----
    gat_dim: int = 128
    gat_heads: int = 4
    gat_layers: int = 3              # at least 2, so every square can tell which way the board faces
    gat_dropout: float = 0.0

    # ----- Extra tokens -----
    token_variant: str = "unpool"    # "squares" | "registers" | "unpool"
    num_registers: int = 32
    unpool_layers: int = 2
    unpool_hidden: int = 128         # hidden width of the MLPs inside each unpooling layer
    unpool_edge_dim: int = 16        # width of the edge features the unpooling layers produce

    # ----- Transformer over the tokens -----
    d_model: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_ff_mult: int = 4
    transformer_dropout: float = 0.0

    # ----- Readout -----
    readout_queries: int = 8
    z_dim: int = 1024                # keep fixed during a DEHB search, so LiDAR scores are comparable

    # ----- Masked-square predictor (a GAT on the chess graph) -----
    pred_dim: int = 128
    pred_gat_layers: int = 3
    pred_heads: int = 4

    # ----- Next-position predictor -----
    move_dim: int = 128
    next_hidden: int = 1024
    next_layers: int = 2

    # ----- Objectives -----
    mask_weight: float = 1.0
    next_weight: float = 1.0
    loss_type: str = "smooth_l1"     # "smooth_l1" | "mse", on layer-normalized targets

    # ----- Anti-collapse regularization on z -----
    anticollapse: str = "sigreg"     # "sigreg" | "vicreg" | "none"
    sigreg_weight: float = 0.05      # LeJEPA's default lambda
    vicreg_weight: float = 1.0       # VICReg's variance weight equals its invariance weight; our losses play that role
    anticollapse_squares: bool = False   # also regularize the square tokens (not only z)
    sigreg_slices: int = 256         # random 1-D projections per step
    sigreg_points: int = 17          # quadrature points of the Epps-Pulley integral on [0, sigreg_t_max]
    sigreg_t_max: float = 3.0
    sigreg_unbiased: bool = True     # finite-sample-unbiased Epps-Pulley statistic (see sigreg)
    vicreg_cov_ratio: float = 0.04   # covariance weight relative to the variance term (VICReg's 1 : 25)
    decorrelation_weight: float = 0.0  # adds VICReg's covariance (decorrelation) term on top of SIGReg: SIGReg's
                                       # random 1-D projections barely see a few strong factors spread across all
                                       # dimensions, which correlated dimensions are; 0 turns it off

    # ----- Rollout curriculum (predicting several moves ahead) -----
    rollout_max_steps: int = 4       # the longest rollout
    rollout_stage_positions: int = 250_000   # training positions per curriculum stage (0: full length from the start)
    rollout_ramp_fraction: float = 0.25      # a newly added step's weight ramps from 0 to 1 over this part of its stage
    rollout_step_decay: float = 1.0  # step k's loss weight is decay ** (k - 1)

    # ----- Unpooling decisions (REINFORCE) -----
    unpool_rl: bool = True           # False freezes the decision policy; everything else still trains
    rl_coef: float = 0.01
    rl_entropy_coef: float = 1e-3

    # ----- Target encoder -----
    ema_start: float = 0.996         # moving-average momentum, raised to ema_end over training
    ema_end: float = 1.0

    # ----- Masks (passed to the data pipeline) -----
    mask_min_ratio: float = 0.15
    mask_max_ratio: float = 0.40
    mask_max_block_radius: int = 1

    # ----- Optimizer (read by the training loop) -----
    lr: float = 3e-4
    weight_decay: float = 0.05

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "JEPAConfig":
        """Build a config from a (possibly partial) dict such as a DEHB sample; unknown keys are ignored."""
        names = {f.name for f in dataclasses.fields(cls)}
        clean = {k: (v.item() if hasattr(v, "item") else v) for k, v in values.items() if k in names}
        return cls(**clean)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def mask_config(self) -> ld.MaskConfig:
        return ld.MaskConfig(self.mask_min_ratio, self.mask_max_ratio, self.mask_max_block_radius)

    @property
    def uses_unpool(self) -> bool:
        return self.token_variant == "unpool"

    @property
    def max_tokens(self) -> int:
        if self.token_variant == "registers":
            return 64 + self.num_registers
        if self.token_variant == "unpool":
            return 64 + 64 * 2 ** self.unpool_layers
        return 64

    def validate(self) -> None:
        if self.token_variant not in TOKEN_VARIANTS:
            raise ValueError(f"token_variant must be one of {TOKEN_VARIANTS}")
        if self.gat_layers < 1 or self.pred_gat_layers < 1:
            raise ValueError("gat_layers and pred_gat_layers must be at least 1")
        if self.d_model % self.transformer_heads:
            raise ValueError("d_model must be divisible by transformer_heads")
        if self.gat_dim % self.gat_heads or self.pred_dim % self.pred_heads:
            raise ValueError("gat_dim and pred_dim must be divisible by their head counts")
        if self.uses_unpool and (self.unpool_layers < 1 or self.gat_dim < 4):
            raise ValueError("the unpool variant needs unpool_layers >= 1 and gat_dim >= 4")
        if self.token_variant == "registers" and self.num_registers < 1:
            raise ValueError("the registers variant needs num_registers >= 1")
        if self.loss_type not in ("smooth_l1", "mse"):
            raise ValueError("loss_type must be smooth_l1 or mse")
        if self.anticollapse not in ANTICOLLAPSE:
            raise ValueError(f"anticollapse must be one of {ANTICOLLAPSE}")
        if self.rollout_max_steps < 1 or self.rollout_stage_positions < 0:
            raise ValueError("need rollout_max_steps >= 1 and rollout_stage_positions >= 0")
        if not 0 < self.rollout_ramp_fraction <= 1 or self.rollout_step_decay <= 0:
            raise ValueError("need 0 < rollout_ramp_fraction <= 1 and rollout_step_decay > 0")
        if not 0 <= self.mask_min_ratio <= self.mask_max_ratio <= 1:
            raise ValueError("need 0 <= mask_min_ratio <= mask_max_ratio <= 1")


def _mlp(sizes: Sequence[int]) -> nn.Sequential:
    """Linear -> LayerNorm -> GELU between every pair of sizes; the last layer is linear."""
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.LayerNorm(sizes[i + 1]), nn.GELU()]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


class _GATBlock(nn.Module):
    """
    A residual GAT layer, h + GELU(LayerNorm(GAT(h))), its width split across the heads.
    add_self_loops=False for the chess graph, which brings its own self-loops (some carry castling
    flags; see chess_graph.to_single_graph); True for unpooled graphs, which have none.
    """

    def __init__(self, dim: int, heads: int, edge_dim: int, dropout: float, add_self_loops: bool = True):
        super().__init__()
        self.gat = GATConv(dim, dim // heads, heads=heads, concat=True, dropout=dropout, edge_dim=edge_dim,
                           add_self_loops=add_self_loops)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return h + F.gelu(self.norm(self.gat(h, edge_index, edge_attr)))


# ##### ##### ##### ##### #####
#       The encoder


@dataclass
class EncoderOutput:
    """
    z          [B, z_dim]    the position vectors
    squares    [B, 64, d_model]  the final square tokens (the masked-square targets / predictor inputs)
    log_prob   [B]  log-probability of each position's sampled unpooling decisions (0 without unpooling)
    entropy    [B]  their entropy
    num_tokens [B]  tokens each position had in the transformer
    num_decisions [B]  how many unpooling decisions each position's log_prob sums over (0 without unpooling)
    actions    one guo v2 UnpoolActions record per unpooling layer (empty without unpooling); pass back
               as actions_to_replay to rebuild the same enlarged graphs
    """
    z: torch.Tensor
    squares: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    num_tokens: torch.Tensor
    num_decisions: torch.Tensor
    actions: List[unpool.UnpoolActions]


class LabyrinthEncoder(nn.Module):
    """Chess graphs -> position vectors. See the module docstring for the structure."""

    def __init__(self, config: JEPAConfig):
        super().__init__()
        config.validate()
        self.config = c = config

        self.input_proj = nn.Linear(c.node_feature_dim, c.gat_dim)
        self.mask_token = nn.Parameter(torch.randn(c.gat_dim) * 0.02)
        self.square_gats = nn.ModuleList(_GATBlock(c.gat_dim, c.gat_heads, c.num_edge_types, c.gat_dropout,
                                                   add_self_loops=False)
                                         for _ in range(c.gat_layers))

        self.unpool_layers = nn.ModuleList()
        self.unpool_gats = nn.ModuleList()
        if c.uses_unpool:
            edge_in = c.num_edge_types
            for _ in range(c.unpool_layers):
                self.unpool_layers.append(unpool.GuoUnpool(
                    dx=c.gat_dim, dw=edge_in, dy=c.gat_dim, du=c.unpool_edge_dim,
                    kv=c.unpool_hidden, kia=c.unpool_hidden, kie=c.unpool_hidden, kw=c.unpool_hidden))
                self.unpool_gats.append(_GATBlock(c.gat_dim, c.gat_heads, c.unpool_edge_dim, c.gat_dropout))
                edge_in = c.unpool_edge_dim
        self.registers = nn.Parameter(torch.randn(c.num_registers, c.d_model) * 0.02) \
            if c.token_variant == "registers" else None

        self.token_proj = nn.Linear(c.gat_dim, c.d_model)
        # Token types (square / extra), not positions: tokens stay an unordered set
        self.token_type = nn.Parameter(torch.randn(2, c.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(c.d_model, c.transformer_heads, c.transformer_ff_mult * c.d_model,
                                           dropout=c.transformer_dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, c.transformer_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(c.d_model)

        self.queries = nn.Parameter(torch.randn(c.readout_queries, c.d_model) * 0.02)
        self.readout_attn = nn.MultiheadAttention(c.d_model, c.transformer_heads, batch_first=True)
        self.readout_norm = nn.LayerNorm(c.d_model)
        self.readout_proj = nn.Linear(c.readout_queries * c.d_model, c.z_dim)

    def forward(
        self,
        graphs: ld.GraphBatch,
        mask: Optional[torch.Tensor] = None,
        rng: Optional[torch.Generator] = None,
        actions_to_replay: Optional[List[unpool.UnpoolActions]] = None,
    ) -> EncoderOutput:
        """
        graphs: B chess graphs of 64 nodes each (labyrinth_data.GraphBatch).
        mask: [B*64] bool, True on hidden squares; they get the mask token instead of their features.
        rng: generator for the unpooling decisions. actions_to_replay: rebuild recorded decisions.
        """
        c = self.config
        B = graphs.num_graphs
        square_edges, square_attr = graphs.edge_index, graphs.edge_attr     # one multi-hot edge per square pair
        h = self.input_proj(graphs.x)
        if mask is not None:
            h = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(h), h)
        for block in self.square_gats:
            h = block(h, square_edges, square_attr)
        square_tokens = self.token_proj(h).view(B, 64, c.d_model) + self.token_type[0]

        log_prob = h.new_zeros(B)
        entropy = h.new_zeros(B)
        num_decisions = torch.zeros(B, dtype=torch.long, device=h.device)
        actions: List[unpool.UnpoolActions] = []
        if c.token_variant == "unpool":
            x, edge_index, edge_attr, batch = h, square_edges, square_attr, graphs.batch
            for k, (layer, block) in enumerate(zip(self.unpool_layers, self.unpool_gats)):
                replay = actions_to_replay[k] if actions_to_replay is not None else None
                out = layer(x, edge_index, edge_attr, actions_to_replay=replay, rng=rng, batch=batch, num_graphs=B)
                x, edge_index, edge_attr, batch = out.y, out.edge_index, out.edge_attr, out.batch
                log_prob = log_prob + out.log_prob
                entropy = entropy + out.entropy
                actions.append(out.actions)
                num_decisions = num_decisions + out.actions.counts.sum(0)
                x = block(x, edge_index, edge_attr)
            extra, extra_pad = self._pad_by_graph(self.token_proj(x) + self.token_type[1], batch, B)
        elif c.token_variant == "registers":
            extra = (self.registers + self.token_type[1]).unsqueeze(0).expand(B, -1, -1)
            extra_pad = torch.zeros(B, c.num_registers, dtype=torch.bool, device=h.device)
        else:
            extra = h.new_zeros(B, 0, c.d_model)
            extra_pad = torch.zeros(B, 0, dtype=torch.bool, device=h.device)

        tokens = torch.cat([square_tokens, extra], dim=1)                          # [B, T, d]
        pad = torch.cat([torch.zeros(B, 64, dtype=torch.bool, device=h.device), extra_pad], dim=1)
        tokens = self.final_norm(self.transformer(tokens, src_key_padding_mask=pad))

        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        read, _ = self.readout_attn(queries, tokens, tokens, key_padding_mask=pad, need_weights=False)
        z = self.readout_proj(self.readout_norm(read).reshape(B, -1))
        return EncoderOutput(z, tokens[:, :64], log_prob, entropy, (~pad).sum(1), num_decisions, actions)

    @staticmethod
    def _pad_by_graph(x: torch.Tensor, batch: torch.Tensor, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nodes grouped by graph -> ([B, max_nodes, d] zero-padded, [B, max_nodes] True where padding)."""
        counts = torch.bincount(batch, minlength=B)
        width = int(counts.max()) if counts.numel() else 0
        slot = torch.arange(x.size(0), device=x.device) - (torch.cumsum(counts, 0) - counts)[batch]
        padded = x.new_zeros(B, width, x.size(1)).index_put((batch, slot), x)
        pad = torch.arange(width, device=x.device).unsqueeze(0) >= counts.unsqueeze(1)
        return padded, pad


# ##### ##### ##### ##### #####
#       Predictors


class MaskPredictor(nn.Module):
    """
    Predicts the target encoder's square tokens on the hidden squares from the context encoder's
    square tokens, with a GAT on the chess graph (the masked view's graph, so the hidden squares'
    castling edges stay hidden). Hidden squares are told apart by a learned flag, not by an ID:
    where a square is comes only from the graph, as everywhere else in the model.
    """

    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = c = config
        self.inp = nn.Linear(c.d_model, c.pred_dim)
        self.target_flag = nn.Parameter(torch.randn(c.pred_dim) * 0.02)
        self.gats = nn.ModuleList(_GATBlock(c.pred_dim, c.pred_heads, c.num_edge_types, 0.0, add_self_loops=False)
                                  for _ in range(c.pred_gat_layers))
        self.out = nn.Sequential(nn.LayerNorm(c.pred_dim), nn.Linear(c.pred_dim, c.d_model))

    def forward(self, squares: torch.Tensor, graphs: ld.GraphBatch, mask: torch.Tensor) -> torch.Tensor:
        """squares [B, 64, d_model], mask [B*64] -> predictions [B*64, d_model] (meaningful on masked squares)."""
        h = self.inp(squares.reshape(-1, squares.size(-1)))
        h = h + mask.unsqueeze(-1).to(h.dtype) * self.target_flag
        edge_index, edge_attr = graphs.edge_index, graphs.edge_attr
        for block in self.gats:
            h = block(h, edge_index, edge_attr)
        return self.out(h)


class NextPredictor(nn.Module):
    """
    Predicts the target encoder's z of the position a move leads to, from z (encoded, or predicted by
    the previous rollout step) and the move. The move's squares are tokens of the starting position, and
    a learned embedding of the step number tells the predictor which side is moving and how far the
    board has drifted from those tokens.
    """

    def __init__(self, config: JEPAConfig):
        super().__init__()
        c = config
        self.promotion = nn.Embedding(7, c.move_dim)          # 0 = none, else the python-chess piece type
        self.step = nn.Embedding(c.rollout_max_steps + 1, c.move_dim)
        self.move = _mlp([2 * c.d_model + 3, c.move_dim, c.move_dim])
        self.body = _mlp([c.z_dim + c.move_dim] + [c.next_hidden] * c.next_layers + [c.z_dim])

    def forward(self, z: torch.Tensor, from_token: torch.Tensor, to_token: torch.Tensor,
                promotion: torch.Tensor, flags: torch.Tensor, step: int = 1) -> torch.Tensor:
        move = self.move(torch.cat([from_token, to_token, flags], dim=-1)) + self.promotion(promotion)
        move = move + self.step.weight[step]
        return z + self.body(torch.cat([z, move], dim=-1))     # residual: predict the change


# ##### ##### ##### ##### #####
#       Anti-collapse regularizers


def sigreg(z: torch.Tensor, generator: Optional[torch.Generator] = None, slices: int = 256, points: int = 17,
           t_max: float = 3.0, unbiased: bool = True) -> torch.Tensor:
    """
    SIGReg, the regularizer of LeJEPA (Balestriero & LeCun, 2025). z [N, D] is projected onto `slices`
    random unit directions; each 1-D projection is compared with a standard normal by the Epps-Pulley
    statistic, N times the integral of |empirical characteristic function - exp(-t^2/2)|^2 under a
    Gaussian window, computed with the trapezoid rule on [0, t_max] (the integrand is even). The mean
    over directions is 0 for an isotropic standard Gaussian and grows with any shared offset, collapsed
    or unused direction, or non-Gaussian shape. New directions every call.

    unbiased=True (the default here) estimates the squared characteristic-function gap without its
    finite-sample bias, which matters most at small batches: the plain form (unbiased=False, as in the
    LeJEPA paper) has an expected extra term that nudges embeddings to narrow. Unbiased, a standard
    Gaussian scores about 0 on average (single values can be slightly negative).
    """
    N, D = z.shape
    dirs = torch.randn(D, slices, device=z.device, dtype=z.dtype, generator=generator)
    dirs = dirs / dirs.norm(dim=0, keepdim=True)
    proj = z @ dirs                                                     # [N, slices]
    t = torch.linspace(0, t_max, points, device=z.device, dtype=z.dtype)
    phi = torch.exp(-0.5 * t ** 2)                                      # characteristic function of N(0, 1)
    dt = t_max / (points - 1)
    weights = torch.full_like(t, 2 * dt)                                # trapezoid on [0, t_max], doubled for [-t_max, t_max]
    weights[0] = weights[-1] = dt
    weights = weights * phi                                             # Gaussian window
    xt = proj.unsqueeze(-1) * t                                         # [N, slices, points]
    cos_mean, sin_mean = torch.cos(xt).mean(0), torch.sin(xt).mean(0)
    if unbiased and N > 1:
        # |phi_hat - phi|^2 = |phi_hat|^2 - 2 phi Re(phi_hat) + phi^2, but E|phi_hat|^2 = |phi_X|^2 + (1 - |phi_X|^2) / N:
        # the plain statistic carries an extra (1 - |phi_X|^2), which does not shrink with N and whose gradient
        # favours narrower distributions. (N |phi_hat|^2 - 1) / (N - 1) estimates |phi_X|^2 without that bias.
        sq = (N * (cos_mean ** 2 + sin_mean ** 2) - 1) / (N - 1)
        err = sq - 2 * phi * cos_mean + phi ** 2
    else:
        err = (cos_mean - phi) ** 2 + sin_mean ** 2
    return (err @ weights).mean() * N


def vicreg_terms(z: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    VICReg's (Bardes, Ponce & LeCun, 2022) variance and covariance terms for z [N, D]: the mean hinge
    loss keeping every dimension's standard deviation across the batch at 1 or more, and the sum of
    squared off-diagonal covariances divided by D.
    """
    N, D = z.shape
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + eps)
    variance = F.relu(1 - std).mean()
    cov = (z.t() @ z) / max(1, N - 1)
    off_diagonal = cov - torch.diag(torch.diag(cov))
    return variance, off_diagonal.pow(2).sum() / D


# ##### ##### ##### ##### #####
#       The JEPA


@dataclass
class JEPALosses:
    total: torch.Tensor
    mask: torch.Tensor          # mean masked-square loss
    next: torch.Tensor          # mean next-position loss
    rl: torch.Tensor            # REINFORCE + baseline + entropy terms (0 without unpooling or with unpool_rl off)
    anticollapse: torch.Tensor  # the SIGReg / VICReg term (0 with anticollapse = "none")
    stats: Dict[str, float]     # diagnostics for logging
    # Each term's weighted share of total, which they add up to: "mask", "next", "anticollapse_<embedding>"
    # (masked, full, squares), "decorrelation", "rl". For measuring what each term alone does to the gradient.
    terms: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)


class LabyrinthJEPA(nn.Module):
    """The encoder, its moving-average target, both predictors, and the training losses."""

    def __init__(self, config: JEPAConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.encoder = LabyrinthEncoder(config)
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.mask_predictor = MaskPredictor(config)
        self.next_predictor = NextPredictor(config)
        # Learned baselines for the unpooling rewards: each position's expected loss, from its z
        self.baselines = nn.ModuleDict({k: _mlp([config.z_dim, 256, 1]) for k in ("mask", "next")}) \
            if config.uses_unpool else None

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if not n.startswith("target.")]

    def momentum_at(self, step: int, total_steps: int) -> float:
        """Moving-average momentum, raised from ema_start to ema_end on a cosine schedule."""
        c = self.config
        t = min(1.0, step / max(1, total_steps))
        return c.ema_end - (c.ema_end - c.ema_start) * (math.cos(math.pi * t) + 1) / 2

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for p_t, p_c in zip(self.target.parameters(), self.encoder.parameters()):
            p_t.mul_(momentum).add_(p_c.detach(), alpha=1 - momentum)
        for b_t, b_c in zip(self.target.buffers(), self.encoder.buffers()):
            b_t.copy_(b_c)

    def predict_rollout(self, z: torch.Tensor, squares: torch.Tensor, steps: Sequence[ld.RolloutStep]) -> List[torch.Tensor]:
        """
        Latent rollout: z [B, z_dim] and squares [B, 64, d_model] of the starting positions (online
        encoder) -> one prediction per step, [B_k, z_dim], each from the previous step's prediction.
        Uses only the starting positions and the moves; the steps' graphs (the intermediate positions)
        are never read here, only encoded as targets elsewhere.
        """
        flat = squares.reshape(-1, squares.size(-1))
        source, predictions = z, []
        for k, step in enumerate(steps, start=1):
            pred = self.next_predictor(source[step.parent], flat[step.move_from], flat[step.move_to],
                                       step.promotion, step.flags, k)
            predictions.append(pred)
            source = pred
        return predictions

    def _distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-row loss against a layer-normalized target."""
        target = F.layer_norm(target, target.shape[-1:])
        if self.config.loss_type == "mse":
            return F.mse_loss(pred, target, reduction="none").mean(-1)
        return F.smooth_l1_loss(pred, target, reduction="none").mean(-1)

    def losses(self, batch: ld.JEPABatch, rng: Optional[torch.Generator] = None,
               rollout_weights: Optional[Sequence[float]] = None) -> JEPALosses:
        """
        All training losses for one batch (already on the model's device). rollout_weights: the loss
        weight of each rollout step (rollout_schedule gives them); its length is how many moves ahead to
        predict, and the batch must reach at least that far where games allow. Default: one move.
        """
        c = self.config
        B = batch.num_graphs
        stats: Dict[str, float] = {}
        weights = list(rollout_weights) if rollout_weights is not None else [1.0]
        steps = batch.rollout[:len(weights)]

        with torch.no_grad():
            self.target.eval()
            target_full = self.target(batch.full, rng=rng)
            target_steps = [self.target(step.graphs, rng=rng).z for step in steps]

        # ---- masked squares ----
        ctx_masked = self.encoder(batch.context, mask=batch.mask, rng=rng)
        pred = self.mask_predictor(ctx_masked.squares, batch.context, batch.mask)
        per_square = self._distance(pred[batch.mask], target_full.squares.reshape(B * 64, -1)[batch.mask])
        owner = batch.full.batch[batch.mask]
        n_masked = torch.bincount(owner, minlength=B).clamp_min(1)
        mask_per_pos = torch.zeros(B, device=per_square.device).index_add(0, owner, per_square) / n_masked
        mask_loss = mask_per_pos.mean()

        # ---- next positions: a latent rollout, a loss at every step ----
        if steps:
            ctx_full = self.encoder(batch.full, rng=rng)
            predictions = self.predict_rollout(ctx_full.z, ctx_full.squares, steps)
            loss_sum = mask_loss.new_zeros(B)
            weight_sum = mask_loss.new_zeros(B)
            weighted, total_weight = mask_loss.new_zeros(()), 0.0
            for k, (step, pred, target_z, w) in enumerate(zip(steps, predictions, target_steps, weights), start=1):
                per = self._distance(pred, target_z)
                loss_sum = loss_sum.index_add(0, step.index, w * per)
                weight_sum = weight_sum.index_add(0, step.index, torch.full_like(per, w))
                weighted = weighted + w * per.mean()
                total_weight += w
                stats[f"loss_next_{k}"] = float(per.detach().mean())
            next_loss = weighted / max(total_weight, 1e-12)
            first = steps[0].index
            next_per_pos = loss_sum[first] / weight_sum[first].clamp_min(1e-12)   # each position's weighted rollout loss
        else:
            ctx_full, next_per_pos, next_loss = None, None, mask_loss.new_zeros(())
        stats["rollout_steps"] = float(len(steps))

        contributions = {"mask": c.mask_weight * mask_loss, "next": c.next_weight * next_loss}
        total = contributions["mask"] + contributions["next"]

        # ---- anti-collapse regularization on the online encoder's embeddings ----
        anticollapse = total.new_zeros(())
        if c.anticollapse != "none":
            embeddings = [("masked", ctx_masked.z)] + ([("full", ctx_full.z)] if ctx_full is not None else [])
            if c.anticollapse_squares:
                squares = ctx_masked.squares.reshape(-1, c.d_model)
                if squares.size(0) > 4096:     # a random subset keeps the cost bounded
                    keep = torch.randperm(squares.size(0), device=squares.device, generator=rng)[:4096]
                    squares = squares[keep]
                embeddings.append(("squares", squares))
            terms = []
            # Measured on the current batch only. (A queue of recent embeddings was tried, to give the
            # statistic more samples: the queue's centre lags the encoder, so the cheapest way to lower
            # the statistic became moving the whole batch away from it, and training collapsed.)
            for name, e in embeddings:
                if c.anticollapse == "sigreg":
                    terms.append(sigreg(e, rng, c.sigreg_slices, c.sigreg_points, c.sigreg_t_max, c.sigreg_unbiased))
                else:
                    variance, covariance = vicreg_terms(e)
                    terms.append(variance + c.vicreg_cov_ratio * covariance)
            anticollapse = torch.stack(terms).mean()
            reg_weight = c.sigreg_weight if c.anticollapse == "sigreg" else c.vicreg_weight
            total = total + reg_weight * anticollapse
            for (name, _), term in zip(embeddings, terms):
                contributions[f"anticollapse_{name}"] = reg_weight * term / len(terms)
                stats[f"anticollapse_{name}"] = float(term.detach())
            if c.decorrelation_weight > 0 and c.anticollapse == "sigreg":
                decorrelation = torch.stack([vicreg_terms(e)[1] for _, e in embeddings]).mean()
                contributions["decorrelation"] = c.decorrelation_weight * decorrelation
                total = total + contributions["decorrelation"]
                stats["decorrelation"] = float(decorrelation.detach())

        # ---- unpooling decisions: REINFORCE with learned baselines ----
        # A position involves thousands of decisions, so its log-probability and entropy are averaged per
        # decision: the same gradient direction, on a scale comparable with the prediction losses.
        rl = total.new_zeros(())
        if c.uses_unpool and c.unpool_rl:
            terms = [("mask", ctx_masked, mask_per_pos, slice(None))]
            if ctx_full is not None:
                terms.append(("next", ctx_full, next_per_pos, steps[0].index))
            for name, out, loss_per_pos, rows in terms:
                reward = -loss_per_pos.detach()
                baseline = self.baselines[name](out.z[rows].detach()).squeeze(-1)
                advantage = reward - baseline.detach()
                if advantage.numel() > 1:      # centred and scaled per batch, so a baseline that is still
                    advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)  # off by a constant cannot dominate
                per_decision = out.num_decisions[rows].clamp_min(1).to(out.log_prob.dtype)
                rl = rl - (advantage * out.log_prob[rows] / per_decision).mean() + F.mse_loss(baseline, reward)
                rl = rl - c.rl_entropy_coef * (out.entropy[rows] / per_decision).mean()
                stats[f"{name}_reward_std"] = float(reward.std()) if reward.numel() > 1 else 0.0
            contributions["rl"] = c.rl_coef * rl
            total = total + contributions["rl"]

        with torch.no_grad():
            z = ctx_masked.z
            stats.update({
                "loss_mask": float(mask_loss), "loss_next": float(next_loss),
                "z_std": float(z.std(0).mean()),                         # falls toward 0 on collapse
                "z_mean_cosine": float(F.cosine_similarity(z, z.mean(0, keepdim=True)).mean()),  # near 1: all z alike
                "anticollapse": float(anticollapse),
                "target_z_std": float(target_full.z.std(0).mean()),
                "tokens": float(ctx_masked.num_tokens.float().mean()),
                "unpool_entropy_per_decision": float((ctx_masked.entropy / ctx_masked.num_decisions.clamp_min(1)).mean()),
            })
        return JEPALosses(total, mask_loss, next_loss, rl, anticollapse, stats, contributions)


def build_jepa(config: JEPAConfig) -> LabyrinthJEPA:
    return LabyrinthJEPA(config)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def describe(config: JEPAConfig) -> Dict[str, Any]:
    """Sizes worth looking at before committing to a configuration."""
    model = LabyrinthJEPA(config)
    return {
        "encoder_params": count_parameters(model.encoder),
        "mask_predictor_params": count_parameters(model.mask_predictor),
        "next_predictor_params": count_parameters(model.next_predictor),
        "max_tokens": config.max_tokens,
        "z_dim": config.z_dim,
    }


def rollout_schedule(config: JEPAConfig, positions_seen: int) -> List[float]:
    """
    The loss weight of each rollout step after positions_seen training positions; the list's length is
    how many moves ahead to predict. One move at first, one more every rollout_stage_positions positions,
    up to rollout_max_steps. A newly added step's weight rises linearly from 0 to its full value over the
    first rollout_ramp_fraction of its stage. Step k's full weight is rollout_step_decay ** (k - 1).
    """
    c = config
    if c.rollout_stage_positions <= 0 or c.rollout_max_steps == 1:
        steps, ramp = c.rollout_max_steps, 1.0
    else:
        steps = min(c.rollout_max_steps, 1 + positions_seen // c.rollout_stage_positions)
        added_at = (steps - 1) * c.rollout_stage_positions
        progress = (positions_seen - added_at) / c.rollout_stage_positions
        ramp = 1.0 if steps == 1 else min(1.0, progress / c.rollout_ramp_fraction)
    weights = [c.rollout_step_decay ** k for k in range(steps)]
    weights[-1] *= ramp
    return weights


# ##### ##### ##### ##### #####
#       Representation metrics (label-free)


def rankme(z: torch.Tensor, eps: float = 1e-7) -> float:
    """
    RankMe (Garrido et al. 2023): exp of the entropy of the normalized singular values of the
    embedding matrix z [N, D]. Between 1 and min(N, D). Rewards using many dimensions, including
    noise, which is why it is a diagnostic here and not the objective.
    """
    s = torch.linalg.svdvals(z.double())
    p = s / s.sum() + eps
    return float(torch.exp(-(p * torch.log(p)).sum()))


def lidar(z: torch.Tensor, delta: float = 1e-4, eps: float = 1e-7) -> float:
    """
    LiDAR (Thilak et al., ICLR 2024). z [N, q, D]: q random encodings (members) of each of N positions
    (classes). The effective rank of Sw^-1/2 Sb Sw^-1/2, where Sb is the covariance of the class means
    and Sw the average within-class covariance plus delta * I. Each direction counts by its ratio of
    between-position to within-position variance: directions that only carry sampling noise add almost
    nothing, while noise mixed into informative directions dilutes them.
    """
    z = z.double()
    N, q, D = z.shape
    means = z.mean(1)                                             # [N, D]
    centered_means = means - means.mean(0)
    sb = centered_means.t() @ centered_means / max(1, N - 1)
    within = z - means.unsqueeze(1)
    sw = torch.einsum("nqd,nqe->de", within, within) / max(1, N * (q - 1)) + delta * torch.eye(D, dtype=z.dtype, device=z.device)
    evals, evecs = torch.linalg.eigh(sw)
    inv_sqrt = evecs @ torch.diag(evals.clamp_min(1e-12).rsqrt()) @ evecs.t()
    lam = torch.linalg.eigvalsh(inv_sqrt @ sb @ inv_sqrt).clamp_min(0)
    p = lam / lam.sum() + eps
    return float(torch.exp(-(p * torch.log(p)).sum()))


def self_consistency(z: torch.Tensor) -> float:
    """
    z [N, q, D]: mean distance between two encodings of the same position, divided by the mean distance
    between encodings of different positions. 0 = perfectly consistent; about 1 = no better than chance.
    """
    N, q, _ = z.shape
    within = torch.cdist(z, z).sum() / max(1, N * q * (q - 1))
    means = z.mean(1)
    between = torch.cdist(means, means).sum() / max(1, N * (N - 1))
    return float(within / between.clamp_min(1e-12))


@torch.no_grad()
def encode_positions(encoder: LabyrinthEncoder, fens: Sequence[str], *, samples: int = 4,
                     mask_config: Optional[ld.MaskConfig] = None, batch_size: int = 256,
                     seed: int = 0, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    [N, samples, z_dim]: each position encoded `samples` times, with fresh unpooling decisions each
    time and, if mask_config is given, a fresh light mask (as LiDAR does for I-JEPA). Without
    unpooling or masks the encodings of a position are identical.
    """
    device = device or next(encoder.parameters()).device
    was_training = encoder.training
    encoder.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    rnd = random.Random(seed)
    out = torch.empty(len(fens), samples, encoder.config.z_dim, device=device)
    for s in range(samples):
        for lo in range(0, len(fens), batch_size):
            chunk = fens[lo:lo + batch_size]
            views, masks = [], []
            for fen in chunk:
                m = ld.sample_mask(rnd, mask_config) if mask_config is not None else torch.zeros(64, dtype=torch.bool)
                full, context, _, _ = ld.build_views(fen, m)
                views.append(context if mask_config is not None else full)
                masks.append(m)
            graphs = ld.pack_views(views).to(device)
            mask = torch.cat(masks).to(device) if mask_config is not None else None
            out[lo:lo + len(chunk), s] = encoder(graphs, mask=mask, rng=gen).z
    encoder.train(was_training)
    return out


def representation_report(encoder: LabyrinthEncoder, fens: Sequence[str], *, samples: int = 4,
                          mask_config: Optional[ld.MaskConfig] = None, seed: int = 0) -> Dict[str, float]:
    """
    LiDAR (the DEHB objective), RankMe and the self-consistency ratio of the encoder on these positions.
    Use a fixed set of held-out positions (the val split) so scores are comparable between runs; LiDAR
    needs more positions than z_dim to reach its full range.
    """
    z = encode_positions(encoder, fens, samples=samples, mask_config=mask_config, seed=seed)
    return {
        "lidar": lidar(z),
        "rankme": rankme(z.mean(1)),
        "rankme_centered": rankme(z.mean(1) - z.mean(1).mean(0)),   # without the component every position shares
        "rankme_single_sample": rankme(z[:, 0]),
        "self_consistency": self_consistency(z),
    }


# ##### ##### ##### ##### #####
#       The DEHB search space


def build_jepa_configspace(seed: int = 0, include_training: bool = True, fixed: Optional[Dict[str, Any]] = None):
    """
    The DEHB search space over the architecture (and, with include_training, the training knobs).
    z_dim is deliberately not searched: LiDAR grows with the dimension, so scores are only comparable
    at a fixed z_dim. Choose it separately and keep it fixed for a search.

    fixed: JEPAConfig values that are not searched. Their hyperparameters are left out, and one that only
    applies under a particular value of a fixed hyperparameter (unpool_layers under token_variant =
    "unpool") is kept, unconditionally, only when the fixed value is that one.
    """
    import ConfigSpace as CS
    fixed = fixed or {}
    H = CS.CategoricalHyperparameter
    I = CS.UniformIntegerHyperparameter
    U = CS.UniformFloatHyperparameter

    # (hyperparameter, None or (the hyperparameter it depends on, the value it needs, or a list of values))
    specs = [
        # square stage
        (I("gat_layers", lower=2, upper=4), None),
        (H("gat_dim", choices=[64, 128, 256]), None),
        (H("gat_heads", choices=[2, 4, 8]), None),
        # extra tokens
        (H("token_variant", choices=list(TOKEN_VARIANTS)), None),
        (H("num_registers", choices=[16, 32, 64]), ("token_variant", "registers")),
        (I("unpool_layers", lower=1, upper=3), ("token_variant", "unpool")),
        (H("unpool_hidden", choices=[64, 128, 256]), ("token_variant", "unpool")),
        (H("unpool_edge_dim", choices=[8, 16, 32]), ("token_variant", "unpool")),
        # transformer and readout
        (H("d_model", choices=[128, 256, 384]), None),
        (I("transformer_layers", lower=2, upper=6), None),
        (H("transformer_heads", choices=[4, 8]), None),
        (H("readout_queries", choices=[4, 8, 16, 32]), None),
        # predictors
        (H("pred_dim", choices=[64, 128, 256]), None),
        (I("pred_gat_layers", lower=2, upper=4), None),
        (H("next_hidden", choices=[512, 1024, 2048]), None),
        (I("next_layers", lower=1, upper=3), None),
        # masks
        (U("mask_min_ratio", lower=0.05, upper=0.25), None),
        (U("mask_max_ratio", lower=0.30, upper=0.60), None),
        (I("mask_max_block_radius", lower=0, upper=2), None),
        # rollout curriculum
        (H("rollout_max_steps", choices=[1, 2, 3, 4, 6]), None),
        (H("rollout_stage_positions", choices=[100_000, 250_000, 500_000]), ("rollout_max_steps", [2, 3, 4, 6])),
    ]
    if include_training:
        specs += [
            (U("lr", lower=5e-5, upper=1e-3, log=True), None),
            (U("weight_decay", lower=1e-3, upper=0.2, log=True), None),
            (U("ema_start", lower=0.99, upper=0.999), None),
            (U("next_weight", lower=0.25, upper=4.0, log=True), None),     # relative to mask_weight = 1
            (H("anticollapse", choices=["sigreg", "vicreg"]), None),
            (U("sigreg_weight", lower=0.005, upper=0.5, log=True), ("anticollapse", "sigreg")),
            (U("vicreg_weight", lower=0.1, upper=10.0, log=True), ("anticollapse", "vicreg")),
            (U("rl_coef", lower=1e-3, upper=1e-1, log=True), ("token_variant", "unpool")),
            (U("rl_entropy_coef", lower=1e-5, upper=1e-2, log=True), ("token_variant", "unpool")),
        ]

    def allows(condition: Tuple[str, Any], value: Any) -> bool:
        return value in condition[1] if isinstance(condition[1], (list, tuple)) else value == condition[1]

    cs = CS.ConfigurationSpace(seed=seed)
    kept: Dict[str, Tuple[Any, Optional[Tuple[str, Any]]]] = {}
    for hp, condition in specs:
        if hp.name in fixed:
            continue
        if condition is not None and condition[0] in fixed and not allows(condition, fixed[condition[0]]):
            continue                       # it only applies under a value the fixed setting rules out
        cs.add_hyperparameter(hp)
        kept[hp.name] = (hp, condition)
    for hp, condition in kept.values():
        if condition is not None and condition[0] in kept:
            parent = kept[condition[0]][0]
            if isinstance(condition[1], (list, tuple)):
                cs.add_condition(CS.InCondition(hp, parent, list(condition[1])))
            else:
                cs.add_condition(CS.EqualsCondition(hp, parent, condition[1]))
    return cs


# ----- ----- -----
# Program Body

if __name__ == "__main__":
    import json

    config = JEPAConfig()
    print("Default configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    print("\nSizes:")
    print(json.dumps(describe(config), indent=2))

    rnd = random.Random(0)
    fens = ["1q1rkr2/pp3pnp/2pn2pQ/3p4/3Pb3/2P2NP1/PP2P2P/3RKRNB b - - 1 15",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"]
    batch = ld.collate_jepa([ld.build_sample(ld.normalize_fen(f), rnd, config.mask_config()) for f in fens])
    model = LabyrinthJEPA(config)
    out = model.losses(batch, rng=torch.Generator().manual_seed(0))
    out.total.backward()
    print(f"\nlosses: total={float(out.total):.4f} mask={float(out.mask):.4f} next={float(out.next):.4f} rl={float(out.rl):.2f}")
    print(json.dumps(out.stats, indent=2))
