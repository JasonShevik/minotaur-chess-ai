import guo_et_al_unpooling as unpool
import chess_graph as cg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import List, Optional


# ##### ##### ##### ##### #####
#       Helpers: chess graph -> single graph for GuoUnpool
# ##### ##### ##### ##### #####


def chess_graph_to_single(
    x: torch.Tensor,
    edge_index_list: List[torch.Tensor],
    num_edge_types: int = 8,
) -> tuple:
    """
    Merge the 8 chess edge lists into one edge_index and edge_attr.

    x: [N, node_feat_dim], edge_index_list: list of 8 tensors each [2, E_i].
    Returns (x, edge_index [2, E_total], edge_attr [E_total, num_edge_types]).

    Edge type encoding: each edge gets a one-hot vector for its type (one 1 per row).
    When the same pair (u, v) is connected by multiple edge types (e.g. e2-e3 by both
    pawn move and king move), we keep multiple edges: one per type. So (u,v) appears
    once per type with that type's one-hot. We do NOT coalesce into one edge with
    multi-hot; downstream (GAT, GuoUnpool) see separate messages per edge type.
    """
    device = x.device
    dtype = x.dtype
    edge_indices = []
    edge_attrs = []
    for i, ei in enumerate(edge_index_list):
        if ei.numel() == 0:
            continue
        ei = ei.to(device)
        E_i = ei.size(1)
        edge_indices.append(ei)
        # One-hot for edge type i
        one_hot = F.one_hot(
            torch.full((E_i,), i, dtype=torch.long, device=device),
            num_classes=num_edge_types,
        ).to(dtype)
        edge_attrs.append(one_hot)
    if not edge_indices:
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        edge_attr = torch.empty(0, num_edge_types, dtype=dtype, device=device)
    else:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)
    return x, edge_index, edge_attr


# ##### ##### ##### ##### #####
#       DGI Encoding

class NodeEncoder(nn.Module):
    """
    Encodes 64 board squares into node embeddings using one GAT per hop with edge
    features. Edge types (pawn move, knight, king, etc.) are passed as edge_attr
    via chess_graph_to_single; the same (u,v) can appear as multiple edges with
    different one-hot type vectors (e.g. e2-e3 as both pawn and king).
    """

    def __init__(
        self,
        in_channels: int,
        edge_dim: int = 8,
        hidden_dim1: int = 32,
        hidden_dim2: int = 128,
        out_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super(NodeEncoder, self).__init__()
        self.activation = nn.ReLU()

        # One GAT per hop; edge_dim so attention uses edge type (one-hot per edge)
        self.gat1 = GATConv(
            in_channels,
            hidden_dim1,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim,
        )
        self.gat2 = GATConv(
            hidden_dim1,
            hidden_dim2,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim,
        )
        self.gat3 = GATConv(
            hidden_dim2,
            out_dim,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        x: [N, in_channels], edge_index_list: list of 8 edge index tensors.
        Returns [N, out_dim] node embeddings (e.g. [64, 512]).
        """
        x, edge_index, edge_attr = chess_graph_to_single(x, edge_index_list)

        h = self.activation(self.gat1(x, edge_index, edge_attr))
        h = self.activation(self.gat2(h, edge_index, edge_attr))
        h = self.gat3(h, edge_index, edge_attr)

        return h


class GlobalSummarizer(nn.Module):
    """
    Summarizes a chess position graph via multiple Guo unpooling layers, then
    global pooling and an MLP to a latent vector. summary_dim is intended to
    be larger than node_dim (e.g. 768 or 1024).
    """

    def __init__(
        self,
        num_guo_layers: int = 2,
        node_feat_dim: int = 8,
        edge_feat_dim: int = 8,
        summary_dim: int = 768,
        unpool_hidden: int = 128,
        mlp_hidden_dim: int = 256,
    ):
        super(GlobalSummarizer, self).__init__()
        self.num_guo_layers = num_guo_layers
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.summary_dim = summary_dim

        # Stack of Guo unpool layers (same dx,dw,dy,du for all)
        self.unpool_layers = nn.ModuleList([
            unpool.GuoUnpool(
                dx=node_feat_dim,
                dw=edge_feat_dim,
                dy=node_feat_dim,
                du=edge_feat_dim,
                kv=unpool_hidden,
                kia=unpool_hidden,
                kie=unpool_hidden,
                kw=unpool_hidden,
            )
            for _ in range(num_guo_layers)
        ])

        # After unpool(s), we mean-pool node features (dim node_feat_dim), then MLP to summary_dim
        self.mlp = nn.Sequential(
            nn.Linear(node_feat_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, summary_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index_list: List[torch.Tensor],
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        x: [N, node_feat_dim], edge_index_list: list of 8 edge index tensors.
        Returns summary vector [summary_dim] (or [1, summary_dim] for batch compatibility).
        """
        x, edge_index, edge_attr = chess_graph_to_single(x, edge_index_list)

        for i in range(self.num_guo_layers):
            x, edge_index, edge_attr, *_ = self.unpool_layers[i](
                x, edge_index, edge_attr, rng=rng
            )

        # Global mean pool over nodes -> [node_feat_dim]
        if x.size(0) == 0:
            h = x.new_zeros(self.node_feat_dim)
        else:
            h = x.mean(dim=0)

        s = self.mlp(h)
        return s


class Discriminator(nn.Module):
    def __init__(self, node_dim, summary_dim, num_bilinear_maps=8, hidden_dim=128):
        super(Discriminator, self).__init__()

        # Multi-bilinear scoring ...
        self.W = nn.Parameter(
            torch.randn(num_bilinear_maps, node_dim, summary_dim) * 0.02
        )
        # ... with non-linear combination
        self.mlp = nn.Sequential(
            nn.Linear(num_bilinear_maps, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y):
        # x: [batch, node_dim], y: [batch, summary_dim]
        bilinear_scores = torch.einsum('bd,kds,bs->bk', x, self.W, y)
        return self.mlp(bilinear_scores).squeeze(-1)


# ----- ----- -----
# Program Body

if __name__ == "__main__":
    pass



