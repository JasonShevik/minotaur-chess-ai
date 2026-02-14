import math
import torch
import torch.nn.functional as F
from torch_geometric.utils import coalesce
from collections import Counter
from typing import Optional, Dict, Tuple


# ------------------------------
# Utilities
# ------------------------------
def _infer_num_nodes(edge_index: torch.Tensor) -> int:
    if edge_index.numel() == 0:
        return 0
    return int(edge_index.max().item()) + 1


def _to_dense_adj(edge_index: torch.Tensor, n: int, directed: bool = True) -> torch.Tensor:
    A = torch.zeros((n, n), dtype=torch.float32)
    if edge_index.numel() == 0 or n == 0:
        return A
    src, dst = edge_index[0].long().cpu(), edge_index[1].long().cpu()
    A[src, dst] = 1.0
    if not directed:
        A[dst, src] = 1.0
    return A


def _degree_hist_cdf(deg: torch.Tensor, max_deg: Optional[int] = None) -> torch.Tensor:
    if deg.numel() == 0:
        return torch.zeros(1)
    deg = deg.long().clamp(min=0)
    m = int(max_deg if max_deg is not None else deg.max().item())
    hist = torch.bincount(deg, minlength=m + 1).float()
    cdf = hist.cumsum(0)
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    return cdf


def _emd1_cdf_score(cdf1: torch.Tensor, cdf2: torch.Tensor) -> float:
    L = max(cdf1.numel(), cdf2.numel())
    if cdf1.numel() < L: cdf1 = F.pad(cdf1, (0, L - cdf1.numel()), value=1.0)
    if cdf2.numel() < L: cdf2 = F.pad(cdf2, (0, L - cdf2.numel()), value=1.0)
    emd = torch.sum(torch.abs(cdf1 - cdf2)).item()
    return max(0.0, 1.0 - emd / 2.0)


def _laplacian_spectrum_score(A_undir: torch.Tensor, topk: int = 32) -> torch.Tensor:
    n = A_undir.shape[0]
    if n == 0: return torch.zeros(0)
    deg = A_undir.sum(1)
    Dinv2 = torch.diag(1.0 / torch.sqrt(torch.clamp(deg, min=1.0)))
    L = torch.eye(n) - Dinv2 @ A_undir @ Dinv2
    evals = torch.linalg.eigvalsh(L).real
    k = min(topk, n)
    return evals[:k]


def _adj_singular_values(A_dir: torch.Tensor, topk: int = 32) -> torch.Tensor:
    n = A_dir.shape[0]
    if n == 0: return torch.zeros(0)
    svals = torch.linalg.svdvals(A_dir)
    k = min(topk, n)
    return svals[:k]


def _vec_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    k = max(a.numel(), b.numel())
    if a.numel() < k: a = F.pad(a, (0, k - a.numel()))
    if b.numel() < k: b = F.pad(b, (0, k - b.numel()))
    dist = torch.norm(a - b).item()
    return 1.0 / (1.0 + dist)


def _feature_to_tuple(feat: torch.Tensor, precision: int = 3) -> tuple:
    """Converts a continuous feature vector into a discrete hashable tuple."""
    if feat is None or feat.numel() == 0:
        return ("none",)
    return tuple(torch.round(feat * (10 ** precision)).tolist())


def _wl_histograms(
        edge_index: torch.Tensor,
        n: int,
        iters: int = 2,
        directed: bool = True,
        x: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None
) -> Dict:
    """
    Enhanced Weisfeiler-Lehman that integrates node and edge features.
    This ensures features are compared structurally, not just in aggregate.
    """
    src_list = edge_index[0].tolist() if edge_index.numel() > 0 else []
    dst_list = edge_index[1].tolist() if edge_index.numel() > 0 else []

    # Adjacency with feature binding
    # out_adj[u] = [(neighbor_v, edge_feat_uv), ...]
    out_adj = [[] for _ in range(n)]
    in_adj = [[] for _ in range(n)]

    for idx, (u, v) in enumerate(zip(src_list, dst_list)):
        e_feat = _feature_to_tuple(edge_attr[idx]) if edge_attr is not None else (1,)
        out_adj[u].append((v, e_feat))
        in_adj[v].append((u, e_feat))
        if not directed:
            out_adj[v].append((u, e_feat))
            in_adj[u].append((v, e_feat))

    # Initial labels: use node features if available, otherwise degree
    if x is not None:
        lbl = [_feature_to_tuple(x[i]) for i in range(n)]
    elif directed:
        lbl = [("din", len(in_adj[i]), "dout", len(out_adj[i])) for i in range(n)]
    else:
        lbl = [("deg", len(out_adj[i])) for i in range(n)]

    hist = Counter(lbl)

    for _ in range(iters):
        new_lbl = []
        for i in range(n):
            # Each neighbor sends its label AND the feature of the edge connecting them
            in_msg = tuple(sorted((lbl[v], e) for v, e in in_adj[i]))
            out_msg = tuple(sorted((lbl[v], e) for v, e in out_adj[i]))

            if directed:
                new_lbl.append((lbl[i], ("I", in_msg), ("O", out_msg)))
            else:
                new_lbl.append((lbl[i], ("N", out_msg)))
        lbl = new_lbl
        hist.update(lbl)

    return dict(hist)


def _hist_intersection_score(h1: Dict, h2: Dict) -> float:
    keys = set(h1) | set(h2)
    s_min = sum(min(h1.get(k, 0), h2.get(k, 0)) for k in keys)
    s_max = sum(max(h1.get(k, 0), h2.get(k, 0)) for k in keys)
    return s_min / s_max if s_max > 0 else 1.0


# ------------------------------
# Main Entry Point
# ------------------------------
def graph_similarity(
        edge_index1: torch.Tensor,
        edge_index2: torch.Tensor,
        *,
        n1: Optional[int] = None,
        n2: Optional[int] = None,
        x1: Optional[torch.Tensor] = None,
        x2: Optional[torch.Tensor] = None,
        edge_attr1: Optional[torch.Tensor] = None,
        edge_attr2: Optional[torch.Tensor] = None,
        directed: bool = True,
        topk_spec: int = 32,
        wl_iters: int = 2,
        weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    if weights is None:
        weights = {
            "size": 0.10,  # Basic density/count
            "degree": 0.15,  # Local connectivity
            "spectral": 0.25,  # Global structure
            "wl_feat": 0.50,  # Structural + Feature similarity (The most important)
        }

    # Pre-process
    edge_index1, edge_attr1 = coalesce(edge_index1, edge_attr1, num_nodes=n1)
    edge_index2, edge_attr2 = coalesce(edge_index2, edge_attr2, num_nodes=n2)
    n1 = _infer_num_nodes(edge_index1) if n1 is None else n1
    n2 = _infer_num_nodes(edge_index2) if n2 is None else n2

    # 1. Size/Density
    s_nodes = 1.0 if (n1 == n2 == 0) else (1.0 - abs(n1 - n2) / max(1, n1, n2))
    dens1 = edge_index1.size(1) / max(1, n1 * (n1 - 1))
    dens2 = edge_index2.size(1) / max(1, n2 * (n2 - 1))
    size_score = 0.5 * s_nodes + 0.5 * (1.0 - abs(dens1 - dens2))

    # 2. Degree Distribution
    A1_dir = _to_dense_adj(edge_index1, n1, directed=True)
    A2_dir = _to_dense_adj(edge_index2, n2, directed=True)
    deg1_out, deg1_in = A1_dir.sum(1), A1_dir.sum(0)
    deg2_out, deg2_in = A2_dir.sum(1), A2_dir.sum(0)
    degree_score = 0.5 * _emd1_cdf_score(_degree_hist_cdf(deg1_out), _degree_hist_cdf(deg2_out)) + \
                   0.5 * _emd1_cdf_score(_degree_hist_cdf(deg1_in), _degree_hist_cdf(deg2_in))

    # 3. Spectral
    A1_und = ((A1_dir + A1_dir.t()) > 0).float()
    A2_und = ((A2_dir + A2_dir.t()) > 0).float()
    s_lap = _vec_similarity(_laplacian_spectrum_score(A1_und, topk_spec),
                            _laplacian_spectrum_score(A2_und, topk_spec))
    s_svd = _vec_similarity(_adj_singular_values(A1_dir, topk_spec),
                            _adj_singular_values(A2_dir, topk_spec))
    spectral_score = 0.5 * s_lap + 0.5 * s_svd

    # 4. Attribute-Aware WL (Structural Feature Comparison)
    h1 = _wl_histograms(edge_index1, n1, iters=wl_iters, directed=directed, x=x1, edge_attr=edge_attr1)
    h2 = _wl_histograms(edge_index2, n2, iters=wl_iters, directed=directed, x=x2, edge_attr=edge_attr2)
    wl_feat_score = _hist_intersection_score(h1, h2)

    parts = {
        "size": size_score,
        "degree": degree_score,
        "spectral": spectral_score,
        "wl_feat": wl_feat_score
    }

    total_w = sum(weights.values())
    final_score = sum(weights[k] * parts[k] for k in parts if k in weights) / total_w
    return float(final_score), parts