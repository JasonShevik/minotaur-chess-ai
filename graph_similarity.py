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
    """Return dense adjacency (float32) on CPU for stability."""
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
    hist = torch.bincount(deg, minlength=m+1).float()
    cdf = hist.cumsum(0)
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    return cdf


def _emd1_cdf_score(cdf1: torch.Tensor, cdf2: torch.Tensor) -> float:
    # Pad to same length
    L = max(cdf1.numel(), cdf2.numel())
    if cdf1.numel() < L: cdf1 = F.pad(cdf1, (0, L - cdf1.numel()), value=1.0)
    if cdf2.numel() < L: cdf2 = F.pad(cdf2, (0, L - cdf2.numel()), value=1.0)
    emd = torch.sum(torch.abs(cdf1 - cdf2)).item()               # ∈ [0, 2]
    # Normalize to [0,1] and convert to similarity
    return max(0.0, 1.0 - emd / 2.0)


def _laplacian_spectrum_score(A_undir: torch.Tensor, topk: int = 32) -> float:
    n = A_undir.shape[0]
    if n == 0:
        return 1.0
    deg = A_undir.sum(1)
    Dinv2 = torch.diag(1.0 / torch.sqrt(torch.clamp(deg, min=1.0)))
    L = torch.eye(n) - Dinv2 @ A_undir @ Dinv2
    # Eigenvalues are real and in [0,2]; sort ascending
    evals = torch.linalg.eigvalsh(L).real
    k = min(topk, n)
    # Compare against itself? Caller passes both graphs; this helper returns vector
    return evals[:k]


def _adj_singular_values(A_dir: torch.Tensor, topk: int = 32) -> torch.Tensor:
    n = A_dir.shape[0]
    if n == 0:
        return torch.zeros(0)
    # Singular values are permutation invariant; largest first
    svals = torch.linalg.svdvals(A_dir)
    k = min(topk, n)
    return svals[:k]


def _vec_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    # Pad shorter with zeros
    k = max(a.numel(), b.numel())
    if a.numel() < k: a = F.pad(a, (0, k - a.numel()))
    if b.numel() < k: b = F.pad(b, (0, k - b.numel()))
    # L2 difference → [0,1] via 1/(1+norm)
    dist = torch.norm(a - b).item()
    return 1.0 / (1.0 + dist)


def _wl_histograms(edge_index: torch.Tensor, n: int, iters: int = 2, directed: bool = True) -> Dict[int, int]:
    # Build adjacency
    src = edge_index[0].tolist() if edge_index.numel() > 0 else []
    dst = edge_index[1].tolist() if edge_index.numel() > 0 else []
    out_adj = [[] for _ in range(n)]
    in_adj  = [[] for _ in range(n)]
    for u, v in zip(src, dst):
        out_adj[u].append(v)
        in_adj[v].append(u)
        if not directed:
            out_adj[v].append(u); in_adj[u].append(v)

    # Initial canonical labels (tuples built from degrees)
    if directed:
        lbl = [("din", len(in_adj[i]), "dout", len(out_adj[i])) for i in range(n)]
    else:
        lbl = [("deg", len(out_adj[i])) for i in range(n)]

    hist = Counter(lbl)

    for _ in range(iters):
        new_lbl = []
        for i in range(n):
            in_multiset  = tuple(sorted(lbl[j] for j in in_adj[i]))
            out_multiset = tuple(sorted(lbl[j] for j in out_adj[i]))
            if directed:
                new_lbl.append((lbl[i], ("I", in_multiset), ("O", out_multiset)))
            else:
                both = tuple(sorted([*(lbl[j] for j in in_adj[i]), *(lbl[j] for j in out_adj[i])]))
                new_lbl.append((lbl[i], ("N", both)))
        lbl = new_lbl
        hist.update(lbl)

    return dict(hist)  # keys are hashable tuples; works with your _hist_intersection_score


def _hist_intersection_score(h1: Dict[int,int], h2: Dict[int,int]) -> float:
    keys = set(h1) | set(h2)
    s_min = sum(min(h1.get(k,0), h2.get(k,0)) for k in keys)
    s_max = sum(max(h1.get(k,0), h2.get(k,0)) for k in keys)
    if s_max == 0:
        return 1.0
    return s_min / s_max


def _rbf_mmd_score(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> float:
    """Return similarity in [0,1] from RBF MMD^2 via 1/(1+MMD^2)."""
    if X is None or Y is None or X.numel()==0 or Y.numel()==0:
        return 1.0
    X = X.float().cpu(); Y = Y.float().cpu()
    def pdist2(a, b):
        a2 = (a*a).sum(1, keepdim=True)
        b2 = (b*b).sum(1, keepdim=True).t()
        return a2 + b2 - 2.0 * (a @ b.t())
    Kxx = torch.exp(-pdist2(X, X) / (2*sigma*sigma))
    Kyy = torch.exp(-pdist2(Y, Y) / (2*sigma*sigma))
    Kxy = torch.exp(-pdist2(X, Y) / (2*sigma*sigma))
    # Remove diagonal for unbiased estimate
    n = X.size(0); m = Y.size(0)
    if n > 1:
        Kxx = (Kxx.sum() - Kxx.diag().sum()) / (n*(n-1))
    else:
        Kxx = torch.tensor(0.0)
    if m > 1:
        Kyy = (Kyy.sum() - Kyy.diag().sum()) / (m*(m-1))
    else:
        Kyy = torch.tensor(0.0)
    Kxy = Kxy.mean()
    mmd2 = (Kxx + Kyy - 2*Kxy).clamp(min=0.0).item()
    return 1.0 / (1.0 + mmd2)


# ------------------------------
# Main entry point
# ------------------------------
def graph_similarity(
    edge_index1: torch.Tensor,
    edge_index2: torch.Tensor,
    *,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
    x1: Optional[torch.Tensor] = None,          # [n1, D1] node features (optional)
    x2: Optional[torch.Tensor] = None,          # [n2, D2] (if dims differ, we compare after PCA-free stats via MMD)
    edge_attr1: Optional[torch.Tensor] = None,  # [E1, F] edge features (optional)
    edge_attr2: Optional[torch.Tensor] = None,  # [E2, F]
    directed: bool = True,
    topk_spec: int = 32,
    wl_iters: int = 2,
    sigma_node: float = 1.0,
    sigma_edge: float = 1.0,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compare two graphs and return (score in [0,1], details).
    Invariance: permutation of node IDs. Robust to small structural noise.
    """
    # Defaults for mixing
    if weights is None:
        weights = {
            "size":      0.10,  # node/edge count, density
            "degree":    0.20,  # in/out CDF-EMD
            "spectral":  0.30,  # Laplacian eigen + adj. singular
            "wl":        0.25,  # Weisfeiler–Lehman hist intersection
            "nodefeat":  0.10,  # RBF-MMD over node features
            "edgefeat":  0.05,  # RBF-MMD over edge features
        }

    # Coalesce duplicates (keep directedness)
    edge_index1, _ = coalesce(edge_index1, None, num_nodes=None)
    edge_index2, _ = coalesce(edge_index2, None, num_nodes=None)

    # Sizes
    n1 = _infer_num_nodes(edge_index1) if n1 is None else n1
    n2 = _infer_num_nodes(edge_index2) if n2 is None else n2
    E1 = edge_index1.size(1)
    E2 = edge_index2.size(1)

    # Dense adjacencies
    A1_dir = _to_dense_adj(edge_index1, n1, directed=True)
    A2_dir = _to_dense_adj(edge_index2, n2, directed=True)
    A1_und = ((A1_dir + A1_dir.t()) > 0).float()
    A2_und = ((A2_dir + A2_dir.t()) > 0).float()

    # ---- size / density ----
    s_nodes = 1.0 if (n1 == 0 and n2 == 0) else (1.0 - abs(n1 - n2) / max(1, max(n1, n2)))
    dens1 = E1 / max(1, n1*(n1 - 1))
    dens2 = E2 / max(1, n2*(n2 - 1))
    s_edges = 1.0 - abs(dens1 - dens2)
    size_score = max(0.0, 0.5*s_nodes + 0.5*s_edges)

    # ---- degree distributions (directed aware) ----
    deg1_out = A1_dir.sum(1); deg1_in = A1_dir.sum(0)
    deg2_out = A2_dir.sum(1); deg2_in = A2_dir.sum(0)
    cdf1_o = _degree_hist_cdf(deg1_out); cdf2_o = _degree_hist_cdf(deg2_out)
    cdf1_i = _degree_hist_cdf(deg1_in);  cdf2_i = _degree_hist_cdf(deg2_in)
    degree_score = 0.5*_emd1_cdf_score(cdf1_o, cdf2_o) + 0.5*_emd1_cdf_score(cdf1_i, cdf2_i)
    if not directed:
        # if you know graphs are undirected, you can simplify to total degree
        pass

    # ---- spectral (permutation invariant) ----
    # Undirected normalized Laplacian eigenvalues
    evals1 = _laplacian_spectrum_score(A1_und, topk=topk_spec)
    evals2 = _laplacian_spectrum_score(A2_und, topk=topk_spec)
    s_lap = _vec_similarity(evals1, evals2)
    # Directed adjacency singular values (structure with directions)
    svals1 = _adj_singular_values(A1_dir, topk=topk_spec)
    svals2 = _adj_singular_values(A2_dir, topk=topk_spec)
    s_svd = _vec_similarity(svals1, svals2)
    spectral_score = 0.5*s_lap + 0.5*s_svd

    # ---- Weisfeiler–Lehman subtree pattern similarity ----
    h1 = _wl_histograms(edge_index1, n1, iters=wl_iters, directed=directed)
    h2 = _wl_histograms(edge_index2, n2, iters=wl_iters, directed=directed)
    wl_score = _hist_intersection_score(h1, h2)

    # ---- Node feature distribution similarity (optional) ----
    nodefeat_score = 1.0
    if x1 is not None and x2 is not None and x1.numel() > 0 and x2.numel() > 0:
        # If feature dims differ, project to common dim by truncation/padding
        d1, d2 = x1.size(1), x2.size(1)
        d = max(d1, d2)
        xx1 = x1.float().cpu(); xx2 = x2.float().cpu()
        if d1 < d: xx1 = F.pad(xx1, (0, d - d1))
        if d2 < d: xx2 = F.pad(xx2, (0, d - d2))
        nodefeat_score = _rbf_mmd_score(xx1, xx2, sigma=sigma_node)

    # ---- Edge feature distribution similarity (optional) ----
    edgefeat_score = 1.0
    if edge_attr1 is not None and edge_attr2 is not None and edge_attr1.numel() > 0 and edge_attr2.numel() > 0:
        f1, f2 = edge_attr1.size(-1), edge_attr2.size(-1)
        f = max(f1, f2)
        ea1 = edge_attr1.float().cpu(); ea2 = edge_attr2.float().cpu()
        if f1 < f: ea1 = F.pad(ea1, (0, f - f1))
        if f2 < f: ea2 = F.pad(ea2, (0, f - f2))
        edgefeat_score = _rbf_mmd_score(ea1, ea2, sigma=sigma_edge)

    # ---- Combine
    parts = {
        "size": size_score,
        "degree": degree_score,
        "spectral": spectral_score,
        "wl": wl_score,
        "nodefeat": nodefeat_score,
        "edgefeat": edgefeat_score,
    }
    total_w = sum(weights.values())
    score = sum(weights[k]*parts[k] for k in parts if k in weights) / max(1e-9, total_w)
    # Clamp to [0,1] for safety
    score = float(min(1.0, max(0.0, score)))
    return score, {k: float(v) for k, v in parts.items()}
