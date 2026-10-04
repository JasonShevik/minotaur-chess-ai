import random
import math
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.optim import Adam
from torch_geometric.utils import coalesce
from torch_geometric.data import Batch
from torch_scatter import scatter_softmax, scatter_sum, scatter_max
from typing import Optional, Dict, Tuple, List, Set, Any
from dataclasses import dataclass, field


def mlp(sizes, last_activation=None, norm="none", lrelu_slope=0.05):
    """A simple MLP factory."""
    layers = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if norm == "batch":
            layers += [nn.BatchNorm1d(sizes[i + 1])]
        elif norm == "layer":
            layers += [nn.LayerNorm(sizes[i + 1])]
        layers += [nn.LeakyReLU(lrelu_slope, inplace=True)]
    layers += [nn.Linear(sizes[-2], sizes[-1])]
    if last_activation is not None:
        layers += [last_activation]
    return nn.Sequential(*layers)


def _segment_logsumexp_with_extra(values: torch.Tensor, seg: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
    """
    For every segment s: log( sum over k with seg[k] == s of exp(values[k])  +  exp(extra[s]) ).

    values [K], seg [K] with ids in [0, S), extra [S]  ->  [S]. Used for the preference score,
    where each child's scores are normalized across all of its parent's neighbors plus one extra
    "zero preference" slot. Numerically stable; segments with no rows reduce to extra[s].
    """
    if values.numel() == 0:
        return extra
    seg_max = scatter_max(values.detach(), seg, dim=0, dim_size=extra.size(0))[0]
    m = torch.maximum(extra.detach(), seg_max)
    total = scatter_sum(torch.exp(values - m[seg]), seg, dim=0, dim_size=extra.size(0))
    return m + torch.log(total + torch.exp(extra - m))


def _bernoulli_terms(p: torch.Tensor, outcome: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Summed log-probability of the observed outcomes, and summed entropy, of Bernoulli(p) draws."""
    logp = torch.where(outcome, torch.log(p), torch.log1p(-p)).sum()
    ent = -(p * torch.log(p) + (1 - p) * torch.log1p(-p)).sum()
    return logp, ent


class GuoUnpool(nn.Module):
    """
    The unpooling layer of Guo, Zou and Lerman, "An Unpooling Layer for Graph Generation"
    (arXiv 2206.01874), for UNDIRECTED graphs, following appendix A of the paper step by step.

    Each node is either kept (static) or split into two children. The layer then decides which
    edges the new graph has, and computes features for every node and edge. Every structural
    decision is sampled, and forward returns the log-probability and entropy of all of them so
    the layer can be trained with REINFORCE or PPO, as in the paper.

    Steps (numbering from the paper's appendix A):
        1a  Decide which nodes are unpooled: Bernoulli(MLP-R(x_j)) for every node in I_r.
        1b  Node features: y = MLP-y(PS1 x) for static nodes; each child gets MLP-y(PS1 x) or
            MLP-y(PS2 x), where PS1 and PS2 are overlapping projections of the parent's features.
        2a  Intra-links: link the two children of j with probability MLP-IA. V_c = the linked ones.
        2b  For every unpooled j without an intra-link, choose one neighbor b_j (categorical over
            all of j's edges, scored by h_C = MLP-IE-2). Both of j's children connect to it, which
            is what guarantees the output graph stays connected.
        2c  Inter-links: for every edge {i, j} and every unpooled endpoint, choose which of that
            endpoint's children take part: child 1 only, child 2 only, or both. All pairs between
            the two endpoints' chosen sets are linked. The edge to b_j uses both children, unsampled.
        2d  For every edge whose two ends were both unpooled, with probability MLP-IE-A add one more
            edge: if each end used one child, link the two leftover children; if one end used one
            child and the other used both, link the leftover child to one of the other end's children
            (chosen with odds p1 : p2 from step 2c); if both used both, nothing is added.
        3   Edge features: u_kl = MLP-u(agg(y_k, y_l)).

    Step 2c scoring. With use_preference=False the three options of an endpoint-edge pair are a
    softmax of MLP-IE-1(y1, w, x_other), MLP-IE-1(y2, w, x_other) and MLP-IE-2(agg(y1, y2), w,
    x_other). With use_preference=True (the paper's preference score, supplement C.2) each of the
    three options is first normalized across ALL of the parent's neighbors together with a learned
    "zero preference" slot, and then the three are renormalized per edge. That is the same as the
    plain softmax with a per-option offset measuring how much the option likes the whole
    neighborhood, so an option wins an edge when that neighbor matters more to it than its other
    neighbors do; the paper uses it to stop one child inheriting every edge. Either way exactly
    three outcomes are sampled, and the probability logged is the probability sampled from.

    Graph conventions:
      * Input: edge_index may list each undirected edge once or in both directions, and may hold
        parallel edges (one per chess edge type, for example). They are merged into one edge per
        unordered pair, combining attributes with an elementwise max, so one-hot edge types become
        a multi-hot. Both directions of an edge should carry the same attributes. Self-loops are
        dropped, because the paper's construction is for simple graphs.
      * Output: each undirected edge in both directions with identical attributes, the PyG
        convention, ready for message passing and for the next unpooling layer.

    Two choices the paper leaves open, made here for undirected graphs:
      * MLP-IE-A is written MLP-IE-A(x_i, x_j, w_ij), which depends on the order of i and j. An
        undirected edge has no order, so the probability is the mean over both orders.
      * In 2d, the odds p1 : p2 come from step 2c for that edge. When the end with both children got
        them through step 2b, those probabilities were not sampled from in 2c, but they are defined
        by the same formula, so they are computed the same way.

    Replay: pass the `actions_recorded` of an earlier call as `actions_to_replay` to rebuild exactly
    the same output graph under the current parameters, with the log-probability of those same
    decisions recomputed differentiably. That is what PPO's probability ratio needs.
    """

    def __init__(
        self,
        dx, dw, dy, du,
        kv=128, kia=128, kie=128, kw=128,
        use_preference=True
    ):
        super().__init__()
        self.dx, self.dw, self.dy, self.du = dx, dw, dy, du
        self.use_preference = use_preference
        # Policy smoothing: floor/ceiling on every action probability. Prevents
        # saturation (p -> 0/1), which bounds d(logP)/dstep and keeps per-update
        # trajectory KL finite
        self.p_eps = 0.01

        # PS1/PS2 projection indices (d' = floor(dx/2) + floor(dx/4))
        ds = dx // 2
        D  = dx // 4
        self.register_buffer("_ps1_idx", torch.tensor(list(range(ds)) + list(range(ds, ds + D)), dtype=torch.long))
        self.register_buffer("_ps2_idx", torch.tensor(list(range(ds)) + list(range(ds + D, ds + 2 * D)), dtype=torch.long))
        d_prime = ds + D

        # heads
        self.mlp_y   = mlp([d_prime, kv, dy], norm="layer")
        self.mlp_ia  = mlp([dy, kia, 1], last_activation=nn.Sigmoid(), norm="layer")
        self.mlp_ie1 = mlp([dy + dw + dx, kie, 1], norm="layer")
        self.mlp_ie2 = mlp([dy + dw + dx, kie, 1], norm="layer")
        self.mlp_c   = self.mlp_ie2  # the paper defines h_C (step 2b) as MLP-IE-2

        if self.use_preference:
            self.mlp_zero_s = mlp([dy, 2 * dy, 1], norm="layer")   # zero preference of one child
            self.mlp_zero_b = mlp([dx, 2 * dx, 1], norm="layer")   # zero preference of "both children"

        self.mlp_r    = mlp([dx, max(1, dx // 2), 1], last_activation=nn.Sigmoid(), norm="layer")
        self.mlp_ie_a = mlp([dx + dx + dw, kie, 1], last_activation=nn.Sigmoid(), norm="layer")
        self.mlp_u    = mlp([dy, kw, du], norm="layer")

    # ----- small helpers -----

    @staticmethod
    def agg(a, b):
        return F.leaky_relu(a + b, negative_slope=0.05)

    def _smooth_bern(self, p: torch.Tensor) -> torch.Tensor:
        """Clamp a Bernoulli probability into [eps, 1-eps]."""
        return p.clamp(self.p_eps, 1.0 - self.p_eps)

    def _smooth_cat(self, probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Mix a categorical distribution with uniform: (1-eps)*p + eps/L."""
        L = probs.size(dim)
        return probs * (1.0 - self.p_eps) + self.p_eps / L

    def _project(self, x):
        return x[:, self._ps1_idx], x[:, self._ps2_idx]

    @staticmethod
    def canonicalize_undirected(
        edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One row per unordered pair {a, b} with a < b, sorted, self-loops dropped. Parallel edges
        and the two directions of an edge are merged with an elementwise max of their attributes.
        Returns (pairs [2, M], attr [M, dw]).
        """
        a = torch.minimum(edge_index[0], edge_index[1])
        b = torch.maximum(edge_index[0], edge_index[1])
        keep = a != b
        pairs = torch.stack([a[keep], b[keep]])
        attr = edge_attr[keep]
        if pairs.size(1) == 0:
            return pairs, attr
        return coalesce(pairs, attr, num_nodes=num_nodes, reduce="max")

    def interlink_probabilities(
        self,
        y1: torch.Tensor, y2: torch.Tensor, w: torch.Tensor, x_other: torch.Tensor,
        seg: torch.Tensor, y1_parent: torch.Tensor, y2_parent: torch.Tensor, x_parent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 2c probabilities for every incidence of an unpooled node e with one of its neighbors.

        Per incidence k: y1, y2 [K, dy] are e's children features, w [K, dw] the edge features,
        x_other [K, dx] the neighbor's input features, seg [K] the index of e among the unpooled
        nodes. Per unpooled node: y1_parent, y2_parent [U, dy] and x_parent [U, dx], used for the
        preference score's zero slots.

        Returns (probs [K, 3] over (child 1 only, child 2 only, both), smoothed, which is exactly
        the distribution sampled from and logged; h_C [K], the raw MLP-IE-2 score, which step 2b
        reuses as the paper prescribes).
        """
        s1 = self.mlp_ie1(torch.cat([y1, w, x_other], dim=1)).squeeze(-1)
        s2 = self.mlp_ie1(torch.cat([y2, w, x_other], dim=1)).squeeze(-1)
        sb = self.mlp_ie2(torch.cat([self.agg(y1, y2), w, x_other], dim=1)).squeeze(-1)
        if self.use_preference:
            # Normalize each option across all of the parent's neighbors plus its zero slot, then
            # renormalize per edge: softmax of (score - log partition) for each of the three options.
            z1 = self.mlp_zero_s(y1_parent).squeeze(-1)
            z2 = self.mlp_zero_s(y2_parent).squeeze(-1)
            zb = self.mlp_zero_b(x_parent).squeeze(-1)
            l1 = s1 - _segment_logsumexp_with_extra(s1, seg, z1)[seg]
            l2 = s2 - _segment_logsumexp_with_extra(s2, seg, z2)[seg]
            lb = sb - _segment_logsumexp_with_extra(sb, seg, zb)[seg]
        else:
            l1, l2, lb = s1, s2, sb
        probs = self._smooth_cat(F.softmax(torch.stack([l1, l2, lb], dim=1), dim=1), dim=1)
        return probs, sb

    # ----- forward -----

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        I_s: torch.Tensor | None = None,
        I_u: torch.Tensor | None = None,
        I_r: torch.Tensor | None = None,
        actions_to_replay: Optional[Dict] = None,
        rng: Optional[torch.Generator] = None,
    ):
        """
        x [N, dx]; edge_index [2, E] (undirected, see the class docstring); edge_attr [E, dw] or None.
        I_s / I_u: nodes forced static / forced unpooled; I_r: nodes decided by MLP-R (default: the rest).

        Returns:
          x_out, edge_index_out, edge_attr_out, logP, total_entropy, parent_map, sets, actions_recorded
        """
        device = x.device
        N = x.size(0)
        replay = actions_to_replay is not None
        rec: Dict[str, Any] = {}
        if edge_attr is None:
            edge_attr = x.new_zeros(edge_index.size(1), self.dw)

        pairs, W = self.canonicalize_undirected(edge_index.to(device), edge_attr.to(device), N)
        A, B = pairs[0], pairs[1]
        M = pairs.size(1)
        long_ = dict(dtype=torch.long, device=device)

        logP = x.new_zeros(())
        total_entropy = x.new_zeros(())

        # ========== Step 1a: which nodes are unpooled ==========
        all_idx = torch.arange(N, device=device)
        I_s = torch.tensor([], **long_) if I_s is None else I_s.to(device)
        I_u = torch.tensor([], **long_) if I_u is None else I_u.to(device)
        if I_r is None:
            mask = torch.ones(N, dtype=torch.bool, device=device)
            mask[I_s] = False
            mask[I_u] = False
            I_r = all_idx[mask]
        else:
            I_r = I_r.to(device)

        pr = self._smooth_bern(self.mlp_r(x[I_r]).squeeze(-1))
        if replay:
            choose_unpool = actions_to_replay["step1a_unpool"][0].to(device)
        else:
            choose_unpool = torch.rand(pr.shape, dtype=pr.dtype, device=device, generator=rng) < pr
        rec["step1a_unpool"] = [choose_unpool]
        lp, ent = _bernoulli_terms(pr, choose_unpool)
        logP, total_entropy = logP + lp, total_entropy + ent

        Iu = torch.cat([I_u, I_r[choose_unpool]])
        Is = torch.cat([I_s, I_r[~choose_unpool]])
        ns, nu = Is.numel(), Iu.numel()

        # ========== Step 1b: output nodes and features ==========
        PS1, PS2 = self._project(x)
        y = torch.cat([self.mlp_y(PS1[Is]), self.mlp_y(PS1[Iu]), self.mlp_y(PS2[Iu])], dim=0)
        f = torch.full((N,), -1, **long_);  f[Is] = torch.arange(ns, **long_)
        f1 = torch.full((N,), -1, **long_); f1[Iu] = ns + torch.arange(nu, **long_)
        f2 = torch.full((N,), -1, **long_); f2[Iu] = ns + nu + torch.arange(nu, **long_)
        unpooled = torch.zeros(N, dtype=torch.bool, device=device); unpooled[Iu] = True
        iu_pos = torch.full((N,), -1, **long_); iu_pos[Iu] = torch.arange(nu, **long_)
        y1u, y2u = y[ns:ns + nu], y[ns + nu:]

        # ========== Step 2a: intra-links ==========
        if nu > 0:
            pc = self._smooth_bern(self.mlp_ia(self.agg(y1u, y2u)).squeeze(-1))
            if replay:
                Vc_mask = actions_to_replay["step2a_intra"][0].to(device)
            else:
                Vc_mask = torch.rand(pc.shape, dtype=pc.dtype, device=device, generator=rng) < pc
            lp, ent = _bernoulli_terms(pc, Vc_mask)
            logP, total_entropy = logP + lp, total_entropy + ent
        else:
            Vc_mask = torch.zeros(0, dtype=torch.bool, device=device)
        rec["step2a_intra"] = [Vc_mask]
        Vc = Iu[Vc_mask]
        in_vc = torch.zeros(N, dtype=torch.bool, device=device); in_vc[Vc] = True
        intra_edges = torch.stack([f1[Vc], f2[Vc]])

        # ---- incidences: (edge, unpooled endpoint e, other endpoint o) ----
        m_idx = torch.arange(M, **long_)
        a_unp, b_unp = unpooled[A], unpooled[B]
        n_a = int(a_unp.sum())
        inc_m = torch.cat([m_idx[a_unp], m_idx[b_unp]])
        inc_e = torch.cat([A[a_unp], B[b_unp]])
        inc_o = torch.cat([B[a_unp], A[b_unp]])
        K = inc_m.numel()
        inc_of_a = torch.full((M,), -1, **long_); inc_of_a[a_unp] = torch.arange(n_a, **long_)
        inc_of_b = torch.full((M,), -1, **long_); inc_of_b[b_unp] = n_a + torch.arange(K - n_a, **long_)

        if K > 0:
            seg = iu_pos[inc_e]
            P, h_c = self.interlink_probabilities(
                y1u[seg], y2u[seg], W[inc_m], x[inc_o], seg, y1u, y2u, x[Iu]
            )
        else:
            P = x.new_zeros(0, 3)
            h_c = x.new_zeros(0)

        # ========== Step 2b: a guaranteed neighbor for children without an intra-link ==========
        b_of = torch.full((N,), -1, **long_)
        picks: Dict[int, int] = {}
        for j in Iu[~Vc_mask].tolist():
            rows = (inc_e == j).nonzero(as_tuple=False).flatten()
            if rows.numel() == 0:
                continue  # an isolated node: nothing to connect to
            probs = self._smooth_cat(F.softmax(h_c[rows], dim=0), dim=0)
            if replay:
                nei = actions_to_replay["step2b_pick"][j]
                hit = (inc_o[rows] == nei).nonzero(as_tuple=False).flatten()
                if hit.numel() == 0:
                    raise RuntimeError(f"Step 2b replay: node {j} has no neighbor {nei}")
                pick = int(hit[0])
            else:
                pick = int(torch.multinomial(probs, 1, generator=rng))
            picks[j] = int(inc_o[rows[pick]])
            b_of[j] = picks[j]
            logP = logP + torch.log(probs[pick])
            total_entropy = total_entropy - (probs * torch.log(probs)).sum()
        rec["step2b_pick"] = picks

        # ========== Step 2c: inter-links ==========
        # An incidence is forced to "both children" when it is the step 2b edge; all others are sampled.
        forced = (~in_vc[inc_e]) & (b_of[inc_e] == inc_o)
        dec = (~forced).nonzero(as_tuple=False).flatten()
        keys = [(int(A[inc_m[k]]), int(B[inc_m[k]]), int(inc_e[k])) for k in dec.tolist()]
        Pd = P[dec]
        if replay:
            recorded = actions_to_replay["step2c_choice"]
            choice_dec = torch.tensor([recorded[key] for key in keys], **long_)
        else:
            u = torch.rand(dec.numel(), dtype=P.dtype, device=device, generator=rng)
            # 0: child 1 only (u < p1); 1: child 2 only; 2: both (u >= p1 + p2)
            choice_dec = (u >= Pd[:, 0]).long() + (u >= Pd[:, 0] + Pd[:, 1]).long()
        rec["step2c_choice"] = {key: int(c) for key, c in zip(keys, choice_dec.tolist())}
        if dec.numel() > 0:
            logP = logP + torch.log(Pd.gather(1, choice_dec.unsqueeze(1))).sum()
            total_entropy = total_entropy - (Pd * torch.log(Pd)).sum()
        choice = torch.full((K,), 2, **long_)
        choice[dec] = choice_dec

        def endpoint_slots(node: torch.Tensor, inc_of: torch.Tensor) -> torch.Tensor:
            """[M, 2] output nodes each edge's endpoint takes part with: (child 1 or the static node, child 2), -1 if absent."""
            slots = torch.full((M, 2), -1, **long_)
            static = ~unpooled[node]
            slots[static, 0] = f[node[static]]
            has = inc_of >= 0
            c = choice[inc_of[has]]
            none = torch.full_like(c, -1)
            slots[has, 0] = torch.where(c != 1, f1[node[has]], none)
            slots[has, 1] = torch.where(c != 0, f2[node[has]], none)
            return slots

        SA, SB = endpoint_slots(A, inc_of_a), endpoint_slots(B, inc_of_b)
        cand_a, cand_b = SA[:, [0, 0, 1, 1]], SB[:, [0, 1, 0, 1]]
        ok = (cand_a >= 0) & (cand_b >= 0)
        inter_edges = torch.stack([cand_a[ok], cand_b[ok]])

        # ========== Step 2d: additional edges between children pairs ==========
        extra_edges: List[torch.Tensor] = []
        rec["step2d_pa"], rec["step2d_r"] = {}, {}
        eu = (a_unp & b_unp).nonzero(as_tuple=False).flatten()
        if eu.numel() > 0:
            xa, xb, w = x[A[eu]], x[B[eu]], W[eu]
            pa = self._smooth_bern(0.5 * (
                self.mlp_ie_a(torch.cat([xa, xb, w], dim=1)) + self.mlp_ie_a(torch.cat([xb, xa, w], dim=1))
            ).squeeze(-1))
            pair_keys = [(int(A[m]), int(B[m])) for m in eu.tolist()]
            if replay:
                chosen = torch.tensor([bool(actions_to_replay["step2d_pa"][k]) for k in pair_keys],
                                      dtype=torch.bool, device=device)
            else:
                chosen = torch.rand(pa.shape, dtype=pa.dtype, device=device, generator=rng) < pa
            rec["step2d_pa"] = {k: bool(c) for k, c in zip(pair_keys, chosen.tolist())}
            lp, ent = _bernoulli_terms(pa, chosen)
            logP, total_entropy = logP + lp, total_entropy + ent

            size_a = (SA[eu] >= 0).sum(1)
            size_b = (SB[eu] >= 0).sum(1)
            # The child an end did NOT use, when it used exactly one
            left_a = torch.where(SA[eu, 0] >= 0, f2[A[eu]], f1[A[eu]])
            left_b = torch.where(SB[eu, 0] >= 0, f2[B[eu]], f1[B[eu]])

            # Case 1: each end used one child -> link the two leftover children
            case1 = chosen & (size_a == 1) & (size_b == 1)
            extra_edges.append(torch.stack([left_a[case1], left_b[case1]]))

            # Case 2: one end used one child, the other both -> link the leftover child to child r of the other end
            case2 = (chosen & ((size_a + size_b) == 3)).nonzero(as_tuple=False).flatten()
            if case2.numel() > 0:
                a_small = size_a[case2] == 1
                m2 = eu[case2]
                inc_big = torch.where(a_small, inc_of_b[m2], inc_of_a[m2])
                p_big = P[inc_big]
                q1 = (p_big[:, 0] / (p_big[:, 0] + p_big[:, 1])).clamp(1e-6, 1 - 1e-6)
                keys2 = [pair_keys[i] for i in case2.tolist()]
                if replay:
                    r = torch.tensor([int(actions_to_replay["step2d_r"][k]) for k in keys2], **long_)
                else:
                    r = 1 + (torch.rand(q1.shape, dtype=q1.dtype, device=device, generator=rng) >= q1).long()
                rec["step2d_r"] = {k: int(v) for k, v in zip(keys2, r.tolist())}
                lp, ent = _bernoulli_terms(q1, r == 1)
                logP, total_entropy = logP + lp, total_entropy + ent
                small_left = torch.where(a_small, left_a[case2], left_b[case2])
                big_node = torch.where(a_small, B[m2], A[m2])
                big_child = torch.where(r == 1, f1[big_node], f2[big_node])
                extra_edges.append(torch.stack([small_left, big_child]))

        # ========== Step 3: assemble the undirected output graph and its edge features ==========
        n_out = y.size(0)
        und = torch.cat([intra_edges, inter_edges] + extra_edges, dim=1)
        if und.size(1) > 0:
            edge_index_out = coalesce(torch.cat([und, und.flip(0)], dim=1), num_nodes=n_out)
            edge_attr_out = self.mlp_u(self.agg(y[edge_index_out[0]], y[edge_index_out[1]]))
        else:
            edge_index_out = torch.empty(2, 0, **long_)
            edge_attr_out = y.new_zeros(0, self.du)

        parent_map = {
            "f": {int(i): int(f[i]) for i in Is.tolist()},
            "f1": {int(i): int(f1[i]) for i in Iu.tolist()},
            "f2": {int(i): int(f2[i]) for i in Iu.tolist()},
        }
        sets = {"Is": Is, "Iu": Iu, "Vc": Vc}

        sig_out = {"N": int(n_out), "E": int(edge_index_out.size(1)),
                   "edges": tuple(map(tuple, edge_index_out.t().tolist()))}
        if replay and "__sig_out__" in actions_to_replay:
            assert actions_to_replay["__sig_out__"] == sig_out, "Replay rebuilt a different output graph"
        rec["__sig_out__"] = sig_out

        return y, edge_index_out, edge_attr_out, logP, total_entropy, parent_map, sets, rec


# ============================================================================
# Test-problem utilities: random target graphs and the unpooling rollout
# ============================================================================

def seed_graph():
    e = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long)
    return e


def random_undirected_graph_with_features(
    n: int,
    *,
    p_extra: float = 0.2,              # probability for each remaining unordered pair
    node_feat_dim: int = 8,            # per-node random features (fixed dim)
    desc: torch.Tensor | None = None,  # optional [desc_dim]; broadcast to all nodes
    desc_dim: int = 0,                 # if desc is None and desc_dim>0, sample a random desc
    include_degree_feats: bool = True, # append normalized degree per node
    edge_feat_dim: int = 0,            # 0 → no edge_attr, >0 → return edge_attr
    edge_feat_style: str = "gaussian", # "gaussian" | "zeros"
    device: torch.device | str | None = None,
    rng: torch.Generator | None = None,
):
    """
    A random connected undirected graph: a random spanning tree, plus every remaining unordered
    pair with probability p_extra. No self-loops.

    Returns:
      x_raw:        [n, D] node features (fixed dimensional)
      edge_index:   [2, E] each undirected edge in both directions (PyG convention)
      edge_attr:    [E, edge_feat_dim] (the same features in both directions) or None
      meta:         dict with helper info

    Node features are fixed-dimensional across graphs (so the encoder can be a single MLP):
      [ desc (broadcast) | per-node gaussian | (optional) normalized degree ]
    """
    assert n >= 3, "Need at least 3 nodes"
    cpu = torch.device("cpu")
    if device is None:
        device = cpu
    if rng is None:
        rng = torch.Generator(device=cpu).manual_seed(torch.seed())

    pairs = set()
    # Connectivity: random spanning tree
    for i in range(1, n):
        p = torch.randint(low=0, high=i, size=(1,), generator=rng, device=cpu).item()
        pairs.add((p, i))
    # Extra edges among the remaining unordered pairs
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in pairs:
                continue
            if torch.rand((), generator=rng, device=cpu) < p_extra:
                pairs.add((u, v))
    ordered = sorted(pairs)

    und = torch.tensor(ordered, dtype=torch.long, device=cpu).t().contiguous()
    edge_index = torch.cat([und, und.flip(0)], dim=1).to(device)

    parts = []
    if desc is None and desc_dim > 0:
        d = torch.randn(desc_dim, generator=rng, device=cpu)
        desc = d / (d.norm(p=2) + 1e-8)
    if desc is not None:
        assert desc.dim() == 1, "desc must be a 1D vector"
        parts.append(desc.to(cpu).unsqueeze(0).repeat(n, 1))
    if node_feat_dim > 0:
        parts.append(torch.randn(n, node_feat_dim, generator=rng, device=cpu))
    if include_degree_feats:
        deg = torch.zeros(n, device=cpu)
        for u, v in ordered:
            deg[u] += 1
            deg[v] += 1
        parts.append((deg / max(1, n - 1)).unsqueeze(1))
    x_raw = torch.cat(parts, dim=1) if parts else torch.zeros(n, 0, device=cpu)
    x_raw = x_raw.to(device)

    edge_attr = None
    if edge_feat_dim > 0:
        if edge_feat_style == "gaussian":
            half = torch.randn(len(ordered), edge_feat_dim, generator=rng, device=cpu)
        elif edge_feat_style == "zeros":
            half = torch.zeros(len(ordered), edge_feat_dim, device=cpu)
        else:
            raise ValueError(f"Unsupported edge_feat_style: {edge_feat_style}")
        edge_attr = torch.cat([half, half], dim=0).to(device)

    meta = {"desc": None if desc is None else desc.to(device), "p_extra": float(p_extra), "n": n}
    return x_raw, edge_index, edge_attr, meta


# ============================================================================
# Components hoisted out of __main__ so the DEHB objective can reuse them.
# Logic is unchanged from the original script except where noted.
# ============================================================================

def unpool_k_fixed(unpool, x0, ei0, ea0=None, k: int = 2,
                   actions_to_replay: Optional[List[Dict]] = None,
                   rng: Optional[torch.Generator] = None):
    x, ei, ea = x0, ei0, ea0
    logP_total = x0.new_zeros(())
    entropy_total = x0.new_zeros(())

    replay_mode = actions_to_replay is not None
    k_actions_recorded = []

    for step_k in range(k):
        actions_for_step = actions_to_replay[step_k] if replay_mode else None

        x, ei, ea, logP, entropy, *_, actions_obj = unpool(
            x, ei, edge_attr=ea, actions_to_replay=actions_for_step, rng=rng
        )

        logP_total = logP_total + logP
        entropy_total = entropy_total + entropy
        if not replay_mode:
            k_actions_recorded.append(actions_obj)

    return x, ei, ea, logP_total, entropy_total, k_actions_recorded


class LosslessGraphEncoder(nn.Module):
    """
    Encodes a graph into two scrambled tensors (integer and float), preserving
    all information perfectly. The scrambling is a fixed, seeded permutation.
    """

    def __init__(self, capN: int, node_dim: int, edge_dim: int, scramble_seed: int = 0):
        super().__init__()
        self.capN = int(capN)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)

        self.int_dim = 1 + self.capN * self.capN
        self.float_dim = self.capN * self.node_dim + self.capN * self.capN * self.edge_dim

        g = torch.Generator(device="cpu").manual_seed(int(scramble_seed))
        int_perm = torch.randperm(self.int_dim, generator=g)
        float_perm = torch.randperm(self.float_dim, generator=g)

        self.register_buffer("int_perm", int_perm)
        self.register_buffer("float_perm", float_perm)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = x.device
        n = x.size(0)
        assert n <= self.capN, f"Graph size n={n} exceeds capacity capN={self.capN}"

        n_tensor = torch.tensor([n], dtype=torch.int64, device=device)

        adj_matrix = torch.zeros(self.capN, self.capN, dtype=torch.bool, device=device)
        if edge_index.numel() > 0:
            adj_matrix[edge_index[0], edge_index[1]] = True

        int_flat = torch.cat([
            n_tensor,
            adj_matrix.view(-1).long()
        ])

        X_pad = torch.zeros(self.capN, self.node_dim, dtype=x.dtype, device=device)
        X_pad[:n, :] = x

        W_pad = torch.zeros(self.capN, self.capN, self.edge_dim, dtype=x.dtype, device=device)
        if edge_attr is not None and edge_index.numel() > 0:
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            W_pad[edge_index[0], edge_index[1]] = edge_attr

        float_flat = torch.cat([
            X_pad.view(-1),
            W_pad.view(-1)
        ])

        int_scrambled = int_flat[self.int_perm.to(device)]
        float_scrambled = float_flat[self.float_perm.to(device)]

        return {
            "int_scrambled": int_scrambled,
            "float_scrambled": float_scrambled,
        }


class LosslessSeedFeaturizer(nn.Module):
    """
    Deterministically converts the scrambled, lossless graph encoding into
    the initial seed features for the generator.
    """

    def __init__(self, capN: int, node_dim: int, edge_dim: int,
                 dx: int, n_seed: int, scramble_seed: int = 0, dim_check: bool = False):
        super().__init__()
        self.capN = int(capN)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.dx = int(dx)
        self.n_seed = int(n_seed)

        self.int_dim = 1 + self.capN * self.capN
        self.float_dim = self.capN * self.node_dim + self.capN * self.capN * self.edge_dim
        self.int_as_float_dim = self.int_dim
        self.total_packed_dim = self.float_dim + self.int_as_float_dim

        seed_capacity = self.n_seed * self.dx
        if not dim_check:
            assert seed_capacity >= self.total_packed_dim, (
                f"Seed capacity {seed_capacity} is less than packed data size "
                f"{self.total_packed_dim}. Increase DX or N_SEED."
            )

        g = torch.Generator(device="cpu").manual_seed(int(scramble_seed))
        int_perm = torch.randperm(self.int_dim, generator=g)
        float_perm = torch.randperm(self.float_dim, generator=g)

        self.register_buffer("int_unperm", torch.argsort(int_perm))
        self.register_buffer("float_unperm", torch.argsort(float_perm))

    def forward(self, encoded_dict: Dict[str, torch.Tensor], noise_std: float = 0.0) -> torch.Tensor:
        device = encoded_dict["float_scrambled"].device

        int_flat = encoded_dict["int_scrambled"][self.int_unperm.to(device)]
        float_flat = encoded_dict["float_scrambled"][self.float_unperm.to(device)]

        int_as_float = int_flat.to(torch.float32)
        z_combined = torch.cat([float_flat, int_as_float], dim=0)

        x_seed = torch.zeros(self.n_seed, self.dx, device=device)
        x_seed_flat = x_seed.view(-1)
        x_seed_flat[:self.total_packed_dim] = z_combined
        x_seed = x_seed_flat.view(self.n_seed, self.dx)

        if noise_std > 0:
            x_seed = x_seed + torch.randn_like(x_seed) * noise_std

        return x_seed


class Critic(nn.Module):
    def __init__(self, node_feature_dim: int, hidden=(512, 256)):
        super().__init__()
        core_in = node_feature_dim * 3
        dims = (core_in,) + tuple(hidden) + (1,)
        self.alpha = nn.Parameter(torch.zeros(core_in))
        self.net = mlp(dims, norm="layer")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = x.reshape(-1)
        feats = (feats * torch.exp(self.alpha)).clone()
        v = self.net(feats)
        return v.squeeze(-1)


# ============================================================================
# Dataset / context construction (built ONCE, shared by every DEHB evaluation
# so all configs are scored on identical data)
# ============================================================================

def make_dataset(n_graphs: int, *, n_min_nodes: int, n_max_nodes: int,
                 node_feat_dim: int, desc_dim: int, include_degree_feats: bool,
                 edge_feat_dim: int, p_extra: float, device):
    ds = []
    for _ in range(n_graphs):
        n = random.randint(n_min_nodes, n_max_nodes)
        x, ei, ea, meta = random_undirected_graph_with_features(
            n,
            p_extra=p_extra,
            node_feat_dim=node_feat_dim,
            desc_dim=desc_dim,
            include_degree_feats=include_degree_feats,
            edge_feat_dim=edge_feat_dim,
            edge_feat_style="gaussian",
            device=device,
        )
        ds.append((x, ei, ea, meta))
    return ds


def build_training_context(
    *,
    n_train: int,
    n_val: int,
    n_min_nodes: int = 7,
    n_max_nodes: int = 11,
    node_feat_dim: int = 5,
    desc_dim: int = 16,
    edge_feat_dim: int = 2,
    include_degree_feats: bool = True,
    p_extra: float = 0.25,
    n_seed: int = 3,
    noise_std: float = 0.0,
    scramble_seed: int = 42,
    data_seed: int = 1234,
    device="cpu",
) -> Dict[str, Any]:
    """Builds datasets + lossless seed featurization exactly as the original
    script did, and returns everything the objective needs."""
    random.seed(data_seed)
    torch.manual_seed(data_seed)

    raw_train = make_dataset(n_train, n_min_nodes=n_min_nodes, n_max_nodes=n_max_nodes,
                             node_feat_dim=node_feat_dim, desc_dim=desc_dim,
                             include_degree_feats=include_degree_feats,
                             edge_feat_dim=edge_feat_dim, p_extra=p_extra, device=device)
    raw_val = make_dataset(n_val, n_min_nodes=n_min_nodes, n_max_nodes=n_max_nodes,
                           node_feat_dim=node_feat_dim, desc_dim=desc_dim,
                           include_degree_feats=include_degree_feats,
                           edge_feat_dim=edge_feat_dim, p_extra=p_extra, device=device)

    capN = n_max_nodes
    raw_node_dim = raw_train[0][0].size(1)

    probe = LosslessSeedFeaturizer(capN=capN, node_dim=raw_node_dim, edge_dim=edge_feat_dim,
                                   dx=1, n_seed=n_seed, dim_check=True)
    packed_dim = probe.total_packed_dim
    dx = math.ceil(packed_dim / n_seed)

    encoder = LosslessGraphEncoder(capN=capN, node_dim=raw_node_dim,
                                   edge_dim=edge_feat_dim, scramble_seed=scramble_seed).to(device).eval()
    featurizer = LosslessSeedFeaturizer(capN=capN, node_dim=raw_node_dim, edge_dim=edge_feat_dim,
                                        dx=dx, n_seed=n_seed, scramble_seed=scramble_seed).to(device).eval()

    with torch.no_grad():
        train_set = [(featurizer(encoder(x, ei, ea), noise_std=noise_std), x, ei, ea)
                     for (x, ei, ea, _) in raw_train]
        val_set = [(featurizer(encoder(x, ei, ea), noise_std=noise_std), x, ei, ea)
                   for (x, ei, ea, _) in raw_val]

    return {
        "train_set": train_set,
        "val_set": val_set,
        "dx": dx,
        "dw": edge_feat_dim,
        "packed_dim": packed_dim,
        "n_seed": n_seed,
        "seed_ei": seed_graph().to(device),
        "device": device,
    }


# ============================================================================
# Evaluation (deterministic RNG so every config is scored identically)
# ============================================================================

@torch.no_grad()
def evaluate_policy(unpool, dataset, similarity, seed_ei, *, k: int, device,
                    eval_seed: int = 9999, n_eval: int = 64, verbose: bool = False) -> float:
    was_training = unpool.training
    unpool.eval()

    # NOTE: a *fresh* generator with a fixed seed, NOT the actor RNG. This makes
    # val scores comparable across DEHB evaluations.
    rng = torch.Generator(device=device)
    rng.manual_seed(eval_seed)

    total = 0.0
    count = 0
    gen_n = tgt_n = gen_e = tgt_e = 0

    for (x_seed, x_t, ei_t, ea_t) in dataset[:n_eval]:
        x_gen, ei_gen, ea_gen, *_ = unpool_k_fixed(unpool, x_seed, seed_ei, None, k=k, rng=rng)

        gen_n += x_gen.size(0); tgt_n += x_t.size(0)
        gen_e += ei_gen.size(1); tgt_e += ei_t.size(1)

        score, _ = similarity.graph_similarity(
            ei_gen.cpu(), ei_t.cpu(),
            x1=x_gen.cpu(), x2=x_t.cpu(),
            edge_attr1=ea_gen.cpu(),
            edge_attr2=(ea_t.cpu() if ea_t is not None else None),
            directed=False, wl_iters=2
        )
        total += float(score); count += 1

    if verbose:
        print(f"Avg Gen N Size: {gen_n / max(1, count):.2f}\tTarget: {tgt_n / max(1, count):.2f}")
        print(f"Avg Gen E Size: {gen_e / max(1, count):.2f}\tTarget: {tgt_e / max(1, count):.2f}")

    if was_training:
        unpool.train()
    return total / max(1, count)


# ============================================================================
# The DEHB objective: config + fidelity (epochs) -> validation similarity
# ============================================================================

def train_and_eval(
    config: Dict[str, Any],
    fidelity: float,
    ctx: Dict[str, Any],
    *,
    k_unpool: int = 2,
    seed: int = 1234,
    value_loss_coef: float = 0.5,
    warmup_frac: float = 0.1,
    eval_n: int = 64,
    eval_seed: int = 9999,
    log_every: int = 0,
    similarity=None,
    similarity_ref=None,   # optional second metric (e.g. graph_similarity2) printed as a reference
    target_kl: Optional[float] = 1.0,    # TRAJECTORY-level KL budget: logP sums over all unpooling
                                         # actions, so this is ~n_actions x per-action KL. Calibrate
                                         # from the kl= diagnostic in the logs (see below).
    log_ratio_bound: float = 10.0,       # hard bound on log(ratio) before exp() — prevents inf/NaN
    adv_std_floor: float = 0.05,         # floor on advantage std: as the policy converges, rollout
                                         # scores homogenize and std -> 0; dividing by it amplifies
                                         # metric noise into huge pseudo-advantages (the late-run
                                         # policy collapse). Floor keeps the scale sane.
    adv_clip: float = 5.0,               # hard cap on normalized advantages
    early_stop_patience: Optional[int] = None,  # stop after this many periodic evals w/o a new best
    return_models: bool = False,
) -> Dict[str, Any]:
    """
    Trains the unpooler from scratch for int(fidelity) epochs with the given
    hyperparameters and returns validation similarity. This is the function
    DEHB drives.

    config keys (must match build_configspace):
      lr, entropy_coef, unpool_size, batch_size, ppo_update_epochs, ppo_clip_eps,
      inter_link_scoring ("preference" or "plain"; defaults to "preference" when absent)
    """
    if similarity is None:
        import graph_similarity as similarity  # local module, import lazily

    device = ctx["device"]
    DX, DW = ctx["dx"], ctx["dw"]
    train_set = ctx["train_set"]
    seed_ei = ctx["seed_ei"]

    epochs = max(1, int(round(float(fidelity))))
    lr = float(config["lr"])
    entropy_coef = float(config["entropy_coef"])
    unpool_size = int(config["unpool_size"])
    batch_size = int(config["batch_size"])
    ppo_update_epochs = int(config["ppo_update_epochs"])
    ppo_clip_eps = float(config["ppo_clip_eps"])

    # Fixed seeding => every config starts from a comparable init / action stream
    torch.manual_seed(seed)
    actor_rng = torch.Generator(device=device)
    actor_rng.manual_seed(seed + 7)

    use_preference = str(config.get("inter_link_scoring", "preference")) == "preference"
    unpool = GuoUnpool(dx=DX, dw=DW, dy=DX, du=DW,
                       kv=unpool_size, kia=unpool_size,
                       kie=unpool_size, kw=unpool_size,
                       use_preference=use_preference).to(device)
    critic = Critic(node_feature_dim=DX, hidden=(512, 256)).to(device)
    opt = torch.optim.AdamW(list(unpool.parameters()) + list(critic.parameters()),
                            lr=lr, weight_decay=0.0)

    warmup = max(1, int(round(epochs * warmup_frac)))
    if epochs > warmup:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup, eta_min=lr * 0.1),
            ],
            milestones=[warmup],
        )
    else:
        sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=epochs)

    import copy as _copy
    experience_buffer: List[Dict[str, Any]] = []
    losses_hist, rewards_hist = [], []
    adv_std_hist, entropy_hist = [], []
    best_val_R = -math.inf
    best_state: Optional[Dict[str, Any]] = None
    evals_since_best = 0

    for epoch in range(1, epochs + 1):
        # ----- Data collection -----
        unpool.eval(); critic.eval()
        experience_buffer.clear()
        with torch.no_grad():
            for (x_seed, x_t, ei_t, ea_t) in train_set:
                x_gen, ei_gen, ea_gen, logP_old, entropy, actions_taken = unpool_k_fixed(
                    unpool, x_seed, seed_ei, k=k_unpool, rng=actor_rng
                )
                predicted_value = critic(x_seed)
                score, _ = similarity.graph_similarity(
                    ei_gen.cpu(), ei_t.cpu(),
                    x1=x_gen.cpu(), x2=x_t.cpu(),
                    edge_attr1=ea_gen.cpu(),
                    edge_attr2=(ea_t.cpu() if ea_t is not None else None),
                    directed=False, wl_iters=2
                )
                experience_buffer.append({
                    "x_seed": x_seed,
                    "seed_ei": seed_ei,
                    "actions": actions_taken,
                    "score": float(score),
                    "logP_old": logP_old.detach(),
                    "entropy": entropy.detach(),
                    "predicted_value": predicted_value.detach(),
                })

        scores_all = torch.tensor([e["score"] for e in experience_buffer], device=device, dtype=torch.float32)
        values_all = torch.stack([e["predicted_value"] for e in experience_buffer]).detach().squeeze(-1)
        logP_old_all = torch.stack([e["logP_old"] for e in experience_buffer]).detach()

        advantages_all = scores_all - values_all
        adv_std_raw = float(advantages_all.std())          # diagnostic: the collapse precursor
        mean_entropy = float(torch.stack([e["entropy"] for e in experience_buffer]).mean())
        advantages_all = (advantages_all - advantages_all.mean()) / advantages_all.std().clamp_min(adv_std_floor)
        advantages_all = advantages_all.clamp(-adv_clip, adv_clip)

        # ----- PPO Update -----
        unpool.train(); critic.train()
        epoch_total_losses = []
        epoch_skipped = 0
        kl_stopped = False
        kl_at_stop = None
        n_minibatches = max(1, math.ceil(len(experience_buffer) / batch_size))
        total_planned_steps = ppo_update_epochs * n_minibatches
        for _ in range(ppo_update_epochs):
            if kl_stopped:
                break
            perm = torch.randperm(len(experience_buffer), device=device)
            for i in range(0, len(experience_buffer), batch_size):
                mb_idx = perm[i:i + batch_size]
                scores_tensor = scores_all[mb_idx]
                logPs_old_tensor = logP_old_all[mb_idx]
                advantages_mb = advantages_all[mb_idx]

                logPs_new_list, entropies_new_list, current_values_list = [], [], []
                for j in mb_idx.tolist():
                    exp = experience_buffer[j]
                    actions_copy = _copy.deepcopy(exp["actions"])
                    _, _, _, logP_new, entropy_new, _ = unpool_k_fixed(
                        unpool, exp["x_seed"], exp["seed_ei"], k=k_unpool,
                        actions_to_replay=actions_copy, rng=actor_rng
                    )
                    logPs_new_list.append(logP_new)
                    entropies_new_list.append(entropy_new)
                    current_values_list.append(critic(exp["x_seed"]))

                logPs_new_tensor = torch.stack(logPs_new_list)
                entropies_new_tensor = torch.stack(entropies_new_list)
                current_values_tensor = torch.stack(current_values_list).squeeze(-1)

                opt.zero_grad(set_to_none=True)
                # Bound the log-ratio BEFORE exponentiating. Unbounded, exp() can
                # overflow to inf (the epoch-50 loss=4079 / epoch-80 NaN failure):
                # for negative advantages min(surr1, surr2) is NOT bounded by the
                # PPO clip, so an exploding ratio explodes the loss.
                log_ratio = (logPs_new_tensor - logPs_old_tensor).clamp(-log_ratio_bound, log_ratio_bound)
                ratio = torch.exp(log_ratio)

                # Early stop on KL drift (k3 estimator). Repeated PPO epochs over
                # the same buffer push the policy off-policy; once KL exceeds the
                # target, further replay updates are noise at best, poison at worst.
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                if target_kl is not None and float(approx_kl) > target_kl:
                    kl_stopped = True
                    kl_at_stop = float(approx_kl)
                    break

                surr1 = ratio * advantages_mb
                surr2 = torch.clamp(ratio, 1 - ppo_clip_eps, 1 + ppo_clip_eps) * advantages_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(current_values_tensor, scores_tensor)
                entropy_bonus = entropies_new_tensor.mean()

                loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_bonus

                # Never let a non-finite loss or gradient reach opt.step():
                # clip_grad_norm_ does NOT sanitize NaN/inf — it multiplies by a
                # NaN norm and opt.step() then writes NaN into every weight.
                if not torch.isfinite(loss):
                    epoch_skipped += 1
                    continue
                loss.backward()
                gn_u = torch.nn.utils.clip_grad_norm_(unpool.parameters(), 1.0)
                gn_c = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                if not (torch.isfinite(gn_u) and torch.isfinite(gn_c)):
                    opt.zero_grad(set_to_none=True)
                    epoch_skipped += 1
                    continue
                opt.step()
                epoch_total_losses.append(loss.item())

        if not epoch_total_losses:
            epoch_total_losses = [float("nan")]
        avg_total_loss = float(torch.tensor(epoch_total_losses).nanmean())
        avg_reward = float(scores_all.mean())
        losses_hist.append(avg_total_loss)
        rewards_hist.append(avg_reward)
        adv_std_hist.append(adv_std_raw)
        entropy_hist.append(mean_entropy)

        # Weight health check: if anything non-finite slipped into the params,
        # restore the last good snapshot instead of burning the remaining epochs
        # on a dead network.
        params_ok = all(torch.isfinite(p).all() for p in unpool.parameters())
        if not params_ok:
            if best_state is not None:
                unpool.load_state_dict(best_state["unpool"])
                critic.load_state_dict(best_state["critic"])
                print(f"[{epoch:04d}/{epochs}] !! non-finite weights detected — "
                      f"restored best snapshot (val_R={best_val_R:.3f} @ epoch {best_state['epoch']})")
            else:
                print(f"[{epoch:04d}/{epochs}] !! non-finite weights detected with no snapshot to restore — stopping")
                break

        if log_every and (epoch % log_every == 0 or epoch == 1):
            val_R_now = evaluate_policy(unpool, ctx["val_set"], similarity, seed_ei,
                                        k=k_unpool, device=device,
                                        eval_seed=eval_seed, n_eval=eval_n)
            if math.isfinite(val_R_now) and val_R_now > best_val_R:
                best_val_R = float(val_R_now)
                evals_since_best = 0
                best_state = {
                    "epoch": epoch,
                    "unpool": _copy.deepcopy(unpool.state_dict()),
                    "critic": _copy.deepcopy(critic.state_dict()),
                }
            else:
                evals_since_best += 1
            line = (f"[{epoch:04d}/{epochs}] loss={avg_total_loss:<7.4f} | "
                    f"train_R={avg_reward:.3f} | val_R={val_R_now:.3f} | "
                    f"adv_std={adv_std_raw:.3f} | H={mean_entropy:.1f}")
            if similarity_ref is not None:
                val_R_ref = evaluate_policy(unpool, ctx["val_set"], similarity_ref, seed_ei,
                                            k=k_unpool, device=device,
                                            eval_seed=eval_seed, n_eval=eval_n)
                line += f" | val_R_ref={val_R_ref:.3f}"
            if kl_stopped:
                line += f" | kl_stop@{len(epoch_total_losses)}/{total_planned_steps} (kl={kl_at_stop:.3f})"
            else:
                line += f" | steps={len(epoch_total_losses)}/{total_planned_steps}"
            if epoch_skipped:
                line += f" | skipped={epoch_skipped}"
            print(line)

            if early_stop_patience is not None and evals_since_best >= early_stop_patience:
                print(f"[{epoch:04d}/{epochs}] no val improvement in {evals_since_best} evals "
                      f"(best={best_val_R:.3f} @ epoch {best_state['epoch'] if best_state else '?'}) — stopping early.")
                break

        sched.step()

    # If the final weights underperform the best periodic snapshot (e.g. the
    # run degraded late), restore the snapshot before final evaluation.
    if best_state is not None:
        final_probe = evaluate_policy(unpool, ctx["val_set"], similarity, seed_ei,
                                      k=k_unpool, device=device,
                                      eval_seed=eval_seed, n_eval=eval_n)
        if not math.isfinite(final_probe) or final_probe < best_val_R:
            unpool.load_state_dict(best_state["unpool"])
            critic.load_state_dict(best_state["critic"])
            if log_every:
                print(f"Final weights (val_R={final_probe:.3f}) underperform best snapshot "
                      f"(val_R={best_val_R:.3f} @ epoch {best_state['epoch']}) — using snapshot.")

    val_R = evaluate_policy(unpool, ctx["val_set"], similarity, seed_ei,
                            k=k_unpool, device=device,
                            eval_seed=eval_seed, n_eval=eval_n, verbose=bool(log_every))
    val_R_ref = None
    if similarity_ref is not None:
        val_R_ref = float(evaluate_policy(unpool, ctx["val_set"], similarity_ref, seed_ei,
                                          k=k_unpool, device=device,
                                          eval_seed=eval_seed, n_eval=eval_n))

    out: Dict[str, Any] = {
        "val_R": float(val_R),
        "val_R_ref": val_R_ref,
        "train_R_last": float(rewards_hist[-1]),
        "best_val_R": (None if best_state is None else float(best_val_R)),
        "best_val_epoch": (None if best_state is None else int(best_state["epoch"])),
        "epochs": epochs,
        "losses_hist": losses_hist,
        "rewards_hist": rewards_hist,
        "adv_std_hist": adv_std_hist,
        "entropy_hist": entropy_hist,
    }
    if return_models:
        out["unpool"] = unpool
        out["critic"] = critic
        out["opt"] = opt
        out["sched"] = sched
    return out


def build_configspace(seed: int = 0):
    import ConfigSpace as CS
    cs = CS.ConfigurationSpace(seed=seed)
    cs.add_hyperparameter(CS.UniformFloatHyperparameter("lr", lower=1e-5, upper=3e-3, log=True))
    cs.add_hyperparameter(CS.UniformFloatHyperparameter("entropy_coef", lower=1e-4, upper=5e-2, log=True))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("unpool_size", choices=[128, 256, 384, 512]))
    cs.add_hyperparameter(CS.CategoricalHyperparameter("batch_size", choices=[64, 128, 256, 384]))
    cs.add_hyperparameter(CS.UniformIntegerHyperparameter("ppo_update_epochs", lower=2, upper=8))
    cs.add_hyperparameter(CS.UniformFloatHyperparameter("ppo_clip_eps", lower=0.05, upper=0.3, log=False))
    # Step 2c scoring: the paper's preference score, or a plain per-edge softmax
    cs.add_hyperparameter(CS.CategoricalHyperparameter("inter_link_scoring", choices=["preference", "plain"]))
    return cs


# ============================================================================
# Entry point: UNPOOL_MODE=tune (default) runs DEHB; UNPOOL_MODE=train does a
# full-length training run (optionally with the tuned config) and saves it.
# ============================================================================

if __name__ == "__main__":
    import json
    from datetime import datetime
    from dehb_helper import DEHBHelper, DEHBRunConfig, ObjectiveResult
    import graph_similarity as GS
    import graph_similarity2 as GS2

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 1234
    K_MAX = 2

    MODE = os.environ.get("UNPOOL_MODE", "tune")  # "tune" | "train"
    DEHB_OUT_DIR = "dehb_unpool_out_undirected"

    # ---- HPO uses a SUBSET of the full data so each eval is tractable.
    N_TRAIN_HPO = 512
    N_VAL_HPO = 128

    N_TRAIN_FULL = 3840
    N_VAL_FULL = 1000
    EPOCHS_FULL = 300

    if MODE == "tune":
        # Auto-resume: if a previous tune run left state in the output dir,
        # pick up where it stopped instead of starting the search over.
        # (DEHB checkpoints after every eval; evals.jsonl existing means at
        # least one eval completed and DEHB state was saved alongside it.)
        # Delete the output directory to force a fresh search. It is not the old
        # dehb_unpool_out/: those results came from the directed layer, before the
        # step 2c / 2d fixes, and their search space has no inter_link_scoring.
        DEHB_OUT = DEHB_OUT_DIR
        RESUME = os.path.exists(os.path.join(DEHB_OUT, "evals.jsonl"))
        if RESUME:
            print(f"Found previous run state in {DEHB_OUT}/ — resuming. "
                  f"(Delete that directory to start fresh.)")

        print("Building HPO datasets (shared across all DEHB evaluations)…")
        ctx = build_training_context(
            n_train=N_TRAIN_HPO, n_val=N_VAL_HPO,
            data_seed=SEED, device=DEVICE,
        )
        print(f"dx={ctx['dx']} packed_dim={ctx['packed_dim']} "
              f"train={len(ctx['train_set'])} val={len(ctx['val_set'])}")

        def objective(cfg: Dict[str, Any], fidelity: float) -> ObjectiveResult:
            out = train_and_eval(cfg, fidelity, ctx, k_unpool=K_MAX, seed=SEED, similarity=GS)
            return ObjectiveResult(
                metric=out["val_R"],
                info={"train_R_last": out["train_R_last"], "epochs": out["epochs"]},
            )

        helper = DEHBHelper(
            configspace=build_configspace(seed=0),
            objective=objective,
            direction="maximize",          # similarity: higher is better
            run_cfg=DEHBRunConfig(
                min_fidelity=8,            # epochs at the lowest rung
                max_fidelity=72,           # epochs at full fidelity (eta=3 → rungs ~8/24/72)
                eta=3,
                fevals=40,                 # total number of (config, fidelity) evaluations
                seed=0,
                n_workers=1,
                output_path=DEHB_OUT,
                resume=RESUME,
            ),
        )

        summary = helper.run()
        print(json.dumps(summary, indent=2, default=str))
        print(f"\nBest config saved to {summary['best_config_path']}")
        print("Re-train at full scale with:  UNPOOL_MODE=train python guo_et_al_unpooling.py")

    elif MODE == "train":
        import matplotlib.pyplot as plt

        # Use tuned config if available, otherwise the original defaults
        best_path = os.path.join(DEHB_OUT_DIR, "best_config.json")
        if os.path.exists(best_path):
            with open(best_path) as f:
                config = json.load(f)["best_config"]
            print(f"Loaded tuned config: {config}")
        else:
            config = {"lr": 3e-4, "entropy_coef": 0.01, "unpool_size": 256,
                      "batch_size": 256, "ppo_update_epochs": 4, "ppo_clip_eps": 0.2,
                      "inter_link_scoring": "preference"}
            print(f"No tuned config found; using defaults: {config}")

        print("Building full datasets…")
        ctx = build_training_context(
            n_train=N_TRAIN_FULL, n_val=N_VAL_FULL,
            data_seed=SEED, device=DEVICE,
        )

        out = train_and_eval(config, EPOCHS_FULL, ctx, k_unpool=K_MAX, seed=SEED,
                             similarity=GS, similarity_ref=GS2, log_every=10, return_models=True,
                             early_stop_patience=10)
        print(f"Final val_R = {out['val_R']:.4f} | val_R_ref (GS2) = {out['val_R_ref']:.4f}")

        os.makedirs("artifacts", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = os.path.join("artifacts", f"unpool_last_{stamp}.pt")
        torch.save({
            "unpool_state_dict": out["unpool"].state_dict(),
            "critic_state_dict": out["critic"].state_dict(),
            "optimizer_state_dict": out["opt"].state_dict(),
            "scheduler_state_dict": out["sched"].state_dict(),
        }, ckpt_path)
        with open(os.path.join("artifacts", f"unpool_last_{stamp}.json"), "w") as f:
            json.dump({
                "dx": ctx["dx"], "dw": ctx["dw"], "dy": ctx["dx"], "du": ctx["dw"],
                "k_max": K_MAX, "packed_dim": ctx["packed_dim"], "n_seed": ctx["n_seed"],
                "seed_ei": ctx["seed_ei"].detach().cpu().tolist(),
                "config": config,
            }, f, indent=2)
        print(f"Saved checkpoint to {ckpt_path}")

        fig1 = plt.figure()
        plt.plot(out["losses_hist"]); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Unpooling PPO Loss")
        fig1.savefig(os.path.join("artifacts", "unpool_loss.png"), dpi=150); plt.close(fig1)

        fig2 = plt.figure()
        plt.plot(out["rewards_hist"]); plt.xlabel("Epoch"); plt.ylabel("Reward (similarity)"); plt.title("Unpooling Reward")
        fig2.savefig(os.path.join("artifacts", "unpool_reward.png"), dpi=150); plt.close(fig2)

        fig3, ax1 = plt.subplots()
        ax1.plot(out["adv_std_hist"], color="tab:blue", label="advantage std (raw)")
        ax1.axhline(0.05, color="tab:blue", linestyle=":", alpha=0.6, label="std floor")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Advantage std", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(out["entropy_hist"], color="tab:orange", label="mean policy entropy")
        ax2.set_ylabel("Entropy", color="tab:orange")
        ax1.set_title("Collapse diagnostics: advantage std & policy entropy")
        fig3.savefig(os.path.join("artifacts", "unpool_diagnostics.png"), dpi=150); plt.close(fig3)

    else:
        raise ValueError(f"Unknown UNPOOL_MODE={MODE!r} (use 'tune' or 'train')")