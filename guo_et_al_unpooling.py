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


class GuoUnpool(nn.Module):
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
        self.mlp_c   = self.mlp_ie2  # alias

        if self.use_preference:
            self.mlp_zero_s = mlp([dy, 2 * dy, 1], norm="layer")
            self.mlp_zero_b = mlp([dx, 2 * dx, 1], norm="layer")

        self.mlp_r    = mlp([dx, max(1, dx // 2), 1], last_activation=nn.Sigmoid(), norm="layer")
        self.mlp_ie_a = mlp([dx + dx + dw, kie, 1], last_activation=nn.Sigmoid(), norm="layer")
        self.mlp_u    = mlp([dy, kw, du], norm="layer")

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
    def _lexsort_edges(edge_index: torch.Tensor, edge_attr: torch.Tensor | None, num_nodes: int):
        if edge_index.numel() == 0:
            return edge_index, edge_attr
        src = edge_index[0].to(torch.long)
        dst = edge_index[1].to(torch.long)
        keys = src * num_nodes + dst
        perm = torch.argsort(keys)
        ei_sorted = edge_index[:, perm]
        ea_sorted = edge_attr[perm] if edge_attr is not None else None
        return ei_sorted, ea_sorted

    @staticmethod
    def _canon_pair(i, j):
        a = int(i) if torch.is_tensor(i) else i
        b = int(j) if torch.is_tensor(j) else j
        return (a, b) if a < b else (b, a)

    # ---- helper: interlink probabilities (batched) ----
    def _p12_both_batched(self, y1s, y2s, w_ij, x_other):
        # inputs: [K, dy], [K, dy], [K, dw], [K, dx]
        if self.use_preference:
            hs1 = self.mlp_ie1(torch.cat([y1s, w_ij, x_other], dim=1))  # [K,1]
            hs2 = self.mlp_ie1(torch.cat([y2s, w_ij, x_other], dim=1))
            hb  = self.mlp_ie2(torch.cat([self.agg(y1s, y2s), w_ij, x_other], dim=1))
            h0s = self.mlp_zero_s(y1s)                                  # [K,1]
            h0b = self.mlp_zero_b(x_other)                              # [K,1]
            logits = torch.cat([hs1, hs2, hb, h0s + h0b], dim=1)        # [K,4]
            Z = self._smooth_cat(F.softmax(logits, dim=1), dim=1)
            p1, p2, pB = Z[:, 0], Z[:, 1], Z[:, 2]
        else:
            s1 = self.mlp_ie1(torch.cat([y1s, w_ij, x_other], dim=1))
            s2 = self.mlp_ie1(torch.cat([y2s, w_ij, x_other], dim=1))
            sb = self.mlp_ie2(torch.cat([self.agg(y1s, y2s), w_ij, x_other], dim=1))
            logits = torch.cat([s1, s2, sb], dim=1)                     # [K,3]
            Z = self._smooth_cat(F.softmax(logits, dim=1), dim=1)
            p1, p2, pB = Z[:, 0], Z[:, 1], Z[:, 2]
        return p1, p2, pB, Z  # Z is [K,3 or 4]

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
        Returns:
          x_out, edge_index_out, edge_attr_out, logP, total_entropy, parent_map, sets, actions_recorded
        """
        device = x.device
        N = x.size(0)

        if edge_attr is None:
            edge_attr = x.new_zeros(edge_index.size(1), self.dw)

        # coalesce + stable lex ordering (critical for replay determinism)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=N)
        edge_index, edge_attr = self._lexsort_edges(edge_index, edge_attr, N)

        replay_mode = actions_to_replay is not None
        actions_recorded: Dict = {
            'step1a_unpool': [],
            'step2a_intra': [],
            'step2b_pick': {},
            'step2c_side': {},
            'step2c_sets': {},
            'step2d_pa': {},
            'step2d_rij': {},
            'step2d_edges': {},
        }

        # ========== Step 1a: partitions / unpool decision ==========
        all_idx = torch.arange(N, device=device)
        I_s = torch.tensor([], dtype=torch.long, device=device) if I_s is None else I_s.to(device)
        I_u = torch.tensor([], dtype=torch.long, device=device) if I_u is None else I_u.to(device)
        if I_r is None:
            mask = torch.ones(N, dtype=torch.bool, device=device)
            mask[I_s] = False
            mask[I_u] = False
            I_r = all_idx[mask]
        else:
            I_r = I_r.to(device)

        logP = x.new_zeros(())
        total_entropy = x.new_zeros(())

        pr = self._smooth_bern(self.mlp_r(x[I_r]).squeeze(-1))  # [|I_r|]
        if replay_mode:
            choose_unpool = actions_to_replay['step1a_unpool'][0]
        else:
            u = torch.rand(pr.shape, dtype=pr.dtype, device=pr.device, generator=rng)
            choose_unpool = (u < pr)
            actions_recorded['step1a_unpool'].append(choose_unpool)

        Iu = torch.cat([I_u, I_r[choose_unpool]], dim=0)
        Is = torch.cat([I_s, I_r[~choose_unpool]], dim=0)

        logP = logP + torch.sum(torch.log(pr.clamp_min(1e-9))[choose_unpool]) \
                    + torch.sum(torch.log((1 - pr).clamp_min(1e-9))[~choose_unpool])

        pr_stable = pr.clamp(1e-9, 1.0 - 1e-9)
        total_entropy = total_entropy + (-(pr_stable * pr_stable.log() + (1 - pr_stable) * (1 - pr_stable).log())).sum()

        # ========== Step 1b: node features / packing ==========
        PS1, PS2 = self._project(x)

        if not replay_mode:
            actions_recorded["__Is_order__"] = [int(i) for i in Is.tolist()]
            actions_recorded["__Iu_order__"] = [int(i) for i in Iu.tolist()]
        else:
            Is = torch.tensor(actions_to_replay["__Is_order__"], dtype=torch.long, device=device)
            Iu = torch.tensor(actions_to_replay["__Iu_order__"], dtype=torch.long, device=device)

        x_static = self.mlp_y(PS1[Is])
        x_c1     = self.mlp_y(PS1[Iu])
        x_c2     = self.mlp_y(PS2[Iu])
        x_out    = torch.cat([x_static, x_c1, x_c2], dim=0)

        f_map  = {int(i): idx for idx, i in enumerate(Is.tolist())}
        base_c1 = len(Is)
        base_c2 = len(Is) + len(Iu)
        f1_map = {int(i): base_c1 + k for k, i in enumerate(Iu.tolist())}
        f2_map = {int(i): base_c2 + k for k, i in enumerate(Iu.tolist())}

        # ========== Step 2a: intra-links ==========
        if len(Iu) > 0:
            y1 = x_out[torch.arange(len(Iu), device=device) + base_c1]
            y2 = x_out[torch.arange(len(Iu), device=device) + base_c2]
            p_intra = self._smooth_bern(self.mlp_ia(self.agg(y1, y2)).squeeze(-1))  # [|Iu|]

            if replay_mode:
                Vc_mask = actions_to_replay['step2a_intra'][0]
            else:
                u = torch.rand(p_intra.shape, dtype=p_intra.dtype, device=p_intra.device, generator=rng)
                Vc_mask = (u < p_intra)
                actions_recorded['step2a_intra'].append(Vc_mask)

            Vc = Iu[Vc_mask]
            logP = logP + torch.sum(torch.log(p_intra.clamp_min(1e-9))[Vc_mask]) \
                        + torch.sum(torch.log((1 - p_intra).clamp_min(1e-9))[~Vc_mask])

            p_intra_stable = p_intra.clamp(1e-9, 1.0 - 1.0e-9)
            total_entropy  = total_entropy + (-(p_intra_stable * p_intra_stable.log()
                                               + (1 - p_intra_stable) * (1 - p_intra_stable).log())).sum()
        else:
            Vc_mask = torch.tensor([], dtype=torch.bool, device=device)
            Vc = Iu

        # ========== Build adjacency (lexsorted) and neighbor lists ==========
        src = edge_index[0]
        dst = edge_index[1]
        M   = edge_index.size(1)

        # Fast vectorized representation of neighbors: for each parent u in Iu_no_intra,
        # we gather its neighbor indices and ek row indices. We’ll build flat lists
        # then compute scores in one MLP call, and finally sample 1 choice per parent.
        iu_no_intra = Iu[~Vc_mask]
        new_edges = []

        # ------- Step 2b: vectorized neighbor scoring -------
        bj_choice: Dict[int, int] = {}
        if iu_no_intra.numel() > 0:
            parent_rows = []
            parent_ptrs = [0]  # prefix to segment rows by parent
            all_neis = []
            all_eks  = []
            for p in iu_no_intra.tolist():
                # neighbors: any edge touching p → neighbor is the other endpoint
                mask_src = (src == p)
                mask_dst = (dst == p)
                idx_src  = torch.nonzero(mask_src, as_tuple=False).flatten()
                idx_dst  = torch.nonzero(mask_dst, as_tuple=False).flatten()
                neis_src = dst[idx_src]  # edges p->nei
                neis_dst = src[idx_dst]  # edges nei->p
                neis     = torch.cat([neis_src, neis_dst], dim=0)
                eks      = torch.cat([idx_src, idx_dst], dim=0)

                # keep lexicographic order already provided by _lexsort_edges
                parent_rows.append((p, neis, eks))
                all_neis.append(neis)
                all_eks.append(eks)
                parent_ptrs.append(parent_ptrs[-1] + neis.numel())

            if parent_ptrs[-1] > 0:
                all_neis_cat = torch.cat(all_neis, dim=0)  # [K_total]
                all_eks_cat  = torch.cat(all_eks,  dim=0)  # [K_total]

                # Build Φ_2b rows: [agg(y1p,y2p), w, x_nei]
                # For each row we need y1p/y2p of its parent → we’ll expand per segment
                # Prepare parent embedding tensors aligned with concatenated rows
                y1p_list, y2p_list = [], []
                for (p, neis, eks) in parent_rows:
                    if neis.numel() == 0:
                        continue
                    y1p_list.append(x_out[f1_map[p]].unsqueeze(0).repeat(neis.numel(), 1))
                    y2p_list.append(x_out[f2_map[p]].unsqueeze(0).repeat(neis.numel(), 1))
                y1p_cat = torch.cat(y1p_list, dim=0) if y1p_list else x_out.new_zeros((0, self.dy))
                y2p_cat = torch.cat(y2p_list, dim=0) if y2p_list else x_out.new_zeros((0, self.dy))

                w_cat   = edge_attr[all_eks_cat]                # [K_total, dw]
                xnei_cat= x[all_neis_cat]                       # [K_total, dx]
                agg_y   = self.agg(y1p_cat, y2p_cat)            # [K_total, dy]
                scores  = self.mlp_c(torch.cat([agg_y, w_cat, xnei_cat], dim=1)).squeeze(-1)  # [K_total]

                # For each parent segment, softmax over its rows, then sample 1 index
                # We’ll do sampling per segment in a light python loop (K segments), probs computed batched.
                start = 0
                for seg_idx, (p, neis, eks) in enumerate(parent_rows):
                    L = neis.numel()
                    if L == 0:
                        continue
                    seg_scores = scores[start:start+L]
                    probs = self._smooth_cat(F.softmax(seg_scores, dim=0), dim=0)

                    if replay_mode:
                        rec = actions_to_replay['step2b_pick'].get(int(p), None)
                        pick = None
                        if rec is not None and isinstance(rec, dict) and "ek" in rec:
                            ek_id = int(rec["ek"])
                            # find matching ek
                            match = (eks == ek_id).nonzero(as_tuple=False)
                            if match.numel() > 0:
                                pick = int(match[0].item())
                        if pick is None:
                            raise RuntimeError(
                                f"Step 2b replay failed for parent {p}. "
                                f"Could not find recorded action: {rec}. "
                                f"Available neighbors (nei, ek): {[(int(n), int(e)) for n, e in zip(neis.tolist(), eks.tolist())]}"
                            )
                    else:
                        pick = torch.multinomial(probs, 1, generator=rng).item()
                        actions_recorded['step2b_pick'][int(p)] = {"nei": int(neis[pick].item()),
                                                                   "ek":  int(eks[pick].item())}

                    bj_choice[p] = int(neis[pick].item())
                    # logP & entropy
                    logP = logP + torch.log(probs[pick].clamp_min(1e-9))
                    ent  = -(probs.clamp_min(1e-9) * torch.log(probs.clamp_min(1e-9))).sum()
                    total_entropy = total_entropy + ent
                    start += L

        if not replay_mode:
            actions_recorded["__bj_choice__"] = {int(k): int(v) for k, v in bj_choice.items()}
        else:
            rec_bj = actions_to_replay.get("__bj_choice__", {})
            assert {int(k): int(v) for k, v in bj_choice.items()} == rec_bj, \
                f"Replay diverged in 2b bj_choice.\nrecord={rec_bj}\nreplay={bj_choice}"

        # ---- batched 2c probabilities for all directed edges ----
        # For each original directed edge (i->j) we might need probabilities for i’s split to j (i->j)
        # and for j’s split to i (j->i). We compute both directions in two batched calls.
        # Prepare masks for which endpoints are static (mapped) vs unpooled.
        # ---- batched 2c probabilities for all directed edges ----
        # We must EXCLUDE edges that were satisfied by the 2b "both-children" pick.
        is_static = torch.zeros(N, dtype=torch.bool, device=device)
        is_static[Is] = True

        src = edge_index[0]
        dst = edge_index[1]
        M   = edge_index.size(1)

        # masks: which endpoints are unpooled (need 2c), per edge
        i_unp_mask = ~is_static[src]  # i->j direction needs 2c if True
        j_unp_mask = ~is_static[dst]  # j->i direction needs 2c if True

        # Build masks for edges that are the special 2b neighbor (we SKIP 2c there)
        bj_edge_mask_ij = torch.zeros(M, dtype=torch.bool, device=device)  # for i->j direction
        bj_edge_mask_ji = torch.zeros(M, dtype=torch.bool, device=device)  # for j->i direction
        if bj_choice:
            # mark edges where (src==i and dst==bj_choice[i]) and where (dst==j and src==bj_choice[j])
            # do in a tiny loop over parents (cheap); avoids big O(N^2) compare
            for pi, pj in bj_choice.items():
                # i->j mask
                hits_ij = (src == pi) & (dst == pj)
                if hits_ij.any():
                    bj_edge_mask_ij |= hits_ij
                # j->i mask
                hits_ji = (dst == pi) & (src == pj)
                if hits_ji.any():
                    bj_edge_mask_ji |= hits_ji

        # 2c really needs probs only for:
        prob_mask_ij = i_unp_mask & ~bj_edge_mask_ij
        prob_mask_ji = j_unp_mask & ~bj_edge_mask_ji

        idx_ij = torch.nonzero(prob_mask_ij, as_tuple=False).flatten()  # rows needing i->j probs
        idx_ji = torch.nonzero(prob_mask_ji, as_tuple=False).flatten()  # rows needing j->i probs

        # Build position lookups so we don't rely on pointer equality with k
        pos_ij = torch.full((M,), -1, dtype=torch.long, device=device)
        pos_ji = torch.full((M,), -1, dtype=torch.long, device=device)
        if idx_ij.numel() > 0:
            pos_ij[idx_ij] = torch.arange(idx_ij.numel(), device=device)
        if idx_ji.numel() > 0:
            pos_ji[idx_ji] = torch.arange(idx_ji.numel(), device=device)

        # === i->j batched evaluation ===
        p1_i = p2_i = pB_i = None
        Z_i  = None
        if idx_ij.numel() > 0:
            i_nodes = src[idx_ij]
            j_nodes = dst[idx_ij]
            y1s = x_out[torch.tensor([f1_map[int(v)] for v in i_nodes.tolist()], device=device)]
            y2s = x_out[torch.tensor([f2_map[int(v)] for v in i_nodes.tolist()], device=device)]
            w_ij = edge_attr[idx_ij]
            x_j  = x[j_nodes]
            p1_i, p2_i, pB_i, Z_i = self._p12_both_batched(y1s, y2s, w_ij, x_j)  # [Kij]

        # === j->i batched evaluation ===
        p1_j = p2_j = pB_j = None
        Z_j  = None
        if idx_ji.numel() > 0:
            j_nodes = dst[idx_ji]
            i_nodes = src[idx_ji]
            y1s = x_out[torch.tensor([f1_map[int(v)] for v in j_nodes.tolist()], device=device)]
            y2s = x_out[torch.tensor([f2_map[int(v)] for v in j_nodes.tolist()], device=device)]
            w_ij = edge_attr[idx_ji]
            x_i  = x[i_nodes]
            p1_j, p2_j, pB_j, Z_j = self._p12_both_batched(y1s, y2s, w_ij, x_i)  # [Kji]

        # Occurrence-indexed recording
        occ_counter = defaultdict(int)
        dir_p12: Dict[Tuple[int,int], Tuple[torch.Tensor, torch.Tensor]] = {}

        # We loop once over edges only to (a) form occurrence keys,
        # (b) materialize final child sets using the batched probs we computed,
        # (c) accumulate logP/entropy.
        new_edges_2c = []
        ptr_ij = ptr_ji = 0  # cursors into the batched arrays
        for k in range(M):
            i = int(src[k]); j = int(dst[k])
            w = edge_attr[k]

            # helper to fetch probabilities/logits already computed in batch arrays
            # for the specific direction if needed
            # ---- i -> j ----
            if is_static[i]:
                S_ij = {f_map[i]}
                lp_i = x.new_tensor(0.); ent_i = x.new_tensor(0.); p12_i = None
            elif (i in bj_choice) and (j == bj_choice[i]):
                S_ij = {f1_map[i], f2_map[i]}
                lp_i = x.new_tensor(0.); ent_i = x.new_tensor(0.); p12_i = None
            else:
                # i is unpooled and this edge is NOT the special 2b neighbor → consume from batched arrays
                if pos_ij[k] == -1:
                    # Shouldn't happen; safety check
                    raise RuntimeError(f"2c: expected i->j probs but pos_ij[{k}] == -1")
                idx = pos_ij[k]
                pi1, pi2, piB = p1_i[idx], p2_i[idx], pB_i[idx]
                Zi = Z_i[idx]
                if replay_mode:
                    key_ij = f"{i}->{j}#{occ_counter[(i, j)]}"
                    occ_counter[(i, j)] += 1
                    set_rec = set(int(s) for s in actions_to_replay['step2c_sets'][key_ij])
                    has1 = (f1_map[i] in set_rec)
                    has2 = (f2_map[i] in set_rec)
                    choice_i = 0 if (has1 and not has2) else 1 if (has2 and not has1) else 2
                else:
                    u = torch.rand((), generator=rng, device=device)
                    if u < pi1:
                        choice_i = 0
                    elif u < (pi1 + pi2):
                        choice_i = 1
                    else:
                        choice_i = 2
                if choice_i == 0:
                    S_ij = {f1_map[i]}
                    lp_i = torch.log(pi1.clamp_min(1e-9))
                elif choice_i == 1:
                    S_ij = {f2_map[i]}
                    lp_i = torch.log(pi2.clamp_min(1e-9))
                else:
                    S_ij = {f1_map[i], f2_map[i]}
                    lp_i = torch.log(piB.clamp_min(1e-9))
                Zi_s = Zi.clamp(1e-9, 1.0)
                ent_i = -(Zi_s * Zi_s.log()).sum()
                p12_i = (pi1, pi2)
                if not replay_mode:
                    key_ij = f"{i}->{j}#{occ_counter[(i, j)]}"
                    occ_counter[(i, j)] += 1
                    actions_recorded['step2c_sets'][key_ij] = [int(a) for a in sorted(S_ij)]

            # ---- j -> i ----
            if is_static[j]:
                S_ji = {f_map[j]}
                lp_j = x.new_tensor(0.); ent_j = x.new_tensor(0.); p12_j = None
            elif (j in bj_choice) and (i == bj_choice[j]):
                S_ji = {f1_map[j], f2_map[j]}
                lp_j = x.new_tensor(0.); ent_j = x.new_tensor(0.); p12_j = None
            else:
                if pos_ji[k] == -1:
                    raise RuntimeError(f"2c: expected j->i probs but pos_ji[{k}] == -1")
                idx = pos_ji[k]
                pj1, pj2, pjB = p1_j[idx], p2_j[idx], pB_j[idx]
                Zj = Z_j[idx]
                if replay_mode:
                    key_ji = f"{j}->{i}#{occ_counter[(j, i)]}"
                    occ_counter[(j, i)] += 1
                    set_rec = set(int(s) for s in actions_to_replay['step2c_sets'][key_ji])
                    has1 = (f1_map[j] in set_rec)
                    has2 = (f2_map[j] in set_rec)
                    choice_j = 0 if (has1 and not has2) else 1 if (has2 and not has1) else 2
                else:
                    u = torch.rand((), generator=rng, device=device)
                    if u < pj1:
                        choice_j = 0
                    elif u < (pj1 + pj2):
                        choice_j = 1
                    else:
                        choice_j = 2
                if choice_j == 0:
                    S_ji = {f1_map[j]}
                    lp_j = torch.log(pj1.clamp_min(1e-9))
                elif choice_j == 1:
                    S_ji = {f2_map[j]}
                    lp_j = torch.log(pj2.clamp_min(1e-9))
                else:
                    S_ji = {f1_map[j], f2_map[j]}
                    lp_j = torch.log(pjB.clamp_min(1e-9))
                Zj_s = Zj.clamp(1e-9, 1.0)
                ent_j = -(Zj_s * Zj_s.log()).sum()
                p12_j = (pj1, pj2)
                if not replay_mode:
                    key_ji = f"{j}->{i}#{occ_counter[(j, i)]}"
                    occ_counter[(j, i)] += 1
                    actions_recorded['step2c_sets'][key_ji] = [int(b) for b in sorted(S_ji)]

            # cursor advance if we consumed from batched arrays
            if not is_static[i] and not (i in bj_choice and j == bj_choice[i]):
                ptr_ij += 1
            if not is_static[j] and not (j in bj_choice and i == bj_choice[j]):
                ptr_ji += 1

            logP = logP + lp_i + lp_j
            total_entropy = total_entropy + ent_i + ent_j
            if p12_i is not None: dir_p12[(j, i)] = p12_i  # store for 2d (note the key)
            if p12_j is not None: dir_p12[(i, j)] = p12_j

            # materialize directed edges from sets
            for a in S_ij:
                for b_ in S_ji:
                    new_edges_2c.append([int(a), int(b_)])

        new_edges.extend(new_edges_2c)

        # add intra-links
        for idx, parent in enumerate(Iu.tolist()):
            if len(Iu) > 0 and Vc_mask[idx]:
                new_edges.append([f1_map[parent], f2_map[parent]])

        # For step 2d we need a stable lookup of existing undirected child edges
        stable_edge_set_lookup = set()
        if new_edges:
            for a, b in new_edges:
                stable_edge_set_lookup.add(tuple(sorted((a, b))))

        # ========== Step 2d: extra edges ==========
        if len(Iu) > 0:
            Eu = set()
            for k in range(M):
                i, j = int(src[k]), int(dst[k])
                if (i in f1_map) and (j in f1_map):
                    Eu.add(tuple(sorted((i, j))))
            eu_pairs = sorted(list(Eu))

            def N_size(node_self, node_other):
                if node_self in f_map: return 1
                c1, c2 = f1_map[node_self], f2_map[node_self]
                count = 0
                imgs = [f_map[node_other]] if node_other in f_map else [f1_map[node_other], f2_map[node_other]]
                for c in (c1, c2):
                    for im in imgs:
                        if tuple(sorted((c, im))) in stable_edge_set_lookup:
                            count += 1
                            break
                return max(1, min(2, count))

            logP_A = x.new_zeros(())
            added_edges_2d = []

            for (i, j) in eu_pairs:
                pair = self._canon_pair(i, j)
                # locate ek for (i,j) undirected
                ek = None
                for kk in range(M):
                    a, b = int(src[kk]), int(dst[kk])
                    if {a, b} == {i, j}: ek = kk; break
                w = edge_attr[ek] if ek is not None else x.new_zeros(self.dw)

                pa = self._smooth_bern(self.mlp_ie_a(torch.cat([x[i], x[j], w], dim=0).unsqueeze(0)).squeeze(0).squeeze(-1))

                if replay_mode:
                    chosen = bool(actions_to_replay["step2d_pa"][pair])
                else:
                    u1 = torch.rand((), generator=rng, device=device)
                    chosen = (u1 < pa)
                    actions_recorded["step2d_pa"][pair] = int(chosen)

                logP_A = logP_A + (torch.log(pa.clamp_min(1e-9)) if chosen else torch.log((1 - pa).clamp_min(1e-9)))
                pa_stable = pa.clamp(1e-9, 1.0 - 1e-9)
                total_entropy = total_entropy + (-(pa_stable * pa_stable.log() + (1 - pa_stable) * (1 - pa_stable).log()))

                if not chosen:
                    if not replay_mode:
                        actions_recorded["step2d_rij"][pair] = ("none", 0)
                    continue

                n_i, n_j = N_size(i, j), N_size(j, i)
                if replay_mode:
                    side_tag, pick_idx = actions_to_replay["step2d_rij"][pair]
                    if side_tag in ("pick_j", "pick_i"):
                        assert self._canon_pair(i, j) in actions_to_replay.get("step2d_edges", {}), \
                            f"Missing step2d_edges for pair {self._canon_pair(i, j)}"
                    if side_tag == "pick_j":
                        key = (i, j)
                        p1, p2 = dir_p12.get(key, (x.new_tensor(0.5), x.new_tensor(0.5)))
                        denom = (p1 + p2 + 1e-9)
                        pick_j = int(pick_idx)
                        ci, cj = actions_to_replay["step2d_edges"][self._canon_pair(i, j)]
                        added_edges_2d.append([int(ci), int(cj)])
                        logP_A = logP_A + torch.log((p1 if pick_j == 1 else p2).div(denom).clamp_min(1e-9))
                        prob_pick1 = (p1 / denom).clamp(1e-9, 1 - 1e-9)
                        total_entropy = total_entropy + (-(prob_pick1 * prob_pick1.log()
                                                          + (1 - prob_pick1) * (1 - prob_pick1).log()))
                    elif side_tag == "pick_i":
                        key = (j, i)
                        p1, p2 = dir_p12.get(key, (x.new_tensor(0.5), x.new_tensor(0.5)))
                        denom = (p1 + p2 + 1e-9)
                        pick_i = int(pick_idx)
                        ci, cj = actions_to_replay["step2d_edges"][self._canon_pair(i, j)]
                        added_edges_2d.append([int(ci), int(cj)])
                        logP_A = logP_A + torch.log((p1 if pick_i == 1 else p2).div(denom).clamp_min(1e-9))
                        prob_pick1 = (p1 / denom).clamp(1e-9, 1 - 1e-9)
                        total_entropy = total_entropy + (-(prob_pick1 * prob_pick1.log()
                                                          + (1 - prob_pick1) * (1 - prob_pick1).log()))
                    else:
                        pass
                else:
                    if (n_i + n_j) == 3:
                        if n_i == 1 and n_j == 2:
                            key = (i, j)
                            p1, p2 = dir_p12.get(key, (torch.tensor(0.5, device=device), torch.tensor(0.5, device=device)))
                            denom = (p1 + p2 + 1e-9)
                            prob_pick1 = (p1 / denom).clamp(1e-9, 1 - 1e-9)
                            u2 = torch.rand((), generator=rng, device=device)
                            pick_j = 1 if u2 < prob_pick1 else 2
                            actions_recorded["step2d_rij"][pair] = ("pick_j", int(pick_j))
                            logP_A = logP_A + torch.log((p1 if pick_j == 1 else p2).div(denom).clamp_min(1e-9))
                            ci_options = sorted(list({f1_map[i], f2_map[i]}))
                            cj_options = sorted(list({f1_map[j], f2_map[j]}))
                            ci = next(c for c in ci_options if not any(tuple(sorted((c, t))) in stable_edge_set_lookup for t in cj_options))
                            cj = f1_map[j] if pick_j == 1 else f2_map[j]
                            added_edges_2d.append([ci, cj])
                            actions_recorded.setdefault("step2d_edges", {})[self._canon_pair(i, j)] = [int(ci), int(cj)]
                            total_entropy = total_entropy + (-(prob_pick1 * prob_pick1.log()
                                                              + (1 - prob_pick1) * (1 - prob_pick1).log()))
                        elif n_j == 1 and n_i == 2:
                            key = (j, i)
                            p1, p2 = dir_p12.get(key, (torch.tensor(0.5, device=device), torch.tensor(0.5, device=device)))
                            denom = (p1 + p2 + 1e-9)
                            prob_pick1 = (p1 / denom).clamp(1e-9, 1 - 1e-9)
                            u2 = torch.rand((), generator=rng, device=device)
                            pick_i = 1 if u2 < prob_pick1 else 2
                            actions_recorded["step2d_rij"][pair] = ("pick_i", int(pick_i))
                            logP_A = logP_A + torch.log((p1 if pick_i == 1 else p2).div(denom).clamp_min(1e-9))
                            cj_options = sorted(list({f1_map[j], f2_map[j]}))
                            ci_options = sorted(list({f1_map[i], f2_map[i]}))
                            cj = next(c for c in cj_options if not any(tuple(sorted((c, t))) in stable_edge_set_lookup for t in ci_options))
                            ci = f1_map[i] if pick_i == 1 else f2_map[i]
                            added_edges_2d.append([ci, cj])
                            actions_recorded.setdefault("step2d_edges", {})[self._canon_pair(i, j)] = [int(ci), int(cj)]
                            total_entropy = total_entropy + (-(prob_pick1 * prob_pick1.log()
                                                              + (1 - prob_pick1) * (1 - prob_pick1).log()))
                    else:
                        actions_recorded["step2d_rij"][pair] = ("none", 0)
                        pass

            if added_edges_2d:
                new_edges.extend(added_edges_2d)
            logP = logP + logP_A

        # ========== Finalize edges, attrs ==========
        if not new_edges:
            edge_index_out = torch.empty(2, 0, dtype=torch.long, device=device)
        else:
            edge_index_out = torch.tensor(new_edges, dtype=torch.long, device=device).t().contiguous()
            edge_index_out, _ = coalesce(edge_index_out, None, num_nodes=x_out.size(0))

        if edge_index_out.size(1) == 0:
            edge_attr_out = torch.empty(0, self.du, dtype=x.dtype, device=device)
        else:
            yk, yl = x_out[edge_index_out[0]], x_out[edge_index_out[1]]
            edge_attr_out = self.mlp_u(self.agg(yk, yl))

        parent_map = {"f": f_map, "f1": f1_map, "f2": f2_map}
        sets = {"Is": Is, "Iu": Iu, "Vc": Vc}

        sig_out = {
            "N": int(x_out.size(0)),
            "E": int(edge_index_out.size(1)),
            "edges": tuple(zip(edge_index_out[0].tolist(), edge_index_out[1].tolist())),
        }
        if replay_mode:
            recsig = actions_to_replay.get("__sig_out__")
            assert recsig == sig_out, f"Replay graph mismatch at step OUTPUT:\nrecord={recsig}\nreplay={sig_out}"
        else:
            actions_recorded["__sig_out__"] = sig_out

        return x_out, edge_index_out, edge_attr_out, logP, total_entropy, parent_map, sets, actions_recorded


def seed_graph():
    e = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long)
    return e


def random_directed_graph_with_features(
    n: int,
    *,
    strongly_connected: bool = False,
    allow_self_loops: bool = False,
    p_extra: float = 0.2,              # probability for each *remaining* directed edge
    node_feat_dim: int = 8,            # per-node random features (fixed dim)
    desc: torch.Tensor | None = None,  # optional [desc_dim]; broadcast to all nodes
    desc_dim: int = 0,                 # if desc is None and desc_dim>0, sample a random desc
    include_degree_feats: bool = True, # append normalized in/out degree per node
    edge_feat_dim: int = 0,            # 0 → no edge_attr, >0 → return edge_attr
    edge_feat_style: str = "gaussian", # "gaussian" | "zeros"
    device: torch.device | str | None = None,
    rng: torch.Generator | None = None,
):
    """
    Returns:
      x_raw:        [n, D] node features (fixed dimensional)
      edge_index:   [2, E] directed edges (no duplicates). Underlying undirected graph is connected.
      edge_attr:    [E, edge_feat_dim] or None
      meta:         dict with helper info: {"desc": ..., "strongly_connected": ..., "p_extra": ...}

    Notes:
      - Connectivity:
          weak  (default): build a random spanning tree (undirected), then orient each tree edge randomly.
          strong: add a directed Hamiltonian cycle (over a random permutation).
      - Extra edges: sampled with probability p_extra from all remaining directed pairs.
      - Node features are FIXED-DIM across graphs (so the encoder can be a single MLP):
          [ desc (broadcast) | per-node gaussian | (optional) normalized degrees ]
      - If you prefer ID one-hots, add them yourself; they make input dim depend on n.
    """
    assert n >= 3, "Need at least 3 nodes"
    cpu = torch.device("cpu")
    if device is None:
        device = cpu
    if rng is None:
        rng = torch.Generator(device=cpu).manual_seed(torch.seed())

    edges = set()

    # ---- Base edges to ensure connectivity ----
    if strongly_connected:
        # Directed Hamiltonian cycle over a random permutation
        perm = torch.randperm(n, generator=rng, device=cpu).tolist()
        for i in range(n):
            u = perm[i]
            v = perm[(i + 1) % n]
            if allow_self_loops or (u != v):
                edges.add((u, v))
    else:
        # Weak connectivity: random spanning tree (undirected), then random orientation for each tree edge
        for i in range(1, n):
            # connect i to a random previous node (classic random tree)
            p = torch.randint(low=0, high=i, size=(1,), generator=rng, device=cpu).item()
            if torch.rand((), generator=rng, device=cpu) < 0.5:
                u, v = p, i
            else:
                u, v = i, p
            if allow_self_loops or (u != v):
                edges.add((u, v))

    # ---- Extra edges ----
    # Candidate directed pairs not yet used
    for u in range(n):
        for v in range(n):
            if (not allow_self_loops) and (u == v):
                continue
            if (u, v) in edges:
                continue
            if torch.rand((), generator=rng, device=cpu) < p_extra:
                edges.add((u, v))

    # ---- Tensors: edge_index [2, E] ----
    if len(edges) == 0:
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
    else:
        ei = torch.tensor(list(edges), dtype=torch.long, device=cpu).t().contiguous()
        edge_index = ei.to(device)

    # ---- Node features ----
    parts = []

    # (a) description vector broadcast to all nodes
    if desc is None and desc_dim > 0:
        # sample a random description (unit-normalized Gaussian)
        d = torch.randn(desc_dim, generator=rng, device=cpu)
        desc = d / (d.norm(p=2) + 1e-8)
    if desc is not None:
        assert desc.dim() == 1, "desc must be a 1D vector"
        parts.append(desc.to(cpu).unsqueeze(0).repeat(n, 1))

    # (b) per-node Gaussian features
    if node_feat_dim > 0:
        parts.append(torch.randn(n, node_feat_dim, generator=rng, device=cpu))

    # (c) optional degree features (normalized in/out degree)
    if include_degree_feats:
        if edge_index.numel() == 0:
            deg_in = torch.zeros(n, device=cpu)
            deg_out = torch.zeros(n, device=cpu)
        else:
            deg_out = torch.zeros(n, device=cpu)
            deg_in  = torch.zeros(n, device=cpu)
            for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                deg_out[u] += 1
                deg_in[v]  += 1
            norm = max(1, n - 1)
            deg_out = deg_out / norm
            deg_in  = deg_in  / norm
        parts.append(torch.stack([deg_in, deg_out], dim=1))

    x_raw = torch.cat(parts, dim=1) if parts else torch.zeros(n, 0, device=cpu)
    x_raw = x_raw.to(device)

    # ---- Edge features ----
    edge_attr = None
    E = edge_index.size(1)
    if edge_feat_dim > 0:
        if edge_feat_style == "gaussian":
            edge_attr = torch.randn(E, edge_feat_dim, generator=rng, device='cpu').to(device)
        elif edge_feat_style == "zeros":
            edge_attr = torch.zeros(E, edge_feat_dim, device=device)
        else:
            raise ValueError(f"Unsupported edge_feat_style: {edge_feat_style}")

    meta = {
        "desc": None if desc is None else desc.to(device),
        "strongly_connected": strongly_connected,
        "p_extra": float(p_extra),
        "n": n,
    }
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
        x, ei, ea, meta = random_directed_graph_with_features(
            n,
            strongly_connected=False,
            allow_self_loops=False,
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
            directed=True, wl_iters=2
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
      lr, entropy_coef, unpool_size, batch_size, ppo_update_epochs, ppo_clip_eps
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

    unpool = GuoUnpool(dx=DX, dw=DW, dy=DX, du=DW,
                       kv=unpool_size, kia=unpool_size,
                       kie=unpool_size, kw=unpool_size).to(device)
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
                    directed=True, wl_iters=2
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
        # Delete the dehb_unpool_out/ directory to force a fresh search.
        DEHB_OUT = "dehb_unpool_out"
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
        best_path = os.path.join("dehb_unpool_out", "best_config.json")
        if os.path.exists(best_path):
            with open(best_path) as f:
                config = json.load(f)["best_config"]
            print(f"Loaded tuned config: {config}")
        else:
            config = {"lr": 3e-4, "entropy_coef": 0.01, "unpool_size": 256,
                      "batch_size": 256, "ppo_update_epochs": 4, "ppo_clip_eps": 0.2}
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