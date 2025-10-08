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
from typing import Optional, Dict, Tuple, List, Set
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

        # PS1/PS2 projection indices (d' = floor(dx/2) + floor(dx/4))
        ds = dx // 2
        D = dx // 4
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

        pr = self.mlp_r(x[I_r]).squeeze(-1)
        if replay_mode:
            choose_unpool = actions_to_replay['step1a_unpool'][0]
        else:
            u = torch.rand(pr.shape, dtype=pr.dtype, device=pr.device, generator=rng)
            choose_unpool = (u < pr)
            actions_recorded['step1a_unpool'].append(choose_unpool)

        Iu = torch.cat([I_u, I_r[choose_unpool]], dim=0)
        Is = torch.cat([I_s, I_r[~choose_unpool]], dim=0)

        logP = logP + torch.sum(torch.log(pr + 1e-9)[choose_unpool]) \
                    + torch.sum(torch.log(1 - pr + 1e-9)[~choose_unpool])

        pr_stable = pr.clamp(1e-9, 1.0 - 1e-9)
        entropy_r = -(pr_stable * torch.log(pr_stable) + (1 - pr_stable) * torch.log(1 - pr_stable))
        total_entropy = total_entropy + entropy_r.sum()

        # ========== Step 1b: node features / packing ==========
        PS1, PS2 = self._project(x)

        if not replay_mode:
            actions_recorded["__Is_order__"] = [int(i) for i in Is.tolist()]
            actions_recorded["__Iu_order__"] = [int(i) for i in Iu.tolist()]
        else:
            Is_rec = actions_to_replay.get("__Is_order__")
            Iu_rec = actions_to_replay.get("__Iu_order__")
            assert Is_rec is not None and Iu_rec is not None, "Missing recorded Is/Iu ordering"
            Is = torch.tensor(Is_rec, dtype=torch.long, device=device)
            Iu = torch.tensor(Iu_rec, dtype=torch.long, device=device)

        x_static = self.mlp_y(PS1[Is])
        x_c1 = self.mlp_y(PS1[Iu])
        x_c2 = self.mlp_y(PS2[Iu])
        x_out = torch.cat([x_static, x_c1, x_c2], dim=0)

        f_map  = {int(i): idx for idx, i in enumerate(Is.tolist())}
        base_c1 = len(Is)
        base_c2 = len(Is) + len(Iu)
        f1_map = {int(i): base_c1 + k for k, i in enumerate(Iu.tolist())}
        f2_map = {int(i): base_c2 + k for k, i in enumerate(Iu.tolist())}

        # ========== Step 2a: intra-links ==========
        if len(Iu) > 0:
            y1 = x_out[torch.arange(len(Iu), device=device) + base_c1]
            y2 = x_out[torch.arange(len(Iu), device=device) + base_c2]
            p_intra = self.mlp_ia(self.agg(y1, y2)).squeeze(-1)

            if replay_mode:
                Vc_mask = actions_to_replay['step2a_intra'][0]
            else:
                u = torch.rand(p_intra.shape, dtype=p_intra.dtype, device=p_intra.device, generator=rng)
                Vc_mask = (u < p_intra)
                actions_recorded['step2a_intra'].append(Vc_mask)

            Vc = Iu[Vc_mask]
            logP = logP + torch.sum(torch.log(p_intra + 1e-9)[Vc_mask]) \
                        + torch.sum(torch.log(1 - p_intra + 1e-9)[~Vc_mask])

            p_intra_stable = p_intra.clamp(1e-9, 1.0 - 1e-9)
            entropy_ia = -(p_intra_stable * torch.log(p_intra_stable) + (1 - p_intra_stable) * torch.log(1 - p_intra_stable))
            total_entropy = total_entropy + entropy_ia.sum()
        else:
            Vc_mask = torch.tensor([], dtype=torch.bool, device=device)
            Vc = Iu

        # Build adjacency (deterministic by earlier sort) for neighbors and 2b
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        M = len(src)
        adj = [[] for _ in range(N)]
        for k in range(M):
            i, j = src[k], dst[k]
            adj[i].append((j, k))
            adj[j].append((i, k))
        for u_idx in range(N):
            adj[u_idx].sort(key=lambda t: t[0])

        # Ensure replay has 2b picks for all needed parents
        if replay_mode:
            needed = set(int(u) for u in Iu[~Vc_mask].tolist())
            have   = set(int(k) for k in actions_to_replay.get("step2b_pick", {}).keys())
            missing = needed - have
            assert not missing, f"Replay missing step2b picks for parents: {sorted(missing)}"

        # ========== Step 2b: pick b_j for parents without intra-link ==========
        iu_no_intra = Iu[~Vc_mask]
        bj_choice: Dict[int, int] = {}
        for parent in iu_no_intra.tolist():
            nbrs = adj[parent]
            if not nbrs:
                continue

            y1p = x_out[f1_map[parent]]
            y2p = x_out[f2_map[parent]]

            # score each neighbor (match original: add batch dim for Linear)
            scores = []
            for (nei, ek) in nbrs:
                w = edge_attr[ek]
                x_nei = x[nei]
                s = self.mlp_c(torch.cat([self.agg(y1p, y2p), w, x_nei], dim=0).unsqueeze(0)).squeeze(0)
                scores.append(s)
            scores = torch.stack(scores, dim=0).squeeze(-1)
            probs = F.softmax(scores, dim=0)

            if replay_mode:
                rec = actions_to_replay['step2b_pick'].get(int(parent), None)
                pick = None
                if rec is not None:
                    # Prefer matching by ek (robust to duplicate neighbors / ordering)
                    if isinstance(rec, dict) and "ek" in rec:
                        ek_id = int(rec["ek"])
                        for i_n, (_n, _ek) in enumerate(nbrs):
                            if int(_ek) == ek_id:
                                pick = i_n
                                break
                if pick is None:
                    raise RuntimeError(
                        f"Step 2b replay failed for parent {parent}. "
                        f"Could not find recorded action: {rec}. "
                        f"Available neighbors (nei, ek): {[(int(n), int(ek)) for n, ek in nbrs]}"
                    )
            else:
                pick = torch.multinomial(probs, 1, generator=rng).item()
                actions_recorded['step2b_pick'][int(parent)] = {
                    "nei": int(nbrs[pick][0]),
                    "ek":  int(nbrs[pick][1]),
                }

            bj_choice[parent] = nbrs[pick][0]
            logP = logP + torch.log(probs[pick].clamp(min=1e-9))
            entropy_bj = -(probs.clamp(min=1e-9) * torch.log(probs.clamp(min=1e-9))).sum()
            total_entropy = total_entropy + entropy_bj

        if not replay_mode:
            actions_recorded["__bj_choice__"] = {int(k): int(v) for k, v in bj_choice.items()}
        else:
            rec_bj = actions_to_replay.get("__bj_choice__", {})
            assert {int(k): int(v) for k, v in bj_choice.items()} == rec_bj, \
                f"Replay diverged in 2b bj_choice.\nrecord={rec_bj}\nreplay={bj_choice}"

        # ---- helper: interlink probabilities (with optional preference scores)
        def get_p12_both(y1s, y2s, w_ij, x_other):
            if self.use_preference:
                hs1 = self.mlp_ie1(torch.cat([y1s, w_ij, x_other], dim=0).unsqueeze(0)).squeeze(0)
                hs2 = self.mlp_ie1(torch.cat([y2s, w_ij, x_other], dim=0).unsqueeze(0)).squeeze(0)
                hb  = self.mlp_ie2(torch.cat([self.agg(y1s, y2s), w_ij, x_other], dim=0).unsqueeze(0)).squeeze(0)
                h0s = self.mlp_zero_s(y1s.unsqueeze(0)).squeeze(0)
                h0b = self.mlp_zero_b(x_other.unsqueeze(0)).squeeze(0)
                Z_probs = F.softmax(torch.stack([hs1, hs2, hb, h0s + h0b], dim=0).squeeze(-1), dim=0)
                p1, p2, p_both = Z_probs[0], Z_probs[1], Z_probs[2]
            else:
                s1 = self.mlp_ie1(torch.cat([y1s, w_ij, x_other], dim=0).unsqueeze(0)).squeeze(0)
                s2 = self.mlp_ie1(torch.cat([y2s, w_ij, x_other], dim=0).unsqueeze(0)).squeeze(0)
                sb = self.mlp_ie2(torch.cat([self.agg(y1s, y2s), w_ij, x_other], dim=0).unsqueeze(0)).squeeze(0)
                logits = torch.cat([s1, s2, sb], dim=-1).squeeze(0)
                Z_probs = F.softmax(logits, dim=-1)
                p1, p2, p_both = Z_probs[0], Z_probs[1], Z_probs[2]
            return p1, p2, p_both, Z_probs

        # ========== Step 2c: expand edges per direction (occurrence-indexed) ==========
        new_edges = []
        dir_p12: Dict[Tuple[int,int], Tuple[torch.Tensor, torch.Tensor]] = {}

        def step2c_key_occ(i, j, counter: Dict[Tuple[int,int], int]):
            k = counter[(int(i), int(j))]
            counter[(int(i), int(j))] += 1
            return f"{int(i)}->{int(j)}#{k}"

        occ_counter = defaultdict(int)

        for k in range(M):
            i, j = src[k], dst[k]
            w_ij = edge_attr[k]

            if replay_mode:
                key_ij = step2c_key_occ(i, j, occ_counter)
                key_ji = step2c_key_occ(j, i, occ_counter)
                S_ij = set(int(s) for s in actions_to_replay["step2c_sets"][key_ij])
                S_ji = set(int(s) for s in actions_to_replay["step2c_sets"][key_ji])
                assert len(S_ij) in (1, 2), f"Bad step2c_sets[{key_ij}]"
                assert len(S_ji) in (1, 2), f"Bad step2c_sets[{key_ji}]"

                # compute lp/entropy using recorded choices (no RNG)
                if i in f_map:
                    lp_i = x.new_tensor(0.); ent_i = x.new_tensor(0.); p12_i = None
                else:
                    y1s_i = x_out[f1_map[i]]; y2s_i = x_out[f2_map[i]]
                    p1_i, p2_i, pB_i, Z_i = get_p12_both(y1s_i, y2s_i, w_ij, x[j])
                    has1 = (f1_map[i] in S_ij); has2 = (f2_map[i] in S_ij)
                    choice_i = 0 if (has1 and not has2) else 1 if (has2 and not has1) else 2
                    lp_i = torch.log((p1_i if choice_i == 0 else p2_i if choice_i == 1 else pB_i).clamp_min(1e-9))
                    Zs = Z_i.clamp(1e-9, 1.0); ent_i = -(Zs * torch.log(Zs)).sum()
                    p12_i = (p1_i, p2_i)

                if j in f_map:
                    lp_j = x.new_tensor(0.); ent_j = x.new_tensor(0.); p12_j = None
                else:
                    y1s_j = x_out[f1_map[j]]; y2s_j = x_out[f2_map[j]]
                    p1_j, p2_j, pB_j, Z_j = get_p12_both(y1s_j, y2s_j, w_ij, x[i])
                    has1 = (f1_map[j] in S_ji); has2 = (f2_map[j] in S_ji)
                    choice_j = 0 if (has1 and not has2) else 1 if (has2 and not has1) else 2
                    lp_j = torch.log((p1_j if choice_j == 0 else p2_j if choice_j == 1 else pB_j).clamp_min(1e-9))
                    Zs = Z_j.clamp(1e-9, 1.0); ent_j = -(Zs * torch.log(Zs)).sum()
                    p12_j = (p1_j, p2_j)

                logP = logP + lp_i + lp_j
                total_entropy = total_entropy + ent_i + ent_j
                if p12_i is not None: dir_p12[(j, i)] = p12_i
                if p12_j is not None: dir_p12[(i, j)] = p12_j

                # materialize edges from recorded sets (directed)
                for a in S_ij:
                    for b_ in S_ji:
                        new_edges.append([int(a), int(b_)])

            else:
                # sample for i->j
                if i in f_map:
                    S_ij = {f_map[i]}
                    lp_i = x.new_tensor(0.); ent_i = x.new_tensor(0.); p12_i = None
                elif (i in bj_choice) and (j == bj_choice[i]):
                    S_ij = {f1_map[i], f2_map[i]}
                    lp_i = x.new_tensor(0.); ent_i = x.new_tensor(0.); p12_i = None
                else:
                    y1s_i = x_out[f1_map[i]]; y2s_i = x_out[f2_map[i]]
                    p1_i, p2_i, pB_i, Z_i = get_p12_both(y1s_i, y2s_i, w_ij, x[j])
                    u = torch.rand((), generator=rng, device=device)
                    if u < p1_i:
                        choice_i = 0; S_ij = {f1_map[i]}; lp_i = torch.log(p1_i.clamp_min(1e-9))
                    elif u < (p1_i + p2_i):
                        choice_i = 1; S_ij = {f2_map[i]}; lp_i = torch.log(p2_i.clamp_min(1e-9))
                    else:
                        choice_i = 2; S_ij = {f1_map[i], f2_map[i]}; lp_i = torch.log(pB_i.clamp_min(1e-9))
                    Zs = Z_i.clamp(1e-9, 1.0); ent_i = -(Zs * torch.log(Zs)).sum()
                    p12_i = (p1_i, p2_i)

                # sample for j->i
                if j in f_map:
                    S_ji = {f_map[j]}
                    lp_j = x.new_tensor(0.); ent_j = x.new_tensor(0.); p12_j = None
                elif (j in bj_choice) and (i == bj_choice[j]):
                    S_ji = {f1_map[j], f2_map[j]}
                    lp_j = x.new_tensor(0.); ent_j = x.new_tensor(0.); p12_j = None
                else:
                    y1s_j = x_out[f1_map[j]]; y2s_j = x_out[f2_map[j]]
                    p1_j, p2_j, pB_j, Z_j = get_p12_both(y1s_j, y2s_j, w_ij, x[i])
                    u = torch.rand((), generator=rng, device=device)
                    if u < p1_j:
                        choice_j = 0; S_ji = {f1_map[j]}; lp_j = torch.log(p1_j.clamp_min(1e-9))
                    elif u < (p1_j + p2_j):
                        choice_j = 1; S_ji = {f2_map[j]}; lp_j = torch.log(p2_j.clamp_min(1e-9))
                    else:
                        choice_j = 2; S_ji = {f1_map[j], f2_map[j]}; lp_j = torch.log(pB_j.clamp_min(1e-9))
                    Zs = Z_j.clamp(1e-9, 1.0); ent_j = -(Zs * torch.log(Zs)).sum()
                    p12_j = (p1_j, p2_j)

                # record occurrence-indexed sets
                key_ij = step2c_key_occ(i, j, occ_counter)
                key_ji = step2c_key_occ(j, i, occ_counter)
                actions_recorded['step2c_sets'][key_ij] = [int(a) for a in sorted(S_ij)]
                actions_recorded['step2c_sets'][key_ji] = [int(b) for b in sorted(S_ji)]

                logP = logP + lp_i + lp_j
                total_entropy = total_entropy + ent_i + ent_j
                if p12_i is not None: dir_p12[(j, i)] = p12_i
                if p12_j is not None: dir_p12[(i, j)] = p12_j

                for a in S_ij:
                    for b_ in S_ji:
                        new_edges.append([int(a), int(b_)])

        # add intra-links
        for idx, parent in enumerate(Iu.tolist()):
            if len(Iu) > 0 and Vc_mask[idx]:
                new_edges.append([f1_map[parent], f2_map[parent]])

        stable_edge_set_lookup = set()
        if new_edges:
            for a, b in new_edges:
                stable_edge_set_lookup.add(tuple(sorted((a, b))))

        # ========== Step 2d: extra edges ==========
        if len(Iu) > 0:
            Eu = set()
            for k in range(M):
                i, j = src[k], dst[k]
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
                ek = None
                for kk in range(M):
                    a, b = src[kk], dst[kk]
                    if {a, b} == {i, j}: ek = kk; break
                w = edge_attr[ek] if ek is not None else x.new_zeros(self.dw)

                pa = self.mlp_ie_a(torch.cat([x[i], x[j], w], dim=0).unsqueeze(0)).squeeze(0).squeeze(-1)

                if replay_mode:
                    chosen = bool(actions_to_replay["step2d_pa"][pair])
                else:
                    u1 = torch.rand((), generator=rng, device=device)
                    chosen = (u1 < pa)
                    actions_recorded["step2d_pa"][pair] = int(chosen)

                logP_A = logP_A + (torch.log(pa + 1e-9) if chosen else torch.log(1 - pa + 1e-9))
                pa_stable = pa.clamp(1e-9, 1.0 - 1e-9)
                entropy_pa = -(pa_stable * torch.log(pa_stable) + (1 - pa_stable) * torch.log(1 - pa_stable))
                total_entropy = total_entropy + entropy_pa

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
                        key = (i, j)  # p12 for node_self=j stored under (i,j)
                        p1, p2 = dir_p12.get(key, (x.new_tensor(0.5), x.new_tensor(0.5)))
                        denom = (p1 + p2 + 1e-9)
                        pick_j = int(pick_idx)
                        ci, cj = actions_to_replay["step2d_edges"][self._canon_pair(i, j)]
                        added_edges_2d.append([int(ci), int(cj)])
                        logP_A = logP_A + torch.log((p1 if pick_j == 1 else p2).div(denom).clamp(min=1e-9))
                        prob_pick1 = (p1 / denom).clamp(1e-9, 1 - 1e-9)
                        entropy_rij = -(prob_pick1 * prob_pick1.log() + (1 - prob_pick1) * (1 - prob_pick1).log())
                        total_entropy = total_entropy + entropy_rij

                    elif side_tag == "pick_i":
                        key = (j, i)  # p12 for node_self=i stored under (j,i)
                        p1, p2 = dir_p12.get(key, (x.new_tensor(0.5), x.new_tensor(0.5)))
                        denom = (p1 + p2 + 1e-9)
                        pick_i = int(pick_idx)
                        ci, cj = actions_to_replay["step2d_edges"][self._canon_pair(i, j)]
                        added_edges_2d.append([int(ci), int(cj)])
                        logP_A = logP_A + torch.log((p1 if pick_i == 1 else p2).div(denom).clamp(min=1e-9))
                        prob_pick1 = (p1 / denom).clamp(1e-9, 1 - 1e-9)
                        entropy_rij = -(prob_pick1 * prob_pick1.log() + (1 - prob_pick1) * (1 - prob_pick1).log())
                        total_entropy = total_entropy + entropy_rij

                    else:
                        pass  # "none"

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

                            logP_A = logP_A + torch.log((p1 if pick_j == 1 else p2).div(denom).clamp(min=1e-9))
                            ci_options = sorted(list({f1_map[i], f2_map[i]}))
                            cj_options = sorted(list({f1_map[j], f2_map[j]}))
                            ci = next(c for c in ci_options if not any(tuple(sorted((c, t))) in stable_edge_set_lookup for t in cj_options))
                            cj = f1_map[j] if pick_j == 1 else f2_map[j]
                            added_edges_2d.append([ci, cj])

                            actions_recorded.setdefault("step2d_edges", {})[self._canon_pair(i, j)] = [int(ci), int(cj)]

                            entropy_rij = -(prob_pick1 * prob_pick1.log() + (1 - prob_pick1) * (1 - prob_pick1).log())
                            total_entropy = total_entropy + entropy_rij
                        elif n_j == 1 and n_i == 2:
                            key = (j, i)
                            p1, p2 = dir_p12.get(key, (torch.tensor(0.5, device=device), torch.tensor(0.5, device=device)))
                            denom = (p1 + p2 + 1e-9)
                            prob_pick1 = (p1 / denom).clamp(1e-9, 1 - 1e-9)

                            u2 = torch.rand((), generator=rng, device=device)
                            pick_i = 1 if u2 < prob_pick1 else 2
                            actions_recorded["step2d_rij"][pair] = ("pick_i", int(pick_i))

                            logP_A = logP_A + torch.log((p1 if pick_i == 1 else p2).div(denom).clamp(min=1e-9))
                            cj_options = sorted(list({f1_map[j], f2_map[j]}))
                            ci_options = sorted(list({f1_map[i], f2_map[i]}))
                            cj = next(c for c in cj_options if not any(tuple(sorted((c, t))) in stable_edge_set_lookup for t in ci_options))
                            ci = f1_map[i] if pick_i == 1 else f2_map[i]
                            added_edges_2d.append([ci, cj])

                            actions_recorded.setdefault("step2d_edges", {})[self._canon_pair(i, j)] = [int(ci), int(cj)]

                            entropy_rij = -(prob_pick1 * prob_pick1.log() + (1 - prob_pick1) * (1 - prob_pick1).log())
                            total_entropy = total_entropy + entropy_rij
                    else:  # n_i + n_j == 4 → no extra edges needed
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
      - Node features are FIXED-DIM across graphs (so your encoder can be a single MLP):
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


if __name__ == "__main__":
    # ---------- imports & tiny patch ----------
    import json
    import copy
    from datetime import datetime
    import matplotlib.pyplot as plt
    from torch_geometric.nn import global_mean_pool
    import graph_similarity as GS

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 1234
    ACTOR_RNG = torch.Generator(device=DEVICE)
    ACTOR_RNG.manual_seed(SEED + 7)

    # ---------- config ----------
    LR           = 1e-4
    ENTROPY_COEF = 0.01
    EPOCHS       = 25
    BATCH_SIZE   = 64
    NOISE_STD    = 0
    PRINT_EVERY  = 1

    # dataset sizes
    N_TRAIN = 960
    N_VAL   = 1000
    N_MIN_NODES = 7
    N_MAX_NODES = 11

    # random graph generator knobs
    NODE_FEAT_DIM        = 5
    DW                   = 2  # input edge features
    DESC_BROADCAST_DIM   = 16
    EDGE_FEAT_DIM        = DW
    INCLUDE_DEG_FEATURES = True
    P_EXTRA              = 0.25

    K_MAX = 2

    def unpool_k_fixed(unpool, x0, ei0, ea0=None, k: int = K_MAX,
                       actions_to_replay: list[Dict] | None = None, rng: torch.Generator | None = None):
        x, ei, ea = x0, ei0, ea0
        logP_total = x0.new_zeros(())
        entropy_total = x0.new_zeros(())

        replay_mode = actions_to_replay is not None
        k_actions_recorded = []

        for step_k in range(k):
            actions_for_step = actions_to_replay[step_k] if replay_mode else None

            # The unpool module now returns the actions object directly
            x, ei, ea, logP, entropy, *_, actions_obj = unpool(
                x, ei, edge_attr=ea, actions_to_replay=actions_for_step, rng=rng
            )

            logP_total = logP_total + logP
            entropy_total = entropy_total + entropy
            if not replay_mode:
                k_actions_recorded.append(actions_obj)  # Append the dataclass object

        return x, ei, ea, logP_total, entropy_total, k_actions_recorded


    class LosslessGraphEncoder(nn.Module):
        """
        Encodes a graph into two scrambled tensors (integer and float), preserving
        all information perfectly. The scrambling is a fixed, seeded permutation.

        This avoids the floating-point precision issues of packing discrete data
        into a float tensor before scrambling.
        """

        def __init__(self, capN: int, node_dim: int, edge_dim: int, scramble_seed: int = 0):
            super().__init__()
            self.capN = int(capN)
            self.node_dim = int(node_dim)
            self.edge_dim = int(edge_dim)

            # Calculate the total size of discrete/integer components
            # 1 for n + capN*capN for the adjacency matrix
            self.int_dim = 1 + self.capN * self.capN

            # Calculate the total size of continuous/float components
            # capN*node_dim for node features + capN*capN*edge_dim for edge features
            self.float_dim = self.capN * self.node_dim + self.capN * self.capN * self.edge_dim

            # --- Create fixed, seeded permutations for scrambling ---
            g = torch.Generator(device="cpu").manual_seed(int(scramble_seed))

            # Permutation for integer data
            int_perm = torch.randperm(self.int_dim, generator=g)
            # Permutation for float data
            float_perm = torch.randperm(self.float_dim, generator=g)

            self.register_buffer("int_perm", int_perm)
            self.register_buffer("float_perm", float_perm)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Dict[
            str, torch.Tensor]:
            """
            Packs and scrambles the graph into a dictionary of two tensors.

            Args:
                x: Node features [n, node_dim]
                edge_index: Edge index [2, E] (directed)
                edge_attr: Edge attributes [E, edge_dim]

            Returns:
                A dictionary containing the scrambled integer and float data.
                {
                    "int_scrambled": [int_dim],
                    "float_scrambled": [float_dim]
                }
            """
            device = x.device
            n = x.size(0)
            assert n <= self.capN, f"Graph size n={n} exceeds capacity capN={self.capN}"

            # --- Pack Integer Data ---
            # 1. Node count
            n_tensor = torch.tensor([n], dtype=torch.int64, device=device)

            # 2. Adjacency Matrix (using bool/byte is efficient)
            adj_matrix = torch.zeros(self.capN, self.capN, dtype=torch.bool, device=device)
            if edge_index.numel() > 0:
                # Note: Assumes no self-loops, consistent with your generator
                adj_matrix[edge_index[0], edge_index[1]] = True

            # Flatten and concatenate all integer components
            int_flat = torch.cat([
                n_tensor,
                adj_matrix.view(-1).long()  # Flatten and cast to long for consistency
            ])

            # --- Pack Float Data ---
            # 3. Padded Node Features
            X_pad = torch.zeros(self.capN, self.node_dim, dtype=x.dtype, device=device)
            X_pad[:n, :] = x

            # 4. Padded Edge Features (in canonical adjacency matrix order)
            W_pad = torch.zeros(self.capN, self.capN, self.edge_dim, dtype=x.dtype, device=device)
            if edge_attr is not None and edge_index.numel() > 0:
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                W_pad[edge_index[0], edge_index[1]] = edge_attr

            # Flatten and concatenate all float components
            float_flat = torch.cat([
                X_pad.view(-1),
                W_pad.view(-1)
            ])

            # --- Apply Scrambling Permutations ---
            int_scrambled = int_flat[self.int_perm.to(device)]
            float_scrambled = float_flat[self.float_perm.to(device)]

            return {
                "int_scrambled": int_scrambled,
                "float_scrambled": float_scrambled,
            }


    class LosslessSeedFeaturizer(nn.Module):
        """
        Deterministically converts the scrambled, lossless graph encoding into
        the initial seed features for the generator. This is the inverse of the
        encoder, followed by a packing into the desired feature shape.
        """

        def __init__(self, capN: int, node_dim: int, edge_dim: int,
                     dx: int, n_seed: int, scramble_seed: int = 0, dim_check: bool = False):
            super().__init__()
            self.capN = int(capN)
            self.node_dim = int(node_dim)
            self.edge_dim = int(edge_dim)
            self.dx = int(dx)
            self.n_seed = int(n_seed)

            # Calculate dimensions to match the encoder
            self.int_dim = 1 + self.capN * self.capN
            self.float_dim = self.capN * self.node_dim + self.capN * self.capN * self.edge_dim

            # An int64 takes the space of two float32s
            # ... but we only use one because we don't need the full capacity
            self.int_as_float_dim = self.int_dim

            self.total_packed_dim = self.float_dim + self.int_as_float_dim

            seed_capacity = self.n_seed * self.dx
            if not dim_check:
                assert seed_capacity >= self.total_packed_dim, (
                    f"Seed capacity {seed_capacity} is less than packed data size "
                    f"{self.total_packed_dim}. Increase DX or N_SEED."
                )

            # --- Create INVERSE permutations for unscrambling ---
            g = torch.Generator(device="cpu").manual_seed(int(scramble_seed))
            int_perm = torch.randperm(self.int_dim, generator=g)
            float_perm = torch.randperm(self.float_dim, generator=g)

            # The inverse of a permutation is its argsort
            self.register_buffer("int_unperm", torch.argsort(int_perm))
            self.register_buffer("float_unperm", torch.argsort(float_perm))

        def forward(self, encoded_dict: Dict[str, torch.Tensor], noise_std: float = 0.0) -> torch.Tensor:
            """
            Unscrambles and packs the graph representation into seed features.

            Args:
                encoded_dict: The output from LosslessGraphEncoder.
                noise_std: Standard deviation for optional Gaussian noise.

            Returns:
                A tensor of shape [n_seed, dx] representing the initial node features.
            """
            device = encoded_dict["float_scrambled"].device

            # --- Apply Inverse Permutations to Unscramble ---
            int_flat = encoded_dict["int_scrambled"][self.int_unperm.to(device)]
            float_flat = encoded_dict["float_scrambled"][self.float_unperm.to(device)]

            # --- Lossless Union of Integer and Float Data ---
            # To combine the int64 tensor with the float32 tensor without losing
            # information, we could view the int64 data as raw bytes and interpret
            # those bytes as two float32s. This is a lossless "bit-cast".
            # [int_dim] (int64) -> [int_dim, 2] (int32) -> [int_dim * 2] (float32)

            # However, we're unlikely to need that much capacity.
            int_as_float = int_flat.to(torch.float32)

            # Concatenate the two float tensors into one large vector
            z_combined = torch.cat([float_flat, int_as_float], dim=0)

            # --- Pack into Seed Feature Matrix ---
            x_seed = torch.zeros(self.n_seed, self.dx, device=device)

            # Flatten the seed matrix to easily copy the data
            x_seed_flat = x_seed.view(-1)
            x_seed_flat[:self.total_packed_dim] = z_combined

            # Reshape back to the desired [n_seed, dx]
            x_seed = x_seed_flat.view(self.n_seed, self.dx)

            if noise_std > 0:
                x_seed = x_seed + torch.randn_like(x_seed) * noise_std

            return x_seed

    # ----------

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

    # ---------- dataset ----------
    def make_dataset(n_graphs: int):
        ds = []
        for _ in range(n_graphs):
            n = random.randint(N_MIN_NODES, N_MAX_NODES)
            x, ei, ea, meta = random_directed_graph_with_features(
                n,
                strongly_connected=False,
                allow_self_loops=False,
                p_extra=P_EXTRA,
                node_feat_dim=NODE_FEAT_DIM,
                desc_dim=DESC_BROADCAST_DIM,
                include_degree_feats= INCLUDE_DEG_FEATURES,
                edge_feat_dim=EDGE_FEAT_DIM,
                edge_feat_style="gaussian",
                device=DEVICE,
            )
            ds.append((x, ei, ea, meta))

        return ds

    print("Building datasets…")
    train_set = make_dataset(N_TRAIN)
    val_set   = make_dataset(N_VAL)
    print(f"train={len(train_set)} graphs, val={len(val_set)} graphs")

    # 1. Calculate dimensions (this part remains the same)
    capN = N_MAX_NODES
    RAW_NODE_DIM = train_set[0][0].size(1)
    N_SEED = 3
    encoder_for_dims = LosslessGraphEncoder(
        capN=capN, node_dim=RAW_NODE_DIM, edge_dim=EDGE_FEAT_DIM
    )
    featurizer_for_dims = LosslessSeedFeaturizer(
        capN=capN, node_dim=RAW_NODE_DIM, edge_dim=EDGE_FEAT_DIM, dx=1, n_seed=N_SEED, dim_check=True
    )
    packed_dim = featurizer_for_dims.total_packed_dim
    print(f"Truly lossless packed dim (float equivalent): {packed_dim}")

    DX = math.ceil(packed_dim / N_SEED)
    seed_ei = seed_graph().to(DEVICE)

    # 2. Instantiate the encoder and featurizer
    encoder = LosslessGraphEncoder(
        capN=capN,
        node_dim=RAW_NODE_DIM,
        edge_dim=EDGE_FEAT_DIM,
        scramble_seed=42  # Use a fixed seed for reproducibility
    ).to(DEVICE).eval()

    featurizer = LosslessSeedFeaturizer(
        capN=capN,
        node_dim=RAW_NODE_DIM,
        edge_dim=EDGE_FEAT_DIM,
        dx=DX,
        n_seed=N_SEED,
        scramble_seed=42  # Must be the same seed as the encoder!
    ).to(DEVICE).eval()

    # 3. Update the dataset creation loop (the featurizer now takes the dict directly)
    train_set = [(featurizer(encoder(x_t, ei_t, ea_t), noise_std=NOISE_STD), x_t, ei_t, ea_t) for (x_t, ei_t, ea_t, _) in train_set]
    val_set = [(featurizer(encoder(x_t, ei_t, ea_t), noise_std=NOISE_STD), x_t, ei_t, ea_t) for (x_t, ei_t, ea_t, _) in val_set]

    # ----- Unpooler stuff -----
    unpool_size = 256
    unpool = GuoUnpool(dx=DX, dw=DW, dy=DX, du=DW, kv=unpool_size, kia=unpool_size, kie=unpool_size, kw=unpool_size).to(DEVICE)
    critic = Critic(
        node_feature_dim=DX,  # same DX you compute for the seed features
        hidden=(512, 256),  # (256,128) also works if you want it smaller
    ).to(DEVICE)
    opt = torch.optim.AdamW(
        list(unpool.parameters()) + list(critic.parameters()),
        lr=LR,
        weight_decay=0.0
    )
    warmup = 10
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - warmup, eta_min=LR * 0.1),
        ],
        milestones=[warmup],
    )

    # ---------- batching ----------
    def iterate_minibatches(dataset, batch_size):
        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            yield [dataset[j] for j in idxs[i:i+batch_size]]

    # ---------- validation ----------
    @torch.no_grad()
    def evaluate(dataset, n_eval=64):
        unpool.eval()
        # snapshot actor RNG
        rng_state = ACTOR_RNG.get_state()
        try:
            total = 0.0
            count = 0
            avg_gen_n_size = avg_tgt_n_size = avg_gen_e_size = avg_tgt_e_size = 0
            for (x_seed, x_t, ei_t, ea_t) in random.sample(dataset, k=min(n_eval, len(dataset))):
                x_gen, ei_gen, ea_gen, _, _, _ = unpool_k_fixed(
                    unpool, x_seed, seed_ei, None, k=K_MAX, rng=ACTOR_RNG
                )

                avg_gen_n_size += x_gen.size(0); avg_tgt_n_size += x_t.size(0)
                avg_gen_e_size += ei_gen.size(1); avg_tgt_e_size += ei_t.size(1)

                score, _ = GS.graph_similarity(
                    ei_gen.cpu(), ei_t.cpu(),
                    x1=x_gen.cpu(), x2=x_t.cpu(),
                    edge_attr1=ea_gen.cpu(),
                    edge_attr2=(ea_t.cpu() if ea_t is not None else None),
                    directed=True, wl_iters=2
                )
                total += float(score); count += 1

            print(f"Avg Gen N Size: {avg_gen_n_size/max(1, count)}\tTarget: {avg_tgt_n_size/max(1, count)}")
            print(f"Avg Gen E Size: {avg_gen_e_size/max(1, count)}\tTarget: {avg_tgt_e_size/max(1, count)}")
            result = total / max(1, count)

            unpool.train()
            return result

        finally:
            ACTOR_RNG.set_state(rng_state)

    # ---------- train ----------
    print("Training…")
    t0 = time.time()

    experience_buffer = []
    losses_hist = []
    rewards_hist = []

    for epoch in range(1, EPOCHS + 1):
        # ----- Data collection -----
        unpool.eval()
        critic.eval()
        experience_buffer.clear()
        with torch.no_grad():
            # For each encoded seed graph in the batch
            for (x_seed, x_t, ei_t, ea_t) in train_set:
                # Generate graph and get the logP from the current acting policy
                x_gen, ei_gen, ea_gen, logP_old, entropy, actions_taken = unpool_k_fixed(unpool, x_seed, seed_ei, k=K_MAX, rng=ACTOR_RNG)

                # Also get the predicted_value and score
                predicted_value = critic(x_seed)
                score, _ = GS.graph_similarity(
                    ei_gen.cpu(), ei_t.cpu(),
                    x1=x_gen.cpu(), x2=x_t.cpu(),
                    edge_attr1=ea_gen.cpu(),
                    edge_attr2=(ea_t.cpu() if ea_t is not None else None),
                    directed=True, wl_iters=2
                )

                # And store them
                experience_buffer.append({
                    "x_seed": x_seed,
                    "seed_ei": seed_ei,
                    "actions": actions_taken,
                    "score": float(score),
                    "logP_old": logP_old.detach(),
                    "entropy": entropy.detach(),
                    "predicted_value": predicted_value.detach(),
                })

        # ----- Build epoch-wide tensors (fixed during PPO epochs) -----
        scores_all = torch.tensor([exp["score"] for exp in experience_buffer], device=DEVICE, dtype=torch.float32)
        values_all = torch.stack([exp["predicted_value"] for exp in experience_buffer]).detach().squeeze(-1)
        logP_old_all = torch.stack([exp["logP_old"] for exp in experience_buffer]).detach()

        advantages_all = scores_all - values_all
        adv_mean = advantages_all.mean()
        adv_std = advantages_all.std().clamp_min(1e-8)
        advantages_all = (advantages_all - adv_mean) / adv_std

        # ----- PPO Update -----
        unpool.train()
        critic.train()

        PPO_UPDATE_EPOCHS = 4
        PPO_CLIP_EPSILON = 0.2
        VALUE_LOSS_COEF = 0.5

        epoch_total_losses, epoch_policy_losses, epoch_value_losses, epoch_entropies = [], [], [], []
        for _ in range(PPO_UPDATE_EPOCHS):
            # make a fresh random permutation each epoch pass
            perm = torch.randperm(len(experience_buffer), device=DEVICE)
            for i in range(0, len(experience_buffer), BATCH_SIZE):
                mb_idx = perm[i:i + BATCH_SIZE]

                # Collate fixed epoch-wide views
                scores_tensor = scores_all[mb_idx]
                logPs_old_tensor = logP_old_all[mb_idx]
                advantages_mb = advantages_all[mb_idx]

                # --- Re-evaluate with current policy (same as before) ---
                logPs_new_list, entropies_new_list, current_values_list = [], [], []
                for j in mb_idx.tolist():
                    exp = experience_buffer[j]
                    actions_copy = copy.deepcopy(exp["actions"])
                    _, _, _, logP_new, entropy_new, _ = unpool_k_fixed(
                        unpool, exp["x_seed"], exp["seed_ei"], k=K_MAX,
                        actions_to_replay=actions_copy, rng=ACTOR_RNG
                    )
                    current_value = critic(exp["x_seed"])
                    logPs_new_list.append(logP_new)
                    entropies_new_list.append(entropy_new)
                    current_values_list.append(current_value)

                logPs_new_tensor = torch.stack(logPs_new_list)
                entropies_new_tensor = torch.stack(entropies_new_list)
                current_values_tensor = torch.stack(current_values_list).squeeze(-1)

                # --- PPO losses (unchanged except using advantages_mb) ---
                opt.zero_grad()
                ratio = torch.exp(logPs_new_tensor - logPs_old_tensor)
                surr1 = ratio * advantages_mb
                surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON) * advantages_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(current_values_tensor, scores_tensor)
                entropy_bonus = entropies_new_tensor.mean()

                loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy_bonus
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unpool.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt.step()

                epoch_total_losses.append(loss.item())
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(entropy_bonus.item())

        # Calculate final stats for the epoch
        avg_total_loss = torch.tensor(epoch_total_losses).mean().item()
        avg_policy_loss = torch.tensor(epoch_policy_losses).mean().item()
        avg_value_loss = torch.tensor(epoch_value_losses).mean().item()
        avg_entropy = torch.tensor(epoch_entropies).mean().item()
        avg_reward = torch.tensor([exp["score"] for exp in experience_buffer]).mean().item()
        avg_critic_value = torch.stack([exp["predicted_value"] for exp in experience_buffer]).mean().item()

        # Append to history for plotting
        losses_hist.append(avg_total_loss)
        rewards_hist.append(avg_reward)

        # Print detailed log every PRINT_EVERY epochs, otherwise print a concise one
        if (epoch % PRINT_EVERY) == 0 or epoch == 1:
            val_R = evaluate(val_set)
            print(f"[{epoch:04d}] loss={avg_total_loss:<7.4f} | p_loss={avg_policy_loss:<7.4f} | v_loss={avg_value_loss:<7.4f} | "
                  f"train_R={avg_reward:.3f} | val_R={val_R:.3f} | critic_V={avg_critic_value:.3f} | entropy={avg_entropy:.3f}")
        else:
            print(f"[{epoch:04d}] loss={avg_total_loss:<7.4f} | train_R={avg_reward:.3f} | critic_V={avg_critic_value:.3f}")

        sched.step()

    t1 = time.time()
    print(f"Done in {t1 - t0:.1f}s. Saving…")
    os.makedirs("artifacts", exist_ok=True)

    # ---------- save model ----------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt = {
        "unpool_state_dict": unpool.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "unpool_config": {
            "dx": DX, "dw": DW, "dy": DX, "du": DW,
            "k_max": K_MAX,
            "packed_dim": packed_dim,
            "n_seed": N_SEED,
            "seed_ei": seed_ei.detach().cpu().tolist(),  # for reproducible seeding graph
        }
    }
    ckpt_path = os.path.join("artifacts", f"unpool_last_{stamp}.pt")
    torch.save({
        "unpool_state_dict": unpool.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        # ... your config ...
    }, ckpt_path)
    with open(os.path.join("artifacts", f"unpool_last_{stamp}.json"), "w") as f:
        json.dump(ckpt["unpool_config"], f, indent=2)
    print(f"Saved unpool checkpoint to {ckpt_path}")

    # ---------- save training curves ----------
    fig1 = plt.figure()
    plt.plot(losses_hist)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Unpooling REINFORCE Loss")
    loss_path = os.path.join("artifacts", "unpool_loss.png")
    fig1.savefig(loss_path, dpi=150); plt.close(fig1)

    fig2 = plt.figure()
    plt.plot(rewards_hist)
    plt.xlabel("Epoch"); plt.ylabel("Reward (similarity)"); plt.title("Unpooling Reward")
    rew_path = os.path.join("artifacts", "unpool_reward.png")
    fig2.savefig(rew_path, dpi=150); plt.close(fig2)

    print(f"Saved: {loss_path} and {rew_path}")

    """
    ckpt = torch.load("artifacts/unpool_best.pt", map_location=DEVICE)
    cfg = ckpt["config"]
    unpool_reuse = GuoUnpool(dx=cfg["dx"], dw=cfg["dw"], dy=cfg["dy"], du=cfg["du"]).to(DEVICE)
    unpool_reuse.load_state_dict(ckpt["unpool_state_dict"])
    unpool_reuse.eval()
    """

