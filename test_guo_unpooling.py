"""
Correctness tests for the unpooling layer in guo_et_al_unpooling.py.

Run with plain Python (no pytest needed):

    python test_guo_unpooling.py

The layer follows appendix A of Guo, Zou and Lerman for undirected graphs. These tests check it
against the paper's rules, independently of how the layer is implemented, so they can serve as
the reference for any faster rewrite later:

  * calibration: every sampled step 2c choice is drawn with exactly the probability it logs;
  * the preference score matches the paper's formula computed by hand;
  * the output edge set is exactly what the paper's rules give for the recorded decisions,
    including both step 2d cases that add an edge;
  * the output graph is undirected, simple, and connected whenever the input is connected;
  * replay rebuilds the same graph and log-probability, differentiably;
  * input direction and parallel typed edges do not change anything;
  * the chess graph itself is undirected, and the standalone PPO harness still runs.
"""
import collections
import random
import sys
import traceback
from typing import Callable, Dict, List, Set, Tuple

import torch

import chess_graph as cg
import guo_et_al_unpooling as gu

CHESS_FEN = "1q1rkr2/pp3pnp/2pn2pQ/3p4/3Pb3/2P2NP1/PP2P2P/3RKRNB b KQkq - 1 15"


# ##### ##### ##### ##### #####
#   Helpers


def make_layer(dx: int = 16, dw: int = 3, use_preference: bool = True, seed: int = 0) -> gu.GuoUnpool:
    torch.manual_seed(seed)
    return gu.GuoUnpool(dx=dx, dw=dw, dy=dx, du=dw, kv=32, kia=32, kie=32, kw=32,
                        use_preference=use_preference).eval()


def random_graph(n: int, seed: int, dx: int = 16, dw: int = 3, p_extra: float = 0.3):
    """A random connected undirected graph in both-direction form, with random features."""
    g = torch.Generator().manual_seed(seed)
    _, ei, _, _ = gu.random_undirected_graph_with_features(n, p_extra=p_extra, node_feat_dim=1,
                                                           include_degree_feats=False, rng=g)
    x = torch.randn(n, dx, generator=g)
    w_half = torch.randn(ei.size(1) // 2, dw, generator=g)
    return x, ei, torch.cat([w_half, w_half])


def undirected_set(edge_index: torch.Tensor) -> Set[Tuple[int, int]]:
    return {(min(a, b), max(a, b)) for a, b in edge_index.t().tolist()}


def is_connected(num_nodes: int, edge_index: torch.Tensor) -> bool:
    adj: Dict[int, List[int]] = collections.defaultdict(list)
    for a, b in edge_index.t().tolist():
        adj[a].append(b)
        adj[b].append(a)
    seen, stack = {0}, [0]
    while stack:
        for nb in adj[stack.pop()]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == num_nodes


def paper_edge_set(pairs: List[Tuple[int, int]], rec: Dict, pm: Dict, iu_order: List[int]) -> Set[Tuple[int, int]]:
    """
    Rebuild the output edge set from the recorded decisions using only the rules in the paper's
    appendix A, written as plainly as possible. Independent of the layer's tensor code.
    """
    f, f1, f2 = pm["f"], pm["f1"], pm["f2"]
    vc = {j for j, linked in zip(iu_order, rec["step2a_intra"][0].tolist()) if linked}
    bj = rec["step2b_pick"]
    choice = rec["step2c_choice"]
    edges: Set[Tuple[int, int]] = set()

    def add(k: int, l: int) -> None:
        edges.add((min(k, l), max(k, l)))

    for j in vc:                                            # step 2a
        add(f1[j], f2[j])

    def n_set(e: int, other: int, a: int, b: int) -> List[int]:
        if e in f:
            return [f[e]]
        if e not in vc and bj.get(e) == other:              # step 2b edge: both children
            return [f1[e], f2[e]]
        c = choice[(a, b, e)]                               # step 2c
        return [f1[e]] if c == 0 else [f2[e]] if c == 1 else [f1[e], f2[e]]

    sizes = {}
    for a, b in pairs:
        na, nb = n_set(a, b, a, b), n_set(b, a, a, b)
        sizes[(a, b)] = (na, nb)
        for k in na:
            for l in nb:
                add(k, l)

    for (a, b), chosen in rec["step2d_pa"].items():         # step 2d
        if not chosen:
            continue
        na, nb = sizes[(a, b)]
        left_a = [c for c in (f1[a], f2[a]) if c not in na]
        left_b = [c for c in (f1[b], f2[b]) if c not in nb]
        if len(na) == 1 and len(nb) == 1:                   # case 1: link the leftovers
            add(left_a[0], left_b[0])
            assert (a, b) not in rec["step2d_r"]
        elif len(na) + len(nb) == 3:                        # case 2: leftover to child r of the other end
            r = rec["step2d_r"][(a, b)]
            if len(na) == 1:
                add(left_a[0], f1[b] if r == 1 else f2[b])
            else:
                add(left_b[0], f1[a] if r == 1 else f2[a])
        else:                                               # case 3: nothing
            assert (a, b) not in rec["step2d_r"]
    return edges


# ##### ##### ##### ##### #####
#   Tests


def test_chess_graph_is_undirected():
    edges, _ = cg.create_filled_chess_graphs(CHESS_FEN)
    assert len(edges) == cg.NUM_EDGE_TYPES
    for t, ei in enumerate(edges):
        d = set(map(tuple, ei.t().tolist()))
        assert len(d) == ei.size(1), f"type {t} has duplicate edges"
        assert all((b, a) in d for a, b in d), f"type {t} is not symmetric"


def test_canonicalize_merges_directions_types_and_self_loops():
    ei_one = torch.tensor([[0, 1, 2], [1, 2, 0]])
    attr_one = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
    ei_both = torch.cat([ei_one, ei_one.flip(0)], dim=1)
    attr_both = torch.cat([attr_one, attr_one])
    # a parallel edge of a second type on {0, 1}, plus a self-loop
    ei_typed = torch.cat([ei_both, torch.tensor([[1, 0, 2], [0, 1, 2]])], dim=1)
    attr_typed = torch.cat([attr_both, torch.tensor([[0., 1.], [0., 1.], [1., 1.]])])
    p1, a1 = gu.GuoUnpool.canonicalize_undirected(ei_one, attr_one, 3)
    p2, a2 = gu.GuoUnpool.canonicalize_undirected(ei_both, attr_both, 3)
    p3, a3 = gu.GuoUnpool.canonicalize_undirected(ei_typed, attr_typed, 3)
    assert p1.tolist() == p2.tolist() == p3.tolist() == [[0, 0, 1], [1, 2, 2]]
    assert torch.equal(a1, a2)
    assert a3[0].tolist() == [1., 1.], "parallel typed edges must merge into a multi-hot"


def test_input_direction_does_not_matter():
    layer = make_layer()
    x, ei, ea = random_graph(10, seed=1)
    one_dir = ei[0] < ei[1]
    with torch.no_grad():
        a = layer(x, ei, ea, rng=torch.Generator().manual_seed(5))
        b = layer(x, ei[:, one_dir], ea[one_dir], rng=torch.Generator().manual_seed(5))
    assert torch.equal(a[1], b[1]) and torch.allclose(a[0], b[0]) and torch.allclose(a[3], b[3])


def test_output_is_undirected_and_simple():
    for pref in (True, False):
        layer = make_layer(use_preference=pref)
        for s in range(10):
            x, ei, ea = random_graph(9, seed=s)
            with torch.no_grad():
                y, eo, ao, *_ = layer(x, ei, ea, rng=torch.Generator().manual_seed(s))
            d = {tuple(e): i for i, e in enumerate(eo.t().tolist())}
            assert len(d) == eo.size(1), "duplicate edges"
            assert not any(a == b for a, b in d), "self-loop"
            for (a, b), i in d.items():
                assert (b, a) in d, "missing reverse edge"
                assert torch.allclose(ao[i], ao[d[(b, a)]]), "the two directions must carry the same features"


def test_output_connected_whenever_input_connected():
    for pref in (True, False):
        layer = make_layer(use_preference=pref)
        for s in range(40):
            x, ei, ea = random_graph(random.Random(s).randint(4, 14), seed=100 + s, p_extra=0.15)
            with torch.no_grad():
                y, eo, *_ = layer(x, ei, ea, rng=torch.Generator().manual_seed(s))
            assert is_connected(y.size(0), eo), f"disconnected output, preference={pref}, seed={s}"
    # and on the chess graph, through two stacked layers
    import labyrinth_dgi_encoder as lde
    edges, x = cg.create_filled_chess_graphs(CHESS_FEN)
    xs, ei, ea = lde.chess_graph_to_single(x, edges)
    l1 = gu.GuoUnpool(dx=8, dw=cg.NUM_EDGE_TYPES, dy=8, du=4).eval()
    l2 = gu.GuoUnpool(dx=8, dw=4, dy=8, du=4).eval()
    with torch.no_grad():
        for s in range(3):
            h, e1, a1, *_ = l1(xs, ei, ea, rng=torch.Generator().manual_seed(s))
            assert is_connected(h.size(0), e1)
            h, e2, *_ = l2(h, e1, a1, rng=torch.Generator().manual_seed(s))
            assert is_connected(h.size(0), e2)


def test_output_edges_follow_the_paper_exactly():
    """Every output edge, and no other, is what appendix A's rules give for the recorded decisions."""
    case_counts = collections.Counter()
    for pref in (True, False):
        layer = make_layer(use_preference=pref)
        for s in range(60):
            n = random.Random(s).randint(5, 12)
            x, ei, ea = random_graph(n, seed=200 + s)
            pairs = sorted(undirected_set(ei))
            # unpool every node, so step 2d (which needs both ends unpooled) applies to every edge
            iu = list(range(n))
            with torch.no_grad():
                y, eo, _, _, _, pm, sets, rec = layer(x, ei, ea, I_u=torch.tensor(iu),
                                                      rng=torch.Generator().manual_seed(s))
            assert sets["Iu"].tolist() == iu
            assert undirected_set(eo) == paper_edge_set(pairs, rec, pm, iu), f"seed {s}"
            for (a, b), chosen in rec["step2d_pa"].items():
                if chosen:
                    case_counts["case 2" if (a, b) in rec["step2d_r"] else "case 1 or 3"] += 1
    assert case_counts["case 2"] > 0 and case_counts["case 1 or 3"] > 0, case_counts


def test_step2d_case1_adds_the_leftover_edge():
    """Force a situation where both ends of an edge used a single child and step 2d fires."""
    layer = make_layer()
    found = 0
    for s in range(400):
        x, ei, ea = random_graph(6, seed=300 + s)
        with torch.no_grad():
            y, eo, _, _, _, pm, sets, rec = layer(x, ei, ea, I_u=torch.arange(6),
                                                  rng=torch.Generator().manual_seed(s))
        vc = {j for j, linked in zip(range(6), rec["step2a_intra"][0].tolist()) if linked}
        out = undirected_set(eo)
        for (a, b), chosen in rec["step2d_pa"].items():
            ca, cb = rec["step2c_choice"].get((a, b, a)), rec["step2c_choice"].get((a, b, b))
            if chosen and ca in (0, 1) and cb in (0, 1):
                left_a = pm["f2"][a] if ca == 0 else pm["f1"][a]
                left_b = pm["f2"][b] if cb == 0 else pm["f1"][b]
                assert (min(left_a, left_b), max(left_a, left_b)) in out
                found += 1
    assert found > 0, "never produced a case 1 situation; the test needs more seeds"


def test_step2c_calibration():
    """
    The decisive test for the old logging bug: the empirical frequency of each step 2c choice
    must match the probability the layer logs for it. One node is unpooled and everything else is
    static, so each of its edges' choice probabilities is fixed across runs.
    """
    n_runs = 1500
    for pref in (True, False):
        layer = make_layer(dx=12, dw=2, use_preference=pref, seed=7)
        x, ei, ea = random_graph(7, seed=11, dx=12, dw=2, p_extra=0.5)
        pairs, W = gu.GuoUnpool.canonicalize_undirected(ei, ea, 7)
        e = 0
        nbr_rows = [(m, int(pairs[1, m]) if int(pairs[0, m]) == e else int(pairs[0, m]))
                    for m in range(pairs.size(1)) if e in (int(pairs[0, m]), int(pairs[1, m]))]
        assert len(nbr_rows) >= 3
        with torch.no_grad():
            ps1, ps2 = layer._project(x[e:e + 1])
            y1, y2 = layer.mlp_y(ps1), layer.mlp_y(ps2)
            K = len(nbr_rows)
            P, _ = layer.interlink_probabilities(
                y1.repeat(K, 1), y2.repeat(K, 1), W[[m for m, _ in nbr_rows]], x[[o for _, o in nbr_rows]],
                torch.zeros(K, dtype=torch.long), y1, y2, x[e:e + 1])
        expected = {o: P[k] for k, (_, o) in enumerate(nbr_rows)}

        counts = collections.defaultdict(lambda: torch.zeros(3))
        others = torch.tensor([i for i in range(7) if i != e])
        for s in range(n_runs):
            with torch.no_grad():
                *_, rec = layer(x, ei, ea, I_s=others, I_u=torch.tensor([e]), I_r=torch.tensor([], dtype=torch.long),
                                rng=torch.Generator().manual_seed(s))
            for (a, b, node), c in rec["step2c_choice"].items():
                counts[b if a == node else a][c] += 1
        for o, cnt in counts.items():
            total = cnt.sum()
            if total < 300:
                continue
            freq = cnt / total
            tol = 4 * torch.sqrt(expected[o] * (1 - expected[o]) / total) + 0.01
            assert torch.all((freq - expected[o]).abs() <= tol), (
                f"preference={pref}, neighbor {o}: sampled {freq.tolist()} vs logged {expected[o].tolist()}")


def test_preference_score_matches_the_paper_formula():
    """Supplement C.2, computed by hand with plain loops (note the paper's printed formula repeats y1 where y2 is meant)."""
    layer = make_layer(dx=10, dw=2, seed=3)
    g = torch.Generator().manual_seed(0)
    K = 5
    y1, y2 = torch.randn(1, 10, generator=g), torch.randn(1, 10, generator=g)
    w, xo, xp = torch.randn(K, 2, generator=g), torch.randn(K, 10, generator=g), torch.randn(1, 10, generator=g)
    with torch.no_grad():
        P, _ = layer.interlink_probabilities(y1.repeat(K, 1), y2.repeat(K, 1), w, xo,
                                             torch.zeros(K, dtype=torch.long), y1, y2, xp)
        hs1 = [float(layer.mlp_ie1(torch.cat([y1[0], w[k], xo[k]]).unsqueeze(0))) for k in range(K)]
        hs2 = [float(layer.mlp_ie1(torch.cat([y2[0], w[k], xo[k]]).unsqueeze(0))) for k in range(K)]
        hb = [float(layer.mlp_ie2(torch.cat([layer.agg(y1[0], y2[0]), w[k], xo[k]]).unsqueeze(0))) for k in range(K)]
        z1, z2, zb = float(layer.mlp_zero_s(y1)), float(layer.mlp_zero_s(y2)), float(layer.mlp_zero_b(xp))
    import math
    def pref(scores, zero):
        denom = sum(math.exp(v) for v in scores) + math.exp(zero)
        return [math.exp(v) / denom for v in scores]
    p1s, p2s, pbs = pref(hs1, z1), pref(hs2, z2), pref(hb, zb)
    eps = layer.p_eps
    for k in range(K):
        Z = p1s[k] + p2s[k] + pbs[k]
        want = [(v / Z) * (1 - eps) + eps / 3 for v in (p1s[k], p2s[k], pbs[k])]
        assert all(abs(a - b) < 1e-5 for a, b in zip(P[k].tolist(), want)), (P[k].tolist(), want)


def test_replay_is_exact_and_differentiable():
    for pref in (True, False):
        layer = make_layer(use_preference=pref)
        x, ei, ea = random_graph(10, seed=21)
        with torch.no_grad():
            first = layer(x, ei, ea, I_u=torch.arange(4), rng=torch.Generator().manual_seed(3))
        replay = layer(x, ei, ea, I_u=torch.arange(4), actions_to_replay=first[7])
        assert torch.equal(first[1], replay[1]) and torch.allclose(first[0], replay[0])
        assert torch.allclose(first[3], replay[3]) and torch.allclose(first[4], replay[4])
        layer.zero_grad()
        replay[3].backward()
        heads = ["mlp_r", "mlp_ia", "mlp_ie1", "mlp_ie2", "mlp_ie_a"] + (["mlp_zero_s", "mlp_zero_b"] if pref else [])
        for head in heads:
            grads = [p.grad for p in getattr(layer, head).parameters()]
            assert any(g is not None and g.abs().sum() > 0 for g in grads), f"no gradient reached {head}"


def test_replay_ignores_the_rng():
    layer = make_layer()
    x, ei, ea = random_graph(8, seed=4)
    with torch.no_grad():
        first = layer(x, ei, ea, rng=torch.Generator().manual_seed(1))
        again = layer(x, ei, ea, actions_to_replay=first[7], rng=torch.Generator().manual_seed(999))
    assert torch.equal(first[1], again[1]) and torch.allclose(first[3], again[3])


def test_harness_runs_end_to_end():
    """The standalone PPO harness (undirected targets, both scoring rules) still trains and evaluates."""
    ctx = gu.build_training_context(n_train=6, n_val=4, data_seed=5)
    for ei_t in (s[2] for s in ctx["train_set"]):
        d = set(map(tuple, ei_t.t().tolist()))
        assert all((b, a) in d for a, b in d)
    assert "inter_link_scoring" in list(gu.build_configspace())
    for scoring in ("preference", "plain"):
        cfg = {"lr": 3e-4, "entropy_coef": 0.01, "unpool_size": 64, "batch_size": 4,
               "ppo_update_epochs": 2, "ppo_clip_eps": 0.2, "inter_link_scoring": scoring}
        out = gu.train_and_eval(cfg, 2, ctx, k_unpool=2, seed=1, eval_n=4)
        assert out["val_R"] == out["val_R"], "validation similarity is NaN"


# ##### ##### ##### ##### #####
#   Runner

def main() -> int:
    tests: List[Callable[[], None]] = [
        obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
