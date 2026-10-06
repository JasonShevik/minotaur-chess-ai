"""
Correctness tests for the batched unpooling layer in guo_et_al_unpooling_v2.py.

Run with plain Python (no pytest needed):

    python test_guo_unpooling_v2.py

v1 (guo_et_al_unpooling_v1.py) is the reference. These tests check that v2 is the same layer:

  * cross-replay: decisions sampled by v2, replayed through v1 with the same weights, rebuild the
    identical graph with the same log-probability and entropy; and the other way round;
  * a batch of graphs in one call gives exactly what each graph gives on its own;
  * sampling is calibrated, including step 2b, which v2 samples with the Gumbel-max trick;
  * the paper's rules, connectivity and output conventions hold, as in v1's tests;
  * action records can be split, selected and collated, and replay detects a different graph;
  * v1 checkpoints load, the encoder's summarizer works with v2 swapped in, and the harness runs.
"""
import collections
import random
import sys
import traceback
from typing import Callable, List

import torch

import chess_graph as cg
import guo_et_al_unpooling_v1 as gu1
import guo_et_al_unpooling_v2 as gu2
import test_guo_unpooling_v1 as t1

CHESS_FEN = t1.CHESS_FEN


# ##### ##### ##### ##### #####
#   Helpers


def make_pair(dx: int = 16, dw: int = 3, use_preference: bool = True, seed: int = 0):
    """A v1 layer and a v2 layer with identical weights."""
    v1 = t1.make_layer(dx=dx, dw=dw, use_preference=use_preference, seed=seed)
    v2 = gu2.GuoUnpool(dx=dx, dw=dw, dy=dx, du=dw, kv=32, kia=32, kie=32, kw=32,
                       use_preference=use_preference).eval()
    v2.load_state_dict(v1.state_dict())
    return v1, v2


def assert_same(a, b, what: str) -> None:
    """Same output graph, features, log-probability and entropy (the first 5 values of either version)."""
    assert torch.equal(a[1], b[1]), f"{what}: different edges"
    assert torch.allclose(a[0], b[0], atol=1e-5), f"{what}: different node features"
    assert torch.allclose(a[2], b[2], atol=1e-5), f"{what}: different edge features"
    assert torch.allclose(a[3], b[3], atol=1e-4), f"{what}: logP {float(a[3])} vs {float(b[3])}"
    assert torch.allclose(a[4], b[4], atol=1e-4), f"{what}: entropy {float(a[4])} vs {float(b[4])}"


def chess_input():
    import labyrinth_dgi_encoder as lde
    edges, x = cg.create_filled_chess_graphs(CHESS_FEN)
    return lde.chess_graph_to_single(x, edges)


def graph_slice(out, g: int):
    """One graph's (y, local edge_index, edge_attr) from a batched UnpoolOutput."""
    nodes = (out.batch == g).nonzero().flatten()
    off = int(nodes[0]) if nodes.numel() else 0
    keep = out.batch[out.edge_index[0]] == g
    return out.y[nodes], out.edge_index[:, keep] - off, out.edge_attr[keep]


# ##### ##### ##### ##### #####
#   Tests


def test_v1_checkpoints_load():
    for pref in (True, False):
        v1, v2 = make_pair(use_preference=pref)
        assert set(v1.state_dict()) == set(v2.state_dict())
        v2.load_state_dict(v1.state_dict(), strict=True)


def test_single_graph_unpacks_like_v1():
    _, v2 = make_pair()
    x, ei, ea = t1.random_graph(9, seed=2)
    with torch.no_grad():
        out = v2(x, ei, ea, rng=torch.Generator().manual_seed(0))
    y, eo, ao, logP, ent, pm, sets, rec = out
    assert logP.dim() == 0 and ent.dim() == 0, "one graph must give scalars, as in v1"
    assert out.y is y and out.log_prob is logP and out.actions is rec
    assert isinstance(rec, gu2.UnpoolActions) and rec.num_graphs == 1
    assert out.batch.tolist() == [0] * y.size(0)
    *_, rec_k = gu2.unpool_k_fixed(v2, x, ei, ea, k=2, rng=torch.Generator().manual_seed(0))
    assert len(rec_k) == 2


def test_v2_decisions_replay_identically_in_v1():
    for pref in (True, False):
        v1, v2 = make_pair(use_preference=pref)
        for s in range(25):
            x, ei, ea = t1.random_graph(random.Random(s).randint(4, 13), seed=500 + s)
            kwargs = {"I_u": torch.arange(x.size(0))} if s % 3 == 0 else {}   # all unpooled: every 2d case
            with torch.no_grad():
                a = v2(x, ei, ea, rng=torch.Generator().manual_seed(s), record_v1=True, **kwargs)
                b = v1(x, ei, ea, actions_to_replay=a.actions_v1, **kwargs)
            assert_same(a, b, f"preference={pref} seed={s}")
            assert gu2.parent_map_as_dicts(a[5]) == b[5]
            assert torch.equal(a.sets["Iu"], b[6]["Iu"]) and torch.equal(a.sets["Vc"], b[6]["Vc"])


def test_v1_decisions_replay_identically_in_v2():
    for pref in (True, False):
        v1, v2 = make_pair(use_preference=pref)
        for s in range(25):
            x, ei, ea = t1.random_graph(random.Random(s).randint(4, 13), seed=700 + s)
            kwargs = {"I_u": torch.arange(x.size(0))} if s % 3 == 0 else {}
            with torch.no_grad():
                a = v1(x, ei, ea, rng=torch.Generator().manual_seed(s), **kwargs)
                b = v2(x, ei, ea, actions_to_replay=a[7], **kwargs)          # v1 dict replayed directly
                assert_same(a, b, f"dict replay, preference={pref} seed={s}")
                rec = gu2.actions_from_v1(v2, x, ei, ea, a[7], **kwargs)     # converted, then replayed
                c = v2(x, ei, ea, actions_to_replay=rec, **kwargs)
                assert_same(a, c, f"converted replay, preference={pref} seed={s}")
                back = gu2.actions_to_v1(v2, x, ei, ea, rec, **kwargs)       # and converted back
            for key in ("step2b_pick", "step2c_choice", "step2d_pa", "step2d_r", "__sig_out__"):
                assert back[key] == a[7][key], key


def test_cross_replay_on_the_chess_graph_through_two_layers():
    xs, ei, ea = chess_input()
    v1a = gu1.GuoUnpool(dx=8, dw=cg.NUM_EDGE_TYPES, dy=8, du=4).eval()
    v1b = gu1.GuoUnpool(dx=8, dw=4, dy=8, du=4).eval()
    v2a = gu2.GuoUnpool(dx=8, dw=cg.NUM_EDGE_TYPES, dy=8, du=4).eval(); v2a.load_state_dict(v1a.state_dict())
    v2b = gu2.GuoUnpool(dx=8, dw=4, dy=8, du=4).eval(); v2b.load_state_dict(v1b.state_dict())
    with torch.no_grad():
        for s in range(3):
            a1 = v2a(xs, ei, ea, rng=torch.Generator().manual_seed(s), record_v1=True)
            a2 = v2b(a1[0], a1[1], a1[2], rng=torch.Generator().manual_seed(s), record_v1=True)
            b1 = v1a(xs, ei, ea, actions_to_replay=a1.actions_v1)
            b2 = v1b(b1[0], b1[1], b1[2], actions_to_replay=a2.actions_v1)
            assert_same(a1, b1, f"layer 1 seed {s}")
            assert_same(a2, b2, f"layer 2 seed {s}")


def test_batch_equals_separate_calls():
    for pref in (True, False):
        _, v2 = make_pair(use_preference=pref, seed=4)
        graphs = [t1.random_graph(n, seed=900 + n) for n in (5, 11, 3, 8, 13)]
        x, ei, ea, batch = gu2.pack_graphs([g[0] for g in graphs], [g[1] for g in graphs], [g[2] for g in graphs])
        G = len(graphs)
        with torch.no_grad():
            # sampled as a batch, replayed one graph at a time
            out = v2(x, ei, ea, batch=batch, rng=torch.Generator().manual_seed(1))
            assert out.log_prob.shape == (G,) and out.entropy.shape == (G,)
            assert torch.all(out.batch[1:] >= out.batch[:-1]), "output nodes must stay grouped by graph"
            assert torch.equal(out.batch[out.edge_index[0]], out.batch[out.edge_index[1]]), "edge between graphs"
            for g, (rec_g, (xg, eig, eag)) in enumerate(zip(out.actions.split(), graphs)):
                single = v2(xg, eig, eag, actions_to_replay=rec_g)
                y_g, e_g, a_g = graph_slice(out, g)
                assert torch.equal(e_g, single[1]) and torch.allclose(y_g, single[0], atol=1e-5)
                assert torch.allclose(a_g, single[2], atol=1e-5)
                assert torch.allclose(out.log_prob[g], single[3], atol=1e-4)
                assert torch.allclose(out.entropy[g], single[4], atol=1e-4)
            # sampled one graph at a time, collated, replayed as a batch
            singles = [v2(*g, rng=torch.Generator().manual_seed(10 + i)) for i, g in enumerate(graphs)]
            again = v2(x, ei, ea, batch=batch, actions_to_replay=gu2.UnpoolActions.collate([s[7] for s in singles]))
            for g, single in enumerate(singles):
                y_g, e_g, _ = graph_slice(again, g)
                assert torch.equal(e_g, single[1]) and torch.allclose(y_g, single[0], atol=1e-5)
                assert torch.allclose(again.log_prob[g], single[3], atol=1e-4)


def test_batch_with_forced_nodes_equals_separate_calls():
    """I_s / I_u / I_r given in global indices, out of graph order, still match per-graph calls with local indices."""
    _, v2 = make_pair(seed=5)
    graphs = [t1.random_graph(n, seed=950 + n) for n in (6, 9, 7)]
    x, ei, ea, batch = gu2.pack_graphs([g[0] for g in graphs], [g[1] for g in graphs], [g[2] for g in graphs])
    offsets = [0, 6, 15]
    forced_u = {2: [3, 0], 0: [5]}          # graph -> local nodes forced unpooled (listed out of order)
    forced_s = {1: [2], 2: [1]}
    I_u = torch.tensor([offsets[g] + v for g in (2, 0) for v in forced_u[g]])
    I_s = torch.tensor([offsets[g] + v for g in (2, 1) for v in forced_s[g]])
    with torch.no_grad():
        out = v2(x, ei, ea, batch=batch, I_u=I_u, I_s=I_s, rng=torch.Generator().manual_seed(4))
        for g, (rec_g, (xg, eig, eag)) in enumerate(zip(out.actions.split(), graphs)):
            single = v2(xg, eig, eag, I_u=torch.tensor(forced_u.get(g, []), dtype=torch.long),
                        I_s=torch.tensor(forced_s.get(g, []), dtype=torch.long), actions_to_replay=rec_g)
            y_g, e_g, _ = graph_slice(out, g)
            assert torch.equal(e_g, single[1]) and torch.allclose(y_g, single[0], atol=1e-5), f"graph {g}"
            assert torch.allclose(out.log_prob[g], single[3], atol=1e-4)


def test_minibatch_replay_of_any_graphs_through_two_layers():
    """What PPO does: roll out a batch, then replay an arbitrary subset, in any order, in one call."""
    _, v2 = make_pair(seed=6)
    seeds = torch.randn(7, 3, 16)
    seed_ei = gu2.seed_graph()
    with torch.no_grad():
        ro = gu2.batched_rollout(v2, seeds, seed_ei, k=2, rng=torch.Generator().manual_seed(2))
        pick = torch.tensor([5, 0, 3])
        re = gu2.batched_rollout(v2, seeds[pick], seed_ei, k=2, actions_to_replay=[a.select(pick) for a in ro.actions])
    assert torch.allclose(re.log_prob, ro.log_prob[pick], atol=1e-4)
    assert torch.allclose(re.entropy, ro.entropy[pick], atol=1e-4)
    full = gu2.unbatch_graphs(ro.x, ro.edge_index, ro.edge_attr, ro.batch, 7)
    part = gu2.unbatch_graphs(re.x, re.edge_index, re.edge_attr, re.batch, 3)
    for i, g in enumerate(pick.tolist()):
        assert torch.equal(part[i][1], full[g][1]) and torch.allclose(part[i][0], full[g][0], atol=1e-5)


def test_action_records_split_select_collate():
    _, v2 = make_pair()
    graphs = [t1.random_graph(n, seed=40 + n) for n in (6, 9, 4, 12)]
    x, ei, ea, batch = gu2.pack_graphs([g[0] for g in graphs], [g[1] for g in graphs], [g[2] for g in graphs])
    with torch.no_grad():
        rec = v2(x, ei, ea, batch=batch, rng=torch.Generator().manual_seed(3)).actions

    def same(a, b):
        return all(torch.equal(getattr(a, f), getattr(b, f)) for f in gu2.ACTION_FIELDS + ("counts", "signature"))

    parts = rec.split()
    assert len(parts) == 4 and same(gu2.UnpoolActions.collate(parts), rec)
    order = [2, 0, 3, 1]
    assert same(rec.select(order), gu2.UnpoolActions.collate([parts[i] for i in order]))
    assert same(rec.select([1]), parts[1])


def test_replay_detects_a_different_graph():
    _, v2 = make_pair()
    x, ei, ea = t1.random_graph(10, seed=8)
    x2, ei2, ea2 = t1.random_graph(10, seed=9)
    with torch.no_grad():
        rec = v2(x, ei, ea, I_u=torch.arange(10), rng=torch.Generator().manual_seed(0)).actions
        try:
            v2(x2, ei2, ea2, I_u=torch.arange(10), actions_to_replay=rec)
        except RuntimeError:
            pass
        else:
            raise AssertionError("replaying on another graph must fail")
        v2(x, ei, ea, I_u=torch.arange(10), actions_to_replay=rec)        # and on the right one it works


def test_paper_rules_hold():
    """v1's independent oracle: every output edge, and no other, follows appendix A for the recorded decisions."""
    case_counts = collections.Counter()
    for pref in (True, False):
        _, v2 = make_pair(use_preference=pref)
        for s in range(40):
            n = random.Random(s).randint(5, 12)
            x, ei, ea = t1.random_graph(n, seed=200 + s)
            pairs = sorted(t1.undirected_set(ei))
            iu = list(range(n))
            with torch.no_grad():
                out = v2(x, ei, ea, I_u=torch.tensor(iu), rng=torch.Generator().manual_seed(s), record_v1=True)
            rec, pm = out.actions_v1, gu2.parent_map_as_dicts(out.parent_map)
            assert out.sets["Iu"].tolist() == iu
            assert t1.undirected_set(out.edge_index) == t1.paper_edge_set(pairs, rec, pm, iu), f"seed {s}"
            for (a, b), chosen in rec["step2d_pa"].items():
                if chosen:
                    case_counts["case 2" if (a, b) in rec["step2d_r"] else "case 1 or 3"] += 1
    assert case_counts["case 2"] > 0 and case_counts["case 1 or 3"] > 0, case_counts


def test_output_undirected_simple_and_connected_in_batches():
    for pref in (True, False):
        _, v2 = make_pair(use_preference=pref)
        graphs = [t1.random_graph(random.Random(s).randint(4, 14), seed=100 + s, p_extra=0.15) for s in range(20)]
        x, ei, ea, batch = gu2.pack_graphs([g[0] for g in graphs], [g[1] for g in graphs], [g[2] for g in graphs])
        with torch.no_grad():
            out = v2(x, ei, ea, batch=batch, rng=torch.Generator().manual_seed(int(pref)))
        for g, (y_g, e_g, a_g) in enumerate(gu2.unbatch_graphs(out.y, out.edge_index, out.edge_attr, out.batch, 20)):
            d = {tuple(e): i for i, e in enumerate(e_g.t().tolist())}
            assert len(d) == e_g.size(1) and not any(a == b for a, b in d)
            for (a, b), i in d.items():
                # equal up to float rounding: a row's matmul result can depend on where it sits in the batch
                assert (b, a) in d and torch.allclose(a_g[i], a_g[d[(b, a)]], atol=1e-6)
            assert t1.is_connected(y_g.size(0), e_g), f"graph {g} disconnected, preference={pref}"


def test_step2b_gumbel_sampling_is_calibrated():
    """v2 samples step 2b with Gumbel-max: the picked neighbor must follow the logged probabilities."""
    n_runs = 1500
    _, v2 = make_pair(dx=12, dw=2, seed=7)
    x, ei, ea = t1.random_graph(7, seed=11, dx=12, dw=2, p_extra=0.5)
    e = 0
    others = torch.tensor([i for i in range(7) if i != e])
    kwargs = dict(I_s=others, I_u=torch.tensor([e]), I_r=torch.tensor([], dtype=torch.long))
    counts, expected = collections.Counter(), None
    with torch.no_grad():
        # Graphs where node e has no intra-link: its neighbors' probabilities are fixed across runs
        pairs, W = v2.canonicalize_undirected(ei, ea, 7)
        nbrs = sorted(int(b) if int(a) == e else int(a) for a, b in pairs.t().tolist() if e in (a, b))
        ps1, ps2 = v2._project(x[e:e + 1])
        y1, y2 = v2.mlp_y(ps1), v2.mlp_y(ps2)
        rows = [m for m in range(pairs.size(1)) if e in pairs[:, m].tolist()]
        rows.sort(key=lambda m: int(pairs[1, m]) if int(pairs[0, m]) == e else int(pairs[0, m]))
        K = len(rows)
        _, h_c = v2.interlink_probabilities(y1.repeat(K, 1), y2.repeat(K, 1), W[rows], x[nbrs],
                                            torch.zeros(K, dtype=torch.long), y1, y2, x[e:e + 1])
        expected = v2._smooth_cat(torch.softmax(h_c, 0), 0)
        n_b = 0
        for s in range(n_runs):
            out = v2(x, ei, ea, rng=torch.Generator().manual_seed(s), **kwargs)
            if out.actions.pick.numel():
                counts[int(out.actions.pick[0])] += 1
                n_b += 1
    assert n_b > 300, n_b
    for k in range(K):
        freq = counts[k] / n_b
        tol = 4 * (float(expected[k]) * (1 - float(expected[k])) / n_b) ** 0.5 + 0.01
        assert abs(freq - float(expected[k])) <= tol, f"neighbor {k}: sampled {freq:.3f} vs logged {float(expected[k]):.3f}"


def test_step2c_sampling_is_calibrated():
    n_runs = 1200
    for pref in (True, False):
        _, v2 = make_pair(dx=12, dw=2, use_preference=pref, seed=7)
        x, ei, ea = t1.random_graph(7, seed=11, dx=12, dw=2, p_extra=0.5)
        pairs, W = v2.canonicalize_undirected(ei, ea, 7)
        e = 0
        nbr_rows = [(m, int(pairs[1, m]) if int(pairs[0, m]) == e else int(pairs[0, m]))
                    for m in range(pairs.size(1)) if e in (int(pairs[0, m]), int(pairs[1, m]))]
        with torch.no_grad():
            ps1, ps2 = v2._project(x[e:e + 1])
            y1, y2 = v2.mlp_y(ps1), v2.mlp_y(ps2)
            K = len(nbr_rows)
            P, _ = v2.interlink_probabilities(
                y1.repeat(K, 1), y2.repeat(K, 1), W[[m for m, _ in nbr_rows]], x[[o for _, o in nbr_rows]],
                torch.zeros(K, dtype=torch.long), y1, y2, x[e:e + 1])
        expected = {o: P[k] for k, (_, o) in enumerate(nbr_rows)}
        counts = collections.defaultdict(lambda: torch.zeros(3))
        others = torch.tensor([i for i in range(7) if i != e])
        for s in range(n_runs):
            with torch.no_grad():
                out = v2(x, ei, ea, I_s=others, I_u=torch.tensor([e]), I_r=torch.tensor([], dtype=torch.long),
                         rng=torch.Generator().manual_seed(s), record_v1=True)
            for (a, b, node), c in out.actions_v1["step2c_choice"].items():
                counts[b if a == node else a][c] += 1
        for o, cnt in counts.items():
            total = cnt.sum()
            if total < 300:
                continue
            freq = cnt / total
            tol = 4 * torch.sqrt(expected[o] * (1 - expected[o]) / total) + 0.01
            assert torch.all((freq - expected[o]).abs() <= tol), (
                f"preference={pref}, neighbor {o}: sampled {freq.tolist()} vs logged {expected[o].tolist()}")


def test_batched_replay_is_differentiable():
    for pref in (True, False):
        _, v2 = make_pair(use_preference=pref)
        graphs = [t1.random_graph(n, seed=60 + n) for n in (7, 10, 5)]
        x, ei, ea, batch = gu2.pack_graphs([g[0] for g in graphs], [g[1] for g in graphs], [g[2] for g in graphs])
        with torch.no_grad():
            first = v2(x, ei, ea, batch=batch, I_u=torch.tensor([0, 1, 7, 8, 17]), rng=torch.Generator().manual_seed(3))
        replay = v2(x, ei, ea, batch=batch, I_u=torch.tensor([0, 1, 7, 8, 17]), actions_to_replay=first.actions)
        assert torch.allclose(first.log_prob, replay.log_prob, atol=1e-5)
        v2.zero_grad()
        replay.log_prob.sum().backward()
        heads = ["mlp_r", "mlp_ia", "mlp_ie1", "mlp_ie2", "mlp_ie_a"] + (["mlp_zero_s", "mlp_zero_b"] if pref else [])
        for head in heads:
            grads = [p.grad for p in getattr(v2, head).parameters()]
            assert any(g is not None and g.abs().sum() > 0 for g in grads), f"no gradient reached {head}"


def test_summarizer_works_with_v2_swapped_in():
    """The encoder's GlobalSummarizer, built on v2 instead of v1, runs and replays unchanged."""
    import labyrinth_dgi_encoder as lde
    old = lde.unpool
    try:
        lde.unpool = gu2
        cfg = lde.DGIConfig(sum_in_dim=16, sum_unpool_hidden=32, sum_edge_dim=8, sum_flat_dim=4,
                            sum_mlp_hidden=64, summary_dim=32)
        summarizer = lde.GlobalSummarizer(cfg).eval()
        assert isinstance(summarizer.unpool_layers[0], gu2.GuoUnpool)
        edges, x = cg.create_filled_chess_graphs(CHESS_FEN)
        with torch.no_grad():
            first = summarizer(x, edges, rng=torch.Generator().manual_seed(0))
        again = summarizer(x, edges, actions_to_replay=first.actions)
        assert torch.allclose(first.summary, again.summary, atol=1e-5)
        assert torch.allclose(first.log_prob, again.log_prob, atol=1e-4)
        again.log_prob.backward()
    finally:
        lde.unpool = old


def test_runs_on_gpu_when_available():
    if not torch.cuda.is_available():
        print("        (no GPU; skipped)")
        return
    dev = torch.device("cuda")
    _, v2 = make_pair(seed=2)
    v2 = v2.to(dev)
    seeds = torch.randn(32, 3, 16, device=dev)
    seed_ei = gu2.seed_graph().to(dev)
    with torch.no_grad():
        ro = gu2.batched_rollout(v2, seeds, seed_ei, k=2, rng=torch.Generator(device=dev).manual_seed(0))
    re = gu2.batched_rollout(v2, seeds, seed_ei, k=2, actions_to_replay=ro.actions)
    assert torch.allclose(re.log_prob, ro.log_prob, atol=1e-4)
    re.log_prob.sum().backward()
    # and the same decisions give the same result on the CPU
    v2_cpu = gu2.GuoUnpool(dx=16, dw=3, dy=16, du=3, kv=32, kia=32, kie=32, kw=32).eval()
    v2_cpu.load_state_dict({k: v.cpu() for k, v in v2.state_dict().items()})
    with torch.no_grad():
        cpu = gu2.batched_rollout(v2_cpu, seeds.cpu(), gu2.seed_graph(), k=2,
                                  actions_to_replay=[a.to("cpu") for a in ro.actions])
    assert torch.allclose(cpu.log_prob, ro.log_prob.cpu(), atol=1e-3)


def test_harness_runs_end_to_end():
    """The batched PPO harness trains and evaluates with both scoring rules."""
    ctx = gu2.build_training_context(n_train=6, n_val=4, data_seed=5)
    assert ctx["train_seeds"].shape[0] == 6
    assert "inter_link_scoring" in list(gu2.build_configspace())
    for scoring in ("preference", "plain"):
        cfg = {"lr": 3e-4, "entropy_coef": 0.01, "unpool_size": 64, "batch_size": 4,
               "ppo_update_epochs": 2, "ppo_clip_eps": 0.2, "inter_link_scoring": scoring}
        out = gu2.train_and_eval(cfg, 2, ctx, k_unpool=2, seed=1, eval_n=4, rollout_chunk=4, log_every=1)
        assert out["val_R"] == out["val_R"], "validation similarity is NaN"
        assert all(l == l for l in out["losses_hist"]), "a PPO epoch produced no finite loss"


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
