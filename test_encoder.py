"""
Tests for the three Labyrinth models in labyrinth_dgi_encoder.py.

Run with plain Python (no pytest needed):

    python test_encoder.py

These check shapes and contracts rather than learning: that every model builds from a config,
that the summarizer can replay its own unpooling decisions exactly, that stacking positions
into one disconnected graph encodes them identically to one at a time, that the discriminator's
paired and matrix scoring agree, and that configurations sampled from the DEHB search space
build and run.
"""
import random
import sys
import traceback
from typing import Callable, List

import torch

import chess_graph as cg
import labyrinth_dgi_encoder as lde

FENS: List[str] = [
    "1q1rkr2/pp3pnp/2pn2pQ/3p4/3Pb3/2P2NP1/PP2P2P/3RKRNB b KQkq - 1 15",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "2k5/ppp3pp/6N1/8/8/2b1P1P1/2NrK1BP/6B1 w - - 1 26",
    "8/7p/kR2p1p1/5p2/N3RPP1/1P6/1K5P/8 b - - 1 37",
]
GRAPHS = [cg.create_filled_chess_graphs(f) for f in FENS]


def test_graph_builder_matches_config():
    edges, x = GRAPHS[0]
    assert len(edges) == lde.DGIConfig().num_edge_types == cg.NUM_EDGE_TYPES
    assert x.shape == (64, lde.DGIConfig().node_feature_dim)


def test_default_models_shapes():
    cfg = lde.DGIConfig()
    enc, summ, disc = lde.build_models(cfg)
    edges, x = GRAPHS[0]
    h = enc(x, edges)
    out = summ(x, edges, rng=torch.Generator().manual_seed(0))
    assert h.shape == (64, cfg.node_dim)
    assert out.summary.shape == (cfg.summary_dim,)
    assert 64 <= out.num_nodes <= cfg.max_unpooled_nodes
    assert out.log_prob.dim() == 0 and out.entropy.dim() == 0
    assert len(out.actions) == cfg.sum_num_unpool
    assert disc(h, out.summary.unsqueeze(0).expand(64, -1)).shape == (64,)
    assert disc.score_matrix(h, out.summary.unsqueeze(0)).shape == (64, 1)


def test_summarizer_replay_is_exact():
    cfg = lde.DGIConfig(sum_num_unpool=3)
    summ = lde.GlobalSummarizer(cfg).eval()
    edges, x = GRAPHS[1]
    with torch.no_grad():
        first = summ(x, edges, rng=torch.Generator().manual_seed(7))
        replay = summ(x, edges, actions_to_replay=first.actions)
        other = summ(x, edges, rng=torch.Generator().manual_seed(8))
    assert replay.num_nodes == first.num_nodes
    assert torch.allclose(replay.summary, first.summary)
    assert torch.allclose(replay.log_prob, first.log_prob)
    # a different seed should (almost always) produce a different enlarged graph
    assert other.num_nodes != first.num_nodes or not torch.allclose(other.summary, first.summary)


def test_summarizer_gradients_reach_every_stage():
    cfg = lde.DGIConfig(sum_num_unpool=1, summary_dim=256, sum_mlp_hidden=128)
    summ = lde.GlobalSummarizer(cfg)
    edges, x = GRAPHS[2]
    out = summ(x, edges, rng=torch.Generator().manual_seed(0))
    (out.summary.sum() + out.log_prob).backward()
    for name, p in summ.named_parameters():
        if p.grad is None:
            # the unpooling layer's preference heads are only used on some graphs; everything else must train
            assert "mlp_zero" in name or "mlp_ie_a" in name, f"no gradient reached {name}"


def test_batching_matches_single_encoding():
    cfg = lde.DGIConfig()
    enc = lde.NodeEncoder(cfg).eval()
    with torch.no_grad():
        single = torch.cat([enc(x, e) for e, x in GRAPHS])
        x, ei, ea, batch = lde.batch_chess_graphs(GRAPHS)
        stacked = enc.forward_merged(x, ei, ea)
    assert batch.tolist() == sorted(batch.tolist())
    assert batch.bincount().tolist() == [64] * len(GRAPHS)
    assert torch.allclose(single, stacked, atol=1e-5)


def test_discriminator_paired_equals_matrix_diagonal():
    cfg = lde.DGIConfig(node_dim=64, summary_dim=96)
    disc = lde.Discriminator(cfg).eval()
    h = torch.randn(6, cfg.node_dim)
    s = torch.randn(6, cfg.summary_dim)
    with torch.no_grad():
        assert torch.allclose(disc(h, s), disc.score_matrix(h, s).diagonal(), atol=1e-5)
        assert disc.score_matrix(h[:4], s).shape == (4, 6)


def test_config_from_dict_and_validation():
    d = lde.DGIConfig().to_dict()
    d["lr"] = 1e-3               # a training knob the config does not own
    d["sum_num_unpool"] = 4
    cfg = lde.DGIConfig.from_dict(d)
    assert cfg.sum_num_unpool == 4 and cfg.max_unpooled_nodes == 64 * 16
    assert cfg.flattened_size == cfg.max_unpooled_nodes * cfg.sum_flat_dim
    try:
        lde.DGIConfig(sum_in_dim=2).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("sum_in_dim below 4 must be rejected")


def test_search_space_samples_build_and_run():
    cs = lde.build_architecture_configspace(seed=3)
    edges, x = GRAPHS[3]
    rng = random.Random(0)
    tried = 0
    while tried < 6:
        sample = dict(cs.sample_configuration())
        cfg = lde.DGIConfig.from_dict(sample)
        if cfg.sum_num_unpool > 3:      # larger stacks are exercised by the smoke run, not here (they are slow)
            continue
        tried += 1
        enc, summ, disc = lde.build_models(cfg)
        h = enc(x, edges)
        out = summ(x, edges, rng=torch.Generator().manual_seed(rng.randint(0, 999)))
        sc = disc.score_matrix(h, out.summary.unsqueeze(0))
        assert h.shape == (64, cfg.node_dim) and out.summary.shape == (cfg.summary_dim,) and sc.shape == (64, 1)
        assert out.num_nodes <= cfg.max_unpooled_nodes
    # the two extremes of the space must at least build
    for extreme in (dict(sum_in_dim=16, sum_num_unpool=5, sum_flat_dim=32), dict(sum_in_dim=128, sum_num_unpool=1, sum_flat_dim=4)):
        lde.build_models(lde.DGIConfig.from_dict(extreme))


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
