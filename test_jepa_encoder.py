"""
Tests for the JEPA encoder in labyrinth_jepa_encoder.py.

Run with plain Python (no pytest needed):

    python test_jepa_encoder.py

They check contracts, not learning:
  * all three token variants build from a config, run, and give the documented shapes;
  * the encoder and the masked-square predictor never see what stands on hidden squares;
  * a position encodes the same whether it is alone or in a batch of other positions;
  * no gradient reaches the target encoder, which moves only by the moving average;
  * the losses reach every trainable part, and the unpooling policy only when unpool_rl is on;
  * LiDAR, RankMe and the self-consistency ratio behave as defined on constructed embeddings;
  * the DEHB search space samples valid configurations.
"""
import math
import random
import sys
import traceback
from typing import Callable, List

import chess
import torch

import labyrinth_data as ld
import labyrinth_jepa_encoder as lje

FENS = [
    "1q1rkr2/pp3pnp/2pn2pQ/3p4/3Pb3/2P2NP1/PP2P2P/3RKRNB b - - 1 15",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "2k5/ppp3pp/6N1/8/8/2b1P1P1/2NrK1BP/6B1 w - - 1 26",
    "8/7p/kR2p1p1/5p2/N3RPP1/1P6/1K5P/8 b - - 1 37",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
]
FENS = [ld.normalize_fen(f) for f in FENS]


def small_config(variant: str = "unpool", **kw) -> lje.JEPAConfig:
    base = dict(token_variant=variant, gat_dim=32, gat_heads=2, gat_layers=2, num_registers=8, unpool_layers=1,
                unpool_hidden=32, unpool_edge_dim=8, d_model=32, transformer_layers=2, transformer_heads=4,
                readout_queries=4, z_dim=64, pred_dim=32, pred_gat_layers=2, pred_heads=2, move_dim=32,
                next_hidden=64, next_layers=1)
    base.update(kw)
    return lje.JEPAConfig(**base)


def make_batch(config: lje.JEPAConfig, fens=FENS, seed: int = 0) -> ld.JEPABatch:
    rnd = random.Random(seed)
    return ld.collate_jepa([ld.build_sample(f, rnd, config.mask_config()) for f in fens])


# ##### ##### ##### ##### #####
#   Tests


def test_every_variant_builds_and_has_the_documented_shapes():
    for variant in lje.TOKEN_VARIANTS:
        cfg = small_config(variant)
        model = lje.LabyrinthJEPA(cfg)
        batch = make_batch(cfg)
        out = model.encoder(batch.full, rng=torch.Generator().manual_seed(0))
        B = len(FENS)
        assert out.z.shape == (B, cfg.z_dim) and out.squares.shape == (B, 64, cfg.d_model)
        assert out.log_prob.shape == out.entropy.shape == out.num_tokens.shape == (B,)
        if variant == "squares":
            assert out.num_tokens.tolist() == [64] * B and float(out.log_prob.abs().sum()) == 0
        elif variant == "registers":
            assert out.num_tokens.tolist() == [64 + cfg.num_registers] * B
        else:
            assert torch.all(out.num_tokens > 64) and torch.all(out.num_tokens <= cfg.max_tokens)
            assert torch.all(out.log_prob < 0) and torch.all(out.num_decisions > 0)
            assert len(out.actions) == cfg.unpool_layers
        losses = model.losses(batch, rng=torch.Generator().manual_seed(1))
        assert torch.isfinite(losses.total), variant


def test_hidden_squares_never_reach_the_encoder_or_the_predictor():
    """Change what stands on the hidden squares: the masked encoding and the predictions must not change."""
    for variant in ("unpool", "registers"):
        cfg = small_config(variant)
        model = lje.LabyrinthJEPA(cfg).eval()
        rnd = random.Random(3)
        checked = 0
        for fen in FENS:
            board = chess.Board(fen, chess960=True)
            mask = ld.sample_mask(rnd, ld.MaskConfig(0.3, 0.5))
            orient = (lambda sq: sq) if board.turn == chess.WHITE else (lambda sq: sq ^ 56)
            other = board.copy(stack=False)
            for sq in chess.SQUARES:
                if mask[orient(sq)] and other.piece_at(sq) is not None:
                    other.remove_piece_at(sq)
            if other.board_fen() == board.board_fen():
                continue
            checked += 1
            outs = []
            for f in (ld.fen_of(board), ld.fen_of(other)):
                _, ctx, _, _ = ld.build_views(f, mask)
                graphs = ld.pack_views([ctx])
                with torch.no_grad():
                    enc = model.encoder(graphs, mask=mask, rng=torch.Generator().manual_seed(5))
                    pred = model.mask_predictor(enc.squares, graphs, mask)
                outs.append((enc.z, enc.squares, pred))
            for a, b in zip(*outs):
                assert torch.equal(a, b), f"{variant}: hidden squares leaked for {fen}"
        assert checked >= 3


def test_a_position_encodes_the_same_alone_or_in_a_batch():
    for variant in lje.TOKEN_VARIANTS:
        cfg = small_config(variant)
        enc = lje.LabyrinthEncoder(cfg).eval()
        views = [ld.build_views(f, torch.zeros(64, dtype=torch.bool))[0] for f in FENS]
        with torch.no_grad():
            together = enc(ld.pack_views(views), rng=torch.Generator().manual_seed(0))
            for b, view in enumerate(views):
                replay = [a.select([b]) for a in together.actions] if together.actions else None
                alone = enc(ld.pack_views([view]), actions_to_replay=replay)
                assert torch.allclose(alone.z[0], together.z[b], atol=1e-4), (variant, b)
                assert torch.allclose(alone.squares[0], together.squares[b], atol=1e-4), (variant, b)


def test_target_encoder_moves_only_by_the_moving_average():
    cfg = small_config("unpool")
    model = lje.LabyrinthJEPA(cfg)
    batch = make_batch(cfg)
    before = [p.clone() for p in model.target.parameters()]
    model.losses(batch, rng=torch.Generator().manual_seed(0)).total.backward()
    assert all(p.grad is None for p in model.target.parameters()), "gradient reached the target encoder"
    assert all(not n.startswith("target.") for n, p in model.named_parameters()
               if any(p is q for q in model.trainable_parameters()))
    # perturb the online encoder, then a moving-average step moves the target a fraction of the way
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)
    model.update_target(0.9)
    for p_t, p_old, p_c in zip(model.target.parameters(), before, model.encoder.parameters()):
        assert torch.allclose(p_t, 0.9 * p_old + 0.1 * p_c, atol=1e-6)
    assert abs(model.momentum_at(0, 100) - cfg.ema_start) < 1e-9 and abs(model.momentum_at(100, 100) - cfg.ema_end) < 1e-9


def test_losses_reach_every_trainable_part():
    for rl_on in (True, False):
        cfg = small_config("unpool", unpool_rl=rl_on)
        model = lje.LabyrinthJEPA(cfg)
        losses = model.losses(make_batch(cfg), rng=torch.Generator().manual_seed(2))
        losses.total.backward()

        def reached(module) -> bool:
            return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters())
        for name in ("input_proj", "square_gats", "token_proj", "transformer", "readout_proj"):
            assert reached(getattr(model.encoder, name)), f"no gradient reached encoder.{name}"
        assert reached(model.mask_predictor) and reached(model.next_predictor)
        assert reached(model.encoder.unpool_layers[0].mlp_y), "the unpool's feature MLP trains through the losses"
        policy = model.encoder.unpool_layers[0].mlp_r
        assert reached(policy) == rl_on, f"unpool decision policy gradient with unpool_rl={rl_on}"
        assert reached(model.baselines) == rl_on


def test_encoder_layouts_without_transformer_mean_readout_and_predictor_transformer():
    batch = make_batch(small_config("squares"))

    def reached(module) -> bool:
        return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in module.parameters())
    for variant in lje.TOKEN_VARIANTS:
        for kw in (dict(transformer_layers=0), dict(readout="mean"), dict(pred_transformer_layers=2),
                   dict(transformer_layers=0, readout="mean", pred_transformer_layers=2)):
            cfg = small_config(variant, **kw)
            model = lje.LabyrinthJEPA(cfg)
            losses = model.losses(make_batch(cfg), rng=torch.Generator().manual_seed(3))
            losses.total.backward()
            unreached = [n for n, p in model.named_parameters()
                         if not n.startswith("target.") and p.requires_grad and p.grad is None
                         and not n.startswith(("baselines.", "encoder.unpool_layers"))]
            assert not unreached, (variant, kw, unreached[:5])
            if cfg.pred_transformer_layers:
                assert reached(model.mask_predictor.transformer)
    # without a transformer the final square tokens are the GAT tokens, normalized; the mean readout projects
    # the tokens' mean
    cfg = small_config("squares", transformer_layers=0, readout="mean")
    encoder = lje.LabyrinthEncoder(cfg).eval()
    captured = {}
    encoder.transformer.register_forward_pre_hook(lambda m, args: captured.__setitem__("in", args[0]))
    with torch.no_grad():
        out = encoder(batch.full)
        assert torch.allclose(out.squares, encoder.final_norm(captured["in"]), atol=1e-6)
        assert torch.allclose(out.z, encoder.readout_proj(out.squares.mean(1)), atol=1e-5)
    assert not hasattr(encoder, "queries") and sum(p.numel() for p in encoder.transformer.parameters()) == 0


def test_mlp_token_mixers():
    """token_mixer "mlp" mixes nothing across tokens and keeps the mirror symmetry; "mixer" mixes and breaks it."""
    fen = FENS[3]
    mirror = ld.normalize_fen(chess.Board(fen).transform(chess.flip_horizontal).fen(en_passant="xfen"))
    graphs = ld.pack_views([ld.build_views(f, torch.zeros(64, dtype=torch.bool))[0] for f in (fen, mirror)])
    for mixer in ("mlp", "mixer"):
        for variant in ("squares", "registers"):
            cfg = small_config(variant, token_mixer=mixer, readout="mean")
            torch.manual_seed(0)
            model = lje.LabyrinthJEPA(cfg)
            model.losses(make_batch(cfg), rng=torch.Generator().manual_seed(0)).total.backward()
            assert all(p.grad is not None for p in model.encoder.transformer.parameters()), (mixer, variant)
        encoder = lje.LabyrinthEncoder(small_config("squares", token_mixer=mixer, readout="mean")).eval()
        # with "mlp", the square stage's output token for a square depends on nothing else after it
        captured = {}
        encoder.transformer.register_forward_pre_hook(lambda m, args: captured.__setitem__("in", args[0]))
        with torch.no_grad():
            out = encoder(graphs)
        z_gap = float((out.z[0] - out.z[1]).norm() / out.z[0].norm())
        if mixer == "mlp":
            with torch.no_grad():
                alone = encoder.final_norm(encoder.transformer(captured["in"][:, :1]))
            assert torch.allclose(alone, out.squares[:, :1], atol=1e-5) and z_gap < 1e-5, z_gap
        else:
            assert z_gap > 1e-3, z_gap
    try:
        small_config("unpool", token_mixer="mixer").validate()
        assert False, "the mixer needs a fixed number of tokens"
    except ValueError:
        pass


def test_representation_metrics():
    torch.manual_seed(0)
    # RankMe: an identity-like matrix uses every dimension, a rank-1 matrix one
    assert abs(lje.rankme(torch.eye(16)) - 16) < 1e-3
    assert lje.rankme(torch.randn(100, 1) @ torch.randn(1, 16)) < 1.01
    # LiDAR: 400 positions separated along 5 directions of a 32-dim space, small noise between samples -> about 5
    means = torch.zeros(400, 32)
    means[:, :5] = torch.randn(400, 5) * 3
    z = means.unsqueeze(1) + 0.05 * torch.randn(400, 4, 32)
    assert 4.0 < lje.lidar(z) < 6.5, lje.lidar(z)
    # an encoder that also fills its unused directions with sampling noise: RankMe rises sharply, LiDAR does not,
    # because LiDAR weighs each direction by how much positions differ along it relative to sampling noise
    noise = torch.zeros(400, 4, 32)
    noise[:, :, 5:] = 3 * torch.randn(400, 4, 27)
    noisy = means.unsqueeze(1) + noise
    assert lje.rankme(noisy[:, 0]) > 3 * lje.rankme(z[:, 0]), (lje.rankme(noisy[:, 0]), lje.rankme(z[:, 0]))
    assert lje.lidar(noisy) < 6.5, lje.lidar(noisy)
    # self-consistency: near 0 when samples agree, near 1 when they are noise
    assert lje.self_consistency(z) < 0.1
    assert lje.self_consistency(torch.randn(200, 4, 16)) > 0.8


def test_representation_report_runs():
    for variant in ("unpool", "squares"):
        cfg = small_config(variant)
        enc = lje.LabyrinthEncoder(cfg)
        report = lje.representation_report(enc, FENS * 4, samples=3)
        assert set(report) == {"lidar", "rankme", "rankme_centered", "rankme_single_sample", "self_consistency"}
        assert all(v == v for v in report.values())
        if variant == "squares":      # deterministic encoder: every sample of a position is identical
            assert report["self_consistency"] < 1e-5
        masked = lje.representation_report(enc, FENS * 4, samples=2, mask_config=ld.MaskConfig(0.05, 0.1))
        assert masked["self_consistency"] > 0


def test_anticollapse_regularizers():
    g = lambda seed: torch.Generator().manual_seed(seed)
    torch.manual_seed(0)
    gauss = torch.randn(256, 64)
    collapsed = 0.5 + 0.01 * torch.randn(256, 64)
    shifted = torch.randn(256, 64) + 2.0
    # SIGReg: near its sampling-noise floor for a standard normal, large for anything else
    assert float(lje.sigreg(gauss, g(1))) < 3
    for bad in (collapsed, shifted, 0.1 * torch.randn(256, 64)):
        assert float(lje.sigreg(bad, g(1))) > 50
    # new random directions each call, the same ones for the same generator state
    assert float(lje.sigreg(gauss, g(1))) == float(lje.sigreg(gauss, g(1)))
    assert float(lje.sigreg(gauss, g(1))) != float(lje.sigreg(gauss, g(2)))
    # VICReg: the variance hinge fires on collapse, not on a standard normal (it ignores a shared offset)
    assert float(lje.vicreg_terms(collapsed)[0]) > 0.9 and float(lje.vicreg_terms(gauss)[0]) < 0.05
    assert float(lje.vicreg_terms(shifted)[0]) < 0.05
    # both are differentiable
    for fn in (lambda z: lje.sigreg(z, g(0)), lambda z: sum(lje.vicreg_terms(z))):
        z = collapsed.clone().requires_grad_(True)
        fn(z).backward()
        assert float(z.grad.abs().sum()) > 0


def test_each_anticollapse_option_trains_the_encoder():
    for option in lje.ANTICOLLAPSE:
        for squares in (False, True):
            cfg = small_config("registers", anticollapse=option, anticollapse_squares=squares)
            model = lje.LabyrinthJEPA(cfg)
            losses = model.losses(make_batch(cfg), rng=torch.Generator().manual_seed(0))
            assert torch.isfinite(losses.total) and "z_mean_cosine" in losses.stats
            if option == "none":
                assert float(losses.anticollapse) == 0
                continue
            losses.anticollapse.backward()
            assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
                       for p in model.encoder.readout_proj.parameters()), option
            assert all(p.grad is None for p in model.target.parameters())


def test_castling_self_loop_reaches_the_gat_layers():
    """
    The castling flag on a king's self-loop must change what the chess-graph GAT layers compute. PyG's
    default (add_self_loops=True) would delete that self-loop and the flag with it.
    """
    import labyrinth_dgi_encoder as lde
    fen = ld.normalize_fen("rnbqrbkr/pppppppp/8/8/8/8/PPPPPPPP/RNBQRBKR w KQkq - 0 1")   # kings on g1 / g8
    full, _, _, _ = ld.build_views(fen, torch.zeros(64, dtype=torch.bool))
    graphs = ld.pack_views([full])
    loop = ((graphs.edge_index[0] == 6) & (graphs.edge_index[1] == 6)).nonzero().flatten()
    assert loop.numel() == 1 and float(graphs.edge_attr[loop, 9]) == 1
    without = ld.GraphBatch(graphs.x, graphs.edge_index, graphs.edge_attr.clone(), graphs.batch, 1)
    without.edge_attr[loop, 9] = 0
    torch.manual_seed(0)
    enc = lje.LabyrinthEncoder(small_config("squares")).eval()
    node = lde.NodeEncoder(lde.DGIConfig(enc_hidden_dim=32, node_dim=32, enc_heads=2)).eval()
    with torch.no_grad():
        a, b = enc(graphs).squares[0, 6], enc(without).squares[0, 6]
        assert not torch.allclose(a, b), "the JEPA encoder ignored the castling self-loop"
        a = node.forward_merged(graphs.x, graphs.edge_index, graphs.edge_attr)[6]
        b = node.forward_merged(without.x, without.edge_index, without.edge_attr)[6]
        assert not torch.allclose(a, b), "the DGI encoder ignored the castling self-loop"


def test_unbiased_sigreg_has_no_pull_at_small_batches():
    """At batch 48, the plain statistic expects about +1 on a standard normal and pushes the spread down; unbiased does neither."""
    def averages(unbiased):
        values, pulls = [], []
        for seed in range(150):
            scale = torch.tensor(1.0, requires_grad=True)
            z = torch.randn(48, 32, generator=torch.Generator().manual_seed(seed)) * scale
            v = lje.sigreg(z, torch.Generator().manual_seed(seed), unbiased=unbiased)
            v.backward()
            values.append(float(v.detach()))
            pulls.append(float(scale.grad))
        return sum(values) / len(values), sum(pulls) / len(pulls)
    biased_value, biased_pull = averages(False)
    value, pull = averages(True)
    assert biased_value > 0.5 and biased_pull > 0.4, (biased_value, biased_pull)
    assert abs(value) < 0.2 and abs(pull) < 0.25, (value, pull)


def test_search_space_leaves_out_fixed_values():
    names = lambda cs: set(cs.keys())
    full = names(lje.build_jepa_configspace())
    squares = lje.build_jepa_configspace(fixed={"token_variant": "squares"})
    assert not names(squares) & {"token_variant", "num_registers", "unpool_layers", "unpool_hidden", "rl_coef"}
    unpool = lje.build_jepa_configspace(fixed={"token_variant": "unpool", "lr": 1e-4})
    assert {"unpool_layers", "unpool_hidden", "rl_coef"} <= names(unpool) and "lr" not in names(unpool)
    for sample in unpool.sample_configuration(5):       # the unpool settings are now always sampled
        assert "unpool_layers" in dict(sample)
    assert not names(lje.build_jepa_configspace(fixed={"anticollapse": "none"})) & {"sigreg_weight", "vicreg_weight"}
    assert "rollout_stage_positions" not in names(lje.build_jepa_configspace(fixed={"rollout_max_steps": 1}))
    assert "rollout_stage_positions" in names(lje.build_jepa_configspace(fixed={"rollout_max_steps": 4}))
    for sample in lje.build_jepa_configspace(seed=3).sample_configuration(30):
        values = dict(sample)
        assert ("rollout_stage_positions" in values) == (values["rollout_max_steps"] > 1)
    assert names(squares) < full


def test_rollout_schedule():
    cfg = small_config(rollout_max_steps=3, rollout_stage_positions=1000, rollout_ramp_fraction=0.25,
                       rollout_step_decay=0.5)
    assert lje.rollout_schedule(cfg, 0) == [1.0]
    assert lje.rollout_schedule(cfg, 999) == [1.0]
    assert lje.rollout_schedule(cfg, 1000) == [1.0, 0.0]                # a new step starts at weight 0 ...
    assert abs(lje.rollout_schedule(cfg, 1125)[1] - 0.25) < 1e-9        # ... ramps over a quarter of its stage ...
    assert lje.rollout_schedule(cfg, 1250) == [1.0, 0.5]                # ... to decay ** 1
    assert len(lje.rollout_schedule(cfg, 10 ** 9)) == 3                 # never past rollout_max_steps
    assert lje.rollout_schedule(small_config(rollout_max_steps=3, rollout_stage_positions=0), 0) == [1.0, 1.0, 1.0]
    # starting longer: rollout_start_steps at once, then one more per stage
    start = small_config(rollout_start_steps=3, rollout_max_steps=20, rollout_stage_positions=1000)
    assert lje.rollout_schedule(start, 0) == [1.0, 1.0, 1.0]
    assert lje.rollout_schedule(start, 1000) == [1.0, 1.0, 1.0, 0.0]
    assert len(lje.rollout_schedule(start, 17_000)) == 20 and lje.rollout_schedule(start, 17_250)[-1] == 1.0
    # which steps are graded
    assert lje.rollout_target_steps(start, 20) is None
    log = small_config(rollout_targets="log", rollout_max_steps=20)
    assert lje.rollout_target_steps(log, 20) == [1, 2, 3, 4, 5, 8, 9, 16, 17, 19, 20]   # odd and even plies
    assert lje.rollout_target_steps(log, 1) == [1] and lje.rollout_target_steps(log, 3) == [1, 2, 3]
    for bad in (dict(rollout_start_steps=5, rollout_max_steps=4), dict(rollout_targets="some")):
        try:
            small_config(**bad).validate()
        except ValueError:
            continue
        raise AssertionError(f"{bad} must be refused")


def test_long_rollouts_grade_only_the_target_steps():
    cfg = small_config("registers", rollout_max_steps=10, rollout_targets="log")
    model = lje.LabyrinthJEPA(cfg)
    targets = lje.rollout_target_steps(cfg, 10)
    rnd = random.Random(32)
    batch = ld.collate_jepa([ld.build_sample(f, rnd, cfg.mask_config(), rollout_steps=10, target_steps=targets)
                             for f in FENS[:4]])
    losses = model.losses(batch, rng=torch.Generator().manual_seed(0), rollout_weights=[1.0] * 10)
    graded = {k for k in range(1, 11) if f"loss_next_{k}" in losses.stats}
    assert graded == {1, 2, 3, 4, 5, 8, 9, 10}, graded
    losses.total.backward()
    # the ungraded steps are still on the way to the graded ones, so their step embeddings train too
    assert float(model.next_predictor.step.weight.grad[6].abs().sum()) > 0
    assert float(model.next_predictor.step.weight.grad[10].abs().sum()) > 0


def test_sibling_moves_are_graded_with_the_first_step():
    cfg = small_config("squares", rollout_max_steps=3, rollout_siblings=3, move_encoding="rank_file",
                       inverse_weight=0.1, inverse_hidden=32)
    torch.manual_seed(0)
    model = lje.LabyrinthJEPA(cfg)

    def batch_with(siblings: int) -> ld.JEPABatch:      # one generator per position, as the dataset has
        return ld.collate_jepa([ld.build_sample(f, random.Random(33 + i), cfg.mask_config(), rollout_steps=3,
                                                siblings=siblings) for i, f in enumerate(FENS[:4])])

    alone, batch = batch_with(1), batch_with(cfg.rollout_siblings)
    rows = alone.rollout[0].index.numel()
    more = sum(min(2, chess.Board(f, chess960=True).legal_moves.count() - 1) for f in FENS[:4])
    assert more > 4 and batch.rollout[0].index.numel() == rows + more, "up to two more first moves per position"
    with torch.no_grad():
        enc = model.encoder(batch.full)
        a = model.predict_rollout(enc.z, enc.squares, alone.rollout)
        b = model.predict_rollout(enc.z, enc.squares, batch.rollout)
    # the siblings are extra rows of step 1; the rollouts themselves are untouched
    assert all(torch.allclose(x, y[:x.size(0)], atol=1e-6) for x, y in zip(a, b))
    assert b[1].size(0) == a[1].size(0) and b[2].size(0) == a[2].size(0)
    # different moves from one position get different predictions
    own, sibling = b[0][0], b[0][rows]
    assert int(batch.rollout[0].index[rows]) == 0 and float((own - sibling).abs().max()) > 0
    losses = model.losses(batch, rng=torch.Generator().manual_seed(0), rollout_weights=[1.0, 1.0, 1.0])
    assert torch.isfinite(losses.total) and "loss_inverse" in losses.stats
    losses.total.backward()
    assert float(model.next_predictor.names[0].weight.grad.abs().sum()) > 0


def test_latent_rollout_trains_every_step_without_peeking():
    cfg = small_config("registers", rollout_max_steps=3)
    model = lje.LabyrinthJEPA(cfg)
    rnd = random.Random(31)
    batch = ld.collate_jepa([ld.build_sample(f, rnd, cfg.mask_config(), rollout_steps=3) for f in FENS[:4]])
    assert len(batch.rollout) == 3
    losses = model.losses(batch, rng=torch.Generator().manual_seed(0), rollout_weights=[1.0, 1.0, 1.0])
    assert torch.isfinite(losses.total) and {"loss_next_1", "loss_next_2", "loss_next_3"} <= set(losses.stats)
    losses.total.backward()
    assert float(model.next_predictor.step.weight.grad[3].abs().sum()) > 0, "step 3 must train"
    # the rollout reads only the starting positions and the moves: garbage intermediate graphs change nothing
    with torch.no_grad():
        enc = model.encoder(batch.full)
        clean = model.predict_rollout(enc.z, enc.squares, batch.rollout)
        for step in batch.rollout:
            step.graphs.x.fill_(7.0)
            step.graphs.edge_attr.fill_(3.0)
        dirty = model.predict_rollout(enc.z, enc.squares, batch.rollout)
    assert all(torch.equal(a, b) for a, b in zip(clean, dirty))
    # a step with weight 0 contributes nothing to the loss
    a = model.losses(batch, rng=torch.Generator().manual_seed(0), rollout_weights=[1.0, 0.0]).next
    b = model.losses(batch, rng=torch.Generator().manual_seed(0), rollout_weights=[1.0]).next
    assert torch.allclose(a, b, atol=1e-6)


def test_decorrelation_term_adds_to_sigreg():
    """decorrelation_weight adds VICReg's covariance term on top of SIGReg, and trains the encoder through it."""
    # it separates a few factors spread across all dimensions from isotropic embeddings
    torch.manual_seed(0)
    assert float(lje.vicreg_terms(torch.randn(256, 20) @ torch.randn(20, 512) / 20 ** 0.5)[1]) >         5 * float(lje.vicreg_terms(torch.randn(256, 512))[1])
    batch = make_batch(small_config("registers"))
    plain = lje.LabyrinthJEPA(small_config("registers"))
    assert "decorrelation" not in plain.losses(batch, rng=torch.Generator().manual_seed(0)).stats
    cfg = small_config("registers", decorrelation_weight=10.0)
    model = lje.LabyrinthJEPA(cfg)
    model.load_state_dict(plain.state_dict())
    with_term = model.losses(batch, rng=torch.Generator().manual_seed(0))
    without = plain.losses(batch, rng=torch.Generator().manual_seed(0))
    assert with_term.stats["decorrelation"] > 0
    assert abs(float(with_term.total - without.total) - 10.0 * with_term.stats["decorrelation"]) < 1e-3
    with_term.total.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.encoder.readout_proj.parameters())


def test_loss_terms_add_up_to_the_total():
    """JEPALosses.terms splits total into each term's weighted share, one per regularized embedding."""
    cases = [("unpool", dict()), ("registers", dict(decorrelation_weight=0.5, anticollapse_squares=True)),
             ("squares", dict(anticollapse="vicreg")), ("squares", dict(anticollapse="none")),
             ("squares", dict(zpred_weight=0.5, projector_layers=2, projector_hidden=32, projector_dim=16)),
             ("unpool", dict(transformer_layers=0, readout="mean", anticollapse_square_mean=True)),
             ("squares", dict(inverse_weight=0.5, move_encoding="rank_file", move_piece=True)),
             ("squares", dict(gat_style="concat", gat_sum=True, transformer_layers=0, readout="mean_max",
                              readout_hidden=32))]
    for variant, kw in cases:
        cfg = small_config(variant, **kw)
        torch.manual_seed(0)
        out = lje.LabyrinthJEPA(cfg).losses(make_batch(cfg), rng=torch.Generator().manual_seed(0))
        assert abs(float(sum(out.terms.values()) - out.total)) < 1e-5, (variant, kw)
        expected = {"mask", "next"}
        if cfg.anticollapse != "none":
            expected |= {"anticollapse_masked", "anticollapse_full"}
            expected |= {"anticollapse_squares"} if cfg.anticollapse_squares else set()
            expected |= ({"anticollapse_square_mean_masked", "anticollapse_square_mean_full"}
                         if cfg.anticollapse_square_mean else set())
            expected |= {"decorrelation"} if cfg.decorrelation_weight > 0 else set()
            assert all(f"anticollapse_{n}" in out.stats for n in ("masked", "full"))
        expected |= {"rl"} if cfg.uses_unpool and cfg.unpool_rl else set()
        expected |= {"zpred"} if cfg.zpred_weight > 0 else set()
        expected |= {"inverse"} if cfg.inverse_weight > 0 else set()
        assert set(out.terms) == expected, (variant, kw, sorted(out.terms))


def test_heads_graded_against_the_board():
    """The piece-count decoder and the legal-move head: their labels, their losses, and the other weights' start."""
    kw = dict(transformer_layers=0, readout="mean")
    cfg = small_config("squares", count_weight=1.0, count_hidden=32, legal_weight=1.0, legal_hidden=32, **kw)
    batch = make_batch(cfg, seed=4)

    counts = lje.piece_counts(batch.full)
    graded, legal = lje.legal_move_edges(batch)
    src, dst = batch.full.edge_index
    for i, fen in enumerate(FENS):
        board = chess.Board(fen, chess960=True)
        mine = [len(board.pieces(t, board.turn)) for t in chess.PIECE_TYPES]
        theirs = [len(board.pieces(t, not board.turn)) for t in chess.PIECE_TYPES]
        assert counts[i].tolist() == mine + theirs
        orient = (lambda sq: sq) if board.turn == chess.WHITE else (lambda sq: sq ^ 56)
        pairs = {(orient(m.from_square), orient(m.to_square)) for m in board.legal_moves}
        here = batch.full.batch[src] == i
        found = {(int(a) % 64, int(b) % 64) for a, b in zip(src[here & legal], dst[here & legal])}
        assert found == pairs, "the edges labelled legal are exactly the position's legal moves"
        assert not bool((here & legal & ~graded).any()), "and every one of them is graded"
        own = {s for s in chess.SQUARES if board.color_at(s) == board.turn}
        assert {int(a) % 64 for a in src[here & graded]} == {orient(s) for s in own}
    assert not bool((graded & (src == dst)).any())

    torch.manual_seed(0)
    model = lje.LabyrinthJEPA(cfg)
    torch.manual_seed(0)
    plain = lje.LabyrinthJEPA(small_config("squares", **kw))
    assert plain.count_decoder is None and plain.legal_head is None
    theirs = plain.state_dict()
    assert theirs and all(torch.equal(v, model.state_dict()[k]) for k, v in theirs.items()),         "the heads are built last: every other weight starts as without them"

    out = model.losses(batch, rng=torch.Generator().manual_seed(0))
    assert {"count", "legal"} <= set(out.terms) and abs(float(sum(out.terms.values()) - out.total)) < 1e-5
    for key in ("loss_count", "count_error", "count_exact", "loss_legal", "legal_rate", "legal_precision", "legal_recall"):
        assert key in out.stats and math.isfinite(out.stats[key]), key
    assert abs(out.stats["legal_rate"] - float(legal[graded].float().mean())) < 1e-6
    # the count loss reaches the encoder through z, the legal-move loss through the square tokens
    encoder = [p for p in model.encoder.parameters()]
    for term, head in (("count", model.count_decoder), ("legal", model.legal_head)):
        grads = torch.autograd.grad(out.terms[term], encoder + list(head.parameters()), retain_graph=True,
                                    allow_unused=True)
        assert float(sum(g.abs().sum() for g in grads[:len(encoder)] if g is not None)) > 0, term
        assert all(g is not None and float(g.abs().sum()) > 0 for g in grads[len(encoder):]), term
    readout = torch.autograd.grad(out.terms["legal"], model.encoder.readout_proj.weight, allow_unused=True)[0]
    assert readout is None, "the legal-move head reads the square tokens, not z"

    # each head alone, and neither without a rollout to share the whole board's encoding with
    for only in (dict(count_weight=0.5, count_hidden=32), dict(legal_weight=0.5, legal_hidden=32)):
        single = lje.LabyrinthJEPA(small_config("squares", **only, **kw))
        o = single.losses(batch, rng=torch.Generator().manual_seed(0))
        assert ("count" in o.terms) == ("count_weight" in only) and ("legal" in o.terms) == ("legal_weight" in only)
        ended = ld.collate_jepa([ld.build_sample("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", random.Random(i), cfg.mask_config())
                                 for i in range(2)])     # stalemate
        assert not ended.rollout and torch.isfinite(single.losses(ended, rng=torch.Generator().manual_seed(0)).total)
    for bad in (dict(count_weight=-1.0), dict(legal_weight=-1.0)):
        try:
            small_config("squares", **bad).validate()
            raise AssertionError("a negative weight should be rejected")
        except ValueError:
            pass


def test_position_decoder_reads_every_square_from_z():
    """The position decoder: its labels are the board, its loss reaches the encoder through z, it is built last."""
    kw = dict(transformer_layers=0, readout="mean")
    cfg = small_config("squares", decode_weight=1.0, decode_hidden=48, count_weight=1.0, count_hidden=32,
                       legal_weight=1.0, legal_hidden=32, **kw)
    batch = make_batch(cfg, seed=4)
    classes = lje.square_classes(batch.full)
    assert classes.shape == (len(FENS), 64) and classes.dtype == torch.long
    en_passant = 0
    for i, fen in enumerate(FENS):
        board = chess.Board(fen, chess960=True)
        orient = (lambda sq: sq) if board.turn == chess.WHITE else (lambda sq: sq ^ 56)
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is not None:
                expected = piece.piece_type + (0 if piece.color == board.turn else 6)
            else:
                expected = lje.SQUARE_CLASSES - 1 if sq == board.ep_square else 0
            en_passant += expected == lje.SQUARE_CLASSES - 1
            assert int(classes[i, orient(sq)]) == expected, (fen, chess.square_name(sq))
    assert en_passant == 1, "one of the positions has an en passant square"

    torch.manual_seed(0)
    model = lje.LabyrinthJEPA(cfg)
    torch.manual_seed(0)
    plain = lje.LabyrinthJEPA(small_config("squares", count_weight=1.0, count_hidden=32, legal_weight=1.0,
                                           legal_hidden=32, **kw))
    assert plain.board_decoder is None
    theirs, ours = plain.state_dict(), model.state_dict()
    assert all(torch.equal(v, ours[k]) for k, v in theirs.items()), "built last: the other weights start as without it"
    assert all(k.startswith("board_decoder.") for k in set(ours) - set(theirs)) and len(ours) > len(theirs)
    assert model.board_decoder(torch.randn(3, cfg.z_dim)).shape == (3, 64, lje.SQUARE_CLASSES)

    out = model.losses(batch, rng=torch.Generator().manual_seed(0))
    assert "decode" in out.terms and abs(float(sum(out.terms.values()) - out.total)) < 1e-5
    for key in ("loss_decode", "decode_squares", "decode_pieces", "decode_exact"):
        assert key in out.stats and math.isfinite(out.stats[key]) and out.stats[key] >= 0, key
    assert abs(out.stats["loss_decode"] - math.log(lje.SQUARE_CLASSES)) < 1.0, "an untrained decoder guesses"
    encoder = list(model.encoder.parameters())
    grads = torch.autograd.grad(out.terms["decode"], encoder + list(model.board_decoder.parameters()),
                                retain_graph=True, allow_unused=True)
    assert float(sum(g.abs().sum() for g in grads[:len(encoder)] if g is not None)) > 0
    assert all(g is not None and float(g.abs().sum()) > 0 for g in grads[len(encoder):])
    readout = torch.autograd.grad(out.terms["decode"], model.encoder.readout_proj.weight)[0]
    assert float(readout.abs().sum()) > 0, "the decoder reads z"
    assert all(p.grad is None for p in model.target.parameters())

    # alone, and without a rollout to share the whole board's encoding with
    single = lje.LabyrinthJEPA(small_config("squares", decode_weight=0.5, decode_hidden=48, **kw))
    assert "decode" in single.losses(batch, rng=torch.Generator().manual_seed(0)).terms
    ended = ld.collate_jepa([ld.build_sample("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", random.Random(i), cfg.mask_config())
                             for i in range(2)])         # stalemate
    assert not ended.rollout and torch.isfinite(single.losses(ended, rng=torch.Generator().manual_seed(0)).total)
    try:
        small_config("squares", decode_weight=-1.0).validate()
        raise AssertionError("a negative weight should be rejected")
    except ValueError:
        pass


def test_z_heads_judge_predictions_with_the_head_held_fixed():
    """
    The heads that read only z (the position decoder, the z-level legal-move head), asked about the position's
    own z, the predicted next z and the whole-board z predicted from the masked board. Their labels are the
    board's; a head learns from the target encoder's z (nothing reaches the encoder that way) and is held
    fixed while it judges a prediction (that loss reaches the predictor and the encoder, not the head).
    """
    kw = dict(transformer_layers=0, readout="mean", move_encoding="rank_file", orientation_edges=True,
              decode_hidden=48, zlegal_hidden=48, zlegal_dim=24, zpred_hidden=64)
    rnd = random.Random(2)
    cfg = small_config("squares", **kw)
    samples = [ld.build_sample(f, rnd, cfg.mask_config(), rollout_steps=2) for f in FENS]
    batch = ld.collate_jepa(samples)
    weights = [1.0, 0.5]

    # the moves asked about, and their labels
    pairs = lje.move_pairs()
    assert pairs.shape == (1792, 2) and len({(int(a), int(b)) for a, b in pairs}) == 1792
    head = lje.ZLegalHead(cfg)
    assert head.num_pairs == 1792 and not [k for k in head.state_dict() if "pair" in k or "parts" in k]
    now = head.labels(batch.legal, len(FENS))
    for i, fen in enumerate(FENS):
        assert {(int(a), int(b)) for a, b in pairs[now[i]]} == {(code // 64, code % 64) for code in samples[i].legal}
        assert samples[i].legal == ld.legal_codes(chess.Board(fen, chess960=True))
    for step in batch.rollout:
        after = head.labels(step.legal, step.graphs.num_graphs)
        assert int(after.sum()) == step.legal.numel() and after.shape == (step.index.numel(), 1792)
    try:
        head.labels(torch.tensor([0 * 64 + 11]), 1)                     # a1 to d2 is no queen or knight path
        raise AssertionError("a move outside move_pairs() should be rejected")
    except ValueError:
        pass
    z = torch.randn(3, cfg.z_dim)
    assert head(z).shape == (3, 1792) and torch.allclose(head(z), head(2.5 * z + 1.0), atol=1e-4), "it reads z normalized"
    assert torch.equal(lje.held_fixed(head, z), head(z))
    loss, scores = lje.legal_scores(torch.tensor([[4.0, -4.0, -4.0, 4.0]]), torch.tensor([[True, False, False, False]]))
    assert scores["found"] == 1.0 and abs(scores["rejected"] - 2 / 3) < 1e-6
    assert abs(float(loss) - 0.5 * (0.0181 + (2 * 0.0181 + 4.0181) / 3)) < 1e-3, "legal and illegal weigh the same"

    def build(seed=0, **weights_kw):
        torch.manual_seed(seed)
        return lje.LabyrinthJEPA(small_config("squares", **kw, **weights_kw))

    def reached(term, *parts):
        """Which of the parts a loss term's gradient reaches."""
        out = []
        for part in parts:
            params = [p for p in part.parameters() if p.requires_grad]
            grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            out.append(any(g is not None and float(g.abs().sum()) > 0 for g in grads))
        return out

    base, control = build(), build(zpred_weight=1.0)
    assert base.board_decoder is None and base.zlegal_head is None
    arms = {
        "decode_next": (dict(decode_next_weight=1.0), base, "board_decoder", "next_predictor"),
        "decode_masked": (dict(decode_masked_weight=1.0, zpred_weight=1.0), control, "board_decoder", "full_predictor"),
        "zlegal_next": (dict(zlegal_next_weight=1.0), base, "zlegal_head", "next_predictor"),
        "zlegal_masked": (dict(zlegal_masked_weight=1.0, zpred_weight=1.0), control, "zlegal_head", "full_predictor"),
    }
    for name, (weights_kw, without, head_name, predictor_name) in arms.items():
        model = build(**weights_kw)
        theirs, ours = without.state_dict(), model.state_dict()
        assert all(torch.equal(v, ours[k]) for k, v in theirs.items()), f"{name}: the other weights start as without it"
        assert {k.split(".")[0] for k in set(ours) - set(theirs)} == {head_name}
        head_module, predictor = getattr(model, head_name), getattr(model, predictor_name)
        other = model.zlegal_head if head_name == "board_decoder" else model.board_decoder
        assert other is None and (head_name != "board_decoder" or head_module.normalize)
        out = model.losses(batch, rng=torch.Generator().manual_seed(0), rollout_weights=weights)
        fit = name.split("_")[0] + "_fit"
        assert {name, fit} <= set(out.terms) and abs(float(sum(out.terms.values()) - out.total)) < 1e-4, name
        assert reached(out.terms[fit], head_module, model.encoder, predictor) == [True, False, False], \
            f"{name}: the head learns from the target encoder's z alone"
        assert reached(out.terms[name], head_module, model.encoder, predictor) == [False, True, True], \
            f"{name}: a prediction is judged with the head held fixed"
        assert all(p.grad is None for p in model.target.parameters())
        for key in (f"loss_{fit}", f"loss_{name}"):
            assert math.isfinite(out.stats[key]) and out.stats[key] > 0, key
        scores = ("pieces", "squares", "exact") if head_name == "board_decoder" else ("found", "rejected")
        for key in scores:
            assert 0 <= out.stats[f"{name}_{key}"] <= 1 and 0 <= out.stats[f"{fit}_{key}"] <= 1
            if name.endswith("_next"):                                  # each graded step, and their weighted mean
                each = [out.stats[f"{name}_{key}_{k}"] for k in (1, 2)]
                assert abs(out.stats[f"{name}_{key}"] - (each[0] + 0.5 * each[1]) / 1.5) < 1e-6
        # without a rollout the next-z term is left out and the rest still stands
        ended = ld.collate_jepa([ld.build_sample("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", random.Random(i), cfg.mask_config())
                                 for i in range(2)])                     # stalemate
        o = model.losses(ended, rng=torch.Generator().manual_seed(0))
        assert not ended.rollout and torch.isfinite(o.total) and fit in o.terms
        assert (name in o.terms) == name.endswith("_masked")

    # asked about the position's own z, a head and the encoder learn together (not a prediction)
    for name, weights_kw, head_name in (("zlegal", dict(zlegal_weight=1.0), "zlegal_head"),
                                        ("decode", dict(decode_weight=1.0), "board_decoder")):
        model = build(**weights_kw)
        theirs, ours = base.state_dict(), model.state_dict()
        assert all(torch.equal(v, ours[k]) for k, v in theirs.items())
        out = model.losses(batch, rng=torch.Generator().manual_seed(0), rollout_weights=weights)
        assert name in out.terms and name + "_fit" not in out.terms
        assert reached(out.terms[name], getattr(model, head_name), model.encoder, model.next_predictor) == [True, True, False]
        assert head_name != "board_decoder" or not model.board_decoder.normalize     # as in the runs before these
    both = build(zlegal_weight=1.0, zlegal_next_weight=1.0, decode_weight=1.0, decode_masked_weight=1.0, zpred_weight=1.0)
    out = both.losses(batch, rng=torch.Generator().manual_seed(0), rollout_weights=weights)
    assert {"zlegal", "zlegal_fit", "zlegal_next", "decode", "decode_fit", "decode_masked"} <= set(out.terms)
    assert "zlegal_masked" not in out.terms and "decode_next" not in out.terms and both.board_decoder.normalize
    assert abs(float(sum(out.terms.values()) - out.total)) < 1e-4

    # zhead_fit "online": a head judging predictions learns from the online z of the whole board instead, so its
    # fit term reaches the encoder too; a prediction is still judged with the head held fixed
    for name, weights_kw, head_name, predictor_name in (
            ("decode_next", dict(decode_next_weight=1.0), "board_decoder", "next_predictor"),
            ("decode_masked", dict(decode_masked_weight=1.0, zpred_weight=1.0), "board_decoder", "full_predictor"),
            ("zlegal_next", dict(zlegal_next_weight=1.0), "zlegal_head", "next_predictor")):
        target_model, model = build(**weights_kw), build(zhead_fit="online", **weights_kw)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in target_model.state_dict().items())
        out = model.losses(batch, rng=torch.Generator().manual_seed(0), rollout_weights=weights)
        fit = name.split("_")[0] + "_fit"
        head_module, predictor = getattr(model, head_name), getattr(model, predictor_name)
        assert reached(out.terms[fit], head_module, model.encoder, predictor) == [True, True, False], name
        assert reached(out.terms[name], head_module, model.encoder, predictor) == [False, True, True], name
        assert abs(float(sum(out.terms.values()) - out.total)) < 1e-4
        if head_name == "board_decoder":                                # the fit sees the whole board alone
            whole = model.encoder(batch.full, rng=torch.Generator().manual_seed(0))
            expected, _ = lje.decode_scores(model.board_decoder(whole.z), lje.square_classes(batch.full))
            assert abs(float(out.terms[fit]) - float(expected)) < 1e-5
    for bad in (dict(zhead_fit="both"), dict(zhead_fit="online", decode_next_weight=1.0, decode_weight=1.0),
                dict(zhead_fit="online", zlegal_masked_weight=1.0, zpred_weight=1.0, zlegal_weight=1.0)):
        try:
            small_config("squares", **kw, **bad).validate()
            raise AssertionError(f"{bad} should be rejected")
        except ValueError:
            pass

    stripped = ld.collate_jepa(samples)
    stripped.rollout[0].legal = None
    try:
        build(zlegal_next_weight=1.0).losses(stripped, rng=torch.Generator().manual_seed(0), rollout_weights=weights)
        raise AssertionError("judging next positions' legal moves needs them")
    except ValueError:
        pass
    for bad in (dict(decode_masked_weight=1.0), dict(zlegal_masked_weight=1.0), dict(zlegal_weight=-1.0),
                dict(decode_next_weight=-1.0)):
        try:
            small_config("squares", **bad).validate()
            raise AssertionError(f"{bad} should be rejected")
        except ValueError:
            pass


def test_whole_board_predictor_projector_and_regularized_views():
    batch = make_batch(small_config("squares"))
    rng = lambda: torch.Generator().manual_seed(0)
    # the whole-board term grades z of the masked board, through the readout
    cfg = small_config("squares", zpred_weight=1.0, anticollapse="none")
    model = lje.LabyrinthJEPA(cfg)
    out = model.losses(batch, rng=rng())
    assert out.stats["loss_zpred"] > 0
    out.terms["zpred"].backward()
    assert float(model.full_predictor.body[0].weight.grad.abs().sum()) > 0
    assert float(model.encoder.readout_proj.weight.grad.abs().sum()) > 0
    assert lje.LabyrinthJEPA(small_config("squares")).full_predictor is None
    # which views are regularized
    for views, names in (("both", {"masked", "full"}), ("full", {"full"}), ("masked", {"masked"})):
        out = lje.LabyrinthJEPA(small_config("squares", anticollapse_views=views)).losses(batch, rng=rng())
        assert {k[len("anticollapse_"):] for k in out.terms if k.startswith("anticollapse_")} == names, views
    # with a projector, the regularizer sees the head's output, and its gradient reaches z through the head
    cfg = small_config("squares", projector_layers=2, projector_hidden=48, projector_dim=24)
    model = lje.LabyrinthJEPA(cfg)
    seen = []
    original = lje.sigreg
    try:
        lje.sigreg = lambda e, *a, **k: (seen.append(e.shape[-1]), original(e, *a, **k))[1]
        out = model.losses(batch, rng=rng())
    finally:
        lje.sigreg = original
    assert seen == [24, 24]
    out.terms["anticollapse_full"].backward()
    assert float(model.projector[0].weight.grad.abs().sum()) > 0
    assert float(model.encoder.readout_proj.weight.grad.abs().sum()) > 0
    assert not any(n.startswith("projector") for n, _ in model.encoder.named_parameters())


def test_square_mean_regularizer_weighs_like_z():
    """anticollapse_square_mean adds each view's mean square token at its z's weight; z's own terms stay as they were."""
    def run(**kw):
        cfg = small_config("unpool", transformer_layers=0, readout="mean", **kw)
        torch.manual_seed(0)
        model = lje.LabyrinthJEPA(cfg)
        return model, model.losses(make_batch(cfg), rng=torch.Generator().manual_seed(0))
    _, plain = run()
    model, out = run(anticollapse_square_mean=True)
    for view in ("masked", "full"):
        assert abs(float(plain.terms[f"anticollapse_{view}"] - out.terms[f"anticollapse_{view}"])) < 1e-6, view
        share = float(out.terms[f"anticollapse_square_mean_{view}"])
        assert share > 0 and abs(share - 0.05 / 2 * out.stats[f"anticollapse_square_mean_{view}"]) < 1e-6, view
    # it trains the square stage directly, not through z's readout
    out.terms["anticollapse_square_mean_masked"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.encoder.square_gats.parameters())
    assert model.encoder.readout_proj.weight.grad is None


def test_orientation_anchor_breaks_the_mirror_symmetry():
    """A position without castling rights and its a <-> h mirror encode alike, unless orientation_anchor is on."""
    fen = FENS[3]
    assert not chess.Board(fen).castling_rights
    mirror = ld.normalize_fen(chess.Board(fen).transform(chess.flip_horizontal).fen(en_passant="xfen"))
    graphs = ld.pack_views([ld.build_views(f, torch.zeros(64, dtype=torch.bool))[0] for f in (fen, mirror)])
    marked = lje.with_orientation_anchor(graphs.edge_index, graphs.edge_attr)[:, -1].bool()
    assert graphs.edge_index[:, marked].tolist() == [[0, 64], [0, 64]]               # a1's self-loop, in each graph
    flip = torch.tensor([s ^ 7 for s in range(64)])                                 # a <-> h, ranks kept
    for anchor in (False, True):
        torch.manual_seed(0)
        out = lje.LabyrinthEncoder(small_config("squares", orientation_anchor=anchor)).eval()(graphs)
        z_gap = float((out.z[0] - out.z[1]).norm() / out.z[0].norm())
        token_gap = float((out.squares[0] - out.squares[1][flip]).norm() / out.squares[0].norm())
        # (untrained, the anchor only nudges attention weights, so the gap is small, but far above rounding)
        if anchor:
            assert z_gap > 1e-6 and token_gap > 1e-6, (z_gap, token_gap)
        else:
            assert z_gap < 1e-6 and token_gap < 1e-6, (z_gap, token_gap)
    # every variant trains with it (the predictor's GAT and the unpooling layers read the wider edges too)
    for variant in ("squares", "unpool"):
        cfg = small_config(variant, orientation_anchor=True)
        lje.LabyrinthJEPA(cfg).losses(make_batch(cfg), rng=torch.Generator().manual_seed(0)).total.backward()


def test_orientation_flag_keeps_the_other_weights_and_breaks_the_mirror_symmetry():
    """
    orientation_flag draws its start from its own generator: same seed, same other weights and the same
    random numbers afterwards as without it, so runs with and without it stay paired. A position and its
    mirror encode apart from the start, even with a1 hidden, and the flag trains.
    """
    fen = FENS[3]                                                                   # no castling rights
    mirror = ld.normalize_fen(chess.Board(fen).transform(chess.flip_horizontal).fen(en_passant="xfen"))
    graphs = ld.pack_views([ld.build_views(f, torch.zeros(64, dtype=torch.bool))[0] for f in (fen, mirror)])
    flip = torch.tensor([s ^ 7 for s in range(64)])                                 # a <-> h, ranks kept
    models, after = [], []
    for flag in (False, True):
        torch.manual_seed(0)
        models.append(lje.LabyrinthJEPA(small_config("squares", orientation_flag=flag)))
        after.append(torch.rand(4))
    plain, flagged = models[0].state_dict(), models[1].state_dict()
    assert all(torch.equal(value, flagged[name]) for name, value in plain.items())
    assert sorted(set(flagged) - set(plain)) == ["encoder.a1_flag", "target.a1_flag"]
    assert torch.equal(after[0], after[1])
    bound = models[1].config.node_feature_dim ** -0.5                               # input_proj's init range
    assert 0 < float(flagged["encoder.a1_flag"].abs().max()) <= bound
    assert torch.equal(flagged["encoder.a1_flag"], flagged["target.a1_flag"])
    torch.manual_seed(1)
    assert not torch.equal(lje.LabyrinthEncoder(small_config("squares", orientation_flag=True)).a1_flag,
                           flagged["encoder.a1_flag"])                              # it follows the seed
    model = models[1]
    batch = make_batch(model.config)
    model.losses(batch, rng=torch.Generator().manual_seed(0)).total.backward()
    assert float(model.encoder.a1_flag.grad.abs().sum()) > 0

    def gaps(mask=None):
        out = model.encoder.eval()(graphs, mask)
        return (float((out.z[0] - out.z[1]).norm() / out.z[0].norm()),
                float((out.squares[0] - out.squares[1][flip]).norm() / out.squares[0].norm()))

    # (untrained, most of z's norm is the vector every position shares, so its gap looks smaller)
    with torch.no_grad():
        z_gap, token_gap = gaps()
        assert z_gap > 1e-4 and token_gap > 1e-3, (z_gap, token_gap)
        hidden = torch.zeros(128, dtype=torch.bool)
        hidden[[0, 64]] = True                                                      # a1 hidden in both
        z_gap, token_gap = gaps(hidden)
        assert z_gap > 1e-4 and token_gap > 1e-3, (z_gap, token_gap)


def test_file_marks_keep_the_other_weights_and_tell_left_from_right():
    """
    orientation_files tells every square its file. Like the a1 flag it starts from its own generator, so
    the other weights and the random numbers afterwards are those of a run without it; unlike the flag it
    needs no relay across the board, so a position and its mirror encode much further apart.
    """
    fen = FENS[3]                                                                   # no castling rights
    mirror = ld.normalize_fen(chess.Board(fen).transform(chess.flip_horizontal).fen(en_passant="xfen"))
    graphs = ld.pack_views([ld.build_views(f, torch.zeros(64, dtype=torch.bool))[0] for f in (fen, mirror)])
    flip = torch.tensor([s ^ 7 for s in range(64)])                                 # a <-> h, ranks kept
    models, after = {}, {}
    for name, kw in (("plain", {}), ("flag", dict(orientation_flag=True)), ("files", dict(orientation_files=True)),
                     ("both", dict(orientation_flag=True, orientation_files=True))):
        torch.manual_seed(0)
        models[name] = lje.LabyrinthJEPA(small_config("squares", **kw))
        after[name] = torch.rand(4)
    plain, marked = models["plain"].state_dict(), models["files"].state_dict()
    assert all(torch.equal(value, marked[name]) for name, value in plain.items())
    assert sorted(set(marked) - set(plain)) == ["encoder.file_marks", "target.file_marks"]
    assert all(torch.equal(after["plain"], a) for a in after.values())
    marks = marked["encoder.file_marks"]
    assert marks.shape == (8, models["files"].config.gat_dim) and torch.equal(marks, marked["target.file_marks"])
    assert 0 < float(marks.abs().max()) <= models["files"].config.node_feature_dim ** -0.5
    assert len({tuple(row.tolist()) for row in marks}) == 8, "every file has its own mark"
    assert not torch.equal(marks[0], models["flag"].state_dict()["encoder.a1_flag"])   # not the flag's numbers
    assert torch.equal(models["both"].state_dict()["encoder.file_marks"], marks)

    def gaps(model, mask=None):
        out = model.encoder.eval()(graphs, mask)
        return (float((out.z[0] - out.z[1]).norm() / out.z[0].norm()),
                float((out.squares[0] - out.squares[1][flip]).norm() / out.squares[0].norm()))

    with torch.no_grad():
        assert max(gaps(models["plain"])) < 1e-5, "without a mark, mirror images encode alike"
        z_flag, token_flag = gaps(models["flag"])
        z_files, token_files = gaps(models["files"])
        assert z_files > 2 * z_flag and token_files > 2 * token_flag, (z_files, z_flag, token_files, token_flag)
        hidden = torch.rand(128, generator=torch.Generator().manual_seed(0)) < 0.5
        hidden[64:] = hidden[:64][flip]                                             # the mirrored squares, hidden
        z_hidden, token_hidden = gaps(models["files"], hidden)
        assert z_hidden > 2 * z_flag and token_hidden > 2 * token_flag, "hidden squares keep their marks"
        # the same file on every rank, whichever side is to move: only the files tell the squares apart
        empty = ld.pack_views([ld.build_views("8/8/8/8/8/8/8/8 w - - 0 1", torch.zeros(64, dtype=torch.bool))[0]])
        encoder = models["files"].encoder
        start = encoder.input_proj(empty.x) + encoder.file_marks[torch.arange(64) % 8]
        assert torch.allclose(start.view(8, 8, -1)[0], start.view(8, 8, -1)[5])
    model = models["files"]
    model.losses(make_batch(model.config), rng=torch.Generator().manual_seed(0)).total.backward()
    assert float(model.encoder.file_marks.grad.abs().sum()) > 0


def test_edge_marks_keep_the_other_weights_and_tell_left_from_right():
    """
    orientation_edges marks which way each edge runs across the board. The weights that read the marks come
    from their own generators, so every other weight, and the random numbers afterwards, are those of a run
    without them. A position and its mirror image then encode apart, and the marks' weights train.
    """
    # in the second graph of a batch: messages into a1 from c1, a3 and c3, into c1 from a1, and a self-loop
    a1, c1, a3, c3, f1 = 64, 66, 80, 82, 69
    edge_index = torch.tensor([[c1, a1, a3, c3, f1], [a1, c1, a1, a1, f1]])
    marked = lje.with_orientation_edges(edge_index, torch.zeros(5, 10))
    assert marked.shape == (5, 12) and float(marked[:, :10].abs().sum()) == 0
    assert marked[:, 10:].tolist() == [[1, 0], [0, 1], [0, 0], [1, 0], [0, 0]]      # (from the right, from the left)

    fen = FENS[3]                                                                   # no castling rights
    mirror = ld.normalize_fen(chess.Board(fen).transform(chess.flip_horizontal).fen(en_passant="xfen"))
    graphs = ld.pack_views([ld.build_views(f, torch.zeros(64, dtype=torch.bool))[0] for f in (fen, mirror)])
    flip = torch.tensor([s ^ 7 for s in range(64)])                                 # a <-> h, ranks kept
    models, after = {}, {}
    for name, kw in (("plain", {}), ("edges", dict(orientation_edges=True))):
        torch.manual_seed(0)
        models[name] = lje.LabyrinthJEPA(small_config("squares", **kw))
        after[name] = torch.rand(4)
    cfg = models["edges"].config
    plain, state = models["plain"].state_dict(), models["edges"].state_dict()
    assert set(plain) == set(state) and torch.equal(after["plain"], after["edges"])
    wider = [k for k in plain if plain[k].shape != state[k].shape]
    assert len(wider) == 2 * cfg.gat_layers + cfg.pred_gat_layers                   # encoder, target, predictor
    assert all(k.endswith("gat.lin_edge.weight") for k in wider)
    for k, value in plain.items():
        if k in wider:
            assert state[k].shape == (value.size(0), cfg.edge_dim + 2) and torch.equal(state[k][:, :-2], value)
            assert 0 < float(state[k][:, -2:].abs().max()) <= float(value.abs().max())
        else:
            assert torch.equal(state[k], value), k
    first, second = (state[f"encoder.square_gats.{i}.gat.lin_edge.weight"][:, -2:] for i in (0, 1))
    assert not torch.equal(first, second), "each layer draws its own"
    assert torch.equal(state["target.square_gats.0.gat.lin_edge.weight"], state["encoder.square_gats.0.gat.lin_edge.weight"])
    assert not torch.equal(state["mask_predictor.gats.0.gat.lin_edge.weight"][:, -2:], first)
    torch.manual_seed(1)
    other = lje.LabyrinthEncoder(small_config("squares", orientation_edges=True))
    assert not torch.equal(other.square_gats[0].gat.lin_edge.weight[:, -2:], first)  # they follow the seed

    def gaps(model, mask=None):
        out = model.encoder.eval()(graphs, mask)
        return (float((out.z[0] - out.z[1]).norm() / out.z[0].norm()),
                float((out.squares[0] - out.squares[1][flip]).norm() / out.squares[0].norm()))

    with torch.no_grad():
        assert max(gaps(models["plain"])) < 1e-5, "without marks, mirror images encode alike"
        z_gap, token_gap = gaps(models["edges"])
        assert z_gap > 1e-4 and token_gap > 1e-3, (z_gap, token_gap)
        hidden = torch.rand(128, generator=torch.Generator().manual_seed(0)) < 0.5
        hidden[64:] = hidden[:64][flip]                                             # the mirrored squares, hidden
        z_gap, token_gap = gaps(models["edges"], hidden)
        assert z_gap > 1e-4 and token_gap > 1e-3, (z_gap, token_gap)
    model = models["edges"]
    model.losses(make_batch(cfg), rng=torch.Generator().manual_seed(0)).total.backward()
    for block in (model.encoder.square_gats[0], model.mask_predictor.gats[0]):
        assert float(block.gat.lin_edge.weight.grad[:, -2:].abs().sum()) > 0

    # the concatenating layers and their summed branch read the marks too; the unpooling layers do not
    kw = dict(gat_style="concat", gat_sum=True, transformer_layers=0, readout="mean")
    torch.manual_seed(0)
    concat = lje.LabyrinthJEPA(small_config("squares", orientation_edges=True, **kw))
    torch.manual_seed(0)
    base = lje.LabyrinthJEPA(small_config("squares", **kw))
    for marked_block, block in zip(concat.encoder.square_gats, base.encoder.square_gats):
        for a, b in ((marked_block.gat.lin_edge.weight, block.gat.lin_edge.weight), (marked_block.edge.weight, block.edge.weight)):
            assert a.shape == (b.size(0), b.size(1) + 2) and torch.equal(a[:, :-2], b)
        assert torch.equal(marked_block.mlp[0].weight, block.mlp[0].weight)
    with torch.no_grad():
        assert max(gaps(base)) < 1e-5 and min(gaps(concat)) > 1e-4
    assert torch.isfinite(concat.losses(make_batch(concat.config), rng=torch.Generator().manual_seed(0)).total)
    unpool_cfg = small_config("unpool", orientation_edges=True)
    assert unpool_cfg.edge_dim == unpool_cfg.num_edge_types and unpool_cfg.mark_dim == 2
    assert torch.isfinite(lje.LabyrinthJEPA(unpool_cfg).losses(make_batch(unpool_cfg),
                                                             rng=torch.Generator().manual_seed(0)).total)


def test_mark_init_scale_enlarges_only_the_encoders_mark_weights():
    """
    mark_init_scale multiplies the encoder's starting weights on the left / right marks: the same draw, made
    larger, in both layer styles and the summed branch. The masked-square predictor's mark weights and every
    other weight stay those of a run without it, and the encoder tells a position from its mirror image by more.
    """
    fen = FENS[3]
    mirror = ld.normalize_fen(chess.Board(fen).transform(chess.flip_horizontal).fen(en_passant="xfen"))
    graphs = ld.pack_views([ld.build_views(f, torch.zeros(64, dtype=torch.bool))[0] for f in (fen, mirror)])
    for kw in (dict(), dict(gat_style="concat", gat_residual=True, gat_sum=True, transformer_layers=0, readout="mean")):
        models = {}
        for scale in (1.0, 4.0):
            torch.manual_seed(0)
            models[scale] = lje.LabyrinthJEPA(small_config("squares", orientation_edges=True, mark_init_scale=scale, **kw))
        plain, state = models[1.0].state_dict(), models[4.0].state_dict()
        scaled = [k for k in plain if k.startswith(("encoder.square_gats", "target.square_gats"))
                  and k.endswith(("gat.lin_edge.weight", ".edge.weight"))]
        assert len(scaled) == (2 if kw else 1) * 2 * models[1.0].config.gat_layers
        for k, value in plain.items():
            if k in scaled:
                assert torch.equal(state[k][:, :-2], value[:, :-2]) and torch.allclose(state[k][:, -2:], 4 * value[:, -2:]), k
            else:
                assert torch.equal(state[k], value), k
        with torch.no_grad():
            gap = {scale: float((lambda z: (z[0] - z[1]).norm() / z[0].norm())(m.encoder.eval()(graphs).z))
                   for scale, m in models.items()}
        assert gap[4.0] > gap[1.0] > 1e-5, gap
    try:
        small_config("squares", orientation_edges=True, mark_init_scale=0.0).validate()
        raise AssertionError("a scale of 0 should be rejected")
    except ValueError:
        pass


def test_sum_branch_keeps_the_other_weights_and_counts_neighbours():
    """
    gat_sum gives every square GAT layer a branch that sums the neighbours' messages. The heads average, so a
    square hears one neighbour and two identical ones alike; with the branch it does not. The branch's weights
    come from their own generators, so every other weight, and the random numbers afterwards, are those of a
    run without it: in both layer styles, with and without the left / right edge marks.
    """
    x = torch.randn(64, 8)
    x[2] = x[1]
    loops = torch.arange(1, 64).repeat(2, 1)                                        # the other squares hear themselves
    one = torch.cat([loops, torch.tensor([[1], [0]])], 1)                           # square 0 hears square 1
    two = torch.cat([loops, torch.tensor([[1, 2], [0, 0]])], 1)                     # ... and its twin, square 2
    for summed in (False, True):
        torch.manual_seed(0)
        block = lje._GATBlock(8, 2, edge_dim=3, dropout=0.0, add_self_loops=False, summed=summed)
        with torch.no_grad():
            heard_one = block(x, one, torch.ones(one.size(1), 3))[0]
            heard_two = block(x, two, torch.ones(two.size(1), 3))[0]
        assert (float((heard_one - heard_two).abs().max()) > 1e-4) == summed, summed
    messages = lje._summed_messages(x, two, torch.ones(two.size(1), 3), torch.nn.Linear(3, 8))
    alone = lje._summed_messages(x, one, torch.ones(one.size(1), 3), lambda attr: 0.0)
    assert messages.shape == x.shape and float(alone[0].abs().max()) > 0
    both = lje._summed_messages(x, two, torch.ones(two.size(1), 3), lambda attr: 0.0)
    assert torch.allclose(both[0], 2 * alone[0]) and torch.equal(both[1:], alone[1:])

    def build(seed=0, **kw):
        torch.manual_seed(seed)
        model = lje.LabyrinthJEPA(small_config("squares", transformer_layers=0, readout="mean", **kw))
        return model, model.state_dict(), torch.rand(4)

    fen = FENS[3]                                                                   # no castling rights
    mirror = ld.normalize_fen(chess.Board(fen).transform(chess.flip_horizontal).fen(en_passant="xfen"))
    graphs = ld.pack_views([ld.build_views(f, torch.zeros(64, dtype=torch.bool))[0] for f in (fen, mirror)])
    for style in (dict(), dict(gat_style="concat", gat_residual=True)):
        concat = bool(style)
        own = ("edge.weight", "edge.bias") + (() if concat else ("sum_out.weight",))
        states = {}
        for marks in (dict(), dict(orientation_edges=True)):
            _, base, base_after = build(**style, **marks)
            model, state, after = build(gat_sum=True, **style, **marks)
            states[bool(marks)] = state
            cfg = model.config
            assert torch.equal(base_after, after), "the global random stream is left alone"
            added = sorted(set(state) - set(base))
            assert set(base) <= set(state) and len(added) == 2 * cfg.gat_layers * len(own)    # encoder and target
            assert all(k.split(".")[1] == "square_gats" and k.endswith(own) for k in added), added
            for k, value in base.items():
                if concat and k.endswith("mlp.0.weight") and ".square_gats." in k:
                    width = state[k].size(1) - value.size(1)                        # the sum's columns come last
                    assert width > 0 and torch.equal(state[k][:, :-width], value), k
                    assert 0 < float(state[k][:, -width:].abs().max()) <= float(value.abs().max())
                elif concat and ".square_gats." in k and k.split(".")[-2] == "norm":
                    assert torch.equal(state[k][:value.numel()], value), k           # the LayerNorm over the parts
                else:
                    assert torch.equal(state[k], value), k                          # the marks' weights too
            for k in added:
                assert float(state[k].abs().max()) > 0
                assert torch.equal(state[k], state["target." + k.split(".", 1)[1]]), "the target starts as a copy"
            _, other, _ = build(seed=1, gat_sum=True, **style, **marks)
            assert not torch.equal(other["encoder.square_gats.0.edge.weight"], state["encoder.square_gats.0.edge.weight"])
            if not concat:                                                          # each layer draws its own
                assert not torch.equal(state["encoder.square_gats.0.edge.weight"], state["encoder.square_gats.1.edge.weight"])
                assert not torch.equal(state["encoder.square_gats.0.sum_out.weight"], state["encoder.square_gats.1.sum_out.weight"])
            with torch.no_grad():
                out = model.encoder.eval()(graphs)
                gap = float((out.z[0] - out.z[1]).norm() / out.z[0].norm())
            assert (gap > 1e-4) == bool(marks), (style, marks, gap)                 # the sum alone is mirror-blind
            model.train().losses(make_batch(cfg), rng=torch.Generator().manual_seed(0)).total.backward()
            for block in model.encoder.square_gats:
                assert float(block.edge.weight.grad.abs().sum()) > 0
                if concat:
                    assert float(block.mlp[0].weight.grad[:, -block.edge.out_features:].abs().sum()) > 0
                else:
                    assert float(block.sum_out.weight.grad.abs().sum()) > 0
        # the marks leave the branch's other weights alone too
        for k, value in states[False].items():
            marked = states[True][k]
            if k.endswith(("lin_edge.weight", "edge.weight")) and ".square_gats." in k or ".gats." in k and k.endswith("lin_edge.weight"):
                assert marked.shape == (value.size(0), value.size(1) + 2) and torch.equal(marked[:, :-2], value), k
            else:
                assert torch.equal(marked, value), k

    # hidden squares stay hidden: the branch reads the tokens, where a hidden square is the mask token
    torch.manual_seed(0)
    model = lje.LabyrinthJEPA(small_config("squares", gat_sum=True, orientation_edges=True)).eval()
    board = chess.Board(FENS[0], chess960=True)
    mask = torch.zeros(64, dtype=torch.bool)
    orient = (lambda sq: sq) if board.turn == chess.WHITE else (lambda sq: sq ^ 56)
    other = board.copy(stack=False)
    hidden = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None and board.piece_type_at(sq) != chess.KING][:6]
    for sq in hidden:
        mask[orient(sq)] = True
        other.remove_piece_at(sq)
    outs = []
    for f in (ld.fen_of(board), ld.fen_of(other)):
        ctx = ld.pack_views([ld.build_views(f, mask)[1]])
        with torch.no_grad():
            enc = model.encoder(ctx, mask=mask)
            outs.append((enc.z, enc.squares, model.mask_predictor(enc.squares, ctx, mask)))
    assert all(torch.equal(a, b) for a, b in zip(*outs))
    # the unpooling variant takes the branch in its square layers
    cfg = small_config("unpool", gat_sum=True)
    assert torch.isfinite(lje.LabyrinthJEPA(cfg).losses(make_batch(cfg), rng=torch.Generator().manual_seed(0)).total)


def test_concatenated_gat_layers_and_the_mean_max_readout():
    """
    gat_style "concat": full-width heads side by side with the square's own features, reduced by a per-square
    MLP to a wider width, layer by layer. The heads average, so a square hears one neighbour and two identical
    ones alike; gat_sum's branch tells them apart. Neither, nor the mean_max readout, tells left from right.
    The heads, computed densely, are GATConv's, values and gradients.
    """
    x = torch.randn(64, 8)
    x[2] = x[1]
    loops = torch.arange(1, 64).repeat(2, 1)                                        # the other squares hear themselves
    one = torch.cat([loops, torch.tensor([[1], [0]])], 1)                           # square 0 hears square 1
    two = torch.cat([loops, torch.tensor([[1, 2], [0, 0]])], 1)                     # ... and its twin, square 2
    for summed in (False, True):
        torch.manual_seed(0)
        block = lje._ConcatGATBlock(8, 12, heads=2, edge_dim=3, dropout=0.0, summed=summed)
        with torch.no_grad():
            heard_one = block(x, one, torch.ones(one.size(1), 3))[0]
            heard_two = block(x, two, torch.ones(two.size(1), 3))[0]
        assert heard_one.shape == (12,)
        assert (float((heard_one - heard_two).abs().max()) > 1e-4) == summed, summed
    graphs = ld.pack_views([ld.build_views(f, torch.zeros(64, dtype=torch.bool))[0] for f in FENS[:3]])
    attr = lje.with_orientation_anchor(graphs.edge_index, graphs.edge_attr)
    torch.manual_seed(0)
    block = lje._ConcatGATBlock(16, 24, heads=3, edge_dim=attr.size(1), dropout=0.0, summed=False)
    h = torch.randn(3 * 64, 16, requires_grad=True)
    dense, sparse = block._attend(h, graphs.edge_index, attr), block.gat(h, graphs.edge_index, attr)
    assert float((dense - sparse).detach().abs().max()) < 1e-5
    weights = torch.randn_like(dense)
    grads = [torch.autograd.grad((out * weights).sum(), [h] + list(block.gat.parameters())) for out in (dense, sparse)]
    assert all(float((a - b).abs().max()) < 1e-4 for a, b in zip(*grads))
    torch.manual_seed(0)
    plain = lje._ConcatGATBlock(16, 24, heads=3, edge_dim=attr.size(1), dropout=0.0, summed=True)
    torch.manual_seed(0)
    residual = lje._ConcatGATBlock(16, 24, heads=3, edge_dim=attr.size(1), dropout=0.0, summed=True, residual=True)
    with torch.no_grad():
        assert plain.state_dict().keys() == residual.state_dict().keys()            # no weights of its own
        skip = residual(h, graphs.edge_index, attr) - plain(h, graphs.edge_index, attr)
        assert float((skip[:, :16] - h).abs().max()) < 1e-5 and float(skip[:, 16:].abs().max()) < 1e-5
    fen = FENS[3]                                                                   # no castling rights
    mirror = ld.normalize_fen(chess.Board(fen).transform(chess.flip_horizontal).fen(en_passant="xfen"))
    graphs = ld.pack_views([ld.build_views(f, torch.zeros(64, dtype=torch.bool))[0] for f in (fen, mirror)])
    flip = torch.tensor([s ^ 7 for s in range(64)])                                 # a <-> h, ranks kept
    for kw in (dict(readout="mean"), dict(gat_sum=True, gat_residual=True, readout="mean_max", readout_hidden=48)):
        cfg = small_config("squares", gat_style="concat", transformer_layers=0, **kw)
        torch.manual_seed(0)
        model = lje.LabyrinthJEPA(cfg)
        assert cfg.gat_widths == [32, 48, 72]
        assert [block.mlp[-1].out_features for block in model.encoder.square_gats] == [48, 72]
        with torch.no_grad():
            out = model.encoder.eval()(graphs)
        assert out.squares.shape == (2, 64, cfg.d_model) and out.z.shape == (2, cfg.z_dim)
        assert float((out.z[0] - out.z[1]).norm() / out.z[0].norm()) < 1e-5
        assert float((out.squares[0] - out.squares[1][flip]).norm() / out.squares[0].norm()) < 1e-5
        model.train().losses(make_batch(cfg), rng=torch.Generator().manual_seed(0)).total.backward()
        for part in (model.encoder.square_gats, model.encoder.readout_proj):
            assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in part.parameters()), kw
    for variant, bad in (("squares", dict(gat_style="wide")), ("unpool", dict(gat_style="concat")),
                         ("squares", dict(gat_style="concat", gat_growth=0.5)), ("squares", dict(gat_residual=True)),
                         ("squares", dict(readout="max")), ("squares", dict(readout_hidden=-1))):
        try:
            small_config(variant, **bad).validate()
            raise AssertionError(f"{bad} should be rejected")
        except ValueError:
            pass


def test_move_encodings_and_the_move_predictor():
    """
    Moves named by squares (not tokens) make the next-position predictor tell a position from its mirror
    image, which the tokens never do; the move predictor names the first move's squares from z.
    """
    fen = FENS[3]                                                                   # no castling rights
    mirror = ld.normalize_fen(chess.Board(fen).transform(chess.flip_horizontal).fen(en_passant="xfen"))
    first = ld.collate_jepa([ld.build_sample(fen, random.Random(4))]).rollout[0]
    flip = lambda nodes: 64 + ((nodes % 64) ^ 7)                                   # a <-> h, into graph 1
    both = ld.RolloutStep(None, torch.tensor([0, 1]), torch.tensor([0, 1]),
                          torch.cat([first.move_from, flip(first.move_from)]), torch.cat([first.move_to, flip(first.move_to)]),
                          first.promotion.repeat(2), first.flags.repeat(2, 1), first.piece.repeat(2))
    graphs = ld.pack_views([ld.build_views(f, torch.zeros(64, dtype=torch.bool))[0] for f in (fen, mirror)])
    for encoding in lje.MOVE_ENCODINGS:
        for piece in (False, True):
            cfg = small_config("squares", move_encoding=encoding, move_piece=piece)
            torch.manual_seed(0)
            model = lje.LabyrinthJEPA(cfg)
            losses = model.losses(make_batch(cfg), rng=torch.Generator().manual_seed(0))
            assert torch.isfinite(losses.total)
            losses.total.backward()
            if encoding != "tokens":
                assert all(e.weight.grad is not None for e in model.next_predictor.names)
            assert (model.next_predictor.piece is not None) == piece
            if piece:
                assert model.next_predictor.piece.weight.grad is not None
            with torch.no_grad():
                out = model.encoder.eval()(graphs)
                pred = model.predict_rollout(out.z, out.squares, [both])[0]
                gap = float((pred[0] - pred[1]).norm() / pred[0].norm())
                if encoding == "tokens":                # the same z, mirrored tokens: the same prediction
                    assert gap < 1e-5, gap
                    continue
                assert gap > 1e-3, (encoding, gap)      # "b1 to c3" is not "g1 to f3"
                # named squares never read the square tokens
                noisy = model.predict_rollout(out.z, torch.randn_like(out.squares), [both])[0]
                assert torch.equal(pred, noisy)
    # the move predictor: a 64-way choice for each square, trained through z
    cfg = small_config("squares", inverse_weight=1.0)
    model = lje.LabyrinthJEPA(cfg)
    losses = model.losses(make_batch(cfg), rng=torch.Generator().manual_seed(0))
    assert 2.0 < losses.stats["loss_inverse"] < 8.0 and 0.0 <= losses.stats["inverse_accuracy"] <= 1.0
    losses.terms["inverse"].backward()
    assert all(p.grad is not None for p in model.move_predictor.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.encoder.readout_proj.parameters())
    assert lje.LabyrinthJEPA(small_config("squares")).move_predictor is None
    for bad in (dict(move_encoding="names"), dict(inverse_weight=-1.0), dict(move_encoding="rank_file", d_model=33)):
        try:
            small_config("squares", **bad).validate()
            raise AssertionError(f"{bad} should be rejected")
        except ValueError:
            pass


def test_dehb_search_space_samples_build():
    cs = lje.build_jepa_configspace(seed=1)
    seen = set()
    for sample in cs.sample_configuration(12):
        values = dict(sample)
        cfg = lje.JEPAConfig.from_dict(values)
        cfg.validate()
        seen.add(cfg.token_variant)
        if cfg.token_variant != "unpool":
            assert "unpool_layers" not in values and "rl_coef" not in values
        if cfg.token_variant != "registers":
            assert "num_registers" not in values
        assert cfg.z_dim == lje.JEPAConfig().z_dim, "z_dim must stay fixed during a search"
    assert len(seen) >= 2
    # a small sample from the space actually runs
    for sample in cs.sample_configuration(3):
        cfg = lje.JEPAConfig.from_dict({**dict(sample), "unpool_layers": 1, "z_dim": 64, "next_hidden": 64,
                                        "d_model": 32, "readout_queries": 4})
        model = lje.LabyrinthJEPA(cfg)
        assert torch.isfinite(model.losses(make_batch(cfg, FENS[:2]), rng=torch.Generator().manual_seed(0)).total)


def test_next_predictor_can_predict_the_next_z_outright():
    """next_residual = False drops the "+ z": same weights, and the two predictions differ by exactly z."""
    outputs = {}
    for residual in (True, False):
        cfg = small_config("squares", next_residual=residual)
        torch.manual_seed(0)
        model = lje.LabyrinthJEPA(cfg).eval()
        outputs[residual] = model.state_dict()
        batch = make_batch(cfg)
        with torch.no_grad():
            ctx = model.encoder(batch.full)
            outputs[residual, "z"] = ctx.z
            outputs[residual, "pred"] = model.predict_rollout(ctx.z, ctx.squares, batch.rollout[:1])[0]
    assert all(torch.equal(v, outputs[False][k]) for k, v in outputs[True].items())
    parent = batch.rollout[0].parent
    assert float((outputs[True, "pred"] - outputs[False, "pred"] - outputs[True, "z"][parent]).abs().max()) < 1e-5


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
