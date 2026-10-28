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
             ("squares", dict(zpred_weight=0.5, projector_layers=2, projector_hidden=32, projector_dim=16))]
    for variant, kw in cases:
        cfg = small_config(variant, **kw)
        torch.manual_seed(0)
        out = lje.LabyrinthJEPA(cfg).losses(make_batch(cfg), rng=torch.Generator().manual_seed(0))
        assert abs(float(sum(out.terms.values()) - out.total)) < 1e-5, (variant, kw)
        expected = {"mask", "next"}
        if cfg.anticollapse != "none":
            expected |= {"anticollapse_masked", "anticollapse_full"}
            expected |= {"anticollapse_squares"} if cfg.anticollapse_squares else set()
            expected |= {"decorrelation"} if cfg.decorrelation_weight > 0 else set()
            assert all(f"anticollapse_{n}" in out.stats for n in ("masked", "full"))
        expected |= {"rl"} if cfg.uses_unpool and cfg.unpool_rl else set()
        expected |= {"zpred"} if cfg.zpred_weight > 0 else set()
        assert set(out.terms) == expected, (variant, kw, sorted(out.terms))


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
