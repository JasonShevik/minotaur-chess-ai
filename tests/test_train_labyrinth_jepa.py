"""
Tests for the JEPA training script, train_labyrinth_jepa.py.

Run with plain Python (no pytest needed):

    python tests/test_train_labyrinth_jepa.py

Tiny models on a tiny position cache built in a temporary folder, on the CPU. They check that a run
trains, evaluates, checkpoints and resumes; that the schedules and weight-decay groups are right; that
the memory guard trips; that the exam grades every run alike; that a run or a search can be stopped
and taken up again; and that the search's objective scores runs and handles runs that do not fit.
"""
import csv
import json
import math
import os
import random
import sys
import tempfile
import traceback
from dataclasses import replace
from typing import Callable, List

import chess
import torch

import repo_root  # noqa: F401  (first: it puts the repo's root folder, where the code under test is, on sys.path)
import labyrinth_data as ld
import labyrinth_jepa_encoder as lje
import piece_reach as pr
import probe_labyrinth as pl
import reach_labeler as rl
import train_labyrinth_jepa as tj


def tiny_config(variant: str = "unpool", **kw) -> lje.JEPAConfig:
    base = dict(token_variant=variant, gat_dim=16, gat_heads=2, gat_layers=2, num_registers=4, unpool_layers=1,
                unpool_hidden=16, unpool_edge_dim=4, d_model=16, transformer_layers=1, transformer_heads=2,
                readout_queries=2, z_dim=16, pred_dim=16, pred_gat_layers=1, pred_heads=2, move_dim=16,
                next_hidden=32, next_layers=1, lr=1e-3)
    base.update(kw)
    return lje.JEPAConfig(**base)


def make_cache(folder: str, n_starts: int = 30) -> str:
    rnd = random.Random(0)
    fens = []
    for i in range(n_starts):
        board = chess.Board() if i == 0 else chess.Board.from_chess960_pos(rnd.randrange(960))
        for _ in range(8):
            fens.append(board.fen())
            for _ in range(rnd.randint(1, 5)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rnd.choice(moves))
    path = os.path.join(folder, "fens.txt")
    with open(path, "w") as f:
        f.write("\n".join(fens))
    cache = os.path.join(folder, "cache")
    ld.build_position_cache_from_file(path, cache, val_fraction=0.2, test_fraction=0.0, verbose=False)
    return cache


def tiny_run(cache: str, out_dir, **kw) -> tj.RunConfig:
    base = dict(cache_dir=cache, out_dir=out_dir, total_steps=6, batch_size=8, warmup_steps=2, num_workers=0,
                log_every=1, eval_every=3, eval_positions=24, eval_samples=2, val_loss_batches=1,
                checkpoint_every=3, device="cpu")
    base.update(kw)
    return tj.RunConfig(**base)


def quiet(_msg: str) -> None:
    pass


# ##### ##### ##### ##### #####
#   Tests


def test_a_run_trains_evaluates_checkpoints_and_resumes():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = make_cache(tmp)
        out = os.path.join(tmp, "run")
        result = tj.train(tiny_config(), tiny_run(cache, out, grad_terms_every=2, snapshot_first=2), log=quiet)
        assert result["steps"] == 6
        steps = [h for h in result["history"] if "loss" in h]
        evals = [h for h in result["history"] if "eval" in h]
        assert [h["step"] for h in steps] == [1, 2, 3, 4, 5, 6]
        assert [h["step"] for h in evals] == [3, 6]
        assert all(v == v for h in evals for v in h["eval"].values()), "a metric is NaN"
        assert {"lidar", "rankme", "self_consistency", "val_loss_mask", "val_loss_next"} <= set(evals[-1]["eval"])
        for name in ("last.pt", "best.pt", "history.jsonl", "run_config.json"):
            assert os.path.exists(os.path.join(out, name)), name
        ckpt = torch.load(os.path.join(out, "last.pt"), weights_only=False)
        assert ckpt["step"] == 6 and ckpt["jepa_config"]["token_variant"] == "unpool"

        # the target moved by the moving average, the online encoder by the optimizer
        model = lje.LabyrinthJEPA(tiny_config())
        model.load_state_dict(ckpt["model"])
        assert any(not torch.equal(a, b) for a, b in zip(model.encoder.parameters(), model.target.parameters()))

        # per-term gradients: the masked-square loss never reaches the readout, the regularizers and the
        # next-position loss do
        grads = [h for h in result["history"] if "grad_terms" in h]
        assert [h["step"] for h in grads] == [2, 4, 6]
        norms = grads[-1]["grad_terms"]["norms"]
        assert {"mask", "next", "anticollapse_masked", "anticollapse_full", "rl", "total"} <= set(norms)
        assert "readout" not in norms["mask"] and norms["mask"]["transformer"] > 0
        assert norms["next"]["readout"] > 0 and norms["anticollapse_masked"]["readout"] > 0
        assert norms["total"]["mask_predictor"] > 0 and "mask_predictor" not in norms["next"]
        assert all(-1 - 1e-6 <= c <= 1 + 1e-6 for pair in grads[-1]["grad_terms"]["cosine"].values()
                   for c in pair.values())
        assert "readout" not in grads[-1]["grad_terms"]["cosine"]["mask|next"]

        # probes: the baselines once, then the target encoder at every evaluation, every view
        baselines = [h for h in result["history"] if "probe_baselines" in h]
        assert len(baselines) == 1 and set(baselines[0]["probe_baselines"]) == {"untrained", "board_onehot"}
        probes = [h for h in result["history"] if "probe" in h]
        assert [h["step"] for h in probes] == [3, 6] and result["probe"] == probes[-1]["probe"]
        assert set(probes[-1]["probe"]) == {"z", "token_mean", "pre_transformer_mean"}
        assert "val/material_balance" in probes[-1]["probe"]["z"]
        with open(os.path.join(out, "probes.csv")) as f:
            rows = list(csv.DictReader(f))
        assert {(r["step"], r["model"]) for r in rows} == {("0", "untrained"), ("0", "board_onehot"),
                                                           ("3", "trained"), ("6", "trained")}

        # snapshots at step 0, 2, 4 and the end; step 0 is the starting point the untrained baseline rebuilds
        snaps = os.path.join(out, "snapshots")
        assert sorted(os.listdir(snaps)) == [f"step_{s:07d}.pt" for s in (0, 2, 4, 6)]
        torch.manual_seed(0)
        start = lje.LabyrinthEncoder(tiny_config())
        first = pl.load_encoder(os.path.join(snaps, "step_0000000.pt"), "cpu")
        assert all(torch.equal(a, b) for a, b in zip(start.parameters(), first.parameters()))
        assert not torch.equal(pl.load_encoder(os.path.join(snaps, "step_0000006.pt"), "cpu").queries, first.queries)

        # resuming continues from step 6 to step 9
        resumed = tj.train(tiny_config(), tiny_run(cache, out, total_steps=9, grad_terms_every=2, snapshot_first=2),
                           log=quiet)
        assert resumed["steps"] == 9
        assert [h["step"] for h in resumed["history"] if "loss" in h] == [7, 8, 9]
        with open(os.path.join(out, "history.jsonl")) as f:
            lines = [json.loads(l) for l in f]
        assert [h["step"] for h in lines if "loss" in h] == list(range(1, 10))
        assert not any("probe_baselines" in h for h in resumed["history"]), "baselines are not probed again"
        assert sorted(os.listdir(snaps))[-2:] == ["step_0000008.pt", "step_0000009.pt"]


def test_rollout_curriculum_grows_during_a_run():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = make_cache(tmp)
        cfg = tiny_config("registers", rollout_max_steps=3, rollout_stage_positions=16, rollout_ramp_fraction=0.5)
        messages = []
        out = os.path.join(tmp, "run")
        result = tj.train(cfg, tiny_run(cache, out, total_steps=7, eval_every=0), log=messages.append)
        lengths = [len(h["rollout_weights"]) for h in result["history"] if "loss" in h]
        # 8 positions per step, 16 per stage: one move ahead for steps 1-2, two for 3-4, three from step 5
        assert lengths == [1, 1, 2, 2, 3, 3, 3], lengths
        assert sum("now predicting" in m for m in messages) == 2
        # resuming picks the curriculum up where it was
        resumed = tj.train(cfg, tiny_run(cache, out, total_steps=9, eval_every=0), log=messages.append)
        assert [len(h["rollout_weights"]) for h in resumed["history"] if "loss" in h] == [3, 3]


def test_long_sparse_rollouts_whole_board_term_and_projector_train():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = make_cache(tmp)
        cfg = tiny_config("squares", rollout_start_steps=10, rollout_max_steps=12, rollout_stage_positions=8,
                          rollout_targets="log", zpred_weight=1.0, projector_layers=2, projector_hidden=32,
                          projector_dim=16, anticollapse_views="full")
        result = tj.train(cfg, tiny_run(cache, None, total_steps=6, eval_every=3, grad_terms_every=3), log=quiet)
        steps = [h for h in result["history"] if "loss" in h]
        assert [len(h["rollout_weights"]) for h in steps] == [10, 11, 12, 12, 12, 12]
        assert all("loss_zpred" in h for h in steps)
        assert "loss_next_6" not in steps[-1] and "loss_next_10" not in steps[-1]       # ungraded steps
        assert {f"loss_next_{k}" for k in (1, 2, 3, 8, 9)} <= set(steps[-1])
        assert "val_loss_zpred" in [h for h in result["history"] if "eval" in h][-1]["eval"]
        norms = [h for h in result["history"] if "grad_terms" in h][-1]["grad_terms"]["norms"]
        assert set(norms) == {"mask", "next", "zpred", "anticollapse_full", "total"}
        assert "projector" in norms["anticollapse_full"] and norms["zpred"]["readout"] > 0


def test_every_variant_trains():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = make_cache(tmp)
        for variant in lje.TOKEN_VARIANTS:
            result = tj.train(tiny_config(variant), tiny_run(cache, None, total_steps=3, eval_every=0), log=quiet)
            assert result["steps"] == 3 and result["final"]["lidar"] >= 1.0 - 1e-6, variant


def test_schedules_and_weight_decay_groups():
    # warm-up rises linearly to the peak, cosine decays to min_fraction at the end
    assert abs(tj.lr_factor(0, 1000, 100, 0.01) - 0.01) < 1e-9
    assert abs(tj.lr_factor(99, 1000, 100, 0.01) - 1.0) < 1e-9
    assert abs(tj.lr_factor(1000, 1000, 100, 0.01) - 0.01) < 1e-9
    assert tj.lr_factor(550, 1000, 100, 0.01) < tj.lr_factor(300, 1000, 100, 0.01)
    # min_fraction 1: the rate stays at its peak after the warm-up
    assert all(abs(tj.lr_factor(s, 1000, 100, 1.0) - 1.0) < 1e-9 for s in (99, 400, 1000))
    # warm-up is capped at a tenth of a short run
    assert abs(tj.lr_factor(9, 100, 1000, 0.0) - 1.0) < 1e-9
    model = lje.LabyrinthJEPA(tiny_config("registers"))
    decay, no_decay = tj.parameter_groups(model, 0.05)
    names = {id(p): n for n, p in model.named_parameters()}
    no_decay_names = [names[id(p)] for p in no_decay["params"]]
    decay_names = [names[id(p)] for p in decay["params"]]
    assert not any(n.startswith("target.") for n in decay_names + no_decay_names)
    assert all(model.get_parameter(n).ndim >= 2 for n in decay_names)
    for token in ("encoder.mask_token", "encoder.token_type", "encoder.registers", "encoder.queries"):
        assert token in no_decay_names, token
    assert len(decay_names) + len(no_decay_names) == len(model.trainable_parameters())


def test_evaluations_use_one_random_draw_of_the_validation_split():
    run = tj.RunConfig(eval_positions=50, val_loss_batches=2)
    eval_rows, val_rows = tj.evaluation_positions(run, 1000, 16)
    assert len(eval_rows) == 50 and len(set(eval_rows)) == 50 and val_rows == eval_rows[:32]
    assert max(eval_rows) > 100, "not the split's first rows"
    assert tj.evaluation_positions(replace(run, seed=5), 1000, 16) == (eval_rows, val_rows)     # the same in every run
    assert tj.evaluation_positions(replace(run, eval_subset="first"), 1000, 16) == (list(range(50)), list(range(32)))
    small_eval, small_val = tj.evaluation_positions(run, 20, 16)                                # a split smaller than asked
    assert sorted(small_eval) == list(range(20)) and len(small_val) == 20
    try:
        tj.evaluation_positions(replace(run, eval_subset="last"), 1000, 16)
        raise AssertionError("an unknown eval_subset should be rejected")
    except ValueError:
        pass


def test_parameter_parts_and_snapshot_steps():
    for variant in lje.TOKEN_VARIANTS:
        parts = {n: tj.parameter_part(n) for n, _ in lje.LabyrinthJEPA(tiny_config(variant)).named_parameters()
                 if not n.startswith("target.")}
        assert all(p in tj.ENCODER_PARTS for n, p in parts.items() if n.startswith("encoder.")), variant
        assert set(tj.ENCODER_PARTS) <= set(parts.values())
    assert tj.parameter_part("encoder.queries") == "readout" == tj.parameter_part("encoder.readout_proj.weight")
    assert tj.parameter_part("encoder.registers") == "transformer" == tj.parameter_part("encoder.final_norm.bias")
    assert tj.parameter_part("encoder.square_gats.0.conv.lin.weight") == "square_stage"
    assert tj.parameter_part("next_predictor.step.weight") == "next_predictor"
    assert [s for s in range(5000) if tj.snapshot_due(s, 250)] == [0, 250, 500, 1000, 2000, 4000]
    assert not any(tj.snapshot_due(s, 0) for s in range(10))


def test_memory_guard():
    tj.check_memory_budget(8 * 2**30, 16 * 2**30, 0.9)
    try:
        tj.check_memory_budget(15 * 2**30, 16 * 2**30, 0.9)
    except tj.MemoryBudgetExceeded:
        pass
    else:
        raise AssertionError("15 of 16 GB must exceed a 90% budget")


def test_a_run_with_the_legal_threat_head_reads_the_labellers_database():
    """
    The labeller walks the positions a run will see, in its order, with its rollout moves. So a run on a
    labelled split finds every training position's label in the database, and every rollout position's.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = make_cache(tmp)
        db = os.path.join(tmp, "labels.db")
        labelled = rl.label(db_path=db, cache_dir=cache, split="train", depth=2, workers=0, order_seed=0,
                            rollout_plies=2, kind="threats", log=quiet)
        cfg = tiny_config("squares", transformer_layers=0, move_encoding="rank_file", rollout_start_steps=3,
                          rollout_max_steps=3, zhead_fit="online", zthreat_next_weight=1.0, zthreat_masked_weight=1.0,
                          zthreat_plies=2, zthreat_mode="count", zthreat_head="bilinear", zthreat_dim=8,
                          zthreat_start_plies=0, zthreat_stage_positions=16, rollout_ramp_fraction=0.5)
        messages = []
        run = tiny_run(cache, os.path.join(tmp, "run"), total_steps=6, eval_every=0, probe_positions=0, threat_db=db,
                       threat_compute=False, grad_terms_every=3)
        result = tj.train(cfg, run, log=messages.append)
        steps = [h for h in result["history"] if "loss" in h]
        assert len(steps) == 6 and labelled["done"] >= 6 * 8
        assert any(f"the database holds the first {labelled['done']:,} samples" in m for m in messages)
        for h in steps:
            assert h["threats_stored"] == 1.0 and h["threats_searched"] == 0.0, "every position's label was in the database"
            assert h["zthreat_fit_labelled"] == 1.0 and h["zthreat_next_labelled"] == 1.0, "and every rollout position's"
            assert {"loss_zthreat_fit", "loss_zthreat_next", "loss_zthreat_masked", "zthreat_fit_found"} <= set(h)
            assert "zthreat_next_found_step2" in h and "zthreat_next_found_step3" not in h, "two rollout positions have labels"
        # 8 positions a step, 16 a stage: ply 0 alone for two steps, ply 1 joins at half weight, then ply 2
        assert [[round(w, 6) for w in h["threat_weights"]] for h in steps] == \
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1]], [h["threat_weights"] for h in steps]
        assert "zthreat_fit_found_1" not in steps[0] and "zthreat_fit_found_1" in steps[3] and "zthreat_fit_close_2" in steps[5]
        assert sum("threat curriculum" in m for m in messages) == 2
        assert any("threats real z=" in m and "counts" in m and "cells wrong" in m for m in messages), "the log shows the head"
        norms = [h for h in result["history"] if "grad_terms" in h][-1]["grad_terms"]["norms"]
        assert {"zthreat_fit", "zthreat_next", "zthreat_masked"} <= set(norms) and norms["zthreat_fit"]["encoder"] > 0
        # validation positions are not in the database and nothing may be searched: they go ungraded
        final = result["final"]
        assert final["val_zthreat_fit_labelled"] == 0.0 and final["val_loss_zthreat_fit"] == 0.0

        # with nothing to take labels from, the run says so at once
        was = pr.USING_C
        try:
            pr.use_compiled(False)
            tj.train(cfg, tiny_run(cache, None, total_steps=1, eval_every=0, probe_positions=0,
                                   threat_db=os.path.join(tmp, "absent.db")), log=quiet)
            raise AssertionError("no labels database and no search: the run should refuse")
        except ValueError as error:
            assert "labels" in str(error)
        finally:
            pr.use_compiled(was)
        if pr.USING_C:                                  # the search fills in for the database, validation included
            result = tj.train(cfg, tiny_run(cache, None, total_steps=2, eval_every=0, probe_positions=0, threat_db=None),
                              log=quiet)
            last = [h for h in result["history"] if "loss" in h][-1]
            assert last["threats_stored"] == 0.0 and last["threats_searched"] == 1.0
            assert result["final"]["val_zthreat_fit_labelled"] == 1.0 and result["final"]["val_loss_zthreat_fit"] > 0


def heads_config(**kw) -> lje.JEPAConfig:
    """A tiny concat encoder with the decoder judging its own z, the predicted next z and the predicted whole z."""
    base = dict(gat_style="concat", readout="concat", readout_hidden=32, transformer_layers=0, move_encoding="rank_file",
                zhead_fit="online", zpred_weight=1.0, decode_next_weight=1.0, decode_masked_weight=1.0, decode_hidden=32,
                rollout_start_steps=1, rollout_max_steps=3, rollout_stage_positions=16)
    base.update(kw)
    return tiny_config("squares", **base)


def test_the_exam_grades_the_whole_rollout_whatever_the_curriculum():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = make_cache(tmp)
        exam = dict(exam=True, exam_positions=16, exam_batch_size=8, exam_workers=0, probe_positions=0, eval_every=0)
        # a run whose curriculum never gets past one move is still examined on all three
        stuck = heads_config(rollout_stage_positions=10 ** 9)
        result = tj.train(stuck, tiny_run(cache, None, total_steps=3, **exam), log=quiet)
        assert [len(h["rollout_weights"]) for h in result["history"] if "loss" in h] == [1, 1, 1]
        final = result["final"]
        assert {"exam_loss", "exam_loss_decode_fit", "exam_loss_decode_next", "exam_loss_decode_masked",
                "exam_decode_fit_exact", "exam_decode_next_exact_3", "exam_latent_next"} <= set(final)
        parts = [v for k, v in final.items() if k.startswith("exam_loss_")]
        assert len(parts) == 3 and abs(final["exam_loss"] - sum(parts)) < 1e-6, "the exam loss is the heads' terms"
        assert "exam_loss_mask" not in final and "exam_loss_zpred" not in final, "the latent terms are not in it"
        messages = []
        tj.train(stuck, tiny_run(cache, None, total_steps=1, **exam), log=messages.append)
        assert any("exam  loss=" in m and "boards decoded exactly" in m for m in messages)

        # the exam's positions, masks and moves do not depend on the run's batch size or seed
        torch.manual_seed(0)
        model = lje.LabyrinthJEPA(stuck)
        val = ld.PositionDataset(cache, "val")
        rows = random.Random(tj.EVAL_SUBSET_SEED).sample(range(len(val)), 16)

        def sit(batch: int) -> dict:
            loader = ld.JEPALoader(cache, "val", batch, num_workers=0, mask_config=ld.MaskConfig(), seed=tj.EXAM_SEED,
                                   shuffle=False, indices=rows)
            return tj.exam(model, loader, torch.device("cpu"))
        first, again = sit(8), sit(8)
        assert first == again, "the same model sits the same exam twice"
        assert abs(sit(16)["exam_loss"] - first["exam_loss"]) < 1e-4, "(equal batches: the mean over batches is the mean)"


def test_eager_rollouts_train_like_restarted_loaders():
    """Carrying the whole rollout in every batch and using what the curriculum has reached gives the same
    training as restarting the loader each time the curriculum adds a move."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = make_cache(tmp)
        for cfg in (heads_config(), heads_config(rollout_start_steps=2, rollout_max_steps=6, rollout_targets="log",
                                                 rollout_stage_positions=8)):
            runs = [tj.train(cfg, tiny_run(cache, None, total_steps=6, eval_every=0, probe_positions=0,
                                           grad_terms_every=0, eager_rollouts=eager), log=quiet)
                    for eager in (False, True)]
            losses = [[h["loss"] for h in r["history"] if "loss" in h] for r in runs]
            lengths = [[len(h["rollout_weights"]) for h in r["history"] if "loss" in h] for r in runs]
            assert lengths[0] == lengths[1] and len(set(lengths[0])) > 1, lengths
            assert all(abs(a - b) < 1e-5 for a, b in zip(*losses)), losses
            graded = [sorted(k for k in r["history"][-2] if k.startswith("loss_next_")) for r in runs]
            assert graded[0] == graded[1], "the same steps are graded"


def test_a_stopped_run_saves_and_carries_on():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = make_cache(tmp)
        cfg = heads_config()
        settings = dict(total_steps=6, eval_every=0, probe_positions=0, grad_terms_every=0, checkpoint_every=0,
                        snapshot_first=0, snapshot_every=4)
        whole = tj.train(cfg, tiny_run(cache, os.path.join(tmp, "whole"), **settings), log=quiet)
        assert os.listdir(os.path.join(tmp, "whole", "snapshots")) == ["step_0000004.pt"], "snapshot_every"

        out = os.path.join(tmp, "run")
        asked = []

        def stop() -> bool:
            asked.append(1)
            return len(asked) > 4                       # before the fifth step
        messages = []
        first = tj.train(cfg, tiny_run(cache, out, **settings), log=messages.append, stop=stop)
        assert first["stopped"] and first["steps"] == 4 and first["final"] == {}
        assert any("stopped on request" in m for m in messages)
        assert torch.load(os.path.join(out, "last.pt"), weights_only=False)["step"] == 4
        rest = tj.train(cfg, tiny_run(cache, out, **settings), log=quiet, stop=lambda: False)
        assert not rest["stopped"] and rest["steps"] == 6
        assert [h["step"] for h in rest["history"] if "loss" in h] == [5, 6]
        with open(os.path.join(out, "history.jsonl")) as f:
            losses = [json.loads(line)["loss"] for line in f if '"loss"' in line]
        assert len(losses) == 6
        expected = [h["loss"] for h in whole["history"] if "loss" in h]
        assert all(abs(a - b) < 1e-4 for a, b in zip(losses, expected)), "a stop loses nothing"

        # the command line's switch: a file named "stop" in the run's folder
        switch = tj.stop_switch(out, log=quiet)
        assert not switch()
        open(os.path.join(out, "stop"), "w").close()
        assert switch() and switch() and not os.path.exists(os.path.join(out, "stop"))


def test_a_run_can_be_the_first_part_of_a_longer_one():
    """
    stop_at_step: the run follows the longer run's schedules (the target encoder's momentum is spread over
    total_steps), saves and closes at that step, and the same run without it carries on to the same end.
    Also: a run with no masked board (JEPAConfig.uses_masked_view False) trains, evaluates and sits the exam.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = make_cache(tmp)
        cfg = heads_config(mask_weight=0.0, zpred_weight=0.0, decode_masked_weight=0.0, zlegal_masked_weight=0.0,
                           anticollapse_views="full")
        assert not cfg.uses_masked_view
        settings = dict(total_steps=6, eval_every=0, probe_positions=0, grad_terms_every=0, checkpoint_every=0,
                        snapshot_first=0, snapshot_every=4, exam=True, exam_positions=8, exam_batch_size=4, exam_workers=0)
        whole = tj.train(cfg, tiny_run(cache, os.path.join(tmp, "whole"), **settings), log=quiet)
        assert not whole["stopped"] and whole["final"]["exam_loss"] > 0
        assert not any(key.startswith(("exam_loss_decode_masked", "exam_loss_zlegal_masked")) for key in whole["final"])
        assert all(h["loss_mask"] == 0.0 for h in whole["history"] if "loss" in h)

        out = os.path.join(tmp, "run")
        messages = []
        first = tj.train(cfg, tiny_run(cache, out, stop_at_step=4, **settings), log=messages.append)
        assert first["stopped"] and first["steps"] == 4 and first["final"] == {}
        assert any("stopped at step 4 as planned" in m for m in messages)
        assert os.listdir(os.path.join(out, "snapshots")) == ["step_0000004.pt"]
        again = tj.train(cfg, tiny_run(cache, out, stop_at_step=4, **settings), log=quiet)
        assert again["stopped"] and again["steps"] == 4, "the same command stays where it stopped"
        rest = tj.train(cfg, tiny_run(cache, out, **settings), log=quiet)
        assert not rest["stopped"] and rest["steps"] == 6
        with open(os.path.join(out, "history.jsonl")) as f:
            entries = [json.loads(line) for line in f if '"loss"' in line]
        expected = [h for h in whole["history"] if "loss" in h]
        assert len(entries) == 6
        assert all(abs(a["loss"] - b["loss"]) < 1e-4 for a, b in zip(entries, expected)), "the same run as the long one"
        assert all(a["momentum"] == b["momentum"] for a, b in zip(entries, expected) if "momentum" in a)
        assert abs(rest["final"]["exam_loss"] - whole["final"]["exam_loss"]) < 1e-3
        # a stop_at_step at or past the end changes nothing
        late = tj.train(cfg, tiny_run(cache, os.path.join(tmp, "late"), stop_at_step=6, **settings), log=quiet)
        assert not late["stopped"] and late["steps"] == 6


def test_a_stopped_run_carries_on_on_the_gpu():
    """The checkpoint is loaded onto the run's device, the random generator's state with it; resuming on a GPU
    must still restore it (it once failed there, and only there)."""
    if not torch.cuda.is_available():
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = make_cache(tmp)
        cfg = heads_config()
        settings = dict(total_steps=4, eval_every=0, probe_positions=0, grad_terms_every=0, checkpoint_every=0,
                        snapshot_first=0, device="cuda")
        out = os.path.join(tmp, "run")
        asked = []

        def stop() -> bool:
            asked.append(1)
            return len(asked) > 2
        first = tj.train(cfg, tiny_run(cache, out, **settings), log=quiet, stop=stop)
        assert first["stopped"] and first["steps"] == 2
        rest = tj.train(cfg, tiny_run(cache, out, **settings), log=quiet)
        assert not rest["stopped"] and rest["steps"] == 4 and [h["step"] for h in rest["history"] if "loss" in h] == [3, 4]


def test_search_space_and_how_a_point_becomes_a_run():
    names = lambda cs: set(cs.keys())
    full = names(tj.build_search_space())
    assert full == {name for name, _, _, _ in tj.SEARCH_DIMENSIONS}
    assert {"token_variant", "unpool_layers", "gat_heads", "z_dim", "batch_size", "rollout_curriculum"} <= full
    assert not full & {"zthreat_hidden", "zthreat_head", "decode_hidden", "next_weight", "mask_weight"}, \
        "the heads and the loss weights are not searched"
    squares = names(tj.build_search_space(fixed={"token_variant": "squares", "lr": 1e-4}))
    assert not squares & {"token_variant", "unpool_layers", "unpool_hidden", "unpool_dim", "rl_coef", "lr"}
    unpool = tj.build_search_space(fixed={"token_variant": "unpool"})
    assert {"unpool_layers", "unpool_hidden", "unpool_dim", "rl_coef"} <= names(unpool)
    assert all("unpool_layers" in dict(s) for s in unpool.sample_configuration(5))

    fixed = dict(gat_style="concat", readout="concat", transformer_layers=0, move_encoding="rank_file",
                 rollout_start_steps=2, rollout_max_steps=10, rollout_targets="all", zthreat_plies=3,
                 zthreat_start_plies=1, zthreat_next_weight=1.0, zhead_fit="online")
    seen = set()
    for sample in tj.build_search_space(seed=1).sample_configuration(40):
        point = {**dict(sample), **fixed}
        values, how = tj.resolve_search(point, 3_000_000)
        config = lje.JEPAConfig.from_dict(values)
        config.validate()
        seen.add(config.token_variant)
        assert ("unpool_layers" in dict(sample)) == (config.token_variant == "unpool")
        cap = tj.UNPOOL_MAX_BATCH[config.unpool_layers] if config.token_variant == "unpool" else 256
        assert how["batch_size"] == min(point["batch_size"], cap) and how["warmup_steps"] == 256_000 // how["batch_size"]
        assert not set(values) & set(tj.SEARCH_ONLY)
        # the curricula's shares of the run, as stage lengths: 8 stages from 2 to 10 moves, 2 from 1 to 3 plies
        assert config.rollout_stage_positions == round(point["rollout_curriculum"] * 3_000_000 / 8)
        assert config.zthreat_stage_positions == round(point["threat_curriculum"] * 3_000_000 / 2)
        full_length_at = (config.rollout_max_steps - config.rollout_start_steps) * config.rollout_stage_positions
        assert len(lje.rollout_schedule(config, full_length_at)) == 10 and full_length_at <= 0.8 * 3_000_000 + 8
        assert abs(config.sigreg_weight - point["sigreg_weight"] * 256 / how["batch_size"]) < 1e-12
    assert seen == {"squares", "unpool"}
    # an unpooling setting that is fixed for the whole search does not make the squares configurations invalid
    for variant in ("squares", "unpool"):
        values, _ = tj.resolve_search({**fixed, "token_variant": variant, "unpool_dim": 64, "unpool_layers": 1,
                                       "gat_heads": 4}, 1000)
        lje.JEPAConfig.from_dict(values).validate()
        assert ("unpool_dim" in values) == (variant == "unpool")
    # a run that did not fit is resolved again at half the batch: more steps of warm-up, a stronger weight
    values, how = tj.resolve_search({**fixed, "batch_size": 256, "sigreg_weight": 0.01}, 1000, batch_size=64)
    assert how == {"batch_size": 64, "warmup_steps": 4000} and abs(values["sigreg_weight"] - 0.04) < 1e-12
    assert abs(values["ema_start"] - 0.999) < 1e-12, "0.996 at 256 a step: the same horizon in positions at 64"
    assert abs(tj.resolve_search({**fixed, "token_variant": "squares", "batch_size": 256}, 1000)[0]["ema_start"] - 0.996) < 1e-12
    # an unpooling configuration starts no larger than fits: its batch is capped by its number of layers
    for layers, cap in tj.UNPOOL_MAX_BATCH.items():
        how = tj.resolve_search({**fixed, "token_variant": "unpool", "unpool_layers": layers, "unpool_dim": 64,
                                 "batch_size": 256}, 1000)[1]
        assert how["batch_size"] == cap
    assert tj.resolve_search({**fixed, "token_variant": "unpool", "unpool_layers": 3, "unpool_dim": 64, "batch_size": 256},
                             1000, batch_size=16)[1]["batch_size"] == 16, "a retry's batch is taken as given"


def test_search_objective_scores_runs_and_handles_failures():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = make_cache(tmp)
        base = tiny_run(cache, None, exam_positions=16, exam_batch_size=8, exam_workers=0)
        fixed = heads_config().to_dict()
        objective = tj.make_objective(base, fixed=fixed, log=quiet)
        result = objective({"batch_size": 8, "rollout_curriculum": 0.5}, 0.048)     # 48 positions = 6 steps of 8
        info = result.info
        assert 0 < result.metric < tj.FAILED and abs(result.metric - info["exam_loss"]) < 1e-9
        assert info["steps"] == 6 and info["batch_size"] == 8 and info["positions"] == 48
        assert {"exam_loss_decode_fit", "exam_decode_next_exact", "positions_per_s", "seconds"} <= set(info)
        ran = info["run"]
        assert ran["batch_size"] == 8 and ran["jepa_config"]["rollout_stage_positions"] == 12   # half of 48, in 2 stages
        assert "batch_size" not in ran["jepa_config"] and ran["jepa_config"]["gat_style"] == "concat"
        assert not any(key.startswith("unpool_") for key in info), "nothing unpools in this run"

        # an unpooling run also says how far its graphs grew and whether its decisions settled
        def unpools(config, run, log=None, stop=None):
            logged = [{"step": s + 1, "positions_per_s": 10.0, "tokens": 64 + 64 + s,
                       "unpool_entropy_per_decision": 0.6 - 0.005 * s} for s in range(100)]
            return {"final": {"exam_loss": 2.0}, "steps": run.total_steps, "history": logged + [{"step": 100, "lidar": 1.0}],
                    "stopped": False}
        info = tj.make_objective(base, fixed={**fixed, "token_variant": "unpool"}, log=quiet, train_fn=unpools)(
            {"batch_size": 8}, 1.0).info
        assert info["unpool_nodes_first"] == 64 and abs(info["unpool_nodes_half"] - 113.5) < 1e-9, "64 nodes before unpooling"
        assert abs(info["unpool_nodes_ninth_tenth"] - 148.5) < 1e-9 and abs(info["unpool_nodes_last_tenth"] - 158.5) < 1e-9
        assert abs(info["unpool_entropy_first"] - 0.6) < 1e-9 and abs(info["unpool_entropy_last_tenth"] - 0.1275) < 1e-9
        assert info["unpool_entropy_ninth_tenth"] > info["unpool_entropy_last_tenth"], "still falling at the end"
        assert tj.unpool_course([{"step": 1, "positions_per_s": 10.0}]) == {}
        assert tj.unpool_course([{"tokens": 100.0, "unpool_entropy_per_decision": 0.3}])["unpool_nodes_last_tenth"] == 36

        def finished(run) -> dict:
            return {"final": {"exam_loss": 2.5, "exam_loss_decode_fit": 2.5}, "steps": run.total_steps,
                    "history": [{"positions_per_s": 10.0}], "stopped": False}

        # a run that does not fit is tried again at half the batch size, down to the smallest allowed
        calls = []

        def never_fits(config, run, log=None, stop=None):
            calls.append(run.batch_size)
            raise tj.MemoryBudgetExceeded("too big")
        result = tj.make_objective(base, fixed=fixed, log=quiet, train_fn=never_fits)({"batch_size": 64}, 1.0)
        assert calls == [64, 32, 16] and result.metric == tj.FAILED and result.info["failed"] == "memory"

        seen = []

        def fits_at_32(config, run, log=None, stop=None):
            calls.append(run.batch_size)
            seen.append((config, run))
            if run.batch_size > 32:
                if run.batch_size == 64:
                    raise torch.cuda.OutOfMemoryError("CUDA out of memory")      # handled like the memory guard
                raise tj.MemoryBudgetExceeded("too big")
            return finished(run)
        calls.clear()
        searched = {k: v for k, v in fixed.items() if k != "sigreg_weight"}
        result = tj.make_objective(base, fixed=searched, log=quiet, train_fn=fits_at_32)(
            {"batch_size": 128, "sigreg_weight": 0.01}, 3.2)
        assert calls == [128, 64, 32] and result.metric == 2.5 and result.info["batch_size"] == 32
        config, run = seen[-1]
        assert run.total_steps == 100 and run.warmup_steps == 8000, "the same 3,200 positions, in more steps"
        assert abs(config.sigreg_weight - 0.08) < 1e-12 and run.exam and run.eager_rollouts and run.min_lr_fraction == 1.0
        assert run.eval_every == 0 and run.probe_positions == 0 and run.val_loss_batches == 0

        # fixed values win over anything DEHB samples; --batch-size style overrides go through `fixed` too
        seen.clear()
        tj.make_objective(base, fixed={**fixed, "token_variant": "squares", "batch_size": 16}, log=quiet,
                          train_fn=fits_at_32)({"token_variant": "unpool", "batch_size": 32}, 1.6)
        assert seen[-1][0].token_variant == "squares" and seen[-1][1].batch_size == 16

        def diverges(config, run, log=None, stop=None):
            raise tj.TrainingDiverged("nan")
        result = tj.make_objective(base, fixed=fixed, log=quiet, train_fn=diverges)({"batch_size": 8}, 1.0)
        assert result.metric == tj.FAILED and result.info["failed"] == "diverged"
        result = tj.make_objective(base, fixed={**fixed, "gat_heads": 5}, log=quiet, train_fn=fits_at_32)({"batch_size": 8}, 1.0)
        assert result.metric == tj.FAILED and result.info["failed"] == "invalid", "16 wide does not split into 5 heads"

        # a stop in the middle of an evaluation keeps its checkpoint; the same point carries on from it
        work = os.path.join(tmp, "search")
        asked = []

        def stop() -> bool:
            asked.append(1)
            return len(asked) == 4
        objective = tj.make_objective(base, fixed=fixed, log=quiet, work_dir=work, stop=stop)
        point = {"batch_size": 8, "rollout_curriculum": 0.5}
        try:
            objective(point, 0.048)
            raise AssertionError("the stop should have ended the evaluation")
        except tj.SearchStopped:
            pass
        current = os.path.join(work, "current")
        assert torch.load(os.path.join(current, "last.pt"), weights_only=False)["step"] == 3
        messages = []
        result = tj.make_objective(base, fixed=fixed, log=messages.append, work_dir=work, stop=stop)(point, 0.048)
        assert any("resumed from" in m and "at step 3" in m for m in messages)
        assert result.info["steps"] == 6 and not os.path.exists(os.path.join(current, "last.pt"))
        # another job (here: the same point, a longer run) does not take a stale checkpoint up
        try:
            asked.clear()
            objective(point, 0.048)
        except tj.SearchStopped:
            pass
        assert os.path.exists(os.path.join(current, "last.pt"))
        messages.clear()
        other = tj.make_objective(base, fixed=fixed, log=messages.append, work_dir=work)(point, 0.064)
        assert not any("resumed from" in m for m in messages) and other.info["steps"] == 8


def test_an_evaluation_runs_in_a_process_of_its_own():
    """train_in_subprocess gives what train gives, tells a run to stop through its stop file, and turns the
    ways a run can fail into the exceptions the search's objective handles."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = make_cache(tmp)
        cfg = heads_config()
        settings = dict(total_steps=6, eval_every=0, probe_positions=0, grad_terms_every=0, checkpoint_every=0,
                        snapshot_first=0, exam=True, exam_positions=16, exam_batch_size=8, exam_workers=0)
        here = tj.train(cfg, tiny_run(cache, None, **settings), log=quiet)
        out = os.path.join(tmp, "eval")
        asked = []

        def stop() -> bool:                                 # asked twice a second while the run's process lives
            asked.append(1)
            return False
        there = tj.train_in_subprocess(cfg, tiny_run(cache, out, **settings), stop=stop)
        assert there["steps"] == 6 and not there["stopped"] and asked
        assert abs(there["final"]["exam_loss"] - here["final"]["exam_loss"]) < 1e-4, "the same run, in another process"
        assert len(there["history"]) == 6 and all(set(h) == set(tj.HISTORY_KEPT) for h in there["history"])

        # a stop is passed on through the run's stop file: it saves and says it stopped
        stopped = tj.train_in_subprocess(cfg, tiny_run(cache, os.path.join(tmp, "stopped"), **{**settings, "total_steps": 4000}),
                                         stop=lambda: True)
        assert stopped["stopped"] and stopped["steps"] < 4000
        assert os.path.exists(os.path.join(tmp, "stopped", "last.pt"))

        # failures come back as the exceptions train raises
        try:
            tj.train_in_subprocess(cfg, tiny_run(os.path.join(tmp, "no such cache"), os.path.join(tmp, "bad"), **settings))
            raise AssertionError("a run that cannot start should raise")
        except tj.EvaluationCrashed as error:
            assert "positions_train" in str(error) or "No such file" in str(error) or "FileNotFound" in str(error), error
        assert tj._is_out_of_memory(RuntimeError("CUDA error: out of memory")) and not tj._is_out_of_memory(RuntimeError("x"))
        assert tj._is_out_of_memory(torch.cuda.OutOfMemoryError("CUDA out of memory"))

        # the objective scores a crash as a failure, and gives up when the same thing fails three times running
        def crashes(config, run, log=None, stop=None):
            raise tj.EvaluationCrashed("boom")
        objective = tj.make_objective(tiny_run(cache, None), fixed=cfg.to_dict(), log=quiet, train_fn=crashes)
        for _ in range(2):
            result = objective({"batch_size": 8}, 1.0)
            assert result.metric == tj.FAILED and result.info["failed"] == "crashed"
        try:
            objective({"batch_size": 8}, 1.0)
            raise AssertionError("three crashes in a row should end the search")
        except tj.EvaluationCrashed:
            pass

        def kernel_oom_then_fits(config, run, log=None, stop=None):
            if run.batch_size > 16:
                raise RuntimeError("CUDA error: out of memory")
            return {"final": {"exam_loss": 1.5}, "steps": run.total_steps, "history": [], "stopped": False}
        result = tj.make_objective(tiny_run(cache, None), fixed=cfg.to_dict(), log=quiet,
                                   train_fn=kernel_oom_then_fits)({"batch_size": 32}, 1.6)
        assert result.metric == 1.5 and result.info["batch_size"] == 16


def test_a_search_can_be_stopped_and_taken_up_again():
    """DEHB one evaluation at a time: a stop between evaluations or in the middle of one loses nothing."""
    from dehb_helper import DEHBHelper, DEHBRunConfig, ObjectiveResult

    class Interrupted(Exception):
        pass

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out = os.path.join(tmp, "search")
        space = tj.build_search_space(seed=0, fixed={"token_variant": "squares"})
        evaluated, state = [], {"interrupt_at": None}

        def objective(cfg, fidelity):
            if state["interrupt_at"] == len(evaluated):
                state["interrupt_at"] = None
                state["interrupted"] = (dict(cfg), fidelity)
                raise Interrupted()
            evaluated.append((dict(cfg), fidelity))
            # (longer runs score better, as real ones do)
            return ObjectiveResult(metric=abs(math.log10(cfg["lr"]) + 3.5) + 1.0 / fidelity,
                                   info={"run": {"jepa_config": {"lr": cfg["lr"]}, "batch_size": 8}})

        def helper(resume: bool) -> DEHBHelper:
            return DEHBHelper(configspace=space, objective=objective, direction="minimize",
                              run_cfg=DEHBRunConfig(min_fidelity=1, max_fidelity=9, eta=3, seed=0, n_workers=1,
                                                    output_path=out, resume=resume, save_freq="step",
                                                    best_at_highest_fidelity=True, log_level="ERROR"))

        summary = helper(False).run_stoppable(lambda: False, fevals=5, interrupted=Interrupted)
        assert summary["evaluations"] == 5 and summary["ended_because"] == "evaluations" and len(evaluated) == 5
        assert not os.path.exists(os.path.join(out, "pending_job.json"))

        state["interrupt_at"] = 7                           # the eighth evaluation is cut short
        summary = helper(True).run_stoppable(lambda: False, fevals=20, interrupted=Interrupted)
        assert summary["evaluations"] == 7 and summary["ended_because"] == "stopped"
        assert os.path.exists(os.path.join(out, "pending_job.json"))

        summary = helper(True).run_stoppable(lambda: False, fevals=14, interrupted=Interrupted)
        assert summary["evaluations"] == 14 and len(evaluated) == 14
        again, cut = evaluated[7], state["interrupted"]
        assert again[1] == cut[1] and again[0].keys() == cut[0].keys(), "the interrupted evaluation came first"
        assert all(again[0][k] == cut[0][k] or abs(again[0][k] - cut[0][k]) < 1e-9 * abs(cut[0][k])
                   for k in cut[0]), (again[0], cut[0])
        with open(os.path.join(out, "evals.jsonl")) as f:
            logged = [json.loads(line) for line in f]
        assert len(logged) == 14

        # the best is the best of the longest runs, and its file says how to run it again
        with open(os.path.join(out, "best_config.json")) as f:
            best = json.load(f)
        top = max(e["fidelity"] for e in logged)
        assert best["best_fidelity"] == top
        assert best["best_metric_raw"] == min(e["metric_raw"] for e in logged if e["fidelity"] == top)
        assert tj._load_config(os.path.join(out, "best_config.json")) == {"lr": best["best_config"]["lr"]}
        assert tj._load_run_settings(os.path.join(out, "best_config.json")) == {"batch_size": 8}

        # a stop asked for between evaluations ends the search at once
        summary = helper(True).run_stoppable(lambda: True, fevals=30, interrupted=Interrupted)
        assert summary["evaluations"] == 14 and summary["ended_because"] == "stopped"


def test_config_loading_accepts_search_and_run_files():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "c.json")
        for payload in ({"best_config": {"gat_dim": 64}}, {"jepa_config": {"gat_dim": 64}}, {"gat_dim": 64},
                        {"best_config": {"gat_dim": 32, "batch_size": 64},
                         "best_info": {"run": {"jepa_config": {"gat_dim": 64}, "batch_size": 64, "warmup_steps": 4000}}}):
            with open(path, "w") as f:
                json.dump(payload, f)
            assert tj._load_config(path) == {"gat_dim": 64}
        assert tj._load_run_settings(path) == {"batch_size": 64, "warmup_steps": 4000}
        with open(path, "w") as f:
            json.dump({"jepa_config": {"gat_dim": 64}, "run_config": {}, "batch_size": 128}, f)
        assert tj._load_run_settings(path) == {"batch_size": 128}
    assert tj._load_config(None) == {} and tj._load_run_settings(None) == {}


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
