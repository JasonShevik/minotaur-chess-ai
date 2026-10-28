"""
Tests for the JEPA training script, train_labyrinth_jepa.py.

Run with plain Python (no pytest needed):

    python test_train_labyrinth_jepa.py

Tiny models on a tiny position cache built in a temporary folder, on the CPU. They check that a run
trains, evaluates, checkpoints and resumes; that the schedules and weight-decay groups are right; that
the memory guard trips; and that the DEHB objective scores runs and handles runs that do not fit.
"""
import csv
import json
import os
import random
import sys
import tempfile
import traceback
from dataclasses import replace
from typing import Callable, List

import chess
import torch

import labyrinth_data as ld
import labyrinth_jepa_encoder as lje
import probe_labyrinth as pl
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


def test_dehb_objective_scores_runs_and_handles_failures():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = make_cache(tmp)
        base = tiny_run(cache, None, batch_size=8)
        fixed = {k: v for k, v in tiny_config().to_dict().items() if k not in ("token_variant",)}
        objective = tj.make_objective(base, fixed=fixed, log=quiet)
        result = objective({"token_variant": "squares"}, 0.024)     # 24 positions = 3 steps of 8
        assert result.metric >= 1.0 - 1e-6 and result.info["steps"] == 3 and result.info["batch_size"] == 8

        # a run that does not fit is retried once at half the batch size, then scores 0
        calls = []

        def never_fits(config, run, log=None):
            calls.append(run.batch_size)
            raise tj.MemoryBudgetExceeded("too big")
        result = tj.make_objective(replace(base, batch_size=32), log=quiet, train_fn=never_fits)({}, 1.0)
        assert calls == [32, 16] and result.metric == 0.0 and result.info["failed"] == "memory"

        def fits_at_half(config, run, log=None):
            calls.append(run.batch_size)
            if run.batch_size > 16:
                raise tj.MemoryBudgetExceeded("too big")
            return {"final": {"lidar": 7.0, "online_lidar": 11.0, "rankme": 9.0, "self_consistency": 0.1},
                    "steps": run.total_steps}
        calls.clear()
        result = tj.make_objective(replace(base, batch_size=32), log=quiet, train_fn=fits_at_half)({}, 1.6)
        assert calls == [32, 16] and result.info["steps"] == 100   # 1,600 positions / 16
        assert result.metric == 11.0 and result.info["target_lidar"] == 7.0, "the objective scores the online encoder"

        # a real CUDA out-of-memory error is handled like the memory guard, not allowed to end the search
        calls.clear()

        def cuda_oom_then_fits(config, run, log=None):
            calls.append(run.batch_size)
            if run.batch_size > 16:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return fits_at_half(config, run)
        calls_before = len(calls)
        result = tj.make_objective(replace(base, batch_size=32), log=quiet, train_fn=cuda_oom_then_fits)({}, 1.6)
        assert result.metric == 11.0 and 16 in calls

        # fixed values win over anything DEHB samples
        seen = []

        def record(config, run, log=None):
            seen.append(config.token_variant)
            return fits_at_half(config, replace(run, batch_size=16))
        tj.make_objective(replace(base, batch_size=16), fixed={"token_variant": "squares"}, log=quiet,
                          train_fn=record)({"token_variant": "unpool"}, 1.6)
        assert seen == ["squares"]

        def diverges(config, run, log=None):
            raise tj.TrainingDiverged("nan")
        assert tj.make_objective(base, log=quiet, train_fn=diverges)({}, 1.0).metric == 0.0


def test_config_loading_accepts_dehb_and_run_files():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for payload in ({"best_config": {"gat_dim": 64}}, {"jepa_config": {"gat_dim": 64}}, {"gat_dim": 64}):
            path = os.path.join(tmp, "c.json")
            with open(path, "w") as f:
                json.dump(payload, f)
            assert tj._load_config(path) == {"gat_dim": 64}
    assert tj._load_config(None) == {}


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
