"""
Tests for the JEPA training script, train_labyrinth_jepa.py.

Run with plain Python (no pytest needed):

    python test_train_labyrinth_jepa.py

Tiny models on a tiny position cache built in a temporary folder, on the CPU. They check that a run
trains, evaluates, checkpoints and resumes; that the schedules and weight-decay groups are right; that
the memory guard trips; and that the DEHB objective scores runs and handles runs that do not fit.
"""
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
        result = tj.train(tiny_config(), tiny_run(cache, out), log=quiet)
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

        # resuming continues from step 6 to step 9
        resumed = tj.train(tiny_config(), tiny_run(cache, out, total_steps=9), log=quiet)
        assert resumed["steps"] == 9
        assert [h["step"] for h in resumed["history"] if "loss" in h] == [7, 8, 9]
        with open(os.path.join(out, "history.jsonl")) as f:
            lines = [json.loads(l) for l in f]
        assert [h["step"] for h in lines if "loss" in h] == list(range(1, 10))


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
            return {"final": {"lidar": 7.0, "rankme": 9.0, "self_consistency": 0.1}, "steps": run.total_steps}
        calls.clear()
        result = tj.make_objective(replace(base, batch_size=32), log=quiet, train_fn=fits_at_half)({}, 1.6)
        assert calls == [32, 16] and result.metric == 7.0 and result.info["steps"] == 100   # 1,600 positions / 16

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
