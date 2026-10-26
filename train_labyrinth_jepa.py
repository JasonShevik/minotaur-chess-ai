"""
Training and hyperparameter search for the Labyrinth JEPA encoder (labyrinth_jepa_encoder.py).

    python train_labyrinth_jepa.py train [--config best_config.json] [--steps 20000] [--out runs/jepa]
    python train_labyrinth_jepa.py tune  [--out dehb_jepa_out] [--fevals 40]

Build the position cache first (python labyrinth_data.py build ...).

train(model_config, run_config) is one training run:
  * AdamW (no weight decay on biases, norms and learned tokens), linear warm-up then cosine decay;
  * the target encoder follows the online encoder by a moving average whose momentum rises from
    ema_start to ema_end over the run;
  * non-finite losses or gradients skip the step (too many in a row stop the run);
  * a memory guard: on Windows a GPU that runs out of memory does not fail, it silently spills into
    system memory and slows down a hundredfold, so a run whose peak memory exceeds memory_budget of the
    card stops with MemoryBudgetExceeded instead. It checks at step 3 and every memory_check_every steps,
    since the unpool variant's graphs grow as it learns to split more nodes;
  * every eval_every steps: LiDAR, RankMe and the self-consistency ratio of the target encoder on a fixed
    set of validation positions, plus the losses on validation batches;
  * logs (history.jsonl), last.pt every checkpoint_every steps and at the end, best.pt by LiDAR, and
    resuming from last.pt.

tune runs DEHB over build_jepa_configspace(). Each evaluation trains from scratch on `fidelity`
thousand positions and reports the ONLINE encoder's LiDAR on the validation positions (higher is
better; RunConfig.objective_metric). Not the target's: its moving average is stretched over the run, so
over DEHB's short runs the target is still largely its initial weights. Values given with --config or
--variant are fixed: they are left out of the search space. A configuration that does not fit in memory
(MemoryBudgetExceeded, or a CUDA out-of-memory error) is retried once at half the batch size; one that
still does not fit, or diverges, scores 0.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional

import torch

import labyrinth_data as ld
import labyrinth_jepa_encoder as lje

# Batch sizes that fit a 16 GB card with the default sizes; the unpool variant stores far more
# activations per position. The memory guard catches configurations these do not fit.
DEFAULT_BATCH_SIZE = {"squares": 128, "registers": 128, "unpool": 48}


class MemoryBudgetExceeded(RuntimeError):
    """A training step used more GPU memory than the run allows."""


class TrainingDiverged(RuntimeError):
    """Too many consecutive steps had a non-finite loss or gradient."""


@dataclass
class RunConfig:
    """How to run training. What to train is the JEPAConfig."""
    cache_dir: str = ld.DEFAULT_CACHE_DIR
    out_dir: Optional[str] = "runs/labyrinth_jepa"   # None: write nothing (DEHB evaluations)
    total_steps: int = 20_000
    batch_size: Optional[int] = None                 # None: DEFAULT_BATCH_SIZE for the token variant
    warmup_steps: int = 1_000                        # capped at a tenth of the run
    min_lr_fraction: float = 0.01                    # the cosine decays to this fraction of the peak lr
    grad_clip: float = 1.0
    num_workers: int = 8
    max_train_positions: Optional[int] = None        # train on a subset (for short runs)
    log_every: int = 50
    eval_every: int = 2_000                          # 0: evaluate only at the end
    eval_positions: int = 2_048                      # validation positions for LiDAR; keep above z_dim
    eval_samples: int = 4                            # encodings per position (LiDAR's class members)
    eval_mask_min_ratio: float = 0.05                # members also get a fresh light mask, as LiDAR does for
    eval_mask_max_ratio: float = 0.15                # I-JEPA, so variants without unpooling are measured fairly
    val_loss_batches: int = 8
    checkpoint_every: int = 2_000                    # 0: only at the end
    resume: bool = True
    seed: int = 0
    device: Optional[str] = None                     # None: cuda if available
    memory_budget: float = 0.9                       # fraction of the GPU's memory a run may use
    memory_check_every: int = 100                    # also check the budget every this many steps (0: only at step 3)
    objective_metric: str = "online_lidar"           # what the DEHB objective reports from the final evaluation
    max_consecutive_skips: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ##### ##### ##### ##### #####
#       Optimizer and schedules


def parameter_groups(model: lje.LabyrinthJEPA, weight_decay: float) -> List[Dict[str, Any]]:
    """Weight decay on weight matrices only: not on biases, norms, or learned tokens and queries."""
    no_decay_names = ("mask_token", "token_type", "registers", "queries", "target_flag")
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if name.startswith("target.") or not p.requires_grad:
            continue
        if p.ndim < 2 or any(n in name for n in no_decay_names) or "promotion" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_factor(step: int, total_steps: int, warmup_steps: int, min_fraction: float) -> float:
    """Linear warm-up to 1, then cosine decay to min_fraction at total_steps."""
    warmup = max(1, min(warmup_steps, total_steps // 10))
    if step < warmup:
        return (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return min_fraction + (1 - min_fraction) * (math.cos(math.pi * t) + 1) / 2


def check_memory_budget(peak_bytes: int, total_bytes: int, budget: float) -> None:
    if peak_bytes > budget * total_bytes:
        raise MemoryBudgetExceeded(f"peak GPU memory {peak_bytes / 2**30:.1f} GB exceeds {budget:.0%} of "
                                   f"{total_bytes / 2**30:.1f} GB; lower the batch size")


# ##### ##### ##### ##### #####
#       Evaluation


@torch.no_grad()
def evaluate(model: lje.LabyrinthJEPA, eval_fens: List[str], val_loader: Optional[ld.JEPALoader],
             run: RunConfig, device: torch.device) -> Dict[str, float]:
    """
    Label-free metrics on fixed validation positions, of the target encoder (the plain keys: the encoder
    that is kept) and of the online encoder ("online_" keys: over short runs the moving-average target
    still lags far behind it), and validation losses.
    """
    was_training = model.training
    model.eval()
    mask = ld.MaskConfig(run.eval_mask_min_ratio, run.eval_mask_max_ratio, 1) if run.eval_mask_max_ratio > 0 else None
    metrics = lje.representation_report(model.target, eval_fens, samples=run.eval_samples, mask_config=mask,
                                        seed=run.seed)
    online = lje.representation_report(model.encoder, eval_fens, samples=run.eval_samples, mask_config=mask,
                                       seed=run.seed)
    metrics.update({f"online_{k}": v for k, v in online.items()})
    if val_loader is not None:
        val_loader.set_epoch(0)
        gen = torch.Generator(device=device).manual_seed(run.seed + 12345)
        sums: Dict[str, float] = {"val_loss_mask": 0.0, "val_loss_next": 0.0}
        n = 0
        for batch in val_loader:
            out = model.losses(batch.to(device), rng=gen)
            sums["val_loss_mask"] += float(out.mask)
            sums["val_loss_next"] += float(out.next)
            n += 1
        metrics.update({k: v / max(1, n) for k, v in sums.items()})
    model.train(was_training)
    return metrics


# ##### ##### ##### ##### #####
#       Training


def _save(path: str, model, opt, sched, step: int, best: float, gen: torch.Generator, config, run) -> None:
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
        "step": step, "best_lidar": best, "generator": gen.get_state(),
        "jepa_config": config.to_dict(), "run_config": run.to_dict(),
    }, tmp)
    os.replace(tmp, path)


def train(config: lje.JEPAConfig, run: RunConfig, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """
    One training run. Returns {"final": metrics at the end, "best_lidar", "steps", "history"}.
    With run.out_dir set, writes run_config.json, history.jsonl, last.pt and best.pt there, and resumes
    from last.pt when run.resume is set and it exists.
    """
    config.validate()
    device = torch.device(run.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = run.batch_size or DEFAULT_BATCH_SIZE[config.token_variant]
    torch.manual_seed(run.seed)
    random.seed(run.seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = lje.LabyrinthJEPA(config).to(device)
    opt = torch.optim.AdamW(parameter_groups(model, config.weight_decay), lr=config.lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, run.total_steps, run.warmup_steps, run.min_lr_fraction))
    gen = torch.Generator(device=device).manual_seed(run.seed + 1)
    step, best_lidar = 0, -math.inf
    history: List[Dict[str, Any]] = []

    paths = {}
    if run.out_dir:
        os.makedirs(run.out_dir, exist_ok=True)
        paths = {name: os.path.join(run.out_dir, name) for name in ("last.pt", "best.pt", "history.jsonl")}
        with open(os.path.join(run.out_dir, "run_config.json"), "w") as f:
            json.dump({"jepa_config": config.to_dict(), "run_config": run.to_dict(), "batch_size": batch_size}, f, indent=2)
        if run.resume and os.path.exists(paths["last.pt"]):
            ckpt = torch.load(paths["last.pt"], map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["scheduler"])
            gen.set_state(ckpt["generator"])
            step, best_lidar = ckpt["step"], ckpt["best_lidar"]
            log(f"resumed from {paths['last.pt']} at step {step}")

    def record(entry: Dict[str, Any]) -> None:
        history.append(entry)
        if paths:
            with open(paths["history.jsonl"], "a") as f:
                f.write(json.dumps(entry) + "\n")

    train_size = len(ld.PositionDataset(run.cache_dir, "train"))
    indices = None
    if run.max_train_positions is not None and run.max_train_positions < train_size:
        indices = random.Random(run.seed).sample(range(train_size), run.max_train_positions)
    loader = ld.JEPALoader(run.cache_dir, "train", batch_size, num_workers=run.num_workers,
                           mask_config=config.mask_config(), seed=run.seed, indices=indices,
                           pin_memory=device.type == "cuda", drop_last=True)
    if len(loader) == 0:
        raise ValueError(f"the training set ({len(loader.dataset)} positions) is smaller than one batch ({batch_size})")
    val_dataset = ld.PositionDataset(run.cache_dir, "val")
    eval_fens = [val_dataset.fen(i) for i in range(min(run.eval_positions, len(val_dataset)))]
    n_val = min(len(val_dataset), run.val_loss_batches * batch_size)
    val_loader = ld.JEPALoader(run.cache_dir, "val", batch_size, num_workers=0, mask_config=config.mask_config(),
                               seed=run.seed + 1, shuffle=False, indices=list(range(n_val))) if n_val else None

    epoch = step // len(loader)
    loader.set_epoch(epoch)
    batches = iter(loader)
    skips = 0
    t_last, positions_since = time.perf_counter(), 0
    log(f"training {config.token_variant} for {run.total_steps} steps of {batch_size} positions on {device} "
        f"({len(loader.dataset):,} training positions, {len(eval_fens)} evaluation positions)")

    def run_eval() -> Dict[str, float]:
        nonlocal best_lidar
        metrics = evaluate(model, eval_fens, val_loader, run, device)
        record({"step": step, "eval": metrics})
        log(f"[{step:>7}] eval  lidar={metrics['lidar']:.1f}  rankme={metrics['rankme']:.1f}  "
            f"rankme_centered={metrics['rankme_centered']:.1f}  online_lidar={metrics['online_lidar']:.1f}  "
            f"online_rankme_centered={metrics['online_rankme_centered']:.1f}  "
            f"self_consistency={metrics['self_consistency']:.3f}  "
            f"val_mask={metrics.get('val_loss_mask', float('nan')):.4f}  val_next={metrics.get('val_loss_next', float('nan')):.4f}")
        if metrics["lidar"] > best_lidar:
            best_lidar = metrics["lidar"]
            if paths:
                _save(paths["best.pt"], model, opt, sched, step, best_lidar, gen, config, run)
        return metrics

    final: Dict[str, float] = {}
    while step < run.total_steps:
        try:
            batch = next(batches)
        except StopIteration:
            epoch += 1
            loader.set_epoch(epoch)
            batches = iter(loader)
            continue
        batch = batch.to(device, non_blocking=True)
        model.train()
        out = model.losses(batch, rng=gen)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(out.total):
            skips += 1
        else:
            out.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), run.grad_clip)
            if not torch.isfinite(grad_norm):
                opt.zero_grad(set_to_none=True)
                skips += 1
            else:
                skips = 0
                opt.step()
                sched.step()
                step += 1
                model.update_target(model.momentum_at(step, run.total_steps))
                positions_since += batch.num_graphs
        if skips:
            if skips >= run.max_consecutive_skips:
                raise TrainingDiverged(f"{skips} consecutive steps with a non-finite loss or gradient at step {step}")
            continue

        if device.type == "cuda" and (step == 3 or (run.memory_check_every and step % run.memory_check_every == 0)):
            check_memory_budget(torch.cuda.max_memory_allocated(device),
                                torch.cuda.get_device_properties(device).total_memory, run.memory_budget)

        if step % run.log_every == 0 or step == run.total_steps:
            now = time.perf_counter()
            entry = {"step": step, "epoch": epoch, "loss": float(out.total.detach()), "lr": sched.get_last_lr()[0],
                     "momentum": model.momentum_at(step, run.total_steps), "grad_norm": float(grad_norm),
                     "positions_per_s": positions_since / max(1e-9, now - t_last), **out.stats}
            if device.type == "cuda":
                entry["gpu_peak_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
            record(entry)
            log(f"[{step:>7}] loss={entry['loss']:.4f} mask={entry['loss_mask']:.4f} next={entry['loss_next']:.4f} "
                f"reg={entry['anticollapse']:.3f} z_std={entry['z_std']:.3f} z_cos={entry['z_mean_cosine']:.3f} "
                f"lr={entry['lr']:.2e} {entry['positions_per_s']:.0f} pos/s")
            t_last, positions_since = now, 0

        if run.eval_every and step % run.eval_every == 0 and step < run.total_steps:
            run_eval()
        if paths and run.checkpoint_every and step % run.checkpoint_every == 0:
            _save(paths["last.pt"], model, opt, sched, step, best_lidar, gen, config, run)

    final = run_eval()
    if paths:
        _save(paths["last.pt"], model, opt, sched, step, best_lidar, gen, config, run)
    result = {"final": final, "best_lidar": best_lidar, "steps": step, "batch_size": batch_size, "history": history}
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ##### ##### ##### ##### #####
#       DEHB


def _release_gpu_memory() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_objective(base_run: RunConfig, fixed: Optional[Dict[str, Any]] = None,
                   train_fn: Callable[..., Dict[str, Any]] = train, log: Callable[[str], None] = print):
    """
    The DEHB objective: config dict and fidelity (thousands of training positions) -> base_run's
    objective_metric (the online encoder's LiDAR). Fidelity counts positions rather than steps so
    configurations with different batch sizes get the same amount of data. fixed: JEPAConfig values every
    evaluation shares (e.g. z_dim, or a token_variant); they override anything sampled.
    """
    from dehb_helper import ObjectiveResult

    def objective(cfg: Dict[str, Any], fidelity: float) -> ObjectiveResult:
        config = lje.JEPAConfig.from_dict({**cfg, **(fixed or {})})
        batch_size = base_run.batch_size or DEFAULT_BATCH_SIZE[config.token_variant]
        positions = int(round(fidelity * 1000))
        for attempt in range(2):
            run = replace(base_run, out_dir=None, resume=False, eval_every=0, checkpoint_every=0,
                          batch_size=batch_size, total_steps=max(1, positions // batch_size))
            failure = None
            try:
                result = train_fn(config, run, log=log)
            except (MemoryBudgetExceeded, torch.cuda.OutOfMemoryError) as e:
                failure = f"{type(e).__name__}: {e}"
            except TrainingDiverged as e:
                return ObjectiveResult(metric=0.0, info={"failed": "diverged", "error": str(e)})
            if failure is not None:
                _release_gpu_memory()      # outside the except block, once the failed run's frames are gone
                log(f"  does not fit at batch size {batch_size}: {failure}")
                if attempt == 0 and batch_size >= 16:
                    batch_size //= 2
                    continue
                return ObjectiveResult(metric=0.0, info={"failed": "memory", "batch_size": batch_size})
            final = result["final"]
            return ObjectiveResult(metric=float(final[base_run.objective_metric]), info={
                "target_lidar": final.get("lidar"),
                "rankme": final["rankme"], "rankme_centered": final.get("rankme_centered"),
                "online_lidar": final.get("online_lidar"),
                "self_consistency": final["self_consistency"],
                "val_loss_mask": final.get("val_loss_mask"), "val_loss_next": final.get("val_loss_next"),
                "batch_size": batch_size, "steps": result["steps"]})
        return ObjectiveResult(metric=0.0, info={"failed": "memory"})

    return objective


# ##### ##### ##### ##### #####
#       Command line


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    """A JEPAConfig dict, a DEHB best_config.json ({"best_config": {...}}), or a run_config.json."""
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("best_config") or data.get("jepa_config") or data


def _plot(history: List[Dict[str, Any]], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps = [h for h in history if "loss" in h]
    evals = [h for h in history if "eval" in h]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot([h["step"] for h in steps], [h["loss_mask"] for h in steps], label="masked squares")
    axes[0].plot([h["step"] for h in steps], [h["loss_next"] for h in steps], label="next position")
    axes[0].set_title("Training losses"); axes[0].set_xlabel("step"); axes[0].legend()
    axes[1].plot([h["step"] for h in evals], [h["eval"]["lidar"] for h in evals], marker="o", label="LiDAR")
    axes[1].plot([h["step"] for h in evals], [h["eval"]["rankme_centered"] for h in evals], marker="o",
                 label="RankMe (centred)")
    axes[1].set_title("Representation rank (target encoder)"); axes[1].set_xlabel("step"); axes[1].legend()
    axes[2].plot([h["step"] for h in evals], [h["eval"]["self_consistency"] for h in evals], marker="o")
    axes[2].set_title("Self-consistency (lower = more consistent)"); axes[2].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training.png"), dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or tune the Labyrinth JEPA encoder.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "tune"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=None, help="JEPAConfig values (JSON), e.g. a DEHB best_config.json")
        p.add_argument("--variant", choices=lje.TOKEN_VARIANTS, default=None)
        p.add_argument("--cache", default=ld.DEFAULT_CACHE_DIR)
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--max-train-positions", type=int, default=None)
        p.add_argument("--seed", type=int, default=0)
    t = sub.choices["train"]
    t.add_argument("--steps", type=int, default=20_000)
    t.add_argument("--out", default="runs/labyrinth_jepa")
    t.add_argument("--eval-every", type=int, default=2_000)
    t.add_argument("--no-resume", action="store_true")
    d = sub.choices["tune"]
    d.add_argument("--out", default="dehb_jepa_out")
    d.add_argument("--fevals", type=int, default=40)
    d.add_argument("--min-fidelity", type=float, default=30, help="thousands of training positions")
    d.add_argument("--max-fidelity", type=float, default=810, help="thousands of training positions")
    args = parser.parse_args()

    values = _load_config(args.config)
    if args.variant:
        values["token_variant"] = args.variant
    run = RunConfig(cache_dir=args.cache, batch_size=args.batch_size, num_workers=args.workers,
                    max_train_positions=args.max_train_positions, seed=args.seed)

    if args.command == "train":
        config = lje.JEPAConfig.from_dict(values)
        run = replace(run, out_dir=args.out, total_steps=args.steps, eval_every=args.eval_every,
                      resume=not args.no_resume)
        result = train(config, run)
        _plot(result["history"], args.out)
        print(json.dumps({"final": result["final"], "best_lidar": result["best_lidar"]}, indent=2))
    else:
        from dehb_helper import DEHBHelper, DEHBRunConfig
        known = {f.name for f in dataclasses.fields(lje.JEPAConfig)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise SystemExit(f"--config has keys JEPAConfig does not know: {unknown}")
        if values:
            print(f"Fixed for this search (not searched): {values}")
        resume = os.path.exists(os.path.join(args.out, "evals.jsonl"))
        if resume:
            print(f"Found previous search state in {args.out}/ - resuming (delete it to start fresh).")
        helper = DEHBHelper(
            configspace=lje.build_jepa_configspace(seed=args.seed, fixed=values),
            objective=make_objective(run, fixed=values),
            direction="maximize",
            run_cfg=DEHBRunConfig(min_fidelity=args.min_fidelity, max_fidelity=args.max_fidelity, eta=3,
                                  fevals=args.fevals, seed=args.seed, n_workers=1, output_path=args.out,
                                  resume=resume),
        )
        summary = helper.run()
        print(json.dumps(summary, indent=2, default=str))
        print(f"\nTrain the best configuration with:  python train_labyrinth_jepa.py train --config {summary['best_config_path']}")


if __name__ == "__main__":
    main()
