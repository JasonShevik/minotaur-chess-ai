"""
Training and hyperparameter search for the Labyrinth JEPA encoder (labyrinth_jepa_encoder.py).

    python train_labyrinth_jepa.py train [--config best_config.json] [--steps 20000] [--out runs/jepa] \\
        [--probe-db minotaur_data.db --probe-puzzles lichess_db_puzzle.csv.zst]
    python train_labyrinth_jepa.py tune  --config base.json --out dehb_jepa_out --hours 36

Build the position cache first (python labyrinth_data.py build ...). Both commands stop when q is pressed in
their window (or a file named "stop" appears in --out): they save and close, and the same command carries on
from there.

train(model_config, run_config) is one training run:
  * AdamW (no weight decay on biases, norms and learned tokens), linear warm-up then cosine decay;
  * the target encoder follows the online encoder by a moving average whose momentum rises from
    ema_start to ema_end over the run;
  * the rollout curriculum (labyrinth_jepa_encoder.rollout_schedule): how many moves ahead the model
    predicts grows with the number of training positions seen; when it grows, the loader continues the
    current epoch with longer rollouts, without repeating positions, building graphs only for the steps
    that are graded (rollout_target_steps). With eager_rollouts the batches carry the whole rollout from
    the start and each step uses what the curriculum has reached: the same training, no loader restarts;
  * the loaders hand the batches' graphs over in a compact form (labyrinth_data.CompactGraphs) and they are
    expanded on the GPU: every chess graph repeats the same edges, and with ten-move rollouts a loader's
    ready batches would otherwise fill the machine's memory;
  * with the legal-threat head (JEPAConfig.zthreat_*), batches carry threat labels from the labels database
    reach_labeler.py fills (threat_db). Positions it does not hold, the validation positions among them, are
    searched on the spot when the compiled search is built and go ungraded when it is not. The head's own
    curriculum (labyrinth_jepa_encoder.threat_schedule) adds a ply of play at a time;
  * non-finite losses or gradients skip the step (too many in a row stop the run);
  * a memory guard: on Windows a GPU that runs out of memory does not fail, it silently spills into
    system memory and slows down a hundredfold, so a run whose peak memory exceeds memory_budget of the
    card stops with MemoryBudgetExceeded instead. It checks at step 3 and every memory_check_every steps,
    since the unpool variant's graphs grow as it learns to split more nodes. It measures the memory
    PyTorch has reserved, not just allocated: the unpool variant's graphs change size every batch, which
    fragments the allocator's cache until the reserved memory is far above the allocated. So whenever the
    reserved memory passes cache_release_fraction of the card, the unused cache is handed back first;
  * every eval_every steps: LiDAR, RankMe and the self-consistency ratio of the target encoder on a fixed
    set of validation positions, plus the losses on validation batches, graded as training is at that
    step. With exam, also the exam (exam()): the heads' losses with the whole rollout and every threat ply
    at full weight, on positions, masks and moves that are the same for every run, so runs with different
    curricula and batch sizes compare;
  * probes at every evaluation (probe_positions > 0): linear and nearest-neighbour probes of the target
    encoder at several views (probe_views: z and vectors inside the encoder), on fixed positions with
    fixed labels: board facts of validation positions, plus engine evaluations and best moves (probe_db)
    and puzzle themes (probe_puzzles) when given. The untrained encoder (the run's own starting weights)
    and the raw board are probed once, as baselines. Rows go to probes.csv, averages to the history;
  * every grad_terms_every steps, the gradient each loss term alone gives each part of the model (the
    square stage, the transformer, the readout, the predictors), and the cosine between terms' gradients on
    the encoder's parts: which signal shapes which layers, and whether signals pull against each other;
  * weights-only snapshots at step 0 and at snapshot_first, 2x, 4x, ... steps, at every multiple of
    snapshot_every and at the end, in snapshots/ (probe_labyrinth.py --snapshots probes them all);
  * logs (history.jsonl), last.pt every checkpoint_every steps, at a stop and at the end, best.pt by LiDAR,
    and resuming from last.pt.

tune runs DEHB over build_search_space(): the square stage's and the readout's sizes, z's size, whether to
unpool and how, the next-position predictor's size, the learning rate, the batch size, SIGReg's weight and
how fast the two curricula go (SEARCH_DIMENSIONS). Each evaluation trains from scratch on `fidelity`
thousand positions, at a constant learning rate after the warm-up, and reports its exam loss (lower is
better): the loss terms graded against the board and the rules, measured the same way for every run. Not
LiDAR, which went the other way from everything else we measured, and not the whole training loss: its
prediction terms are graded against what the encoder itself makes, so an encoder that keeps less scores
better on them. For the same reason the search does not vary the heads or the loss weights. Values given
with --config or --variant are fixed: they are left out of the search space. Every evaluation runs in a
process of its own. One that does not fit in memory is tried again at half the batch size; one that never
fits, is not a valid configuration, diverges or crashes scores FAILED. The search writes its state after
every evaluation and keeps the checkpoint of the one under way when stopped, so nothing is lost.
<out>/best_config.json holds the best of the longest runs and the configuration it ran: train --config
takes it as it is.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import random
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

import labyrinth_data as ld
import labyrinth_jepa_encoder as lje
import probe_labyrinth as pl

# Batch sizes that fit a 16 GB card with the default sizes; the unpool variant stores far more
# activations per position. The memory guard catches configurations these do not fit.
DEFAULT_BATCH_SIZE = {"squares": 256, "registers": 256, "unpool": 48}


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
    min_lr_fraction: float = 0.01                    # the cosine decays to this fraction of the peak lr (1: the
                                                     # rate stays at its peak after the warm-up)
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
    eval_subset: str = "random"                      # the validation rows evaluations use: "random" (one fixed draw,
                                                     # the same in every run) or "first" (the split's first rows, which
                                                     # are sorted by FEN and endgame-heavy; all runs before 2026-11-14)
    checkpoint_every: int = 2_000                    # 0: only at the end
    resume: bool = True
    seed: int = 0
    device: Optional[str] = None                     # None: cuda if available
    memory_budget: float = 0.9                       # fraction of the GPU's memory a run may use
    memory_check_every: int = 100                    # also check the budget every this many steps (0: only at step 3)
    cache_release_fraction: float = 0.7              # hand the allocator's unused cache back to the GPU whenever the
                                                     # reserved memory passes this fraction of the card (0: never)
    exam: bool = False                               # every evaluation also sits the exam (see exam()): the heads'
                                                     # losses with the whole rollout and every threat ply at full
                                                     # weight, whatever the curricula have reached
    exam_positions: int = 2_048                      # validation positions the exam is set on (the same for every run)
    exam_batch_size: int = 128                       # and how many are graded at once (the same for every run, so the
                                                     # exam does not depend on the training batch size)
    exam_workers: int = 8                            # loader processes for the exam (its positions' threat labels are
                                                     # searched on the spot); 0: in the main process
    eager_rollouts: bool = False                     # batches always carry the full rollout and the trainer uses as much
                                                     # as the curriculum has reached, instead of restarting the loader
                                                     # every time the curriculum adds a move
    compact_batches: bool = True                     # the loaders hand the graphs over without what every chess graph
                                                     # repeats (labyrinth_data.CompactGraphs): the same batches on the
                                                     # GPU, a fraction of the memory kept ready on the way there
    prefetch_factor: int = 4                         # batches each loader process keeps ready
    max_consecutive_skips: int = 50
    probe_positions: int = 8_000                     # positions per probe set, probed at every evaluation (0: no probes)
    probe_views: Tuple[str, ...] = ("z", "token_mean", "pre_transformer_mean")   # see probe_labyrinth.VIEWS
    probe_db: Optional[str] = None                   # minotaur_data.db: engine evaluations and best moves
    probe_puzzles: Optional[str] = None              # lichess_db_puzzle.csv.zst: puzzle themes (loading takes ~1 min)
    grad_terms_every: int = 200                      # per-term gradient norms every this many steps (0: never)
    snapshot_first: int = 250                        # snapshots at step 0, this, twice this, ... (0: none)
    snapshot_every: int = 0                          # and at every multiple of this (0: only the doubling ones): a long
                                                     # run's doubling snapshots grow hours apart
    threat_db: Optional[str] = "minotaur_reach.db"   # the labels database (reach_labeler.py), for the legal-threat head
    threat_compute: bool = True                      # search the positions it does not hold on the spot (needs the
                                                     # compiled search, build_fast_reach.py); False: leave them ungraded
    stop_at_step: Optional[int] = None               # save and close at this step, as a stop request does: the first
                                                     # part of a longer run, on that run's schedules (total_steps sets
                                                     # the target encoder's and the learning rate's), to carry on later
                                                     # or to compare with the longer run's snapshot at the same step

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ##### ##### ##### ##### #####
#       Optimizer and schedules


def parameter_groups(model: lje.LabyrinthJEPA, weight_decay: float) -> List[Dict[str, Any]]:
    """Weight decay on weight matrices only: not on biases, norms, or learned tokens and queries."""
    no_decay_names = ("mask_token", "token_type", "registers", "queries", "target_flag", "file_marks")
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
             run: RunConfig, device: torch.device, rollout_weights: Optional[List[float]] = None,
             encode_batch: int = 256, threat_weights: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Label-free metrics on fixed validation positions, of the target encoder (the plain keys: the encoder
    that is kept) and of the online encoder ("online_" keys: over short runs the moving-average target
    still lags far behind it), and validation losses. encode_batch: positions encoded at once.
    """
    was_training = model.training
    model.eval()
    mask = ld.MaskConfig(run.eval_mask_min_ratio, run.eval_mask_max_ratio, 1) if run.eval_mask_max_ratio > 0 else None
    metrics = lje.representation_report(model.target, eval_fens, samples=run.eval_samples, mask_config=mask,
                                        seed=run.seed, batch_size=encode_batch)
    online = lje.representation_report(model.encoder, eval_fens, samples=run.eval_samples, mask_config=mask,
                                       seed=run.seed, batch_size=encode_batch)
    metrics.update({f"online_{k}": v for k, v in online.items()})
    if val_loader is not None:
        val_loader.set_epoch(0)
        steps = len(rollout_weights) if rollout_weights else 1
        val_loader.set_rollout_steps(steps, lje.rollout_target_steps(model.config, steps))
        gen = torch.Generator(device=device).manual_seed(run.seed + 12345)
        sums: Dict[str, float] = {"val_loss_mask": 0.0, "val_loss_next": 0.0}
        n = 0
        for batch in val_loader:
            out = model.losses(batch.to(device), rng=gen, rollout_weights=rollout_weights, threat_weights=threat_weights)
            sums["val_loss_mask"] += float(out.mask)
            sums["val_loss_next"] += float(out.next)
            for key, value in out.stats.items():       # the optional predictors' and heads' losses and scores
                if key.startswith(("loss_zpred", "loss_inverse", "inverse_accuracy", "loss_count", "count_", "loss_legal",
                                   "legal_", "loss_decode", "decode_", "loss_zlegal", "zlegal_", "loss_zthreat", "zthreat_")):
                    sums[f"val_{key}"] = sums.get(f"val_{key}", 0.0) + value
            n += 1
        metrics.update({k: v / max(1, n) for k, v in sums.items()})
    model.train(was_training)
    return metrics


# The loss terms graded against the board and the rules of chess (not against the target encoder's own output)
BOARD_TERMS = ("count", "legal", "decode", "zlegal", "zthreat")
EXAM_SEED = 20261120             # the exam's masks, moves and unpooling draws: the same for every run and seed


@torch.no_grad()
def exam(model: lje.LabyrinthJEPA, loader: ld.JEPALoader, device: torch.device) -> Dict[str, float]:
    """
    The exam: the heads' losses on fixed validation positions, with the whole rollout (rollout_max_steps moves,
    every one of them graded, each counting the same) and every threat ply at full weight, whatever the run's
    curricula have reached and whichever steps it trains on. So two runs' exams compare whatever their
    curricula, batch sizes and learning rates: a run cannot look better by keeping its task easy.

      exam_loss            the sum of the loss terms graded against the board (BOARD_TERMS), each at its training
                           weight: the part of the training loss an encoder cannot lower by keeping less. Lower
                           is better; this is what the search minimizes.
      exam_loss_<term>     each of those terms
      exam_<score>         the heads' scores (boards decoded exactly, threat cells wrong, ...), as in training
      exam_latent_<term>   the prediction losses graded against the target encoder (mask, next, zpred), for the
                           record only: they depend on what the encoder itself makes, so they do not compare
                           between encoders
    loader: validation positions with the full rollout as targets (train() builds it: every run the same
    positions, masks and moves).
    """
    config = model.config
    was_training = model.training
    model.eval()
    loader.set_epoch(0)
    loader.set_rollout_steps(config.rollout_max_steps, None)
    rollout_weights = [1.0] * config.rollout_max_steps
    threat_weights = [1.0] * (config.zthreat_plies + 1) if config.uses_threats else None
    gen = torch.Generator(device=device).manual_seed(EXAM_SEED)
    sums: Dict[str, float] = {"exam_loss": 0.0}
    n = 0
    for batch in loader:
        out = model.losses(batch.to(device), rng=gen, rollout_weights=rollout_weights, threat_weights=threat_weights)
        for name, value in out.terms.items():
            if name.startswith(BOARD_TERMS):
                sums["exam_loss"] += float(value)
                sums[f"exam_loss_{name}"] = sums.get(f"exam_loss_{name}", 0.0) + float(value)
            elif name in ("mask", "next", "zpred"):
                sums[f"exam_latent_{name}"] = sums.get(f"exam_latent_{name}", 0.0) + float(value)
        for key, value in out.stats.items():
            if key.startswith(("count_", "legal_", "decode_", "zlegal_", "zthreat_")):
                sums[f"exam_{key}"] = sums.get(f"exam_{key}", 0.0) + value
        n += 1
    model.train(was_training)
    return {k: v / max(1, n) for k, v in sums.items()}


def exam_line(metrics: Dict[str, float]) -> str:
    """The exam's headline scores, for the log."""
    names = (("", "own z"), ("_fit", "own z"), ("_next", "next z"), ("_masked", "whole z"))
    boards = [f"{label} {100 * metrics[key]:.1f}%" for key, label in
              ((f"exam_decode{part}_exact", label) for part, label in names) if key in metrics]
    wrong = [f"{label} {metrics[key]:.0f}" for key, label in
             ((f"exam_zthreat{part}_wrong", label) for part, label in names) if key in metrics]
    return "; ".join(([f"boards decoded exactly: {', '.join(boards)}"] if boards else [])
                     + ([f"threat cells wrong a position per ply: {', '.join(wrong)}"] if wrong else []))


EVAL_SUBSET_SEED = 20261114      # one draw for every run and seed, so evaluations compare across runs


def evaluation_positions(run: RunConfig, val_size: int, batch_size: int) -> Tuple[List[int], List[int]]:
    """
    The validation rows an evaluation uses: (for RankMe / LiDAR / self-consistency, for the validation losses).
    eval_subset "random": a fixed random draw of the split, the same for every run and seed. "first": the
    split's first rows; the split is sorted by FEN, so those are endgame-heavy (about 18 pieces a position
    against 24 for a random draw). Every run before 2026-11-14 used "first": their validation losses, RankMe
    and LiDAR compare with each other, not with "random" runs. The probes always drew random positions.
    """
    n_eval, n_val = min(run.eval_positions, val_size), min(val_size, run.val_loss_batches * batch_size)
    if run.eval_subset == "first":
        order = list(range(max(n_eval, n_val)))
    elif run.eval_subset == "random":
        order = random.Random(EVAL_SUBSET_SEED).sample(range(val_size), max(n_eval, n_val))
    else:
        raise ValueError("eval_subset must be 'random' or 'first'")
    return order[:n_eval], order[:n_val]


def build_probes(run: RunConfig, val_dataset: ld.PositionDataset, log: Callable[[str], None] = print,
                 encode_batch: int = 512) -> Optional[pl.TrainingProbes]:
    """
    The run's fixed probe set (None with probe_positions 0): board facts of random validation positions,
    plus the engine-analyzed positions (probe_db) and puzzles (probe_puzzles) when given, each cut to
    probe_positions positions.
    """
    if run.probe_positions <= 0:
        return None
    t0 = time.perf_counter()
    picks = sorted(random.Random(run.seed + 7).sample(range(len(val_dataset)), min(run.probe_positions, len(val_dataset))))
    sets = [pl.board_label_set("val", [val_dataset.fen(i) for i in picks])]
    if run.probe_db:
        sets.append(pl.subsample(pl.load_labeled(run.probe_db, log=log), run.probe_positions, seed=run.seed))
    if run.probe_puzzles:
        sets.append(pl.subsample(pl.load_puzzles(run.probe_puzzles, seed=run.seed, log=log), run.probe_positions,
                                 seed=run.seed))
    log(f"probe sets ({time.perf_counter() - t0:.0f}s): " + ", ".join(
        f"{s.name} {len(s.fens):,} positions, {len(s.binary) + len(s.numeric)} labels" for s in sets))
    return pl.TrainingProbes(sets, views=run.probe_views, seed=run.seed, batch_size=encode_batch)


def threat_labels(config: lje.JEPAConfig, run: RunConfig, batch_size: int,
                  log: Callable[[str], None] = print) -> Optional[ld.ThreatLabels]:
    """
    Where the run's batches get legal-threat labels (None: the model has no legal-threat head), and a line in
    the log on how much of the run the labels database covers. The positions after rollout moves are labelled
    only when something is graded on them.
    """
    if not config.uses_threats:
        return None
    on_steps = config.zthreat_next_weight > 0 or (config.zthreat_judges_predictions and config.zhead_fit == "target")
    source = ld.ThreatLabels(config.zthreat_plies, run.threat_db, compute=run.threat_compute,
                             rollout_steps=config.zthreat_rollout_steps if on_steps else 0,
                             branches=config.zthreat_branches)
    if source.connection is None and not source.searches:
        raise ValueError(f"the legal-threat head needs labels: there is no labels database at {run.threat_db!r} "
                         f"(reach_labeler.py makes one; --threat-db says where it is) and the compiled search "
                         f"that could work them out on the spot is {ld.pr.C_UNAVAILABLE or 'switched off'}")
    # The database follows the order of a run on the whole training split with default masks
    stored = source.stored_samples("train", run.seed) if run.max_train_positions is None else 0
    others = ("searched on the spot" if source.searches
              else f"left ungraded (the compiled search is {ld.pr.C_UNAVAILABLE or 'switched off'})")
    log(f"legal-threat labels to {config.zthreat_plies} plies: the database holds the first {stored:,} samples of this "
        f"run's order (the run takes {run.total_steps * batch_size:,}); positions it does not hold are {others}")
    return source


# What the log shows of each probed view: (summary key, name), as linear / nearest-neighbour / whitened
# nearest-neighbour scores
PROBE_HEADLINES = (("val/material_balance", "material r2"), ("labeled/eval_win_prob", "eval r2"),
                   ("puzzles/theme/auc", "themes auc"), ("labeled/best_move/auc", "best move auc"))


def _probe_line(view_summary: Dict[str, Dict[str, float]]) -> str:
    parts = [f"{name} " + "/".join(f"{view_summary[key].get(probe, math.nan):.2f}" for probe in ("linear", "knn", "knn_white"))
             for key, name in PROBE_HEADLINES if key in view_summary]
    return "  ".join(parts) + "  (linear/knn/whitened knn)"


def _append_probe_rows(path: str, rows: Sequence[Dict[str, Any]], step: int) -> None:
    columns = ["step"] + pl.COLUMNS
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow({"step": step, **{c: f"{v:.4f}" if isinstance(v, float) else v
                                              for c, v in ((c, row.get(c, "")) for c in pl.COLUMNS)}})


# ##### ##### ##### ##### #####
#       Gradient diagnostics


# The encoder's parts, from input to output
ENCODER_PARTS = ("square_stage", "transformer", "readout")


def parameter_part(name: str) -> str:
    """Which part of the model a parameter belongs to, by its name in LabyrinthJEPA."""
    if not name.startswith("encoder."):
        return name.split(".", 1)[0]                   # mask_predictor, next_predictor, baselines
    name = name[len("encoder."):]
    if name.startswith(("queries", "readout_")):
        return "readout"
    if name.startswith(("token_proj", "token_type", "registers", "transformer.", "final_norm")):
        return "transformer"
    return "square_stage"                              # input_proj, mask_token, the square GATs, unpooling


def _sums_by_part(parts: Sequence[str], a: Sequence[Optional[torch.Tensor]],
                  b: Sequence[Optional[torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Per part, and for the encoder's parts together ("encoder"): the sum of a * b over the part's
    parameters. A missing gradient (a term that does not reach a parameter) leaves the parameter out."""
    sums: Dict[str, torch.Tensor] = {}
    for part, x, y in zip(parts, a, b):
        if x is None or y is None:
            continue
        v = (x * y).sum()
        for key in (part, "encoder") if part in ENCODER_PARTS else (part,):
            sums[key] = sums[key] + v if key in sums else v
    return sums


def _trainable(model: lje.LabyrinthJEPA) -> Tuple[List[torch.nn.Parameter], List[str]]:
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and not n.startswith("target.")]
    return [p for _, p in named], [parameter_part(n) for n, _ in named]


def gradient_by_term(model: lje.LabyrinthJEPA, terms: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    What each loss term alone does to the gradient, before clipping: {"norms": {term: {part: norm}},
    "cosine": {"a|b": {part: cosine}}}, the cosines between two terms' gradients on each part of the
    encoder and on the whole encoder (negative: they pull against each other). A part a term does not
    reach is missing from its norms. One extra backward pass per term, so it runs only now and then.
    """
    params, parts = _trainable(model)
    grads = {term: torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
             for term, value in terms.items() if value.requires_grad}
    squares = {term: _sums_by_part(parts, g, g) for term, g in grads.items()}
    norms = {term: {part: float(v.sqrt()) for part, v in sq.items()} for term, sq in squares.items()}
    cosine: Dict[str, Dict[str, float]] = {}
    names = list(grads)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            dots = _sums_by_part(parts, grads[a], grads[b])
            cos = {part: float(dot / (squares[a][part] * squares[b][part]).sqrt().clamp_min(1e-30))
                   for part, dot in dots.items() if part in ENCODER_PARTS + ("encoder",)}
            if cos:
                cosine[f"{a}|{b}"] = cos
    return {"norms": norms, "cosine": cosine}


def gradient_norms(model: lje.LabyrinthJEPA) -> Dict[str, float]:
    """The norm of the accumulated gradient (.grad) on each part of the model."""
    params, parts = _trainable(model)
    grads = [p.grad for p in params]
    return {part: float(v.sqrt()) for part, v in _sums_by_part(parts, grads, grads).items()}


def _gradient_line(norms: Dict[str, Dict[str, float]]) -> str:
    def by_part(n: Dict[str, float]) -> str:
        return "/".join(f"{n[p]:.2g}" if p in n else "0" for p in ENCODER_PARTS)
    return "  ".join(f"{term} {by_part(n)}" for term, n in norms.items())


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


def snapshot_due(step: int, first: int) -> bool:
    """Snapshots at step 0 and at first, 2 * first, 4 * first, ...: dense early, where most changes, few in all."""
    if first <= 0 or (step and step % first):
        return False
    m = step // first
    return m & (m - 1) == 0


def _snapshot(folder: str, model, step: int, config, run) -> None:
    """The weights alone (no optimizer), in a file probe_labyrinth.load_encoder reads."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"step_{step:07d}.pt")
    torch.save({"model": model.state_dict(), "step": step, "jepa_config": config.to_dict(),
                "run_config": run.to_dict()}, path + ".tmp")
    os.replace(path + ".tmp", path)


def _trim_rollout(batch: ld.JEPABatch, steps: int, targets: Optional[Sequence[int]]) -> None:
    """(eager rollouts) Cut a batch's rollout to its first `steps` moves, and take the target graphs (and
    with them the labels) off the moves that are not among `targets` (None: every move is one)."""
    rollout = batch.rollout[:steps]
    if targets is not None:
        graded = set(targets)
        rollout = [step if k in graded else replace(step, graphs=None, legal=None, threats=None)
                   for k, step in enumerate(rollout, start=1)]
    batch.rollout = rollout


def train(config: lje.JEPAConfig, run: RunConfig, log: Callable[[str], None] = print,
          stop: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """
    One training run. Returns {"final": metrics at the end, "probe": the last probe averages (or None),
    "best_lidar", "steps", "history", "stopped"}. With run.out_dir set, writes run_config.json,
    history.jsonl, probes.csv, last.pt, best.pt and snapshots/ there, and resumes from last.pt when
    run.resume is set and it exists.
    stop: asked before every step; when it answers True the run saves last.pt and returns at once, with
    "stopped" set and no final evaluation. Running the same command again carries on from there.
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
        paths = {name: os.path.join(run.out_dir, name)
                 for name in ("last.pt", "best.pt", "history.jsonl", "probes.csv", "snapshots")}
        with open(os.path.join(run.out_dir, "run_config.json"), "w") as f:
            json.dump({"jepa_config": config.to_dict(), "run_config": run.to_dict(), "batch_size": batch_size}, f, indent=2)
        if run.resume and os.path.exists(paths["last.pt"]):
            ckpt = torch.load(paths["last.pt"], map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["scheduler"])
            gen.set_state(ckpt["generator"].cpu())     # (loaded onto the run's device with the rest; a state is kept on the CPU)
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
    threats = threat_labels(config, run, batch_size, log)
    loader = ld.JEPALoader(run.cache_dir, "train", batch_size, num_workers=run.num_workers,
                           mask_config=config.mask_config(), seed=run.seed, indices=indices,
                           pin_memory=device.type == "cuda", drop_last=True, siblings=config.rollout_siblings,
                           branches=config.rollout_branches,
                           threats=threats, compact=run.compact_batches, prefetch_factor=run.prefetch_factor)
    if len(loader) == 0:
        raise ValueError(f"the training set ({len(loader.dataset)} positions) is smaller than one batch ({batch_size})")
    val_dataset = ld.PositionDataset(run.cache_dir, "val")
    eval_rows, val_rows = evaluation_positions(run, len(val_dataset), batch_size)
    eval_fens = [val_dataset.fen(i) for i in eval_rows]
    n_val = len(val_rows)
    val_loader = ld.JEPALoader(run.cache_dir, "val", batch_size, num_workers=0, mask_config=config.mask_config(),
                               seed=run.seed + 1, shuffle=False, indices=val_rows,
                               siblings=config.rollout_siblings, branches=config.rollout_branches, threats=threats,
                               compact=run.compact_batches) if n_val else None
    exam_loader = None
    if run.exam:
        # The same positions, masks and moves for every run, whatever its seed, batch size and mask settings
        # (one line of play each: no sibling moves or branches, so runs with and without them sit the same exam)
        exam_rows = random.Random(EVAL_SUBSET_SEED).sample(range(len(val_dataset)), min(run.exam_positions, len(val_dataset)))
        exam_loader = ld.JEPALoader(run.cache_dir, "val", run.exam_batch_size, num_workers=run.exam_workers,
                                    mask_config=ld.MaskConfig(), seed=EXAM_SEED, shuffle=False, indices=exam_rows,
                                    threats=threats, compact=run.compact_batches, prefetch_factor=2)

    # Evaluation and probes encode no more positions at once than a training step does: the unpool
    # variant's enlarged graphs make a few hundred positions at once far heavier than its training batch
    probes = build_probes(run, val_dataset, log, encode_batch=batch_size)
    if probes is not None and (step == 0 or not (paths and os.path.exists(paths["probes.csv"]))):
        # Baselines, once a run (at the start, or on resuming a run that has none): the untrained encoder
        # (rebuilt from the seed: the run's own starting weights) and the raw board
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(run.seed)
            untrained = lje.LabyrinthEncoder(config).to(device)
        rows = probes.probe(untrained, device, "untrained", "random init") + probes.probe(None, device, "board_onehot", "")
        del untrained
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if paths:
            _append_probe_rows(paths["probes.csv"], rows, step)
        baselines = {m: pl.TrainingProbes.summarize([r for r in rows if r["model"] == m])
                     for m in ("untrained", "board_onehot")}
        record({"step": step, "probe_baselines": baselines})
        for m, summary in baselines.items():
            for view, s in summary.items():
                log(f"[{step:>7}] probe {m} {view}: {_probe_line(s)}")
    if paths and step == 0 and snapshot_due(0, run.snapshot_first):
        _snapshot(paths["snapshots"], model, 0, config, run)

    # Where in the data and the curriculum this run starts (step 0, or a resumed step)
    epoch = step // len(loader)
    loader.set_epoch(epoch)
    loader.resume_at((step - epoch * len(loader)) * batch_size)
    epoch_start_step = epoch * len(loader)
    rollout_weights = lje.rollout_schedule(config, step * batch_size)
    if run.eager_rollouts:
        # Every batch carries the whole rollout, every move a target; each step uses what the curriculum has reached
        loader.set_rollout_steps(config.rollout_max_steps, None)
    else:
        loader.set_rollout_steps(len(rollout_weights), lje.rollout_target_steps(config, len(rollout_weights)))
    threat_weights = lje.threat_schedule(config, step * batch_size) if threats is not None else None
    batches = iter(loader)
    skips = 0
    cache_releases = 0
    total_gpu_memory = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
    t_last, positions_since = time.perf_counter(), 0
    log(f"training {config.token_variant} for {run.total_steps} steps of {batch_size} positions on {device} "
        f"({len(loader.dataset):,} training positions, {len(eval_fens)} evaluation positions)")

    last_probe: Optional[Dict[str, Any]] = None

    def run_eval() -> Dict[str, float]:
        nonlocal best_lidar, last_probe
        if probes is not None:
            t0 = time.perf_counter()
            rows = probes.probe(model.target, device)
            if paths:
                _append_probe_rows(paths["probes.csv"], rows, step)
            last_probe = pl.TrainingProbes.summarize(rows)
            record({"step": step, "probe": last_probe})
            for view, s in last_probe.items():
                log(f"[{step:>7}] probe {view}: {_probe_line(s)}")
            log(f"[{step:>7}] probes took {time.perf_counter() - t0:.0f}s")
        metrics = evaluate(model, eval_fens, val_loader, run, device, rollout_weights, encode_batch=batch_size,
                           threat_weights=threat_weights)
        if exam_loader is not None:
            t0 = time.perf_counter()
            metrics.update(exam(model, exam_loader, device))
            metrics["exam_seconds"] = time.perf_counter() - t0
        if device.type == "cuda":
            torch.cuda.empty_cache()           # hand evaluation's odd-sized blocks back before training resumes
        record({"step": step, "eval": metrics})
        log(f"[{step:>7}] eval  lidar={metrics['lidar']:.1f}  rankme={metrics['rankme']:.1f}  "
            f"rankme_centered={metrics['rankme_centered']:.1f}  online_lidar={metrics['online_lidar']:.1f}  "
            f"online_rankme_centered={metrics['online_rankme_centered']:.1f}  "
            f"self_consistency={metrics['self_consistency']:.3f}  "
            f"val_mask={metrics.get('val_loss_mask', float('nan')):.4f}  val_next={metrics.get('val_loss_next', float('nan')):.4f}")
        if "exam_loss" in metrics:
            log(f"[{step:>7}] exam  loss={metrics['exam_loss']:.4f}  " + exam_line(metrics)
                + f"  ({metrics['exam_seconds']:.0f}s)")
        if metrics["lidar"] > best_lidar:
            best_lidar = metrics["lidar"]
            if paths:
                _save(paths["best.pt"], model, opt, sched, step, best_lidar, gen, config, run)
        return metrics

    final: Dict[str, float] = {}
    stopped = False
    stop_at = run.stop_at_step if run.stop_at_step is not None and run.stop_at_step < run.total_steps else None
    while step < run.total_steps:
        if (stop is not None and stop()) or (stop_at is not None and step >= stop_at):
            stopped = True
            break
        new_weights = lje.rollout_schedule(config, step * batch_size)
        if len(new_weights) != len(rollout_weights):
            if not run.eager_rollouts:
                # The curriculum adds a move: continue this epoch with longer rollouts, without repeating positions
                loader.set_rollout_steps(len(new_weights), lje.rollout_target_steps(config, len(new_weights)))
                loader.resume_at((step - epoch_start_step) * batch_size)
                batches = iter(loader)
            log(f"[{step:>7}] rollout curriculum: now predicting {len(new_weights)} move(s) ahead")
        rollout_weights = new_weights
        if threats is not None:
            new_weights = lje.threat_schedule(config, step * batch_size)
            if sum(w > 0 for w in new_weights) != sum(w > 0 for w in threat_weights):
                log(f"[{step:>7}] threat curriculum: now grading plies 0 to {sum(w > 0 for w in new_weights) - 1}")
            threat_weights = new_weights
        try:
            batch = next(batches)
        except StopIteration:
            epoch += 1
            epoch_start_step = step
            loader.set_epoch(epoch)
            batches = iter(loader)
            continue
        if run.eager_rollouts:
            _trim_rollout(batch, len(rollout_weights), lje.rollout_target_steps(config, len(rollout_weights)))
        batch = batch.to(device, non_blocking=True)
        model.train()
        # The heads' scores (boards decoded, threats found, ...) only on the steps that are logged: scoring
        # takes a good part of a small batch's step and changes nothing in the losses
        logged = (step + 1) % run.log_every == 0 or step + 1 == run.total_steps
        out = model.losses(batch, rng=gen, rollout_weights=rollout_weights, threat_weights=threat_weights,
                           with_scores=logged)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(out.total):
            skips += 1
        else:
            by_term = None
            if run.grad_terms_every and (step + 1) % run.grad_terms_every == 0:
                by_term = gradient_by_term(model, out.terms)
            out.total.backward()
            if by_term is not None:
                by_term["norms"]["total"] = gradient_norms(model)
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
                if by_term is not None:
                    record({"step": step, "grad_terms": by_term})
                    log(f"[{step:>7}] gradient norms ({' / '.join(ENCODER_PARTS)}): {_gradient_line(by_term['norms'])}")
        if skips:
            if skips >= run.max_consecutive_skips:
                raise TrainingDiverged(f"{skips} consecutive steps with a non-finite loss or gradient at step {step}")
            continue

        if (device.type == "cuda" and run.cache_release_fraction > 0
                and torch.cuda.memory_reserved(device) > run.cache_release_fraction * total_gpu_memory):
            # The unpool variant's graphs change size every batch, so the cache fills with blocks that fit no
            # later request; left alone, reserved memory creeps past the card and spills into system memory
            torch.cuda.empty_cache()
            cache_releases += 1
        if device.type == "cuda" and (step == 3 or (run.memory_check_every and step % run.memory_check_every == 0)):
            check_memory_budget(torch.cuda.max_memory_reserved(device),
                                torch.cuda.get_device_properties(device).total_memory, run.memory_budget)

        if step % run.log_every == 0 or step == run.total_steps:
            now = time.perf_counter()
            entry = {"step": step, "epoch": epoch, "loss": float(out.total.detach()), "lr": sched.get_last_lr()[0],
                     "rollout_weights": list(rollout_weights),
                     "momentum": model.momentum_at(step, run.total_steps), "grad_norm": float(grad_norm),
                     "positions_per_s": positions_since / max(1e-9, now - t_last), **out.stats}
            if device.type == "cuda":
                entry["gpu_peak_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
                entry["gpu_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 2**30
                entry["cache_releases"] = cache_releases
            if batch.threats is not None:              # where this batch's own positions' threat labels came from
                entry["threat_weights"] = list(threat_weights)
                entry["threats_stored"] = float((batch.threats.origin == ld.THREATS_STORED).float().mean())
                entry["threats_searched"] = float((batch.threats.origin == ld.THREATS_COMPUTED).float().mean())
            record(entry)
            zpred = f"zpred={entry['loss_zpred']:.4f} " if "loss_zpred" in entry else ""
            if "loss_inverse" in entry:
                zpred += f"inverse={entry['loss_inverse']:.3f} ({entry['inverse_accuracy']:.0%}) "
            if "loss_count" in entry:
                zpred += f"count={entry['loss_count']:.4f} (off by {entry['count_error']:.2f}) "
            if "loss_legal" in entry:
                zpred += (f"legal={entry['loss_legal']:.4f} (precision {entry['legal_precision']:.0%}, "
                          f"recall {entry['legal_recall']:.0%}) ")
            if "loss_decode" in entry:
                zpred += (f"decode={entry['loss_decode']:.4f} (pieces {entry['decode_pieces']:.0%}, "
                          f"boards {entry['decode_exact']:.0%}) ")
            for name, label in (("decode_fit", "real z"), ("decode_next", "next z"), ("decode_masked", "masked z")):
                if f"loss_{name}" in entry:
                    zpred += f"decode {label}={entry[f'loss_{name}']:.3f} (pieces {entry[f'{name}_pieces']:.0%}) "
            for name, label in (("zlegal", "z"), ("zlegal_fit", "real z"), ("zlegal_next", "next z"),
                                ("zlegal_masked", "masked z")):
                if f"loss_{name}" in entry:
                    zpred += (f"legal {label}={entry[f'loss_{name}']:.3f} (found {entry[f'{name}_found']:.0%}, "
                              f"rejected {entry[f'{name}_rejected']:.0%}) ")
            for name, label in (("zthreat", "z"), ("zthreat_fit", "real z"), ("zthreat_next", "next z"),
                                ("zthreat_masked", "masked z")):
                if f"loss_{name}" in entry:
                    more = "".join(f", {word} {entry[f'{name}_{key}']:.0%}" for key, word in (("new", "new"), ("close", "counts"))
                                   if f"{name}_{key}" in entry)
                    zpred += (f"threats {label}={entry[f'loss_{name}']:.3f} (found {entry[f'{name}_found']:.0%}, "
                              f"rejected {entry[f'{name}_rejected']:.0%}{more}; {entry[f'{name}_wrong']:.0f} cells wrong "
                              f"a position per ply) ")
            log(f"[{step:>7}] loss={entry['loss']:.4f} mask={entry['loss_mask']:.4f} next={entry['loss_next']:.4f} "
                f"{zpred}K={len(rollout_weights)} "
                f"reg={entry['anticollapse']:.3f} z_std={entry['z_std']:.3f} z_cos={entry['z_mean_cosine']:.3f} "
                f"lr={entry['lr']:.2e} {entry['positions_per_s']:.0f} pos/s")
            t_last, positions_since = now, 0

        if run.eval_every and step % run.eval_every == 0 and step < run.total_steps:
            run_eval()
        if paths and run.checkpoint_every and step % run.checkpoint_every == 0:
            _save(paths["last.pt"], model, opt, sched, step, best_lidar, gen, config, run)
        if paths and (snapshot_due(step, run.snapshot_first) or (run.snapshot_first and step == run.total_steps)
                      or (run.snapshot_every and step % run.snapshot_every == 0)):
            _snapshot(paths["snapshots"], model, step, config, run)

    if stopped:
        planned = stop_at is not None and step >= stop_at
        how = "the same command with a later stop_at_step, or none, carries on" if planned \
            else "the same command carries on from here"
        log(f"[{step:>7}] " + (f"stopped at step {stop_at} as planned" if planned else "stopped on request")
            + (f": saved {paths['last.pt']}; {how}" if paths else ""))
    else:
        final = run_eval()
    if paths:
        _save(paths["last.pt"], model, opt, sched, step, best_lidar, gen, config, run)
    result = {"final": final, "probe": last_probe, "best_lidar": best_lidar, "steps": step, "batch_size": batch_size,
              "history": history, "stopped": stopped}
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ##### ##### ##### ##### #####
#       The search (DEHB)


def _release_gpu_memory() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SearchStopped(Exception):
    """A stop was asked for in the middle of an evaluation. Its checkpoint is kept, to carry on from."""


class EvaluationCrashed(RuntimeError):
    """A training run in a process of its own ended with an error that is not about memory or divergence."""


def _is_out_of_memory(error: BaseException) -> bool:
    """A GPU out of memory: PyTorch's allocator says so with OutOfMemoryError, a CUDA library that could not
    get its own workspace with some other RuntimeError ("CUDA error: out of memory")."""
    return isinstance(error, torch.cuda.OutOfMemoryError) or \
        (isinstance(error, RuntimeError) and "out of memory" in str(error).lower())


def train_in_subprocess(config: lje.JEPAConfig, run: RunConfig, log: Callable[[str], None] = print,
                        stop: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """
    train(config, run) in a process of its own, as a search runs its evaluations: a run that exhausts the
    GPU can leave CUDA unusable for the rest of its process, and a search is a hundred runs, some of them
    meant to find out what fits. The run needs run.out_dir: the job goes in as <out_dir>/evaluation.json and
    the outcome comes back as <out_dir>/evaluation_result.json. Returns what train returns (of the history,
    only HISTORY_KEPT), and raises what train raises: MemoryBudgetExceeded for any way of running out of memory,
    TrainingDiverged, and EvaluationCrashed for anything else.
    stop: asked twice a second; when it answers True the run is told through its stop file, saves and returns.
    """
    import subprocess
    import sys
    if not run.out_dir:
        raise ValueError("a run in a process of its own needs an out_dir")
    os.makedirs(run.out_dir, exist_ok=True)
    result_path = os.path.join(run.out_dir, "evaluation_result.json")
    if os.path.exists(result_path):
        os.remove(result_path)
    with open(os.path.join(run.out_dir, "evaluation.json"), "w") as f:
        json.dump({"jepa_config": config.to_dict(), "run_config": run.to_dict()}, f)
    process = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "evaluate", run.out_dir])
    told = False
    while process.poll() is None:
        if stop is not None and not told and stop():
            open(os.path.join(run.out_dir, "stop"), "w").close()
            told = True
        time.sleep(0.5)
    if not os.path.exists(result_path):
        raise EvaluationCrashed(f"the run's process ended with exit code {process.returncode} and no result")
    with open(result_path) as f:
        outcome = json.load(f)
    if outcome["status"] == "memory":
        raise MemoryBudgetExceeded(outcome["error"])
    if outcome["status"] == "diverged":
        raise TrainingDiverged(outcome["error"])
    if outcome["status"] != "done":
        raise EvaluationCrashed(outcome["error"])
    return outcome["result"]


def _evaluate_job(folder: str) -> int:
    """The other end of train_in_subprocess: run the job in `folder` and write its outcome there."""
    with open(os.path.join(folder, "evaluation.json")) as f:
        job = json.load(f)
    config = lje.JEPAConfig.from_dict(job["jepa_config"])
    run = RunConfig(**{**job["run_config"], "probe_views": tuple(job["run_config"]["probe_views"])})
    try:
        result = train(config, run, stop=stop_switch(folder, keys=False))
        outcome: Dict[str, Any] = {"status": "done", "result": {
            "final": result["final"], "steps": result["steps"], "batch_size": result["batch_size"],
            "stopped": result["stopped"],
            "history": [{key: h[key] for key in HISTORY_KEPT if key in h}
                        for h in result["history"] if "positions_per_s" in h]}}
    except TrainingDiverged as error:
        outcome = {"status": "diverged", "error": str(error)}
    except Exception as error:
        if isinstance(error, MemoryBudgetExceeded) or _is_out_of_memory(error):
            outcome = {"status": "memory", "error": f"{type(error).__name__}: {error}"}
        else:
            import traceback
            traceback.print_exc()
            outcome = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    path = os.path.join(folder, "evaluation_result.json")
    with open(path + ".tmp", "w") as f:
        json.dump(outcome, f)
    os.replace(path + ".tmp", path)
    return 0


# What the search varies: (name, kind, values, None or (the setting it depends on, the value that needs)).
# kind "choice": one of the listed values (numbers in rising order, so neighbours in the search are neighbours
# in size); "log" / "float": a number between the two, on a log or a plain scale. A name that is a JEPAConfig
# field goes into the configuration as it is; the others (SEARCH_ONLY) are the search's own, resolved by
# resolve_search. The heads are NOT here, nor are the loss weights: the exam is the heads' losses, so a
# search that could enlarge the heads or reweigh the terms would improve its score without improving z.
SEARCH_DIMENSIONS: Tuple[Tuple[str, str, Any, Optional[Tuple[str, Any]]], ...] = (
    # the square stage (gat_style "concat": every layer's per-square MLP is gat_growth times as wide as its input)
    ("gat_layers", "choice", (2, 3, 4), None),
    ("gat_dim", "choice", (64, 128, 192), None),
    ("gat_heads", "choice", (2, 4, 8), None),
    ("gat_growth", "choice", (1.0, 1.25, 1.5), None),
    ("gat_mlp_layers", "choice", (1, 2, 3), None),
    # the square tokens, the readout MLP and z
    ("d_model", "choice", (128, 256, 384), None),
    ("readout_hidden", "choice", (1024, 2048, 4096), None),
    ("readout_layers", "choice", (1, 2, 3), None),
    ("z_dim", "choice", (512, 1024, 2048), None),
    # unpooling: whether, how many layers, how wide its MLPs and its nodes, how hard its decisions are trained
    ("token_variant", "choice", ("squares", "unpool"), None),
    ("unpool_layers", "choice", (1, 2, 3), ("token_variant", "unpool")),
    ("unpool_hidden", "choice", (64, 128, 256), ("token_variant", "unpool")),
    ("unpool_dim", "choice", (64, 128), ("token_variant", "unpool")),
    ("rl_coef", "log", (1e-3, 1e-1), ("token_variant", "unpool")),
    # the next-position predictor, which has to carry a rollout to its last move
    ("next_hidden", "choice", (1024, 2048), None),
    ("next_layers", "choice", (2, 3, 4), None),
    # training
    ("lr", "log", (1e-4, 1e-3), None),
    ("batch_size", "choice", (64, 128, 256), None),
    ("sigreg_weight", "log", (0.002, 0.05), None),
    ("rollout_curriculum", "float", (0.0, 0.8), None),
    ("threat_curriculum", "float", (0.0, 0.5), None),
)
# The search's own settings (not JEPAConfig fields):
#   batch_size           positions a step. Every evaluation gets the same number of POSITIONS, so a smaller batch
#                        takes more steps on the same data
#   rollout_curriculum   the share of the run after which the rollout has grown to its full length (0: full
#                        length from the first step). A share, not a number of positions: every evaluation, short
#                        or long, then finishes its curriculum and is examined on what it trained on
#   threat_curriculum    the same for the legal-threat head's plies
SEARCH_ONLY = ("batch_size", "rollout_curriculum", "threat_curriculum")
SIGREG_BATCH = 256              # sigreg_weight and ema_start are given for this batch size (see resolve_search)
WARMUP_POSITIONS = 256_000      # the learning rate's warm-up, in positions (1,000 steps of 256), at most a tenth of a run
FAILED = 1e3                    # the exam loss given to a configuration that does not fit, is not valid or diverges
# The largest batch an unpooling configuration starts at, by its number of unpooling layers: the enlarged graphs
# take several times the memory of the squares alone, and finding that out by running out of it costs minutes.
# (What fits a 16 GB card at the smallest unpooling sizes; the memory guard still halves the batch when needed.)
UNPOOL_MAX_BATCH = {1: 128, 2: 64, 3: 32}


def build_search_space(seed: int = 0, fixed: Optional[Dict[str, Any]] = None):
    """
    The DEHB search space: SEARCH_DIMENSIONS without the settings `fixed` gives a value for. A setting that
    only applies under one value of another (unpool_layers under token_variant "unpool") is left out when
    the other is fixed to something else, and kept unconditionally when it is fixed to that value.
    """
    import ConfigSpace as CS
    fixed = fixed or {}
    cs = CS.ConfigurationSpace(seed=seed)
    kept: Dict[str, Tuple[Any, Optional[Tuple[str, Any]]]] = {}
    for name, kind, values, condition in SEARCH_DIMENSIONS:
        if name in fixed:
            continue
        if condition is not None and condition[0] in fixed and fixed[condition[0]] != condition[1]:
            continue
        if kind == "choice":
            hp = CS.CategoricalHyperparameter(name, choices=list(values))
        else:
            hp = CS.UniformFloatHyperparameter(name, lower=values[0], upper=values[1], log=kind == "log")
        cs.add_hyperparameter(hp)
        kept[name] = (hp, condition)
    for hp, condition in kept.values():
        if condition is not None and condition[0] in kept:
            cs.add_condition(CS.EqualsCondition(hp, kept[condition[0]][0], condition[1]))
    return cs


def resolve_search(point: Dict[str, Any], positions: int,
                   batch_size: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    A point of the search space (with the fixed values merged in) -> (JEPAConfig values, {"batch_size",
    "warmup_steps"}) for a run of `positions` training positions. batch_size: use this one instead of the
    point's (a configuration that did not fit is tried again at half).

      * the curricula's shares of the run become stage lengths in positions;
      * sigreg_weight is taken as the weight at SIGREG_BATCH positions a batch and scaled by SIGREG_BATCH /
        batch size: SIGReg's statistic is multiplied by the batch size, so its pull on one position does not
        shrink as the batch grows, while that of the other losses (means over the batch) does. Unscaled, a
        smaller batch would also be a weaker regularizer;
      * ema_start is taken as the target encoder's momentum at SIGREG_BATCH positions a batch too, and set so
        that the target averages over the same number of POSITIONS at any batch size (a quarter of the batch:
        four times the steps, each moving the target a quarter as far);
      * the warm-up is WARMUP_POSITIONS positions, whatever the batch size;
      * an unpooling configuration's batch is at most UNPOOL_MAX_BATCH for its number of layers.
    So the batch size changes how many gradient steps the same data makes, and nothing else on the way.
    """
    point = {k: (v.item() if hasattr(v, "item") else v) for k, v in point.items()}
    values = {k: v for k, v in point.items() if k not in SEARCH_ONLY}
    for name, _, _, condition in SEARCH_DIMENSIONS:
        # a setting that only applies under another's value (a fixed unpool_dim, say) is left at its default
        # where it does not apply: the configuration is then valid whichever way the other went
        if condition is not None and name in values and lje.JEPAConfig.from_dict(values).to_dict()[condition[0]] != condition[1]:
            del values[name]
    base = lje.JEPAConfig.from_dict(values)
    batch = int(batch_size or point.get("batch_size") or DEFAULT_BATCH_SIZE[base.token_variant])
    if base.uses_unpool and not batch_size:
        batch = min(batch, UNPOOL_MAX_BATCH.get(base.unpool_layers, min(UNPOOL_MAX_BATCH.values())))
    if "rollout_curriculum" in point:
        stages = base.rollout_max_steps - base.rollout_start_steps
        values["rollout_stage_positions"] = int(round(point["rollout_curriculum"] * positions / stages)) if stages > 0 else 0
    if "threat_curriculum" in point:
        stages = base.zthreat_plies - base.zthreat_start_plies
        values["zthreat_stage_positions"] = int(round(point["threat_curriculum"] * positions / stages)) if stages > 0 else 0
    values["sigreg_weight"] = base.sigreg_weight * SIGREG_BATCH / batch
    values["ema_start"] = 1.0 - (1.0 - base.ema_start) * batch / SIGREG_BATCH
    return values, {"batch_size": batch, "warmup_steps": max(1, round(WARMUP_POSITIONS / batch))}


def _plain(value: Any) -> Any:
    """Numbers as plain Python ones (rounded) and tuples as lists, so a value read back from a file compares equal."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    return round(value, 12) if isinstance(value, float) else value


# What an evaluation in a process of its own hands back of each logged step
HISTORY_KEPT = ("step", "positions_per_s", "tokens", "unpool_entropy_per_decision")


def unpool_course(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    What the unpooling did over a run, from its logged steps: the nodes a position's graph has after unpooling
    (64 before it; at most 64 x 2 ** unpool_layers) and the entropy of an unpooling decision (about 1.0 for an
    untrained layer: most decisions are three-way choices about an edge, 1.10 when the three are equally
    likely; it falls as the decisions settle), each at the first logged step, half-way through, and over the
    run's ninth and last tenth (so the last two say which way it was still going). {} when nothing was logged.
    """
    logged = [h for h in history if "tokens" in h and "unpool_entropy_per_decision" in h]
    if not logged:
        return {}

    def mean(key: str, start: float, end: float) -> float:
        rows = logged[int(start * len(logged)):max(int(start * len(logged)) + 1, int(end * len(logged)))]
        return sum(h[key] for h in rows) / len(rows)

    course: Dict[str, float] = {}
    for name, key, shift in (("unpool_nodes", "tokens", -64.0), ("unpool_entropy", "unpool_entropy_per_decision", 0.0)):
        course.update({f"{name}_first": logged[0][key] + shift, f"{name}_half": mean(key, 0.45, 0.55) + shift,
                       f"{name}_ninth_tenth": mean(key, 0.8, 0.9) + shift, f"{name}_last_tenth": mean(key, 0.9, 1.0) + shift})
    return course


def make_objective(base_run: RunConfig, fixed: Optional[Dict[str, Any]] = None,
                   train_fn: Callable[..., Dict[str, Any]] = train, log: Callable[[str], None] = print,
                   work_dir: Optional[str] = None, stop: Optional[Callable[[], bool]] = None,
                   min_batch_size: int = 16, retry_wait: float = 0.0):
    """
    The DEHB objective: a point of the search space and a fidelity (thousands of training positions) -> the
    exam loss of a run trained from scratch on that many positions (exam(): lower is better). Fidelity counts
    positions, not steps, so every batch size gets the same data. fixed: settings every evaluation shares; they
    override anything sampled.

    Every evaluation runs at a constant learning rate after the warm-up, as a long run will, and sits the exam
    once, at its end. One that does not fit in memory is tried again at half the batch size (after retry_wait
    seconds), down to min_batch_size; one that never fits, is not a valid configuration, diverges or fails
    some other way gets FAILED. train_fn: train, or train_in_subprocess to give every evaluation a process
    of its own.

    work_dir: where the evaluation under way keeps its checkpoint (<work_dir>/current). stop: passed to the
    training run; when it stops a run, the checkpoint stays and SearchStopped is raised, and the next call
    with the same point and fidelity carries on from the checkpoint instead of starting over.
    """
    from dehb_helper import ObjectiveResult
    import shutil

    current = os.path.join(work_dir, "current") if work_dir else None
    crashes = {"in a row": 0}

    def objective(cfg: Dict[str, Any], fidelity: float) -> ObjectiveResult:
        positions = int(round(fidelity * 1000))
        point = _plain({**cfg, **(fixed or {})})
        job = {"point": point, "positions": positions}
        batch_size: Optional[int] = None if "batch_size" in point else base_run.batch_size   # (the point's, if it has one)
        resume = False
        if current:
            job_path = os.path.join(current, "job.json")
            if os.path.exists(job_path) and os.path.exists(os.path.join(current, "last.pt")):
                with open(job_path) as f:
                    kept = json.load(f)
                if kept.get("point") == point and kept.get("positions") == positions:
                    resume, batch_size = True, kept["batch_size"]       # the evaluation that was stopped
            if not resume:
                shutil.rmtree(current, ignore_errors=True)

        started = time.perf_counter()
        while True:
            try:
                values, how = resolve_search(point, positions, batch_size)
                config = lje.JEPAConfig.from_dict(values)
                config.validate()
            except ValueError as e:
                return ObjectiveResult(metric=FAILED, info={"failed": "invalid", "error": str(e)})
            batch_size = how["batch_size"]
            if current:
                os.makedirs(current, exist_ok=True)
                with open(os.path.join(current, "job.json"), "w") as f:
                    json.dump({**job, "batch_size": batch_size}, f)
            run = replace(base_run, out_dir=current, resume=resume, batch_size=batch_size,
                          total_steps=max(1, positions // batch_size), warmup_steps=how["warmup_steps"],
                          min_lr_fraction=1.0, eval_every=0, checkpoint_every=0, val_loss_batches=0,
                          probe_positions=0, grad_terms_every=0, snapshot_first=0, snapshot_every=0,
                          exam=True, eager_rollouts=True)
            failure = None
            try:
                result = train_fn(config, run, log=log, stop=stop)
            except TrainingDiverged as e:
                return ObjectiveResult(metric=FAILED, info={"failed": "diverged", "error": str(e), "batch_size": batch_size})
            except EvaluationCrashed as e:
                # One odd configuration must not end a search; the same failure three times running is not a
                # configuration's fault (a missing file, a bug), and scoring on would only waste the search
                crashes["in a row"] += 1
                log(f"  the evaluation failed: {e}")
                if crashes["in a row"] >= 3:
                    raise
                return ObjectiveResult(metric=FAILED, info={"failed": "crashed", "error": str(e), "batch_size": batch_size})
            except RuntimeError as e:
                if not (isinstance(e, MemoryBudgetExceeded) or _is_out_of_memory(e)):
                    raise
                failure = f"{type(e).__name__}: {e}"
            crashes["in a row"] = 0
            if failure is not None:
                _release_gpu_memory()      # outside the except block, once the failed run's frames are gone
                time.sleep(retry_wait)     # the driver takes a moment to hand a dead run's memory back
                log(f"  does not fit at batch size {batch_size}: {failure}")
                if current:
                    shutil.rmtree(current, ignore_errors=True)
                if batch_size // 2 >= min_batch_size:
                    batch_size, resume = batch_size // 2, False
                    continue
                return ObjectiveResult(metric=FAILED, info={"failed": "memory", "batch_size": batch_size})
            if result.get("stopped"):
                raise SearchStopped()
            if current:                    # done: nothing of it is to be taken up again
                shutil.rmtree(current, ignore_errors=True)
            final = result["final"]
            if not math.isfinite(final.get("exam_loss", math.nan)):
                return ObjectiveResult(metric=FAILED, info={"failed": "no exam loss", "batch_size": batch_size,
                                                            "error": "the run has no finite exam loss (no head graded "
                                                                     "against the board, or the exam went non-finite)"})
            rates = [h["positions_per_s"] for h in result["history"] if "positions_per_s" in h]
            info: Dict[str, Any] = {key: final[key] for key in sorted(final) if key.startswith("exam_loss")}
            info.update({key: final[key] for key in sorted(final) if key.startswith("exam_") and key.endswith(
                ("_exact", "_wrong", "_close", "_found", "_rejected"))})
            # (and how decoding the predicted next z holds up along the rollout, move by move)
            info.update({key: final[key] for key in sorted(final) if key.startswith("exam_decode_next_exact_")})
            info.update({
                "online_lidar": final.get("online_lidar"), "online_rankme_centered": final.get("online_rankme_centered"),
                "self_consistency": final.get("self_consistency"),
                "batch_size": batch_size, "steps": result["steps"], "positions": positions,
                "positions_per_s": sorted(rates)[len(rates) // 2] if rates else None,
                "seconds": time.perf_counter() - started,
                # how to run this configuration again: train --config <the search's best_config.json> reads it
                "run": {"jepa_config": config.to_dict(), "batch_size": batch_size, "warmup_steps": how["warmup_steps"]},
            })
            if config.uses_unpool:         # how far the graphs grew and whether the decisions settled
                info.update(unpool_course(result["history"]))
            return ObjectiveResult(metric=float(final["exam_loss"]), info=info)

    return objective


# ##### ##### ##### ##### #####
#       Command line


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    """
    JEPAConfig values from a file: a plain dict of them, a run's run_config.json ({"jepa_config": {...}}), or
    a search's best_config.json (the configuration its best evaluation ran, curricula and all).
    """
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    return (data.get("best_info") or {}).get("run", {}).get("jepa_config") or data.get("best_config") \
        or data.get("jepa_config") or data


def _load_run_settings(path: Optional[str]) -> Dict[str, int]:
    """The batch size and warm-up a configuration file was run with, where it says: a search's
    best_config.json (both), a run's run_config.json (the batch size)."""
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    found = (data.get("best_info") or {}).get("run") or data
    return {key: int(found[key]) for key in ("batch_size", "warmup_steps") if isinstance(found.get(key), (int, float))}


def stop_switch(folder: Optional[str], log: Callable[[str], None] = print, keys: bool = True) -> Callable[[], bool]:
    """
    A run's stop switch, as the labelling programs have: it answers True once q has been pressed in this
    console window (only this one: typing elsewhere is safe; keys=False: it does not listen), or once a file
    named "stop" appears in `folder` (for a run started without a console; the file is removed when seen).
    """
    import threading
    event = threading.Event()
    if keys:
        from reach_labeler import watch_for_q
        watch_for_q(event)
    stop_file = os.path.join(folder, "stop") if folder else None
    said = False

    def stop() -> bool:
        nonlocal said
        if stop_file and not event.is_set() and os.path.exists(stop_file):
            os.remove(stop_file)
            event.set()
        if event.is_set() and not said:
            said = True
            log("stop requested: saving and closing...")
        return event.is_set()

    return stop


def _read_history(out_dir: str) -> List[Dict[str, Any]]:
    """The whole history.jsonl, including the entries from before a resume."""
    path = os.path.join(out_dir, "history.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _plot(history: List[Dict[str, Any]], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps = [h for h in history if "loss" in h]
    evals = [h for h in history if "eval" in h]
    grads = [h for h in history if "grad_terms" in h]
    probes = [h for h in history if "probe" in h]
    baselines = next((h["probe_baselines"] for h in reversed(history) if "probe_baselines" in h), {})
    fig, ((a0, a1, a2), (b0, b1, b2)) = plt.subplots(2, 3, figsize=(15, 8))
    a0.plot([h["step"] for h in steps], [h["loss_mask"] for h in steps], label="masked squares")
    a0.plot([h["step"] for h in steps], [h["loss_next"] for h in steps], label="next position")
    if any("loss_zpred" in h for h in steps):
        a0.plot([h["step"] for h in steps], [h.get("loss_zpred", math.nan) for h in steps], label="whole board z")
    a0.set_title("Training losses"); a0.set_xlabel("step"); a0.legend()
    a1.plot([h["step"] for h in evals], [h["eval"]["lidar"] for h in evals], marker="o", label="LiDAR")
    a1.plot([h["step"] for h in evals], [h["eval"]["rankme_centered"] for h in evals], marker="o",
            label="RankMe (centred)")
    a1.set_title("Representation rank (target encoder)"); a1.set_xlabel("step"); a1.legend()
    a2.plot([h["step"] for h in evals], [h["eval"]["self_consistency"] for h in evals], marker="o")
    a2.set_title("Self-consistency (lower = more consistent)"); a2.set_xlabel("step")

    for term in sorted({t for h in grads for t in h["grad_terms"]["norms"]}):
        b0.plot([h["step"] for h in grads],
                [h["grad_terms"]["norms"].get(term, {}).get("encoder", math.nan) for h in grads], label=term)
    b0.set_yscale("log"); b0.set_title("Gradient norm on the encoder, by loss term"); b0.set_xlabel("step")
    if grads:
        b0.legend(fontsize=7)
    has_eval = any("labeled/eval_win_prob" in v for h in probes for v in h["probe"].values())
    panels = [(b1, "val/material_balance", "linear", "Probe: material balance, linear R^2"),
              (b2, "labeled/eval_win_prob", "knn", "Probe: engine eval, nearest-neighbour R^2") if has_eval else
              (b2, "val/material_balance", "knn", "Probe: material balance, nearest-neighbour R^2")]
    views = list(dict.fromkeys(view for h in probes for view in h["probe"]))
    for ax, key, probe, title in panels:
        for i, view in enumerate(views):
            ax.plot([h["step"] for h in probes], [h["probe"].get(view, {}).get(key, {}).get(probe, math.nan)
                                                  for h in probes], marker="o", color=f"C{i}", label=view)
            base = baselines.get("untrained", {}).get(view, {}).get(key, {}).get(probe)
            if base is not None:
                ax.axhline(base, color=f"C{i}", linestyle="--", linewidth=1)
        ax.set_title(title + " (dashed: untrained)", fontsize=10); ax.set_xlabel("step")
        if views:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training.png"), dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or tune the Labyrinth JEPA encoder.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "tune"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=None,
                       help="JEPAConfig values (JSON): a plain dict, a run's run_config.json or a search's best_config.json")
        p.add_argument("--variant", choices=lje.TOKEN_VARIANTS, default=None)
        p.add_argument("--cache", default=ld.DEFAULT_CACHE_DIR)
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--max-train-positions", type=int, default=None)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--threat-db", default=RunConfig.threat_db,
                       help="the labels database reach_labeler.py fills, for the legal-threat head")
        p.add_argument("--exam-workers", type=int, default=RunConfig.exam_workers,
                       help="loader processes for the exam's positions (their threat labels are searched on the spot)")
        p.add_argument("--no-stop-key", action="store_true",
                       help="do not listen for q in this window (a file named 'stop' in --out still stops the run)")
    t = sub.choices["train"]
    t.add_argument("--min-lr-fraction", type=float, default=RunConfig.min_lr_fraction,
                   help="the cosine decays to this fraction of the peak learning rate; 1 keeps it constant after the warm-up")
    t.add_argument("--warmup-steps", type=int, default=None,
                   help="warm-up steps (default: what the --config file was run with, else 1,000); at most a tenth of the run")
    t.add_argument("--steps", type=int, default=20_000)
    t.add_argument("--out", default="runs/labyrinth_jepa")
    t.add_argument("--eval-every", type=int, default=2_000)
    t.add_argument("--checkpoint-every", type=int, default=RunConfig.checkpoint_every,
                   help="save last.pt every this many steps (a stop always saves it)")
    t.add_argument("--no-resume", action="store_true")
    t.add_argument("--stop-at", type=int, default=None,
                   help="save and close at this step (before --steps): the first part of a longer run, on that run's "
                        "schedules; the same command with a later --stop-at, or none, carries on")
    t.add_argument("--exam", action="store_true",
                   help="every evaluation also sits the exam: the heads' losses with the whole rollout and every "
                        "threat ply at full weight, whatever the curricula have reached")
    t.add_argument("--eager-rollouts", action="store_true",
                   help="batches always carry the full rollout, so the loader is not restarted when the curriculum adds a move")
    t.add_argument("--probe-db", default=None, help="minotaur_data.db: engine evaluations and best moves to probe")
    t.add_argument("--probe-puzzles", default=None, help="lichess_db_puzzle.csv.zst: puzzle themes to probe")
    t.add_argument("--probe-positions", type=int, default=8_000, help="positions per probe set (0: no probes)")
    t.add_argument("--probe-views", default=",".join(RunConfig.probe_views),
                   help=f"comma-separated, from: {', '.join(pl.VIEWS)}")
    t.add_argument("--grad-terms-every", type=int, default=200, help="per-term gradient norms (0: never)")
    t.add_argument("--snapshot-first", type=int, default=250,
                   help="snapshots at step 0, this, twice this, ... and at the end (0: none)")
    t.add_argument("--snapshot-every", type=int, default=0, help="also a snapshot at every multiple of this (0: no)")
    t.add_argument("--prefetch-factor", type=int, default=RunConfig.prefetch_factor,
                   help="batches each loader process keeps ready")
    d = sub.choices["tune"]
    d.add_argument("--out", default="dehb_jepa_out")
    d.add_argument("--fevals", type=int, default=None, help="stop after this many evaluations in all (earlier starts included)")
    d.add_argument("--hours", type=float, default=None,
                   help="stop asking for new evaluations after this many hours of this start (the one under way finishes)")
    d.add_argument("--max-fidelity", type=float, default=3000,
                   help="the longest evaluations, in thousands of training positions")
    d.add_argument("--rungs", type=int, default=4,
                   help="how many lengths of evaluation there are: each one --eta times the one below")
    d.add_argument("--eta", type=int, default=3, help="each rung keeps one in this many and trains them this many times longer")
    d.add_argument("--prefetch-factor", type=int, default=2, help="batches each loader process keeps ready")
    d.add_argument("--in-process", action="store_true",
                   help="run the evaluations in this process instead of giving each a process of its own")
    e = sub.add_parser("evaluate", help="(used by tune) run the evaluation whose job is in this folder")
    e.add_argument("folder")
    args = parser.parse_args()
    if args.command == "evaluate":
        raise SystemExit(_evaluate_job(args.folder))

    values = _load_config(args.config)
    how = _load_run_settings(args.config)
    if args.variant:
        values["token_variant"] = args.variant
    run = RunConfig(cache_dir=args.cache, batch_size=args.batch_size or how.get("batch_size"), num_workers=args.workers,
                    max_train_positions=args.max_train_positions, seed=args.seed, threat_db=args.threat_db,
                    exam_workers=args.exam_workers, prefetch_factor=args.prefetch_factor)

    if args.command == "train":
        config = lje.JEPAConfig.from_dict({k: v for k, v in values.items() if k not in SEARCH_ONLY})
        run = replace(run, out_dir=args.out, total_steps=args.steps, eval_every=args.eval_every,
                      min_lr_fraction=args.min_lr_fraction,
                      warmup_steps=args.warmup_steps or how.get("warmup_steps") or RunConfig.warmup_steps,
                      checkpoint_every=args.checkpoint_every, resume=not args.no_resume, stop_at_step=args.stop_at,
                      exam=args.exam, eager_rollouts=args.eager_rollouts,
                      probe_db=args.probe_db, probe_puzzles=args.probe_puzzles,
                      probe_positions=args.probe_positions,
                      probe_views=tuple(v.strip() for v in args.probe_views.split(",") if v.strip()),
                      grad_terms_every=args.grad_terms_every, snapshot_first=args.snapshot_first,
                      snapshot_every=args.snapshot_every)
        os.makedirs(args.out, exist_ok=True)
        if not args.no_stop_key:
            print(f"Press q in this window to stop: the run saves and closes, and the same command carries on "
                  f"from there. (Or create a file named 'stop' in {args.out}.)")
        result = train(config, run, stop=stop_switch(args.out, keys=not args.no_stop_key))
        _plot(_read_history(args.out) or result["history"], args.out)
        print(json.dumps({"final": result["final"], "best_lidar": result["best_lidar"], "steps": result["steps"],
                          "stopped": result["stopped"]}, indent=2))
    else:
        from dehb_helper import DEHBHelper, DEHBRunConfig
        if args.fevals is None and args.hours is None:
            raise SystemExit("say how long to search: --fevals (evaluations in all) and / or --hours")
        known = {f.name for f in dataclasses.fields(lje.JEPAConfig)} | set(SEARCH_ONLY)
        unknown = sorted(set(values) - known)
        if unknown:
            raise SystemExit(f"--config has keys the search does not know: {unknown}")
        if run.batch_size:
            values["batch_size"] = run.batch_size       # (--batch-size fixes it: it is then not searched)
            run = replace(run, batch_size=None)
        space = build_search_space(seed=args.seed, fixed=values)
        print(f"Searching {len(space.get_hyperparameters())} settings: {', '.join(space.get_hyperparameter_names())}")
        print(f"Fixed for this search (not searched): {values}")
        os.makedirs(args.out, exist_ok=True)
        resume = os.path.exists(os.path.join(args.out, "dehb_state.json"))
        if resume:
            print(f"Found an earlier search in {args.out}/ - carrying on with it (delete the folder to start afresh).")
        if not args.no_stop_key:
            print(f"Press q in this window to stop: the evaluation under way is saved, and the same command carries "
                  f"on from there. (Or create a file named 'stop' in {args.out}.)")
        stop = stop_switch(args.out, keys=not args.no_stop_key)
        helper = DEHBHelper(
            configspace=space,
            objective=make_objective(run, fixed=values, work_dir=args.out, stop=stop, retry_wait=5.0,
                                     train_fn=train if args.in_process else train_in_subprocess),
            direction="minimize",
            # (a hair under the exact ratio: DEHB counts the rungs by cutting a logarithm off)
            run_cfg=DEHBRunConfig(min_fidelity=args.max_fidelity / args.eta ** (args.rungs - 1) * 0.999999,
                                  max_fidelity=args.max_fidelity, eta=args.eta,
                                  seed=args.seed, n_workers=1, output_path=args.out, resume=resume,
                                  save_freq="step", best_at_highest_fidelity=True),
        )
        summary = helper.run_stoppable(stop, fevals=args.fevals,
                                       seconds=args.hours * 3600 if args.hours else None, interrupted=SearchStopped)
        print(json.dumps(summary, indent=2, default=str))
        print(f"\nTrain the best configuration with:  python train_labyrinth_jepa.py train --config {summary['best_config_path']}")


if __name__ == "__main__":
    main()
