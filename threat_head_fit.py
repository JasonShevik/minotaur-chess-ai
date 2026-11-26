"""
What can each form of the legal-threat head work out, and from what?

    python threat_head_fit.py [name=run_folder:snapshot_step ...] [--heads linear,bilinear,piece_mlp]
        [--plies 2] [--mode count] [--positions 60000] [--steps 4000] [--db minotaur_reach.db]
        [--cache labyrinth_data_cache] [--out fit.json]

labyrinth_jepa_encoder.ZThreatHead is fitted alone, on inputs that stay fixed, to the legal-threat labels
(piece_reach.threat_counts), and scored on positions it was not fitted on. The inputs:

    one-hot board   for each of the 64 squares, which of 14 things stands on it (as the position decoder tells
                    them apart): a plain copy of the position. What a head scores on it, the head works out by
                    itself; what it misses is what z would have to supply.
    raw board       the 64 squares' node features as the encoder is given them
    name=run:step   the z of a trained encoder (that snapshot's target encoder; --online: its online encoder):
                    how much of the threats z makes readable for this form of head. A head on z that beats
                    the same head on the board means the encoder has made something explicit; a strong head
                    that does as well on the board as on z shows only that the head needs no help.

The head reads its input the way it reads z in training (layer-normalized, nothing else), and its three forms
are JEPAConfig.zthreat_head's: "linear", "bilinear", "piece_mlp" (labyrinth_jepa_encoder's docstring).

Positions. The head is fitted on labelled positions of the training split, taken from the labels database in
the order reach_labeler.py wrote them (a training run's order). It is scored on positions of the validation
split, which no encoder was trained on; their labels are searched on the spot, so that needs the compiled
search (build_fast_reach.py). Without it, or with --cache "", the last tenth of the database's positions is
held out instead.

Scores, per ply of play (0: the position as it stands) and on the held-out positions: found (threats the head
says yes to), rejected (non-threats it says no to), new (found, among the threats that were not there a ply
earlier), wrong (cells answered wrongly, per position; some nine cells in ten are non-threats, so this says
more than the two percentages), exact (positions with every cell right) and, in count mode, close (the number
of sequences within 10%).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import random
import sqlite3
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

import labyrinth_data as ld
import labyrinth_jepa_encoder as lje
import piece_reach as pr
import probe_heads as ph
import probe_labyrinth as pl

Log = Callable[[str], None]


# ##### ##### ##### ##### #####
#       Positions and their labels


def stored_positions(db_path: str, plies: int, limit: int) -> Tuple[List[str], List[ld.PositionThreats]]:
    """The first `limit` positions the database holds legal-threat labels for, at least `plies` deep, with them."""
    conn = sqlite3.connect(pathlib.Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    fens, labels = [], []
    query = f"SELECT position, counts FROM {ld.THREAT_TABLE} WHERE depth >= ? ORDER BY rowid LIMIT ?"
    for position, blob in conn.execute(query, (plies, limit)):
        fen = position + " 0 1"
        white = position.split()[1] == "w"
        fens.append(fen)
        labels.append(ld.PositionThreats(*ld.threat_cells(pr.unpack_counts(blob)[:plies + 1], white), ld.THREATS_STORED, plies + 1))
    conn.close()
    return fens, labels


def searched_positions(cache_dir: str, split: str, plies: int, limit: int, seed: int = 0) -> Tuple[List[str], List[ld.PositionThreats]]:
    """`limit` positions of a split of the position cache, drawn at random, with labels searched on the spot."""
    dataset = ld.PositionDataset(cache_dir, split)
    rows = sorted(random.Random(seed).sample(range(len(dataset)), min(limit, len(dataset))))
    fens = [dataset.fen(i) for i in rows]
    labels = [ld.PositionThreats(*ld.threat_cells(pr.threat_counts(fen, plies), fen.split()[1] == "w"),
                                 ld.THREATS_COMPUTED, plies + 1) for fen in fens]
    return fens, labels


# ##### ##### ##### ##### #####
#       Fitting one head


def fit(x: torch.Tensor, pieces: torch.Tensor, labels: Sequence[ld.PositionThreats], test_x: torch.Tensor,
        test_pieces: torch.Tensor, test_labels: Sequence[ld.PositionThreats], head: str, mode: str, plies: int,
        device, steps: int = 4000, batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 0.01,
        seed: int = 0, config: Optional[lje.JEPAConfig] = None) -> Dict[str, float]:
    """
    Fit one ZThreatHead of this form on (x, labels) and score it on the test positions. x [N, dim] (any float
    type), pieces [N, 64] bool (the squares with a piece: the rows that are graded). AdamW, one cycle of the
    learning rate, `steps` batches drawn at random. Returns threat_scores' scores averaged over the test
    positions, with "params" and "seconds". config: the head's sizes (zthreat_dim, zthreat_hidden) are taken
    from it.
    """
    torch.manual_seed(seed)
    base = config if config is not None else lje.JEPAConfig()
    cfg = dataclasses.replace(base, z_dim=x.size(1), zthreat_weight=1.0, zthreat_head=head, zthreat_mode=mode,
                              zthreat_plies=plies, zthreat_start_plies=min(base.zthreat_start_plies, plies))
    model = lje.ZThreatHead(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    generator = torch.Generator().manual_seed(seed)
    started = time.time()

    def batch(inputs, squares, items, rows) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wanted = ld.ThreatBatch.pack([items[i] for i in rows]).to(device)
        index = torch.as_tensor(rows)
        return inputs[index].to(device).float(), wanted.dense(), squares[index].to(device)

    for _ in range(steps):
        rows = torch.randint(len(labels), (batch_size,), generator=generator).tolist()
        inputs, target, graded = batch(x, pieces, labels, rows)
        loss, _ = lje.threat_scores(model(inputs), target, graded, None, cfg.zthreat_count_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    sums: Dict[str, float] = {}
    chunks = 0
    with torch.no_grad():
        for lo in range(0, len(test_labels), 512):
            inputs, target, graded = batch(test_x, test_pieces, test_labels, list(range(lo, min(lo + 512, len(test_labels)))))
            loss, scores = lje.threat_scores(model(inputs), target, graded, None, cfg.zthreat_count_weight)
            for key, value in {"loss": float(loss), **scores}.items():
                sums[key] = sums.get(key, 0.0) + value
            chunks += 1
    result = {key: value / max(chunks, 1) for key, value in sums.items()}
    result.update(params=float(lje.count_parameters(model)), seconds=time.time() - started)
    return result


def table(results: Dict[str, Dict[str, Dict[str, float]]], plies: int, mode: str) -> str:
    """{input: {head: scores}} as a markdown table, a row per input and head."""
    columns: List[Tuple[str, str]] = []
    for k in range(plies + 1):
        columns += [(f"found_{k}", f"{k} plies: found"), (f"rejected_{k}", "rejected")]
        if k > 0:
            columns.append((f"new_{k}", "new"))
        columns += [(f"wrong_{k}", "wrong"), (f"exact_{k}", "exact")]
        if k > 0 and mode == "count":
            columns.append((f"close_{k}", "counts"))
    lines = ["| input | head | " + " | ".join(title for _, title in columns) + " |", "|---|---|" + "---|" * len(columns)]
    for source, by_head in results.items():
        for head, scores in by_head.items():
            cells = [f"{scores[key]:.1f}" if key.startswith("wrong") else f"{scores[key]:.1%}" for key, _ in columns]
            lines.append(f"| {source} | {head} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ##### ##### ##### ##### #####
#       Command line


def run(encoders: Dict[str, torch.nn.Module], db_path: str, cache_dir: Optional[str], heads: Sequence[str],
        mode: str, plies: int, positions: int, test_positions: int, steps: int, device, seed: int = 0,
        config: Optional[lje.JEPAConfig] = None, log: Log = print) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{input name: {head: scores}} for the two boards and every encoder's z. See the top of the file."""
    started = time.time()
    fens, labels = stored_positions(db_path, plies, positions)
    if not fens:
        raise ValueError(f"{db_path} holds no legal-threat labels {plies} plies deep (reach_labeler.py makes them)")
    if cache_dir and pr.USING_C:
        test_fens, test_labels = searched_positions(cache_dir, "val", plies, test_positions, seed)
        where = "validation positions, labels searched on the spot"
    else:
        cut = len(fens) - max(1, len(fens) // 10)
        fens, labels, test_fens, test_labels = fens[:cut], labels[:cut], fens[cut:], labels[cut:]
        where = "the last tenth of the database's positions"
    log(f"{len(fens):,} positions to fit on (the database's first), {len(test_fens):,} to score on ({where}); "
        f"{time.time() - started:.0f}s")
    inputs, test_inputs = ph.encode_inputs(encoders, fens, device), ph.encode_inputs(encoders, test_fens, device)

    def pieces(onehot: torch.Tensor) -> torch.Tensor:
        return onehot.view(onehot.size(0), 64, lje.SQUARE_CLASSES)[:, :, 1:13].sum(-1) > 0

    squares, test_squares = pieces(inputs[ph.ONEHOT]), pieces(test_inputs[ph.ONEHOT])
    log(f"inputs encoded; {time.time() - started:.0f}s")
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for source, x in inputs.items():
        results[source] = {}
        for head in heads:
            scores = fit(x, squares, labels, test_inputs[source], test_squares, test_labels, head, mode, plies, device,
                         steps=steps, seed=seed, config=config)
            results[source][head] = scores
            log(f"{source}, {head}: " + ", ".join(
                f"{k} plies {scores[f'found_{k}']:.1%} / {scores[f'rejected_{k}']:.1%} / {scores[f'wrong_{k}']:.1f}"
                for k in range(plies + 1))
                + f" (found / rejected / cells wrong a position; {scores['params'] / 1e6:.1f}M weights, {scores['seconds']:.0f}s)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("encoders", nargs="*", help="name=run_folder:snapshot_step")
    parser.add_argument("--heads", default=",".join(lje.THREAT_HEADS))
    parser.add_argument("--plies", type=int, default=2)
    parser.add_argument("--mode", choices=lje.THREAT_MODES, default="count")
    parser.add_argument("--positions", type=int, default=60_000, help="positions to fit on")
    parser.add_argument("--test-positions", type=int, default=5_000, help="validation positions to score on")
    parser.add_argument("--steps", type=int, default=4_000, help="batches of 256 a head is fitted for")
    parser.add_argument("--db", default="minotaur_reach.db", help="the labels database")
    parser.add_argument("--cache", default=ld.DEFAULT_CACHE_DIR, help='the position cache ("": hold out database positions)')
    parser.add_argument("--online", action="store_true", help="the online encoder instead of the target encoder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="write the results as JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = {}
    for spec in args.encoders:
        name, _, where = spec.partition("=")
        folder, _, snap = where.rpartition(":")
        path = os.path.join(folder, "snapshots", f"step_{int(snap):07d}.pt")
        encoders[name] = pl.load_encoder(path, device, "online" if args.online else "target")
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    unknown = [h for h in heads if h not in lje.THREAT_HEADS]
    if unknown:
        parser.error(f"--heads: {unknown} not among {lje.THREAT_HEADS}")
    results = run(encoders, args.db, args.cache, heads, args.mode, args.plies, args.positions, args.test_positions,
                  args.steps, device, seed=args.seed)
    print()
    print(table(results, args.plies, args.mode), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"results": results, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
