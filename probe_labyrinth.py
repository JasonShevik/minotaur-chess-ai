"""
Probes for trained Labyrinth encoders: what do the position vectors know about chess?

    python probe_labyrinth.py --model mine=runs/squares_sigreg/best.pt [--model other=...] \\
        --puzzles lichess_db_puzzle.csv.zst --db minotaur_data.db --out probe_results.csv

Every position is encoded by each model's target encoder (the one that is kept), and every label is
measured three ways, on held-out positions:

    linear      a linear probe on the standardized vectors: AUC for yes/no labels, R^2 (and Pearson
                and Spearman correlation of its predictions) for numeric ones. Can a straight line read
                the information off z?
    knn         the k nearest training positions by cosine similarity. For a yes/no label: the share of
                a positive test position's neighbours that are positive, divided by the base rate (lift:
                above 1 means positions with the label cluster), and an AUC scoring each test position
                by its neighbours. For a numeric label: R^2 and Spearman of the neighbours' mean.
    max_abs_dim_corr  the strongest correlation between a single dimension of z and the label.

Labels:
    puzzles (lichess_db_puzzle.csv.zst; the FEN is the position BEFORE the opponent's move that sets
    the puzzle, so that move is played first)
        theme             every one of Lichess's puzzle themes (is_theme = True)
        opening           opening families, where the puzzle's game had one
        puzzle_rating     the puzzle's rating
        puzzle_move       the solution's first move, in the side to move's frame
    labeled positions (minotaur_data.db, the engine-analyzed rows)
        eval              the engine score as the side to move's win probability, clipped centipawns,
                          and whether the engine found a forced mate
        best_move         the engine's best move, in the side to move's frame (so e2e4 also stands for
                          Black's e7e5): do positions with the same best move cluster?
    every position
        board             material balance and total material (side to move's view), legal move count,
                          in check, a capture available, and white to move. The encoder sees every board
                          from the side to move, so white_to_move should score near chance: a check that
                          it does not know the colours.

Baselines, probed the same way (--baselines): an untrained encoder of the same architecture
("untrained") and the raw board encoding, the 64 squares' 8 features flattened ("board_onehot").

Probe train / test split: a hash of the position (80 / 20), so it is the same for every model.
A notes column carries caveats: the labeled positions' castling bias (only ~5% have castling rights,
since the Chess960 repair re-queued most of those), the puzzles' standard-chess origin and theme
oversampling, and white_to_move's expected near-chance result.
Results: one row per (model, label) in --out, with kind and is_theme columns for filtering, and
per-model averages by kind in the _summary file next to it.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import labyrinth_data as ld
import labyrinth_jepa_encoder as lje

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


# ##### ##### ##### ##### #####
#       Labels


@dataclass
class LabelSet:
    """
    The positions of one data source and their labels. binary: name -> 0/1 array (NaN where the label
    does not apply); numeric: name -> float array (NaN where missing). kinds: name -> kind.
    """
    name: str
    fens: List[str]
    binary: Dict[str, np.ndarray] = field(default_factory=dict)
    numeric: Dict[str, np.ndarray] = field(default_factory=dict)
    kinds: Dict[str, str] = field(default_factory=dict)


def oriented_move(board: chess.Board, move: chess.Move) -> str:
    """The move in UCI, with squares in the side to move's frame (Black's e7e5 is e2e4)."""
    orient = (lambda sq: sq) if board.turn == chess.WHITE else (lambda sq: sq ^ 56)
    text = chess.square_name(orient(move.from_square)) + chess.square_name(orient(move.to_square))
    return text + (chess.piece_symbol(move.promotion) if move.promotion else "")


def board_labels(board: chess.Board) -> Dict[str, float]:
    us, them = board.turn, not board.turn
    mine = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == us)
    theirs = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == them)
    legal = list(board.legal_moves)
    return {
        "material_balance": float(mine - theirs),
        "total_material": float(mine + theirs),
        "legal_moves": float(len(legal)),
        "in_check": float(board.is_check()),
        "capture_available": float(any(board.is_capture(m) for m in legal)),
        "white_to_move": float(board.turn == chess.WHITE),
    }


BOARD_NUMERIC = ("material_balance", "total_material", "legal_moves")
BOARD_BINARY = ("in_check", "capture_available", "white_to_move")


def _hash_unit(text: str, salt: str) -> float:
    return int.from_bytes(hashlib.blake2b(f"{salt}:{text}".encode(), digest_size=8).digest(), "big") / 2 ** 64


def load_puzzles(path: str, base_rate: float = 0.02, per_theme: int = 1500, min_count: int = 100,
                 seed: int = 0, limit: Optional[int] = None, log=print) -> LabelSet:
    """
    A sample of the Lichess puzzle database: each puzzle is kept with probability base_rate, raised for
    puzzles with a rare theme so every theme has about per_theme examples (selection by a hash of the
    puzzle id, so it is reproducible). Labels with fewer than min_count examples are dropped.
    """
    import zstandard

    def rows():
        with open(path, "rb") as f:
            reader = csv.reader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8"))
            header = next(reader)
            col = {name: i for i, name in enumerate(header)}
            for n, row in enumerate(reader):
                if limit is not None and n >= limit:
                    break
                yield col, row

    t0 = time.time()
    theme_counts = collections.Counter()
    total = 0
    for col, row in rows():
        theme_counts.update(row[col["Themes"]].split())
        total += 1
    keep_rate = {t: max(base_rate, min(1.0, per_theme / c)) for t, c in theme_counts.items()}

    fens, themes, openings, ratings, moves, boards = [], [], [], [], [], []
    for col, row in rows():
        row_themes = row[col["Themes"]].split()
        rate = max([base_rate] + [keep_rate[t] for t in row_themes])
        if _hash_unit(row[col["PuzzleId"]], f"puzzles{seed}") >= rate:
            continue
        board = chess.Board(row[col["FEN"]], chess960=True)
        solution = row[col["Moves"]].split()
        board.push(chess.Move.from_uci(solution[0]))          # the opponent's move that sets the puzzle
        best = chess.Move.from_uci(solution[1])
        fens.append(ld.fen_of(board))
        themes.append(set(row_themes))
        tags = row[col["OpeningTags"]].split()
        openings.append(tags[0] if tags else None)
        ratings.append(float(row[col["Rating"]]))
        moves.append(oriented_move(board, best))
        boards.append(board_labels(board))
    log(f"puzzles: sampled {len(fens):,} of {total:,} in {time.time() - t0:.0f}s")

    out = LabelSet("puzzles", fens)
    n = len(fens)
    for theme in sorted(theme_counts):
        y = np.array([theme in ts for ts in themes], dtype=np.float32)
        if y.sum() >= min_count:
            out.binary[theme] = y
            out.kinds[theme] = "theme"
    for name, values, kind in (("opening", openings, "opening"), ("move", moves, "puzzle_move")):
        counts = collections.Counter(v for v in values if v)
        for value, c in counts.items():
            if c >= min_count:
                out.binary[f"{name}:{value}"] = np.array([v == value for v in values], dtype=np.float32)
                out.kinds[f"{name}:{value}"] = kind
    out.numeric["puzzle_rating"] = np.array(ratings, dtype=np.float32)
    out.kinds["puzzle_rating"] = "puzzle_rating"
    _add_board_labels(out, boards)
    return out


def score_to_side_to_move(score: str, white_to_move: bool) -> Tuple[float, float, float]:
    """(win probability, centipawns clipped to +-1000, forced mate found) for the side to move."""
    sign = 1.0 if white_to_move else -1.0
    if score.startswith("#"):
        mate_for_white = not score.startswith("#-")
        wins = mate_for_white == white_to_move
        return (1.0 if wins else 0.0), (1000.0 if wins else -1000.0), 1.0
    cp = sign * float(score)
    return 1.0 / (1.0 + 10 ** (-cp / 400.0)), max(-1000.0, min(1000.0, cp)), 0.0


def load_labeled(db_path: str, min_count: int = 100, limit: Optional[int] = None, log=print) -> LabelSet:
    """The engine-analyzed positions of minotaur_data.db with their evaluations and best moves."""
    uri = "file:" + os.path.abspath(db_path).replace("\\", "/") + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        query = 'SELECT fen, score, best_move FROM "960_position_data" WHERE is_analyzed = 1'
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        rows = conn.execute(query).fetchall()
    fens, win, cp, mate, moves, boards = [], [], [], [], [], []
    for fen, score, best in rows:
        norm = ld.normalize_fen(fen)
        if norm is None or score is None or not best:
            continue
        board = chess.Board(norm, chess960=True)
        try:
            move = chess.Move.from_uci(best)
        except ValueError:
            continue
        w, c, m = score_to_side_to_move(str(score), board.turn == chess.WHITE)
        fens.append(norm)
        win.append(w); cp.append(c); mate.append(m)
        moves.append(oriented_move(board, move))
        boards.append(board_labels(board))
    log(f"labeled positions: {len(fens):,}")
    out = LabelSet("labeled", fens)
    out.numeric["eval_win_prob"] = np.array(win, dtype=np.float32)
    out.numeric["eval_cp_clipped"] = np.array(cp, dtype=np.float32)
    out.binary["forced_mate_found"] = np.array(mate, dtype=np.float32)
    for name in ("eval_win_prob", "eval_cp_clipped", "forced_mate_found"):
        out.kinds[name] = "eval"
    counts = collections.Counter(moves)
    for value, c in counts.items():
        if c >= min_count:
            out.binary[f"best_move:{value}"] = np.array([v == value for v in moves], dtype=np.float32)
            out.kinds[f"best_move:{value}"] = "best_move"
    _add_board_labels(out, boards)
    return out


def _add_board_labels(out: LabelSet, boards: List[Dict[str, float]]) -> None:
    for name in BOARD_NUMERIC:
        out.numeric[name] = np.array([b[name] for b in boards], dtype=np.float32)
        out.kinds[name] = "board"
    for name in BOARD_BINARY:
        out.binary[name] = np.array([b[name] for b in boards], dtype=np.float32)
        out.kinds[name] = "board"


# ##### ##### ##### ##### #####
#       Encoding


class _FenGraphs(Dataset):
    def __init__(self, fens: Sequence[str]):
        self.fens = list(fens)

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, i: int) -> ld.GraphView:
        return ld.build_views(self.fens[i], torch.zeros(64, dtype=torch.bool))[0]


def load_target_encoder(path: str, device) -> lje.LabyrinthEncoder:
    """The target encoder of a training checkpoint (last.pt / best.pt)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    encoder = lje.LabyrinthEncoder(lje.JEPAConfig.from_dict(ckpt["jepa_config"]))
    state = {k[len("target."):]: v for k, v in ckpt["model"].items() if k.startswith("target.")}
    encoder.load_state_dict(state)
    return encoder.to(device).eval()


@torch.no_grad()
def encode_all(fens: Sequence[str], encoders: Dict[str, lje.LabyrinthEncoder], board_baseline: bool,
               device, batch_size: int = 512, workers: int = 8, seed: int = 0, log=print) -> Dict[str, torch.Tensor]:
    """{model name: [N, z_dim] on the CPU}, every model reading the same graph batches."""
    loader = DataLoader(_FenGraphs(fens), batch_size=batch_size, num_workers=workers, collate_fn=ld.pack_views,
                        persistent_workers=False)
    out: Dict[str, List[torch.Tensor]] = {name: [] for name in encoders}
    if board_baseline:
        out["board_onehot"] = []
    gens = {name: torch.Generator(device=device).manual_seed(seed) for name in encoders}
    t0 = time.time()
    for graphs in loader:
        if board_baseline:
            out["board_onehot"].append(graphs.x.view(graphs.num_graphs, -1).clone())
        graphs = graphs.to(device)
        for name, encoder in encoders.items():
            out[name].append(encoder(graphs, rng=gens[name]).z.float().cpu())
    log(f"  encoded {len(fens):,} positions with {len(out)} models in {time.time() - t0:.0f}s")
    return {name: torch.cat(parts) for name, parts in out.items()}


# ##### ##### ##### ##### #####
#       Metrics


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Area under the ROC curve (ties averaged); NaN without both classes."""
    pos = labels > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64, device=scores.device)
    # average the ranks of tied scores
    sorted_scores = scores[order]
    _, inverse, counts = torch.unique_consecutive(sorted_scores, return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, 0).double()
    mean_rank = ends - (counts.double() - 1) / 2
    ranks[order] = mean_rank[inverse]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    def rank(x):
        r = torch.empty_like(x, dtype=torch.float64)
        r[torch.argsort(x)] = torch.arange(x.numel(), dtype=torch.float64, device=x.device)
        return r
    return pearson(rank(a), rank(b))


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
    denom = (a.norm() * b.norm()).item()
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def r2(pred: torch.Tensor, y: torch.Tensor) -> float:
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((pred - y) ** 2).sum() / ss) if ss > 0 else float("nan")


def train_logistic(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, steps: int = 300, lr: float = 0.05,
                   weight_decay: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-vs-rest logistic regression for every label column at once (mask: which entries count)."""
    w = torch.zeros(x.size(1), y.size(1), device=x.device, requires_grad=True)
    b = torch.zeros(y.size(1), device=x.device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        logits = x @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = (loss * mask).sum() / mask.sum().clamp_min(1) + weight_decay * (w ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def ridge(x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """Ridge regression coefficients, solved in float64 (the normal equations are badly conditioned in float32)."""
    x64, y64 = x.double(), y.double()
    d = x.size(1)
    coef = torch.linalg.solve(x64.T @ x64 + lam * torch.eye(d, device=x.device, dtype=torch.float64), x64.T @ y64)
    return coef.to(x.dtype)


def knn_indices(queries: torch.Tensor, reference: torch.Tensor, k: int, chunk: int = 2048) -> torch.Tensor:
    q = torch.nn.functional.normalize(queries, dim=1)
    r = torch.nn.functional.normalize(reference, dim=1)
    out = []
    for i in range(0, q.size(0), chunk):
        out.append((q[i:i + chunk] @ r.T).topk(k, dim=1).indices)
    return torch.cat(out)


def probe_one_model(z: torch.Tensor, labels: LabelSet, train: np.ndarray, device, k: int = 20,
                    max_queries: int = 20000, seed: int = 0) -> List[Dict[str, object]]:
    """All metrics for every label of one data source, for one model's vectors z [N, D]."""
    z = z.to(device)
    tr = torch.as_tensor(train, device=device)
    te = ~tr
    mean, std = z[tr].mean(0), z[tr].std(0)
    # A dimension that is nearly constant on the training rows would be blown up by standardizing (tiny
    # variations on the test rows divided by a near-zero spread), so spreads get a floor relative to the median.
    std = std.clamp_min(max(1e-6, 1e-3 * float(std.median())))
    zs = (z - mean) / std
    x_tr, x_te = zs[tr], zs[te]
    test_rows = te.nonzero().flatten()
    g = torch.Generator(device="cpu").manual_seed(seed)
    if test_rows.numel() > max_queries:
        test_rows = test_rows[torch.randperm(test_rows.numel(), generator=g)[:max_queries].to(device)]
    train_rows = tr.nonzero().flatten()
    neighbours = train_rows[knn_indices(z[test_rows], z[train_rows], k)]        # [Q, k] row indices into z

    rows: List[Dict[str, object]] = []
    # ---- yes/no labels: one logistic probe for all of them
    names = list(labels.binary)
    if names:
        y = torch.as_tensor(np.stack([labels.binary[n] for n in names], 1), device=device)
        valid = ~torch.isnan(y)
        y0 = torch.nan_to_num(y, 0.0)
        w, b = train_logistic(x_tr, y0[tr], valid[tr].float())
        scores_te = x_te @ w + b
        y_te, v_te = y0[te], valid[te]
        for j, name in enumerate(names):
            ytr = y0[tr][valid[tr][:, j], j]
            base = float(ytr.mean()) if ytr.numel() else float("nan")
            m = v_te[:, j]
            q_valid = valid[test_rows, j]
            nb = y0[neighbours, j] * valid[neighbours, j]
            nb_rate = nb.sum(1) / valid[neighbours, j].sum(1).clamp_min(1)
            q_y = y0[test_rows, j]
            pos_q = q_valid & (q_y > 0.5)
            lift = float(nb_rate[pos_q].mean() / base) if pos_q.any() and base > 0 else float("nan")
            corr = _max_abs_corr(x_tr[valid[tr][:, j]], ytr)
            rows.append({
                "label": name, "binary": True,
                "n_train": int(valid[tr][:, j].sum()), "n_test": int(m.sum()), "n_test_positive": int((y_te[m, j] > 0.5).sum()),
                "base_rate": base,
                "linear_auc": auc(scores_te[m, j], y_te[m, j]),
                "knn_auc": auc(nb_rate[q_valid], q_y[q_valid]),
                "knn_lift": lift,
                "max_abs_dim_corr": corr,
            })
    # ---- numeric labels: ridge regression, lambda picked on a slice of the training rows
    for name, values in labels.numeric.items():
        y = torch.as_tensor(values, device=device)
        valid = ~torch.isnan(y)
        trv, tev = tr & valid, te & valid
        xa, ya = zs[trv], y[trv]
        perm = torch.randperm(xa.size(0), generator=g).to(device)
        cut = max(1, int(0.85 * xa.size(0)))
        fit, hold = perm[:cut], perm[cut:]
        best_lam, best_score = 1.0, -math.inf
        for lam in (1e-1, 1, 10, 100, 1000, 10000):
            coef = ridge(xa[fit], ya[fit] - ya[fit].mean(), lam)
            score = r2(xa[hold] @ coef + ya[fit].mean(), ya[hold])
            if score > best_score:
                best_lam, best_score = lam, score
        coef = ridge(xa, ya - ya.mean(), best_lam)
        pred = zs[tev] @ coef + ya.mean()
        q_valid = valid[test_rows]
        nb_valid = valid[neighbours]
        nb_mean = (torch.nan_to_num(y[neighbours]) * nb_valid).sum(1) / nb_valid.sum(1).clamp_min(1)
        yq = y[test_rows][q_valid]
        rows.append({
            "label": name, "binary": False,
            "n_train": int(trv.sum()), "n_test": int(tev.sum()), "n_test_positive": "", "base_rate": "",
            "linear_r2": r2(pred, y[tev]), "linear_pearson": pearson(pred, y[tev]),
            "linear_spearman": spearman(pred, y[tev]),
            "knn_r2": r2(nb_mean[q_valid], yq), "knn_spearman": spearman(nb_mean[q_valid], yq),
            "max_abs_dim_corr": _max_abs_corr(xa, ya),
        })
    return rows


def _max_abs_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if y.numel() < 3:
        return float("nan")
    xc = x - x.mean(0)
    yc = y.double() - y.double().mean()
    denom = xc.double().norm(dim=0) * yc.norm()
    corr = (xc.double() * yc[:, None]).sum(0) / denom.clamp_min(1e-12)
    return float(corr.abs().max())


# ##### ##### ##### ##### #####
#       Running everything


COLUMNS = ["model", "dataset", "kind", "is_theme", "label", "n_train", "n_test", "n_test_positive", "base_rate",
           "linear_auc", "knn_auc", "knn_lift", "linear_r2", "linear_pearson", "linear_spearman", "knn_r2",
           "knn_spearman", "max_abs_dim_corr", "notes"]

# Caveats that apply to every row of a data source, and to particular labels
DATASET_NOTES = {
    "labeled": "* sample bias: only about 5% of the engine-analyzed positions have castling rights (about half of "
               "all positions do), since the Chess960 castling repair re-queued most of them; these probes say "
               "little about positions with castling rights",
    "puzzles": "standard-chess games (the encoder trained on Chess960); rare themes were oversampled, so base_rate "
               "is the sample's, not the puzzle database's",
}
LABEL_NOTES = {
    "white_to_move": "expected near chance (AUC ~0.5): boards are always shown from the side to move",
}


def note_for(row: Dict[str, object]) -> str:
    parts = [LABEL_NOTES.get(str(row.get("label")), ""), DATASET_NOTES.get(str(row.get("dataset")), "")]
    return "; ".join(p for p in parts if p)


def run_probes(models: Dict[str, str], label_sets: Sequence[LabelSet], out_path: str, baselines: bool = True,
               device: Optional[str] = None, k: int = 20, workers: int = 8, seed: int = 0, log=print) -> List[Dict]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoders = {name: load_target_encoder(path, device) for name, path in models.items()}
    if baselines:
        torch.manual_seed(seed)
        reference = next(iter(encoders.values())).config if encoders else lje.JEPAConfig(token_variant="squares")
        encoders["untrained"] = lje.LabyrinthEncoder(reference).to(device).eval()
    results: List[Dict[str, object]] = []
    for labels in label_sets:
        log(f"{labels.name}: {len(labels.fens):,} positions, {len(labels.binary)} yes/no and "
            f"{len(labels.numeric)} numeric labels")
        train = np.array([_hash_unit(ld.position_key(f), "probe") < 0.8 for f in labels.fens])
        vectors = encode_all(labels.fens, encoders, baselines, device, workers=workers, seed=seed, log=log)
        for model_name, z in vectors.items():
            t0 = time.time()
            for row in probe_one_model(z, labels, train, device, k=k, seed=seed):
                kind = labels.kinds[row["label"]]
                results.append({"model": model_name, "dataset": labels.name, "kind": kind,
                                "is_theme": kind == "theme", **row})
                results[-1]["notes"] = note_for(results[-1])
            log(f"  probed {model_name} in {time.time() - t0:.0f}s")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({c: _fmt(row.get(c, "")) for c in COLUMNS})
    summary_path = os.path.splitext(out_path)[0] + "_summary.csv"
    write_summary(results, summary_path)
    log(f"wrote {len(results):,} rows to {out_path} and averages to {summary_path}")
    return results


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else v


def write_summary(results: Sequence[Dict], path: str) -> None:
    """Per model and kind: how many labels, and the mean of each metric over them."""
    metrics = ["linear_auc", "knn_auc", "knn_lift", "linear_r2", "knn_r2", "max_abs_dim_corr"]
    groups: Dict[Tuple[str, str, str], List[Dict]] = collections.defaultdict(list)
    for row in results:
        groups[(row["model"], row["dataset"], row["kind"])].append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "dataset", "kind", "labels"] + [f"mean_{m}" for m in metrics] + ["notes"])
        for (model, dataset, kind), rows in sorted(groups.items()):
            means = []
            for m in metrics:
                vals = [r[m] for r in rows if isinstance(r.get(m), float) and not math.isnan(r[m])]
                means.append(f"{sum(vals) / len(vals):.4f}" if vals else "")
            writer.writerow([model, dataset, kind, len(rows)] + means + [DATASET_NOTES.get(dataset, "")])


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe trained Labyrinth encoders.")
    parser.add_argument("--model", action="append", default=[], help="NAME=path/to/checkpoint.pt (repeatable)")
    parser.add_argument("--puzzles", default=None, help="lichess_db_puzzle.csv.zst")
    parser.add_argument("--db", default=None, help="minotaur_data.db (engine-analyzed positions)")
    parser.add_argument("--out", default="probe_results.csv")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--puzzle-rate", type=float, default=0.02)
    parser.add_argument("--per-theme", type=int, default=1500)
    parser.add_argument("--min-count", type=int, default=100, help="drop labels with fewer examples")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    models = dict(m.split("=", 1) for m in args.model)
    sets = []
    if args.puzzles:
        sets.append(load_puzzles(args.puzzles, args.puzzle_rate, args.per_theme, args.min_count, args.seed))
    if args.db:
        sets.append(load_labeled(args.db, args.min_count))
    if not sets:
        raise SystemExit("give --puzzles and/or --db")
    run_probes(models, sets, args.out, baselines=not args.no_baselines, k=args.k, workers=args.workers,
               seed=args.seed)


if __name__ == "__main__":
    main()
