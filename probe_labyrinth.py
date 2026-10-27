"""
Probes for trained Labyrinth encoders: what do the position vectors know about chess?

    python probe_labyrinth.py --model mine=runs/squares_sigreg/best.pt [--model other=...] \\
        --puzzles lichess_db_puzzle.csv.zst --db minotaur_data.db --out probe_results.csv \\
        [--views z,token_mean,readout_in,pre_transformer_mean --online --report-cache labyrinth_ssl_cache]

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

Where to probe (--views, --online): z by default, or also vectors inside the encoder, to find where
information is lost: token_mean (the final square tokens averaged, before the readout), readout_in (the
readout's attention outputs before its last linear layer) and pre_transformer_mean (the square tokens
averaged as they enter the transformer). --online also probes each checkpoint's online encoder, next to
the target encoder. The encoder and view columns say which vector a row measured.

Representation report (--report-cache): LiDAR, centred RankMe and the self-consistency ratio of every
probed vector, baselines included, on the cache's val split under the light masks training evaluates
with, so z's numbers can be read against what the untrained encoder and the raw board score.

A run's snapshots (--snapshots runs/x/snapshots): every step_*.pt that train_labyrinth_jepa saved, each
probed as a model named after its step, to see when during training the probes change.

TrainingProbes is the small fixed version train_labyrinth_jepa runs at every evaluation.

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
import glob
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


def load_encoder(path: str, device, which: str = "target") -> lje.LabyrinthEncoder:
    """
    One encoder of a training checkpoint (last.pt / best.pt): "target" (the moving average, the one that
    is kept) or "online" (the one the optimizer trains).
    """
    prefix = {"target": "target.", "online": "encoder."}[which]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    encoder = lje.LabyrinthEncoder(lje.JEPAConfig.from_dict(ckpt["jepa_config"]))
    state = {k[len(prefix):]: v for k, v in ckpt["model"].items() if k.startswith(prefix)}
    encoder.load_state_dict(state)
    return encoder.to(device).eval()


# The vectors an encoder can be probed at, from the readout's output back to the square stage
VIEWS = {
    "z": "the position vector (the readout's output)",
    "token_mean": "the mean of the 64 final square tokens (after the transformer, before the readout)",
    "readout_in": "the readout's attention outputs, normalized and concatenated, before its last linear layer",
    "pre_transformer_mean": "the mean of the 64 square tokens entering the transformer (after the square GATs)",
}


class EncoderViews:
    """
    Runs an encoder and returns the requested views of each position (see VIEWS). The intermediate ones
    are taken with forward hooks, so the encoder itself is unchanged.
    """

    def __init__(self, encoder: lje.LabyrinthEncoder, views: Sequence[str]):
        unknown = set(views) - set(VIEWS)
        if unknown:
            raise ValueError(f"unknown views {sorted(unknown)}; choose from {sorted(VIEWS)}")
        self.encoder, self.views = encoder, list(views)
        self._captured: Dict[str, torch.Tensor] = {}
        self._hooks = []
        if "readout_in" in self.views:
            self._hooks.append(encoder.readout_proj.register_forward_hook(
                lambda module, args, output: self._captured.__setitem__("readout_in", args[0])))
        if "pre_transformer_mean" in self.views:
            self._hooks.append(encoder.transformer.register_forward_pre_hook(
                lambda module, args: self._captured.__setitem__("pre_transformer", args[0])))

    def remove(self) -> None:
        """Takes the hooks off, for an encoder that goes on being used (or trained) afterwards."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._captured.clear()

    @torch.no_grad()
    def __call__(self, graphs: ld.GraphBatch, mask: Optional[torch.Tensor] = None,
                 rng: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """{view: [B, dim] on the CPU}."""
        out = self.encoder(graphs, mask=mask, rng=rng)
        found = {"z": lambda: out.z,
                 "token_mean": lambda: out.squares.mean(1),
                 "readout_in": lambda: self._captured["readout_in"],
                 "pre_transformer_mean": lambda: self._captured["pre_transformer"][:, :64].mean(1)}
        return {view: found[view]().float().cpu() for view in self.views}


@torch.no_grad()
def encode_views(fens: Sequence[str], viewer: Optional[EncoderViews], device, batch_size: int = 512,
                 workers: int = 8, seed: int = 0) -> Dict[str, torch.Tensor]:
    """
    {view: [N, dim] on the CPU} for one encoder; with viewer None, {"onehot": the raw board}, the 64
    squares' features flattened.
    """
    loader = DataLoader(_FenGraphs(fens), batch_size=batch_size, num_workers=workers, collate_fn=ld.pack_views,
                        persistent_workers=False)
    gen = torch.Generator(device=device).manual_seed(seed)
    parts: Dict[str, List[torch.Tensor]] = collections.defaultdict(list)
    for graphs in loader:
        if viewer is None:
            parts["onehot"].append(graphs.x.view(graphs.num_graphs, -1).clone())
            continue
        for view, vectors in viewer(graphs.to(device), rng=gen).items():
            parts[view].append(vectors)
    return {view: torch.cat(p) for view, p in parts.items()}


# Light masks for the representation report: the ones train_labyrinth_jepa's evaluation uses
# (RunConfig.eval_mask_min_ratio / eval_mask_max_ratio, one-square blocks)
REPORT_MASKS = ld.MaskConfig(0.05, 0.15, 1)


@torch.no_grad()
def representation_rows(sources: Sequence[Tuple[Tuple[str, str], Optional[EncoderViews]]], fens: Sequence[str],
                        device, samples: int = 4, batch_size: int = 256, seed: int = 0) -> List[Dict[str, object]]:
    """
    LiDAR, centred RankMe and the self-consistency ratio of every probed vector, measured as training
    measures z: each position encoded `samples` times, each time under a fresh light mask. Every model sees
    the same masks; the raw board gets them as the encoders do, with the hidden squares' features cleared.
    LiDAR is left out when a view has more than half as many dimensions as there are positions.
    """
    rnd = random.Random(seed)
    masks = [[ld.sample_mask(rnd, REPORT_MASKS) for _ in fens] for _ in range(samples)]
    collected: Dict[Tuple[str, str, str], List[List[torch.Tensor]]] = \
        collections.defaultdict(lambda: [[] for _ in range(samples)])
    gen = torch.Generator(device=device).manual_seed(seed)
    for s in range(samples):
        for lo in range(0, len(fens), batch_size):
            chunk = range(lo, min(lo + batch_size, len(fens)))
            graphs = ld.pack_views([ld.build_views(fens[i], masks[s][i])[1] for i in chunk])
            mask = torch.cat([masks[s][i] for i in chunk])
            for (model, which), viewer in sources:
                if viewer is None:
                    collected[(model, which, "onehot")][s].append(graphs.x.view(graphs.num_graphs, -1).clone())
                    continue
                for view, vectors in viewer(graphs.to(device), mask=mask.to(device), rng=gen).items():
                    collected[(model, which, view)][s].append(vectors)
    rows = []
    for (model, which, view), per_sample in collected.items():
        z = torch.stack([torch.cat(parts) for parts in per_sample], dim=1).to(device)    # [N, samples, D]
        means = z.mean(1)
        rows.append({"model": model, "encoder": which, "view": view, "dims": z.size(2), "positions": z.size(0),
                     "samples": samples, "self_consistency": lje.self_consistency(z),
                     "rankme_centered": lje.rankme(means - means.mean(0)),
                     "lidar": lje.lidar(z) if 2 * z.size(2) <= z.size(0) else None,
                     "notes": VIEWS.get(view, "the raw board: the 64 squares' features flattened")})
    return rows


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


COLUMNS = ["model", "encoder", "view", "dataset", "kind", "is_theme", "label", "n_train", "n_test", "n_test_positive", "base_rate",
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


REPRESENTATION_COLUMNS = ["model", "encoder", "view", "dims", "positions", "samples", "self_consistency",
                          "rankme_centered", "lidar", "notes"]


def run_probes(models: Dict[str, str], label_sets: Sequence[LabelSet], out_path: str, baselines: bool = True,
               device: Optional[str] = None, k: int = 20, workers: int = 8, seed: int = 0,
               views: Sequence[str] = ("z",), online: bool = False, report_fens: Optional[Sequence[str]] = None,
               report_samples: int = 4, log=print) -> List[Dict]:
    """
    Probes every model's target encoder (and its online encoder, with online) at each of `views`, plus
    the baselines, and writes the CSVs. With report_fens, also writes the representation report
    (LiDAR, RankMe, self-consistency of every probed vector) to <out>_representation.csv.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoders: Dict[Tuple[str, str], lje.LabyrinthEncoder] = {}
    for name, path in models.items():
        for which in ("target", "online") if online else ("target",):
            encoders[(name, which)] = load_encoder(path, device, which)
    if baselines:
        torch.manual_seed(seed)
        reference = next(iter(encoders.values())).config if encoders else lje.JEPAConfig(token_variant="squares")
        encoders[("untrained", "random init")] = lje.LabyrinthEncoder(reference).to(device).eval()
    sources: List[Tuple[Tuple[str, str], Optional[EncoderViews]]] = \
        [(key, EncoderViews(encoder, views)) for key, encoder in encoders.items()]
    if baselines:
        sources.append((("board_onehot", ""), None))
    results: List[Dict[str, object]] = []
    for labels in label_sets:
        log(f"{labels.name}: {len(labels.fens):,} positions, {len(labels.binary)} yes/no and "
            f"{len(labels.numeric)} numeric labels")
        train = np.array([_hash_unit(ld.position_key(f), "probe") < 0.8 for f in labels.fens])
        for (model, which), viewer in sources:
            # one encoder at a time: every view of every model at once would not fit in memory
            t0 = time.time()
            vectors = encode_views(labels.fens, viewer, device, workers=workers, seed=seed)
            for view, z in vectors.items():
                for row in probe_one_model(z, labels, train, device, k=k, seed=seed):
                    kind = labels.kinds[row["label"]]
                    results.append({"model": model, "encoder": which, "view": view, "dataset": labels.name,
                                    "kind": kind, "is_theme": kind == "theme", **row})
                    results[-1]["notes"] = note_for(results[-1])
            log(f"  encoded and probed {model} {which} ({', '.join(vectors)}) in {time.time() - t0:.0f}s")
    if results:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow({c: _fmt(row.get(c, "")) for c in COLUMNS})
        summary_path = os.path.splitext(out_path)[0] + "_summary.csv"
        write_summary(results, summary_path)
        log(f"wrote {len(results):,} rows to {out_path} and averages to {summary_path}")
    if report_fens:
        t0 = time.time()
        rows = representation_rows(sources, report_fens, device, samples=report_samples, seed=seed)
        report_path = os.path.splitext(out_path)[0] + "_representation.csv"
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REPRESENTATION_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: _fmt(row[c]) if row[c] is not None else "" for c in REPRESENTATION_COLUMNS})
        log(f"wrote the representation report ({len(rows)} rows, {time.time() - t0:.0f}s) to {report_path}")
    return results


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else v


def write_summary(results: Sequence[Dict], path: str) -> None:
    """Per model, encoder, view and kind: how many labels, and the mean of each metric over them."""
    metrics = ["linear_auc", "knn_auc", "knn_lift", "linear_r2", "knn_r2", "max_abs_dim_corr"]
    groups: Dict[Tuple[str, str, str, str, str], List[Dict]] = collections.defaultdict(list)
    for row in results:
        groups[(row["model"], row["encoder"], row["view"], row["dataset"], row["kind"])].append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "encoder", "view", "dataset", "kind", "labels"] + [f"mean_{m}" for m in metrics]
                        + ["notes"])
        for (model, which, view, dataset, kind), rows in sorted(groups.items()):
            means = []
            for m in metrics:
                vals = [r[m] for r in rows if isinstance(r.get(m), float) and not math.isnan(r[m])]
                means.append(f"{sum(vals) / len(vals):.4f}" if vals else "")
            writer.writerow([model, which, view, dataset, kind, len(rows)] + means + [DATASET_NOTES.get(dataset, "")])


# ##### ##### ##### ##### #####
#       Probes during training


def board_label_set(name: str, fens: Sequence[str]) -> LabelSet:
    """Positions with only the board labels (material, legal moves, check, ...), which need no outside data."""
    out = LabelSet(name, list(fens))
    _add_board_labels(out, [board_labels(chess.Board(f, chess960=True)) for f in out.fens])
    return out


def subsample(labels: LabelSet, n: int, seed: int = 0, min_count: int = 30) -> LabelSet:
    """
    At most n of the positions (the same ones for the same seed) with their labels. Yes/no labels left
    with fewer than min_count positives are dropped: their AUCs would be noise.
    """
    keep = np.arange(len(labels.fens))
    if len(keep) > n:
        keep = np.sort(np.random.default_rng(seed).choice(len(keep), n, replace=False))
    out = LabelSet(labels.name, [labels.fens[i] for i in keep])
    for name, values in labels.binary.items():
        if np.nansum(values[keep]) >= min_count:
            out.binary[name], out.kinds[name] = values[keep], labels.kinds[name]
    for name, values in labels.numeric.items():
        out.numeric[name], out.kinds[name] = values[keep], labels.kinds[name]
    return out


# Labels that TrainingProbes.summarize also reports one by one, besides the averages by kind
HEADLINE_LABELS = ("material_balance", "eval_win_prob")


class TrainingProbes:
    """
    A small fixed probe set for train_labyrinth_jepa's evaluations: the same positions, labels and
    train / test split every time, so a run's curves show when each view gains or loses information.
    Graphs are built in the calling process (no loader workers to start at every evaluation).
    """

    def __init__(self, label_sets: Sequence[LabelSet], views: Sequence[str] = ("z",), k: int = 20,
                 batch_size: int = 512, seed: int = 0):
        unknown = set(views) - set(VIEWS)
        if unknown:
            raise ValueError(f"unknown views {sorted(unknown)}; choose from {sorted(VIEWS)}")
        self.label_sets, self.views = list(label_sets), list(views)
        self.k, self.batch_size, self.seed = k, batch_size, seed
        self.splits = [np.array([_hash_unit(ld.position_key(f), "probe") < 0.8 for f in labels.fens])
                       for labels in self.label_sets]

    def probe(self, encoder: Optional[lje.LabyrinthEncoder], device, model: str = "trained",
              which: str = "target") -> List[Dict[str, object]]:
        """Every label at every view of the encoder (None: the raw board), as rows with run_probes' columns."""
        viewer = EncoderViews(encoder, self.views) if encoder is not None else None
        was_training = encoder is not None and encoder.training
        if encoder is not None:
            encoder.eval()
        rows: List[Dict[str, object]] = []
        try:
            for labels, train in zip(self.label_sets, self.splits):
                vectors = encode_views(labels.fens, viewer, device, batch_size=self.batch_size, workers=0,
                                       seed=self.seed)
                k = max(1, min(self.k, int(train.sum())))
                for view, z in vectors.items():
                    for row in probe_one_model(z, labels, train, device, k=k, seed=self.seed):
                        kind = labels.kinds[row["label"]]
                        rows.append({"model": model, "encoder": which, "view": view, "dataset": labels.name,
                                     "kind": kind, "is_theme": kind == "theme", **row})
                        rows[-1]["notes"] = note_for(rows[-1])
        finally:
            if encoder is not None:
                viewer.remove()
                encoder.train(was_training)
        return rows

    @staticmethod
    def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        One model's rows -> {view: {key: {"linear": x, "knn": y}}}. Keys: "<dataset>/<kind>/auc" (the
        mean AUC of that kind's yes/no labels), "<dataset>/<kind>/r2" (the mean R^2 of its numeric
        ones), and "<dataset>/<label>" for HEADLINE_LABELS. Metrics that are NaN everywhere are left out.
        """
        groups: Dict[Tuple[str, str], Dict[str, List[float]]] = collections.defaultdict(
            lambda: collections.defaultdict(list))
        for row in rows:
            metric = "auc" if row["binary"] else "r2"
            keys = [f"{row['dataset']}/{row['kind']}/{metric}"]
            if row["label"] in HEADLINE_LABELS:
                keys.append(f"{row['dataset']}/{row['label']}")
            for key in keys:
                for probe in ("linear", "knn"):
                    value = row.get(f"{probe}_{metric}")
                    if isinstance(value, float) and math.isfinite(value):
                        groups[(str(row["view"]), key)][probe].append(value)
        out: Dict[str, Dict[str, Dict[str, float]]] = collections.defaultdict(dict)
        for (view, key), values in sorted(groups.items()):
            out[view][key] = {probe: sum(v) / len(v) for probe, v in values.items()}
        return dict(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe trained Labyrinth encoders.")
    parser.add_argument("--model", action="append", default=[], help="NAME=path/to/checkpoint.pt (repeatable)")
    parser.add_argument("--snapshots", default=None,
                        help="a training run's snapshots folder: probes every step_*.pt in it, named by its step")
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
    parser.add_argument("--views", default="z", help=f"comma-separated, from: {', '.join(VIEWS)}")
    parser.add_argument("--online", action="store_true", help="also probe each checkpoint's online encoder")
    parser.add_argument("--report-cache", default=None,
                        help="a position cache: its val split feeds the representation report (LiDAR, RankMe, "
                             "self-consistency of every probed vector)")
    parser.add_argument("--report-positions", type=int, default=2048, help="as train_labyrinth_jepa's evaluation")
    parser.add_argument("--report-samples", type=int, default=4)
    args = parser.parse_args()
    models = dict(m.split("=", 1) for m in args.model)
    if args.snapshots:
        for path in sorted(glob.glob(os.path.join(args.snapshots, "step_*.pt"))):
            models[os.path.splitext(os.path.basename(path))[0]] = path
    sets = []
    if args.puzzles:
        sets.append(load_puzzles(args.puzzles, args.puzzle_rate, args.per_theme, args.min_count, args.seed))
    if args.db:
        sets.append(load_labeled(args.db, args.min_count))
    report_fens = None
    if args.report_cache:
        val = ld.PositionDataset(args.report_cache, "val")
        report_fens = [val.fen(i) for i in range(min(args.report_positions, len(val)))]
    if not sets and not report_fens:
        raise SystemExit("give --puzzles and/or --db, or --report-cache")
    run_probes(models, sets, args.out, baselines=not args.no_baselines, k=args.k, workers=args.workers,
               seed=args.seed, views=[v.strip() for v in args.views.split(",") if v.strip()], online=args.online,
               report_fens=report_fens, report_samples=args.report_samples)


if __name__ == "__main__":
    main()
