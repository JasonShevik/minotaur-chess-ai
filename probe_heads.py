"""
Trained-head probes: what can a small MLP read from a frozen encoder's z?

probe_labyrinth.py asks how z is organized: a linear probe fitted on a few thousand positions, and nearest
neighbours. A z that holds the whole board, square by square, scores badly there however much it holds (a
one-hot board does too). The final encoder is meant to stay frozen under trained heads, so this file measures
it the way it will be used: the SAME MLP, trained the same way, on each encoder's z and on the raw board (the
64 squares' node features, flattened). A head on z that beats the same head on the raw board means the
encoder has made something explicit that the board leaves implicit; a tie means z is a copy of the board.

The probes (every position is seen from the side to move, as the encoder sees it):

  zugzwang      Is this a puzzle whose solution runs through zugzwang? (Lichess's "zugzwang" theme, at the
                position the solver starts from.) The other half are endgame puzzles without the theme,
                matched one for one on both sides' pieces and the number of pawns, so counting material
                tells nothing. Score: AUC.
  quiet_detect  Is the winning move here quiet or forcing? Quiet: positions just before the move that earns a
                puzzle Lichess's "quietMove" theme (no check given or escaped, no capture made, no enemy piece
                attacked by the moved piece, no advanced pawn push, not a king move, not the puzzle's last
                move; lichess-puzzler's cook.quiet_move, applied move by move). Forcing: positions from other
                puzzles, the same number of moves into the solution, where the solver captures or checks.
                Score: AUC.
  quiet_find    Given a position with a quiet winning move, which move is it? The head scores all 1,792
                from-to pairs; illegal ones are masked out. Score: how often the move is the head's first
                choice / among its first three.
  forcing_find  The same for positions whose winning move is a capture or a check: the control. If z helps
                more on quiet moves than on forcing ones, it helps where the board offers no surface cue.
  best_move     The same for the engine's best move on the engine-analyzed positions of minotaur_data.db.
  eval          The engine's score as the side to move's win probability. Score: R2.
  material      Material balance from the side to move's view. Score: R2 (a head should simply get this right).

Each probe: positions split 80 / 10 / 10 by a hash of the position; inputs standardized on the training rows;
an MLP (input -> 1024 -> 1024 -> output) trained with AdamW, the epoch chosen on the validation rows, scored
on the test rows; the mean of `seeds` trainings.

    python probe_heads.py "raw board" C_mean=runs/step21/C_all6_mean:4000 C_concat=runs/step21/C_all6_concat:4000
        [--size 100000 --seeds 2 --online --out results.json]

("raw board" is always included; name=run_folder:snapshot_step for each encoder.)
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
import pickle
import random
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import chess
import numpy as np
import torch
import torch.nn.functional as F

import labyrinth_data as ld
import labyrinth_jepa_encoder as lje
import probe_labyrinth as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.dirname(os.path.dirname(os.path.dirname(ROOT))) if ".claude" in ROOT else ROOT
DEFAULT_PUZZLES = os.path.join(MAIN, "lichess_db_puzzle.csv.zst")
DEFAULT_DB = os.path.join(MAIN, "minotaur_data.db")
RAW = "raw board"


# ##### ##### ##### ##### #####
#       Moves as one of the 1,792 from-to pairs


def _pair_table() -> np.ndarray:
    table = np.full(4096, -1, dtype=np.int16)
    pairs = lje.move_pairs()
    table[(pairs[:, 0] * 64 + pairs[:, 1]).numpy()] = np.arange(len(pairs), dtype=np.int16)
    return table


PAIR_OF = _pair_table()                     # from * 64 + to (in the side to move's frame) -> pair index, or -1
NUM_PAIRS = int((PAIR_OF >= 0).sum())


def move_pair(board: chess.Board, move: chess.Move) -> int:
    """The move's from-to pair in the board's frame (the side to move at the bottom; castling: king to its rook)."""
    return int(PAIR_OF[ld.legal_codes(board, [move])[0]])


def legal_pairs(board: chess.Board) -> np.ndarray:
    return PAIR_OF[np.array(ld.legal_codes(board), dtype=np.int64)]


# ##### ##### ##### ##### #####
#       Puzzles


def puzzle_rows(path: str) -> Iterator[Tuple[str, str, List[str], Set[str]]]:
    """(puzzle id, FEN before the opponent's setting move, the line's moves in UCI, themes) of every puzzle."""
    import zstandard
    with open(path, "rb") as f:
        reader = csv.DictReader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8"))
        for row in reader:
            yield row["PuzzleId"], row["FEN"], row["Moves"].split(), set(row["Themes"].split())


def is_quiet_move(before: chess.Board, move: chess.Move, after: chess.Board, last: bool) -> bool:
    """
    Lichess's rule for the move that makes a puzzle a "quietMove" puzzle (lichess-puzzler, cook.quiet_move),
    for one move of the solver: not the puzzle's last move; no check given or escaped; no capture; the moved
    piece attacks no enemy piece from its new square; no promotion and no pawn push past its fourth rank; not
    a king move (castling included).
    """
    if last or before.is_check() or after.is_check() or before.is_capture(move) or before.is_castling(move):
        return False
    piece = after.piece_type_at(move.to_square)
    if piece is None or piece == chess.KING or move.promotion:
        return False
    mover = before.turn
    for square in after.attacks(move.to_square):
        attacked = after.piece_at(square)
        if attacked is not None and attacked.color != mover:
            return False
    if piece == chess.PAWN:
        rank = chess.square_rank(move.to_square)
        if (rank > 4) if mover == chess.WHITE else (rank < 3):
            return False
    return True


def is_forcing_move(before: chess.Board, move: chess.Move) -> bool:
    return before.is_capture(move) or before.gives_check(move)


def solver_moves(fen: str, moves: Sequence[str]) -> List[Tuple[chess.Board, chess.Move, chess.Board, bool]]:
    """
    The solver's moves of a puzzle line: (the board before the move, the move, the board after it, whether it
    is the line's last move). The line starts with the opponent's move that sets the puzzle. Raises ValueError
    on a line python-chess cannot play.
    """
    board = chess.Board(fen, chess960=True)
    solver = not board.turn
    out = []
    for i, uci in enumerate(moves):
        move = board.parse_uci(uci)                 # reads e1g1 as castling as well as e1h1
        before = board.copy(stack=False) if board.turn == solver else None
        board.push(move)
        if before is not None:
            out.append((before, move, board.copy(stack=False), i == len(moves) - 1))
    return out


def material_signature(board: chess.Board) -> Tuple[str, str, int]:
    """(the side to move's pieces other than pawns and king, the other side's, the number of pawns on the board)."""
    def pieces(color: bool) -> str:
        return "".join(sorted(chess.piece_symbol(p.piece_type) for p in board.piece_map().values()
                              if p.color == color and p.piece_type not in (chess.PAWN, chess.KING)))
    pawns = len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK))
    return pieces(board.turn), pieces(not board.turn), pawns


def match_by_signature(positives: Sequence[Tuple[object, object]], negatives: Sequence[Tuple[object, object]],
                       rng: random.Random, limit: Optional[int] = None) -> Tuple[List[object], List[object]]:
    """
    (signature, item) lists -> equally many positive and negative items for every signature (as many as the
    scarcer side has), at most `limit` of each in all, drawn at random.
    """
    by_sig: Dict[object, Tuple[List[object], List[object]]] = collections.defaultdict(lambda: ([], []))
    for sig, item in positives:
        by_sig[sig][0].append(item)
    for sig, item in negatives:
        by_sig[sig][1].append(item)
    pos, neg = [], []
    for sig in sorted(by_sig, key=repr):
        p, n = by_sig[sig]
        k = min(len(p), len(n))
        pos += rng.sample(p, k)
        neg += rng.sample(n, k)
    order = list(range(len(pos)))
    rng.shuffle(order)
    order = order[:limit] if limit is not None else order
    return [pos[i] for i in order], [neg[i] for i in order]


@dataclass
class ProbeSet:
    """One probe's positions and answers. kind: "binary", "regression" or "move"."""
    name: str
    kind: str
    fens: List[str]
    y: np.ndarray                                   # binary: 0 / 1; regression: the value; move: the pair index
    legal: Optional[List[np.ndarray]] = None        # move: every position's legal pair indices
    notes: str = ""


def _hash(text: str, salt: str) -> float:
    return pl._hash_unit(text, salt)


def build_puzzle_sets(path: str = DEFAULT_PUZZLES, size: int = 100_000, seed: int = 0,
                      log: Callable[[str], None] = print, limit: Optional[int] = None) -> Dict[str, ProbeSet]:
    """zugzwang, quiet_detect, quiet_find and forcing_find from the Lichess puzzle database (see the module docstring)."""
    t0 = time.time()
    rng = random.Random(seed)
    total = collections.Counter()
    for n, (_, _, _, themes) in enumerate(puzzle_rows(path)):
        if limit is not None and n >= limit:
            break
        total["all"] += 1
        total["quiet"] += "quietMove" in themes
        total["endgame_other"] += "endgame" in themes and "zugzwang" not in themes
    quiet_rate = min(1.0, 1.25 * size / max(1, total["quiet"]))
    forcing_rate = min(1.0, 5.0 * size / max(1, total["all"] - total["quiet"]))
    endgame_rate = min(1.0, 12.0 * size / max(1, total["endgame_other"]))

    zug_pos, zug_neg = [], []                       # (signature, fen)
    quiet: List[Tuple[str, int, np.ndarray, int]] = []                     # (fen, pair, legal pairs, solver-move number)
    forcing: List[Dict[int, Tuple[str, int, np.ndarray]]] = []             # per puzzle: solver-move number -> record
    checks = collections.Counter()
    for n, (pid, fen, moves, themes) in enumerate(puzzle_rows(path)):
        if limit is not None and n >= limit:
            break
        is_zug, is_quiet = "zugzwang" in themes, "quietMove" in themes
        want_zug = is_zug or ("endgame" in themes and _hash(pid, f"zug{seed}") < endgame_rate)
        want_quiet = is_quiet and _hash(pid, f"quiet{seed}") < quiet_rate
        want_forcing = not is_quiet and _hash(pid, f"forcing{seed}") < forcing_rate
        if not (want_zug or want_quiet or want_forcing):
            continue
        try:
            if want_quiet or want_forcing:
                line = solver_moves(fen, moves)
                start = line[0][0] if line else None
            else:                                   # the starting position alone: no need to play the line out
                line, start = [], chess.Board(fen, chess960=True)
                start.push(start.parse_uci(moves[0]))
        except (ValueError, IndexError):
            checks["unplayable"] += 1
            continue
        if start is None:
            continue
        if want_zug:
            (zug_pos if is_zug else zug_neg).append((material_signature(start), ld.fen_of(start)))
        if want_quiet or want_forcing:
            found_quiet, by_k = False, {}
            for k, (before, move, after, last) in enumerate(line):
                quiet_here = is_quiet_move(before, move, after, last)
                if want_quiet and quiet_here and not found_quiet:
                    quiet.append((ld.fen_of(before), move_pair(before, move), legal_pairs(before), k))
                found_quiet = found_quiet or quiet_here
                if want_forcing and not last and is_forcing_move(before, move):
                    by_k[k] = (ld.fen_of(before), move_pair(before, move), legal_pairs(before))
            checks["tagged quiet" if is_quiet else "not tagged"] += 1
            checks["tagged quiet, rule finds one" if is_quiet else "not tagged, rule finds one"] += found_quiet
            if want_forcing and by_k:
                forcing.append(by_k)
    log(f"puzzles read in {time.time() - t0:.0f}s: {total['all']:,} puzzles; the quiet-move rule finds a quiet move in "
        f"{checks['tagged quiet, rule finds one']:,} of {checks['tagged quiet']:,} sampled quietMove puzzles and in "
        f"{checks['not tagged, rule finds one']:,} of {checks['not tagged']:,} sampled others "
        f"({checks['unplayable']} unplayable lines skipped)")

    sets: Dict[str, ProbeSet] = {}
    pos, neg = match_by_signature(zug_pos, zug_neg, rng, limit=size // 2)
    sets["zugzwang"] = ProbeSet("zugzwang", "binary", list(pos) + list(neg),
                                np.array([1.0] * len(pos) + [0.0] * len(neg), dtype=np.float32),
                                notes=f"{len(pos):,} zugzwang puzzles of {len(zug_pos):,}, each with an endgame puzzle "
                                      f"of the same material (from {len(zug_neg):,} candidates)")

    rng.shuffle(quiet)
    quiet = quiet[:size]
    k_counts = collections.Counter(k for _, _, _, k in quiet)
    ks, weights = zip(*sorted(k_counts.items()))
    forcing_records = []
    for by_k in forcing:                            # the same number of moves into the solution as the quiet positions
        k = rng.choices(ks, weights)[0]
        if k in by_k:
            forcing_records.append(by_k[k] + (k,))
    rng.shuffle(forcing_records)
    forcing_records = forcing_records[:size]

    def move_set(name: str, records, notes: str) -> ProbeSet:
        return ProbeSet(name, "move", [r[0] for r in records], np.array([r[1] for r in records], dtype=np.int64),
                        legal=[r[2] for r in records], notes=notes)

    k_text = ", ".join(f"move {k + 1}: {c / len(quiet):.0%}" for k, c in sorted(k_counts.items())[:4])
    sets["quiet_find"] = move_set("quiet_find", quiet, f"{len(quiet):,} positions before a quiet move ({k_text})")
    sets["forcing_find"] = move_set("forcing_find", forcing_records,
                                    f"{len(forcing_records):,} positions before a capture or check, the same number "
                                    "of moves into the solution")
    half = min(len(quiet), len(forcing_records), size // 2)
    sets["quiet_detect"] = ProbeSet("quiet_detect", "binary",
                                    [r[0] for r in quiet[:half]] + [r[0] for r in forcing_records[:half]],
                                    np.array([1.0] * half + [0.0] * half, dtype=np.float32),
                                    notes=f"{half:,} quiet and {half:,} forcing positions")
    return sets


def build_engine_sets(db_path: str = DEFAULT_DB, size: int = 100_000, seed: int = 0,
                      log: Callable[[str], None] = print) -> Dict[str, ProbeSet]:
    """best_move, eval and material from the engine-analyzed positions of minotaur_data.db."""
    uri = "file:" + os.path.abspath(db_path).replace("\\", "/") + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute('SELECT fen, score, best_move FROM "960_position_data" WHERE is_analyzed = 1').fetchall()
    random.Random(seed).shuffle(rows)
    fens, win, material, pairs, legal = [], [], [], [], []
    for fen, score, best in rows:
        norm = ld.normalize_fen(fen)
        if norm is None or score is None or not best:
            continue
        board = chess.Board(norm, chess960=True)
        try:
            move = board.parse_uci(best)
        except ValueError:
            continue
        fens.append(norm)
        win.append(pl.score_to_side_to_move(str(score), board.turn == chess.WHITE)[0])
        material.append(pl.board_labels(board)["material_balance"])
        pairs.append(move_pair(board, move))
        legal.append(legal_pairs(board))
        if len(fens) >= size:
            break
    log(f"engine-analyzed positions: {len(fens):,}")
    return {
        "best_move": ProbeSet("best_move", "move", fens, np.array(pairs, dtype=np.int64), legal=legal,
                              notes=f"{len(fens):,} engine-analyzed positions"),
        "eval": ProbeSet("eval", "regression", fens, np.array(win, dtype=np.float32), notes="the same positions"),
        "material": ProbeSet("material", "regression", fens, np.array(material, dtype=np.float32),
                             notes="the same positions"),
    }


def load_sets(puzzles: str, db: str, size: int, seed: int, cache_dir: Optional[str],
              log: Callable[[str], None] = print) -> Dict[str, ProbeSet]:
    """Every probe's set, built once and kept in cache_dir."""
    path = os.path.join(cache_dir, f"probe_heads_sets_{size}_{seed}.pkl") if cache_dir else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    sets = {**build_puzzle_sets(puzzles, size, seed, log), **build_engine_sets(db, size, seed, log)}
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(sets, f)
    return sets


# ##### ##### ##### ##### #####
#       Inputs: each encoder's z, and the raw board


@torch.no_grad()
def encode_inputs(encoders: Dict[str, torch.nn.Module], fens: Sequence[str], device,
                  batch_size: int = 250) -> Dict[str, torch.Tensor]:
    """{name: [N, dim] float16 on the CPU} for every encoder's z, and RAW: the 64 squares' node features, flattened."""
    out: Dict[str, List[torch.Tensor]] = {name: [] for name in [RAW, *encoders]}
    none = torch.zeros(64, dtype=torch.bool)
    for lo in range(0, len(fens), batch_size):
        graphs = ld.pack_views([ld.build_views(f, none)[0] for f in fens[lo:lo + batch_size]]).to(device)
        out[RAW].append(graphs.x.view(graphs.num_graphs, -1).half().cpu())
        for name, encoder in encoders.items():
            out[name].append(encoder(graphs).z.half().cpu())
    return {name: torch.cat(parts) for name, parts in out.items()}


# ##### ##### ##### ##### #####
#       The head


def split_of(fens: Sequence[str]) -> np.ndarray:
    """0 train (80%), 1 validation (10%), 2 test (10%), by a hash of the position."""
    u = np.array([_hash(ld.position_key(f), "heads") for f in fens])
    return np.where(u < 0.8, 0, np.where(u < 0.9, 1, 2))


def legal_mask(legal: Sequence[np.ndarray], device) -> torch.Tensor:
    """[N, NUM_PAIRS] bool, True on every position's legal pairs."""
    rows = np.repeat(np.arange(len(legal)), [len(a) for a in legal])
    cols = np.concatenate(legal).astype(np.int64)
    mask = torch.zeros(len(legal), NUM_PAIRS, dtype=torch.bool, device=device)
    mask[torch.as_tensor(rows, device=device), torch.as_tensor(cols, device=device)] = True
    return mask


def _score(kind: str, out: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    if kind == "binary":
        return {"auc": pl.auc(out.squeeze(1), y), "accuracy": float(((out.squeeze(1) > 0).float() == y).float().mean())}
    if kind == "regression":
        return {"r2": pl.r2(out.squeeze(1), y)}
    top = out.topk(3, dim=1).indices
    return {"top1": float((top[:, 0] == y).float().mean()), "top3": float((top == y[:, None]).any(1).float().mean())}


HEADLINE = {"binary": "auc", "regression": "r2", "move": "top1"}


def fit_head(x: torch.Tensor, y: np.ndarray, split: np.ndarray, kind: str, legal: Optional[torch.Tensor] = None,
             seed: int = 0, device="cuda", hidden: int = 1024, epochs: int = 30, batch_size: int = 512,
             lr: float = 1e-3, weight_decay: float = 0.01, patience: int = 4) -> Dict[str, float]:
    """
    Train one head (input -> hidden -> hidden -> output) on the training rows and score it on the test rows at
    the epoch that did best on the validation rows. x [N, dim] (any float type; standardized here on the
    training rows); kind "binary" (y 0 / 1), "regression" (y standardized on the training rows inside, scored as
    R2 of the original values) or "move" (y the pair index; legal [N, NUM_PAIRS] masks the other pairs out).
    """
    torch.manual_seed(seed)
    x = x.to(device).float()
    parts = [torch.as_tensor(np.flatnonzero(split == s), device=device) for s in (0, 1, 2)]
    mean, std = x[parts[0]].mean(0), x[parts[0]].std(0)
    std = torch.where(std > 1e-3 * std.max().clamp_min(1e-12), std, torch.ones_like(std))
    x = (x - mean) / std
    target = torch.as_tensor(y, device=device)
    shift = scale = None
    if kind == "regression":
        shift, scale = target[parts[0]].mean(), target[parts[0]].std().clamp_min(1e-6)
        target = (target - shift) / scale
    outputs = NUM_PAIRS if kind == "move" else 1
    head = lje._mlp([x.size(1), hidden, hidden, outputs]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * max(1, (len(parts[0]) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    generator = torch.Generator(device=device).manual_seed(seed)

    def forward(rows: torch.Tensor) -> torch.Tensor:
        out = head(x[rows])
        return out.masked_fill(~legal[rows], float("-inf")) if kind == "move" else out

    def evaluate(rows: torch.Tensor) -> Dict[str, float]:
        head.eval()
        with torch.no_grad():
            out = torch.cat([forward(rows[i:i + 4096]) for i in range(0, len(rows), 4096)])
        head.train()
        truth = target[rows]
        if kind == "regression":
            out, truth = out * scale + shift, truth * scale + shift
        return _score(kind, out, truth)

    best, best_state, stale = -float("inf"), None, 0
    for _ in range(epochs):
        order = parts[0][torch.randperm(len(parts[0]), device=device, generator=generator)]
        for i in range(0, len(order), batch_size):
            rows = order[i:i + batch_size]
            out = forward(rows)
            if kind == "binary":
                loss = F.binary_cross_entropy_with_logits(out.squeeze(1), target[rows])
            elif kind == "regression":
                loss = F.mse_loss(out.squeeze(1), target[rows])
            else:
                loss = F.cross_entropy(out, target[rows])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        value = evaluate(parts[1])[HEADLINE[kind]]
        if value > best + 1e-5:
            best, stale = value, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    head.load_state_dict(best_state)
    return evaluate(parts[2])


# ##### ##### ##### ##### #####
#       Running the probes


def run_probes(encoders: Dict[str, torch.nn.Module], sets: Dict[str, ProbeSet], device, seeds: int = 2,
               log: Callable[[str], None] = print) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{probe: {input name: {score: mean over seeds}}}; inputs are RAW and every encoder's z."""
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    encoded: Dict[Tuple[str, ...], Dict[str, torch.Tensor]] = {}
    for name, probe in sets.items():
        t0 = time.time()
        key = tuple(probe.fens)
        if key not in encoded:                      # probes on the same positions share their encodings
            encoded.clear()
            encoded[key] = encode_inputs(encoders, probe.fens, device)
        split = split_of(probe.fens)
        mask = legal_mask(probe.legal, device) if probe.kind == "move" else None
        results[name] = {}
        for source, x in encoded[key].items():
            runs = [fit_head(x, probe.y, split, probe.kind, mask, seed=s, device=device) for s in range(seeds)]
            results[name][source] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            results[name][source]["spread"] = float(np.ptp([r[HEADLINE[probe.kind]] for r in runs]))
        counts = np.bincount(split, minlength=3)
        log(f"{name}: {counts[0]:,} / {counts[1]:,} / {counts[2]:,} train / validation / test positions, "
            f"{time.time() - t0:.0f}s. {probe.notes}")
    return results


COLUMNS = (("zugzwang", "auc", "zugzwang, AUC"), ("quiet_detect", "auc", "quiet or forcing, AUC"),
           ("quiet_find", "top1", "find the quiet move, 1st"), ("quiet_find", "top3", "in first 3"),
           ("forcing_find", "top1", "find the forcing move, 1st"), ("forcing_find", "top3", "in first 3"),
           ("best_move", "top1", "engine's best move, 1st"), ("best_move", "top3", "in first 3"),
           ("eval", "r2", "eval, R2"), ("material", "r2", "material, R2"))


def table(results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    columns = [c for c in COLUMNS if c[0] in results]
    sources = list(next(iter(results.values())))
    lines = ["| input | " + " | ".join(c[2] for c in columns) + " |", "|---|" + "---|" * len(columns)]
    for source in sources:
        cells = []
        for probe, score, _ in columns:
            value = results[probe][source][score]
            cells.append(f"{value:.3f}" if score in ("auc", "r2") else f"{value:.1%}")
        lines.append(f"| {source} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("encoders", nargs="*", help="name=run_folder:snapshot_step")
    parser.add_argument("--puzzles", default=DEFAULT_PUZZLES)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--size", type=int, default=100_000, help="positions per probe (fewer where the data runs out)")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="for drawing the probe sets")
    parser.add_argument("--only", default=None, help="comma-separated probe names")
    parser.add_argument("--online", action="store_true", help="the online encoder instead of the target encoder")
    parser.add_argument("--cache-dir", default=os.path.join(ROOT, "runs", "analysis", "cache"))
    parser.add_argument("--out", default=None, help="write the results as JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = load_sets(args.puzzles, args.db, args.size, args.seed, args.cache_dir)
    if args.only:
        sets = {name: sets[name] for name in args.only.split(",")}
    encoders = {}
    for spec in args.encoders:
        if spec == RAW:
            continue
        name, _, where = spec.partition("=")
        run, _, snap = where.rpartition(":")
        path = os.path.join(run, "snapshots", f"step_{int(snap):07d}.pt")
        encoders[name] = pl.load_encoder(path, device, "online" if args.online else "target")
    results = run_probes(encoders, sets, device, seeds=args.seeds)
    print()
    print(table(results), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"results": results, "notes": {name: s.notes for name, s in sets.items()}}, f, indent=2)


if __name__ == "__main__":
    main()
