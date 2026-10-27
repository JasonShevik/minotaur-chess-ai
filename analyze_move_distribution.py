"""
How lopsided is a uniformly random legal move? The JEPA rollouts play uniform random legal moves; this
measures, by game phase, which pieces those moves move and what kind of moves they are, against the
moves people actually play.

    python analyze_move_distribution.py --games lichess-games [--db minotaur_data.db] --out move_distribution.csv

Positions come from the games (a share of every game's positions, from archives spread across the
folder) and, with --db, from the engine-analyzed rows of the labeled database. Each position counts
once, as it does in training (one sampled move per position), under each scheme:

    uniform     one legal move uniformly at random (what the rollouts do)
    piece       a movable piece uniformly at random, then one of its moves uniformly
    played      the move played in the game (game positions only)
    engine      the engine's best move (labeled positions only)

Columns: the moving piece's share by type (P N B R Q K), and the share of moves that capture, give
check, castle, promote, or hang the moved piece. "Hangs" is a one-ply heuristic: the moved piece ends
on a square the opponent attacks, undefended or attacked by something cheaper, and that loss exceeds
what the move captured (it ignores pins, x-rays and pieces left behind). tv_vs_reference is the total
variation distance between a scheme's piece-type shares and the played (or engine) move's.

Buckets: opening (the first 20 plies), endgame (non-pawn material of at most 6, counting N and B as 1,
R as 2 and Q as 4, of 24 at the start), middlegame (the rest), "all", and ply ranges.

Findings on 12 archives (2026-10): by piece type, uniform moves are close to what people play (total
variation 0.07; queen and rook moves 31% against 26%). The difference is in kind: uniform moves
capture 6% of the time against 23%, and hang the moved piece 22% of the time against 2.5%.
"""
from __future__ import annotations

import argparse
import collections
import csv
import multiprocessing as mp
import random
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

import chess
import chess.pgn

from lichess_position_grabber import find_archives, open_pgn


PIECES = ["P", "N", "B", "R", "Q", "K"]
FLAGS = ["capture", "check", "castle", "promotion", "hangs"]
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
# A king can only take an undefended piece, so as the cheapest attacker of a defended one it never counts.
ATTACKER_VALUE = {**VALUE, chess.KING: 100}
PHASE_WEIGHT = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
PLY_EDGES = (10, 20, 40, 60, 80, 100)
LABELED_NOTE = ("* sample bias: only about 5% of the engine-analyzed positions have castling rights (about half "
                "of all positions do), so castling is missing from the engine moves")


# ##### ##### ##### ##### #####
#       Positions and moves

def material_phase(board: chess.Board) -> int:
    """Non-pawn material, counting knights and bishops as 1, rooks as 2 and queens as 4 (24 at the start)."""
    return sum(PHASE_WEIGHT.get(p.piece_type, 0) for p in board.piece_map().values())


def game_phase(board: chess.Board) -> str:
    if board.ply() < 20:
        return "opening"
    return "endgame" if material_phase(board) <= 6 else "middlegame"


def ply_bucket(ply: int) -> str:
    for edge in PLY_EDGES:
        if ply < edge:
            return f"ply<{edge}"
    return f"ply>={PLY_EDGES[-1]}"


def move_properties(board: chess.Board, move: chess.Move) -> Tuple[str, Dict[str, bool]]:
    """The moving piece's letter (P N B R Q K) and the move's flags (see FLAGS and the module docstring)."""
    mover = board.turn
    castle = board.is_castling(move)
    capture = board.is_capture(move) and not castle
    if board.is_en_passant(move):
        gained = VALUE[chess.PAWN]
    elif capture:
        gained = VALUE[board.piece_at(move.to_square).piece_type]
    else:
        gained = 0
    letter = board.piece_at(move.from_square).symbol().upper()
    check = board.gives_check(move)
    loss = 0
    board.push(move)
    try:
        moved = board.piece_at(move.to_square)
        attackers = board.attackers(not mover, move.to_square)
        if not castle and attackers and moved.piece_type != chess.KING:
            value = VALUE[moved.piece_type]
            if not board.attackers(mover, move.to_square):
                loss = value
            else:
                loss = max(0, value - min(ATTACKER_VALUE[board.piece_at(s).piece_type] for s in attackers))
    finally:
        board.pop()
    return letter, {"capture": capture, "check": check, "castle": castle,
                    "promotion": move.promotion is not None, "hangs": loss > gained}


# ##### ##### ##### ##### #####
#       Counting

class MoveStats:
    """
    Sums over positions, by (source, bucket, scheme), of the chance that the scheme's move has each piece
    type and flag. Dividing by the bucket's position count gives the shares.
    """

    def __init__(self):
        self.positions = collections.Counter()      # (source, bucket) -> positions
        self.legal = collections.Counter()          # ... -> legal moves, summed
        self.queens = collections.Counter()         # ... -> positions with a queen on the board
        self.rook_queen_heavy = collections.Counter()   # ... -> positions where Q and R moves are over half
        self.sums: Dict[Tuple[str, str, str], collections.Counter] = collections.defaultdict(collections.Counter)

    def add(self, source: str, buckets: Sequence[str], board: chess.Board, reference: Optional[chess.Move] = None,
            reference_name: str = "played") -> None:
        moves = list(board.legal_moves)
        if not moves:
            return
        props = [move_properties(board, m) for m in moves]
        by_piece: Dict[int, List[Tuple[str, Dict[str, bool]]]] = collections.defaultdict(list)
        for move, p in zip(moves, props):
            by_piece[move.from_square].append(p)
        uniform = _weighted([(1.0 / len(props), p) for p in props])
        piece = _weighted([(1.0 / len(by_piece) / len(ps), p) for ps in by_piece.values() for p in ps])
        schemes = [("uniform", uniform), ("piece", piece)]
        if reference is not None:
            schemes.append((reference_name, _weighted([(1.0, move_properties(board, reference))])))
        has_queen = any(p.piece_type == chess.QUEEN for p in board.piece_map().values())
        for bucket in buckets:
            key = (source, bucket)
            self.positions[key] += 1
            self.legal[key] += len(moves)
            self.queens[key] += has_queen
            self.rook_queen_heavy[key] += uniform["Q"] + uniform["R"] > 0.5
            for name, shares in schemes:
                self.sums[key + (name,)].update(shares)

    def merge(self, other: "MoveStats") -> None:
        for mine, theirs in [(self.positions, other.positions), (self.legal, other.legal),
                             (self.queens, other.queens), (self.rook_queen_heavy, other.rook_queen_heavy)]:
            mine.update(theirs)
        for key, sums in other.sums.items():
            self.sums[key].update(sums)

    def rows(self) -> List[Dict[str, object]]:
        """One row per (source, bucket, scheme), in report order."""
        order = ["opening", "middlegame", "endgame", "all"] + [ply_bucket(e - 1) for e in PLY_EDGES] + \
                [ply_bucket(PLY_EDGES[-1])]
        out = []
        for source in sorted({s for s, _ in self.positions}):
            for bucket in [b for b in order if (source, b) in self.positions]:
                n = self.positions[(source, bucket)]
                schemes = [s for s in ("uniform", "piece", "played", "engine") if (source, bucket, s) in self.sums]
                shares = {s: {k: self.sums[(source, bucket, s)][k] / n for k in PIECES + FLAGS} for s in schemes}
                reference = next((shares[s] for s in ("played", "engine") if s in shares), None)
                for scheme in schemes:
                    row: Dict[str, object] = {
                        "source": source, "bucket": bucket, "scheme": scheme, "positions": n,
                        "avg_legal_moves": self.legal[(source, bucket)] / n,
                        "queen_on_board": self.queens[(source, bucket)] / n,
                        "qr_over_half_of_legal": self.rook_queen_heavy[(source, bucket)] / n,
                    }
                    row.update({f"share_{k}": shares[scheme][k] for k in PIECES})
                    row.update({k: shares[scheme][k] for k in FLAGS})
                    row["tv_vs_reference"] = (0.5 * sum(abs(shares[scheme][k] - reference[k]) for k in PIECES)
                                              if reference is not None else None)
                    row["notes"] = LABELED_NOTE if source == "labeled" else ""
                    out.append(row)
        return out


def _weighted(items: Sequence[Tuple[float, Tuple[str, Dict[str, bool]]]]) -> collections.Counter:
    total = collections.Counter()
    for weight, (letter, flags) in items:
        total[letter] += weight
        for k in FLAGS:
            total[k] += weight * flags[k]
    return total


# ##### ##### ##### ##### #####
#       Sources

def scan_archive(job: Tuple[str, int, float, int]) -> MoveStats:
    """Positions from the first `games` games of one archive, each kept with chance `keep`."""
    path, games, keep, seed = job
    rng = random.Random(seed)
    stats = MoveStats()
    with open_pgn(path) as f:
        for _ in range(games):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                if rng.random() < keep:
                    stats.add("games", [game_phase(board), ply_bucket(board.ply()), "all"], board, move)
                board.push(move)
    return stats


def scan_labeled(db_path: str, limit: int = 40_000, seed: int = 0) -> MoveStats:
    """Engine-analyzed positions of the labeled database, with the engine's best move as the reference."""
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute('SELECT fen, best_move FROM "960_position_data" '
                           'WHERE is_analyzed = 1 AND best_move IS NOT NULL AND inconsistent = 0').fetchall()
    finally:
        con.close()
    random.Random(seed).shuffle(rows)
    stats = MoveStats()
    for fen, best in rows[:limit]:
        board = chess.Board(fen, chess960=True)
        try:
            move = chess.Move.from_uci(best)
        except ValueError:
            continue
        if move in board.legal_moves:
            stats.add("labeled", [game_phase(board), "all"], board, move, reference_name="engine")
    return stats


def analyze(games: Optional[str], db: Optional[str] = None, files: int = 12, games_per_file: int = 2500,
            keep: float = 0.25, labeled_limit: int = 40_000, workers: int = 12, seed: int = 0,
            log=print) -> MoveStats:
    """Scan `files` archives spread evenly across the `games` folder, plus the labeled database."""
    archives = find_archives(games) if games else []
    picks = archives[:: max(1, len(archives) // files)][:files] if archives else []
    jobs = [(path, games_per_file, keep, seed * 1000 + i) for i, path in enumerate(picks)]
    log(f"scanning {len(picks)} archives" + (" and the labeled database" if db else ""))
    total = MoveStats()
    if workers <= 0:
        for job in jobs:
            total.merge(scan_archive(job))
        if db:
            total.merge(scan_labeled(db, labeled_limit, seed))
        return total
    with mp.Pool(min(workers, len(jobs) + 1)) as pool:
        labeled = pool.apply_async(scan_labeled, (db, labeled_limit, seed)) if db else None
        for stats in pool.imap_unordered(scan_archive, jobs):
            total.merge(stats)
        if labeled is not None:
            total.merge(labeled.get())
    return total


# ##### ##### ##### ##### #####
#       Output

COLUMNS = (["source", "bucket", "scheme", "positions", "avg_legal_moves", "queen_on_board", "qr_over_half_of_legal"]
           + [f"share_{k}" for k in PIECES] + FLAGS + ["tv_vs_reference", "notes"])


def write_csv(rows: Sequence[Dict[str, object]], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v))
                             for k, v in row.items()})


def print_report(rows: Sequence[Dict[str, object]], log=print) -> None:
    current = None
    for row in rows:
        if (row["source"], row["bucket"]) != current:
            current = (row["source"], row["bucket"])
            log(f"\n{row['source']} / {row['bucket']}: {row['positions']} positions, "
                f"{row['avg_legal_moves']:.1f} legal moves on average, a queen on the board in "
                f"{row['queen_on_board']:.0%}, queen and rook moves over half the legal moves in "
                f"{row['qr_over_half_of_legal']:.0%}")
            log(f"  {'scheme':8}" + "".join(f"{k:>7}" for k in PIECES) + "  |" +
                "".join(f"{k:>10}" for k in FLAGS) + "   tv")
        tv = row["tv_vs_reference"]
        log(f"  {row['scheme']:8}" + "".join(f"{row['share_' + k]:7.1%}" for k in PIECES) + "  |" +
            "".join(f"{row[k]:10.1%}" for k in FLAGS) + ("" if tv is None else f"  {tv:.2f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Piece types and kinds of uniform random moves vs played moves.")
    parser.add_argument("--games", default="lichess-games", help="folder of .pgn / .pgn.zst archives, or one file")
    parser.add_argument("--db", default=None, help="labeled database (engine-analyzed positions), optional")
    parser.add_argument("--out", default="move_distribution.csv")
    parser.add_argument("--files", type=int, default=12, help="archives to scan, spread across the folder")
    parser.add_argument("--games-per-file", type=int, default=2500)
    parser.add_argument("--keep", type=float, default=0.25, help="share of each game's positions to use")
    parser.add_argument("--labeled-limit", type=int, default=40_000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    stats = analyze(args.games, args.db, args.files, args.games_per_file, args.keep, args.labeled_limit,
                    args.workers, args.seed)
    rows = stats.rows()
    print_report(rows)
    write_csv(rows, args.out)
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
