"""
Samples positions from Lichess game archives into a text file with one FEN per line.

    python lichess_position_grabber.py [--games lichess-games] [--out ssl_positions.txt] [options]

Reads .pgn files and the .pgn.zst archives that https://database.lichess.org/ distributes (those need
the zstandard package: pip install zstandard), in parallel, one worker per file. From every game it
picks up to max_positions_per_game random positions, at least min_ply_gap plies apart (as many as the
game allows: a short game gives fewer), by default excluding:
  * the starting position (there are only 960 of them, so they would repeat endlessly);
  * the final position (often mate or the moment of resignation);
  * positions with endgame_max_pieces pieces or fewer (skip_endgames): chess is solved there, and
    endgame tablebases are better training data for them. Set skip_endgames=False to keep them.

The output file is what the rest of the pipeline reads: load it into the database with
db_tools.add_ssl_positions (the table for self-supervised positions), or build the training cache
from it directly (python labyrinth_data.py build --fens ...).

FENs are written in X-FEN with the en passant square whenever a pawn could capture there, the
convention labyrinth_data uses. Duplicates are removed within each archive; the database and the
training cache remove duplicates across archives.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import random
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from typing import Dict, Iterator, List, Optional, Sequence, TextIO, Tuple

import chess
import chess.pgn


@dataclass
class GrabOptions:
    max_positions_per_game: int = 10
    min_ply_gap: int = 5              # two positions from the same game are at least this many plies apart
    skip_endgames: bool = True        # skip positions with endgame_max_pieces pieces or fewer
    endgame_max_pieces: int = 7       # 7-piece endgames are solved (Syzygy tablebases)
    skip_start_position: bool = True
    skip_final_position: bool = True
    skip_games_with_errors: bool = True   # games python-chess could not read cleanly
    max_games_per_file: Optional[int] = None   # stop each archive early (for quick runs)
    seed: int = 0


def open_pgn(path: str) -> TextIO:
    """Open a .pgn file, or a zstandard-compressed .pgn.zst archive as a text stream."""
    if path.lower().endswith(".zst"):
        try:
            import zstandard
        except ImportError as e:
            raise ImportError("Reading .pgn.zst archives needs the zstandard package: pip install zstandard") from e
        raw = open(path, "rb")
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw, closefd=True), encoding="utf-8")
    return open(path, encoding="utf-8")


def positions_from_game(game: chess.pgn.Game, options: GrabOptions, rng: random.Random) -> List[str]:
    """
    Random positions of one game, in the order they occurred: as many as fit, up to
    max_positions_per_game, with consecutive positions at least min_ply_gap plies apart.
    """
    moves = list(game.mainline_moves())
    board = game.board()

    # First pass: how many positions are eligible. Material never increases, so once the board is down
    # to endgame_max_pieces pieces every later position is too.
    first = 0 if not options.skip_start_position else 1
    last = len(moves) - 1 if options.skip_final_position else len(moves)     # plies, inclusive
    if options.skip_endgames:
        if chess.popcount(board.occupied) <= options.endgame_max_pieces:
            return []
        for ply, move in enumerate(moves, 1):
            board.push(move)
            if chess.popcount(board.occupied) <= options.endgame_max_pieces:
                last = min(last, ply - 1)
                break
    if last < first:
        return []

    chosen = spaced_sample(rng, first, last, options.max_positions_per_game, options.min_ply_gap)
    if not chosen:
        return []

    # Second pass: replay up to the last chosen ply and record the chosen positions
    board = game.board()
    out = []
    wanted = iter(chosen)
    target = next(wanted)
    for ply in range(0, chosen[-1] + 1):
        if ply > 0:
            board.push(moves[ply - 1])
        if ply == target:
            out.append(board.fen(en_passant="xfen"))
            target = next(wanted, None)
    return out


def spaced_sample(rng: random.Random, first: int, last: int, max_count: int, min_gap: int) -> List[int]:
    """
    Sorted random integers in [first, last], as many as fit up to max_count, every two at least min_gap
    apart, uniform over all such sets of that size: draw k distinct values from a range shortened by
    (k - 1) * (min_gap - 1), then spread the i-th smallest by i * (min_gap - 1).
    """
    gap = max(1, min_gap)
    span = last - first + 1
    if span <= 0 or max_count <= 0:
        return []
    k = min(max_count, (span - 1) // gap + 1)
    base = sorted(rng.sample(range(span - (k - 1) * (gap - 1)), k))
    return [first + v + i * (gap - 1) for i, v in enumerate(base)]


def _file_seed(seed: int, path: str) -> int:
    """
    A seed per archive, from its name without a .zst suffix: results do not depend on the number of
    workers, and an archive samples the same positions compressed or not.
    """
    name = os.path.basename(path)
    if name.lower().endswith(".zst"):
        name = name[:-4]
    digest = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def grab_from_file(path: str, options: GrabOptions) -> Tuple[List[str], Dict[str, int]]:
    """The sampled positions of one archive (duplicates removed, in order of first appearance) and counts."""
    rng = random.Random(_file_seed(options.seed, path))
    seen: set = set()
    positions: List[str] = []
    games = skipped = 0
    with open_pgn(path) as pgn:
        while options.max_games_per_file is None or games < options.max_games_per_file:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games += 1
            if options.skip_games_with_errors and game.errors:
                skipped += 1
                continue
            for fen in positions_from_game(game, options, rng):
                if fen not in seen:
                    seen.add(fen)
                    positions.append(fen)
    return positions, {"games": games, "skipped_games": skipped, "positions": len(positions)}


def _grab_job(job: Tuple[str, GrabOptions]) -> Tuple[str, List[str], Dict[str, int], float]:
    path, options = job
    t = time.time()
    positions, counts = grab_from_file(path, options)
    return path, positions, counts, time.time() - t


def find_archives(source: str) -> List[str]:
    """Every .pgn and .pgn.zst file in a folder (sorted), or the file itself."""
    if os.path.isfile(source):
        return [source]
    names = sorted(n for n in os.listdir(source) if n.lower().endswith((".pgn", ".pgn.zst")))
    return [os.path.join(source, n) for n in names]


def grab_positions(
    sources: Sequence[str] | str = "lichess-games",
    output_path: str = "ssl_positions.txt",
    options: GrabOptions = GrabOptions(),
    workers: int = 1,
    split_into: int = 1,
    log=print,
) -> Dict[str, object]:
    """
    Sample positions from every archive in sources (folders or files) and write them to output_path,
    one FEN per line, archive by archive in sorted order. split_into > 1 also splits the output into
    that many part files (output_part_1.txt, ...). Returns the counts.
    """
    if isinstance(sources, str):
        sources = [sources]
    paths = [p for s in sources for p in find_archives(s)]
    if not paths:
        raise FileNotFoundError(f"no .pgn or .pgn.zst files in {list(sources)}")
    log(f"Sampling up to {options.max_positions_per_game} position(s) per game, {options.min_ply_gap}+ plies apart, "
        f"from {len(paths)} archive(s) with {workers} worker(s); skip_endgames={options.skip_endgames}")
    totals = {"archives": len(paths), "games": 0, "skipped_games": 0, "positions": 0}
    jobs = [(p, options) for p in paths]
    t0 = time.time()
    with open(output_path, "w", encoding="utf-8") as out:
        if workers > 1:
            pool = Pool(min(workers, len(paths)))
            results = pool.imap(_grab_job, jobs)    # in order, so the output is the same for any worker count
        else:
            pool = None
            results = map(_grab_job, jobs)
        try:
            for path, positions, counts, seconds in results:
                if positions:
                    out.write("\n".join(positions) + "\n")
                for k in ("games", "skipped_games", "positions"):
                    totals[k] += counts[k]
                log(f"  {os.path.basename(path)}: {counts['games']:,} games, {counts['positions']:,} positions "
                    f"({seconds:.0f}s)")
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    if split_into > 1:
        totals["parts"] = split_file(output_path, split_into)
    log(f"Wrote {totals['positions']:,} positions from {totals['games']:,} games to {output_path} "
        f"in {time.time() - t0:.0f}s")
    return totals


def split_file(path: str, num_files: int = 5) -> List[str]:
    """Split a text file into num_files parts of nearly equal line counts: <stem>_part_<i>.txt."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    stem = os.path.splitext(path)[0]
    size = -(-len(lines) // num_files)
    parts = []
    for i in range(num_files):
        part = f"{stem}_part_{i + 1}.txt"
        with open(part, "w", encoding="utf-8") as f:
            f.writelines(lines[i * size:(i + 1) * size])
        parts.append(part)
    return parts


def parse_args(argv: Optional[Sequence[str]] = None, defaults: Optional[GrabOptions] = None):
    """
    The command line. Every option defaults to the GrabOptions defaults (edit the class to change them),
    and every switch can be set either way: --skip-endgames / --no-skip-endgames, and so on.
    Returns (options, args).
    """
    defaults = defaults or GrabOptions()
    on_off = argparse.BooleanOptionalAction
    parser = argparse.ArgumentParser(description="Sample positions from Lichess game archives into a FEN text file.")
    parser.add_argument("--games", nargs="+", default=["lichess-games"], help="folders or .pgn / .pgn.zst files")
    parser.add_argument("--out", default="ssl_positions.txt")
    parser.add_argument("--max-per-game", type=int, default=defaults.max_positions_per_game,
                        help="most positions taken from one game")
    parser.add_argument("--min-ply-gap", type=int, default=defaults.min_ply_gap,
                        help="fewest plies between two positions taken from the same game")
    parser.add_argument("--skip-endgames", action=on_off, default=defaults.skip_endgames,
                        help="skip positions with --endgame-max-pieces pieces or fewer")
    parser.add_argument("--endgame-max-pieces", type=int, default=defaults.endgame_max_pieces)
    parser.add_argument("--skip-start", action=on_off, default=defaults.skip_start_position,
                        help="skip starting positions")
    parser.add_argument("--skip-final", action=on_off, default=defaults.skip_final_position,
                        help="skip final positions")
    parser.add_argument("--skip-games-with-errors", action=on_off, default=defaults.skip_games_with_errors)
    parser.add_argument("--max-games-per-file", type=int, default=defaults.max_games_per_file)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--split", type=int, default=1, help="also split the output into this many files")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args(argv)
    options = GrabOptions(max_positions_per_game=args.max_per_game, min_ply_gap=args.min_ply_gap,
                          skip_endgames=args.skip_endgames, endgame_max_pieces=args.endgame_max_pieces,
                          skip_start_position=args.skip_start, skip_final_position=args.skip_final,
                          skip_games_with_errors=args.skip_games_with_errors,
                          max_games_per_file=args.max_games_per_file, seed=args.seed)
    return options, args


def main() -> None:
    options, args = parse_args()
    print("Options:", asdict(options))
    grab_positions(args.games, args.out, options, workers=args.workers, split_into=args.split)


if __name__ == "__main__":
    main()
