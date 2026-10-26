"""
Tests for lichess_position_grabber.py and the self-supervised positions table in db_tools.py.

Run with plain Python (no pytest needed):

    python test_lichess_position_grabber.py

They write small PGN archives of random games (standard and Chess960, some played down to bare
endgames) to a temporary folder and check that:
  * every sampled position really occurs in its game, and respects the options: at most
    max_positions_per_game per game, at least min_ply_gap plies apart, no starting or final position,
    and no position with endgame_max_pieces pieces or fewer unless skip_endgames is off;
  * sampling is uniform over the allowed spacings and reproducible from the seed, whatever the number
    of workers;
  * .pgn.zst archives read the same as .pgn (when zstandard is installed);
  * db_tools loads the output into its own table, skipping duplicates, and labyrinth_data builds its
    cache from that table.
"""
import collections
import io
import os
import random
import sqlite3
import sys
import tempfile
import traceback
from typing import Callable, Dict, List

import chess
import chess.pgn

import db_tools
import labyrinth_data as ld
import lichess_position_grabber as lpg


# ##### ##### ##### ##### #####
#   Helpers


def random_game(rnd: random.Random, chess960: bool, max_plies: int, capture_bias: float) -> chess.pgn.Game:
    """A random game; capture_bias is the chance of preferring a capture, which drives games into endgames."""
    board = chess.Board.from_chess960_pos(rnd.randrange(960)) if chess960 else chess.Board()
    game = chess.pgn.Game.from_board(board)
    node = game
    for _ in range(max_plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        captures = [m for m in moves if board.is_capture(m)]
        move = rnd.choice(captures) if captures and rnd.random() < capture_bias else rnd.choice(moves)
        board.push(move)
        node = node.add_variation(move)
    return game


def write_archive(path: str, n_games: int, seed: int) -> List[chess.pgn.Game]:
    rnd = random.Random(seed)
    games = [random_game(rnd, chess960=i % 2 == 0, max_plies=rnd.randint(0, 220), capture_bias=0.85)
             for i in range(n_games)]
    with open(path, "w", encoding="utf-8") as f:
        for g in games:
            print(g, file=f, end="\n\n")
    return games


def game_fens(game: chess.pgn.Game) -> List[str]:
    """Every position of the game by ply (0 = the start), in the grabber's FEN convention."""
    board = game.board()
    out = [board.fen(en_passant="xfen")]
    for move in game.mainline_moves():
        board.push(move)
        out.append(board.fen(en_passant="xfen"))
    return out


def reread(path: str) -> List[chess.pgn.Game]:
    games = []
    with open(path, encoding="utf-8") as f:
        while (g := chess.pgn.read_game(f)) is not None:
            games.append(g)
    return games


# ##### ##### ##### ##### #####
#   Tests


def test_spaced_sample():
    rnd = random.Random(0)
    for _ in range(2000):
        first, last = rnd.randint(0, 5), rnd.randint(-3, 120)
        k_max, gap = rnd.randint(0, 15), rnd.randint(1, 9)
        s = lpg.spaced_sample(rnd, first, last, k_max, gap)
        assert s == sorted(s) and all(first <= v <= last for v in s)
        assert all(b - a >= gap for a, b in zip(s, s[1:]))
        span = last - first + 1
        expected = 0 if span <= 0 or k_max <= 0 else min(k_max, (span - 1) // gap + 1)
        assert len(s) == expected, (first, last, k_max, gap, s)
    # uniform over every allowed set: 2 values from 1..8 at least 3 apart (15 sets)
    counts = collections.Counter(tuple(lpg.spaced_sample(rnd, 1, 8, 2, 3)) for _ in range(30000))
    assert len(counts) == 15 and max(counts.values()) < 1.15 * min(counts.values()), counts


def test_sampled_positions_follow_the_options():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "games.pgn")
        write_archive(path, 120, seed=1)
        games = reread(path)
        reached_endgame = sum(1 for g in games if any(len(chess.Board(f).piece_map()) <= 7 for f in game_fens(g)))
        assert reached_endgame > 10, "the test archive needs games that reach endgames"
        for opts in (lpg.GrabOptions(), lpg.GrabOptions(max_positions_per_game=3, min_ply_gap=12),
                     lpg.GrabOptions(skip_endgames=False, skip_start_position=False, skip_final_position=False,
                                     min_ply_gap=1, max_positions_per_game=1000)):
            rnd = random.Random(0)
            saw_small_endgame = False
            for g in games:
                fens = game_fens(g)
                picked = lpg.positions_from_game(g, opts, rnd)
                plies = [fens.index(f) for f in picked]      # each picked position occurs in the game
                assert len(picked) <= opts.max_positions_per_game
                assert all(b - a >= opts.min_ply_gap for a, b in zip(plies, plies[1:])), plies
                if opts.skip_start_position:
                    assert 0 not in plies
                if opts.skip_final_position and len(fens) > 1:
                    assert len(fens) - 1 not in plies
                pieces = [len(chess.Board(f, chess960=True).piece_map()) for f in picked]
                if opts.skip_endgames:
                    assert all(p > opts.endgame_max_pieces for p in pieces)
                else:
                    saw_small_endgame |= any(p <= 7 for p in pieces)
            if not opts.skip_endgames:
                assert saw_small_endgame, "with skip_endgames off, endgame positions must be kept"
        # a long game gives the full ten positions by default
        long_games = [g for g in games if len(game_fens(g)) > 80
                      and all(len(chess.Board(f, chess960=True).piece_map()) > 7 for f in game_fens(g)[:80])]
        assert long_games and len(lpg.positions_from_game(long_games[0], lpg.GrabOptions(), random.Random(1))) == 10


def test_grab_positions_writes_a_reproducible_file():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        folder = os.path.join(tmp, "lichess-games")
        os.makedirs(folder)
        for i in range(3):
            write_archive(os.path.join(folder, f"month_{i}.pgn"), 40, seed=10 + i)
        with open(os.path.join(folder, "notes.txt"), "w") as f:
            f.write("not an archive")
        quiet = lambda msg: None
        out1, out2 = os.path.join(tmp, "a.txt"), os.path.join(tmp, "b.txt")
        stats = lpg.grab_positions(folder, out1, lpg.GrabOptions(seed=3), workers=1, log=quiet)
        lpg.grab_positions(folder, out2, lpg.GrabOptions(seed=3), workers=2, log=quiet)
        lines = open(out1).read().split()
        assert stats["archives"] == 3 and stats["games"] == 120 and stats["positions"] == len(open(out1).read().splitlines())
        assert open(out1).read() == open(out2).read(), "the number of workers must not change the output"
        lpg.grab_positions(folder, out2, lpg.GrabOptions(seed=4), log=quiet)
        assert open(out1).read() != open(out2).read(), "another seed must sample other positions"
        # every line is a valid FEN, and max_games_per_file stops each archive early
        for line in open(out1).read().splitlines():
            chess.Board(line, chess960=True)
        few = lpg.grab_positions(folder, out2, lpg.GrabOptions(max_games_per_file=5), log=quiet)
        assert few["games"] == 15
        parts = lpg.grab_positions(folder, os.path.join(tmp, "c.txt"), lpg.GrabOptions(seed=3), split_into=4, log=quiet)["parts"]
        assert len(parts) == 4 and "".join(open(p).read() for p in parts) == open(out1).read()


def test_compressed_archives_read_like_plain_ones():
    try:
        import zstandard
    except ImportError:
        print("        (zstandard not installed; skipped)")
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        plain = os.path.join(tmp, "games.pgn")
        write_archive(plain, 30, seed=5)
        with open(plain, "rb") as f:
            data = zstandard.ZstdCompressor().compress(f.read())
        packed = os.path.join(tmp, "games.pgn.zst")
        with open(packed, "wb") as f:
            f.write(data)
        a, counts_a = lpg.grab_from_file(plain, lpg.GrabOptions())
        b, counts_b = lpg.grab_from_file(packed, lpg.GrabOptions())
        assert a == b and counts_a == counts_b and len(a) > 0


def test_command_line_respects_the_class_defaults():
    """Running the file without arguments uses GrabOptions' defaults, whatever they are set to."""
    options, _ = lpg.parse_args([])
    assert options == lpg.GrabOptions()
    edited = lpg.GrabOptions(skip_endgames=False, skip_start_position=False, max_positions_per_game=3,
                             max_games_per_file=100)
    options, _ = lpg.parse_args([], defaults=edited)
    assert options == edited, "edited defaults must not be overridden by the command line"
    # and every switch works in both directions
    options, _ = lpg.parse_args(["--no-skip-endgames", "--no-skip-start", "--skip-final", "--min-ply-gap", "8"])
    assert (options.skip_endgames, options.skip_start_position, options.skip_final_position, options.min_ply_gap) == (False, False, True, 8)
    options, _ = lpg.parse_args(["--skip-endgames", "--skip-start"], defaults=edited)
    assert options.skip_endgames and options.skip_start_position


def test_database_table_and_training_cache():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        folder = os.path.join(tmp, "games")
        os.makedirs(folder)
        write_archive(os.path.join(folder, "g.pgn"), 60, seed=7)
        out = os.path.join(tmp, "ssl.txt")
        lpg.grab_positions(folder, out, lpg.GrabOptions(), log=lambda m: None)
        fens = open(out).read().splitlines()

        db = os.path.join(tmp, "minotaur_test")          # db_tools names databases without the .db
        db_tools.create_db(db)                             # the engine-labeled table exists alongside
        added = db_tools.add_ssl_positions(db, [out])
        assert added == len(set(fens))
        assert db_tools.add_ssl_positions(db, [out]) == 0, "loading the same file again adds nothing"
        with sqlite3.connect(db + ".db") as conn:
            rows = conn.execute(f'SELECT fen, source FROM "{db_tools.SSL_TABLE}"').fetchall()
            labeled = conn.execute('SELECT COUNT(*) FROM "960_position_data"').fetchone()[0]
        assert {r[0] for r in rows} == set(fens) and all(r[1] == "ssl.txt" for r in rows)
        assert labeled == 0, "the self-supervised positions must not go into the labeled table"

        cache = os.path.join(tmp, "cache")
        meta = ld.build_position_cache(db + ".db", cache, val_fraction=0.1, test_fraction=0.1, verbose=False,
                                       table=db_tools.SSL_TABLE)
        assert sum(meta["positions"].values()) == len({ld.position_key(ld.normalize_fen(f)) for f in fens})
        assert sum(meta["labeled"].values()) == 0
        try:
            ld.build_position_cache(db + ".db", cache, table="no_such_table", verbose=False)
        except ValueError:
            pass
        else:
            raise AssertionError("a missing table must be reported")


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
