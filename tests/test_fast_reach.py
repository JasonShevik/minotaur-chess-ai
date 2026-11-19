"""
Tests for fast_reach.c, the compiled search behind piece_reach.py.

Run with plain Python (no pytest needed):

    python tests/test_fast_reach.py

If the library is not built yet and a C compiler is on the PATH, it is built first (build_fast_reach.py);
without a compiler the tests are skipped, and say so.

The Python searches in piece_reach.py are the reference (tests/test_reach_labeler.py checks those against
every sequence played out). These tests check that the compiled search gives exactly the same arrays:
  * on positions with castling both ways, Chess960 castling (rook and king on, next to and across each other's
    squares, castling through and into attacked squares), en passant (plain, pinned on a rank, a file and a
    diagonal, taking the checking pawn), promotions, checks, double checks, pins, mate and stalemate;
  * on positions from Chess960 games played at random, steered towards castling, en passant and promotion;
  * directly against the play-out too, so that an error the two searches shared would still show;
  * at 4 and 5 plies, where positions are remembered and met again, with a table too small to hold them and
    with none at all;
  * on whole samples through label_threats, which searches a line of play last position first and reuses
    each answer in the search before it;
  * that positions the compiled search refuses still get the Python search's answer.
"""
import ctypes
import os
import random
import sys
import traceback
from typing import Callable, List

import chess
import numpy as np

import repo_root  # noqa: F401  (first: it puts the repo's root folder, where the code under test is, on sys.path)
import build_fast_reach

if not os.path.exists(build_fast_reach.LIBRARY) and build_fast_reach.compiler():     # before piece_reach looks for it
    try:
        build_fast_reach.build()
    except RuntimeError as error:
        print(error)

import piece_reach as pr
import test_reach_labeler as slow       # FENS, SMALL, played_out, threats_played_out

COMPILED: bool = pr.use_compiled(True)  # (test_reach_labeler switches it off for its own tests)


def xfen(board: chess.Board) -> str:
    return board.fen(en_passant="xfen")


# Positions the compiled search must take itself (the tests check that it did not hand them to Python)
EDGE: List[str] = [
    # en passant
    "8/8/8/K2pP2r/8/8/8/7k w - d6 0 1",                         # taking would bare the king to a rook along the rank
    "8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1",                        # the same for black
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "4k3/8/8/8/3pP3/8/8/4K2B b - e3 0 1",                       # by a pawn pinned on the diagonal it takes along
    "4k3/4r3/8/3pP3/8/8/8/4K3 w - d6 0 1",                      # by a pawn pinned on its file
    "8/8/8/2k5/3pP3/8/8/4K3 b - e3 0 1",                        # the pawn that just moved gives check
    "2k5/8/8/3pP3/8/8/8/2K1R3 w - d6 0 1", "4k3/8/8/1pP5/8/8/8/R3K3 w Q b6 0 1",
    "3k4/8/8/2pP4/8/8/8/3KQ3 w - c6 0 1", "k7/8/8/8/1Pp5/8/8/K7 b - b3 0 1",
    "8/8/8/8/k1pP3R/8/8/3K4 b - d3 0 1", "8/8/8/1k6/2pP4/8/8/3K1B2 b - d3 0 1",
    "8/1k6/8/3pP3/8/5B2/8/3K4 w - d6 0 1",
    # promotion
    "8/2P5/8/8/8/8/5k2/K7 w - - 0 1", "1n1r4/2P5/8/8/8/8/5k2/K7 w - - 0 1", "8/P7/8/8/8/8/7p/k6K w - - 0 1",
    "8/8/8/8/8/8/p7/k6K b - - 0 1",
    # castling
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "r3k2r/8/8/8/8/8/6n1/R3K2R w KQkq - 0 1",                   # through attacked squares
    "r3k2r/8/8/8/1b6/8/8/R3K2R w KQkq - 0 1",                   # out of check
    "2r1k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "1r2k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",    # c1 attacked; only b1 attacked
    "1k4r1/8/8/8/8/8/8/R3K2R w KQ - 0 1", "1k3r2/8/8/8/8/8/8/R3K2R w KQ - 0 1", "1k1r4/8/8/8/8/8/8/R3K2R w KQ - 0 1",
    "4k3/8/8/8/8/8/1p6/R3K2R w KQ - 0 1", "4k3/8/8/8/8/8/6p1/R3K2R w KQ - 0 1",
    "r3k2r/8/8/8/8/8/8/1R2K2R b Kkq - 0 1", "4k3/8/8/8/8/8/8/R3K2R w - - 0 1",
    # Chess960 castling
    "bnrqkrnb/pppppppp/8/8/8/8/PPPPPPPP/BNRQKRNB w KQkq - 0 1", "2kr3r/8/8/8/8/8/8/1RK4R w Kk - 0 1",
    "8/8/8/8/8/8/8/RK5k w A - 0 1",                             # the king one step from the rook
    "8/8/8/8/8/8/8/qRK4k w B - 0 1",                            # the castling rook is all that shields the king
    "4k3/8/8/8/8/8/8/1RK3rR w K - 0 1",                         # a piece between the king and its square
    "2k5/8/8/8/8/8/8/5RKR w KQ - 0 1",                          # the king on g1 already, a rook on f1
    "5k2/8/8/8/8/8/8/R4K1R w KQ - 0 1", "5k2/8/8/8/8/8/8/1R3KR1 w KQ - 0 1",
    "7k/8/8/8/8/8/8/RK6 w A - 0 1", "k7/8/8/8/8/8/8/6KR w H - 0 1", "k7/8/8/8/8/8/8/5KR1 w G - 0 1",
    "k7/8/8/8/8/8/8/1KR5 w C - 0 1", "7k/8/8/8/8/8/8/2RK4 w C - 0 1",
    # checks, pins, the end of the game
    "4k3/8/8/8/8/5n2/8/4K3 w - - 0 1", "4k3/8/8/8/8/8/3n1n2/4K3 w - - 0 1",
    "4k3/4r3/8/8/7b/8/8/4K3 w - - 0 1",                         # double check
    "4k3/4r3/8/8/8/8/4R3/4K3 w - - 0 1", "4k3/8/8/8/7b/8/5N2/4K3 w - - 0 1", "4k3/8/8/8/7b/8/5P2/4K3 w - - 0 1",
    "4k3/8/8/b7/8/8/3P4/4K3 w - - 0 1", "4k3/8/8/8/8/2b5/3P4/4K3 w - - 0 1",
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "R6k/8/7K/8/8/8/8/8 b - - 0 1",
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    # busy positions
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
]
# Few moves a side: 4 and 5 plies are quick enough for the Python search
SPARSE: List[str] = [
    "8/8/8/4k3/8/3K4/3P4/8 w - - 0 1", "8/5k2/8/8/3K4/8/2R5/8 w - - 0 1", "8/8/1k6/8/8/3N4/3K4/8 b - - 0 1",
    "8/2k5/8/3p4/3P4/8/2K5/8 w - - 0 1", "8/6k1/8/8/8/8/1p4K1/8 b - - 0 1", "8/8/3k4/8/2p5/8/1P1K4/8 w - - 0 1",
    "8/8/8/8/8/k7/p7/K7 w - - 0 1", "4k3/8/8/8/8/8/3P4/R3K3 w Q - 0 1", "8/1P4k1/8/8/8/8/6K1/8 w - - 0 1",
]
# Positions that cannot arise in a game: the compiled search hands them to the Python one
REFUSED: List[str] = [
    "5k2/8/8/8/8/8/8/5RKR w KQ - 0 1",                          # the side that just moved left its king in check
    "4k3/8/8/8/4pP2/8/8/4K2q b - f3 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - e6 0 1",                           # an en passant square with no pawn to take
    "P3k3/8/8/8/8/8/8/4K3 w - - 0 1",                           # a pawn on the last rank
]


def random_games(seed: int, games: int) -> List[str]:
    """Positions from Chess960 games played at random, leaning towards the special moves."""
    rng = random.Random(seed)
    out = []
    for _ in range(games):
        board = chess.Board.from_chess960_pos(rng.randrange(960))
        board.chess960 = True
        for _ in range(rng.randrange(1, 200)):
            moves = list(board.legal_moves)
            if not moves:
                break
            special = [m for m in moves if board.is_castling(m) or board.is_en_passant(m) or m.promotion or board.is_capture(m)]
            board.push(rng.choice(special) if special and rng.random() < 0.5 else rng.choice(moves))
            if rng.random() < 0.03:
                out.append(xfen(board))
        out.append(xfen(board))
    return out


GAMES: List[str] = random_games(seed=11, games=40)


def same(got: np.ndarray, expected: np.ndarray, what: str) -> None:
    if got.shape != expected.shape or not np.array_equal(got, expected):
        where = tuple(np.argwhere(got != expected)[0]) if got.shape == expected.shape else None
        raise AssertionError(f"{what}: differs" + (f" first at {where}: compiled {int(got[where])}, Python {int(expected[where])}" if where else ""))


def searches() -> int:
    return pr.search_stats()["searches"]


# ##### ##### ##### ##### #####
#   Tests


def test_threats_equal_the_python_search():
    before = searches()
    for fen in slow.FENS + slow.SMALL + EDGE + GAMES:
        expected = pr.threat_counts(fen, 2, use_c=False)
        for plies in (0, 1, 2):
            same(pr.threat_counts(fen, plies), expected[:plies + 1], f"threats {plies} plies {fen}")
    assert searches() - before == 3 * len(slow.FENS + slow.SMALL + EDGE + GAMES), "every one of these is the compiled search's own"
    for fen in slow.FENS[:3] + EDGE[::12] + GAMES[::20]:
        same(pr.threat_counts(fen, 3), pr.threat_counts(fen, 3, use_c=False), f"threats 3 plies {fen}")


def test_reach_equals_the_python_search():
    before = searches()
    for fen in slow.FENS + slow.SMALL + EDGE + GAMES:
        expected = pr.reach_counts(fen, 3, use_c=False)
        for depth in (1, 2, 3):
            same(pr.reach_counts(fen, depth), expected[:depth], f"reach {depth} plies {fen}")
    assert searches() - before == 3 * len(slow.FENS + slow.SMALL + EDGE + GAMES)
    for fen in slow.FENS[:3] + EDGE[::12] + GAMES[::20]:
        same(pr.reach_counts(fen, 4), pr.reach_counts(fen, 4, use_c=False), f"reach 4 plies {fen}")


def test_both_equal_every_sequence_played_out():
    for fen in slow.FENS + EDGE[::5]:
        same(pr.threat_counts(fen, 2), slow.threats_played_out(fen, 2), f"threats {fen}")
        same(pr.reach_counts(fen, 3), slow.played_out(fen, 3), f"reach {fen}")
    for fen in slow.SMALL + SPARSE[:4]:
        same(pr.threat_counts(fen, 4), slow.threats_played_out(fen, 4), f"threats 4 plies {fen}")
        same(pr.reach_counts(fen, 5), slow.played_out(fen, 5), f"reach 5 plies {fen}")


def test_deep_searches_remember_positions_and_stay_right():
    pr.forget()
    before = pr.search_stats()
    for fen in SPARSE + slow.SMALL:
        same(pr.threat_counts(fen, 4), pr.threat_counts(fen, 4, use_c=False), f"threats 4 plies {fen}")
    for fen in SPARSE[:5]:
        same(pr.threat_counts(fen, 5), pr.threat_counts(fen, 5, use_c=False), f"threats 5 plies {fen}")
        same(pr.reach_counts(fen, 5), pr.reach_counts(fen, 5, use_c=False), f"reach 5 plies {fen}")
    after = pr.search_stats()
    assert after["table_stores"] > before["table_stores"] and after["table_hits"] > before["table_hits"], \
        "from the fourth ply on, positions are kept and found again"
    # The same search again, with everything of the first still remembered, and after forgetting it
    first = pr.threat_counts(SPARSE[1], 4)
    same(pr.threat_counts(SPARSE[1], 4), first, "asked twice")
    pr.forget()
    same(pr.threat_counts(SPARSE[1], 4), first, "asked after forgetting")


def test_a_table_too_small_or_absent_changes_nothing():
    lib = pr._LIB
    for megabytes, slots_log2 in ((1, 10), (0, 10)):
        context = lib.fr_new(megabytes, slots_log2)
        try:
            for fen in SPARSE[:6]:
                for plies in (4, 5):
                    out = np.empty((plies + 1, 64, 64), dtype=np.uint64)
                    assert lib.fr_threat_counts(context, fen.encode(), plies, out.ctypes.data) == 0
                    same(out, pr.threat_counts(fen, plies, use_c=False), f"{megabytes} MB table, {plies} plies, {fen}")
            raw = (ctypes.c_uint64 * 8)()
            lib.fr_stats(context, raw)
            if megabytes:
                assert raw[2] > 0 and raw[4] > 0, "the small table filled and was emptied along the way"
            else:
                assert raw[1] == 0 and raw[2] == 0, "no table: nothing kept, nothing found"
        finally:
            lib.fr_free(context)


def test_a_line_of_play_is_searched_backwards_and_reuses_answers():
    rng = random.Random(5)
    for start, plies in ((slow.FENS[0], 2), (slow.FENS[3], 2), (slow.FENS[4], 2), (SPARSE[1], 3), (SPARSE[5], 4)):
        board = chess.Board(start, chess960=True)
        line = [start]
        for _ in range(2):
            board.push(rng.choice(list(board.legal_moves)))
            line.append(xfen(board))
        pr.forget()
        before = pr.search_stats()["answers_reused"]
        rows = pr.label_threats((line, plies))
        assert [row[0] for row in rows] == [pr.position_key(fen) for fen in line], "rows in the order the FENs were given"
        assert all(row[1] == plies for row in rows)
        for fen, row in zip(line, rows):
            same(pr.unpack_counts(row[2]), pr.threat_counts(fen, plies, use_c=False), f"line from {start}, {plies} plies, {fen}")
        assert pr.search_stats()["answers_reused"] - before == 2, "each position found the next one's answer"
    rows = pr.label_positions(([slow.FENS[0], slow.FENS[1]], 2))
    same(pr.unpack_counts(rows[1][2]), pr.reach_counts(slow.FENS[1], 2, use_c=False), "reach labels")
    assert pr.label_threats(([], 2)) == []


def test_refused_positions_get_the_python_answer():
    before = searches()
    for fen in REFUSED:
        same(pr.threat_counts(fen, 1), pr.threat_counts(fen, 1, use_c=False), f"threats {fen}")
        same(pr.reach_counts(fen, 2), pr.reach_counts(fen, 2, use_c=False), f"reach {fen}")
    assert searches() == before, "none of these was searched by the compiled search"
    for bad in ("", "not a fen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN w KQkq - 0 1"):
        try:
            pr.threat_counts(bad, 1)
            raise AssertionError(f"{bad!r} should be rejected")
        except ValueError:              # python-chess's own complaint about the FEN
            pass
    for call in (lambda: pr.threat_counts(slow.FENS[0], -1), lambda: pr.reach_counts(slow.FENS[0], 0)):
        try:
            call()
            raise AssertionError("should be rejected")
        except ValueError:
            pass


def test_switching_the_compiled_search_off_and_on():
    before = searches()
    assert pr.use_compiled(False) is False
    pr.threat_counts(slow.FENS[0], 1)
    assert searches() == before, "off: the Python search"
    assert pr.use_compiled(True) is True
    pr.threat_counts(slow.FENS[0], 1)
    pr.threat_counts(slow.FENS[0], 1, use_c=False)
    assert searches() == before + 1


# ##### ##### ##### ##### #####
#   Runner

def main() -> int:
    if not COMPILED:
        print(f"SKIPPED: the compiled search is not available: {pr.C_UNAVAILABLE}")
        return 0
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
