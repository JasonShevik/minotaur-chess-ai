"""
Tests for analyze_move_distribution.py.

Run with plain Python (no pytest needed):

    python test_analyze_move_distribution.py

They check the move flags (capture, check, castle, promotion, the hanging heuristic) on hand-made
positions, the game phases, the two sampling schemes' shares, and a small end-to-end run writing the CSV.
"""
import csv
import os
import sqlite3
import sys
import tempfile
import traceback
from typing import Callable, List

import chess

import analyze_move_distribution as amd


def flags_of(fen: str, uci: str, chess960: bool = False):
    board = chess.Board(fen, chess960=chess960)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, uci
    letter, flags = amd.move_properties(board, move)
    assert board.fen() == chess.Board(fen, chess960=chess960).fen(), "the board is left as it was"
    return letter, flags


def test_move_flags_and_the_hanging_heuristic():
    queen = "4k3/8/8/2p5/8/8/8/3QK3 w - - 0 1"
    assert flags_of(queen, "d1d4") == ("Q", {"capture": False, "check": False, "castle": False,
                                             "promotion": False, "hangs": True})     # the c5 pawn takes it
    assert not flags_of(queen, "d1a4")[1]["hangs"]
    assert flags_of(queen, "d1d8")[1]["check"] and flags_of(queen, "d1d8")[1]["hangs"]   # the king takes it
    # taking a defended queen with a rook loses the rook but wins more
    letter, flags = flags_of("3k4/8/2p5/3q4/8/8/8/3RK3 w - - 0 1", "d1d5")
    assert letter == "R" and flags["capture"] and not flags["hangs"]
    # a defended piece attacked only by something more expensive is safe
    assert not flags_of("4k3/8/8/7q/8/8/8/4K1N1 w - - 0 1", "g1e2")[1]["hangs"]
    # castling (chess960: the king takes its own rook) is neither a capture nor a hang
    letter, flags = flags_of("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1", "e1h1", chess960=True)
    assert letter == "K" and flags["castle"] and not flags["capture"] and not flags["hangs"]
    assert flags_of("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6")[1]["capture"]          # en passant
    letter, flags = flags_of("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q")
    assert letter == "P" and flags["promotion"] and flags["check"] and not flags["hangs"]


def test_phases_and_ply_buckets():
    assert amd.game_phase(chess.Board()) == "opening" and amd.material_phase(chess.Board()) == 24
    assert amd.game_phase(chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 20")) == "middlegame"
    assert amd.game_phase(chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 30")) == "endgame"
    assert [amd.ply_bucket(p) for p in (0, 9, 10, 99, 100, 250)] == \
        ["ply<10", "ply<10", "ply<20", "ply<100", "ply>=100", "ply>=100"]


def test_uniform_and_piece_first_shares():
    board = chess.Board("4k3/8/8/8/8/8/P7/3QK3 w - - 0 1")
    counts = {t: sum(board.piece_at(m.from_square).symbol().upper() == t for m in board.legal_moves) for t in "PQK"}
    stats = amd.MoveStats()
    stats.add("games", ["all"], board, chess.Move.from_uci("a2a4"))
    rows = {r["scheme"]: r for r in stats.rows()}
    total = sum(counts.values())
    for t in "PQK":
        assert abs(rows["uniform"][f"share_{t}"] - counts[t] / total) < 1e-9
        assert abs(rows["piece"][f"share_{t}"] - 1 / 3) < 1e-9      # three movable pieces, a third each
    assert rows["played"]["share_P"] == 1.0 and rows["played"]["tv_vs_reference"] == 0.0
    assert abs(rows["uniform"]["tv_vs_reference"] - (1 - counts["P"] / total)) < 1e-9
    assert rows["uniform"]["qr_over_half_of_legal"] == 1.0
    stats.add("games", ["all"], chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))     # stalemate: skipped
    assert stats.positions[("games", "all")] == 1


def test_end_to_end_writes_the_csv():
    pgn = ('[Event "a"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *\n\n'
           '[Event "b"]\n[Variant "Chess960"]\n[SetUp "1"]\n[FEN "bqnbrkrn/pppppppp/8/8/8/8/PPPPPPPP/BQNBRKRN w GEge - 0 1"]'
           '\n\n1. O-O g6 2. g3 *\n')
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        games = os.path.join(tmp, "games")
        os.makedirs(games)
        with open(os.path.join(games, "a.pgn"), "w") as f:
            f.write(pgn)
        db = os.path.join(tmp, "labeled.db")
        con = sqlite3.connect(db)
        con.execute('CREATE TABLE "960_position_data" (fen TEXT PRIMARY KEY, is_analyzed INTEGER, best_move TEXT, '
                    'inconsistent INTEGER)')
        con.executemany('INSERT INTO "960_position_data" VALUES (?, 1, ?, 0)',
                        [(chess.STARTING_FEN, "e2e4"), ("4k3/8/8/8/8/8/8/R3K3 w - - 0 30", "a1a8")])
        con.commit()
        con.close()
        stats = amd.analyze(games, db, games_per_file=10, keep=1.0, workers=0, log=lambda m: None)
        out = os.path.join(tmp, "out.csv")
        amd.write_csv(stats.rows(), out)
        with open(out) as f:
            rows = list(csv.DictReader(f))
    assert stats.positions[("games", "all")] == 9 and stats.positions[("labeled", "all")] == 2
    assert stats.positions[("games", "opening")] == 9 and ("games", "ply<10") in stats.positions
    assert {r["scheme"] for r in rows if r["source"] == "games"} == {"uniform", "piece", "played"}
    assert {r["scheme"] for r in rows if r["source"] == "labeled"} == {"uniform", "piece", "engine"}
    played = next(r for r in rows if r["source"] == "games" and r["bucket"] == "all" and r["scheme"] == "played")
    assert abs(float(played["castle"]) - 1 / 9) < 1e-3                 # 1. O-O in the Chess960 game
    for r in rows:
        assert abs(sum(float(r[f"share_{k}"]) for k in amd.PIECES) - 1) < 1e-3
        assert ("sample bias" in r["notes"]) == (r["source"] == "labeled")


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
