"""
Tests for the reach and legal-threat labels: piece_reach.py (the searches and the storage format) and
reach_labeler.py (the program).

Run with plain Python (no pytest needed):

    python tests/test_reach_labeler.py

They check:
  * the counts against every move sequence played out one by one with python-chess (castling in both directions,
    Chess960 castling, en passant, promotions, captures, positions where the game can end early);
  * that every move order is counted, and each piece's counts add up to the number of sequences;
  * the legal threats against the same play-out, asking python-chess at every position what each side could
    play, and against cases worked out by hand (a pin, a check, a line opened by a pawn, castling, en passant);
  * that the storage format gives the counts back exactly, large ones included;
  * that the labeler walks the positions a training run will see, in its order, with its rollout moves;
  * that it stores what the search gives, resumes where it stopped, deepens labels, and runs with workers;
  * that the two kinds of label keep their own tables and their own places in the walk.
"""
import os
import random
import sys
import tempfile
import threading
import traceback
from typing import Callable, List

import chess
import numpy as np

import repo_root  # noqa: F401  (first: it puts the repo's root folder, where the code under test is, on sys.path)
import labyrinth_data as ld
import piece_reach as pr
import reach_labeler as rl

FENS = [
    "1q1rkr2/pp3pnp/2pn2pQ/3p4/3Pb3/2P2NP1/PP2P2P/3RKRNB b - - 1 15",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",       # en passant on offer
    "2k5/ppp3pp/6N1/8/8/2b1P1P1/2NrK1BP/6B1 w - - 1 26",                   # in check
    "8/7p/kR2p1p1/5p2/N3RPP1/1P6/1K5P/8 b - - 1 37",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",                                # castling, both sides and both ways
    "1rk3r1/8/8/8/8/8/8/1RK3R1 w KQkq - 0 1",                              # Chess960 castling
    "4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1",                                   # promotions
    "1n2k3/P7/8/8/8/8/1p6/N3K3 b - - 0 1",                                 # promotions by capture
    "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",                                   # mate in one: some sequences end early
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",                                   # en passant, few pieces
]
FENS = [ld.normalize_fen(f) for f in FENS]
SMALL = ["k7/8/8/8/8/8/8/K7 w - - 0 1", "2k5/8/8/8/8/2n5/8/1NK5 w - - 0 1", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
         "r3k3/8/8/8/8/8/8/4K2R w Kq - 0 1"]


def played_out(fen: str, depth: int) -> np.ndarray:
    """The counts by playing every sequence one by one: slow, and written with python-chess's own questions
    about a move (is it castling, is it en passant) rather than the search's bit tricks."""
    board = chess.Board(fen, chess960=True)
    out = np.zeros((depth, 64, 65), dtype=np.uint64)
    homes = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]

    def walk(at: dict, d: int) -> None:                 # at: square now -> square at the start
        if d:
            now = {home: sq for sq, home in at.items()}
            for home in homes:
                out[d - 1, home, now.get(home, pr.CAPTURED)] += 1
        if d == depth:
            return
        for move in list(board.legal_moves):
            after = dict(at)
            if board.is_castling(move):
                rank = chess.square_rank(move.from_square)
                kingside = board.is_kingside_castling(move)
                king, rook = after.pop(move.from_square), after.pop(move.to_square)
                after[chess.square(6 if kingside else 2, rank)] = king
                after[chess.square(5 if kingside else 3, rank)] = rook
            else:
                if board.is_en_passant(move):
                    del after[chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))]
                after.pop(move.to_square, None)
                after[move.to_square] = after.pop(move.from_square)
            board.push(move)
            walk(after, d + 1)
            board.pop()

    walk({sq: sq for sq in homes}, 0)
    return out


def threats_played_out(fen: str, plies: int) -> np.ndarray:
    """The legal threats by playing every sequence one by one and asking python-chess, at each position, for
    the legal moves of the side to move and then of the other side with the turn handed over."""
    board = chess.Board(fen, chess960=True)
    out = np.zeros((plies + 1, 64, 64), dtype=np.uint64)
    homes = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]

    def could_move_to(b: chess.Board, at: dict) -> set:     # {(the piece's square at the start, a square it could move to)}
        pairs = set()
        for move in b.legal_moves:
            if b.is_castling(move):
                rank = chess.square_rank(move.from_square)
                kingside = b.is_kingside_castling(move)
                king_to, rook_to = chess.square(6 if kingside else 2, rank), chess.square(5 if kingside else 3, rank)
                if king_to != move.from_square:
                    pairs.add((at[move.from_square], king_to))
                if rook_to != move.to_square:
                    pairs.add((at[move.to_square], rook_to))
            else:
                pairs.add((at[move.from_square], move.to_square))
        return pairs

    def walk(at: dict, k: int) -> None:                 # at: square now -> square at the start
        other = board.copy(stack=False)
        other.turn = not board.turn
        other.ep_square = None
        for home, square in could_move_to(board, at) | could_move_to(other, at):
            out[k, home, square] += 1
        if k == plies:
            return
        for move in list(board.legal_moves):
            after = dict(at)
            if board.is_castling(move):
                rank = chess.square_rank(move.from_square)
                kingside = board.is_kingside_castling(move)
                king, rook = after.pop(move.from_square), after.pop(move.to_square)
                after[chess.square(6 if kingside else 2, rank)] = king
                after[chess.square(5 if kingside else 3, rank)] = rook
            else:
                if board.is_en_passant(move):
                    del after[chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))]
                after.pop(move.to_square, None)
                after[move.to_square] = after.pop(move.from_square)
            board.push(move)
            walk(after, k + 1)
            board.pop()

    walk({sq: sq for sq in homes}, 0)
    return out


def perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for move in list(board.legal_moves):
        board.push(move)
        total += perft(board, depth - 1)
        board.pop()
    return total


def make_cache(folder: str, n_starts: int = 12) -> str:
    rnd = random.Random(0)
    fens = []
    for i in range(n_starts):
        board = chess.Board() if i == 0 else chess.Board.from_chess960_pos(rnd.randrange(960))
        for _ in range(6):
            fens.append(board.fen())
            for _ in range(rnd.randint(1, 5)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rnd.choice(moves))
    path = os.path.join(folder, "fens.txt")
    with open(path, "w") as f:
        f.write("\n".join(fens))
    cache = os.path.join(folder, "cache")
    ld.build_position_cache_from_file(path, cache, val_fraction=0.2, test_fraction=0.0, verbose=False)
    return cache


def quiet(_message: str) -> None:
    pass


# ##### ##### ##### ##### #####
#   Tests


def test_counts_match_every_sequence_played_out():
    for fen in FENS:
        got, expected = pr.reach_counts(fen, 3), played_out(fen, 3)
        assert got.dtype == np.uint64 and got.shape == (3, 64, 65)
        assert np.array_equal(got, expected), fen
        assert np.array_equal(pr.reach_counts(fen, 2), expected[:2]), "a shallower search is the deeper one's start"
    for fen in SMALL:                                   # deep enough for move orders to meet (from the third ply)
        assert np.array_equal(pr.reach_counts(fen, 5), played_out(fen, 5)), fen
    assert np.array_equal(pr.reach_counts(FENS[3], 4), played_out(FENS[3], 4))


def test_every_move_order_counts_and_rows_add_up_to_the_number_of_sequences():
    # two kings: after 3 plies the white king is back on a1 by 3 routes, times black's 3 moves in between
    counts = pr.reach_counts("k7/8/8/8/8/8/8/K7 w - - 0 1", 3)
    assert counts[0, chess.A1, chess.A2] == 1 and counts[0, chess.A8, chess.A8] == 3
    assert counts[1, chess.A8, chess.A7] == 3 and counts[1, chess.A1, chess.B2] == 3
    assert counts[2, chess.A1, chess.A1] == 9
    assert not counts[:, chess.C3].any(), "an empty square has no row"
    # the same arrangement by two move orders (Nb1-c3 and Ng1-f3 either way round, the black king shuffling)
    # is two sequences: counted twice
    fen = "4k3/8/8/8/8/8/8/1N2K1N1 w - - 0 1"
    counts = pr.reach_counts(fen, 3)
    board = chess.Board(fen)
    both = 0
    for first, second in (("b1c3", "g1f3"), ("g1f3", "b1c3")):
        board.push_uci(first)
        for reply in list(board.legal_moves):
            board.push(reply)
            both += chess.Move.from_uci(second) in board.legal_moves
            board.pop()
        board.pop()
    assert both == 10, "five king moves, two orders"
    assert counts[2, chess.B1, chess.C3] >= both and np.array_equal(counts, played_out(fen, 3))
    for fen in FENS:
        board, counts = chess.Board(fen, chess960=True), pr.reach_counts(fen, 3)
        for d in range(3):
            sequences = perft(board, d + 1)
            rows = counts[d, [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]].sum(1)
            assert (rows == sequences).all(), (fen, d + 1)
    # a finished game has no sequences at all; a mate in one is not among the 2-ply sequences
    assert not pr.reach_counts("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 3).any()
    mating = pr.reach_counts(FENS[8], 2)
    assert mating[0, chess.A1, chess.A8] == 1 and mating[1, chess.A1, chess.A8] == 0
    # a capture: the rook on a2 can be taken in one way, by the rook on a1
    counts = pr.reach_counts("4k3/8/8/8/8/8/r7/R3K3 w - - 0 1", 1)
    assert counts[0, chess.A2, pr.CAPTURED] == 1 and counts[0, chess.A1, chess.A2] == 1
    try:
        pr.reach_counts(FENS[0], 0)
        raise AssertionError("depth 0 should be rejected")
    except ValueError:
        pass


def test_threats_match_every_sequence_played_out():
    for fen in FENS:
        got, expected = pr.threat_counts(fen, 2), threats_played_out(fen, 2)
        assert got.dtype == np.uint64 and got.shape == (3, 64, 64)
        assert np.array_equal(got, expected), fen
        assert np.array_equal(pr.threat_counts(fen, 1), expected[:2]), "a shallower search is the deeper one's start"
        assert np.array_equal(pr.threat_counts(fen, 0), expected[:1])
    for fen in SMALL:                                   # deep enough for move orders to meet (from the third ply)
        assert np.array_equal(pr.threat_counts(fen, 3), threats_played_out(fen, 3)), fen
    for fen in SMALL[:3]:
        assert np.array_equal(pr.threat_counts(fen, 4), threats_played_out(fen, 4)), fen
    assert np.array_equal(pr.threat_counts(FENS[3], 3), threats_played_out(FENS[3], 3))
    # the side to move's threats in the position itself are its legal moves; every count is at most the
    # number of sequences
    for fen in FENS:
        board, counts = chess.Board(fen, chess960=True), pr.threat_counts(fen, 2)
        mine = [sq for sq in chess.SQUARES if board.color_at(sq) == board.turn]
        plain = {(m.from_square, m.to_square) for m in board.legal_moves if not board.is_castling(m)}
        assert plain <= {(p, s) for p in mine for s in range(64) if counts[0, p, s]}
        assert counts[0].max() <= 1
        for k in (1, 2):
            assert counts[k].max() <= perft(board, k), (fen, k)


def test_threats_are_the_legal_moves_each_side_would_have():
    # the opening position, worked out by hand
    t = pr.threat_counts(chess.STARTING_FEN, 1)
    assert t[0].sum() == 40, "20 moves for white, 20 for black were it black's turn"
    assert t[0, chess.G1, chess.F3] == 1 and t[0, chess.G8, chess.F6] == 1
    assert t[0, chess.E2, chess.E3] == 1 and t[0, chess.E2, chess.E4] == 1
    assert t[0, chess.E2, chess.D3] == 0, "a pawn does not threaten an empty diagonal square"
    assert not t[0, chess.F1].any(), "the bishop is shut in"
    # after one ply: the bishop is let out by e3 or e4, and by g3 or g4
    for square in (chess.E2, chess.D3, chess.C4, chess.B5, chess.A6, chess.G2, chess.H3):
        assert t[1, chess.F1, square] == 2, chess.square_name(square)
    assert t[1, chess.B8, chess.C6] == 20, "black's knight move stays legal after every white move"
    assert t[1, chess.G1, chess.F3] == 17, "20 moves, less the knight's own two and f3 (which blocks it)"
    assert t[1, chess.G1, chess.G1] == 2, "having moved, the knight could come back"
    assert t[1, chess.G1, chess.E2] == 2, "after e3 or e4"
    # a pinned piece threatens nothing; the piece pinning it threatens it, and no further
    for turn in "wb":
        t = pr.threat_counts(f"4k3/4r3/8/8/8/8/4B3/4K3 {turn} - - 0 1", 0)
        assert not t[0, chess.E2].any()
        assert t[0, chess.E7, chess.E2] == 1 and t[0, chess.E7, chess.E1] == 0
    # a check is a threat on the king's square (the turn handed over, the rook could take the king); the king
    # cannot stay on the rook's file
    t = pr.threat_counts("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1", 0)
    assert t[0, chess.E2, chess.E8] == 1
    assert t[0, chess.E8, chess.D8] == 1 and t[0, chess.E8, chess.E7] == 0
    # a captured piece threatens nothing in the sequences where it is gone: white has 6 moves, one takes the rook
    t = pr.threat_counts("4k3/8/8/8/8/8/r7/R3K3 w - - 0 1", 1)
    assert t[1, chess.A2, chess.A3] == 5
    assert t[1, chess.A1, chess.A3] == 1, "only after taking on a2 can the white rook go up the file"
    assert t[0, chess.A1, chess.A2] == 1 and t[0, chess.A2, chess.A1] == 1
    # en passant: the side whose turn it is may take; the other side has no such move
    t = pr.threat_counts("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", 0)
    assert t[0, chess.E5, chess.D6] == 1 and t[0, chess.E5, chess.E6] == 1
    assert t[0, chess.D5, chess.D4] == 1 and t[0, chess.D5].sum() == 1
    # castling: the king and the rook threaten the squares they would castle to; a rook that can also get
    # there by an ordinary move is counted once
    t = pr.threat_counts("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 0)
    assert t[0, chess.E1, chess.G1] == 1 and t[0, chess.E1, chess.C1] == 1
    assert t[0, chess.H1, chess.F1] == 1 and t[0, chess.A1, chess.D1] == 1
    assert t[0, chess.E8, chess.G8] == 1, "black too, were it black's turn"
    # Chess960: the king already stands on its castling square, so only the rook gains a square by castling
    t = pr.threat_counts("1rk3r1/8/8/8/8/8/8/1RK3R1 w KQkq - 0 1", 0)
    assert t[0, chess.B1, chess.D1] == 1 and t[0, chess.C1, chess.C1] == 0
    # promotion: the square is threatened once, not once for each piece the pawn could become
    t = pr.threat_counts("4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1", 0)
    assert t[0, chess.A7, chess.A8] == 1 and t[0, chess.A2, chess.A1] == 1
    assert not pr.threat_counts("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 1)[:, chess.C3].any(), "an empty square has no row"
    try:
        pr.threat_counts(FENS[0], -1)
        raise AssertionError("a negative number of plies should be rejected")
    except ValueError:
        pass


def test_storage_gives_the_counts_back():
    for fen in FENS + ["7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"]:
        counts = pr.reach_counts(fen, 3)
        blob = pr.pack_counts(counts)
        assert np.array_equal(pr.unpack_counts(blob), counts) and pr.packed_depth(blob) == 3
        assert len(blob) < 4000, "a few kB at most"
    big = pr.reach_counts(FENS[0], 2)
    big[1, chess.E8, pr.CAPTURED] = 2 ** 40 + 7             # more than four bytes hold
    assert np.array_equal(pr.unpack_counts(pr.pack_counts(big)), big)
    rows = pr.label_positions(([FENS[0], FENS[1]], 2))
    assert [r[0] for r in rows] == [pr.position_key(FENS[0]), pr.position_key(FENS[1])] and rows[0][1] == 2
    assert np.array_equal(pr.unpack_counts(rows[1][2]), pr.reach_counts(FENS[1], 2))
    assert pr.position_key(FENS[0]) == ld.position_key(FENS[0])
    # threats: the same format, and the blob says which kind it holds
    for fen in FENS + ["7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "k7/8/8/8/8/8/8/K7 w - - 0 1"]:
        counts = pr.threat_counts(fen, 2)
        blob = pr.pack_counts(counts)
        back = pr.unpack_counts(blob)
        assert back.shape == (3, 64, 64) and np.array_equal(back, counts) and pr.packed_depth(blob) == 2
        assert len(blob) < 4000
    assert pr.unpack_counts(pr.pack_counts(pr.threat_counts(FENS[0], 0))).shape == (1, 64, 64)
    rows = pr.label_threats(([FENS[0], FENS[1]], 1))
    assert [r[0] for r in rows] == [pr.position_key(FENS[0]), pr.position_key(FENS[1])] and rows[0][1] == 1
    assert np.array_equal(pr.unpack_counts(rows[1][2]), pr.threat_counts(FENS[1], 1))
    for bad in (np.zeros((2, 64, 63)), np.zeros((2, 63, 64)), np.zeros((64, 65))):
        try:
            pr.pack_counts(bad)
            raise AssertionError("a wrong shape should be rejected")
        except ValueError:
            pass


def test_the_walk_is_the_training_runs_order_and_rollout():
    # the rollout positions are the ones build_sample plays with the same random numbers
    for i, fen in enumerate(FENS):
        for seed in range(3):
            sample = ld.build_sample(fen, random.Random(seed * 100 + i), ld.MaskConfig(), rollout_steps=3)
            mine = rl.rollout_fens(fen, random.Random(seed * 100 + i), 3)
            assert mine == [step.next_fen for step in sample.rollout], fen
            assert rl.rollout_fens(fen, random.Random(seed * 100 + i), 2) == mine[:2]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = make_cache(tmp)
        for seed, epoch in ((0, 0), (3, 1)):
            loader = ld.JEPALoader(cache, "train", batch_size=4, num_workers=0, seed=seed)
            loader.set_epoch(epoch)
            loader.set_rollout_steps(2)
            order = list(loader.sampler)
            walked = list(rl.walk_positions(cache, "train", seed, epoch, rollout_plies=2))
            assert len(walked) == len(order) == len(loader.dataset)
            for i, fens in zip(order, walked):
                sample = loader.dataset[i]
                assert fens == [sample.fen] + [step.next_fen for step in sample.rollout]
            assert list(rl.walk_positions(cache, "train", seed, epoch, 2, start=5)) == walked[5:]
            assert [f[0] for f in rl.walk_positions(cache, "train", seed, epoch, 0)] == [f[0] for f in walked]


def test_labeler_stores_resumes_and_deepens():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache, db = make_cache(tmp), os.path.join(tmp, "reach.db")
        walked = list(rl.walk_positions(cache, "train", 0, 0, rollout_plies=2))
        common = dict(db_path=db, cache_dir=cache, order_seed=0, rollout_plies=2, chunk=3, log=quiet, kind="reach")

        first = rl.label(depth=2, workers=0, max_positions=7, **common)
        assert first["done"] == first["handled"] == 7
        labels = rl.ReachLabels(db)
        expected = {pr.position_key(f) for fens in walked[:7] for f in fens}
        assert first["searched"] == len(expected), "one search per different position"
        for fens in walked[:7]:
            for fen in fens:
                assert np.array_equal(labels.get(fen), pr.reach_counts(fen, 2)), fen
        assert labels.get(walked[7][0]) is None or pr.position_key(walked[7][0]) in expected
        labels.close()

        # the next start carries on from position 7, and what is labelled already is not searched again
        second = rl.label(depth=2, workers=0, max_positions=5, **common)
        assert second["done"] == 12 and second["handled"] == 5
        again = {pr.position_key(f) for fens in walked[7:12] for f in fens}
        assert second["searched"] == len(again - expected) and second["skipped"] >= len(again & expected)

        # a deeper walk starts from the beginning and replaces the shallower labels; a shallower one never does
        deeper = rl.label(depth=3, workers=0, max_positions=4, **common)
        assert deeper["done"] == 4
        labels = rl.ReachLabels(db)
        assert labels.get(walked[0][0]).shape[0] == 3 and labels.get(walked[11][0]).shape[0] == 2
        assert np.array_equal(labels.get(walked[0][0]), pr.reach_counts(walked[0][0], 3))
        labels.close()
        conn = rl.open_db(db)
        rl.write_labels(conn, pr.label_positions(([walked[0][0]], 1)))
        conn.commit()
        assert conn.execute("SELECT depth FROM reach_counts WHERE position = ?",
                            (pr.position_key(walked[0][0]),)).fetchone()[0] == 3
        progress = dict(conn.execute("SELECT walk, done FROM reach_progress"))
        assert progress == {rl.walk_name("train", 0, 0, 2, 2): 12, rl.walk_name("train", 0, 0, 2, 3): 4}
        conn.close()
        back = rl.label(depth=2, workers=0, max_positions=2, **common)       # depth 2 again: on from 12
        assert back["done"] == 14

        # stopping: a set stop event ends the run after the job in hand; a stop file does the same
        stop = threading.Event()
        stop.set()
        stopped = rl.label(depth=2, workers=0, stop=stop, **common)
        assert 14 < stopped["done"] < len(walked)
        open(db + ".stop", "w").close()
        by_file = rl.label(depth=2, workers=0, **common)
        assert stopped["done"] < by_file["done"] < len(walked) and not os.path.exists(db + ".stop")

        # to the end of the split, with worker processes; nothing is left to do afterwards
        rest = rl.label(depth=2, workers=2, **common)
        assert rest["done"] == len(walked)
        labels = rl.ReachLabels(db)
        for fens in walked:
            assert all(labels.get(fen) is not None for fen in fens)
        assert np.array_equal(labels.get(walked[-1][-1])[:2], pr.reach_counts(walked[-1][-1], 2))
        labels.close()
        nothing = rl.label(depth=2, workers=0, **common)
        assert nothing["handled"] == nothing["searched"] == 0 and nothing["done"] == len(walked)
        messages: List[str] = []
        rl.check_db(db, messages.append)
        assert any("labelled to 2 plies" in m for m in messages) and any("walk train seed 0" in m for m in messages)


def test_labeler_keeps_threats_and_reach_apart():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache, db = make_cache(tmp), os.path.join(tmp, "labels.db")
        walked = list(rl.walk_positions(cache, "train", 0, 0, rollout_plies=2))
        common = dict(db_path=db, cache_dir=cache, order_seed=0, rollout_plies=2, chunk=3, log=quiet)
        assert rl.LABEL == "threats" and rl.DEPTH["threats"] == 2, "what a plain start makes"

        reach = rl.label(depth=2, workers=0, max_positions=5, kind="reach", **common)
        first = rl.label(depth=1, workers=0, max_positions=7, kind="threats", **common)
        assert reach["done"] == 5 and first["done"] == first["handled"] == 7, "each kind has its own place in the walk"
        assert first["skipped"] == 0, "a reach label does not stand in for a threat label"
        labels = rl.ReachLabels(db)
        for fens in walked[:7]:
            for fen in fens:
                assert np.array_equal(labels.get(fen, "threats"), pr.threat_counts(fen, 1)), fen
        for fen in walked[4]:
            assert np.array_equal(labels.get(fen), pr.reach_counts(fen, 2)), "the reach labels are as they were"
        seen = {pr.position_key(f) for fens in walked[:7] for f in fens}
        assert labels.get(walked[7][0], "threats") is None or pr.position_key(walked[7][0]) in seen
        labels.close()

        # 0 plies is a depth too (the position's own legal moves, both sides): its walk starts at the beginning,
        # finds everything labelled deeper already, and searches nothing
        shallow = rl.label(depth=0, workers=0, max_positions=3, kind="threats", **common)
        assert shallow["done"] == 3 and shallow["searched"] == 0 and shallow["skipped"] >= 3

        # deeper: from the first sample again, replacing the shallower labels; then the default depth with workers
        deeper = rl.label(depth=2, workers=0, max_positions=4, kind="threats", **common)
        assert deeper["done"] == 4
        rest = rl.label(workers=2, kind="threats", **common)
        assert rest["done"] == len(walked), "no depth given: DEPTH['threats'], carrying on from sample 4"
        labels = rl.ReachLabels(db)
        for fens in walked:
            for fen in fens:
                assert labels.get(fen, "threats").shape == (3, 64, 64)
        assert np.array_equal(labels.get(walked[-1][-1], "threats"), pr.threat_counts(walked[-1][-1], 2))
        assert labels.get(walked[-1][0]) is None or pr.position_key(walked[-1][0]) in {
            pr.position_key(f) for fens in walked[:5] for f in fens}, "no reach label was made for it"
        labels.close()

        conn = rl.open_db(db)
        progress = dict(conn.execute("SELECT walk, done FROM reach_progress"))
        assert progress == {"train seed 0 epoch 0 rollout 2 depth 2": 5, "train seed 0 epoch 0 rollout 2 threats 0": 3,
                            "train seed 0 epoch 0 rollout 2 threats 1": 7, "train seed 0 epoch 0 rollout 2 threats 2": len(walked)}
        assert rl.walk_name("train", 0, 0, 2, 3) == "train seed 0 epoch 0 rollout 2 depth 3", "the name old databases hold"
        conn.close()
        messages: List[str] = []
        rl.check_db(db, messages.append)
        assert any("threats:" in m and "labelled to 2 plies" in m for m in messages)
        assert any("reach:" in m and "labelled to 2 plies" in m for m in messages)
        assert any("samples done" in m for m in messages) and any("after 2 plies" in m for m in messages)
        for bad in (dict(kind="guards"), dict(kind="reach", depth=0), dict(kind="threats", depth=-1)):
            try:
                rl.label(workers=0, **common, **bad)
                raise AssertionError("should be rejected")
            except ValueError:
                pass


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
