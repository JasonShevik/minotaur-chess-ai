"""
Correctness tests for the position perturbations in chess_graph.py.

Run with plain Python (no pytest needed):

    python test_perturbations.py

Every perturbation must produce a position that is different from the input and plausible:
something that could arise in a legal game of Chess960. The tests below check the rules the
perturbations are supposed to obey, the mirror perturbation end to end (including castling
edges), and the graph features built from perturbed positions.

If minotaur_data.db is present next to this file (or MINOTAUR_DB points at one), a sample of
real positions is swept as well; otherwise the sweep runs on a small built-in set.
"""
import collections
import os
import random
import sqlite3
import sys
import traceback
from typing import Callable, List

import chess
import torch

import chess_graph as cg


# ##### ##### ##### ##### #####
#   Fixtures

BUILT_IN_FENS: List[str] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "1q1rkr2/pp3pnp/2pn2pQ/3p4/3Pb3/2P2NP1/PP2P2P/3RKRNB b KQkq - 1 15",
    "bbnrkr2/p2pppp1/1p3n1q/7p/2PN4/1P1N3P/P3PPP1/BB1RKR1Q b KQkq - 0 7",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "2k5/ppp3pp/6N1/8/8/2b1P1P1/2NrK1BP/6B1 w - - 1 26",
    "4kr1r/pppppppp/8/8/8/8/PPPPPPPP/4KRR1 w Kk - 0 1",
    "8/7p/kR2p1p1/5p2/N3RPP1/1P6/1K5P/8 b - - 1 37",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
]

NUM_TYPES = 10
TURN_CHANGE = 9


def _db_path() -> str:
    env = os.environ.get("MINOTAUR_DB")
    if env and os.path.isfile(env):
        return env
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minotaur_data.db")
    return local if os.path.isfile(local) else ""


def sample_positions(n: int, seed: int = 0) -> List[str]:
    """Real positions from the database when available, else the built-in list."""
    path = _db_path()
    if not path:
        return BUILT_IN_FENS
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = [r[0] for r in con.execute('SELECT fen FROM "960_position_data" LIMIT 200000')]
    con.close()
    return random.Random(seed).sample(rows, min(n, len(rows)))


def check_position_properties(original: str, perturbed: str, turn_may_change: bool = False) -> List[str]:
    """Return the list of rules a perturbed position breaks (empty when it is fine)."""
    broken: List[str] = []
    if perturbed == original:
        broken.append("unchanged")
    if not cg.position_is_plausible(perturbed):
        broken.append("fails plausibility gate")
    board = chess.Board(perturbed, chess960=True)
    if len(board.piece_map()) > 32:
        broken.append("more than 32 pieces")
    if board.ep_square is not None:
        if board.piece_at(board.ep_square) is not None:
            broken.append("piece on the en passant square")
        want_rank = 5 if board.turn == chess.WHITE else 2
        if chess.square_rank(board.ep_square) != want_rank:
            broken.append("en passant square on the wrong rank for the side to move")
    if any(c in "ABCDEFGHabcdefgh" for c in perturbed.split()[2]):
        broken.append("X-FEN file letters in castling field")
    if not turn_may_change and perturbed.split()[1] != original.split()[1]:
        broken.append("side to move changed")
    if perturbed.split()[4:6] != original.split()[4:6]:
        broken.append("move counters changed")
    return broken


# ##### ##### ##### ##### #####
#   Tests


def test_gate_accepts_real_positions():
    fens = sample_positions(500)
    rejected = [f for f in fens if not cg.position_is_plausible(f)]
    assert not rejected, f"gate rejected {len(rejected)} real positions, e.g. {rejected[:3]}"


def test_every_type_produces_plausible_different_positions():
    fens = sample_positions(400)
    failures = collections.Counter()
    for fen in fens:
        for t in range(NUM_TYPES):
            for magnitude in ((1, 2, 3) if t in (0, 1) else (1,)):
                out, used = cg.perturb_position(fen, perturb_type=t, magnitude=magnitude, return_type=True)
                # A fallback may land on the turn-change type, which changes the side to move by design
                for rule in check_position_properties(fen, out, turn_may_change=(used == TURN_CHANGE)):
                    failures[(t, rule)] += 1
    assert not failures, dict(failures)


def test_fallbacks_only_happen_for_legitimate_reasons():
    fens = sample_positions(300, seed=1)
    for fen in fens:
        board = chess.Board(fen, chess960=True)
        for t in (3, 6, 7, TURN_CHANGE):
            _, used = cg.perturb_position(fen, perturb_type=t, rng=random.Random(1), return_type=True)
            if used == t:
                continue
            if t == TURN_CHANGE:
                assert board.is_check(), f"turn change fell back although the side to move is not in check: {fen}"
                continue
            if t == 3:
                addable = [
                    (pt, c) for pt in (1, 2, 3, 4, 5) for c in (True, False)
                    if cg._added_piece_is_plausible(board, pt, c)
                ]
                assert not addable, f"addition fell back although {addable} could be added: {fen}"
            elif t == 6:
                flippable = []
                for sq in chess.SQUARES:
                    p = board.piece_at(sq)
                    if p is None or p.piece_type == chess.KING:
                        continue
                    probe = board.copy(stack=False)
                    probe.set_piece_at(sq, chess.Piece(p.piece_type, not p.color))
                    if cg._material_is_plausible(probe):
                        flippable.append(sq)
                assert not flippable, f"color change fell back although a flip was possible: {fen}"
            else:
                rights = cg._castling_rights_in_fen(fen)
                addable = [k for k in "KQkq" if k not in rights and cg.is_castling_right_plausible(board, k, fen)]
                assert not rights and not addable, f"castling fell back with rights={rights} addable={addable}: {fen}"


def test_unchanged_output_is_impossible_and_no_fallback_raises():
    bare = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    for i in range(30):
        out, used = cg.perturb_position(bare, perturb_type=7, rng=random.Random(i), return_type=True)
        assert out != bare
        assert used != 7, "castling perturbation cannot apply to bare kings, so a fallback must be reported"
    try:
        cg.perturb_position(bare, perturb_type=7, fallback_to_other_types=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when no perturbation is possible and fallback is off")


def test_seeded_rng_is_deterministic():
    fen = BUILT_IN_FENS[1]
    outs = {cg.perturb_position(fen, rng=random.Random(42)) for _ in range(5)}
    assert len(outs) == 1


def test_deletion_never_removes_a_king():
    fen = BUILT_IN_FENS[0]
    for i in range(300):
        out = cg.perturb_position(fen, perturb_type=2, rng=random.Random(i))
        board = chess.Board(out, chess960=True)
        assert board.king(chess.WHITE) is not None and board.king(chess.BLACK) is not None


def test_full_board_cannot_gain_pieces():
    full = BUILT_IN_FENS[0]
    for i in range(60):
        _, used = cg.perturb_position(full, perturb_type=3, rng=random.Random(i), return_type=True)
        assert used != 3, "a 32-piece board must not receive an added piece"
        _, used = cg.perturb_position(full, perturb_type=6, rng=random.Random(i), return_type=True)
        assert used != 6, "flipping a color on a 16 v 16 board would give one side 17 pieces"


def test_material_rules():
    ok = cg._material_is_plausible
    assert ok(chess.Board(BUILT_IN_FENS[0], chess960=True))
    assert not ok(chess.Board("4k3/pppppppp/p7/8/8/8/8/4K3 w - - 0 1", chess960=True)), "9 pawns"
    assert not ok(chess.Board("4k3/8/8/8/8/8/PPPPPPPP/Q2QK3 w - - 0 1", chess960=True)), "2 queens with 8 pawns"
    assert ok(chess.Board("4k3/8/8/8/8/8/PPPPPPP1/Q2QK3 w - - 0 1", chess960=True)), "2 queens with 7 pawns"
    assert not ok(chess.Board("4k3/8/8/8/8/8/PPPPPPPP/B1B1K3 w - - 0 1", chess960=True)), "two light-squared bishops with 8 pawns"
    assert ok(chess.Board("4k3/8/8/8/8/8/PPPPPPPP/B1N1KB2 w - - 0 1", chess960=True)), "bishop pair on opposite colors"
    assert ok(chess.Board("4k3/8/8/8/8/8/PPPPPPP1/N1N1KN2 w - - 0 1", chess960=True)), "3 knights with 7 pawns"
    assert not ok(chess.Board("4k3/8/8/8/8/8/8/8 w - - 0 1", chess960=True)), "missing king"


def test_en_passant_addition_rules():
    # White to move, black d-pawn beside white e5-pawn, d7 empty -> only d6 may be added
    fen = "rnbqkbn1/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBN1 w Qq - 0 3"
    eps = {cg.perturb_position(fen, perturb_type=3, rng=random.Random(i), fallback_to_other_types=False).split()[3] for i in range(200)}
    assert eps - {"-"} == {"d6"}, eps
    # Same, but the square the pawn would have come from (d7) is occupied -> no en passant may be added
    fen = "rnb1kbn1/pppqpppp/8/3pP3/8/8/PPPP1PPP/RNBQKBN1 w Qq - 0 3"
    eps = {cg.perturb_position(fen, perturb_type=3, rng=random.Random(i), fallback_to_other_types=False).split()[3] for i in range(200)}
    assert eps == {"-"}, eps
    # Black to move: the en passant square must be on the 3rd rank behind a white pawn on the 4th
    fen = "rnbqkbn1/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBN1 b Qq - 0 3"
    eps = {cg.perturb_position(fen, perturb_type=3, rng=random.Random(i), fallback_to_other_types=False).split()[3] for i in range(200)}
    assert eps - {"-"} == {"d3"}, eps


def test_castling_symmetry_rules():
    # Only white has K, with rooks on f1 and g1. Black rooks on f8 and h8: f is shared -> k may be added
    fen = "4kr1r/pppppppp/8/8/8/8/PPPPPPPP/4KRR1 w K - 0 1"
    assert cg.is_castling_right_plausible(chess.Board(fen, chess960=True), "k", fen)
    # Black rook only on h8: no shared file -> k may not be added
    fen = "4k2r/pppppppp/8/8/8/8/PPPPPPPP/4KRR1 w K - 0 1"
    assert not cg.is_castling_right_plausible(chess.Board(fen, chess960=True), "k", fen)
    # Decoy rook: both castle kingside on f. Delete black's f8 rook, leaving h8 -> black loses k, white keeps K
    before = "4kr1r/pppppppp/8/8/8/8/PPPPPPPP/4KR2 w Kk - 0 1"
    after = "4k2r/pppppppp/8/8/8/8/PPPPPPPP/4KR2 w Kk - 0 1"
    assert cg.fix_castling_in_fen(after, before, random.Random(0)).split()[2] == "K"
    # From KQkq the castling perturbation may only remove single rights (nothing plausible to add)
    fen = "1r2k1r1/pppppppp/8/8/8/8/PPPPPPPP/1R2K1R1 w KQkq - 0 1"
    outcomes = {cg.perturb_position(fen, perturb_type=7, rng=random.Random(i), fallback_to_other_types=False).split()[2] for i in range(200)}
    assert outcomes <= {"Qkq", "Kkq", "KQq", "KQk"}, outcomes
    # After any perturbation, both-sided rights always share a rook file
    for f in sample_positions(200, seed=2):
        for t in range(NUM_TYPES):
            out = cg.perturb_position(f, perturb_type=t, rng=random.Random(t))
            b = chess.Board(out, chess960=True)
            rights = cg._castling_rights_in_fen(out)
            for kingside in (True, False):
                wk, bk = ("K", "k") if kingside else ("Q", "q")
                if wk in rights and bk in rights:
                    assert cg._back_rank_rook_files(b, chess.WHITE, kingside) & cg._back_rank_rook_files(b, chess.BLACK, kingside), out


def test_king_on_corner_file_never_has_castling():
    fen = "k6r/pppppppp/8/8/8/8/PPPPPPPP/K6R w - - 0 1"
    assert not cg.is_castling_right_plausible(chess.Board(fen, chess960=True), "K", fen)
    assert cg.fix_castling_in_fen("k6r/pppppppp/8/8/8/8/PPPPPPPP/K6R w Kk - 0 1").split()[2] == "-"
    assert cg.get_castling_edges(cg.fen_to_vector("k6r/pppppppp/8/8/8/8/PPPPPPPP/K6R w Kk - 0 1")) == set()
    # king on b1 is fine
    fen = "rk5r/pppppppp/8/8/8/8/PPPPPPPP/RK5R w - - 0 1"
    assert cg.is_castling_right_plausible(chess.Board(fen, chess960=True), "K", fen)


def test_graph_time_rook_choice_is_random_when_ambiguous():
    vec = cg.fen_to_vector("4k3/pppppppp/8/8/8/8/PPPPPPPP/4KRR1 w K - 0 1")
    picks = collections.Counter(min(s for s, d in cg.get_castling_edges(vec) if d == 5) for _ in range(400))
    assert set(picks) == {5, 6}, picks          # f1 and g1 both get chosen
    assert min(picks.values()) > 100, picks     # and neither is starved


def test_align_castling_rooks_follows_shared_file():
    # White rooks f1 and g1 with K; black rook f8 with k. python-chess would pick g1 (outermost);
    # the shared file is f, so alignment must move white's castling rook to f1.
    fen = "4kr2/pppppppp/8/8/8/8/PPPPPPPP/4KRR1 w Kk - 0 1"
    board = chess.Board(fen, chess960=True)
    assert chess.G1 in chess.scan_forward(board.castling_rights)
    cg._align_castling_rooks(board, fen, random.Random(0))
    assert set(chess.scan_forward(board.castling_rights)) == {chess.F1, chess.F8}


def test_mirror_perturbation():
    mirror_sq = lambda i: (i // 8) * 8 + (7 - i % 8)
    dest_swap = {6: 2, 2: 6, 5: 3, 3: 5, 62: 58, 58: 62, 61: 59, 59: 61}
    swap = {"K": "Q", "Q": "K", "k": "q", "q": "k"}
    for fen in sample_positions(200, seed=3):
        m = cg.mirror_fen_files(fen)
        assert cg.mirror_fen_files(m) == fen, "double mirror must be the identity"
        assert cg.position_is_plausible(m)
        assert {swap[c] for c in cg._castling_rights_in_fen(fen)} == cg._castling_rights_in_fen(m)
        ep0, ep1 = fen.split()[3], m.split()[3]
        if ep0 != "-":
            assert ep1 == "abcdefgh"[7 - "abcdefgh".index(ep0[0])] + ep0[1]
        e0, x0 = cg.create_filled_chess_graphs(fen)
        e1, x1 = cg.create_filled_chess_graphs(m)
        perm = torch.tensor([mirror_sq(i) for i in range(64)])
        assert torch.equal(x0[perm], x1), "node features must be mirrored"
        b = chess.Board(fen, chess960=True)
        ambiguous = any(len(cg._back_rank_rook_files(b, c, ks)) > 1 for c in (chess.WHITE, chess.BLACK) for ks in (True, False))
        if not ambiguous:
            expect = {(mirror_sq(s), dest_swap[d]) for s, d in e0[-1].t().tolist()}
            got = set(map(tuple, e1[-1].t().tolist()))
            assert expect == got, f"castling edges of the mirror must be the mirrored king/rook castling to the other side: {fen}"
    assert cg.mirror_castling_rook_squares((5, 3, 61, 59)) == (4, 2, 60, 58)
    assert cg.mirror_castling_rook_squares((7, -1, 63, -1)) == (-1, 0, -1, 56)


def test_node_features_match_the_board_exactly():
    """
    Ground truth for create_filled_chess_graphs against python-chess. Orientation: the side to
    move is at the bottom. White to move: vector index i is square i (a1 = 0). Black to move:
    the board is reflected vertically, so index i is file i % 8, rank 7 - i // 8.
    """
    fens = sample_positions(400, seed=7) + [
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",    # white to move, ep, both castle
        "1brqbk1r/pp3ppp/2n2n2/2pp4/3Pp2P/2P3P1/PPN1PPK1/NBRQB2R b kq d3 0 8",  # black to move, ep, black castles
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",          # black to move, both castle
    ]
    for fen in fens:
        for t in range(-1, NUM_TYPES):   # -1 = the unperturbed position itself
            out = fen if t < 0 else cg.perturb_position(fen, perturb_type=t, rng=random.Random(t))
            board = chess.Board(out, chess960=True)
            _, x = cg.create_filled_chess_graphs(out)
            for i in range(64):
                file, row = i % 8, i // 8
                sq = chess.square(file, row if board.turn == chess.WHITE else 7 - row)
                piece = board.piece_at(sq)
                want = [0] * 8
                if piece is not None:
                    want[0] = int(piece.color != board.turn)
                    want[piece.piece_type] = 1
                if board.ep_square == sq:
                    want[7] = 1
                assert x[i].int().tolist() == want, (
                    f"{out}\n  node {i} = {chess.square_name(sq)}: got {x[i].int().tolist()}, want {want}"
                )


def test_turn_change():
    """Type 9 hands the move to the other side: pieces identical, en passant cleared, castling kept."""
    for fen in sample_positions(300, seed=8):
        board = chess.Board(fen, chess960=True)
        out, used = cg.perturb_position(fen, perturb_type=TURN_CHANGE, rng=random.Random(0), return_type=True)
        if board.is_check():
            assert used != TURN_CHANGE, f"cannot give the move away while in check: {fen}"
            assert cg.position_is_plausible(out)
            continue
        assert used == TURN_CHANGE, f"turn change should have been possible: {fen}"
        new = chess.Board(out, chess960=True)
        assert new.piece_map() == board.piece_map(), "no piece may move"
        assert new.turn != board.turn
        assert out.split()[3] == "-", "an en passant square cannot survive a change of turn"
        assert cg._castling_rights_in_fen(out) == cg._castling_rights_in_fen(fen)
        assert out.split()[4:6] == fen.split()[4:6]
    # A position with the side to move in check (here: checkmated) can never have its turn flipped
    mated = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    for i in range(20):
        out, used = cg.perturb_position(mated, perturb_type=TURN_CHANGE, rng=random.Random(i), return_type=True)
        assert used != TURN_CHANGE
        assert chess.Board(out, chess960=True).is_valid()
    try:
        cg.perturb_position(mated, perturb_type=TURN_CHANGE, fallback_to_other_types=False)
    except ValueError:
        pass
    else:
        raise AssertionError("turn change on an in-check position must raise when fallback is off")
    # An en passant square is dropped, and only the turn and en passant fields change
    fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
    out = cg.perturb_position(fen, perturb_type=TURN_CHANGE, fallback_to_other_types=False)
    assert out == "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3", out


def test_node_features_never_mix_piece_and_en_passant():
    for fen in sample_positions(150, seed=5):
        for t in range(NUM_TYPES):
            out = cg.perturb_position(fen, perturb_type=t, rng=random.Random(t))
            _, x = cg.create_filled_chess_graphs(out)
            assert x.shape == (64, 8)
            assert not ((x[:, 7] == 1) & (x[:, 1:7].sum(1) > 0)).any(), out


def test_xfen_letters_are_understood():
    # Shredder-style: white castles with the d1 rook (queenside of the e1 king), black with the f8 rook
    fen = "1q1rkr2/pp3pnp/2pn2pQ/3p4/3Pb3/2P2NP1/PP2P2P/3RKRNB b Df - 1 15"
    assert cg._normalize_castling_field(fen).split()[2] == "Qk"
    vec = cg.fen_to_vector(fen)
    kings = sorted((i, v) for i, v in enumerate(vec) if abs(int(v)) == 6 and v % 1)
    assert kings, "castling must be recognised from file letters"
    assert all(c in "KQkq" for c in cg.perturb_position(fen, perturb_type=7, rng=random.Random(0)).split()[2].replace("-", ""))


# ##### ##### ##### ##### #####
#   Runner

def main() -> int:
    tests: List[Callable[[], None]] = [
        obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)
    ]
    db = _db_path()
    print(f"Database: {db if db else 'not found, using built-in positions only'}")
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
