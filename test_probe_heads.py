"""
Tests for probe_heads.py: the quiet-move rule, the from-to pairs, the matched sets and the trained head.
Run: python test_probe_heads.py
"""
from __future__ import annotations

import random
import sys
import traceback
from typing import Callable, List

import chess
import numpy as np
import torch

import labyrinth_data as ld
import probe_heads as ph


def _move(fen: str, uci: str, last: bool = False) -> bool:
    before = chess.Board(fen, chess960=True)
    move = before.parse_uci(uci)
    after = before.copy(stack=False)
    after.push(move)
    return ph.is_quiet_move(before, move, after, last)


def test_quiet_move_rule_follows_lichess():
    # a rook lift that attacks nothing: quiet, unless it is the puzzle's last move
    quiet_fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
    assert _move(quiet_fen, "a1a4")
    assert not _move(quiet_fen, "a1a4", last=True)
    assert not _move(quiet_fen, "a1a8")                                         # gives check
    assert not _move(quiet_fen, "g1f1")                                         # a king move
    assert not _move("6k1/5ppp/8/8/8/8/r4PPP/R5K1 w - - 0 1", "a1a2")           # a capture
    assert not _move("6k1/5ppp/8/n7/8/8/5PPP/1R4K1 w - - 0 1", "b1a1")          # the rook now attacks the knight
    assert _move("6k1/5ppp/8/n7/8/8/5PPP/1R4K1 w - - 0 1", "b1c1")              # it does not from c1
    assert not _move("6k1/5ppp/8/8/8/8/5PPP/R3K3 b - - 0 1", "g8h8")            # the other side's king move
    assert not _move("6k1/5ppp/8/8/8/8/r4PPP/6K1 w - - 0 1", "g1f1")            # king again (and rook attacks nothing)
    # escaping a check is not quiet
    assert not _move("6k1/5ppp/8/8/8/8/3R1PPP/r5K1 w - - 0 1", "d2d1")           # the rook blocks a check
    # pawns: a push to the fifth rank or beyond is "advanced" for White, to the fourth or below for Black
    assert _move("6k1/8/8/8/8/2P5/8/6K1 w - - 0 1", "c3c4")
    assert not _move("6k1/8/8/2P5/8/8/8/6K1 w - - 0 1", "c5c6")
    assert _move("6k1/8/2p5/8/8/8/8/6K1 b - - 0 1", "c6c5")
    assert not _move("6k1/8/8/8/2p5/8/8/6K1 b - - 0 1", "c4c3")


def test_advanced_pawn_boundaries():
    # White: rank index > 4 (the sixth rank on) is advanced; Black: rank index < 3 (its sixth rank on)
    assert _move("6k1/8/8/8/2P5/8/8/6K1 w - - 0 1", "c4c5")                     # to the fifth: still quiet
    assert not _move("6k1/8/8/2P5/8/8/8/6K1 w - - 0 1", "c5c6")                 # to the sixth: advanced
    assert _move("6k1/8/8/2p5/8/8/8/6K1 b - - 0 1", "c5c4")                     # Black to its fifth: quiet
    assert not _move("6k1/8/8/8/2p5/8/8/6K1 b - - 0 1", "c4c3")                 # to its sixth: advanced


def test_solver_moves_and_forcing_moves():
    # the opponent (White) sets the puzzle with Kg1-h1; Black solves: ...Qd1+ (check), Kh2, ...Qd6+ (last)
    fen = "6k1/5ppp/8/8/8/3q4/5PP1/6K1 w - - 0 1"
    line = ph.solver_moves(fen, ["g1h1", "d3d1", "h1h2", "d1d6"])
    assert len(line) == 2 and [last for _, _, _, last in line] == [False, True]
    before, move, after, _ = line[0]
    assert before.turn == chess.BLACK and after.turn == chess.WHITE
    assert ph.is_forcing_move(before, move) and not ph.is_quiet_move(before, move, after, False)
    # standard castling notation is read on a Chess960 board
    castle = ph.solver_moves("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", ["e8e7", "e1g1", "e7e6", "a1a2"])
    assert castle[0][0].is_castling(castle[0][1])
    try:
        ph.solver_moves(fen, ["g1h1", "d3d9"])
        raise AssertionError("an unplayable line should raise")
    except ValueError:
        pass


def test_pairs_are_in_the_movers_frame():
    assert ph.NUM_PAIRS == 1792
    white = chess.Board()
    black = chess.Board()
    black.push_san("e4")
    # e2e4 for White and e7e5 for Black are the same pair
    assert ph.move_pair(white, white.parse_san("e4")) == ph.move_pair(black, black.parse_san("e5"))
    assert ph.move_pair(white, white.parse_san("Nf3")) == ph.move_pair(black, black.parse_san("Nf6"))
    legal = ph.legal_pairs(white)
    assert len(legal) == 20 and (legal >= 0).all() and ph.move_pair(white, white.parse_san("e4")) in set(legal.tolist())
    mask = ph.legal_mask([legal, ph.legal_pairs(black)], "cpu")
    assert mask.shape == (2, 1792) and mask.sum(1).tolist() == [20, 20]


def test_matching_gives_equal_material_on_both_sides():
    sig = lambda fen: ph.material_signature(chess.Board(fen))
    assert sig("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1") == ("r", "", 6)
    assert sig("6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1") == ("", "r", 6)
    rng = random.Random(0)
    positives = [("a", f"p{i}") for i in range(5)] + [("b", "p5"), ("c", "p6")]
    negatives = [("a", f"n{i}") for i in range(3)] + [("b", f"m{i}") for i in range(4)]
    pos, neg = ph.match_by_signature(positives, negatives, rng)
    assert len(pos) == len(neg) == 4                                            # 3 of "a", 1 of "b", none of "c"
    assert sum(p.startswith("p") for p in pos) == 4 and "p6" not in pos
    count = lambda items, prefix: sum(i.startswith(prefix) for i in items)
    assert count(neg, "n") == 3 and count(neg, "m") == 1
    pos, neg = ph.match_by_signature(positives, negatives, rng, limit=2)
    assert len(pos) == len(neg) == 2


def test_the_head_learns_each_kind_of_probe():
    torch.manual_seed(0)
    n = 3000
    x = torch.randn(n, 24)
    split = np.where(np.arange(n) % 10 < 8, 0, np.where(np.arange(n) % 10 == 8, 1, 2))
    common = dict(device="cpu", hidden=64, epochs=30, batch_size=128, lr=3e-3)
    # binary: an XOR of two signs, which no linear probe reads
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).float().numpy()
    assert ph.fit_head(x, y, split, "binary", **common)["auc"] > 0.95
    assert ph.fit_head(torch.randn(n, 24), y, split, "binary", **common)["auc"] < 0.6          # nothing to read
    # regression: a product
    assert ph.fit_head(x, (x[:, 2] * x[:, 3]).numpy(), split, "regression", **common)["r2"] > 0.8
    # move: the answer is the largest of three inputs, among the legal pairs only
    target = x[:, :3].argmax(1).numpy()
    legal = [np.array([0, 1, 2, 7]) for _ in range(n)]
    mask = ph.legal_mask(legal, "cpu")
    scores = ph.fit_head(x, target, split, "move", mask, **common)
    assert scores["top1"] > 0.9 and scores["top3"] >= scores["top1"]
    # the same seed gives the same head
    a = ph.fit_head(x, y, split, "binary", seed=3, **common)
    b = ph.fit_head(x, y, split, "binary", seed=3, **common)
    assert a == b


def test_split_is_by_position():
    fens = [ld.normalize_fen(f) for f in ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 5 9")]
    assert ph.split_of(fens)[0] == ph.split_of(fens)[1]                         # move counters do not matter
    many = [chess.Board().fen()] * 3
    assert len(set(ph.split_of(many).tolist())) == 1


def test_inputs_are_z_and_the_raw_board():
    import labyrinth_jepa_encoder as lje
    cfg = lje.JEPAConfig(token_variant="squares", gat_dim=32, gat_heads=2, gat_layers=2, d_model=32,
                         transformer_layers=0, readout="mean", z_dim=48, pred_dim=32, pred_gat_layers=2,
                         pred_heads=2, move_dim=32, next_hidden=64, next_layers=1)
    encoder = lje.LabyrinthEncoder(cfg).eval()
    fens = [ld.normalize_fen(chess.Board().fen()), ld.normalize_fen("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")]
    out = ph.encode_inputs({"tiny": encoder}, fens, "cpu", batch_size=1)
    assert out["tiny"].shape == (2, 48) and out[ph.RAW].shape[0] == 2 and out[ph.RAW].shape[1] % 64 == 0
    assert out["tiny"].dtype == torch.float16


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
