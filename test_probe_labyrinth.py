"""
Tests for probe_labyrinth.py.

Run with plain Python (no pytest needed):

    python test_probe_labyrinth.py

They check the metrics against brute-force or known answers, the label conversions (engine scores
to the side to move's view, moves to the side to move's frame, the puzzle's setting move), and a
small end-to-end run writing the CSVs.
"""
import csv
import io
import os
import random
import sqlite3
import sys
import tempfile
import traceback
from typing import Callable, List

import chess
import numpy as np
import torch

import labyrinth_data as ld
import labyrinth_jepa_encoder as lje
import probe_labyrinth as pl


def test_auc_matches_a_brute_force_count():
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        scores = torch.randint(0, 6, (60,), generator=g).float()     # many ties
        labels = (torch.rand(60, generator=g) < 0.4).float()
        pos, neg = scores[labels > 0.5], scores[labels < 0.5]
        brute = ((pos[:, None] > neg[None, :]).double() + 0.5 * (pos[:, None] == neg[None, :]).double()).mean()
        assert abs(pl.auc(scores, labels) - float(brute)) < 1e-9
    assert np.isnan(pl.auc(torch.rand(5), torch.zeros(5)))


def test_ridge_knn_and_correlation_metrics():
    torch.manual_seed(0)
    x = torch.randn(2000, 16)
    y = x @ torch.randn(16) + 0.01 * torch.randn(2000)
    coef = pl.ridge(x, y - y.mean(), 1e-3)
    assert pl.r2(x @ coef + y.mean(), y) > 0.99
    assert abs(pl.spearman(y, y * 3 + 1) - 1) < 1e-9 and abs(pl.pearson(y, -y) + 1) < 1e-9
    # two well-separated clusters: neighbours share the label
    centers = torch.randn(2, 16) * 5
    labels = torch.randint(0, 2, (500,))
    pts = centers[labels] + torch.randn(500, 16)
    nb = pl.knn_indices(pts[:100], pts[100:], 10)
    agree = (labels[100:][nb] == labels[:100, None]).float().mean()
    assert agree > 0.95
    assert pl._max_abs_corr(torch.cat([x, y[:, None]], 1), y) > 0.999


def test_scores_moves_and_board_labels():
    # engine scores are from White's view; probes use the side to move's
    assert pl.score_to_side_to_move("200", True)[1] == 200 and pl.score_to_side_to_move("200", False)[1] == -200
    assert pl.score_to_side_to_move("0", False)[0] == 0.5
    assert pl.score_to_side_to_move("#+3", False)[:2] == (0.0, -1000.0)     # White mates, Black to move
    assert pl.score_to_side_to_move("#-3", False)[:2] == (1.0, 1000.0)
    assert pl.score_to_side_to_move("#+3", True)[2] == 1.0
    # moves in the side to move's frame: Black's e7e5 is e2e4
    black = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    assert pl.oriented_move(black, chess.Move.from_uci("e7e5")) == "e2e4"
    assert pl.oriented_move(chess.Board(), chess.Move.from_uci("g1f3")) == "g1f3"
    labels = pl.board_labels(chess.Board("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1"))
    assert labels["material_balance"] == -9 and labels["in_check"] == 1 and labels["white_to_move"] == 0


def test_puzzles_play_the_setting_move_first():
    import zstandard
    header = "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags,DailyDate"
    rows = ["00008,r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24,f2g3 e6e7 b2b1 b3c1 b1c1 h6c1,1811,76,95,10282,"
            "crushing hangingPiece long middlegame,https://lichess.org/787zsVup/black#48,,",
            "0000D,5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27,d3d6 f8d8 d6d8 f6d8,1529,74,96,37506,"
            "advantage endgame short,https://lichess.org/F8M8OS71#53,Sicilian_Defense Sicilian_Defense_Other,"]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "p.csv.zst")
        with open(path, "wb") as f:
            f.write(zstandard.ZstdCompressor().compress(("\n".join([header] + rows) + "\n").encode()))
        out = pl.load_puzzles(path, base_rate=1.0, per_theme=1, min_count=1, log=lambda m: None)
    first = chess.Board("r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24")
    first.push(chess.Move.from_uci("f2g3"))                   # the puzzle starts after the opponent's move
    assert out.fens[0] == ld.fen_of(first)
    assert out.binary["crushing"].tolist() == [1, 0] and out.kinds["crushing"] == "theme"
    assert out.binary["opening:Sicilian_Defense"].tolist() == [0, 1]
    assert out.binary["move:e6e7"][0] == 1                     # White's solution move, White to move
    assert out.numeric["puzzle_rating"].tolist() == [1811, 1529]


TINY = dict(token_variant="squares", gat_dim=16, gat_heads=2, gat_layers=2, d_model=16, transformer_layers=1,
            transformer_heads=2, readout_queries=2, z_dim=16, pred_dim=16, pred_gat_layers=1, pred_heads=2,
            move_dim=16, next_hidden=16, next_layers=1)


def test_encoder_views_are_the_encoders_own_vectors():
    torch.manual_seed(0)
    encoder = lje.LabyrinthEncoder(lje.JEPAConfig(**TINY)).eval()
    viewer = pl.EncoderViews(encoder, list(pl.VIEWS))
    rnd = random.Random(1)
    board = chess.Board.from_chess960_pos(7)
    fens = []
    for _ in range(6):
        board.push(rnd.choice(list(board.legal_moves)))
        fens.append(ld.fen_of(board))
    graphs = ld.pack_views([ld.build_views(f, torch.zeros(64, dtype=torch.bool))[0] for f in fens])
    views = viewer(graphs)
    with torch.no_grad():
        out = encoder(graphs)
        assert torch.allclose(views["z"], out.z) and torch.allclose(views["token_mean"], out.squares.mean(1))
        # the readout's input gives z through its last layer, and the captured transformer input gives the
        # final square tokens through the transformer and its norm
        assert torch.allclose(encoder.readout_proj(views["readout_in"]), out.z, atol=1e-5)
        tokens = viewer._captured["pre_transformer"]
        assert torch.allclose(encoder.final_norm(encoder.transformer(tokens)), out.squares, atol=1e-5)
        assert torch.allclose(views["pre_transformer_mean"], tokens.mean(1))
    assert views["readout_in"].shape == (6, 2 * 16)
    try:
        pl.EncoderViews(encoder, ["nope"])
        assert False, "unknown views are refused"
    except ValueError:
        pass


def test_end_to_end_writes_the_csvs():
    rnd = random.Random(0)
    boards = []
    for i in range(12):
        b = chess.Board.from_chess960_pos(rnd.randrange(960))
        for _ in range(rnd.randint(2, 12)):
            b.push(rnd.choice(list(b.legal_moves)))
            boards.append(b.copy(stack=False))
    labels = pl.LabelSet("labeled", [ld.fen_of(b) for b in boards])
    labels.binary["best_move:x"] = np.array([i % 3 == 0 for i in range(len(boards))], dtype=np.float32)
    labels.kinds["best_move:x"] = "best_move"
    labels.numeric["eval_win_prob"] = np.random.default_rng(0).random(len(boards)).astype(np.float32)
    labels.kinds["eval_win_prob"] = "eval"
    themes = pl.LabelSet("puzzles", labels.fens[:60])
    themes.binary["fork"] = np.array([i % 2 for i in range(60)], dtype=np.float32)
    themes.kinds["fork"] = "theme"
    pl._add_board_labels(themes, [pl.board_labels(b) for b in boards[:60]])
    cfg = lje.JEPAConfig(**TINY)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ckpt = os.path.join(tmp, "best.pt")
        model = lje.LabyrinthJEPA(cfg)
        with torch.no_grad():
            model.target.readout_proj.bias.add_(1.0)       # so the online and target encoders differ
        torch.save({"model": model.state_dict(), "jepa_config": cfg.to_dict()}, ckpt)
        out = os.path.join(tmp, "results.csv")
        pl.run_probes({"tiny": ckpt}, [themes, labels], out, baselines=True, device="cpu", k=5, workers=0,
                      views=["z", "token_mean"], online=True, report_fens=labels.fens[:24], report_samples=2,
                      log=lambda m: None)
        with open(out) as f:
            rows = list(csv.DictReader(f))
        with open(os.path.splitext(out)[0] + "_summary.csv") as f:
            summary = list(csv.DictReader(f))
        with open(os.path.splitext(out)[0] + "_representation.csv") as f:
            report = list(csv.DictReader(f))
        for which, part in (("online", model.encoder), ("target", model.target)):
            assert torch.equal(pl.load_encoder(ckpt, "cpu", which).readout_proj.bias, part.readout_proj.bias)
    assert {r["model"] for r in rows} == {"tiny", "untrained", "board_onehot"}
    assert {(r["model"], r["encoder"], r["view"]) for r in rows} == {
        ("tiny", "target", "z"), ("tiny", "target", "token_mean"), ("tiny", "online", "z"),
        ("tiny", "online", "token_mean"), ("untrained", "random init", "z"), ("untrained", "random init", "token_mean"),
        ("board_onehot", "", "onehot")}
    assert {(r["model"], r["encoder"], r["view"]) for r in report} == {(r["model"], r["encoder"], r["view"])
                                                                        for r in rows}
    board = next(r for r in report if r["model"] == "board_onehot")
    assert board["dims"] == "512" and 0 < float(board["self_consistency"]) < 1 and board["lidar"] == ""
    assert {s["view"] for s in summary} == {"z", "token_mean", "onehot"}
    theme_rows = [r for r in rows if r["is_theme"] == "True"]
    assert {r["label"] for r in theme_rows} == {"fork"} and all(r["kind"] == "theme" for r in theme_rows)
    assert {"white_to_move", "material_balance", "eval_win_prob", "best_move:x"} <= {r["label"] for r in rows}
    fork = next(r for r in rows if r["label"] == "fork" and r["model"] == "tiny")
    assert fork["linear_auc"] != "" and fork["knn_lift"] != ""
    ev = next(r for r in rows if r["label"] == "eval_win_prob" and r["model"] == "board_onehot")
    assert ev["linear_r2"] != "" and ev["linear_auc"] == ""
    assert {s["kind"] for s in summary} >= {"theme", "board", "eval", "best_move"}
    assert all("sample bias" in r["notes"] for r in rows if r["dataset"] == "labeled")
    assert "expected near chance" in next(r for r in rows if r["label"] == "white_to_move")["notes"]


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
