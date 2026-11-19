"""
Tests for threat_head_fit.py, which fits the legal-threat head alone on fixed inputs.

Run with plain Python (no pytest needed):

    python tests/test_threat_head_fit.py

A small labels database made by the labeller on a small position cache, in a temporary folder, on the CPU.
They check that the positions and labels read back from the database are the labeller's, in the graph's frame;
that every form of the head fits and is scored on held-out positions (database positions held out, and
validation positions searched on the spot when the compiled search is built); that a head can learn a small
set of positions by heart, so the fitting itself works; and that the table shows every score.
"""
import os
import random
import sys
import tempfile
import traceback
from typing import Callable, List

import chess
import numpy as np
import torch

import repo_root  # noqa: F401  (first: it puts the repo's root folder, where the code under test is, on sys.path)
import labyrinth_data as ld
import labyrinth_jepa_encoder as lje
import piece_reach as pr
import probe_heads as ph
import reach_labeler as rl
import threat_head_fit as thf


def make_cache(folder: str, n_starts: int = 30) -> str:
    rnd = random.Random(0)
    fens = []
    for i in range(n_starts):
        board = chess.Board() if i == 0 else chess.Board.from_chess960_pos(rnd.randrange(960))
        for _ in range(8):
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


def labelled(folder: str, plies: int = 1):
    cache = make_cache(folder)
    db = os.path.join(folder, "labels.db")
    rl.label(db_path=db, cache_dir=cache, split="train", depth=plies, workers=0, rollout_plies=1, kind="threats", log=quiet)
    return cache, db


# ##### ##### ##### ##### #####
#   Tests


def test_positions_and_labels_come_back_as_the_labeller_wrote_them():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache, db = labelled(tmp, plies=2)
        fens, labels = thf.stored_positions(db, 2, 40)
        assert len(fens) == len(labels) == 40
        walk = [f for sample in list(rl.walk_positions(cache, "train", 0, 0, 1))[:25] for f in sample]
        assert [ld.position_key(f) for f in fens] == list(dict.fromkeys(ld.position_key(f) for f in walk))[:40], \
            "the database's first positions, in the order they were labelled"
        for fen, label in list(zip(fens, labels))[:6]:
            want = ld.threat_cells(pr.threat_counts(fen, 2), fen.split()[1] == "w")
            assert label.layers == 3 and np.array_equal(label.codes, want[0]) and np.array_equal(label.logs, want[1])
        shallow = thf.stored_positions(db, 1, 5)[1]
        assert shallow[0].layers == 2 and np.array_equal(shallow[0].codes, labels[0].codes[labels[0].codes < 2 * 4096])
        assert thf.stored_positions(db, 3, 5) == ([], []), "nothing is labelled that deep"
        searched = thf.searched_positions(cache, "val", 1, 6, seed=1)
        assert len(searched[0]) == 6 and all(label.origin == ld.THREATS_COMPUTED for label in searched[1])
        want = ld.threat_cells(pr.threat_counts(searched[0][0], 1), searched[0][0].split()[1] == "w")
        assert np.array_equal(searched[1][0].codes, want[0])


def test_every_form_of_the_head_fits_and_is_scored_on_held_out_positions():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache, db = labelled(tmp)
        torch.manual_seed(0)
        tiny = lje.JEPAConfig(token_variant="squares", gat_dim=16, gat_heads=2, gat_layers=2, d_model=16,
                              transformer_layers=0, readout="mean", z_dim=16, zthreat_dim=8, zthreat_hidden=12)
        encoders = {"untrained": lje.LabyrinthEncoder(tiny).eval()}
        messages = []
        results = thf.run(encoders, db, "", lje.THREAT_HEADS, "count", 1, positions=120, test_positions=20, steps=3,
                          device="cpu", config=tiny, log=messages.append)
        assert list(results) == [ph.ONEHOT, ph.RAW, "untrained"] and all(list(r) == list(lje.THREAT_HEADS) for r in results.values())
        assert any("108 positions to fit on" in m and "the last tenth" in m for m in messages)
        for by_head in results.values():
            for scores in by_head.values():
                assert {"found_0", "rejected_1", "new_1", "wrong_1", "exact_0", "close_1", "loss", "params", "seconds"} <= set(scores)
                assert all(0 <= scores[key] <= 1 for key in ("found_0", "rejected_0", "found_1", "new_1", "close_1", "exact_1"))
                assert scores["labelled"] == 1.0
        assert results[ph.ONEHOT]["linear"]["params"] == (64 * lje.SQUARE_CLASSES + 1) * 2 * 2 * 4096, "sized by its input"
        table = thf.table(results, 1, "count")
        assert table.count("\n") == 1 + 3 * 3 and "| one-hot board | linear |" in table and "counts" in table
        assert "counts" not in thf.table(results, 1, "binary")
        if pr.USING_C:              # with the search at hand, scored on validation positions instead
            results = thf.run({}, db, cache, ["linear"], "binary", 1, positions=60, test_positions=10, steps=2,
                              device="cpu", config=tiny, log=messages.append)
            assert any("validation positions" in m and "60 positions to fit on" in m for m in messages)
            assert "close_1" not in results[ph.ONEHOT]["linear"]
        try:
            thf.run({}, db, "", ["linear"], "binary", 4, positions=10, test_positions=2, steps=1, device="cpu", log=quiet)
            raise AssertionError("no labels that deep: it should say so")
        except ValueError:
            pass


def test_a_head_can_learn_a_few_positions_by_heart():
    """Fitted and scored on the same 24 positions, a head gets them right: the fitting and the scoring agree."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        _, db = labelled(tmp)
        fens, labels = thf.stored_positions(db, 1, 24)
        inputs = ph.encode_inputs({}, fens, "cpu")[ph.ONEHOT]
        pieces = inputs.view(len(fens), 64, lje.SQUARE_CLASSES)[:, :, 1:13].sum(-1) > 0
        scores = thf.fit(inputs, pieces, labels, inputs, pieces, labels, "linear", "count", 1, "cpu", steps=150, batch_size=24,
                         lr=3e-3, weight_decay=0.0)
        assert scores["found_0"] > 0.99 and scores["rejected_0"] > 0.99 and scores["found_1"] > 0.99 and scores["rejected_1"] > 0.99, scores
        assert scores["wrong_0"] < 10 and scores["wrong_1"] < 10 and scores["close_1"] > 0.9, scores    # of some 1,500 cells


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
