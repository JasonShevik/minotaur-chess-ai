"""
Tests for the Labyrinth training-data pipeline in labyrinth_data.py.

Run with plain Python (no pytest needed):

    python test_labyrinth_data.py

They build small caches in a temporary folder (from a throwaway SQLite database and a FEN file),
so they do not need minotaur_data.db. They check that:
  * splits depend only on the position, are disjoint, and agree between data sources;
  * the cache drops invalid and duplicate positions and keeps engine labels for the probes;
  * masks have the configured sizes and are reproducible;
  * the masked view never leaks what stands on the masked squares, including through castling
    edges and the en passant flag: changing the masked squares leaves the masked view unchanged;
  * next positions, batches and move indices line up with chess_graph and the encoder's batching;
  * the loader is reproducible for a given (seed, epoch), whatever the number of workers.
"""
import os
import pickle
import random
import sqlite3
import sys
import tempfile
import traceback
from typing import Callable, List

import chess
import torch

import chess_graph as cg
import labyrinth_data as ld


# ##### ##### ##### ##### #####
#   Helpers


def random_positions(n_starts: int, seed: int, per_start: int = 10) -> List[chess.Board]:
    """Boards from random play, from standard and Chess960 starts: castling rights, en passant, both sides to move."""
    rnd = random.Random(seed)
    boards = []
    for i in range(n_starts):
        board = chess.Board() if i == 0 else chess.Board.from_chess960_pos(rnd.randrange(960))
        for _ in range(per_start):
            boards.append(board.copy(stack=False))
            for _ in range(rnd.randint(1, 6)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                # prefer double pawn pushes now and then, so en passant flags appear
                doubles = [m for m in moves if board.piece_type_at(m.from_square) == chess.PAWN
                           and abs(m.to_square - m.from_square) == 16]
                board.push(rnd.choice(doubles) if doubles and rnd.random() < 0.3 else rnd.choice(moves))
    return boards


def make_database(path: str, fens: List[str], analyzed_every: int = 3) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(f'CREATE TABLE "{ld.TABLE}" (fen TEXT PRIMARY KEY, is_analyzed INTEGER, engine_name TEXT, '
                     'depth INTEGER, score TEXT, is_forced_checkmate INTEGER, best_move TEXT)')
        for i, fen in enumerate(fens):
            analyzed = i % analyzed_every == 0
            conn.execute(f'INSERT OR IGNORE INTO "{ld.TABLE}" VALUES (?, ?, ?, ?, ?, ?, ?)',
                         (fen, int(analyzed), "Stockfish 19" if analyzed else None, 20 if analyzed else None,
                          str(i) if analyzed else None, 0, "e2e4" if analyzed else None))


def edge_set(view: ld.GraphView, edge_type: int = None):
    keep = torch.ones(view.edge_type.numel(), dtype=torch.bool) if edge_type is None else view.edge_type == edge_type
    return {(int(t), a, b) for t, (a, b) in zip(view.edge_type[keep].tolist(), view.edge_index[:, keep].t().tolist())}


# ##### ##### ##### ##### #####
#   Tests


def test_splits_depend_only_on_the_position():
    boards = random_positions(60, seed=1)
    for b in boards:
        fen = b.fen(en_passant="xfen")
        parts = fen.split()
        other_counters = " ".join(parts[:4] + ["37", "99"])
        assert ld.split_of(fen) == ld.split_of(other_counters)
    counts = {s: 0 for s in ld.SPLITS}
    for i in range(20000):
        counts[ld.split_of(f"{i}")] += 1
    assert 120 < counts["val"] < 280 and 120 < counts["test"] < 280, counts   # ~1% each of 20,000


def test_cache_from_database_and_from_a_fen_file():
    boards = random_positions(40, seed=2)
    fens = [b.fen(en_passant="xfen") for b in boards]
    dup = " ".join(fens[1].split()[:4] + ["50", "80"])            # same position, other counters
    rows = fens + [dup, "not a fen", "8/8/8/8/8/8/8/8 w - - 0 1"]   # the last has no kings: invalid
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = os.path.join(tmp, "test.db")
        make_database(db, rows)
        meta = ld.build_position_cache(db, os.path.join(tmp, "from_db"), val_fraction=0.2, test_fraction=0.2,
                                       verbose=False)
        assert meta["invalid"] == 2 and meta["duplicates"] >= 1
        splits = {s: set(ld.PositionDataset(os.path.join(tmp, "from_db"), s).fen(i)
                         for i in range(len(ld.PositionDataset(os.path.join(tmp, "from_db"), s))))
                  for s in ld.SPLITS}
        assert not (splits["train"] & splits["val"]) and not (splits["train"] & splits["test"])
        assert sum(len(v) for v in splits.values()) == meta["rows_read"] - meta["invalid"] - meta["duplicates"]
        for s, members in splits.items():
            assert all(ld.split_of(f, 0.2, 0.2) == s for f in members)
            assert all(f == ld.normalize_fen(f) for f in members)
        # labels: exactly the analyzed rows that survived, in their own split
        labeled = 0
        for s in ld.SPLITS:
            lab = ld.load_labels(os.path.join(tmp, "from_db"), s)
            assert set(lab.fen) <= splits[s]
            assert all(e == "Stockfish 19" for e in lab.engine_name) and all(d == 20 for d in lab.depth)
            labeled += len(lab.fen)
        assert labeled > 0

        # the same positions from a text file land in the same splits
        path = os.path.join(tmp, "fens.txt")
        with open(path, "w") as f:
            f.write("# a comment\n\n" + "\n".join(rows) + "\n")
        ld.build_position_cache_from_file(path, os.path.join(tmp, "from_file"), 0.2, 0.2, verbose=False)
        for s in ld.SPLITS:
            ds = ld.PositionDataset(os.path.join(tmp, "from_file"), s)
            assert {ds.fen(i) for i in range(len(ds))} == splits[s], s
            assert len(ld.load_labels(os.path.join(tmp, "from_file"), s).fen) == 0


def test_masks_have_the_configured_size_and_are_reproducible():
    cfg = ld.MaskConfig(min_ratio=0.2, max_ratio=0.3, max_block_radius=2)
    for i in range(200):
        m = ld.sample_mask(random.Random(i), cfg)
        assert round(0.2 * 64) <= int(m.sum()) <= round(0.3 * 64)
        assert torch.equal(m, ld.sample_mask(random.Random(i), cfg))


def test_masked_view_follows_the_rules():
    rng = random.Random(3)
    saw_castling_drop = saw_ep_clear = 0
    for board in random_positions(60, seed=3):
        fen = board.fen(en_passant="xfen")
        mask = ld.sample_mask(rng, ld.MaskConfig(0.3, 0.6))
        full, ctx, _, rooks = ld.build_views(fen, mask)
        assert torch.all(ctx.x[mask] == 0)
        unmasked = ~mask
        ep_diff = (ctx.x[unmasked] != full.x[unmasked])
        # the only difference allowed on unmasked squares is a cleared en passant flag
        assert torch.all(ep_diff[:, :7] == 0)
        saw_ep_clear += int(ep_diff[:, 7].sum())
        # every edge type except castling is untouched; castling edges only ever disappear
        for t in range(cg.NUM_EDGE_TYPES - 1):
            assert edge_set(ctx, t) == edge_set(full, t)
        assert edge_set(ctx, 9) <= edge_set(full, 9)
        saw_castling_drop += edge_set(ctx, 9) != edge_set(full, 9)
    assert saw_castling_drop > 0 and saw_ep_clear > 0, (saw_castling_drop, saw_ep_clear)


def test_masked_view_does_not_leak_the_masked_squares():
    """
    The decisive test: change what stands on the masked squares (remove pieces, kings and rooks included,
    or add pieces) and the masked view must not change at all.
    """
    rng = random.Random(4)
    pieces = [chess.Piece(t, c) for t in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
              for c in (chess.WHITE, chess.BLACK)]
    changed = 0
    for board in random_positions(80, seed=4):
        mask = ld.sample_mask(rng, ld.MaskConfig(0.2, 0.5))
        orient = (lambda sq: sq) if board.turn == chess.WHITE else (lambda sq: sq ^ 56)
        other = board.copy(stack=False)
        for sq in chess.SQUARES:
            if not mask[orient(sq)]:
                continue
            if other.piece_at(sq) is not None and rng.random() < 0.6:
                other.remove_piece_at(sq)
            elif other.piece_at(sq) is None and rng.random() < 0.2 and chess.square_rank(sq) not in (0, 7):
                other.set_piece_at(sq, rng.choice(pieces))
        if other.board_fen() == board.board_fen():
            continue
        changed += 1
        _, ctx_a, _, _ = ld.build_views(ld.fen_of(board), mask)
        _, ctx_b, _, _ = ld.build_views(ld.fen_of(other), mask)
        assert torch.equal(ctx_a.x, ctx_b.x), f"node features leak: {board.fen(en_passant="xfen")} vs {other.fen(en_passant="xfen")}"
        assert edge_set(ctx_a) == edge_set(ctx_b), f"edges leak: {board.fen(en_passant="xfen")} vs {other.fen(en_passant="xfen")}"
    assert changed > 50, changed


def test_next_position_matches_the_move_sampler():
    rng = random.Random(5)
    boards = random_positions(30, seed=5) + [chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")]   # last: checkmate
    for board in boards:
        s = ld.build_sample(board.fen(en_passant="xfen"), rng)
        if board.is_checkmate():
            assert s.next is None
            continue
        after = board.copy(stack=False)
        after.push(chess.Move.from_uci(s.next.move))
        assert s.next.next_fen == after.fen(en_passant="xfen")
        edges, x = cg.create_filled_chess_graphs(after.fen(en_passant="xfen"), cg.castling_rook_squares_from_board(after))
        assert torch.equal(s.next.view.x, x)
        assert edge_set(s.next.view) == {(t, a, b) for t, e in enumerate(edges) for a, b in e.t().tolist()}
        assert s.full.x[s.next.from_square, 0] == 0 and s.full.x[s.next.from_square, 1:7].sum() == 1


def test_batches_line_up_with_the_encoder_batching():
    import labyrinth_dgi_encoder as lde
    rng = random.Random(6)
    fens = [b.fen(en_passant="xfen") for b in random_positions(6, seed=6, per_start=2)] + ["7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"]
    samples = [ld.build_sample(f, rng) for f in fens]
    batch = ld.collate_jepa(samples)
    reference = lde.batch_chess_graphs([cg.create_filled_chess_graphs(f, cg.castling_rook_squares_from_board(
        chess.Board(f, chess960=True))) for f in fens])
    assert torch.equal(batch.full.x, reference[0]) and torch.equal(batch.full.edge_index, reference[1])
    assert torch.equal(batch.full.edge_attr, reference[2]) and torch.equal(batch.full.batch, reference[3])
    assert batch.mask.shape == (len(fens) * 64,) and batch.context.x.shape == batch.full.x.shape
    # the checkmated position has no next position; every other one does, with move indices into its block
    assert batch.next_index.tolist() == list(range(len(fens) - 1))
    assert batch.next.num_graphs == len(fens) - 1
    for k, b in enumerate(batch.next_index.tolist()):
        assert batch.move_from[k] // 64 == b and batch.move_to[k] // 64 == b
        assert batch.move_from[k] % 64 == samples[b].next.from_square
    assert batch.move_flags.shape == (len(fens) - 1, 3)


def test_loader_is_reproducible_and_epochs_differ():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "fens.txt")
        with open(path, "w") as f:
            f.write("\n".join(b.fen(en_passant="xfen") for b in random_positions(30, seed=7)))
        cache = os.path.join(tmp, "cache")
        ld.build_position_cache_from_file(path, cache, 0.0, 0.0, verbose=False)

        def epoch_digest(loader, epoch):
            loader.set_epoch(epoch)
            return [(b.fens, b.mask.tolist(), b.moves) for b in loader]

        serial = ld.JEPALoader(cache, "train", batch_size=16, num_workers=0, seed=3)
        parallel = ld.JEPALoader(cache, "train", batch_size=16, num_workers=2, seed=3)
        e0 = epoch_digest(serial, 0)
        assert e0 == epoch_digest(parallel, 0), "workers must not change what is produced"
        assert e0 == epoch_digest(serial, 0), "the same epoch must be reproducible"
        assert e0 != epoch_digest(serial, 1), "a new epoch must draw a new order, masks and moves"
        assert sum(len(d[0]) for d in e0) == len(serial.dataset)
        # the dataset is picklable for worker processes and never pickles the positions array
        ds = serial.dataset
        ds.fen(0)
        assert pickle.loads(pickle.dumps(ds))._positions is None


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
