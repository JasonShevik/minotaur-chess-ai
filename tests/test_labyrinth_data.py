"""
Tests for the Labyrinth training-data pipeline in labyrinth_data.py.

Run with plain Python (no pytest needed):

    python tests/test_labyrinth_data.py

They build small caches in a temporary folder (from a throwaway SQLite database and a FEN file),
so they do not need minotaur_data.db. They check that:
  * splits depend only on the position, are disjoint, and agree between data sources;
  * the cache drops invalid and duplicate positions and keeps engine labels for the probes;
  * masks have the configured sizes and are reproducible;
  * the masked view never leaks what stands on the masked squares, including through castling
    edges and the en passant flag: changing the masked squares leaves the masked view unchanged;
  * next positions, batches and move indices line up with chess_graph and the encoder's batching;
  * the loader is reproducible for a given (seed, epoch), whatever the number of workers;
  * legal-threat labels are turned into the graph's frame, come from the labels database or the search, and
    ride along in samples and batches without disturbing anything else.
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
import numpy as np
import torch

import repo_root  # noqa: F401  (first: it puts the repo's root folder, where the code under test is, on sys.path)
import chess_graph as cg
import labyrinth_data as ld
import piece_reach as pr
import reach_labeler as rl


class SearchedThreats(ld.ThreatLabels):
    """Labels straight from the search, whichever one is installed (for a handful of positions the Python one will do)."""

    def counts(self, fen):
        return pr.threat_counts(fen, self.plies), ld.THREATS_COMPUTED


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


def edge_set(view: ld.GraphView):
    """{(from, to): tuple of relation-type flags} for a graph with one multi-hot edge per square pair."""
    return {(a, b): tuple(int(v) for v in attr) for (a, b), attr in zip(view.edge_index.t().tolist(), view.edge_attr.tolist())}


def typed_edge_set(edges):
    """The same, built directly from create_filled_chess_graphs's per-type lists (plus every square's self-loop)."""
    out = {(i, i): (0,) * len(edges) for i in range(64)}
    for t, e in enumerate(edges):
        for a, b in e.t().tolist():
            flags = list(out.get((a, b), (0,) * len(edges)))
            flags[t] = 1
            out[(a, b)] = tuple(flags)
    return out


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
        # every relation type except castling is untouched; castling flags only ever disappear
        full_edges, ctx_edges = edge_set(full), edge_set(ctx)
        for pair, flags in full_edges.items():
            kept = ctx_edges.get(pair, (0,) * cg.NUM_EDGE_TYPES)
            assert kept[:9] == flags[:9], (fen, pair)
            assert kept[9] <= flags[9]
            saw_castling_drop += kept[9] < flags[9]
        assert set(ctx_edges) <= set(full_edges)
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
        assert edge_set(s.next.view) == typed_edge_set(edges)
        assert s.full.x[s.next.from_square, 0] == 0 and s.full.x[s.next.from_square, 1:7].sum() == 1


def test_batches_line_up_with_the_encoder_batching():
    import labyrinth_dgi_encoder as lde
    rng = random.Random(6)
    fens = [b.fen(en_passant="xfen") for b in random_positions(6, seed=6, per_start=2)] + ["7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"]
    samples = [ld.build_sample(f, rng) for f in fens]
    batch = ld.collate_jepa(samples)
    reference = cg.batch_graphs([cg.create_filled_chess_graphs(f, cg.castling_rook_squares_from_board(
        chess.Board(f, chess960=True))) for f in fens])
    dgi = lde.batch_chess_graphs([cg.create_filled_chess_graphs(f, cg.castling_rook_squares_from_board(
        chess.Board(f, chess960=True))) for f in fens])
    assert all(torch.equal(a, b) for a, b in zip(reference, dgi)), "the DGI encoder must read the same graph format"
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


def test_streaming_builder_matches_a_simple_reference():
    """
    The streaming, parallel builder must give exactly what the obvious in-memory algorithm gives:
    normalize, keep the first occurrence of each position, split by hash, keep the kept rows' labels.
    Small chunks exercise many chunks and the order of results from several workers.
    """
    boards = random_positions(25, seed=11)
    fens = [b.fen(en_passant="xfen") for b in boards]
    rows = []
    for i, fen in enumerate(fens):
        rows.append(fen)
        if i % 4 == 0:                                   # the same position again, other move counters
            rows.append(" ".join(fen.split()[:4] + ["9", "99"]))
        if i % 9 == 0:
            rows.append("garbage " + str(i))
    labels = [ld.EngineLabel("Stockfish 19", 20, str(i), False, "e2e4") if i % 3 == 0 else None
              for i in range(len(rows))]

    expected = {s: [] for s in ld.SPLITS}
    expected_labels = {s: [] for s in ld.SPLITS}
    seen = set()
    for row, label in zip(rows, labels):
        norm = ld.normalize_fen(row)
        if norm is None or ld.position_key(norm) in seen:
            continue
        seen.add(ld.position_key(norm))
        split = ld.split_of(norm, 0.2, 0.2)
        expected[split].append(norm)
        if label is not None:
            expected_labels[split].append((norm, label.score))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for workers in (1, 2):
            cache = os.path.join(tmp, f"w{workers}")
            meta = ld.build_position_cache_from_fens(rows, cache, labels, 0.2, 0.2, verbose=False,
                                                     workers=workers, chunk_size=7)
            assert not os.path.exists(os.path.join(cache, "_building")), "temporary files left behind"
            assert meta["invalid"] == sum(1 for r in rows if ld.normalize_fen(r) is None)
            assert meta["rows_read"] == len(rows)
            for split in ld.SPLITS:
                ds = ld.PositionDataset(cache, split)
                assert [ds.fen(i) for i in range(len(ds))] == expected[split], (workers, split)
                lab = ld.load_labels(cache, split)
                assert list(zip(lab.fen, lab.score)) == expected_labels[split], (workers, split)


def test_database_source_streams_with_several_workers():
    """The database cursor is read in the main thread even when workers normalize the positions."""
    fens = [b.fen(en_passant="xfen") for b in random_positions(15, seed=12)]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = os.path.join(tmp, "test.db")
        make_database(db, fens)
        one = ld.build_position_cache(db, os.path.join(tmp, "one"), 0.2, 0.2, verbose=False, workers=1)
        two = ld.build_position_cache(db, os.path.join(tmp, "two"), 0.2, 0.2, verbose=False, workers=2)
        for key in ("rows_read", "invalid", "duplicates", "positions", "labeled"):
            assert one[key] == two[key], key


def test_rollouts_play_legal_moves_described_in_the_starting_frame():
    rng = random.Random(21)
    for board in random_positions(25, seed=21):
        fen = ld.fen_of(board)
        seed = rng.randrange(10 ** 9)
        one = ld.build_sample(fen, random.Random(seed), rollout_steps=1)
        three = ld.build_sample(fen, random.Random(seed), rollout_steps=3)
        # the mask and the first move do not depend on the rollout length
        assert torch.equal(one.mask, three.mask)
        assert (one.next is None) == (three.next is None)
        if one.next is not None:
            assert one.next.move == three.next.move
        orient = (lambda sq: sq) if board.turn == chess.WHITE else (lambda sq: sq ^ 56)
        current = board.copy(stack=False)
        for step in three.rollout:
            move = chess.Move.from_uci(step.move)
            assert move in current.legal_moves
            # squares are node indices of the STARTING position's graph, whoever moves
            assert (step.from_square, step.to_square) == (orient(move.from_square), orient(move.to_square))
            assert step.piece == current.piece_type_at(move.from_square)
            current.push(move)
            assert step.next_fen == ld.fen_of(current)
            edges, x = cg.create_filled_chess_graphs(step.next_fen, cg.castling_rook_squares_from_board(current))
            assert torch.equal(step.view.x, x) and edge_set(step.view) == typed_edge_set(edges)
        assert len(three.rollout) == 3 or current.is_game_over()


def test_batches_of_uneven_rollouts_line_up():
    rng = random.Random(22)
    samples = [ld.build_sample(ld.fen_of(b), rng, rollout_steps=3) for b in random_positions(4, seed=22, per_start=2)]
    samples[1].rollout = samples[1].rollout[:1]      # as if its game had ended after one move
    samples[4].rollout = []                          # ... or had no legal move at all
    samples[6].rollout = samples[6].rollout[:2]
    batch = ld.collate_jepa(samples)
    lengths = [len(s.rollout) for s in samples]
    assert len(batch.rollout) == max(lengths)
    previous = None
    for k, step in enumerate(batch.rollout):
        assert step.index.tolist() == [i for i, n in enumerate(lengths) if n > k]
        assert step.graphs.num_graphs == step.index.numel()
        if previous is None:
            assert torch.equal(step.parent, step.index)
        else:
            assert torch.equal(previous.index[step.parent], step.index), "each row continues its own position"
        for row, i in enumerate(step.index.tolist()):
            assert int(step.move_from[row]) == i * 64 + samples[i].rollout[k].from_square
            assert int(step.piece[row]) == samples[i].rollout[k].piece
        previous = step
    assert batch.moves[1] == [samples[1].rollout[0].move] and batch.moves[4] == []


def test_rollouts_build_graphs_only_for_target_steps():
    rng = random.Random(24)
    boards = random_positions(6, seed=24, per_start=2)
    seeds = [rng.randrange(10 ** 9) for _ in boards]
    every = [ld.build_sample(ld.fen_of(b), random.Random(s), rollout_steps=6) for b, s in zip(boards, seeds)]
    sparse = [ld.build_sample(ld.fen_of(b), random.Random(s), rollout_steps=6, target_steps=[1, 2, 4, 6])
              for b, s in zip(boards, seeds)]
    for a, b in zip(every, sparse):
        # the same moves; graphs only where asked, and those the same graphs
        assert [m.move for m in a.rollout] == [m.move for m in b.rollout]
        for k, (ma, mb) in enumerate(zip(a.rollout, b.rollout), start=1):
            assert (mb.view is not None) == (k in (1, 2, 4, 6))
            if mb.view is not None:
                assert torch.equal(ma.view.x, mb.view.x) and torch.equal(ma.view.edge_index, mb.view.edge_index)
    batch = ld.collate_jepa(sparse)
    assert [step.graphs is not None for step in batch.rollout] == [True, True, False, True, False, True]
    moved = batch.to("cpu")
    assert moved.rollout[2].graphs is None and torch.equal(moved.rollout[2].move_from, batch.rollout[2].move_from)


def test_sibling_moves_are_other_first_moves_from_the_same_position():
    rng = random.Random(26)
    boards = random_positions(8, seed=26, per_start=3)
    boards.append(chess.Board("7k/8/8/8/8/8/8/K6R b - - 0 1", chess960=True))   # in check, two legal moves
    seeds = [rng.randrange(10 ** 9) for _ in boards]
    alone = [ld.build_sample(ld.fen_of(b), random.Random(s), rollout_steps=3) for b, s in zip(boards, seeds)]
    samples = [ld.build_sample(ld.fen_of(b), random.Random(s), rollout_steps=3, siblings=4) for b, s in zip(boards, seeds)]
    for board, a, s in zip(boards, alone, samples):
        # the mask and the rollout are the same with and without siblings
        assert torch.equal(a.mask, s.mask) and [m.move for m in a.rollout] == [m.move for m in s.rollout]
        assert a.siblings == []
        legal = {m.uci() for m in chess.Board(ld.fen_of(board), chess960=True).legal_moves}
        assert len(s.siblings) == min(3, len(legal) - 1)
        names = [m.move for m in s.siblings]
        assert len(set(names)) == len(names) and set(names) <= legal - {s.rollout[0].move}
        for m in s.siblings:
            after = chess.Board(ld.fen_of(board), chess960=True)
            after.push_uci(m.move)
            assert m.next_fen == ld.fen_of(after) and m.view is not None
    assert len(samples[-1].siblings) == 1, "a position with two legal moves has one sibling move"
    # no graph for the siblings when step 1 is not a target
    bare = ld.build_sample(ld.fen_of(boards[0]), random.Random(seeds[0]), rollout_steps=3, target_steps=[2], siblings=4)
    assert all(m.view is None for m in bare.siblings)

    batch, plain = ld.collate_jepa(samples), ld.collate_jepa(alone)
    first, n = batch.rollout[0], plain.rollout[0].index.numel()
    extra = [i for i, s in enumerate(samples) for _ in s.siblings]
    # the rollouts' own first moves keep their rows; the siblings follow, each pointing at its position
    assert first.index.tolist() == plain.rollout[0].index.tolist() + extra
    assert torch.equal(first.parent, first.index) and first.graphs.num_graphs == n + len(extra)
    assert torch.equal(first.move_from[:n], plain.rollout[0].move_from)
    row = n
    for i, s in enumerate(samples):
        for m in s.siblings:
            assert int(first.move_from[row]) == i * 64 + m.from_square and int(first.move_to[row]) == i * 64 + m.to_square
            assert int(first.piece[row]) == m.piece and int(first.promotion[row]) == m.promotion
            row += 1
    for with_siblings, without in zip(batch.rollout[1:], plain.rollout[1:]):
        assert torch.equal(with_siblings.parent, without.parent) and torch.equal(with_siblings.index, without.index)


def test_legal_moves_are_listed_in_the_graphs_frame():
    boards = random_positions(12, seed=31, per_start=6)
    boards += [chess.Board(f, chess960=True) for f in (
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",              # castling both ways, black to move
        "8/4P3/8/8/8/8/k7/7K w - - 0 1",                      # four promotions along one edge
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",   # en passant
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")]                    # stalemate: no legal move
    samples = [ld.build_sample(ld.fen_of(b), random.Random(i), rollout_steps=2) for i, b in enumerate(boards)]
    for board, sample in zip(boards, samples):
        board = chess.Board(ld.fen_of(board), chess960=True)
        orient = (lambda sq: sq) if board.turn == chess.WHITE else (lambda sq: sq ^ 56)
        pairs = {(orient(m.from_square), orient(m.to_square)) for m in board.legal_moves}
        assert sample.legal == sorted(a * 64 + b for a, b in pairs)
        # every legal move runs along an edge of the position's graph, from a square holding the mover's piece
        edges = set(zip(sample.full.edge_index[0].tolist(), sample.full.edge_index[1].tolist()))
        assert pairs <= edges
        assert all(float(sample.full.x[a, 1:7].sum()) == 1 and float(sample.full.x[a, 0]) == 0 for a, _ in pairs)
    assert samples[-1].legal == [] and samples[-1].rollout == []
    assert len(samples[-3].legal) == 1 + 3, "one pawn move (its four promotions count once) and three king moves"
    assert any(b.has_castling_rights(b.turn) and any(b.is_castling(m) for m in b.legal_moves) for b in boards)

    batch = ld.collate_jepa(samples)
    assert batch.legal.tolist() == [i * 4096 + code for i, s in enumerate(samples) for code in s.legal]
    assert torch.equal(batch.pin_memory().legal, batch.legal) and torch.equal(batch.to("cpu").legal, batch.legal)

    # the positions the moves lead to list theirs too, each in its own frame (its side to move at the bottom)
    listed = 0
    for sample in samples:
        for step in sample.rollout + sample.siblings:
            after = chess.Board(step.next_fen, chess960=True)
            assert step.legal == ld.legal_codes(after) and step.view is not None
            turn = (lambda sq: sq) if after.turn == chess.WHITE else (lambda sq: sq ^ 56)
            assert step.legal == sorted({turn(m.from_square) * 64 + turn(m.to_square) for m in after.legal_moves})
            edges = set(zip(step.view.edge_index[0].tolist(), step.view.edge_index[1].tolist()))
            assert {(code // 64, code % 64) for code in step.legal} <= edges
            listed += len(step.legal)
    assert listed > 0
    for k, step in enumerate(batch.rollout):
        moves = [s.rollout[k] for s in samples if len(s.rollout) > k]
        assert step.legal.tolist() == [row * 4096 + code for row, m in enumerate(moves) for code in m.legal]
        assert int(step.legal.max()) // 4096 < step.graphs.num_graphs
        assert torch.equal(step.pin_memory().legal, step.legal) and torch.equal(step.to("cpu").legal, step.legal)
    # steps that are not targets list nothing, and the moves drawn are the same either way
    for i, board in enumerate(boards):
        every = ld.build_sample(ld.fen_of(board), random.Random(i), rollout_steps=3)
        some = ld.build_sample(ld.fen_of(board), random.Random(i), rollout_steps=3, target_steps=[2], siblings=3)
        assert [m.move for m in every.rollout] == [m.move for m in some.rollout]
        assert all((m.legal == [] and m.view is None) == (k != 2) or not ld.legal_codes(chess.Board(m.next_fen, chess960=True))
                   for k, m in enumerate(some.rollout, start=1))
        assert all(a.legal == b.legal for k, (a, b) in enumerate(zip(every.rollout, some.rollout), start=1) if k == 2)
        assert all(m.view is None and m.legal == [] for m in some.siblings)
    skipped = ld.collate_jepa([ld.build_sample(ld.fen_of(b), random.Random(i), rollout_steps=2, target_steps=[2])
                               for i, b in enumerate(boards[:4])])
    assert skipped.rollout[0].legal is None and skipped.rollout[0].graphs is None and skipped.rollout[1].legal is not None


def test_loader_rollout_length_and_mid_epoch_resume():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "fens.txt")
        with open(path, "w") as f:
            f.write("\n".join(ld.fen_of(b) for b in random_positions(30, seed=23)))
        cache = os.path.join(tmp, "cache")
        ld.build_position_cache_from_file(path, cache, 0.0, 0.0, verbose=False)
        loader = ld.JEPALoader(cache, "train", batch_size=10, num_workers=0, seed=4)
        loader.set_epoch(2)
        full = [f for b in loader for f in b.fens]
        loader.set_epoch(2)
        loader.set_rollout_steps(2)
        first = []
        for b in loader:
            first += b.fens
            assert len(b.rollout) <= 2 and len(b.rollout) >= 1
            if len(first) >= 40:
                break
        loader.resume_at(40)
        rest = [f for b in loader for f in b.fens]
        assert first[:40] + rest == full, "resuming continues the epoch without repeats or gaps"


def test_threat_labels_are_turned_into_the_graphs_frame():
    """
    The labels go by the board's real squares; a graph shows the side to move at the bottom. Checked against
    what the pipeline already has in the graph's frame: after 0 plies the side to move's threats are its legal
    moves (legal_codes), and every labelled piece square holds a piece of the graph.
    """
    none = torch.zeros(64, dtype=torch.bool)
    for fen in ("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2P1PN2/PP1N1PPP/R2QKB1R b - - 3 8",       # black to move
                "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2P1PN2/PP1NBPPP/R2Q1RK1 w - - 4 9",       # white to move
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1"):
        fen = ld.normalize_fen(fen)
        board = chess.Board(fen, chess960=True)
        white = board.turn == chess.WHITE
        counts = pr.threat_counts(fen, 1)
        codes, logs = ld.threat_cells(counts, white)
        ply, piece, target = codes // 4096, codes // 64 % 64, codes % 64
        assert codes.dtype == np.int32 and logs.dtype == np.float32 and len(codes) == int((counts > 0).sum())
        turn = 0 if white else 56
        assert np.allclose(np.expm1(logs.astype(np.float64)), counts[ply, piece ^ turn, target ^ turn]), "each cell keeps its count"
        x = ld.build_views(fen, none)[0].x
        assert bool((x[torch.as_tensor(piece), 1:7].sum(1) > 0).all()), "every labelled row is a piece of the graph"
        mine = (ply == 0) & (x[torch.as_tensor(piece), 0] == 0).numpy()        # (first feature: the piece is the other side's)
        assert sorted((piece[mine] * 64 + target[mine]).tolist()) == ld.legal_codes(board), \
            "the side to move's threats now are its legal moves, in the graph's squares"
        assert int((ply == 0).sum()) > int(mine.sum()), "the other side's pieces are asked too"
        # the two frames differ exactly by turning the board over, and only for black to move
        plain = ld.threat_cells(counts, True)[0]
        assert white == bool(np.array_equal(plain, codes))
        assert sorted(((plain // 4096 * 64 + (plain // 64 % 64 ^ turn)) * 64 + (plain % 64 ^ turn)).tolist()) == sorted(codes.tolist())


def test_threat_labels_come_from_the_database_or_the_search():
    fens = [ld.normalize_fen(b.fen()) for b in random_positions(3, seed=5, per_start=4)]
    white = [f.split()[1] == "w" for f in fens]
    sources = []

    def source(*args, **kw):
        sources.append(ld.ThreatLabels(*args, **kw))
        return sources[-1]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = os.path.join(tmp, "labels.db")
        conn = rl.open_db(db)
        rl.write_labels(conn, pr.label_threats((fens[:6], 2)), "threats")
        conn.executemany("INSERT INTO reach_progress (walk, done) VALUES (?, ?)", [
            (rl.walk_name("train", 0, 0, 2, 2, "threats"), 123), (rl.walk_name("train", 0, 0, 0, 1, "threats"), 999),
            (rl.walk_name("train", 0, 0, 2, 3, "reach"), 5000), (rl.walk_name("val", 0, 0, 2, 2, "threats"), 7)])
        conn.commit()
        conn.close()
        was = pr.USING_C
        try:
            pr.use_compiled(False)                      # nothing can be searched on the spot now
            stored = source(2, db)
            assert not stored.searches and stored.layers == 3
            # how far the labeller's walks that fit this source have got
            assert stored.stored_samples("train", 0) == 123 and stored.stored_samples("val", 0) == 7
            assert stored.stored_samples("train", 1) == 0 and stored.stored_samples("train", 0, epoch=1) == 0
            assert source(1, db, rollout_steps=0).stored_samples("train", 0) == 999, "a walk without rollout positions will do"
            assert source(1, db).stored_samples("train", 0) == 123 and source(3, db).stored_samples("train", 0) == 0
            got = stored.of(fens[0], white[0])
            want = ld.threat_cells(pr.threat_counts(fens[0], 2), white[0])
            assert got.origin == ld.THREATS_STORED and got.layers == 3
            assert np.array_equal(got.codes, want[0]) and np.array_equal(got.logs, want[1])
            shallow = source(1, db).of(fens[0], white[0])              # a deeper label holds the shallower one
            assert shallow.origin == ld.THREATS_STORED and shallow.layers == 2
            assert np.array_equal(shallow.codes, want[0][want[0] < 2 * 4096])
            for nothing in (stored.of(fens[7], white[7]),                            # not in the database
                            source(3, db).of(fens[0], white[0]),                     # only labelled to 2 plies
                            source(2, None).of(fens[0], white[0]),                   # no database at all
                            source(2, os.path.join(tmp, "absent.db")).of(fens[0], white[0])):
                assert nothing.origin == ld.THREATS_NONE and len(nothing.codes) == 0 and len(nothing.logs) == 0
            copy = pickle.loads(pickle.dumps(stored))                   # a loader worker's copy opens its own
            sources.append(copy)
            assert copy._conn is None and copy.of(fens[1], white[1]).origin == ld.THREATS_STORED
        finally:
            pr.use_compiled(was)
        if pr.USING_C:                                  # with the compiled search, the rest is searched on the spot
            searched = source(2, db).of(fens[7], white[7])
            want = ld.threat_cells(pr.threat_counts(fens[7], 2, use_c=False), white[7])
            assert searched.origin == ld.THREATS_COMPUTED and np.array_equal(searched.codes, want[0])
            assert source(2, db).of(fens[0], white[0]).origin == ld.THREATS_STORED, "the database comes first"
            assert source(2, db, compute=False).of(fens[7], white[7]).origin == ld.THREATS_NONE
        for s in sources:
            s.close()
    for bad in (dict(plies=-1), dict(plies=1, rollout_steps=-1)):
        try:
            ld.ThreatLabels(**bad)
            raise AssertionError(f"{bad} should be rejected")
        except ValueError:
            pass


def test_samples_and_batches_carry_threat_labels():
    fens = [ld.normalize_fen(b.fen()) for b in random_positions(4, seed=9, per_start=3)][::3]
    source = SearchedThreats(1, rollout_steps=2)
    samples = [ld.build_sample(f, random.Random(i), rollout_steps=3, threats=source) for i, f in enumerate(fens)]
    plain = [ld.build_sample(f, random.Random(i), rollout_steps=3) for i, f in enumerate(fens)]
    for s, p in zip(samples, plain):
        assert p.threats is None and all(m.threats is None for m in p.rollout)
        assert torch.equal(s.mask, p.mask) and [m.move for m in s.rollout] == [m.move for m in p.rollout], \
            "labels change nothing in the random draws"
        board = chess.Board(s.fen, chess960=True)
        want = ld.threat_cells(pr.threat_counts(s.fen, 1), board.turn == chess.WHITE)
        assert s.threats.layers == 2 and np.array_equal(s.threats.codes, want[0]) and np.array_equal(s.threats.logs, want[1])
        for k, m in enumerate(s.rollout, start=1):
            board.push_uci(m.move)
            if k > 2:
                assert m.threats is None, "only the first rollout_steps positions are labelled"
                continue
            want = ld.threat_cells(pr.threat_counts(ld.fen_of(board), 1), board.turn == chess.WHITE)
            assert np.array_equal(m.threats.codes, want[0]), "a rollout position's labels are in its own frame"

    batch = ld.collate_jepa(samples)
    dense = batch.threats.dense()
    assert dense.shape == (len(fens), 2, 64, 64) and batch.threats.layers == 2
    assert batch.threats.origin.tolist() == [ld.THREATS_COMPUTED] * len(fens)
    for b, s in enumerate(samples):
        want = torch.zeros(2 * 4096)
        want[torch.as_tensor(s.threats.codes, dtype=torch.long)] = torch.as_tensor(s.threats.logs)
        assert torch.equal(dense[b].flatten(), want)
    assert [step.threats is not None for step in batch.rollout] == [True, True, False]
    for k, step in enumerate(batch.rollout[:2]):
        rows = step.threats.dense()
        assert rows.shape[0] == step.index.numel()
        for row, b in enumerate(step.index.tolist()):
            m = samples[b].rollout[k]
            assert int((rows[row] > 0).sum()) == len(m.threats.codes)
            assert torch.equal(rows[row].flatten().nonzero().flatten(), torch.as_tensor(np.sort(m.threats.codes), dtype=torch.long))
    moved = batch.to("cpu")
    assert torch.equal(moved.threats.dense(), dense) and moved.rollout[0].threats.layers == 2
    assert ld.collate_jepa(plain).threats is None and all(step.threats is None for step in ld.collate_jepa(plain).rollout)

    # a position without a label: its row of the batch is empty and says so
    class Half(SearchedThreats):
        def counts(self, fen):
            return (None, ld.THREATS_NONE) if fen == fens[1] else super().counts(fen)

    mixed = ld.collate_jepa([ld.build_sample(f, random.Random(i), threats=Half(1, rollout_steps=0)) for i, f in enumerate(fens)])
    assert mixed.threats.origin.tolist() == [ld.THREATS_COMPUTED, ld.THREATS_NONE] + [ld.THREATS_COMPUTED] * (len(fens) - 2)
    assert not mixed.threats.dense()[1].any() and mixed.threats.dense()[0].any() and mixed.rollout[0].threats is None

    # labels only where a step is a target, and for sibling moves as for the first step
    sparse = ld.build_sample(fens[0], random.Random(0), rollout_steps=3, target_steps=[2], threats=source)
    assert [m.threats is not None for m in sparse.rollout] == [False, True, False]
    family = ld.build_sample(fens[0], random.Random(0), siblings=3, threats=source)
    assert family.siblings and all(m.threats is not None for m in family.siblings)


def test_loader_hands_out_threat_labels_with_workers():
    boards = random_positions(6, seed=3)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "fens.txt")
        with open(path, "w") as f:
            f.write("\n".join(b.fen() for b in boards))
        cache = os.path.join(tmp, "cache")
        ld.build_position_cache_from_file(path, cache, val_fraction=0.0, test_fraction=0.0, verbose=False)
        seen = []
        for workers in (0, 2):
            loader = ld.JEPALoader(cache, "train", 8, num_workers=workers, seed=3, threats=SearchedThreats(1, rollout_steps=1))
            loader.set_rollout_steps(2)
            batch = next(iter(loader))
            assert batch.threats is not None and batch.threats.dense().shape == (8, 2, 64, 64)
            assert batch.rollout[0].threats is not None and batch.rollout[1].threats is None
            seen.append((batch.fens, batch.threats.index, batch.threats.value, batch.rollout[0].threats.index))
        assert seen[0][0] == seen[1][0] and all(torch.equal(a, b) for a, b in zip(seen[0][1:], seen[1][1:])), \
            "the same labels whatever the number of workers"
        assert next(iter(ld.JEPALoader(cache, "train", 8, num_workers=0, seed=3))).threats is None


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
