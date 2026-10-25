"""
Training data for the Labyrinth encoder: positions from any source, split by hash, prepared in
DataLoader worker processes so the GPU never waits.

    build_position_cache_from_fens(fens, cache_dir, labels=None)
        The generic step: normalizes every FEN through python-chess, drops invalid and duplicate
        positions, assigns each to train / val / test by a hash of the position, and writes one
        memory-mappable file per split, plus engine labels when the source has them (for the probe
        benchmark only, never for training). Two sources feed it:
          build_position_cache(db_path, cache_dir)        minotaur_data.db, with its engine labels
          build_position_cache_from_file(path, cache_dir) a text file with one FEN per line
        (python labyrinth_data.py build --db ... / --fens ...). Because the split depends only on the
        position, a position lands in the same split whichever source it comes from, so positions held
        out for evaluation in one dataset stay held out of training in another.

    PositionDataset(cache_dir, split, ...)
        One item per position: a JEPASample with
          * the full board's graph (what the target encoder sees),
          * a masked view of it (what the context encoder sees for masked-square prediction),
          * one random legal move and the next position's graph (for next-position prediction).
        Every random choice comes from a generator seeded by (seed, epoch, index), so a run is
        reproducible whatever the number of workers.

    collate_jepa(samples) -> JEPABatch
        Packs samples into PyTorch Geometric style batches (one disconnected graph per view).

    JEPALoader(...)
        A DataLoader wrapper with set_epoch(), so each epoch draws new masks and moves.

Conventions shared with chess_graph.py: every graph shows the side to move at the bottom, node
features are [hostile, pawn, knight, bishop, rook, queen, king, en passant], and castling rook
squares come from python-chess, for the position and its successor alike, so both graphs agree on
which rook castles.

Masking never leaks the hidden squares. The masked view is built from the input, before any model
layer sees it:
  * masked squares' node features are zeroed (the model adds its own mask token on top);
  * a castling right whose king or rook stands on a masked square loses both of its edges, since
    those edges exist only because of what stands on those squares;
  * the en passant flag is cleared when the square, the pawn that just moved two squares, or a pawn
    that could capture it is masked. Positions are written in X-FEN (see normalize_fen), where the flag
    appears exactly when a pawn could capture, so it depends only on those squares.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import chess_graph as cg

SPLITS = ("train", "val", "test")
DEFAULT_DB = "minotaur_data.db"
DEFAULT_CACHE_DIR = "labyrinth_data_cache"
TABLE = "960_position_data"


# ##### ##### ##### ##### #####
#       Splits


def position_key(fen: str) -> str:
    """The position part of a FEN (board, side to move, castling, en passant), without the move counters."""
    return " ".join(fen.split()[:4])


def split_of(fen: str, val_fraction: float = 0.01, test_fraction: float = 0.01) -> str:
    """
    "train", "val" or "test", decided by a hash of the position (move counters ignored). The same
    position always lands in the same split, however the database is ordered or grows.
    """
    digest = hashlib.blake2b(position_key(fen).encode(), digest_size=8).digest()
    u = int.from_bytes(digest, "big") / 2 ** 64
    if u < test_fraction:
        return "test"
    if u < test_fraction + val_fraction:
        return "val"
    return "train"


def fen_of(board: chess.Board) -> str:
    """The FEN of a board in the convention every graph uses (see normalize_fen)."""
    return board.fen(en_passant="xfen")


def normalize_fen(fen: str) -> Optional[str]:
    """
    The FEN as python-chess writes it in X-FEN (Chess960 castling letters, and the en passant square
    whenever a pawn could capture there, pins ignored), or None when python-chess cannot read it or the
    position is not valid. The move sampler writes positions the same way, so every graph follows one
    convention. X-FEN rather than python-chess's default matters for masking: the default writes the en
    passant square only when the capture is fully legal, which depends on pins anywhere on the board, so
    the flag could reveal masked squares; under X-FEN it depends only on the squares the masking rule covers.
    """
    try:
        board = chess.Board(fen, chess960=True)
    except ValueError:
        return None
    if not board.is_valid():
        return None
    return fen_of(board)


# ##### ##### ##### ##### #####
#       The position cache


class LabeledPositions(NamedTuple):
    """Engine labels of the analyzed positions in one split (numpy arrays, aligned)."""
    fen: np.ndarray                  # normalized FEN
    engine_name: np.ndarray
    depth: np.ndarray                # int
    score: np.ndarray                # as stored: centipawns or "#+N" / "#-N", from WHITE's point of view
    is_forced_checkmate: np.ndarray  # bool
    best_move: np.ndarray            # UCI


class EngineLabel(NamedTuple):
    """One position's engine analysis, as the database stores it."""
    engine_name: str
    depth: int
    score: str                 # centipawns or "#+N" / "#-N", from WHITE's point of view
    is_forced_checkmate: bool
    best_move: str             # UCI


def build_position_cache_from_fens(
    fens: Iterable[str],
    cache_dir: str = DEFAULT_CACHE_DIR,
    labels: Optional[Iterable[Optional[EngineLabel]]] = None,
    val_fraction: float = 0.01,
    test_fraction: float = 0.01,
    source: str = "",
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Normalize every FEN, drop invalid positions and duplicates (the same position with different move
    counters counts once; the first one seen is kept), split by hash, and write:
        positions_<split>.npy   fixed-width byte strings, opened memory-mapped by PositionDataset
        labels_<split>.npz      engine labels of the labeled positions in that split (may be empty)
        meta.json               the source, the fractions used and the counts
    labels: one EngineLabel or None per FEN, in the same order, when the source has engine analysis.
    Returns the contents of meta.json.
    """
    t0 = time.time()
    seen: set = set()
    positions: Dict[str, List[str]] = {s: [] for s in SPLITS}
    label_cols: Dict[str, Dict[str, list]] = {s: {k: [] for k in LabeledPositions._fields} for s in SPLITS}
    read = invalid = duplicates = 0
    label_iter = iter(labels) if labels is not None else None
    for fen in fens:
        read += 1
        label = next(label_iter) if label_iter is not None else None
        norm = normalize_fen(fen.strip())
        if norm is None:
            invalid += 1
            continue
        key = position_key(norm)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        split = split_of(norm, val_fraction, test_fraction)
        positions[split].append(norm)
        if label is not None:
            col = label_cols[split]
            col["fen"].append(norm)
            col["engine_name"].append(label.engine_name)
            col["depth"].append(label.depth)
            col["score"].append(label.score)
            col["is_forced_checkmate"].append(label.is_forced_checkmate)
            col["best_move"].append(label.best_move)
    labels = label_cols

    os.makedirs(cache_dir, exist_ok=True)
    for split in SPLITS:
        np.save(os.path.join(cache_dir, f"positions_{split}.npy"), np.array(positions[split], dtype=np.bytes_))
        lab = labels[split]
        np.savez(
            os.path.join(cache_dir, f"labels_{split}.npz"),
            fen=np.array(lab["fen"], dtype=np.bytes_),
            engine_name=np.array(lab["engine_name"], dtype=np.bytes_),
            depth=np.array(lab["depth"], dtype=np.int32),
            score=np.array(lab["score"], dtype=np.bytes_),
            is_forced_checkmate=np.array(lab["is_forced_checkmate"], dtype=bool),
            best_move=np.array(lab["best_move"], dtype=np.bytes_),
        )
    meta = {
        "source": source,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "rows_read": read,
        "invalid": invalid,
        "duplicates": duplicates,
        "positions": {s: len(positions[s]) for s in SPLITS},
        "labeled": {s: len(labels[s]["fen"]) for s in SPLITS},
        "built_unix": time.time(),
    }
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"done in {time.time() - t0:.1f}s: {json.dumps(meta['positions'])} positions, "
              f"{json.dumps(meta['labeled'])} labeled, {invalid:,} invalid, {duplicates:,} duplicates")
    return meta


def build_position_cache(
    db_path: str = DEFAULT_DB,
    cache_dir: str = DEFAULT_CACHE_DIR,
    val_fraction: float = 0.01,
    test_fraction: float = 0.01,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """The position cache of minotaur_data.db, with the engine labels of its analyzed rows (read first,
    so a duplicate keeps its label)."""
    t0 = time.time()
    uri = "file:" + os.path.abspath(db_path).replace("\\", "/") + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        query = (f'SELECT fen, is_analyzed, engine_name, depth, score, is_forced_checkmate, best_move '
                 f'FROM "{TABLE}" ORDER BY is_analyzed DESC')
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        rows = conn.execute(query).fetchall()
    if verbose:
        print(f"read {len(rows):,} rows in {time.time() - t0:.1f}s; normalizing...")
    labels = [EngineLabel(engine_name or "", int(depth or 0), str(score), bool(is_mate), best_move or "")
              if is_analyzed else None
              for _, is_analyzed, engine_name, depth, score, is_mate, best_move in rows]
    return build_position_cache_from_fens((r[0] for r in rows), cache_dir, labels, val_fraction, test_fraction,
                                          source=os.path.abspath(db_path), verbose=verbose)


def build_position_cache_from_file(
    path: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    val_fraction: float = 0.01,
    test_fraction: float = 0.01,
    verbose: bool = True,
) -> Dict[str, object]:
    """The position cache of a text file with one FEN per line (blank lines and lines starting with # are skipped)."""
    def lines() -> Iterator[str]:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    yield line
    return build_position_cache_from_fens(lines(), cache_dir, None, val_fraction, test_fraction,
                                          source=os.path.abspath(path), verbose=verbose)


def load_labels(cache_dir: str, split: str) -> LabeledPositions:
    """The engine labels of the analyzed positions in a split, as decoded numpy arrays."""
    with np.load(os.path.join(cache_dir, f"labels_{split}.npz")) as data:
        def text(name: str) -> np.ndarray:
            return np.char.decode(data[name], "utf-8")
        return LabeledPositions(
            fen=text("fen"), engine_name=text("engine_name"), depth=data["depth"].copy(),
            score=text("score"), is_forced_checkmate=data["is_forced_checkmate"].copy(),
            best_move=text("best_move"),
        )


# ##### ##### ##### ##### #####
#       Masks and views


@dataclass
class MaskConfig:
    """
    Masks are unions of square blocks: a random square and every square within a random Chebyshev
    radius of it (radius 0 is one square, 1 a 3x3 block, 2 a 5x5 block), added until the mask
    covers a random fraction of the board between min_ratio and max_ratio.
    """
    min_ratio: float = 0.15
    max_ratio: float = 0.40
    max_block_radius: int = 1


def sample_mask(rng: random.Random, config: MaskConfig) -> torch.Tensor:
    """A [64] bool mask over the squares of the oriented board."""
    target = max(1, round(rng.uniform(config.min_ratio, config.max_ratio) * 64))
    masked: set = set()
    while len(masked) < target:
        centre = rng.randrange(64)
        r = rng.randint(0, config.max_block_radius)
        cr, cf = divmod(centre, 8)
        block = [rr * 8 + ff for rr in range(max(0, cr - r), min(8, cr + r + 1))
                 for ff in range(max(0, cf - r), min(8, cf + r + 1))]
        rng.shuffle(block)
        for sq in block:
            if len(masked) >= target:
                break
            masked.add(sq)
    mask = torch.zeros(64, dtype=torch.bool)
    mask[list(masked)] = True
    return mask


def _orient(board: chess.Board):
    return (lambda sq: sq) if board.turn == chess.WHITE else (lambda sq: sq ^ 56)


def masked_castling_rook_squares(board: chess.Board, rook_squares: Tuple[int, int, int, int],
                                 mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """The rook squares with every right removed whose king or rook stands on a masked square."""
    orient = _orient(board)
    out = list(rook_squares)
    for slot, color in ((0, chess.WHITE), (1, chess.WHITE), (2, chess.BLACK), (3, chess.BLACK)):
        rook = out[slot]
        king = board.king(color)
        if rook >= 0 and king is not None and (mask[orient(rook)] or mask[orient(king)]):
            out[slot] = -1
    return out[0], out[1], out[2], out[3]


def _clear_hidden_en_passant(x: torch.Tensor, mask: torch.Tensor) -> None:
    """Clear the en passant flag when its square, the pawn that moved, or a possible capturer is masked."""
    flagged = (x[:, 7] > 0).nonzero().flatten().tolist()
    for ep in flagged:
        pawn = ep - 8                                  # the opponent's pawn, one rank below on the oriented board
        related = [ep, pawn] + [pawn + d for d in (-1, 1) if 0 <= (pawn % 8) + d < 8]
        if any(bool(mask[s]) for s in related if 0 <= s < 64):
            x[ep, 7] = 0


# ##### ##### ##### ##### #####
#       Samples


@dataclass
class GraphView:
    """
    One chess graph as the models read it (chess_graph.to_single_graph): x [64, F], edge_index [2, E]
    with one edge per square pair, edge_attr [E, NUM_EDGE_TYPES] the multi-hot of its relation types.
    """
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor

    @classmethod
    def from_edge_lists(cls, edges: Sequence[torch.Tensor], x: torch.Tensor) -> "GraphView":
        return cls(*cg.to_single_graph(edges, x))


@dataclass
class NextPosition:
    """The random legal move and the position it leads to."""
    move: str
    next_fen: str
    from_square: int       # node index in the current position's graph
    to_square: int         # node index in the current position's graph (the rook's square for castling)
    promotion: int         # 0, or the python-chess piece type promoted to
    flags: Tuple[bool, bool, bool]   # (is_castling, is_capture, is_en_passant)
    view: GraphView        # the next position's graph


@dataclass
class JEPASample:
    fen: str
    full: GraphView        # the whole board: what the target encoder sees
    context: GraphView     # the masked board: what the context encoder sees for masked-square prediction
    mask: torch.Tensor     # [64] bool, True on the masked squares
    next: Optional[NextPosition]   # None when the position has no legal move (mate or stalemate)


def build_views(fen: str, mask: torch.Tensor) -> Tuple[GraphView, GraphView, chess.Board, Tuple[int, int, int, int]]:
    """
    The full graph of a position and its masked view (see the module docstring for the masking rules).
    Returns (full, context, board, castling rook squares).
    """
    board = chess.Board(fen, chess960=True)
    rook_squares = cg.castling_rook_squares_from_board(board)
    edges, x = cg.create_filled_chess_graphs(fen, rook_squares)
    full = GraphView.from_edge_lists(edges, x)

    context_x = x.clone()
    context_x[mask] = 0
    _clear_hidden_en_passant(context_x, mask)
    ctx_rooks = masked_castling_rook_squares(board, rook_squares, mask)
    context_edges = edges if ctx_rooks == rook_squares else cg.create_filled_chess_graphs(fen, ctx_rooks)[0]
    return full, GraphView.from_edge_lists(context_edges, context_x), board, rook_squares


def build_sample(fen: str, rng: random.Random, mask_config: MaskConfig = MaskConfig(),
                 mask: Optional[torch.Tensor] = None) -> JEPASample:
    """
    Everything one position contributes to a training step. fen should be normalized (normalize_fen);
    the cache already is. mask: use this mask instead of sampling one.
    """
    if mask is None:
        mask = sample_mask(rng, mask_config)
    full, context, board, rook_squares = build_views(fen, mask)

    nxt = None
    moves = list(board.legal_moves)
    if moves:
        t = cg.move_transition(board, rng.choice(moves), rook_squares)
        next_edges, next_x = cg.create_filled_chess_graphs(t.next_fen, t.next_castling_rook_squares)
        nxt = NextPosition(t.move, t.next_fen, t.from_square, t.to_square, t.promotion,
                           (t.is_castling, t.is_capture, t.is_en_passant),
                           GraphView.from_edge_lists(next_edges, next_x))
    return JEPASample(fen, full, context, mask, nxt)


# ##### ##### ##### ##### #####
#       Batches


@dataclass
class GraphBatch:
    """Several chess graphs as one disconnected graph, the PyTorch Geometric way."""
    x: torch.Tensor            # [B*64, F]
    edge_index: torch.Tensor   # [2, E]
    edge_attr: torch.Tensor    # [E, NUM_EDGE_TYPES] multi-hot relation types, one edge per square pair
    batch: torch.Tensor        # [B*64] which graph each node belongs to
    num_graphs: int

    def to(self, device, non_blocking: bool = False) -> "GraphBatch":
        return GraphBatch(self.x.to(device, non_blocking=non_blocking),
                          self.edge_index.to(device, non_blocking=non_blocking),
                          self.edge_attr.to(device, non_blocking=non_blocking),
                          self.batch.to(device, non_blocking=non_blocking), self.num_graphs)


def pack_views(views: Sequence[GraphView]) -> GraphBatch:
    n = views[0].x.size(0)
    offsets = torch.arange(len(views)) * n
    x = torch.cat([v.x for v in views])
    edge_index = torch.cat([v.edge_index + off for v, off in zip(views, offsets.tolist())], dim=1)
    edge_attr = torch.cat([v.edge_attr for v in views])
    batch = torch.arange(len(views)).repeat_interleave(n)
    return GraphBatch(x, edge_index, edge_attr, batch, len(views))


@dataclass
class JEPABatch:
    """
    A training batch of B positions.
      full, context     GraphBatch of all B positions (whole board / masked board)
      mask              [B*64] bool, aligned with the nodes of full and context
      next              GraphBatch of the next positions, for the B_next positions that have a legal move
      next_index        [B_next] which of the B positions each next position belongs to
      move_from, move_to  [B_next] the move's squares as node indices into full / context (b*64 + square)
      promotion         [B_next] 0, or the piece type promoted to
      move_flags        [B_next, 3] (is_castling, is_capture, is_en_passant) as floats
      fens, moves       for inspection
    """
    full: GraphBatch
    context: GraphBatch
    mask: torch.Tensor
    next: Optional[GraphBatch]
    next_index: torch.Tensor
    move_from: torch.Tensor
    move_to: torch.Tensor
    promotion: torch.Tensor
    move_flags: torch.Tensor
    fens: List[str] = field(default_factory=list)
    moves: List[str] = field(default_factory=list)

    @property
    def num_graphs(self) -> int:
        return self.full.num_graphs

    def to(self, device, non_blocking: bool = False) -> "JEPABatch":
        def mv(t: torch.Tensor) -> torch.Tensor:
            return t.to(device, non_blocking=non_blocking)
        return JEPABatch(self.full.to(device, non_blocking), self.context.to(device, non_blocking), mv(self.mask),
                         None if self.next is None else self.next.to(device, non_blocking),
                         mv(self.next_index), mv(self.move_from), mv(self.move_to), mv(self.promotion),
                         mv(self.move_flags), self.fens, self.moves)

    def pin_memory(self) -> "JEPABatch":
        def pin(t: torch.Tensor) -> torch.Tensor:
            return t.pin_memory()

        def pin_graphs(g: Optional[GraphBatch]) -> Optional[GraphBatch]:
            return None if g is None else GraphBatch(pin(g.x), pin(g.edge_index), pin(g.edge_attr), pin(g.batch),
                                                     g.num_graphs)
        return JEPABatch(pin_graphs(self.full), pin_graphs(self.context), pin(self.mask), pin_graphs(self.next),
                         pin(self.next_index), pin(self.move_from), pin(self.move_to), pin(self.promotion),
                         pin(self.move_flags), self.fens, self.moves)


def collate_jepa(samples: Sequence[JEPASample]) -> JEPABatch:
    with_next = [i for i, s in enumerate(samples) if s.next is not None]
    nexts = [samples[i].next for i in with_next]
    index = torch.tensor(with_next, dtype=torch.long)
    return JEPABatch(
        full=pack_views([s.full for s in samples]),
        context=pack_views([s.context for s in samples]),
        mask=torch.cat([s.mask for s in samples]),
        next=pack_views([n.view for n in nexts]) if nexts else None,
        next_index=index,
        move_from=index * 64 + torch.tensor([n.from_square for n in nexts], dtype=torch.long),
        move_to=index * 64 + torch.tensor([n.to_square for n in nexts], dtype=torch.long),
        promotion=torch.tensor([n.promotion for n in nexts], dtype=torch.long),
        move_flags=torch.tensor([n.flags for n in nexts], dtype=torch.float32).reshape(-1, 3),
        fens=[s.fen for s in samples],
        moves=[n.move for n in nexts],
    )


# ##### ##### ##### ##### #####
#       Dataset and loader


class PositionDataset(Dataset):
    """
    The positions of one split, each turned into a JEPASample on demand. The positions file is opened
    memory-mapped in each worker, so workers share it through the OS page cache instead of copying it.
    indices: use only these positions of the split (e.g. a subset for small runs).
    """

    def __init__(self, cache_dir: str, split: str, mask_config: MaskConfig = MaskConfig(), seed: int = 0,
                 indices: Optional[Sequence[int]] = None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.path = os.path.join(cache_dir, f"positions_{split}.npy")
        self.mask_config = mask_config
        self.seed = int(seed)
        self.epoch = 0
        self._positions: Optional[np.ndarray] = None
        total = len(np.load(self.path, mmap_mode="r"))
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        self._len = total if self.indices is None else len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_positions"] = None        # reopened in each worker, never pickled
        return state

    @property
    def positions(self) -> np.ndarray:
        if self._positions is None:
            self._positions = np.load(self.path, mmap_mode="r")
        return self._positions

    def __len__(self) -> int:
        return self._len

    def fen(self, i: int) -> str:
        j = int(self.indices[i]) if self.indices is not None else i
        return self.positions[j].decode("utf-8")

    def rng_for(self, i: int) -> random.Random:
        return random.Random((self.seed * 1_000_003 + self.epoch) * 10_000_019 + i)

    def __getitem__(self, i: int) -> JEPASample:
        return build_sample(self.fen(i), self.rng_for(i), self.mask_config)


class _EpochShuffle(Sampler):
    """A shuffled order that depends only on (seed, epoch)."""

    def __init__(self, dataset: PositionDataset, shuffle: bool):
        self.dataset, self.shuffle = dataset, shuffle

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            return iter(range(len(self.dataset)))
        g = torch.Generator().manual_seed(self.dataset.seed * 7919 + self.dataset.epoch)
        return iter(torch.randperm(len(self.dataset), generator=g).tolist())


class JEPALoader:
    """
    Batches of JEPASamples from one split, prepared by worker processes. Call set_epoch(e) before
    each epoch: the order, masks and moves then depend only on (seed, epoch).
    """

    def __init__(self, cache_dir: str, split: str, batch_size: int, *, num_workers: int = 4,
                 mask_config: MaskConfig = MaskConfig(), seed: int = 0, shuffle: bool = True,
                 drop_last: bool = False, indices: Optional[Sequence[int]] = None, pin_memory: bool = False,
                 prefetch_factor: int = 4):
        self.dataset = PositionDataset(cache_dir, split, mask_config, seed, indices)
        self.sampler = _EpochShuffle(self.dataset, shuffle)
        self.loader_kwargs = dict(batch_size=batch_size, sampler=self.sampler, num_workers=num_workers,
                                  collate_fn=collate_jepa, drop_last=drop_last, pin_memory=pin_memory)
        if num_workers > 0:
            self.loader_kwargs["prefetch_factor"] = prefetch_factor

    def set_epoch(self, epoch: int) -> None:
        self.dataset.epoch = int(epoch)

    def __len__(self) -> int:
        bs = self.loader_kwargs["batch_size"]
        n = len(self.dataset)
        return n // bs if self.loader_kwargs["drop_last"] else -(-n // bs)

    def __iter__(self) -> Iterator[JEPABatch]:
        # A fresh DataLoader per epoch, so the workers see the current epoch
        return iter(DataLoader(self.dataset, **self.loader_kwargs))


# ##### ##### ##### ##### #####
#       Command line: build the cache, measure throughput


def _benchmark(cache_dir: str, split: str, batch_size: int, workers: Sequence[int], n_batches: int) -> None:
    for nw in workers:
        loader = JEPALoader(cache_dir, split, batch_size, num_workers=nw, seed=0)
        it = iter(loader)
        next(it)                     # worker start-up is not part of the steady state
        t = time.perf_counter()
        for _ in range(n_batches):
            next(it)
        dt = time.perf_counter() - t
        print(f"{nw} workers: {n_batches * batch_size / dt:8.0f} positions/s  ({1000 * dt / n_batches:.1f} ms per batch of {batch_size})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build the position cache from the database or a FEN file")
    source = b.add_mutually_exclusive_group()
    source.add_argument("--db", default=None, help=f"SQLite database (default {DEFAULT_DB})")
    source.add_argument("--fens", default=None, help="text file with one FEN per line")
    b.add_argument("--out", default=DEFAULT_CACHE_DIR)
    b.add_argument("--val", type=float, default=0.01)
    b.add_argument("--test", type=float, default=0.01)
    b.add_argument("--limit", type=int, default=None)
    m = sub.add_parser("bench", help="measure how many positions per second the loader prepares")
    m.add_argument("--cache", default=DEFAULT_CACHE_DIR)
    m.add_argument("--split", default="train")
    m.add_argument("--batch-size", type=int, default=256)
    m.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    m.add_argument("--batches", type=int, default=20)
    args = parser.parse_args()
    if args.command == "build" and args.fens:
        build_position_cache_from_file(args.fens, args.out, args.val, args.test)
    elif args.command == "build":
        build_position_cache(args.db or DEFAULT_DB, args.out, args.val, args.test, args.limit)
    else:
        _benchmark(args.cache, args.split, args.batch_size, args.workers, args.batches)
