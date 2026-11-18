"""
Where every piece of a position can be after 1, 2, ... plies of legal play, and in how many ways.

Take a position and a number of plies d, and look at every sequence of d legal moves from it, the two sides
moving in turn as in a game. For each piece on the board now, and each square:

    counts[d - 1, p, s]    how many of those sequences leave the piece that stands on square p NOW standing on
                           square s afterwards; s = 64 (CAPTURED): how many leave it off the board.

Squares are python-chess's (a1 = 0, b1 = 1, ..., h8 = 63), as the position is written, not turned to the side
to move. A piece is followed through the sequence, so two knights are told apart by where they stand now, and
a pawn that promotes stays in that pawn's row. Every sequence counts: an arrangement reached by two move orders
counts twice, and promoting to each of the four pieces is four sequences. A sequence that cannot be continued
(mate or stalemate) is not among the longer ones, so at each depth every piece's row adds up to the number of
sequences of that length (perft). Draws that take no move away (repetition, the 50-move rule, too little
material) do not end a sequence. Rows of empty squares are zero.

From the counts one can read whether a piece can reach a square within k plies (a count above 0 at some depth
up to k), the fewest plies that takes, how many move orders do it, and how often each piece is captured.

This module is the search and the storage format, and imports only python-chess and numpy, so the labelling
program's worker processes stay small. reach_labeler.py is the program that fills a database with it.
"""
from __future__ import annotations

import struct
import zlib
from typing import Dict, List, Sequence, Tuple

import chess
import numpy as np

CAPTURED: int = 64          # the column for "no longer on the board"


def position_key(fen: str) -> str:
    """The position part of a FEN (board, side to move, castling, en passant), without the move counters:
    what the labels are stored under (labyrinth_data.position_key, repeated here to keep this module light)."""
    return " ".join(fen.split()[:4])


def reach_counts(fen: str, depth: int) -> np.ndarray:
    """
    [depth, 64, 65] uint64: the counts described at the top of this file, for sequences of 1 to `depth` plies.

    The sequences are not played out one by one. The search goes a ply at a time over the arrangements that can
    arise (who stands where, and whatever else decides the moves to come), and each arrangement carries the
    number of sequences that lead to it. Arrangements that several move orders reach are searched once with
    their numbers added, which changes nothing in the counts: every move order is still counted. The last ply
    is not played either; its moves are only counted. Both savings grow with depth.
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")
    board = chess.Board(fen, chess960=True)
    squares = [sq for sq in chess.SQUARES if board.piece_type_at(sq) is not None]      # the pieces, by their square now
    n = len(squares)
    is_white = [board.color_at(sq) for sq in squares]
    who = [-1] * 64                             # square -> piece number
    for i, sq in enumerate(squares):
        who[sq] = i
    totals = [[0] * (n * 65) for _ in range(depth)]         # [ply][piece * 65 + square], plain ints (no overflow)
    layer: List[Tuple[chess.Board, List[int], List[int], int]] = [(board, who, list(squares), 1)]
    for d in range(1, depth + 1):
        total = totals[d - 1]
        last = d == depth
        merge = d >= 3 and not last             # two move orders cannot meet before the third ply
        merged: Dict[tuple, list] = {}
        children: List[Tuple[chess.Board, List[int], List[int], int]] = []
        for b, who, where, m in layer:          # m: how many sequences lead to this arrangement
            turn, ep, pawns = b.turn, b.ep_square, b.pawns
            moves = 0
            for move in b.generate_legal_moves():
                moves += 1
                f, t = move.from_square, move.to_square
                i, j = who[f], who[t]
                if not last:
                    w, wh = who[:], where[:]
                if j >= 0 and is_white[j] == turn:
                    # Castling: python-chess writes it as the king taking its own rook (Chess960 style)
                    rank = f & 56
                    king_to, rook_to = (rank | 2, rank | 3) if t < f else (rank | 6, rank | 5)
                    total[i * 65 + f] -= m
                    total[i * 65 + king_to] += m
                    total[j * 65 + t] -= m
                    total[j * 65 + rook_to] += m
                    if not last:
                        w[f] = w[t] = -1
                        w[king_to], w[rook_to] = i, j
                        wh[i], wh[j] = king_to, rook_to
                else:
                    taken = -1
                    if j >= 0:
                        taken, taken_on = j, t
                    elif t == ep and (pawns >> f) & 1:
                        taken_on = t - 8 if turn else t + 8         # en passant: the pawn stands beside, not on t
                        taken = who[taken_on]
                    if taken >= 0:
                        total[taken * 65 + taken_on] -= m
                        total[taken * 65 + CAPTURED] += m
                    total[i * 65 + f] -= m
                    total[i * 65 + t] += m
                    if not last:
                        if taken >= 0:
                            w[taken_on] = -1
                            wh[taken] = CAPTURED
                        w[f] = -1
                        w[t] = i
                        wh[i] = t
                if last:
                    continue
                child = b.copy(stack=False)
                child.push(move)
                if merge:
                    key = (tuple(wh), child.knights, child.bishops, child.rooks, child.queens,
                           child.castling_rights, child.ep_square)
                    known = merged.get(key)
                    if known is None:
                        merged[key] = [child, w, wh, m]
                    else:
                        known[3] += m
                else:
                    children.append((child, w, wh, m))
            # Every move left every piece where it was, except as corrected above
            weight = m * moves
            if weight:
                for k in range(n):
                    total[k * 65 + where[k]] += weight
        layer = [tuple(state) for state in merged.values()] if merge else children
    out = np.zeros((depth, 64, 65), dtype=np.uint64)
    if n:
        out[:, squares, :] = np.array(totals, dtype=np.uint64).reshape(depth, n, 65)
    return out


# ##### ##### ##### ##### #####
#       Storage

_HEADER = struct.Struct("<BBBQI")       # version, depth, bytes per count, occupied squares, how many counts are not 0
_VERSION = 1


def pack_counts(counts: np.ndarray) -> bytes:
    """
    reach_counts' array as a compact blob: only the counts above 0 are kept (their places as gaps from the one
    before, then the counts themselves in 4 or 8 bytes), deflated. Exact; unpack_counts gives the array back.
    """
    depth = counts.shape[0]
    if counts.shape != (depth, 64, 65) or not 1 <= depth <= 255:
        raise ValueError("counts must be [depth, 64, 65] with depth 1 to 255")
    counts = counts.astype(np.uint64, copy=False)
    occupied = counts.any(axis=(0, 2))                                  # every piece's row has counts at depth 1
    rows = counts[:, occupied, :].reshape(-1)
    places = np.flatnonzero(rows)
    values = rows[places]
    width = 4 if values.size == 0 or int(values.max()) < 2 ** 32 else 8
    gaps = np.diff(places, prepend=0).astype("<u4")
    mask = sum(1 << int(sq) for sq in np.flatnonzero(occupied))
    body = gaps.tobytes() + values.astype("<u4" if width == 4 else "<u8").tobytes()
    return _HEADER.pack(_VERSION, depth, width, mask, places.size) + zlib.compress(body, 6)


def unpack_counts(blob: bytes) -> np.ndarray:
    """A pack_counts blob -> [depth, 64, 65] uint64."""
    version, depth, width, mask, size = _HEADER.unpack_from(blob)
    if version != _VERSION:
        raise ValueError(f"unknown reach label format {version}")
    body = zlib.decompress(blob[_HEADER.size:])
    places = np.cumsum(np.frombuffer(body, dtype="<u4", count=size).astype(np.int64))
    values = np.frombuffer(body, dtype="<u4" if width == 4 else "<u8", count=size, offset=4 * size)
    squares = [sq for sq in range(64) if mask >> sq & 1]
    rows = np.zeros(depth * len(squares) * 65, dtype=np.uint64)
    rows[places] = values
    out = np.zeros((depth, 64, 65), dtype=np.uint64)
    if squares:
        out[:, squares, :] = rows.reshape(depth, len(squares), 65)
    return out


def packed_depth(blob: bytes) -> int:
    """How many plies a pack_counts blob covers, without unpacking it."""
    return _HEADER.unpack_from(blob)[1]


def label_positions(job: Tuple[Sequence[str], int]) -> List[Tuple[str, int, bytes]]:
    """What a worker does: (FENs, depth) -> for each FEN, (its position key, depth, its packed counts)."""
    fens, depth = job
    return [(position_key(fen), depth, pack_counts(reach_counts(fen, depth))) for fen in fens]
