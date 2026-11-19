"""
Where every piece of a position can be after 1, 2, ... plies of legal play, and what it could move to then,
each with the number of ways: the reach labels and the legal-threat labels.

Reach (reach_counts). Take a position and a number of plies d, and look at every sequence of d legal moves from it, the two sides
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

Legal threats (threat_counts) are the second kind of label, from the same sequences. A piece threatens a
square when moving there would be a legal move if it were its side's turn:

    threats[k, p, s]       how many sequences of k plies (k = 0: the position as it is) leave the piece that
                           stands on square p NOW able to move to square s, were it its side's turn.

Both sides' pieces are asked after every sequence, whoever moved last, so each ply adds something for both.
What follows from "a legal move":
  * a pinned piece threatens nothing off the line of its pin, and a piece freed by another piece moving out
    of its way (or by a capture) threatens its new squares from that ply on;
  * a pawn threatens the squares ahead of it while they are empty, and a diagonal square only when there is
    something to take there (the en passant square included, for the side whose turn it really is);
  * a piece giving check threatens the king's square: with the turn handed over it could take the king;
  * castling counts as the king and the rook each threatening the square they would castle to;
  * a pawn that could promote threatens the square once, not once per promotion piece;
  * a captured piece threatens nothing in the sequences where it is gone.
Not counted, because it is no legal move: standing guard over one's own piece, and a pawn covering an empty
diagonal square. Both show a ply later, once something stands there to be taken. (A separate "guards" count
would show them at once: left out on purpose, a possible later addition.)
A count above 0 is the yes/no form (can this threat arise within k plies); the count is in how many ways.

This module is the search and the storage format, and imports only python-chess and numpy, so the labelling
program's worker processes stay small. reach_labeler.py is the program that fills a database with it.

Two searches give the same arrays. The ones written here in Python are the reference: short, checked against
every sequence played out one by one (tests/test_reach_labeler.py). fast_reach.c is the same counting compiled,
some hundreds of times faster; `python build_fast_reach.py` builds it, and once the library lies next to this
file reach_counts and threat_counts use it (USING_C says whether they do). tests/test_fast_reach.py checks
that the two agree. A position the compiled search refuses (one that cannot arise in a game) goes to the
Python search, so the answer never depends on which is installed.
"""
from __future__ import annotations

import ctypes
import os
import struct
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import chess
import numpy as np

CAPTURED: int = 64          # the column for "no longer on the board"

# ---------- ---------- ----------
#   The compiled search, when it has been built

C_VERSION: int = 1          # fast_reach.c's fr_version(): a library built from another version is not used
# Memory, per process, for the positions the compiled search remembers within and between searches. Only
# allocated by threat searches of 4 plies or more, which are the ones that meet positions again.
TABLE_MB: int = int(os.environ.get("FAST_REACH_TABLE_MB", "128"))
TABLE_SLOTS_LOG2: int = 18

LIBRARY: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fast_reach.dll" if os.name == "nt" else "fast_reach.so")


def _load_library() -> Tuple[Optional[ctypes.CDLL], str]:
    """(the library, "") or (None, why not)."""
    if not os.path.exists(LIBRARY):
        return None, "not built (python build_fast_reach.py)"
    try:
        lib = ctypes.CDLL(LIBRARY)
        lib.fr_version.restype = ctypes.c_int
        if lib.fr_version() != C_VERSION:
            return None, "built from another version of fast_reach.c (python build_fast_reach.py)"
        lib.fr_new.argtypes = [ctypes.c_size_t, ctypes.c_int]
        lib.fr_new.restype = ctypes.c_void_p
        for call in (lib.fr_clear, lib.fr_free):
            call.argtypes = [ctypes.c_void_p]
            call.restype = None
        lib.fr_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.fr_stats.restype = None
        for search in (lib.fr_threat_counts, lib.fr_reach_counts):
            search.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
            search.restype = ctypes.c_int
    except (OSError, AttributeError) as error:
        return None, f"could not be loaded ({error})"
    return lib, ""


_LIB, C_UNAVAILABLE = _load_library()
USING_C: bool = _LIB is not None    # whether the searches below use the compiled one; C_UNAVAILABLE says why not
_CONTEXT: Optional[int] = None      # this process's search context (a process pool's workers each make their own)


def use_compiled(on: bool) -> bool:
    """Switch the compiled search on or off for this process (off: to run the Python searches where the library
    is built, as their own tests do). Returns whether it is in use now."""
    global USING_C
    USING_C = bool(on) and _LIB is not None
    return USING_C


def _context() -> int:
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = _LIB.fr_new(TABLE_MB, TABLE_SLOTS_LOG2)
        if not _CONTEXT:
            raise MemoryError("fast_reach: no memory for a search context")
    return _CONTEXT


def forget() -> None:
    """Drop what the compiled search remembers from earlier searches (tests; it never changes an answer)."""
    if _LIB is not None:
        _LIB.fr_clear(_context())


def search_stats() -> Dict[str, int]:
    """Counters of the compiled search in this process since it started ({} without it)."""
    if _LIB is None:
        return {}
    raw = (ctypes.c_uint64 * 8)()
    _LIB.fr_stats(_context(), raw)
    names = ("positions", "table_hits", "table_stores", "answers_reused", "table_clears", "searches")
    return {name: int(raw[i]) for i, name in enumerate(names)}


def _compiled(search, fen: str, depth: int, shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    """The compiled search's array, or None when it refuses the position (the Python search then takes it)."""
    out = np.empty(shape, dtype=np.uint64)
    if search(_context(), fen.encode("ascii", "replace"), depth, out.ctypes.data):
        return None
    return out


def position_key(fen: str) -> str:
    """The position part of a FEN (board, side to move, castling, en passant), without the move counters:
    what the labels are stored under (labyrinth_data.position_key, repeated here to keep this module light)."""
    return " ".join(fen.split()[:4])


def reach_counts(fen: str, depth: int, use_c: bool = True) -> np.ndarray:
    """
    [depth, 64, 65] uint64: the counts described at the top of this file, for sequences of 1 to `depth` plies.
    use_c=False: the Python search below even when the compiled one is there (the two give the same array).

    The sequences are not played out one by one. The search goes a ply at a time over the arrangements that can
    arise (who stands where, and whatever else decides the moves to come), and each arrangement carries the
    number of sequences that lead to it. Arrangements that several move orders reach are searched once with
    their numbers added, which changes nothing in the counts: every move order is still counted. The last ply
    is not played either; its moves are only counted. Both savings grow with depth.
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if use_c and USING_C:
        out = _compiled(_LIB.fr_reach_counts, fen, depth, (depth, 64, 65))
        if out is not None:
            return out
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


def threat_counts(fen: str, plies: int, use_c: bool = True) -> np.ndarray:
    """
    [plies + 1, 64, 64] uint64: the legal threats described at the top of this file, after sequences of 0 to
    `plies` plies. Layer k holds what every piece of both sides could legally move to after k plies.
    use_c=False: the Python search below even when the compiled one is there (the two give the same array).

    The search is reach_counts' (arrangements a ply at a time, each with the number of sequences that lead to
    it, move orders that meet searched once with their numbers added), except that the last ply's positions
    are needed too: threats are read off a position, where reach only counts the moves into it. At each
    position the side to move's legal moves are listed, then the other side's with the turn handed to it (and
    the en passant square, which is not that side's to use, set aside).
    """
    if plies < 0:
        raise ValueError("plies must be at least 0")
    if use_c and USING_C:
        out = _compiled(_LIB.fr_threat_counts, fen, plies, (plies + 1, 64, 64))
        if out is not None:
            return out
    board = chess.Board(fen, chess960=True)
    squares = [sq for sq in chess.SQUARES if board.piece_type_at(sq) is not None]      # the pieces, by their square now
    n = len(squares)
    is_white = [board.color_at(sq) for sq in squares]
    who = [-1] * 64                             # square -> piece number
    for i, sq in enumerate(squares):
        who[sq] = i
    totals = [[0] * (n * 64) for _ in range(plies + 1)]     # [ply][piece * 64 + square], plain ints (no overflow)
    layer: List[Tuple[chess.Board, List[int], int]] = [(board, who, 1)]
    for k in range(plies + 1):
        total = totals[k]
        last = k == plies
        merge = k >= 2 and not last             # the positions after the next ply: move orders meet from the third on
        merged: Dict[tuple, list] = {}
        children: List[Tuple[chess.Board, List[int], int]] = []
        for b, who, m in layer:                 # m: how many sequences lead to this arrangement
            turn, ep, pawns = b.turn, b.ep_square, b.pawns
            moves = list(b.generate_legal_moves())
            # A set, so that a threat with two moves behind it counts once: the four promotions of a pawn move,
            # or a rook that can reach its castling square by an ordinary move as well
            threats = set()
            for mover in (turn, not turn):
                if mover == turn:
                    side_moves = moves
                else:
                    b.turn, b.ep_square = mover, None           # as if it were the other side's turn
                    side_moves = b.generate_legal_moves()
                for move in side_moves:
                    f, t = move.from_square, move.to_square
                    i, j = who[f], who[t]
                    if j >= 0 and is_white[j] == mover:
                        # Castling: python-chess writes it as the king taking its own rook (Chess960 style)
                        rank = f & 56
                        king_to, rook_to = (rank | 2, rank | 3) if t < f else (rank | 6, rank | 5)
                        if king_to != f:
                            threats.add(i * 64 + king_to)
                        if rook_to != t:
                            threats.add(j * 64 + rook_to)
                    else:
                        threats.add(i * 64 + t)                 # (with the turn handed over, t can be the king's square)
            b.turn, b.ep_square = turn, ep
            for code in threats:
                total[code] += m
            if last:
                continue
            for move in moves:
                f, t = move.from_square, move.to_square
                w = who[:]
                i, j = w[f], w[t]
                if j >= 0 and is_white[j] == turn:
                    rank = f & 56
                    king_to, rook_to = (rank | 2, rank | 3) if t < f else (rank | 6, rank | 5)
                    w[f] = w[t] = -1
                    w[king_to], w[rook_to] = i, j
                else:
                    if j < 0 and t == ep and (pawns >> f) & 1:
                        w[t - 8 if turn else t + 8] = -1        # en passant: the pawn stands beside, not on t
                    w[f] = -1
                    w[t] = i
                child = b.copy(stack=False)
                child.push(move)
                if merge:
                    key = (tuple(w), child.knights, child.bishops, child.rooks, child.queens,
                           child.castling_rights, child.ep_square)
                    known = merged.get(key)
                    if known is None:
                        merged[key] = [child, w, m]
                    else:
                        known[2] += m
                else:
                    children.append((child, w, m))
        layer = [tuple(state) for state in merged.values()] if merge else children
    out = np.zeros((plies + 1, 64, 64), dtype=np.uint64)
    if n:
        out[:, squares, :] = np.array(totals, dtype=np.uint64).reshape(plies + 1, n, 64)
    return out


# ##### ##### ##### ##### #####
#       Storage

_HEADER = struct.Struct("<BBBQI")       # kind, layers, bytes per count, squares with a row, how many counts are not 0
_REACH, _THREATS = 1, 2                 # the kind says which array a blob holds
_COLUMNS = {_REACH: 65, _THREATS: 64}


def pack_counts(counts: np.ndarray) -> bytes:
    """
    reach_counts' [depth, 64, 65] array or threat_counts' [plies + 1, 64, 64] one as a compact blob: only the
    counts above 0 are kept (their places as gaps from the one before, then the counts themselves in 4 or 8
    bytes), deflated. Exact; unpack_counts gives the array back, and the blob says which of the two it is.
    """
    layers = counts.shape[0] if counts.ndim == 3 else 0
    kind = {65: _REACH, 64: _THREATS}.get(counts.shape[2]) if counts.ndim == 3 else None
    if kind is None or counts.shape[1] != 64 or not 1 <= layers <= 255:
        raise ValueError("counts must be [depth, 64, 65] (reach) or [plies + 1, 64, 64] (threats), 1 to 255 layers")
    counts = counts.astype(np.uint64, copy=False)
    rowed = counts.any(axis=(0, 2))                                     # the squares whose row holds anything
    rows = counts[:, rowed, :].reshape(-1)
    places = np.flatnonzero(rows)
    values = rows[places]
    width = 4 if values.size == 0 or int(values.max()) < 2 ** 32 else 8
    gaps = np.diff(places, prepend=0).astype("<u4")
    mask = sum(1 << int(sq) for sq in np.flatnonzero(rowed))
    body = gaps.tobytes() + values.astype("<u4" if width == 4 else "<u8").tobytes()
    return _HEADER.pack(kind, layers, width, mask, places.size) + zlib.compress(body, 6)


def unpack_counts(blob: bytes) -> np.ndarray:
    """A pack_counts blob -> the array it was made from: [depth, 64, 65] or [plies + 1, 64, 64], uint64."""
    kind, layers, width, mask, size = _HEADER.unpack_from(blob)
    if kind not in _COLUMNS:
        raise ValueError(f"unknown label format {kind}")
    columns = _COLUMNS[kind]
    body = zlib.decompress(blob[_HEADER.size:])
    places = np.cumsum(np.frombuffer(body, dtype="<u4", count=size).astype(np.int64))
    values = np.frombuffer(body, dtype="<u4" if width == 4 else "<u8", count=size, offset=4 * size)
    squares = [sq for sq in range(64) if mask >> sq & 1]
    rows = np.zeros(layers * len(squares) * columns, dtype=np.uint64)
    rows[places] = values
    out = np.zeros((layers, 64, columns), dtype=np.uint64)
    if squares:
        out[:, squares, :] = rows.reshape(layers, len(squares), columns)
    return out


def packed_depth(blob: bytes) -> int:
    """How many plies of play a pack_counts blob covers, without unpacking it (threats have a layer for 0 plies)."""
    kind, layers = _HEADER.unpack_from(blob)[:2]
    return layers - 1 if kind == _THREATS else layers


def label_positions(job: Tuple[Sequence[str], int]) -> List[Tuple[str, int, bytes]]:
    """What a worker does for reach labels: (FENs, depth) -> for each FEN, (its position key, depth, its packed counts)."""
    fens, depth = job
    return [(position_key(fen), depth, pack_counts(reach_counts(fen, depth))) for fen in fens]


def label_threats(job: Tuple[Sequence[str], int]) -> List[Tuple[str, int, bytes]]:
    """
    What a worker does for threat labels: (FENs, plies) -> for each FEN, (its position key, plies, its packed counts).

    The FENs are searched last one first. The labeller hands over each sample as a position followed by the
    positions after it in its rollout, and the compiled search keeps the last few whole answers: searched in
    that order, every position finds the one that follows it already answered and skips that branch (about one
    branch in thirty). The rows come back in the order the FENs were given.
    """
    fens, plies = job
    rows = [(position_key(fen), plies, pack_counts(threat_counts(fen, plies))) for fen in reversed(fens)]
    return rows[::-1]
