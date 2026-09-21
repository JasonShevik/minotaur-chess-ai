import chess.engine
import chess
import torch
import math
import random
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from typing import List, Tuple, Callable, Dict, Any, Optional, Union


# ##### ##### ##### ##### #####
#       Core functions

def get_chess_graph_edges() -> List[set[Tuple[int, int]]]:
    """

    :return:
    """
    # Neighborhood functions for different pieces, used for depth first search to get edges
    pieces_list: List[Callable[Tuple[int, int],
                               set[Tuple[int, int]]]] = [
        get_knight_neighbors,
        get_bishop_neighbors,
        get_rook_neighbors,
        get_king_neighbors
    ]

    # A list of lists of pairwise edges between chessboard squares 0 through 63
    edges_lists: List[set[Tuple[int, int]]] = [get_pawn_move_edges(),    # 0 Pawn move
                                               get_pawn_attack_edges(),  # 1 Pawn attack
                                               (),                       # 2 Knight move
                                               (),                       # 3 Bishop move
                                               (),                       # 4 Rook move
                                               (),                       # 5 King move
                                               (),                       # 6 Queen move
                                               ()]                       # 7 Castle

    edges_index: int = 2
    # Go through the structure, calling each piece function and updating edges_list and edge_types_list
    for neighbor_function in pieces_list:
        # Get the list of new edges that are specific to this piece_type
        edges_lists[edges_index] = depth_first_recursive(visited=[False for _ in range(64)],
                                                         current_coordinates=(0, 0),
                                                         edges=set(),
                                                         get_neighbors=neighbor_function)
        # If this is the bishop, we need to perform another search for the light squares
        if edges_index == 3:
            edges_lists[edges_index].update(depth_first_recursive(visited=[False for _ in range(64)],
                                                                  current_coordinates=(0, 1),
                                                                  edges=set(),
                                                                  get_neighbors=neighbor_function))
        edges_index += 1

    # Queen move edges are the union of bishop and rook moves
    edges_lists[6] = edges_lists[3].union(edges_lists[4])

    # Don't add any castling edges because their existence depends on the position

    return edges_lists


# Castling destination squares (standard / Chess960): king g1/f1, c1/d1; g8/f8, c8/d8
_CASTLING_DESTS = [
    ("K", chess.WHITE, True, chess.square(6, 0), chess.square(5, 0), 0),
    ("Q", chess.WHITE, False, chess.square(2, 0), chess.square(3, 0), 0),
    ("k", chess.BLACK, True, chess.square(6, 7), chess.square(5, 7), 7),
    ("q", chess.BLACK, False, chess.square(2, 7), chess.square(3, 7), 7),
]

# Castling right letter -> (color, kingside)
_CASTLING_KEYS: Dict[str, Tuple[chess.Color, bool]] = {
    "K": (chess.WHITE, True),
    "Q": (chess.WHITE, False),
    "k": (chess.BLACK, True),
    "q": (chess.BLACK, False),
}


# ##### ##### ##### ##### #####
#   Position plausibility
#
# Perturbations must produce positions that are different from the original but still
# plausible: something that could arise in a legal game of Chess960. The helpers below
# encode the rules that the perturbations and the final gate in perturb_position rely on.


def _normalize_castling_field(fen: str) -> str:
    """
    Return the FEN with its castling field expressed strictly as K/Q/k/q letters.

    X-FEN and Shredder-FEN may name the castling rook by its file letter (e.g. "Fd") when the
    castling rook is not the outermost rook. The rest of this project only understands the
    K/Q/k/q form, so file letters are mapped onto the side of the king they sit on. The
    letters are emitted in the canonical KQkq order.
    """
    parts = fen.split()
    if len(parts) < 3 or parts[2] == "-":
        return fen
    field = parts[2]
    if all(c in "KQkq" for c in field):
        return fen

    board = chess.Board(fen, chess960=True)
    rights: set[str] = set()
    for c in field:
        if c in "KQkq":
            rights.add(c)
            continue
        if c in "ABCDEFGH":
            color, keys = chess.WHITE, "KQ"
        elif c in "abcdefgh":
            color, keys = chess.BLACK, "kq"
        else:
            continue
        king_sq = board.king(color)
        if king_sq is None:
            continue
        rook_file = "abcdefgh".index(c.lower())
        rights.add(keys[0] if rook_file > chess.square_file(king_sq) else keys[1])
    parts[2] = "".join(k for k in "KQkq" if k in rights) or "-"
    return " ".join(parts)


def _castling_rights_in_fen(fen: str) -> set[str]:
    """The castling rights present in a FEN, as a set of K/Q/k/q letters."""
    return set(c for c in _normalize_castling_field(fen).split()[2] if c in "KQkq")


def _with_castling_rights(fen: str, rights: set[str]) -> str:
    """Return the FEN with its castling field replaced by the given set of K/Q/k/q letters."""
    parts = fen.split()
    parts[2] = "".join(k for k in "KQkq" if k in rights) or "-"
    return " ".join(parts)


def _king_can_hold_castling_rights(board: chess.Board, color: chess.Color) -> bool:
    """
    A king can only still hold a castling right if it is on its own back rank and not on the
    a or h file. In Chess960 the king always starts somewhere between its two rooks, so it
    never starts on a corner file, and a king that has moved has lost its rights.
    """
    king_sq = board.king(color)
    if king_sq is None:
        return False
    back_rank = 0 if color == chess.WHITE else 7
    return chess.square_rank(king_sq) == back_rank and 0 < chess.square_file(king_sq) < 7


def _back_rank_rook_files(board: chess.Board, color: chess.Color, kingside: bool) -> set[int]:
    """
    The files of every rook of `color` standing on its back rank on the given side of its
    king. Any one of them could be the castling rook as far as the FEN can tell. Empty if
    the king is not on its back rank.
    """
    king_sq = board.king(color)
    back_rank = 0 if color == chess.WHITE else 7
    if king_sq is None or chess.square_rank(king_sq) != back_rank:
        return set()
    king_file = chess.square_file(king_sq)
    files: set[int] = set()
    for f in range(8):
        if (f > king_file) if kingside else (f < king_file):
            if board.piece_at(chess.square(f, back_rank)) == chess.Piece(chess.ROOK, color):
                files.add(f)
    return files


def _reference_castling_files(reference_fen: str, kingside: bool) -> Optional[set[int]]:
    """
    From a trusted reference position, work out which files the castling rook for a given
    side (kingside or queenside) could be on.

    Both colors hold the right  -> the files where both have a rook (they must match).
    One color holds the right   -> that color's rook files on that side.
    Neither holds the right     -> None, meaning the reference says nothing about it.
    """
    board = chess.Board(reference_fen, chess960=True)
    rights = _castling_rights_in_fen(reference_fen)
    w_key, b_key = ("K", "k") if kingside else ("Q", "q")
    w_files = _back_rank_rook_files(board, chess.WHITE, kingside) if w_key in rights else set()
    b_files = _back_rank_rook_files(board, chess.BLACK, kingside) if b_key in rights else set()
    if w_key in rights and b_key in rights:
        common = w_files & b_files
        return common if common else None
    if w_key in rights:
        return w_files or None
    if b_key in rights:
        return b_files or None
    return None


def _plausible_castling_rights(
    board: chess.Board,
    requested: set[str],
    reference_fen: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> set[str]:
    """
    Return the largest subset of `requested` castling rights that is plausible on `board`.

    A right is kept only if that king can still hold rights (back rank, not on a corner
    file) and a rook of that color stands on its back rank on that side of the king. When
    both colors hold the right on the same side, their candidate rooks must share a file,
    because in Chess960 both players start with the same setup and a castling rook never
    moves. When a reference position is supplied, the candidate files are additionally
    restricted to the files the castling rook could have been on in that position, which
    is how a perturbation that removes the real castling rook is told apart from one that
    leaves a decoy rook on the same side.

    If both colors hold a right on a side but no shared file remains and the reference
    cannot resolve it, exactly one of the two rights must be bogus but there is no way to
    tell which, so one of them is dropped at random.
    """
    rand = rng if rng is not None else random
    kept: set[str] = set()
    for kingside in (True, False):
        w_key, b_key = ("K", "k") if kingside else ("Q", "q")
        w_ok = w_key in requested and _king_can_hold_castling_rights(board, chess.WHITE)
        b_ok = b_key in requested and _king_can_hold_castling_rights(board, chess.BLACK)
        w_files = _back_rank_rook_files(board, chess.WHITE, kingside) if w_ok else set()
        b_files = _back_rank_rook_files(board, chess.BLACK, kingside) if b_ok else set()

        if reference_fen is not None:
            allowed = _reference_castling_files(reference_fen, kingside)
            if allowed is not None:
                w_files &= allowed
                b_files &= allowed

        w_ok = w_ok and bool(w_files)
        b_ok = b_ok and bool(b_files)

        if w_ok and b_ok and not (w_files & b_files):
            if rand.random() < 0.5:
                w_ok = False
            else:
                b_ok = False

        if w_ok:
            kept.add(w_key)
        if b_ok:
            kept.add(b_key)
    return kept


def is_castling_right_plausible(
    board: chess.Board,
    key: str,
    reference_fen: Optional[str] = None,
) -> bool:
    """
    Return True if the castling right `key` (K, Q, k or q) could be added to this board
    without contradicting anything: the king is on its back rank and not on a corner file,
    a rook of that color stands on that side of the king, and if the other color already
    holds the same-side right, the two candidate rooks share a file. Blocking pieces and
    attacks are ignored, because the edge represents the right to castle rather than
    whether castling is legal right now.
    """
    if key not in _CASTLING_KEYS:
        return False
    current = _castling_rights_in_fen(board.fen())
    kept = _plausible_castling_rights(board, current | {key}, reference_fen)
    return key in kept and current <= kept


def fix_castling_in_fen(
    fen_str: str,
    reference_fen: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Return the FEN with its castling field pruned to the rights that are still plausible.
    Never adds a right. Call this after any perturbation that could have moved or removed a
    king or rook. Pass the unperturbed position as `reference_fen` so the pruning knows
    which rooks were the castling rooks; see _plausible_castling_rights for the rules.
    """
    fen_str = _normalize_castling_field(fen_str)
    current = _castling_rights_in_fen(fen_str)
    if not current:
        return fen_str
    board = chess.Board(fen_str, chess960=True)
    kept = _plausible_castling_rights(board, current, reference_fen, rng)
    return _with_castling_rights(fen_str, kept)


def _align_castling_rooks(
    board: chess.Board,
    reference_fen: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> None:
    """
    Make python-chess's internal choice of castling rook agree with this project's rules.

    A K/Q/k/q letter does not say which rook castles when two rooks stand on the same side
    of the king; python-chess resolves it to the outermost one. This project instead requires
    that when both colors hold the same-side right their castling rooks share a file, and
    that the file agrees with the reference position. This rewrites board.castling_rights in
    place so that legal move generation (which rook takes part in castling, and which rook's
    move forfeits the right) follows the same rook the rest of the pipeline would choose.
    Only positions with two rooks on one side of a king are affected; when several files
    remain possible one is chosen at random, the same file for both colors.
    """
    rand = rng if rng is not None else random
    rights = _castling_rights_in_fen(board.fen())
    if not rights:
        return

    def current_file(color: chess.Color, kingside: bool) -> Optional[int]:
        king_sq = board.king(color)
        if king_sq is None:
            return None
        back = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
        king_file = chess.square_file(king_sq)
        for sq in chess.scan_forward(board.castling_rights & back & board.occupied_co[color]):
            f = chess.square_file(sq)
            if (f > king_file) if kingside else (f < king_file):
                return f
        return None

    new_rights = 0
    for kingside in (True, False):
        w_key, b_key = ("K", "k") if kingside else ("Q", "q")
        w_has, b_has = w_key in rights, b_key in rights
        if not (w_has or b_has):
            continue
        w_files = _back_rank_rook_files(board, chess.WHITE, kingside) if w_has else set()
        b_files = _back_rank_rook_files(board, chess.BLACK, kingside) if b_has else set()
        if reference_fen is not None:
            allowed = _reference_castling_files(reference_fen, kingside)
            if allowed is not None:
                if w_files & allowed:
                    w_files &= allowed
                if b_files & allowed:
                    b_files &= allowed
        shared: Optional[set[int]] = None
        if w_has and b_has and (w_files & b_files):
            shared = w_files & b_files
            w_files = b_files = shared

        cw, cb = current_file(chess.WHITE, kingside), current_file(chess.BLACK, kingside)
        if shared is not None:
            if cw in shared and cb == cw:
                fw = fb = cw
            else:
                fw = fb = rand.choice(sorted(shared))
        else:
            fw = cw if cw in w_files else (rand.choice(sorted(w_files)) if w_files else None)
            fb = cb if cb in b_files else (rand.choice(sorted(b_files)) if b_files else None)
        if fw is not None:
            new_rights |= chess.BB_SQUARES[chess.square(fw, 0)]
        if fb is not None:
            new_rights |= chess.BB_SQUARES[chess.square(fb, 7)]
    board.castling_rights = new_rights


def _material_is_plausible(board: chess.Board) -> bool:
    """
    Each side must have exactly one king, at most 16 pieces and at most 8 pawns, and every
    piece beyond the starting set (a second queen, a third rook or knight, a second bishop
    on the same square color) must be accounted for by a promoted pawn, so the number of
    such extras cannot exceed the number of pawns that side is missing.
    """
    for color in (chess.WHITE, chess.BLACK):
        own = board.occupied_co[color]
        if chess.popcount(own) > 16:
            return False
        if chess.popcount(own & board.kings) != 1:
            return False
        pawns = chess.popcount(own & board.pawns)
        if pawns > 8:
            return False
        extras = (
            max(0, chess.popcount(own & board.queens) - 1)
            + max(0, chess.popcount(own & board.rooks) - 2)
            + max(0, chess.popcount(own & board.knights) - 2)
            + max(0, chess.popcount(own & board.bishops & chess.BB_LIGHT_SQUARES) - 1)
            + max(0, chess.popcount(own & board.bishops & chess.BB_DARK_SQUARES) - 1)
        )
        if extras > 8 - pawns:
            return False
    return True


def _added_piece_is_plausible(board: chess.Board, piece_type: chess.PieceType, color: chess.Color) -> bool:
    """Would adding one more piece of this type and color keep the material plausible?"""
    probe = board.copy(stack=False)
    # Any empty square will do for the count check, except bishops, whose square color matters
    # to the caller; they are re-checked on the real square.
    for sq in chess.SQUARES:
        if probe.piece_at(sq) is None:
            probe.set_piece_at(sq, chess.Piece(piece_type, color))
            return _material_is_plausible(probe)
    return False


def position_is_plausible(fen: str) -> bool:
    """
    The final gate every perturbation must pass. A position is plausible when:
      - python-chess accepts it as valid: both kings present, no more than 16 pieces or 8
        pawns per side, no pawn on the first or eighth rank, the side not to move is not in
        check, the en passant square agrees with the side to move and is empty with an
        empty square behind it, the castling rights point at real rooks, and the checks on
        the side to move are geometrically possible;
      - the material could have arisen from a real game (see _material_is_plausible);
      - no king holds castling rights while standing on the a or h file;
      - when both colors hold castling rights on the same side, their candidate rooks
        share a file.
    """
    try:
        board = chess.Board(fen, chess960=True)
    except ValueError:
        return False
    if not board.is_valid():
        return False
    if not _material_is_plausible(board):
        return False
    rights = _castling_rights_in_fen(fen)
    if rights:
        if any(k in rights for k in "KQ") and not _king_can_hold_castling_rights(board, chess.WHITE):
            return False
        if any(k in rights for k in "kq") and not _king_can_hold_castling_rights(board, chess.BLACK):
            return False
        for kingside in (True, False):
            w_key, b_key = ("K", "k") if kingside else ("Q", "q")
            if w_key in rights and b_key in rights:
                w_files = _back_rank_rook_files(board, chess.WHITE, kingside)
                b_files = _back_rank_rook_files(board, chess.BLACK, kingside)
                if not (w_files & b_files):
                    return False
    return True


def mirror_fen_files(fen: str) -> str:
    """
    Reflect a position left to right, so the a file swaps with the h file, b with g, c with f
    and d with e. Ranks are untouched. Each color's kingside and queenside castling rights
    swap with each other, the castling rooks move with the board, and the en passant square
    is mirrored. Because a Chess960 start position mirrored is still a Chess960 start
    position, the mirror of a plausible position is always plausible. Applying this twice
    returns the original position.
    """
    board = chess.Board(fen, chess960=True)
    mirrored = board.transform(chess.flip_horizontal)
    return _normalize_castling_field(mirrored.fen())


def mirror_castling_rook_squares(
    castling_rook_squares: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """
    Mirror a (K, Q, k, q) tuple of castling rook squares to match mirror_fen_files. Each
    square's file is reflected, and because kingside becomes queenside, the K and Q slots
    swap, as do k and q. -1 entries stay -1. Use this so that the castling edges built for a
    mirrored position pick the same rook as the original when there is more than one rook on
    a side.
    """
    def mirror(sq: int) -> int:
        return sq if sq < 0 else (sq // 8) * 8 + (7 - sq % 8)

    k_rook, q_rook, k_rook_b, q_rook_b = castling_rook_squares
    return (mirror(q_rook), mirror(k_rook), mirror(q_rook_b), mirror(k_rook_b))


def _keep_move_counters(perturbed: str, original: str) -> str:
    """
    Copy the halfmove clock and fullmove number from the original onto the perturbed FEN.
    The graph never looks at them, and keeping them fixed means a perturbed FEN differs from
    the original only where the position itself differs.
    """
    p, o = perturbed.split(), original.split()
    if len(p) >= 6 and len(o) >= 6:
        p[4], p[5] = o[4], o[5]
    return " ".join(p)


# ##### ##### ##### ##### #####
#   Perturbations


def perturb_position(
    fen: str,
    perturb_type: Optional[int] = None,
    magnitude: int = 1,
    type_distribution: Optional[Union[torch.Tensor, Callable[[], int]]] = None,
    rng: Optional[random.Random] = None,
    max_attempts: int = 30,
    fallback_to_other_types: bool = True,
    return_type: bool = False,
) -> Union[str, Tuple[str, int]]:
    """
    Return a perturbed position as a new FEN string.

    The result is guaranteed to differ from the input and to pass position_is_plausible.
    Each perturbation type is sampled up to `max_attempts` times until an acceptable result
    appears; if a type cannot produce one (for example the castling perturbation on a
    position with no castling rights and no rooks on the back rank), the remaining types are
    tried in random order when `fallback_to_other_types` is True. If nothing works a
    ValueError is raised rather than silently returning the original position, because an
    unchanged "negative" would poison contrastive training.

    Perturbation types:
        0  Legal move: one piece (of either color) makes `magnitude` legal moves in a row.
        1  Illegal move: one piece jumps to an empty square within `magnitude` king steps
           that it could not legally reach.
        2  Deletion: remove one non-king piece, or the en passant square.
        3  Addition: add one piece on an empty square, or an en passant square where a
           double pawn push could just have happened.
        4  Swap: exchange two pieces that differ in type or color.
        5  Piece change: change one non-king piece to a different type of the same color.
        6  Color change: flip the color of one non-king piece.
        7  Castling rights: remove one existing right or add one plausible right.
        8  Mirror: reflect the whole position left to right (a file <-> h file).
        9  Turn change: give the move to the other side; pieces stay put, en passant is
           cleared. Impossible when the side that was to move is in check.

    :param fen: FEN string of the position to perturb.
    :param perturb_type: Which perturbation to apply (0..9). If None, one is chosen from
        type_distribution or uniformly at random.
    :param magnitude: Interpretation depends on perturb_type, but it is the size of a
        single perturbation, not the total number of perturbations.
    :param type_distribution: When perturb_type is None, how to choose the type.
        - If a 1D tensor of shape (num_types,): if all values are in [0, 1], treated as
          probabilities (normalized by sum); otherwise treated as logits (softmax).
          Sampled via torch.multinomial.
        - If a callable: call with no args to get an int in [0, num_types-1].
    :param rng: Optional random.Random for reproducible sampling.
    :param max_attempts: How many times to sample a type before giving up on it.
    :param fallback_to_other_types: Try the other types if the requested one cannot
        produce a plausible, different position.
    :param return_type: When True, return (fen, type_used) instead of just the FEN, so the
        caller can tell when a fallback happened.
    :return: Perturbed position as a FEN string (or a (fen, type) tuple).
    """
    rand = rng if rng is not None else random
    fen = _normalize_castling_field(fen)

    def perturb_fen_piece_move_legal(fen_str: str) -> str:      # ----- Perturbation Type 0 -----
        # Do 'magnitude' legal moves of the same piece in a row. The piece can be of either color;
        # we temporarily set the board's turn to that piece's color to get legal moves, then restore
        # the original turn after each move so the final FEN keeps e.g. white to move. Promotions
        # are excluded so a pawn never lands on the first or eighth rank.
        n = max(1, int(magnitude))
        board = chess.Board(fen_str, chess960=True)
        # When two rooks share a side of a king, castle with (and forfeit rights via) the rook that
        # matches the other color's castling rook, rather than python-chess's default outermost one.
        _align_castling_rooks(board, fen_str, rand)
        turn_white = board.turn  # turn to show in final FEN (unchanged by our moves)
        order = [s for s in chess.SQUARES if board.piece_at(s) is not None]
        rand.shuffle(order)

        def try_complete_moves(
            current: chess.Board,
            piece_square: chess.Square,
            visited: set[chess.Square],
            moves_done: int,
            target: int,
        ) -> Optional[str]:
            """Try to complete exactly `target` moves; at each step try all candidates (shuffled)."""
            if moves_done == target:
                return fix_castling_in_fen(current.fen(), fen_str, rand)
            piece_color = current.color_at(piece_square)
            if piece_color is None:
                return None
            current.turn = piece_color
            candidates = [
                m for m in current.legal_moves
                if m.from_square == piece_square and m.to_square not in visited and m.promotion is None
            ]
            if not candidates:
                return None
            rand.shuffle(candidates)
            for move in candidates:
                next_board = current.copy()
                next_board.push(move)
                new_sq = move.to_square
                new_visited = visited | {new_sq}
                if moves_done + 1 == target:
                    next_board.turn = turn_white
                    return fix_castling_in_fen(next_board.fen(), fen_str, rand)
                next_board.turn = turn_white
                result = try_complete_moves(next_board, new_sq, new_visited, moves_done + 1, target)
                if result is not None:
                    return result
            return None

        for start_square in order:
            result = try_complete_moves(board.copy(), start_square, {start_square}, 0, n)
            if result is not None:
                return result
        # No piece could do n moves; try fewer (n-1, ..., 1).
        for target in range(n - 1, 0, -1):
            for start_square in order:
                result = try_complete_moves(board.copy(), start_square, {start_square}, 0, target)
                if result is not None:
                    return result
        return fen_str

    def perturb_fen_piece_move_illegal(fen_str: str) -> str:      # ----- Perturbation Type 1 -----
        # Pick a random piece, find unoccupied squares within magnitude (Chebyshev) radius,
        # exclude squares that are legal moves for that piece, then move the piece to a random
        # illegal destination (starting square becomes empty). Pawns are never placed on the
        # first or eighth rank, and nothing is ever placed on the en passant square.
        board = chess.Board(fen_str, chess960=True)
        turn_white = board.turn
        radius = max(1, int(magnitude))
        order = [s for s in chess.SQUARES if board.piece_at(s) is not None]
        if not order:
            return fen_str
        rand.shuffle(order)
        for start_square in order:
            piece = board.piece_at(start_square)
            if piece is None:
                continue
            f0, r0 = chess.square_file(start_square), chess.square_rank(start_square)
            in_radius = []
            for f in range(8):
                for r in range(8):
                    if max(abs(f - f0), abs(r - r0)) <= radius:
                        sq = chess.square(f, r)
                        if sq == start_square:
                            continue
                        if board.piece_at(sq) is not None:
                            continue
                        if sq == board.ep_square:
                            continue
                        if piece.piece_type == chess.PAWN and r in (0, 7):
                            continue
                        in_radius.append(sq)
            if not in_radius:
                continue
            piece_color = piece.color
            board.turn = piece_color
            legal_to = {m.to_square for m in board.legal_moves if m.from_square == start_square}
            illegal_dest = [sq for sq in in_radius if sq not in legal_to]
            if not illegal_dest:
                continue
            to_square = rand.choice(illegal_dest)
            board.remove_piece_at(start_square)
            board.set_piece_at(to_square, chess.Piece(piece.piece_type, piece.color))
            board.turn = turn_white
            return fix_castling_in_fen(board.fen(), fen_str, rand)
        board.turn = turn_white
        return fen_str

    def perturb_fen_piece_deletion(fen_str: str) -> str:      # ----- Perturbation Type 2 -----
        # Delete exactly one "thing" at random: any non-king piece of either color, or the en
        # passant target square if present. Kings are never deleted because a position without
        # both kings is not a chess position. Magnitude is ignored.
        board = chess.Board(fen_str, chess960=True)
        options: List[Optional[chess.Square]] = [
            s for s in chess.SQUARES
            if board.piece_at(s) is not None and board.piece_at(s).piece_type != chess.KING
        ]
        if board.ep_square is not None:
            options.append(None)  # sentinel: clear en passant
        if not options:
            return fen_str
        choice = rand.choice(options)
        if choice is None:
            board.ep_square = None
        else:
            board.remove_piece_at(choice)
        return fix_castling_in_fen(board.fen(), fen_str, rand)

    def perturb_fen_piece_addition(fen_str: str) -> str:      # ----- Perturbation Type 3 -----
        # Add one thing at random: a piece (any non-king type and color) on an empty square, or an
        # en passant target. Additions that would push a side past 16 pieces or 8 pawns, or that
        # would need more promotions than that side has missing pawns, are not offered. Pawns are
        # never added to the first or eighth rank and nothing is added on the en passant square.
        # If en passant is not already set, possible ep squares are inferred from 4th/5th rank
        # pawn pairs (adjacent files with one white and one black pawn), and the square the pawn
        # would have come from must be empty. Magnitude is ignored.
        board = chess.Board(fen_str, chess960=True)
        options: List[Tuple[str, Any]] = []

        # Only consider the rank where the side to move could capture en passant:
        # Black to move -> a white pawn just double-pushed to the 4th rank, ep square on the 3rd.
        # White to move -> a black pawn just double-pushed to the 5th rank, ep square on the 6th.
        if board.ep_square is None:
            if board.turn == chess.BLACK:
                pawn_rank, ep_rank, origin_rank, mover = 3, 2, 1, chess.WHITE
            else:
                pawn_rank, ep_rank, origin_rank, mover = 4, 5, 6, chess.BLACK
            possible_ep_squares: set[chess.Square] = set()
            for f in range(7):
                sq_a, sq_b = chess.square(f, pawn_rank), chess.square(f + 1, pawn_rank)
                pa, pb = board.piece_at(sq_a), board.piece_at(sq_b)
                if pa is None or pb is None:
                    continue
                if pa.piece_type != chess.PAWN or pb.piece_type != chess.PAWN or pa.color == pb.color:
                    continue
                mover_file = f if pa.color == mover else f + 1
                ep_sq = chess.square(mover_file, ep_rank)
                origin_sq = chess.square(mover_file, origin_rank)
                if board.piece_at(ep_sq) is None and board.piece_at(origin_sq) is None:
                    possible_ep_squares.add(ep_sq)
            for ep_sq in possible_ep_squares:
                options.append(("ep", ep_sq))

        # Piece additions: only the (type, color) combinations that keep the material plausible
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        allowed: List[Tuple[chess.PieceType, chess.Color]] = [
            (pt, color)
            for pt in piece_types
            for color in (chess.WHITE, chess.BLACK)
            if _added_piece_is_plausible(board, pt, color)
        ]
        for s in chess.SQUARES:
            if board.piece_at(s) is not None or s == board.ep_square:
                continue
            rank = chess.square_rank(s)
            for pt, color in allowed:
                if pt == chess.PAWN and rank in (0, 7):
                    continue
                options.append(("piece", s, pt, color))

        if not options:
            return fen_str
        choice = rand.choice(options)
        if choice[0] == "ep":
            parts = board.fen().split()
            parts[3] = chess.square_name(choice[1])
            return " ".join(parts)
        _tag, square, piece_type, color = choice
        board.set_piece_at(square, chess.Piece(piece_type, color))
        return fix_castling_in_fen(board.fen(), fen_str, rand)

    def perturb_fen_piece_swap(fen_str: str) -> str:      # ----- Perturbation Type 4 -----
        # Swap two randomly chosen pieces that differ in type or color (so the position actually
        # changes). Swaps that would leave a pawn on the first or eighth rank are not offered.
        # Magnitude ignored. En passant is untouched because the ep square is always empty.
        board = chess.Board(fen_str, chess960=True)
        occupied = [s for s in chess.SQUARES if board.piece_at(s) is not None]
        pairs: List[Tuple[chess.Square, chess.Square]] = []
        for i, sq1 in enumerate(occupied):
            p1 = board.piece_at(sq1)
            for sq2 in occupied[i + 1:]:
                p2 = board.piece_at(sq2)
                if p1.piece_type == p2.piece_type and p1.color == p2.color:
                    continue
                if p1.piece_type == chess.PAWN and chess.square_rank(sq2) in (0, 7):
                    continue
                if p2.piece_type == chess.PAWN and chess.square_rank(sq1) in (0, 7):
                    continue
                pairs.append((sq1, sq2))
        if not pairs:
            return fen_str
        sq1, sq2 = rand.choice(pairs)
        p1, p2 = board.piece_at(sq1), board.piece_at(sq2)
        board.remove_piece_at(sq1)
        board.remove_piece_at(sq2)
        board.set_piece_at(sq1, p2)
        board.set_piece_at(sq2, p1)
        return fix_castling_in_fen(board.fen(), fen_str, rand)

    def perturb_fen_piece_change(fen_str: str) -> str:      # ----- Perturbation Type 5 -----
        # Pick a random non-king piece and change it to a different piece type of the same color.
        # A piece on the first or eighth rank is never turned into a pawn, and changes that would
        # make the material implausible are not offered. Magnitude ignored.
        board = chess.Board(fen_str, chess960=True)
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        options: List[Tuple[chess.Square, chess.PieceType]] = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type == chess.KING:
                continue
            for new_type in piece_types:
                if new_type == piece.piece_type:
                    continue
                if new_type == chess.PAWN and chess.square_rank(square) in (0, 7):
                    continue
                probe = board.copy(stack=False)
                probe.set_piece_at(square, chess.Piece(new_type, piece.color))
                if _material_is_plausible(probe):
                    options.append((square, new_type))
        if not options:
            return fen_str
        square, new_type = rand.choice(options)
        board.set_piece_at(square, chess.Piece(new_type, board.piece_at(square).color))
        return fix_castling_in_fen(board.fen(), fen_str, rand)

    def perturb_fen_piece_color_change(fen_str: str) -> str:      # ----- Perturbation Type 6 -----
        # Pick a random non-king piece and flip its color. Flips that would push the receiving
        # side past 16 pieces or 8 pawns, or past what its missing pawns can account for, are
        # not offered.
        board = chess.Board(fen_str, chess960=True)
        options: List[chess.Square] = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type == chess.KING:
                continue
            probe = board.copy(stack=False)
            probe.set_piece_at(square, chess.Piece(piece.piece_type, not piece.color))
            if _material_is_plausible(probe):
                options.append(square)
        if not options:
            return fen_str
        square = rand.choice(options)
        piece = board.piece_at(square)
        board.set_piece_at(square, chess.Piece(piece.piece_type, not piece.color))
        return fix_castling_in_fen(board.fen(), fen_str, rand)

    def perturb_fen_castling_rights_change(fen_str: str) -> str:     # ----- Perturbation Type 7 -----
        # One of: remove an existing castling right, or add a castling right that is plausible
        # (Chess960-aware, see is_castling_right_plausible). Removing and adding are weighted by
        # how many options each has, so every individual option is equally likely.
        board = chess.Board(fen_str, chess960=True)
        current = _castling_rights_in_fen(fen_str)
        actions: List[Tuple[str, str]] = []  # ('remove'|'add', 'K'|'Q'|'k'|'q')
        for key in "KQkq":
            if key in current:
                actions.append(("remove", key))
            elif is_castling_right_plausible(board, key, fen_str):
                actions.append(("add", key))
        if not actions:
            return fen_str
        op, key = rand.choice(actions)
        new_set = current - {key} if op == "remove" else current | {key}
        return fix_castling_in_fen(_with_castling_rights(fen_str, new_set), fen_str, rand)

    def perturb_fen_mirror_files(fen_str: str) -> str:     # ----- Perturbation Type 8 -----
        # Reflect the board left to right. Castling rights swap sides for each color, the
        # castling rooks move with the board, and the en passant square is mirrored. The result
        # is only identical to the input for a perfectly left-right symmetric position, in which
        # case perturb_position falls through to another type. Magnitude ignored.
        return mirror_fen_files(fen_str)

    def perturb_fen_turn_change(fen_str: str) -> str:     # ----- Perturbation Type 9 -----
        # Hand the move to the other side without touching a single piece. The en passant square is
        # cleared because it can only exist for the side that was about to move. From the encoder's
        # point of view this reorients the whole board (the side to move is always at the bottom) and
        # flips every hostility flag. If the side that was to move is in check, giving the move away
        # would leave a king in check on the opponent's turn, which is illegal; the gate rejects that
        # and perturb_position moves on to another type. Magnitude ignored.
        parts = fen_str.split()
        parts[1] = "b" if parts[1] == "w" else "w"
        parts[3] = "-"
        return " ".join(parts)

    perturbation_dispatch_table: Dict[int, Callable[[str], str]] = {
        0: perturb_fen_piece_move_legal,
        1: perturb_fen_piece_move_illegal,
        2: perturb_fen_piece_deletion,
        3: perturb_fen_piece_addition,
        4: perturb_fen_piece_swap,
        5: perturb_fen_piece_change,
        6: perturb_fen_piece_color_change,
        7: perturb_fen_castling_rights_change,
        8: perturb_fen_mirror_files,
        9: perturb_fen_turn_change,
    }

    num_types = len(perturbation_dispatch_table)

    # These always give the same answer for a given position, so retrying them is pointless.
    deterministic_types = {8, 9}

    # Choose perturbation type if not specified
    if perturb_type is None:
        if type_distribution is None:
            perturb_type = rand.randint(0, num_types - 1)
        elif callable(type_distribution):
            perturb_type = type_distribution()
        else:
            probs = type_distribution.to(torch.float64)
            if probs.dim() == 1 and probs.size(0) == num_types:
                # Treat as logits (softmax) if any value is outside [0, 1]; otherwise treat as
                # probabilities (possibly unnormalized) and normalize by sum.
                in_unit_interval = (probs >= 0).all().item() and (probs <= 1).all().item()
                if in_unit_interval and probs.sum().item() > 0:
                    probs = probs / probs.sum()
                else:
                    probs = torch.softmax(probs, dim=0)
                gen = torch.Generator(device=probs.device)
                if rng is not None:
                    gen.manual_seed(rng.randint(0, 2**31 - 1))
                perturb_type = int(torch.multinomial(probs, 1, generator=gen).item())
            else:
                perturb_type = rand.randint(0, num_types - 1)
    perturb_type = int(perturb_type) % num_types

    # Try the requested type first, then (optionally) every other type in random order.
    order = [perturb_type]
    if fallback_to_other_types:
        others = [t for t in range(num_types) if t != perturb_type]
        rand.shuffle(others)
        order += others

    for this_type in order:
        perturb = perturbation_dispatch_table[this_type]
        attempts = 1 if this_type in deterministic_types else max(1, int(max_attempts))
        for _ in range(attempts):
            candidate = _keep_move_counters(perturb(fen), fen)
            if candidate != fen and position_is_plausible(candidate):
                return (candidate, this_type) if return_type else candidate

    raise ValueError(
        f"No perturbation could produce a different, plausible position from {fen!r} "
        f"(tried types {order}, {max_attempts} attempts each)."
    )


def create_filled_chess_graphs(
    fen: str,
    castling_rook_squares: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    A function to get the graph representations of all piece interactions for a specified chess position.
    :param fen: The string that identifies the position to make graphs for.
    :param castling_rook_squares: Optional tuple of 4 square indices (0-63) in order K, Q, k, q for the
        castling rook on each side; use -1 for no right. When provided (e.g. from the DB column), castling
        edges use these squares; when None, rooks are inferred from the position (legacy behavior).
    :return: A tuple with the information needed for all of the graph networks. The first value in the tuple is
        a list of edge index tensors, one for a graph for each piece movement type. The second value in the tuple
        is the tensor with all of the node features, which includes piece locations and en passant information.
    """
    # Create a list of 64 floats that contains the information from the fen
    position_vector: List[float] = fen_to_vector(fen)

    # Initialize the working variables that I will eventually return
    edges_lists: List[set[Tuple[int, int]]] = get_chess_graph_edges()
    # Add the castling edges
    edges_lists.append(get_castling_edges(position_vector, castling_rook_squares))

    # node_features indices mean the following:
    # 0: Enemy controlled square flag
    # 1: Pawn flag
    # 2: Knight flag
    # 3: Bishop flag
    # 4: Rook flag
    # 5: Queen flag
    # 6: King flag
    # 7: En Passant flag
    node_features: List[List[int]] = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(64)]

    # Check for the existence of an en passant square
    if -0.5 in position_vector:
        # Set the En Passant flag for this square
        node_features[position_vector.index(-0.5)][7] = 1

    # Strip the fractional parts so the integers can be used as feature indices. This must truncate
    # toward zero, not floor: floor(-0.5) is -1, which would turn the en passant marker into an enemy
    # pawn, and floor(-6.3) is -7, which would give an enemy king with castling rights the en passant
    # flag instead of the king flag. int() maps -0.5 -> 0 and -6.3 -> -6 as intended.
    position_vector: List[int] = [int(value) for value in position_vector]

    # Loop through each of the 64 squares to set each node feature vector the correct piece vector
    square: int
    for square in range(64):
        # If there is no piece on this square, continue
        if position_vector[square] == 0:
            continue

        # We are going to shift the values: [-6, 6] -> 8 dimensional vector

        # If this square has an enemy piece, set the offensive
        if position_vector[square] < 0:
            node_features[square][0] = 1

        # Whatever piece is on this square, set that flag to 1
        # We know there is a piece since we would have continued earlier
        node_features[square][abs(position_vector[square])] = 1

    # Initialize the list of edge tensors that will eventually be returned
    edges_tensors: List[torch.Tensor] = []

    # Convert the pairwise edges to two lists for source and destination for compatibility with pytorch
    this_type_edges: set[Tuple[int, int]]
    for this_type_edges in edges_lists:
        x_list: List[int] = []
        y_list: List[int] = []
        for x, y in this_type_edges:
            x_list.append(x)
            y_list.append(y)

        # Set this graph tuple
        edges_tensors.append(torch.tensor(data=[x_list, y_list], dtype=torch.int64))

    # Create the node_features_tensor using the node_features list of lists
    node_features_tensor: torch.Tensor = torch.tensor(node_features, dtype=torch.float32)

    return edges_tensors, node_features_tensor


# ##### ##### ##### ##### #####
#   Piece connection getters


def get_pawn_move_edges() -> set[Tuple[int, int]]:
    """
    Deterministically find all paths where pawns can move.
    :return: A list of edges (tuples of start and end squares) that show where all pawn moves may be possible.
    """
    # Create the empty set of edges
    pawn_edges: set[Tuple[int, int]] = set()

    # Light square 2 square moves
    pawn_edges.update([(x, x + 16) for x in range(8, 16)])
    # Dark square 2 square moves
    pawn_edges.update([(x, x - 16) for x in range(48, 56)])

    # Light square 1 square moves
    for row_start in range(8, 49, 8):
        for column_offset in range(0, 8):
            origin_square = row_start + column_offset
            pawn_edges.update([(origin_square, origin_square + 8)])

    # Dark square 1 square moves
    for row_start in range(48, 7, -8):
        for column_offset in range(0, 8):
            origin_square = row_start + column_offset
            pawn_edges.update([(origin_square, origin_square - 8)])

    return pawn_edges


def get_pawn_attack_edges() -> set[Tuple[int, int]]:
    """
    Deterministically find all paths where pawns can attack.
    :return: A list of edges (tuples of start and end squares) that show where all pawn attacks may be possible.
    """
    # Go through A through H for both sides.
    edges: set[Tuple[int, int]] = set()
    # Front perspective
    spot: int
    row: int
    for row in range(1, 7, 1):
        spot = row * 8
        edges.add((spot, spot + 9))

        column: int
        for column in range(1, 7, 1):
            spot = (row * 8) + column
            edges.add((spot, spot + 7))
            edges.add((spot, spot + 9))

        spot = (row * 8) + 7
        edges.add((spot, spot + 7))

    # Back perspective
    spot: int
    row: int
    for row in range(6, 0, -1):
        spot = row * 8
        edges.add((spot, spot - 7))

        column: int
        for column in range(1, 7, 1):
            spot = (row * 8) + column
            edges.add((spot, spot - 7))
            edges.add((spot, spot - 9))

        spot = (row * 8) + 7
        edges.add((spot, spot - 9))

    return edges


def get_knight_neighbors(start_coordinates: Tuple[int, int]) -> set[Tuple[int, int]]:
    """
    Gets the list of all possible knight moves from this position if the board were infinite.
    :param start_coordinates: The coordinates that the knight starts on in the format [row, column].
    :return: A lit of coordinates of where the knight could move from the start given an infinite board.
    """
    row: int
    column: int
    row, column = start_coordinates

    return remove_invalid_coordinates({(row + 2, column + 1),  # Two up, one over
                                       (row + 2, column - 1),
                                       (row + 1, column + 2),  # Two over, one up
                                       (row + 1, column - 2),
                                       (row - 1, column + 2),  # Two over, one down
                                       (row - 1, column - 2),
                                       (row - 2, column + 1),  # Two down, one over
                                       (row - 2, column - 1)})


def get_bishop_neighbors(start_coordinates: Tuple[int, int]) -> set[Tuple[int, int]]:
    """
    Gets the list of all possible light bishop moves from this position.
    :param start_coordinates: The coordinates that the light bishop starts on in the format [row, column].
    :return: A lit of coordinates of where the light bishop could move from the start.
    """
    row: int
    column: int
    row, column = start_coordinates

    neighbors: set[Tuple[int, int]] = set()
    for count in range(1, 8):
        neighbors.update([(row + count, column + count),   # NE Direction
                          (row - count, column + count),   # SE
                          (row - count, column - count),   # SW
                          (row + count, column - count)])  # NW

    return remove_invalid_coordinates(neighbors)


def get_rook_neighbors(start_coordinates: Tuple[int, int]) -> set[Tuple[int, int]]:
    """
    Gets the list of all possible rook moves from this position.
    :param start_coordinates: The coordinates that the rook starts on in the format [row, column].
    :return: A lit of coordinates of where the rook could move from the start.
    """
    row: int
    column: int
    row, column = start_coordinates

    neighbors: set[Tuple[int, int]] = set()
    for count in range(1, 8):
        neighbors.update([(row + count, column        ),   # N Direction
                          (row,         column + count),   # E
                          (row - count, column        ),   # S
                          (row,         column - count)])  # W

    return remove_invalid_coordinates(neighbors)


def get_king_neighbors(start_coordinates: Tuple[int, int]) -> set[Tuple[int, int]]:
    """
    Gets the list of all possible king moves from this position.
    :param start_coordinates: The coordinates that the king starts on in the format [row, column].
    :return: A lit of coordinates of where the king could move from the start.
    """
    row: int
    column: int
    row, column = start_coordinates

    return remove_invalid_coordinates({(row + row_offset, column + column_offset)
                                        for row_offset in [-1, 0, 1]
                                        for column_offset in [-1, 0, 1]
                                        if not (row_offset == 0 and column_offset == 0)})


def get_castling_edges(
    board_vector: List[float],
    castling_rook_squares: Optional[Tuple[int, int, int, int]] = None,
    infer_symmetric: bool = True,
) -> set[Tuple[int, int]]:
    """
    Add castling edges to the graph. When castling_rook_squares is provided (K, Q, k, q rook square
    indices; -1 for none), those squares are used. When None, rooks are inferred from the board (legacy).

    If infer_symmetric is True and castling_rook_squares is None, infer rooks so K/k and Q/q share
    the same file. When both colors have the right but no common file, skip those edges (inconsistent).
    When only one color has the right, pick randomly among that side's rooks.

    board_vector: 6.1 = king-side, 6.2 = queen-side, 6.3 = either. Destinations: white K (g1,f1)=6,5;
    white Q (c1,d1)=2,3; black K (g8,f8)=62,61; black Q (c8,d8)=58,59.
    """
    edges: set[Tuple[int, int]] = set()

    # White: king index and rights from board_vector
    white_king = [i for i, pv in enumerate(board_vector) if int(pv) == 6 and pv % 1]
    # Black: king index and rights
    black_king = [i for i, pv in enumerate(board_vector) if int(pv) == -6 and pv % 1]

    # A king on the a or h file can never hold castling rights: a Chess960 king starts between
    # its rooks, so it never starts on a corner file, and a king that has moved has lost them.
    white_king = [i for i in white_king if 0 < i % 8 < 7]
    black_king = [i for i in black_king if 0 < i % 8 < 7]

    if castling_rook_squares is not None:
        # Use provided rook squares (K, Q, k, q); -1 means no right / skip
        k_rook, q_rook, k_rook_b, q_rook_b = castling_rook_squares
        if white_king:
            king_idx = white_king[0]
            pv = board_vector[king_idx]
            if (pv == 6.3 or pv == 6.1) and k_rook >= 0:
                edges.add((king_idx, 6))
                edges.add((k_rook, 5))
            if (pv == 6.3 or pv == 6.2) and q_rook >= 0:
                edges.add((king_idx, 2))
                edges.add((q_rook, 3))
        if black_king:
            king_idx = black_king[0]
            pv = board_vector[king_idx]
            if (pv == -6.3 or pv == -6.1) and k_rook_b >= 0:
                edges.add((king_idx, 62))
                edges.add((k_rook_b, 61))
            if (pv == -6.3 or pv == -6.2) and q_rook_b >= 0:
                edges.add((king_idx, 58))
                edges.add((q_rook_b, 59))
        return edges

    # Helper: file of square index (0-7)
    def _file(sq: int) -> int:
        return sq % 8

    # Legacy: infer rooks from board (with optional symmetric inference)
    if infer_symmetric and white_king and black_king:
        # Symmetric mode: when both have a right, use common file; skip if no common file
        w_pv = board_vector[white_king[0]]
        b_pv = board_vector[black_king[0]]
        wk_has = w_pv == 6.3 or w_pv == 6.1
        wq_has = w_pv == 6.3 or w_pv == 6.2
        bk_has = b_pv == -6.3 or b_pv == -6.1
        bq_has = b_pv == -6.3 or b_pv == -6.2

        w_k_rooks = [i for i in range(8) if board_vector[i] == 4 and _file(i) > _file(white_king[0])]
        w_q_rooks = [i for i in range(8) if board_vector[i] == 4 and _file(i) < _file(white_king[0])]
        b_k_rooks = [i for i in range(56, 64) if board_vector[i] == -4 and _file(i) > _file(black_king[0])]
        b_q_rooks = [i for i in range(56, 64) if board_vector[i] == -4 and _file(i) < _file(black_king[0])]

        # Kingside: both have right -> common file or skip
        if wk_has and bk_has:
            w_f = {_file(r) for r in w_k_rooks}
            b_f = {_file(r) for r in b_k_rooks}
            common = w_f & b_f
            if common:
                cf = random.choice(list(common))
                wr = next(r for r in w_k_rooks if _file(r) == cf)
                br = next(r for r in b_k_rooks if _file(r) == cf)
                edges.add((white_king[0], 6))
                edges.add((wr, 5))
                edges.add((black_king[0], 62))
                edges.add((br, 61))
        elif wk_has:
            if w_k_rooks:
                the_rook = random.choice(w_k_rooks)
                edges.add((white_king[0], 6))
                edges.add((the_rook, 5))
        elif bk_has:
            if b_k_rooks:
                the_rook = random.choice(b_k_rooks)
                edges.add((black_king[0], 62))
                edges.add((the_rook, 61))

        # Queenside: both have right -> common file or skip
        if wq_has and bq_has:
            w_f = {_file(r) for r in w_q_rooks}
            b_f = {_file(r) for r in b_q_rooks}
            common = w_f & b_f
            if common:
                cf = random.choice(list(common))
                wr = next(r for r in w_q_rooks if _file(r) == cf)
                br = next(r for r in b_q_rooks if _file(r) == cf)
                edges.add((white_king[0], 2))
                edges.add((wr, 3))
                edges.add((black_king[0], 58))
                edges.add((br, 59))
        elif wq_has:
            if w_q_rooks:
                the_rook = random.choice(w_q_rooks)
                edges.add((white_king[0], 2))
                edges.add((the_rook, 3))
        elif bq_has:
            if b_q_rooks:
                the_rook = random.choice(b_q_rooks)
                edges.add((black_king[0], 58))
                edges.add((the_rook, 59))
        return edges

    # Legacy non-symmetric: infer rooks independently (random when multiple)
    if black_king:
        rooks = [i for i, pv in enumerate(board_vector) if pv == -4 and i > 55]
        pv = board_vector[black_king[0]]
        if pv == -6.3 or pv == -6.2:
            cand = [r for r in rooks if r % 8 < black_king[0] % 8]
            if cand:
                the_rook = random.choice(cand)
                edges.add((black_king[0], 58))
                edges.add((the_rook, 59))
        if pv == -6.3 or pv == -6.1:
            cand = [r for r in rooks if r % 8 > black_king[0] % 8]
            if cand:
                the_rook = random.choice(cand)
                edges.add((black_king[0], 62))
                edges.add((the_rook, 61))
    if white_king:
        rooks = [i for i, pv in enumerate(board_vector) if pv == 4 and i < 8]
        pv = board_vector[white_king[0]]
        if pv == 6.3 or pv == 6.2:
            cand = [r for r in rooks if r % 8 < white_king[0] % 8]
            if cand:
                the_rook = random.choice(cand)
                edges.add((white_king[0], 2))
                edges.add((the_rook, 3))
        if pv == 6.3 or pv == 6.1:
            cand = [r for r in rooks if r % 8 > white_king[0] % 8]
            if cand:
                the_rook = random.choice(cand)
                edges.add((white_king[0], 6))
                edges.add((the_rook, 5))
    return edges


# ##### ##### ##### ##### #####
#       Helper functions

def coordinates_to_index(coordinates: Tuple[int, int]) -> int:
    """
    A quick helper function to transform the coordinates representation of a square into the index representation.
    :param coordinates: The coordinate pair to transform, in the format [row, column]
    :return: The index within the range [0, 63] that describes a specific square.
    """
    return (8 * coordinates[0]) + coordinates[1]


def remove_invalid_coordinates(coordinate_list: set[Tuple[int, int]]) -> set[Tuple[int, int]]:
    """
    Removes the invalid moves (those not falling on the standard 8x8 board) from a list.
    :param coordinate_list: The list of theoretical moves that may go off the board.
    :return: A subset of the coordinate_list where all coordinates are within the bounds of a chess board.
    """
    return {(row, column) for row, column in coordinate_list if all(0 <= coord <= 7 for coord in (row, column))}


def depth_first_recursive(visited: List[bool],
                          current_coordinates: Tuple[int, int],
                          edges: set[Tuple[int, int]],
                          get_neighbors: Callable[[Tuple[int, int]], set[Tuple[int, int]]]) \
                          -> set[Tuple[int, int]]:
    """
    The depth first graph traversal algorithm implemented recursively. Given a function to find the piece's neighbors.
    :param visited: A list representing the chess board that holds booleans for whether each square has been visited.
    :param current_coordinates: The coordinates of the square currently being analyzed.
    :param edges: A carry over variable to hold all of the edges.
    :param get_neighbors: A higher order function that returns a list of all squares the current piece type can move to.
    :return: A list of all possible paths that the piece type can take.
    """
    # Set the current square to visited
    visited[coordinates_to_index(current_coordinates)] = True

    # For each destination square in the set of valid neighbors
    for destination in get_neighbors(current_coordinates):
        # Create bidirectional connections from the current square to the destination
        edge_to: Tuple[int, int] = (coordinates_to_index(current_coordinates), coordinates_to_index(destination))
        edge_from: Tuple[int, int] = (coordinates_to_index(destination), coordinates_to_index(current_coordinates))
        # If we haven't recorded these connections before, append them
        if edge_to not in edges:
            edges.add(edge_to)
            edges.add(edge_from)

            # If we haven't visited this destination yet, recursively call the function for that one
            if not visited[coordinates_to_index(destination)]:
                edges.update(depth_first_recursive(visited=visited,
                                                   current_coordinates=destination,
                                                   edges=edges,
                                                   get_neighbors=get_neighbors))

    return edges


# Takes in a FEN string and returns a list of 64 numbers
# https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
def fen_to_vector(fen: str) -> List[float]:
    piece_values: Dict[str, int] = {"p": 1,
                                    "n": 2,
                                    "b": 3,
                                    "r": 4,
                                    "q": 5,
                                    "k": 6}

    # Split the fen by spaces
    fen_parts: List[str] = fen.split(" ")

    # X-FEN / Shredder-FEN may name a castling rook by its file letter (e.g. "Fd") when it is not
    # the outermost rook. Everything below only understands K/Q/k/q, so map letters onto sides.
    if len(fen_parts) > 2 and any(c in "ABCDEFGHabcdefgh" for c in fen_parts[2]):
        fen_parts = _normalize_castling_field(fen).split(" ")

    # The first part is the board portion. Split it by '/' to get each row
    row_strings: List[str] = fen_parts[0].split("/")

    # Initialize some variables for constructing the vector
    vector_version: List[float] = [0 for _ in range(64)]
    index_buffer_num_rows: int = 0

    # Put the player to move at the bottom (reflect vertically). Do not reverse files, so kingside/queenside stay correct.
    if fen_parts[1] == "w":
        row_strings = list(reversed(row_strings))
    # If black to move, row_strings stays in FEN order (rank 8 first) so black is at bottom; rows stay a->h (no horizontal flip).

    if fen_parts[1] not in ("w", "b"):
        print(f"Invalid FEN: {fen_parts[0]}")
        return [0]

    # Iterate over the rows backwards (start from row 1 and go up)
    current_row: str
    for current_row in row_strings:
        index_buffer_this_row: int = 0

        # Loop through each character in this row of the chess board
        index: int
        character: str
        for index, character in enumerate(current_row):
            # If the character is numerical...
            if character.isdigit():
                # Record the number of sequential empty squares
                index_buffer_this_row += int(character) - 1

                # For each of the empty squares
                index_to_zero: int
                for index_to_zero in range(int(character)):
                    # Set that index of the vector_version to zero
                    # (index_buffer_num_rows * 8) because we need to offset by the number of rows we've already done
                    # index because we need to offset by the number of characters in this row we've already done
                    # index_to_zero because we need to count up how many zeros we're adding based on the character
                    # (index_buffer_this_row - int(character) + 1) because ...
                    # if not first digit in row, need offset by more
                    vector_version[index + index_to_zero + (index_buffer_num_rows * 8) + (index_buffer_this_row - int(character) + 1)] = 0
            # If the character is alphabetical...
            else:
                # Set the value in the vector to the piece value
                true_index: int = index + (index_buffer_num_rows * 8) + index_buffer_this_row
                vector_version[true_index] = piece_values[character.lower()]

                # If the piece is black
                if character.islower():
                    # And the AI is playing as white
                    if fen_parts[1] == "w":
                        # Then multiply it by -1 because it's on the opponent's team
                        vector_version[true_index] *= -1
                # If the piece is white
                else:
                    # And the AI is playing as black
                    if fen_parts[1] == "b":
                        # Then multiply it by -1 because it's on the opponent's team
                        vector_version[true_index] *= -1

        index_buffer_num_rows += 1

    # ----- Castling -----

    # King value of +/-6 is modified by +/-0.1 and 0.2
    # This means the square can have 4 possible values:
    # 6: May not castle
    # 6.1: May castle king-side
    # 6.2: May castle queen-side
    # 6.3: May castle either side

    # Establish what side we're on so that we know if the king is a positive or negative number
    if fen_parts[1] == "w":
        white_mod: int = 1
        black_mod: int = -1
    else:
        white_mod: int = -1
        black_mod: int = 1

    # Since we may modify kings multiple times in a row...
    # Doing this work on separate list using indices of the old list so .index() works properly
    working_castle_vector = vector_version[:]

    # White may castle king-side
    if "K" in fen_parts[2]:
        working_castle_vector[vector_version.index(white_mod * 6)] += (white_mod * 0.1)
    # White may castle queen-side
    if "Q" in fen_parts[2]:
        working_castle_vector[vector_version.index(white_mod * 6)] += (white_mod * 0.2)
    # Black may castle king-side
    if "k" in fen_parts[2]:
        working_castle_vector[vector_version.index(black_mod * 6)] += (black_mod * 0.1)
    # Black may castle queen-side
    if "q" in fen_parts[2]:
        working_castle_vector[vector_version.index(black_mod * 6)] += (black_mod * 0.2)

    # Save changes to the original list
    vector_version = working_castle_vector[:]

    # Snap castling fractions (0.1 / 0.2 / 0.3) so float noise does not break == checks downstream
    for i, v in enumerate(vector_version):
        base = int(v)
        if base in (6, -6):
            frac = abs(v) - 6
            if frac > 1e-9:
                sign = 1 if v > 0 else -1
                vector_version[i] = sign * (6 + round(frac * 10) / 10)

    # ----- En Passant -----

    # If there is no En Passant, finish
    if fen_parts[3][0] == "-":
        return vector_version

    character_values = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
    ep_file: int = character_values[fen_parts[3][0]]
    ep_rank: int = int(fen_parts[3][1]) - 1  # 0-based
    if black_mod == 1:
        # Black to move: the board above was reflected vertically (rank 8 is row 0) with the files
        # left alone, so the en passant square is reflected the same way. (63 - index would be a
        # 180 degree rotation and would land on the mirrored file.)
        en_passant_square = (7 - ep_rank) * 8 + ep_file
    else:
        en_passant_square = ep_rank * 8 + ep_file

    # noinspection PyTypeChecker
    vector_version[en_passant_square] = -0.5

    return vector_version


def visualize_graph(edge_list: set[Tuple[int, int]]) -> None:
    # Create a directed graph
    G = nx.DiGraph()

    # Define positions for 8x8 grid
    pos = {i: (i % 8, i // 8) for i in range(64)}

    # Add nodes and edges
    G.add_nodes_from(range(64))
    G.add_edges_from(edge_list)

    # Draw the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', edge_color='black', arrows=True,
            font_size=8)

    # Show the visualization
    plt.show()


# ##### ##### ##### ##### #####
#       Program Body

if __name__ == "__main__":
    # Compute the graph


    # Save it
    #torch.save(computed_graph, 'blank_graph.pt')

    # Visualize the graph
    # 0 Pawn move
    # 1 Pawn attack
    # 2 Knight move
    # 3 Bishop move
    # 4 Rook move
    # 5 King move
    # 6 Queen move

    #visualize_graph(get_chess_graph_edges()[2])
    visualize_graph(get_castling_edges(fen_to_vector("rnr1k1nq/pp6/2p3p1/3P4/1b1PQ3/1PN1P3/P2N1P1P/R3KR2 b KQq - 0 15")))


    """
    # Visualize board perturbations
    # Example FEN: "r1bqk2r/p1ppbpp1/2n2n1p/Pp2p3/4P3/2N2N2/1PPPBPPP/R1BQK2R w Kk - 0 1" # En passant example
    # Example FEN: "r1bq1b1r/ppp3pp/2n1k3/3np3/2B5/5Q2/PPPP1PPP/RNB1K2R w K - 0 1" # Fried Liver Attack
    fen = "r1bqk2r/p1ppbpp1/2n2n1p/Pp2p3/4P3/2N2N2/1PPPBPPP/R1BQK2R w - - 0 1"
    board = chess.Board(fen, chess960=True)
    print(board)
    print(fen)
    print("\n")
    fen = perturb_position(fen, perturb_type=4, magnitude=1)
    board = chess.Board(fen, chess960=True)
    print(board)
    print(fen)
    """



