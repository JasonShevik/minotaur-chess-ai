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


def is_castling_right_plausible(board: chess.Board, key: str) -> bool:
    """
    Return True if the given castling right (K, Q, k, q) is still plausible on this board:
    the king is on the correct back rank and there is at least one rook of that color on
    the correct side of the king (kingside = right of king, queenside = left of king).
    We do not check for blocking pieces or attacks on the path; edges represent the
    right to castle, not whether castling is currently legal.
    """
    if key not in "KQkq":
        return False
    params = {k: (c, ks, kd, rd, r) for k, c, ks, kd, rd, r in _CASTLING_DESTS}
    color, kingside, king_dest, rook_dest, rank = params[key]
    king_sq = board.king(color)
    if king_sq is None or chess.square_rank(king_sq) != rank:
        return False
    kf = chess.square_file(king_sq)
    rooks_on_rank = [
        s for s in chess.SQUARES
        if chess.square_rank(s) == rank
        and board.piece_at(s) == chess.Piece(chess.ROOK, color)
    ]
    candidates = [s for s in rooks_on_rank if (chess.square_file(s) > kf if kingside else chess.square_file(s) < kf)]
    return len(candidates) > 0


def fix_castling_in_fen(fen_str: str) -> str:
    """
    Return FEN with castling string pruned to only plausible rights. Only removes rights
    that are no longer plausible; never adds castling. Call after any perturbation that
    might have made castling implausible (e.g. king/rook moved, piece deleted, etc.).
    """
    parts = fen_str.split()
    if len(parts) < 3:
        return fen_str
    current = set(c for c in parts[2] if c in "KQkq")
    if not current:
        return fen_str
    board = chess.Board(fen_str)
    plausible = {k for k in current if is_castling_right_plausible(board, k)}
    new_castling = "".join(c for c in "KQkq" if c in plausible) if plausible else "-"
    parts[2] = new_castling
    return " ".join(parts)


def perturb_position(
    fen: str,
    perturb_type: Optional[int] = None,
    magnitude: int = 1,
    type_distribution: Optional[Union[torch.Tensor, Callable[[], int]]] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Return a perturbed position as a new FEN string.

    :param fen: FEN string of the position to perturb.
    :param perturb_type: Which perturbation to apply (0..7). If None, one is chosen
        from type_distribution or uniformly at random.
    :param magnitude: Interpretation depends on perturb_type, but it is the size 
        of a single perturbation, not the total number of perturbations.
    :param type_distribution: When perturb_type is None, how to choose the type.
        - If a 1D tensor of shape (num_types,): if all values are in [0, 1], treated
          as probabilities (normalized by sum); otherwise treated as logits (softmax).
          Sampled via torch.multinomial.
        - If a callable: call with no args to get an int in [0, num_types-1].
    :param rng: Optional random.Random for reproducible sampling when using
        type_distribution tensor.
    :return: Perturbed position as a FEN string.
    """

    def perturb_fen_piece_move_legal(fen_str: str) -> str:      # ----- Perturbation Type 0 -----
        # Do 'magnitude' legal moves of the same piece in a row. The piece can be of either color;
        # we temporarily set the board's turn to that piece's color to get legal moves, then restore
        # the original turn after each move so the final FEN keeps e.g. white to move.
        rand = rng if rng is not None else random
        n = max(1, int(magnitude))
        board = chess.Board(fen_str)
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
                return fix_castling_in_fen(current.fen())
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
                    return fix_castling_in_fen(next_board.fen())
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
        # illegal destination (starting square becomes empty).
        rand = rng if rng is not None else random
        board = chess.Board(fen_str)
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
            return fix_castling_in_fen(board.fen())
        board.turn = turn_white
        return fen_str

    def perturb_fen_piece_deletion(fen_str: str) -> str:      # ----- Perturbation Type 2 -----
        # Delete exactly one "thing" at random: any piece (either color, including kings) or the
        # en passant target square if present. Magnitude is ignored.
        rand = rng if rng is not None else random
        board = chess.Board(fen_str)
        options: List[Optional[chess.Square]] = [
            s for s in chess.SQUARES if board.piece_at(s) is not None
        ]
        if board.ep_square is not None:
            options.append(None)  # sentinel: clear en passant
        if not options:
            return fen_str
        choice = rand.choice(options)
        if choice is None:
            board.ep_square = None
            return fix_castling_in_fen(board.fen())
        else:
            board.remove_piece_at(choice)
            return fix_castling_in_fen(board.fen())

    def perturb_fen_piece_addition(fen_str: str) -> str:      # ----- Perturbation Type 3 -----
        # Add one thing at random: a piece (any type and color) on an empty square, or an en passant
        # target. If en passant is not already set, possible ep squares are inferred from 4th/5th rank
        # pawn pairs (adjacent files with one white and one black pawn). Magnitude is ignored.
        rand = rng if rng is not None else random
        board = chess.Board(fen_str)
        options: List[Tuple[str, Any]] = []

        # Possible en passant squares only when ep is not already set (each square at most once).
        # Only consider the rank where the side to move could capture en passant: Black to move -> 4th rank (white just moved); White to move -> 5th rank (black just moved).
        possible_ep_squares: set[chess.Square] = set()
        if board.ep_square is None:
            if board.turn == chess.BLACK:
                # 4th rank: white could have just moved two -> 3rd rank ep squares (Black captures)
                for f in range(7):
                    sq_a, sq_b = chess.square(f, 3), chess.square(f + 1, 3)
                    pa, pb = board.piece_at(sq_a), board.piece_at(sq_b)
                    if pa is not None and pb is not None and pa.piece_type == chess.PAWN and pb.piece_type == chess.PAWN:
                        if pa.color != pb.color:
                            ep_sq = chess.square(f, 2) if pa.color == chess.WHITE else chess.square(f + 1, 2)
                            if board.piece_at(ep_sq) is None:
                                possible_ep_squares.add(ep_sq)
            else:
                # 5th rank: black could have just moved two -> 6th rank ep squares (White captures)
                for f in range(7):
                    sq_a, sq_b = chess.square(f, 4), chess.square(f + 1, 4)
                    pa, pb = board.piece_at(sq_a), board.piece_at(sq_b)
                    if pa is not None and pb is not None and pa.piece_type == chess.PAWN and pb.piece_type == chess.PAWN:
                        if pa.color != pb.color:
                            ep_sq = chess.square(f, 5) if pa.color == chess.BLACK else chess.square(f + 1, 5)
                            if board.piece_at(ep_sq) is None:
                                possible_ep_squares.add(ep_sq)
            for ep_sq in possible_ep_squares:
                options.append(("ep", ep_sq))

        # All piece additions: each empty square × each (piece_type, color)
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        for s in chess.SQUARES:
            if board.piece_at(s) is not None:
                continue
            for pt in piece_types:
                for color in (chess.WHITE, chess.BLACK):
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
        return fix_castling_in_fen(board.fen())

    def perturb_fen_piece_swap(fen_str: str) -> str:      # ----- Perturbation Type 4 -----
        # Swap two randomly chosen pieces that differ in type or color (so the position actually changes).
        # Magnitude ignored. No en passant.
        rand = rng if rng is not None else random
        board = chess.Board(fen_str)
        occupied = [s for s in chess.SQUARES if board.piece_at(s) is not None]
        if len(occupied) < 2:
            return fen_str
        if len({(board.piece_at(s).piece_type, board.piece_at(s).color) for s in occupied}) < 2:
            return fen_str
        while True:
            sq1, sq2 = rand.sample(occupied, 2)
            p1, p2 = board.piece_at(sq1), board.piece_at(sq2)
            if p1.piece_type != p2.piece_type or p1.color != p2.color:
                break
        board.remove_piece_at(sq1)
        board.remove_piece_at(sq2)
        board.set_piece_at(sq1, p2)
        board.set_piece_at(sq2, p1)
        return fix_castling_in_fen(board.fen())

    def perturb_fen_piece_change(fen_str: str) -> str:      # ----- Perturbation Type 5 -----
        # Pick a random piece on the board and change it to a random different piece type (same color).
        # En passant not considered; all pieces except kings count. Magnitude ignored.
        rand = rng if rng is not None else random
        board = chess.Board(fen_str)
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        occupied = [s for s in chess.SQUARES if (board.piece_at(s) is not None and board.piece_at(s).piece_type != chess.KING)]
        if not occupied:
            return fen_str
        square = rand.choice(occupied)
        piece = board.piece_at(square)
        other_types = [t for t in piece_types if t != piece.piece_type]
        new_type = rand.choice(other_types)
        board.set_piece_at(square, chess.Piece(new_type, piece.color))
        return fix_castling_in_fen(board.fen())

    def perturb_fen_piece_color_change(fen_str: str) -> str:      # ----- Perturbation Type 6 -----
        # Pick a random piece (not a king; en passant not considered) and flip its color.
        rand = rng if rng is not None else random
        board = chess.Board(fen_str)
        swappable = [s for s in chess.SQUARES if board.piece_at(s) is not None and board.piece_at(s).piece_type != chess.KING]
        if not swappable:
            return fen_str
        square = rand.choice(swappable)
        piece = board.piece_at(square)
        new_color = chess.BLACK if piece.color == chess.WHITE else chess.WHITE
        board.set_piece_at(square, chess.Piece(piece.piece_type, new_color))
        return fix_castling_in_fen(board.fen())

    def perturb_fen_castling_rights_change(fen_str: str) -> str:     # ----- Perturbation Type 7 -----
        # One of: remove an existing castling right, or add a castling right if sensical (Chess960-aware).
        # Uses is_castling_right_plausible for add-check; final FEN is cleaned with fix_castling_in_fen.
        rand = rng if rng is not None else random
        board = chess.Board(fen_str)

        castling_str = board.fen().split()[2]
        current = set(c for c in castling_str if c in "KQkq")
        actions: List[Tuple[str, str]] = []  # ('remove'|'add', 'K'|'Q'|'k'|'q')
        for key in "KQkq":
            if key in current:
                actions.append(("remove", key))
            elif is_castling_right_plausible(board, key):
                actions.append(("add", key))

        if not actions:
            return fix_castling_in_fen(fen_str)
        op, key = rand.choice(actions)
        if op == "remove":
            new_set = current - {key}
        else:
            new_set = current | {key}
        new_castling = "".join(c for c in "KQkq" if c in new_set) if new_set else "-"
        parts = board.fen().split()
        parts[2] = new_castling
        return fix_castling_in_fen(" ".join(parts))

    perturbation_dispatch_table: Dict[int, Callable[[str], str]] = {
        0: perturb_fen_piece_move_legal,
        1: perturb_fen_piece_move_illegal,
        2: perturb_fen_piece_deletion,
        3: perturb_fen_piece_addition,
        4: perturb_fen_piece_swap,
        5: perturb_fen_piece_change,
        6: perturb_fen_piece_color_change,
        7: perturb_fen_castling_rights_change,
    }

    num_types = len(perturbation_dispatch_table)

    # Choose perturbation type if not specified
    if perturb_type is None:
        if type_distribution is None:
            perturb_type = random.randint(0, num_types - 1) if rng is None else rng.randint(0, num_types - 1)
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
                perturb_type = random.randint(0, num_types - 1) if rng is None else rng.randint(0, num_types - 1)
    perturb_type = int(perturb_type) % num_types

    return perturbation_dispatch_table[perturb_type](fen)


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

    # Remove the decimals so that the integers can be used in calculating index values
    position_vector: List[int] = list(map(math.floor, position_vector))

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

    # ----- En Passant -----

    # If there is no En Passant, finish
    if fen_parts[3][0] == "-":
        return vector_version

    character_values = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
    en_passant_square = ((int(fen_parts[3][1]) - 1) * 8) + character_values[fen_parts[3][0]]
    if black_mod == 1:
        en_passant_square = 63 - en_passant_square

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
    board = chess.Board(fen)
    print(board)
    print(fen)
    print("\n")
    fen = perturb_position(fen, perturb_type=4, magnitude=1)
    board = chess.Board(fen)
    print(board)
    print(fen)
    """



