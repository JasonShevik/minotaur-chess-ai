import chess.engine
import chess
import torch
import math
import random
import helper_utils as hu
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
    :param magnitude: Interpretation depends on perturb_type.
    :param type_distribution: When perturb_type is None, how to choose the type.
        - If a 1D tensor of shape (num_types,): if all values are in [0, 1], treated
          as probabilities (normalized by sum); otherwise treated as logits (softmax).
          Sampled via torch.multinomial.
        - If a callable: call with no args to get an int in [0, num_types-1].
    :param rng: Optional random.Random for reproducible sampling when using
        type_distribution tensor.
    :return: Perturbed position as a FEN string.
    """

    def perturb_fen_piece_move_legal(fen_str: str) -> str:
        # Do 'magnitude' legal moves of the same piece in a row (same side to move after each).
        # Exclude moves that transpose back to a square the piece has already visited.
        rand = rng if rng is not None else random
        n = max(1, int(magnitude))
        board = chess.Board(fen_str)
        turn_white = board.turn
        color = chess.WHITE if turn_white else chess.BLACK
        order = [s for s in chess.SQUARES if board.piece_at(s) is not None and board.color_at(s) == color] # List of all of our squares
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
                return current.fen()
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
                    return next_board.fen()
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

    def perturb_fen_piece_move_illegal(fen_str: str) -> str:
        # TODO: move a piece to a square it can reach (neighbor) but that is not legal (blocked, check, etc.)
        return fen_str

    def perturb_fen_piece_deletion(fen_str: str) -> str:
        # TODO: remove one or more pieces
        return fen_str

    def perturb_fen_piece_addition(fen_str: str) -> str:
        # TODO: add a piece on an empty square
        return fen_str

    def perturb_fen_piece_swap(fen_str: str) -> str:
        # TODO: swap two pieces
        return fen_str

    def perturb_fen_piece_change(fen_str: str) -> str:
        # TODO: change piece type (e.g. knight -> bishop)
        return fen_str

    def perturb_fen_piece_color_change(fen_str: str) -> str:
        # TODO: flip color of a piece
        return fen_str

    def perturb_fen_castling_rights_change(fen_str: str) -> str:
        # TODO: change castling rights
        return fen_str

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


def create_filled_chess_graphs(fen: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    A function to get the graph representations of all piece interactions for a specified chess position.
    :param fen: The string that identifies the position to make graphs for.
    :return: A tuple with the information needed for all of the graph networks. The first value in the tuple is
        a list of edge index tensors, one for a graph for each piece movement type. The second value in the tuple
        is the tensor with all of the node features, which includes piece locations and en passant information.
    """
    # Create a list of 64 floats that contains the information from the fen
    position_vector: List[float] = hu.fen_to_vector(fen)

    # Initialize the working variables that I will eventually return
    edges_lists: List[set[Tuple[int, int]]] = get_chess_graph_edges()
    # Add the castling edges
    edges_lists.append(get_castling_edges(position_vector))

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


def get_castling_edges(board_vector: List[float]) -> set[Tuple[int, int]]:
    """
    This is a helper function to add castling edges to the graph. It needs to be done after the blank graph has
    been made and after the piece placement has been decided.

    6.1: May castle king-side
    6.2: May castle queen-side
    6.3: May castle either side
    :param board_vector: A vector with values -6.3 to 6.3 representing the pieces on the board, plus
    castling and en passant.
    :return: A list of edge pairs that show where castling may be possible. (king g1,g8,c1,c8 and rook f1,f8,d1,d8)
    """
    edges: set[Tuple[int, int]] = set()

    # ----- Enemy -----
    castleable_king = [i for i, piece_value in enumerate(board_vector) if int(piece_value) == -6 and piece_value % 1]
    if castleable_king:
        rooks = [i for i, piece_value in enumerate(board_vector) if piece_value == -4]
        rooks = [x for x in rooks if x > 55]

        # Queen Side
        if board_vector[castleable_king[0]] == -6.3 or board_vector[castleable_king[0]] == -6.2:
            the_rook = random.choice([rook for rook in rooks if rook < castleable_king[0]])
            edges.add((castleable_king[0], 58))
            edges.add((the_rook, 59))

        # King side
        if board_vector[castleable_king[0]] == -6.3 or board_vector[castleable_king[0]] == -6.1:
            the_rook = random.choice([rook for rook in rooks if rook > castleable_king[0]])
            edges.add((castleable_king[0], 62))
            edges.add((the_rook, 61))

    # ----- Ally -----
    castleable_king = [i for i, piece_value in enumerate(board_vector) if int(piece_value) == 6 and piece_value % 1]
    if castleable_king:
        rooks = [i for i, piece_value in enumerate(board_vector) if piece_value == 4]
        rooks = [x for x in rooks if x < 8]

        # Queen Side
        if board_vector[castleable_king[0]] == 6.3 or board_vector[castleable_king[0]] == 6.2:
            the_rook = random.choice([rook for rook in rooks if rook < castleable_king[0]])
            edges.add((castleable_king[0], 2))
            edges.add((the_rook, 3))

        # King side
        if board_vector[castleable_king[0]] == 6.3 or board_vector[castleable_king[0]] == 6.1:
            the_rook = random.choice([rook for rook in rooks if rook > castleable_king[0]])
            edges.add((castleable_king[0], 6))
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

    visualize_graph(get_chess_graph_edges()[2])
    #visualize_graph(get_castling_edges(hu.fen_to_vector("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")))

    # Confirmed correct:
    # Pawn move edges
    # Pawn attack edges
    # Knight edges

    # Bishop move
    # Rook move
    # King move









