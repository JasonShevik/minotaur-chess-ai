import chess.engine
import chess
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Callable


def create_blank_chess_graph() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Used to create a graph of a chess board that shows where moves may be possible not including piece locations.
    :return: A tuple of edge_index and edge_type which are used to create a homogenous graph in PyTorch Geometric.
    """
    # This list defines the connections that exist
    edges_list: List[Tuple[int, int]] = []
    # This list defines the type that each connection in edges_list is
    edge_types_list: List[int] = []

    # This holds all of the different information for each piece type
    # A function that returns a list for edges_list, and the corresponding value for edge_types_list
    pieces_list: List[Tuple[Callable[[Tuple[int, int]],
                                      List[Tuple[int, int]]],
                            Tuple[int, int],
                            int]] = [
        (get_pawn_edges,        (1, 0),  0),
        (get_knight_neighbors,  (0, 0),  1),
        (get_bishop_neighbors,  (0, 1),  2), # Dark and light bishops have same connection type.
        (get_bishop_neighbors,  (0, 0),  2), # No dark or light squares will be connected to each other, though.
        (get_rook_neighbors,    (0, 0),  3),
        (get_queen_neighbors,   (0, 0),  4),
        (get_king_neighbors,    (0, 0),  5),
        (get_en_passant_edges,  (3, 0),  6)
    ]

    # Go through the structure, calling each piece function and updating edges_list and edge_types_list
    for neighbor_function, start_square, piece_type in pieces_list:
        # Pawn edges are hard coded and don't use DFS
        if piece_type == 0:
            new_edges: List[Tuple[int, int]] = get_pawn_edges()
        # En Passant edges are hard coded and don't use DFS
        elif piece_type == 6:
            new_edges: List[Tuple[int, int]] = get_en_passant_edges()
        else:
            # Get the list of new edges that are specific to this piece_type
            new_edges: List[Tuple[int, int]] = depth_first_recursive(visited=[False for _ in range(64)],
                                                                     current_coordinates=start_square,
                                                                     edges=[],
                                                                     get_neighbors=neighbor_function)

        # Extend the edges_list to contain all of the newly gotten edges
        edges_list.extend(new_edges)
        # Extend the edge_types_list to tell us the piece_type for all the new edges that were just added
        edge_types_list.extend([piece_type] * len(new_edges))

    edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types_list, dtype=torch.long)

    # The blank chess graph will not include castling because that depends on the square that the king starts on.
    # A helper function get_castling_edges() should be called after adding piece placement to the graph.

    return edge_index, edge_type


# ##### ##### ##### ##### #####
#   Piece connection getters

def get_pawn_edges() -> List[Tuple[int, int]]:
    """
    Deterministically find all paths where pawns can move.
    :return: A list of edges that show where all pawn moves may be possible.
    """
    #
    def add_column(col_num: int) -> List[Tuple[int, int]]:
        # Can move 2 squares on first move
        # Front and back perspective
        pawn_column: List[Tuple[int, int]] = [(col_num + (8 * 1), col_num + (8 * 3)),
                                              (col_num + (8 * 6), col_num + (8 * 4))]

        # All single square moves
        for row_num in range(1, 7, 1): # From the front perspective
            pawn_column.append((col_num + (8 * row_num), col_num + (8 * (row_num + 1))))
        for row_num in range(6, 0, -1): # From the back perspective
            pawn_column.append((col_num + (8 * row_num), col_num + (8 * (row_num - 1))))

        return pawn_column

    # Go through A through H for both sides.
    edges: List[Tuple[int, int]] = []
    for column in range(8):
        edges.extend(add_column(column))

    return edges


def get_knight_neighbors(start_coordinates: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Gets the list of all possible knight moves from this position if the board were infinite.
    :param start_coordinates: The coordinates that the knight starts on in the format [row, column].
    :return: A lit of coordinates of where the knight could move from the start given an infinite board.
    """
    row: int
    column: int
    row, column = start_coordinates

    return remove_invalid_coordinates([(row + 2, column + 1), # Two up, one over
                                       (row + 2, column - 1),
                                       (row + 1, column + 2), # Two over, one up
                                       (row + 1, column - 2),
                                       (row - 1, column + 2), # Two over, one down
                                       (row - 1, column - 2),
                                       (row - 2, column + 1), # Two down, one over
                                       (row - 2, column - 1)])


def get_bishop_neighbors(start_coordinates: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Gets the list of all possible light bishop moves from this position.
    :param start_coordinates: The coordinates that the light bishop starts on in the format [row, column].
    :return: A lit of coordinates of where the light bishop could move from the start.
    """
    row: int
    column: int
    row, column = start_coordinates

    neighbors: List[Tuple[int, int]] = []
    for count in range(1, 8):
        neighbors.extend([(row + count, column + count),  # NE Direction
                          (row - count, column + count),  # SE
                          (row - count, column - count),  # SW
                          (row + count, column - count)]) # NW

    return remove_invalid_coordinates(neighbors)


def get_rook_neighbors(start_coordinates: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Gets the list of all possible rook moves from this position.
    :param start_coordinates: The coordinates that the rook starts on in the format [row, column].
    :return: A lit of coordinates of where the rook could move from the start.
    """
    row: int
    column: int
    row, column = start_coordinates

    neighbors: List[Tuple[int, int]] = []
    for count in range(1, 8):
        neighbors.extend([(row + count, column        ),  # N Direction
                          (row,         column + count),  # E
                          (row - count, column        ),  # S
                          (row,         column - count)]) # W

    return remove_invalid_coordinates(neighbors)


def get_queen_neighbors(start_coordinates: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Gets the list of all possible queen moves from this position.
    :param start_coordinates: The coordinates that the queen starts on in the format [row, column].
    :return: A lit of coordinates of where the queen could move from the start.
    """
    return get_bishop_neighbors(start_coordinates) + get_rook_neighbors(start_coordinates)


def get_king_neighbors(start_coordinates: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Gets the list of all possible king moves from this position.
    :param start_coordinates: The coordinates that the king starts on in the format [row, column].
    :return: A lit of coordinates of where the king could move from the start.
    """
    row: int
    column: int
    row, column = start_coordinates

    return remove_invalid_coordinates([(row + row_offset, column + column_offset)
                                        for row_offset in [-1, 0, 1]
                                        for column_offset in [-1, 0, 1]
                                        if not (row_offset == 0 and column_offset == 0)])


def get_en_passant_edges() -> List[Tuple[int, int]]:
    """
    Gets the list of all possible en passant moves.
    :return: A list of edges that show where all en passant moves may be possible.
    """
    def get_diagonal_left_and_right(start: int) -> List[Tuple[int, int]]:
        """
        Gets the en passant edges for a specific square
        :param start: The index of the start square
        :return: A list of all of the en passant edges stemming from the start square
        """
        # Determine if we're going toward rank 6 or rank 3
        if start >= 32:
            spot: int = start + 8
        else:
            spot: int = start - 8

        # Add all of the (correct) edges to the list
        piece_edges: List[Tuple[int, int]] = []
        for destination in [x for x in [spot + 1, spot - 1] if (x != 39 and x != 24)]:
            piece_edges.append((start, destination))

        return piece_edges

    edges: List[Tuple[int, int]] = []

    # For every square on the fifth rank
    for square in range(32, 39, 1): # (your perspective)
        edges.extend(get_diagonal_left_and_right(start=square))
    for square in range(24, 31, 1): # (opponent's perspective)
        edges.extend(get_diagonal_left_and_right(start=square))

    return edges


def get_castling_edges(board_vector: List[int]) -> List[Tuple[int, int]]:
    """
    This is a helper function to add castling edges to the graph. It needs to be done after the blank graph has
    been made and after the piece placement has been decided
    :param board_vector: A vector with values -6.3 to 6.3 representing the pieces on the board, plus
    castling and en passant.
    :return: A list of edge pairs that show where castling may be possible.
    """
    edges: List[Tuple[int, int]] = []

    # Check if either side has castling rights. If so, we need to add edges. Otherwise, return the empty list.
    if any(abs(x) > 6 for x in board_vector):
        # Find the squares with the king and rooks
        # Connect them with castling edges

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


def remove_invalid_coordinates(coordinate_list: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Removes the invalid moves (those not falling on the standard 8x8 board) from a list.
    :param coordinate_list: The list of theoretical moves that may go off the board.
    :return: A subset of the coordinate_list where all coordinates are within the bounds of a chess board.
    """
    return [(row, column) for row, column in coordinate_list if all(0 <= coord <= 7 for coord in (row, column))]


def depth_first_recursive(visited: List[bool],
                          current_coordinates: Tuple[int, int],
                          edges: List[Tuple[int, int]],
                          get_neighbors: Callable[[Tuple[int, int]], List[Tuple[int, int]]]) \
                          -> List[Tuple[int, int]]:
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
            edges.append(edge_to)
            edges.append(edge_from)

            # If we haven't visited this destination yet, recursively call the function for that one
            if not visited[coordinates_to_index(destination)]:
                edges.extend(depth_first_recursive(visited=visited,
                                                   current_coordinates=destination,
                                                   edges=edges,
                                                   get_neighbors=get_neighbors))

    return edges


# ##### ##### ##### ##### #####
#       Program Body

if __name__ == "__main__":
    # Compute the graph and save it so that I don't have to compute it again.












