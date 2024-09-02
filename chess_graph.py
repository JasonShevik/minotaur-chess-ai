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
    edges_list: List[List[int]] = []
    # This list defines the type that each connection in edges_list is
    edge_types_list: List[int] = []

    # This holds all of the different information for each piece type
    # A function that returns a list for edges_list, and the corresponding value for edge_types_list
    pieces_list: List[Tuple[Callable[[], List[List[int]]], int]] = [
        (get_pawn_edges, 0),
        (get_knight_edges, 1),
        (get_light_bishop_edges, 2),  # Dark and light bishops have same connection type.
        (get_dark_bishop_edges, 2),  # No dark or light squares will be connected to each other, though.
        (get_rook_edges, 3),
        (get_queen_edges, 4),
        (get_king_edges, 5),
        (get_en_passant_edges, 6)
    ]

    def add_to_edges_list(the_list: List[List[int]], edge_function: Callable[[], List[List[int]]]) \
            -> Tuple[List[List[int]], int]:
        """
        This function appends all of the edges in the current edge_function to the_list, and returns the number
        of edges that there were.
        :param the_list:
        :param edge_function:
        :return:
        """
        count: int = 0
        for edge in edge_function():
            edge: List[int]
            the_list.append(edge)
            count += 1
        return the_list, count

    #
    for this_edge_function, piece_type in pieces_list:
        edges_list, num = add_to_edges_list(edges_list, this_edge_function)
        edge_types_list.extend([piece_type] * num)

    edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types_list, dtype=torch.long)

    # The blank chess graph will not include castling because that depends on the square that the king starts on.
    # A helper function get_castling_edges() should be called after adding piece placement to the graph.

    return edge_index, edge_type


# ##### ##### ##### ##### #####
#   Piece connection getters

def get_pawn_edges() -> List[List[int]]:
    """
    Deterministically find all paths where pawns can move.
    :return: A list of edge pairs that show where pawn moves may be possible.
    """
    #
    def add_column(col_num: int) -> List[List[int]]:
        pawn_column: List[List[int]] = []

        # Can move 2 squares on first move
        # From the front perspective
        pawn_column.append([col_num + (8 * 1), col_num + (8 * 3)])
        # From the back perspective
        pawn_column.append([col_num + (8 * 6), col_num + (8 * 4)])

        # All single square moves
        # From the front perspective
        for row_num in range(1, 7, 1):
            pawn_column.append([col_num + (8 * row_num), col_num + (8 * (row_num + 1))])
        # From the back perspective
        for row_num in range(6, 0, -1):
            pawn_column.append([col_num + (8 * row_num), col_num + (8 * (row_num - 1))])

        return pawn_column

    edges: List[List[int]] = []

    # Go through A through H for both sides.
    for column in range(0, 8, 1):
        edges.extend(add_column(column))

    return edges


def get_knight_edges() -> List[List[int]]:
    """

    :return:
    """
    def get_move_coordinates(start_coordinates: List[int]) -> List[List[int]]:
        """

        :param start_coordinates:
        :return:
        """

        return []

    def remove_invalid_coordinates(coordinate_list: List[List[int]]) -> List[List[int]]:
        """

        :param coordinate_list:
        :return:
        """

        return []

    edges: List[List[int]] = []

    # Implement depth first search
    current_square = [0, 0]
    for destination in remove_invalid_coordinates(get_move_coordinates(current_square)):
        edges.append([coordinates_to_index(current_square), coordinates_to_index(destination)])

    return edges


def get_light_bishop_edges() -> List[List[int]]:
    """

    :return:
    """
    edges: List[List[int]] = []

    return edges


def get_dark_bishop_edges() -> List[List[int]]:
    """

    :return:
    """
    edges: List[List[int]] = []

    return edges


def get_rook_edges() -> List[List[int]]:
    """

    :return:
    """
    edges: List[List[int]] = []

    return edges


def get_queen_edges() -> List[List[int]]:
    """

    :return:
    """
    edges: List[List[int]] = []

    return edges


def get_king_edges() -> List[List[int]]:
    """

    :return:
    """
    edges: List[List[int]] = []

    return edges


def get_en_passant_edges() -> List[List[int]]:
    """

    :return:
    """
    edges: List[List[int]] = []

    return edges


def get_castling_edges(board_vector) -> List[List[int]]:
    """
    This is a helper function to add castling edges to the graph. It needs to be done after the blank graph has
    been made and after the piece placement has been decided
    :param board_vector: A vector with values -6.3 to 6.3 representing the pieces on the board, plus
    castling and en passant.
    :return: A list of edge pairs that show where castling may be possible.
    """
    # Check if either side has castling rights. If not, return.
    if not any(abs(x) > 6 for x in board_vector):
        return []

    edges: List[List[int]] = []

    # Find the squares with the king and rooks
    # Connect them with castling edges

    return edges


# ##### ##### ##### ##### #####
#       Helper functions

def coordinates_to_index(coord_pair: List[int]) -> int:
    """

    :param coord_pair:
    :return:
    """

    return 0












