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
    edges_list: List[List[int]] = []
    edge_types_list: List[int] = []

    pieces_list: List[Tuple[Callable[[], List[List[int]]], int]] = [
        (get_pawn_edges, 0),
        (get_knight_edges, 1),
        (get_light_bishop_edges, 2),
        (get_dark_bishop_edges, 3),
        (get_rook_edges, 4),
        (get_queen_edges, 5),
        (get_king_edges, 6),
        (get_en_passant_edges, 7)
    ]

    def add_to_edges_list(the_list: List[List[int]], edge_function: Callable[[], List[List[int]]]) \
            -> Tuple[List[List[int]], int]:
        count = 0
        for edge in edge_function():
            the_list.append(edge)
            count += 1
        return the_list, count

    for this_edge_function, piece_type in pieces_list:
        edges_list, num = add_to_edges_list(edges_list, this_edge_function)
        edge_types_list.extend([piece_type] * num)

    edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types_list, dtype=torch.long)

    # The blank chess graph will not include castling because that depends on the square that the king is on.
    # A helper function get_castling_edges() should be called after adding piece placement to the graph.

    return edge_index, edge_type


def get_pawn_edges() -> List[List[int]]:
    """
    Deterministically find all paths where pawns can move.
    :return: A list of edge pairs that show where pawn moves may be possible.
    """
    edges: List[List[int]] = []

    # Go through A through H for both sides. Include that you can move 1 or 2 on the first move.

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






