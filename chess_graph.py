import chess.engine
import chess
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from typing import List, Tuple, Callable


def create_blank_chess_graphs() -> List[Tuple[torch.tensor, torch.tensor]]:
    """

    :return: List of tuples of tensors, which are a tensor for all of the edges for that lists type, and a tensor
     to contain the blank node features
    """

    # This holds all of the different information for each piece type
    # A function that returns a list for edges_list, and the corresponding value for edge_types_list
    pieces_list: List[Callable[Tuple[int, int],
                                     set[Tuple[int, int]]]] = [
        get_knight_neighbors,
        get_bishop_neighbors,
        get_rook_neighbors,
        get_king_neighbors
    ]

    # A list of lists of pairwise edges between chessboard squares 0 through 63
    edges_list: List[set[Tuple[int, int]]] = [get_pawn_move_edges(),    # 0 Pawn move
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
        edges_list[edges_index] = depth_first_recursive(visited=[False for _ in range(64)],
                                                        current_coordinates=(0, 0),
                                                        edges=set(),
                                                        get_neighbors=neighbor_function)
        # If this is the bishop, we need to perform another search for the light squares
        if edges_index == 3:
            edges_list[edges_index].update(depth_first_recursive(visited=[False for _ in range(64)],
                                                                 current_coordinates=(0, 1),
                                                                 edges=set(),
                                                                 get_neighbors=neighbor_function))
        edges_index += 1

    # Queen move edges are the union of bishop and rook moves
    edges_list[6] = edges_list[3].union(edges_list[4])

    # Don't add any castling edges because their existence depends on the position

    # This will be the return
    # List of tuples of tensors, which are the edges tensor and blank node features tensors
    empty_graphs: List[Tuple[torch.tensor, torch.tensor]] = [(0, 0) for _ in range(len(edges_list))]

    # Create the empty node features tensor: 64x13
    node_features = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(64)], dtype=torch.float)

    # Convert the pairwise edges to two lists for source and destination for compatibility with pytorch
    for index, this_type_edges in enumerate(edges_list):
        x_list: List[int] = []
        y_list: List[int] = []
        for x, y in this_type_edges:
            x_list.append(x)
            y_list.append(y)

        # Set this graph tuple
        empty_graphs[index] = (torch.tensor(data=[x_list, y_list], dtype=torch.long), node_features)

    return empty_graphs


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

    :return:
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
    been made and after the piece placement has been decided
    :param board_vector: A vector with values -6.3 to 6.3 representing the pieces on the board, plus
    castling and en passant.
    :return: A list of edge pairs that show where castling may be possible. (king g1,g8,c1,c8 and rook f1,f8,d1,d8)
    """
    edges: set[Tuple[int, int]] = set()

    for c in [1, -1]:
        # Find the square with the king
        for decimal in [0.1, 0.2, 0.3]:
            num = board_vector.index((6 + decimal) * c)
            if num :
                # If it returns an index, connect that to the rooks

                # Will this give me a list of indices?
                rook = board_vector.index(4 * c)

                # Connect king to the relevant side...
                # If I need to check what decimal is to know what rook to connect to, then maybe I should remove the loop

                # What if someone gets a pawn to the other side, chooses a rook, and moves it to their back rank?
                # Can I distinguish between the original rook??

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


# ##### ##### ##### ##### #####
#       Helper functions

def visualize_graph(tensor):
    # Extract edges from the tensor
    x_list, y_list = tensor.tolist()

    # Create a directed graph
    G = nx.DiGraph()

    # Define positions for 8x8 grid
    pos = {i: (i % 8, i // 8) for i in range(64)}

    # Add nodes and edges
    G.add_nodes_from(range(64))
    edges = list(zip(x_list, y_list))
    G.add_edges_from(edges)

    # Draw the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', edge_color='black', arrows=True,
            font_size=8)

    # Show the visualization
    plt.show()


#
def perturb_graph():
    # Randomly choose between 1 and X piece perturbations

    # Randomly choose to either perturb by legal moves or proximal squares for each piece perturbation
    # Randomly choose a magnitude for each piece perturbation
    # (?) Randomly decide to either swap or take resulting square (?)

    # Randomly step through each piece perturbation according to parameters

    # Return perturbed graph

    pass


# ##### ##### ##### ##### #####
#       Program Body

if __name__ == "__main__":
    # Compute the graph
    computed_graph = create_blank_chess_graphs()

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
    # 7 Castle
    visualize_graph(computed_graph[5][0])

    # Confirmed correct:
    # Pawn move edges
    # Pawn attack edges
    # Knight edges

    # Bishop move
    # Rook move
    # King move









