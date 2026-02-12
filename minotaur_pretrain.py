import pandas as pd
import random
import chess
from typing import Dict, List


# ##### ##### ##### ##### #####
# Functions

# A function that runs a monte carlo simulation on a list of chess positions
# randomly choosing a move for each position sim_num times
def _monte_carlo_sim(position_vector_batch_df: pd.DataFrame, sim_num: int) -> Dict[str, int]:
    """

    :param position_vector_batch_df:
    :param sim_num:
    :return:
    """
    # A dictionary to hold a count of randomly chosen / simulated moves
    sim_dict: Dict[str, int] = {}
    # A set to keep track of all legal moves that appear
    legal_set: set = set()

    # Simulate sim_num times
    for _ in range(0, sim_num):
        # Loop through each position
        for i in range(0, len(position_vector_batch_df)):
            # Get the current row
            this_row: pd.DataFrame = position_vector_batch_df.iloc[i]

            # Get the list of legal moves using the chess library and the FEN string, then convert SAN to UCI
            legal_moves: List[str] = get_legal_moves_strings(this_row["FEN"])

            # Make sure that these moves are in the legal_set
            legal_set.update(legal_moves)

            # Randomly choose a legal move
            sim_move: str = random.choice(legal_moves)
            # Increment or initialize the simulated move in sim_dict
            if sim_move in sim_dict:
                sim_dict[sim_move] += 1
            else:
                sim_dict[sim_move] = 1

    # Normalize the results
    move: str
    for move in sim_dict:
        sim_dict[move] /= sim_num

    # Make sure that the sim_dict contains all legal moves, even if it never picked some by chance
    for move in legal_set:
        if move not in sim_dict:
            sim_dict[move] = 0

    return sim_dict


#
def _score_random(position_vector_batch_df: pd.DataFrame, choices_dict: Dict[str, List[str]], illegal_move_penalty_factor: float = 1) -> float:
    """

    :param position_vector_batch_df:
    :param choices_dict:
    :param illegal_move_penalty_factor:
    :return:
    """

    total_penalty: float = 0
    total_reward: float = 0

    # Start by calculating the number of illegal moves and summing a penalty
    total_penalty += (illegal_move_penalty_factor *
                      position_vector_batch_df['FEN'].apply(lambda x: _get_num_invalid_moves(x, choices_dict[x])).sum())


    sim_dict: Dict[str, int] = _monte_carlo_sim(position_vector_batch_df, 1000)
    move: int
    for move in sim_dict:
        # Calculate the sum of absolute difference in values for choices_dict and sim_dict
        total_penalty +=

    return total_reward - total_penalty


# A function to return the number of invalid moves predicted by the network
def _get_num_invalid_moves(fen: str, moves: List[str]) -> int:
    """

    :param fen:
    :param moves:
    :return:
    """
    # If we've reached the base case with no moves left, then all moves were valid
    if not moves:
        return 0
    # If the first move in the list is legal
    if moves[0] in get_legal_moves_strings(fen):
        # Recursively call this function using the fen that results from that move and return the list shortened by 1
        return _get_num_invalid_moves(chess.Board(fen).push(chess.Move.from_uci(moves[0])).fen(), moves[1:])
    else:
        # If the move was illegal, then all moves left in the list are illegal
        return len(moves)


#
def get_legal_moves_strings(fen: str) -> List[str]:
    return [board.parse_san(move).uci() for move in list(chess.Board(fen).legal_moves)]





