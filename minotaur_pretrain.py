import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import threading
import random
import chess


# ##### ##### ##### ##### #####
# Functions

# A function that runs a monte carlo simulation on a list of chess positions
# randomly choosing a move for each position sim_num times
def _monte_carlo_sim(position_vector_batch_df, sim_num):
    """

    :param position_vector_batch_df:
    :param sim_num:
    :return:
    """
    # A dictionary to hold a count of randomly chosen / simulated moves
    sim_dict = {}
    # A set to keep track of all legal moves that appear
    legal_set = set()

    # Simulate sim_num times
    for _ in range(0, sim_num):
        # Loop through each position
        for i in range(0, len(position_vector_batch_df)):
            # Get the current row
            this_row = position_vector_batch_df.iloc[i]

            # Get the list of legal moves using the chess library and the FEN string
            legal_moves = list(chess.Board(this_row["FEN"]).legal_moves)
            # Convert the SAN format moves to UCI
            legal_moves = [board.parse_san(move).uci() for move in legal_moves]

            # Make sure that these moves are in the legal_set
            legal_set.update(legal_moves)

            # Randomly choose a legal move
            sim_move = random.choice(legal_moves)
            # Increment or initialize the simulated move in sim_dict
            if sim_move in sim_dict:
                sim_dict[sim_move] += 1
            else:
                sim_dict[sim_move] = 1

    # Normalize the results
    for move in sim_dict:
        sim_dict[move] /= sim_num

    # Make sure that the sim_dict contains all legal moves, even if it never picked some by chance
    for move in legal_set:
        if move not in sim_dict:
            sim_dict[move] = 0

    return sim_dict


#
def _score_random(position_vector_batch_df, choices_dict, illegal_move_penalty_factor=1):
    """

    :param position_vector_batch_df:
    :param choices_dict:
    :param illegal_move_penalty_factor:
    :return:
    """

    total_penalty = 0
    total_reward = 0

    # Start by calculating the number of illegal moves and summing a penalty
    total_penalty += (illegal_move_penalty_factor *
                      position_vector_batch_df['FEN'].apply(lambda x: _get_num_invalid_moves(x, choices_dict[x])).sum())


    sim_dict = _monte_carlo_sim(position_vector_batch_df, 1000)
    for move in sim_dict:
        # Calculate the sum of absolute difference in values for choices_dict and sim_dict
        total_penalty +=

    return total_reward - total_penalty


# A function to return the number of invalid moves predicted by the network
def _get_num_invalid_moves(fen, moves):
    """

    :param fen:
    :param moves:
    :return:
    """
    # If we've reached the base case with no moves left, then all moves were valid
    if not moves:
        return 0
    # If the first move in the list is legal
    if moves[0] in list(chess.Board(fen).legal_moves):
        # Recursively call this function using the fen that results from that move and return the list shortened by 1
        return _get_num_invalid_moves(chess.Board(fen).push(chess.Move.from_uci(moves[0])).fen(), moves[1:])
    else:
        # If the move was illegal, then all moves left in the list are illegal
        return len(moves)