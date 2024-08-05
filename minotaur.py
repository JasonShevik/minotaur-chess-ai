import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import threading
import random
import chess


# ##### ##### ##### ##### #####
# Functions

# Pretraining Functions:

# A function that runs a monte carlo simulation on a list of chess positions
# randomly choosing a move for each position sim_num times
def _monte_carlo_sim(position_vector_batch_df, sim_num):
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
def _score_random(position_vector_batch_df, choices_dict):
    total_penalty = 0
    total_reward = 0

    # Start by calculating the number of illegal moves and summing a penalty

    sim_dict = _monte_carlo_sim(position_vector_batch_df, 1000)

    for move in sim_dict:
        # Calculate the sum of absolute difference in values for choices_dict and sim_dict
        total_penalty +=

    return total_reward - total_penalty


# Reinforcement Learning Testing Functions:

def analyze_distribution(method="uniform"):

    generated_points = []

    # REDO THIS using lambda functions
    if method == "uniform":
        for _ in range(1000):
            generated_points.append(get_random_vector_uniform(3, 1))
    elif method == "spherical_coordinates":
        for _ in range(1000):
            generated_points.append(get_random_vector_spherical_coordinates(3, 1))
    elif method == "gaussian_noise":
        for _ in range(1000):
            generated_points.append(get_random_vector_gaussian_noise(3, 1))

    # Find a bunch of statistics about the points such as:
        # Average distance to nearest neighbor
        # Average total distance from all other points

        # Average distance from the origin

    # Scale the points outward to the surface of the sphere and add those points to a new list
    surface_points = []

    # Calculate some of the same statistics as before for the new points

    # Make a heatmap of point density on the surface of the sphere, transform, and plot it


def get_random_vector_uniform(num_dimensions, max_radius):
    """
    https://mathworld.wolfram.com/HyperspherePointPicking.html

    :param num_dimensions:
    :param max_radius:
    :return:
    """


def get_random_vector_spherical_coordinates(num_dimensions, max_radius):
    """
    Picks a random dimension for a starting vector with radius max_radius.
    It will then apply random rotations in every remaining dimension in a random order.
    Then it will randomly scale the point inward.

    :param num_dimensions:
    :param max_radius:
    :return:
    """


def get_random_vector_gaussian_noise(num_dimensions, max_radius):
    """
    Random walk where each step is drawn from gaussian distribution.
    Results in points centered around original point.

    :param num_dimensions:
    :param max_radius:
    :return:
    """


def get_nearby_net(input_net, dist_type, max_radius):
    """

    :param input_net:
    :param dist_type:
    :param max_radius:
    :return:
    """
    # Flatten the neural net
    # Get the length of the resulting 1D vector
    # Pass the length to the get_random_vector function specified by dist_type
    # Apply the random vector to the flattened vector
    # Unflatten the resulting vector
    # Return


# ##### ##### ##### ##### #####
# Architecture

class Minotaur(nn.Module):
    def __init__(self, mode="normal", pretrain_model=None):
        super(Minotaur, self).__init__()
        self.input_layer = nn.Linear(64, 128)  # 64 squares input
        self.hidden_layers = nn.ModuleList([nn.Linear(128, 128) for _ in range(99)])

        if mode == "pretrain":
            self.output_layer = nn.Linear(128, 40)
        elif mode == "normal":
            if pretrain_model is not None:
                pretrain_dict = {k: v for k, v in pretrain_model.state_dict().items() if "output_layer" not in k}
                self.load_state_dict(pretrain_dict, strict=False)
            self.output_layer = nn.Linear(128, 4)
        else:
            # Handle an exception?
            self.output_layer = nn.Linear(128, 4)

    #
    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = torch.tanh(self.output_layer(x))
        x = (x + 1) * 3.5 + 1  # Transforms: [-1, 1] -> [1, 8]
        return x.round().tolist()

# ##### ##### ##### ##### #####
# Pre-training

    #
    def pretrain(self, position_vector_df, batch_size, stop_event):
        for batch_start_index in range(0, len(position_vector_df) - batch_size, batch_size):
            if not stop_event.is_set():
                positions_now_and_future = []

                for batch_progress_index in range(0, batch_size):
                    this_row = position_vector_df.iloc[batch_start_index + batch_progress_index]

                    legal_moves = list(chess.Board(this_row["FEN"]).legal_moves)

                    # Append FEN column value to positions_now_and_future
                    positions_now_and_future.update(this_row["FEN"])

                    # Do a forward pass using this_row minus the FEN column
                    # Store output which is N*4 nodes where N is the number of moves predicted
                    output = self.forward(positions_now_and_future.drop(columns=["FEN"]))

                    # Convert output to a list of N moves



                    # Append the FEN for board states that result from those N moves to positions_now_and_future

                    # If a move is not in the list of legal moves then use a string like "bad" or something

            else:
                break

        pass


# ##### ##### ##### ##### #####
# Supervised learning

# The model will be trained via supervised learning on chess960 positions analyzed at high depth.
# It will also be trained on endgame databases, and forced checkmate sequences.
# The latter two represent perfect quality data. I'm interested in how each of these sources impacts performance.


# ##### ##### ##### ##### #####
# Reinforcement learning


# ##### ##### ##### ##### #####
# Adversarial learning

# I would then like to train an adversarial network against it to attempt to find gaps in its knowledge.
# Then I will feed positions from games it loses against the adversarial network into Stockfish at very high
# depth and repeat the process.


# ##### ##### ##### ##### #####
# Program Body

if __name__ == "__main__":
    stop_event = threading.Event()

















