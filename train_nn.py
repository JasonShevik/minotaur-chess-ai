import torch
import torch.nn as nn
import torch.optim as optim


# ##### ##### ##### ##### #####
# Creation

# Since this Chess AI only looks at the next move, and does not do search, then it will be a classifier AI.
# Have two output nodes, one for the starting square, and one for the destination square.
# Nodes can choose any one of 64 squares, plus two additional squares represent castling. (65,65), (66,66) mean castle.

# A board position should look like this:
# The pieces on the board
# [[x,x,x,x,x,x,x,x],
#  [x,x,x,x,x,x,x,x],
#  [x,x,x,x,x,x,x,x],
#  [x,x,x,x,x,x,x,x],
#  [x,x,x,x,x,x,x,x],
#  [x,x,x,x,x,x,x,x],
#  [x,x,x,x,x,x,x,x],
#  [x,x,x,x,x,x,x,x],
# Right to castle
#  [x,x],
# Right to en passant
#  [x,x,x,x,x,x,x,x]]

# ##### ##### ##### ##### #####
# Architecture

def create_minotaur(name):
    hidden_depth = 100
    hidden_width = 74

    layers = []

    for i in range(hidden_depth):
        layers.append(nn.Linear(hidden_width, i))
        layers.append(nn.ReLU)

    net = nn.Sequential(*layers)

    torch.save(net.state_dict(),f"{name}.pt")

# ##### ##### ##### ##### #####
# Pre-training

# The model will be trained on a large number of chess positions to generate legal moves.
# The model will be rewarded for legal moves, and punished for illegal ones
# ~~The reward will be boosted if it predicts legal moves that it hasn't predicted recently~~
# This is to increase the diversity of legal moves that it generates and prevent bias toward certain moves
# The more diverse and unbiased the moves it learns to suggest, the better situated we'll be for the next step

# Change of plans:
# Boosting the reward based on recency of the move it chooses will create bias.
# It would bias the model to choose the most obscure move in any position, like an overcorrection.
# Instead, the model should choose a move for N positions, and the reward will be boosted based on
# the ratio between how often it had the opportunity to play each move versus how often it chose that.
# OR instead:
# I could do a monte carlo simulation for the set of positions where I randomly choose moves, then
# compare the found distribution to the one the AI does, and boost reward based on closeness to the random dist.



# ##### ##### ##### ##### #####
# Supervised learning

# The model will be trained via supervised learning on very high quality single-move chess puzzles created
# by giving chess positions to engines to be evaluated at very high depth, as well as positions
# from endgame databases where correct moves are proven.
# This will bring the model to a base level. I'm interested to see what this level will be.





# ##### ##### ##### ##### #####
# Reinforcement learning

# The bot will then be fine-tuned using reinforcement learning through self play.
# Rather than having two versions play each other and picking the winner, this fine-tuning will be to have
# a small tournament with multiple versions of the AI and choosing the winner. This method should
# reduce gaps in knowledge, and ensure that the model converges to a well-rounded style that is resistant
# to exploitation.

# ##### Testing functions

def analyze_distribution(method):

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


# ##### Random vector functions

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

# ##### Training functions

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
# Adversarial learning

# I would then like to train an adversarial network against it to attempt to find gaps in its knowledge.
# Then I will feed positions from games it loses against the adversarial network into Stockfish at very high
# depth and repeat the process.











