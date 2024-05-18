import torch
import torch.nn as nn
import torch.optim as optim


# ##### ##### ##### ##### #####
# Architecture

class Minotaur(nn.Module):
    def __init__(self, mode="normal", pretrain_model=None):
        super(Minotaur, self).__init__()
        self.input_layer = nn.Linear(66, 128)  # 64 squares + castling + en passant = 66 inputs
        self.hidden_layers = nn.ModuleList([nn.Linear(128, 128) for _ in range(99)])

        if mode == "pretrain":
            self.output_layer = nn.Linear(128, 40)
        elif mode == "normal":
            if pretrain_model is not None:
                self.load_pretrained_weights(pretrain_model)

            self.output_layer = nn.Linear(128, 4)
        else:
            # Handle an exception?
            self.output_layer = nn.Linear(128, 4)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:    # Is it slow to do this with a loop? What about map or something?
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        x = (torch.tanh(x) + 1) * 3.5 + 1
        return x  # Should this have a rounding function?

    def load_pretrained_weights(self, pretrain_model):
        pretrain_dict = pretrain_model.state_dict()
        model_dict = self.state_dict()

        pretrain_dict = {k: v for k, v in pretrain_dict.items() if "output_layer" not in k}  # Is this right?

        model_dict.update(pretrain_dict)
        self.load_state_dict(model_dict)

# ##### ##### ##### ##### #####
# Pre-training

# The model will be trained on a large number of chess positions to generate legal moves.
# The model will be rewarded for legal moves, and punished for illegal ones
# I could do a monte carlo simulation for the set of positions where I randomly choose moves, then
# compare the found distribution to the one the AI does, and boost reward based on closeness to the random dist.

# ##### ##### ##### ##### #####
# Supervised learning

# The model will be trained via supervised learning on chess960 positions analyzed at high depth.
# It will also be trained on endgame databases, and forced checkmate sequences.
# The latter two represent perfect quality data. I'm interested in how each of these sources impacts performance.

# ##### ##### ##### ##### #####
# Reinforcement learning


# ##### Testing functions

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


# ##### Random vectors in hyper-spheres

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
# Adversarial learning

# I would then like to train an adversarial network against it to attempt to find gaps in its knowledge.
# Then I will feed positions from games it loses against the adversarial network into Stockfish at very high
# depth and repeat the process.




