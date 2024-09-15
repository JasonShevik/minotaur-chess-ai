import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import threading
import random
import chess
from typing import List


# ##### ##### ##### ##### #####
# Functions

#
def analyze_distribution(method="uniform"):

    generated_points:  = []

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






