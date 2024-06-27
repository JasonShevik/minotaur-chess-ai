import chess.engine
import chess
from heapdict import heapdict
import helper_functions as hf
import threading
import itertools
import keyboard
import logging
import csv


# ---------- ---------- ----------
#   Expand positions

def expander_stop_conditions():
    pass


def expander_write_results():
    pass


def expand_position(initial_position):
    #to_explore_up = heapdict()
    #to_explore_down = heapdict()
    # Add initial_position to to_explore_down

    #positions_finished = set()
    # positions_finished.add()
    # "" in positions_finished


    # There are two functions that will make this function work: expand_up and expand_down
    # expand_down finds all the sequences going toward the checkmate.
    # It may take in a #+10 position as input and output a dictionary of FENs and best moves going down to checkmate.
    # Along the way, it also adds every position to the to_explore_up heapdict.
    # Once expand_down is done, you go through every position in to_explore_up and expand_up each one.
    # expand_up looks at all positions that could precede the current position, and checks if any are forced checkmate.
    # If a position is a forced checkmate, it is added to to_explore_down
    # After checking every possible preceding position, that position is popped from to_explore_up.
    # Then you run expand_down on each of positions, if any, in to_explore_down
    # Repeat until to_explore_up and to_explore_down are both empty.

    # Try to use some elegant recursion, but keep it efficient.

    # For maximum speed/efficiency, don't analyze from the losing side.
    # Just find the set of legal responses from losing, then analyze the winning moves (forced checkmate moves).
    # I can make a separate function to deeply analyze losing positions at a later date to improve defensiveness.
    pass

#
def expand_up(initial_position):
    pass

#
def expand_down(initial_position):
    pass