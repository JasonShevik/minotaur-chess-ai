import chess.engine
import chess
import helper_utils as hu
import threading
import itertools
import keyboard
import logging
import heapq
import csv


# ---------- ---------- ----------
#   Expand positions

# If starting from a fresh position, the heaps will have only one element, otherwise they're from the checkpoint
def expand_position(down_heap, up_heap):

    # Expand down until down_heap is empty
    while down_heap:



    # Don't analyze from the losing side
    # Add them to a database of losing positions; they could be useful later



    pass

#
def expand_up(up_heap):
    pass

#
def expand_down(down_heap):
    pass


# ##### ##### ##### ##### #####
#       Program body

# This file contains the current progress in the positions file, and whether there is a checkpoint
progress_filepath = ""
# This file contains the list of positions that the checkmate_finder found
positions_filepath = ""
# This file contains the progress in expanding the current starting position (if the last one wasn't finished)
checkpoint_filepath = ""
# This file contains
losing_filepath = ""


# Start by checking the checkpoint_filepath
with open(checkpoint_filepath, "r") as checkpoint_file:













