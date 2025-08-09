import threading
import chess.engine
import chess
import itertools
import sqlite3
import queue
import time
import math
import csv
import os
from typing import List, Tuple, Dict, Any, Callable


# This function creates and returns a chess engine object according to the configuration settings
def initialize_engine(which_engine: str, configure_options: Dict[str, Any]) -> chess.engine.SimpleEngine:
    if which_engine == "leela":
        engine_dir: str = "lc0-v0.31.2-windows-gpu-nvidia-cuda/lc0.exe"
    elif which_engine == "stockfish":
        engine_dir: str = "stockfish/stockfish-windows-x86-64-avx2.exe"
    else:
        print("Invalid Engine Choice")
        engine_dir: str = ""

    this_engine = chess.engine.SimpleEngine.popen_uci(engine_dir)
    this_engine.configure(configure_options)
    return this_engine


# Function to write analysis results to the database
def db_writer_loop(db_name: str, write_queue: queue.Queue, thread_list: List[threading.Thread]) -> None:
    """
    A function to indefinitely write updates to the database as they are added to a queue for thread safety.

    :param db_name: The filename of the database to write to
    :param write_queue: The queue object that the analysis thread(s) are adding to. This function updates the database
        as entries get added to the queue
    :param thread_list: A list of the analysis threads. This writer only keeps writing while those threads are alive
    """
    # Connect to the database
    with sqlite3.connect(f"{db_name}.db") as conn:
        cursor: sqlite3.Cursor = conn.cursor()

        count: int = 0
        # Will only terminate when both stop_event is set, and when write_queue is empty
        # If something is added to queue after stop event due to race condition, this will still write everything
        while any(thread.is_alive() for thread in thread_list) or not write_queue.empty():
            try:
                # Get the next item in the queue
                # Timeout is necessary because if stop_event happens while waiting for get, the thread will hang forever
                data = write_queue.get(timeout=5)

                # Get the info object first, because it is used to derive the variables to write
                info: Dict[str, Any] = data[0]

                # Assign each of the variables that will be updated, using data from the queue
                engine_name: str = data[2]
                depth: int = int(info.get("depth"))
                score: str = str(info.get("score").pov(True))
                is_forced_checkmate: int = 1 if "#" in score else 0
                best_move: str = str(info.get("pv")[0])
                fen: str = data[1]

                cursor.execute('''
                                UPDATE "960_position_data" 
                                SET is_analyzed = 1, 
                                    engine_name = ?, 
                                    depth = ?, 
                                    score = ?, 
                                    is_forced_checkmate = ?, 
                                    best_move = ?
                                WHERE fen = ?
                                ''',
                               (engine_name, depth, score, is_forced_checkmate, best_move, fen))
                conn.commit()
                count += 1
                print(f"Wrote: {count}")

            # If it times out, just continue
            # Continuing allows us to check for stop_event.is_set() on the next loop to see if we should stop or not
            except queue.Empty:
                pass

    print("Writer done!")


# A configurable function to continuously analyze positions
def engine_loop(engine: chess.engine.SimpleEngine, position_list: List[str], data_dict: Dict[str, Any],
                stop_event: threading.Event, stop_conditions: Callable, write_queue: queue.Queue) -> None:
    """
    A customizable function that continuously analyzes chess positions until a user-specified stop condition is met
    depending on the specific application.

    :param engine: The chess engine object
    :param position_list: The list of chess positions that this function will loop through to analyze
    :param data_dict: The breakdowns of what depth to analyze to based on the score of the position
    :param stop_event: A threading event that, when set to true, indicates that the loop should wrap up and finish
    :param stop_conditions: A function that checks whether the engine should stop its current analysis or continue
        - info: The dictionary with analysis statistics
        - data_dict: A dictionary that holds any data needed for the specific stop_conditions implementation
        - returns: A Tuple of two booleans that say whether it should_stop and should_write the result
    :param write_queue: A queue object which this thread adds to for a writer thread to utilize for writing results
    """
    count: int = 0

    # Loop through each position in the source file starting at the current row
    position: str
    for position in position_list:
        if not stop_event.is_set():
            # Print the progress
            count += 1
            print(f"{data_dict["Name"]}: {count}")

            # No nodes have been visited yet
            data_dict["Nodes"] = 0

            # Parse the position and make an object
            board: chess.Board = chess.Board(position)

            # Begin the analysis of this board position
            analysis: chess.engine.SimpleAnalysisResult

            # Start the analysis
            with engine.analysis(board) as analysis:
                # Analyze continuously, depth by depth, until we meet a break condition
                info: Dict[str, Any]
                for info in analysis:
                    # Get the current depth
                    depth: int = info.get("depth")
                    if depth is None:
                        continue

                    # Get the current score
                    score: chess.engine.Score = info.get("score")
                    # Score can be None when Stockfish looks at sidelines - Skip those iterations
                    if score is None:
                        continue

                    # If the user-specified stop conditions are met
                    should_stop, should_write = stop_conditions(info, data_dict)
                    if should_stop:
                        if should_write:
                            # Add to queue
                            write_queue.put([info, position, data_dict["Name"]])
                        break

                    # Update the number of explored nodes
                    data_dict["Nodes"] = info.get("nodes")

        # If the stop_event was set
        else:
            engine.quit()
            print(f"{data_dict["Name"]} done!")
            break


# Takes in a FEN string and returns a list of 64 numbers
# https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
def fen_to_vector(fen: str) -> List[float]:
    piece_values: Dict[str, int] = {"p": 1,
                                    "n": 2,
                                    "b": 3,
                                    "r": 4,
                                    "q": 5,
                                    "k": 6}

    # Split the fen by spaces
    fen_parts: List[str] = fen.split(" ")

    # The first part is the board portion. Split it by '/' to get each row
    row_strings: List[str] = fen_parts[0].split("/")

    # Initialize some variables for constructing the vector
    vector_version: List[float] = [0 for _ in range(64)]
    index_buffer_num_rows: int = 0

    # Vectors go a1, b1, ... a2, b2, ..., so if we're white then we need to go through the strings in reverse order
    if fen_parts[1] == "w":
        row_strings = list(reversed(row_strings))

    # If playing as black, then we want to go through the strings in the current order
    # BUT we need to reverse all of the individual strings like we're rotating the board
    elif fen_parts[1] == "b":
        for index, _ in enumerate(row_strings):
            row_strings[index] = row_strings[index][::-1]

    # This value can only be 'w' or 'b'
    else:
        print(f"Invalid FEN: {fen_parts[0]}")
        return [0]

    # Iterate over the rows backwards (start from row 1 and go up)
    current_row: str
    for current_row in row_strings:
        index_buffer_this_row: int = 0

        # Loop through each character in this row of the chess board
        index: int
        character: str
        for index, character in enumerate(current_row):
            # If the character is numerical...
            if character.isdigit():
                # Record the number of sequential empty squares
                index_buffer_this_row += int(character) - 1

                # For each of the empty squares
                index_to_zero: int
                for index_to_zero in range(int(character)):
                    # Set that index of the vector_version to zero
                    # (index_buffer_num_rows * 8) because we need to offset by the number of rows we've already done
                    # index because we need to offset by the number of characters in this row we've already done
                    # index_to_zero because we need to count up how many zeros we're adding based on the character
                    # (index_buffer_this_row - int(character) + 1) because ...
                    # if not first digit in row, need offset by more
                    vector_version[index + index_to_zero + (index_buffer_num_rows * 8) + (index_buffer_this_row - int(character) + 1)] = 0
            # If the character is alphabetical...
            else:
                # Set the value in the vector to the piece value
                true_index: int = index + (index_buffer_num_rows * 8) + index_buffer_this_row
                vector_version[true_index] = piece_values[character.lower()]

                # If the piece is black
                if character.islower():
                    # And the AI is playing as white
                    if fen_parts[1] == "w":
                        # Then multiply it by -1 because it's on the opponent's team
                        vector_version[true_index] *= -1
                # If the piece is white
                else:
                    # And the AI is playing as black
                    if fen_parts[1] == "b":
                        # Then multiply it by -1 because it's on the opponent's team
                        vector_version[true_index] *= -1

        index_buffer_num_rows += 1

    # ----- Castling -----

    # King value of +/-6 is modified by +/-0.1 and 0.2
    # This means the square can have 4 possible values:
    # 6: May not castle
    # 6.1: May castle king-side
    # 6.2: May castle queen-side
    # 6.3: May castle either side

    # Establish what side we're on so that we know if the king is a positive or negative number
    if fen_parts[1] == "w":
        white_mod: int = 1
        black_mod: int = -1
    else:
        white_mod: int = -1
        black_mod: int = 1

    # Since we may modify kings multiple times in a row...
    # Doing this work on separate list using indices of the old list so .index() works properly
    working_castle_vector = vector_version[:]

    # White may castle king-side
    if "K" in fen_parts[2]:
        working_castle_vector[vector_version.index(white_mod * 6)] += (white_mod * 0.1)
    # White may castle queen-side
    if "Q" in fen_parts[2]:
        working_castle_vector[vector_version.index(white_mod * 6)] += (white_mod * 0.2)
    # Black may castle king-side
    if "k" in fen_parts[2]:
        working_castle_vector[vector_version.index(black_mod * 6)] += (black_mod * 0.1)
    # Black may castle queen-side
    if "q" in fen_parts[2]:
        working_castle_vector[vector_version.index(black_mod * 6)] += (black_mod * 0.2)

    # Save changes to the original list
    vector_version = working_castle_vector[:]

    # ----- En Passant -----

    # If there is no En Passant, finish
    if fen_parts[3][0] == "-":
        return vector_version

    character_values = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
    en_passant_square = ((int(fen_parts[3][1]) - 1) * 8) + character_values[fen_parts[3][0]]
    if black_mod == 1:
        en_passant_square = 63 - en_passant_square

    # noinspection PyTypeChecker
    vector_version[en_passant_square] = -0.5

    return vector_version




