import threading
import chess.engine
import chess
import itertools
import sqlite3
import queue
import time
import csv
import os
from typing import List, Tuple, Dict, Any, Callable


# This function creates and returns a chess engine object according to the configuration settings
def initialize_engine(which_engine: str, configure_options: Dict[str, Any]) -> chess.engine.SimpleEngine:
    if which_engine == "leela":
        engine_dir: str = "lc0-v0.31.1-windows-gpu-nvidia-cuda/lc0.exe"
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
def engine_loop(engine: chess.engine.SimpleEngine, position_list: List[str], name: str, data_dict: Dict[str, Any],
                stop_event: threading.Event, stop_conditions: Callable, write_queue: queue.Queue) -> None:
    """
    A customizable function that continuously analyzes chess positions until a user-specified stop condition is met
    depending on the specific application.

    :param engine: The chess engine object
    :param position_list: The list of chess positions that this function will loop through to analyze
    :param name: The name and version of the chess engine analyzing positions
    :param data_dict: The breakdowns of what depth to analyze to based on the score of the position
    :param stop_event: A threading event that, when set to true, indicates that the loop should wrap up and finish
    :param stop_conditions: A function that checks whether the engine should stop its current analysis or continue
        - info: The dictionary with analysis statistics
        - data_dict: A dictionary that holds any data needed for the specific stop_conditions implementation
    :param write_queue: A queue object which this thread adds to for a writer thread to utilize for writing results
    """
    count: int = 0

    # Loop through each position in the source file starting at the current row
    position: str
    for position in position_list:
        if not stop_event.is_set():
            # Print the progress
            count += 1
            print(f"{name}: {count}")

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
                    if stop_conditions(info, data_dict):
                        # Add to queue
                        write_queue.put([info, position, name])
                        break

        # If the stop_event was set
        else:
            engine.quit()
            print(f"{name} done!")
            break







