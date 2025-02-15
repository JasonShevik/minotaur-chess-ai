import chess.engine
import chess
import helper_utils as hu
import db_tools as dbt
import threading
import keyboard
import logging
import queue
import csv
from typing import List, Tuple, Callable, Dict, Any


# ---------- ---------- ----------
#   Stop Conditions Functions:
#       - Return [bool, bool] which describe whether it [should_stop, should_write]

# Stop conditions function for depth breaks analysis mode
def stop_conditions_breaks(info: Dict[str, Any], data_dict: Dict[str, Any]) -> Tuple[bool, bool]:
    # Get the important stuff from info
    depth: int = info.get("depth")
    score: str = str(info.get('score').pov(True))

    # Check to see if there is a forced checkmate
    if "#" in score:
        data_dict["Threshold Index"] = 0
        return True, True
    # If we haven't explored any new nodes, then we're not making progress and should stop
    if data_dict["Nodes"] >= info.get("nodes"):
        data_dict["Threshold Index"] = 0
        return True, True

    # Retrieve the absolute value of the score
    score: int = abs(int(score))

    # If we've reached a depth milestone specified by the breaks
    if depth == data_dict["Breaks"][data_dict["Threshold Index"]][0]:
        # If the score is greater than allowed by the breaks for this depth
        if score > data_dict["Breaks"][data_dict["Threshold Index"]][1]:
            data_dict["Threshold Index"] = 0
            return True, True
        else:
            # The evaluation is close enough that we want to analyze deeper
            data_dict["Threshold Index"] += 1

    return False, False


# Stop conditions function for checkmate finder mode
def stop_conditions_checkmates(info: Dict[str, Any], data_dict: Dict[str, Any]) -> Tuple[bool, bool]:
    # Get the important stuff from info
    depth: int = info.get("depth")
    score: str = str(info.get('score').pov(True))

    # Check to see if there is a forced checkmate
    if "#" in score:
        # Stop and write
        return True, True

    # Set the thresholds for how deep to look for checkmates depending on the engine
    if "leela" in data_dict["Name"].lower():
        first_thresh = 6
        second_thresh = 8
    elif "stockfish" in data_dict["Name"].lower():
        first_thresh = 15
        second_thresh = 25
    else:
        first_thresh = 15
        second_thresh = 25

    # If we haven't explored any new nodes, then we're not making progress and should stop
    if data_dict["Nodes"] >= info.get("nodes"):
        # Stop and write
        # Even though it's not a checkmate, this should be maximally analyzed and therefore worth writing
        return True, True
    # We're making progress, but...
    # If we haven't reached a depth of 15 yet, then lets keep analyzing
    elif depth < first_thresh:
        # Don't stop, don't write
        return False, False
    # If we've reached a depth of 15, then lets check if it's worth continuing
    elif first_thresh <= depth < second_thresh:
        # Retrieve the absolute value of the score
        score: int = abs(int(score))

        # If a side is winning by more than this much, then it's worth continuing
        if score >= 300:
            # Don't stop, don't write
            return False, False
        # Otherwise, lets stop
        else:
            # Stop but don't write
            return True, False
    # If we're at depth 25, and it's not a forced checkmate, then lets stop
    else:
        # Stop but don't write
        return True, False


# ----- -----
# Stockfish

# These breakdowns represent the depth that the engine evaluates to based on the score
# If you reach the move and the evaluation is greater than the score, you stop analyzing
# This is to optimize the data labeling. Positions where one side has a significant advantage are less critical
#                   [move, score (centipawns)]
stockfish_breaks = [[20, 400],
                    [30, 200],
                    [40, 100],
                    [50, -1]]  # Maximum depth

stockfish_config = {"Threads": 4,
                    "Hash": 20000}

stockfish_dict = {"Name": "Stockfish 17",
                  "Breaks": stockfish_breaks,
                  "Threshold Index": 0,
                  "Nodes": 0}

# ----- -----
# Leela
#               [move, score (centipawns)]
leela_breaks = [[10, 400],
                [13, 200],
                [16, 100],
                [19, -1]]  # Maximum depth

leela_config = {"Threads": 2,
                "NNCacheSize": 1000000,
                "MinibatchSize": 1024,
                # "WeightsFile": "lc0-v0.30.0-windows-gpu-nvidia-cuda/768x15x24h-t82-2-swa-5230000.pb",
                "RamLimitMb": 20000}

leela_dict = {"Name": "Leela 0.31.2",
              "Breaks": leela_breaks,
              "Threshold Index": 0,
              "Nodes": 0}

# ----- ----- -----
# Program Body

if __name__ == "__main__":
    # Booleans for which analyses to run:
    # Depth breaks
    stockfish_breaks_yes = False
    leela_breaks_yes = True
    # Checkmate finder
    stockfish_checkmates_yes = True
    leela_checkmates_yes = False

    # Start the log
    logging.basicConfig(filename="depth_breaks_log.log",
                        level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    # Create a stop event object, so that we can end the analysis on demand
    stop_event = threading.Event()

    #
    stockfish_engine_breaks = hu.initialize_engine("stockfish", stockfish_config) if stockfish_breaks_yes else None
    leela_engine_breaks = hu.initialize_engine("leela", leela_config) if leela_breaks_yes else None
    stockfish_engine_checkmates = hu.initialize_engine("stockfish", stockfish_config) if stockfish_checkmates_yes else None
    leela_engine_checkmates = hu.initialize_engine("leela", leela_config) if leela_checkmates_yes else None

    #
    position_lists = dbt.get_slices(db_name="minotaur_data", num_slices=4)

    #
    write_queue = queue.Queue()

    # Create the threads
    stockfish_breaks_thread = threading.Thread(target=hu.engine_loop, args=(stockfish_engine_breaks, position_lists[0], stockfish_dict, stop_event, stop_conditions_breaks, write_queue)) if stockfish_breaks_yes else None
    leela_breaks_thread = threading.Thread(target=hu.engine_loop, args=(leela_engine_breaks, position_lists[1], leela_dict, stop_event, stop_conditions_breaks, write_queue)) if leela_breaks_yes else None
    stockfish_checkmates_thread = threading.Thread(target=hu.engine_loop, args=(stockfish_engine_checkmates, position_lists[2], stockfish_dict, stop_event, stop_conditions_checkmates, write_queue)) if stockfish_checkmates_yes else None
    leela_checkmates_thread = threading.Thread(target=hu.engine_loop, args=(leela_engine_checkmates, position_lists[3], leela_dict, stop_event, stop_conditions_checkmates, write_queue)) if leela_checkmates_yes else None

    #
    thread_list = []
    thread_list.append(stockfish_breaks_thread) if stockfish_breaks_yes else None
    thread_list.append(leela_breaks_thread) if leela_breaks_yes else None
    thread_list.append(stockfish_checkmates_thread) if stockfish_checkmates_yes else None
    thread_list.append(leela_checkmates_thread) if leela_checkmates_yes else None

    writer_thread = threading.Thread(target=hu.db_writer_loop, args=("minotaur_data", write_queue, thread_list))

    # Start the threads
    for thread in thread_list:
        thread.start()
    writer_thread.start()

    # Wait for the user to press the key
    keyboard.wait("q")

    # Indicate that the stop condition has been met, so the engine loop should finish up
    stop_event.set()

    print()
    print("Stop key detected!")
    print("Finishing final calculations...\n")

    # Join the threads
    for thread in thread_list:
        thread.join()
    writer_thread.join()

    print(dbt.check_db("minotaur_data"))

    # Quit the engines
    stockfish_engine_breaks.quit() if stockfish_breaks_yes else None
    leela_engine_breaks.quit() if leela_breaks_yes else None
    stockfish_engine_checkmates.quit() if stockfish_checkmates_yes else None
    leela_engine_checkmates.quit() if leela_checkmates_yes else None










