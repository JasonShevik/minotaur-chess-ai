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


# A function that returns True if the engine_loop should stop and False if it should continue.
def stop_conditions(info: Dict[str, Any], data_dict: Dict[str, Any]) -> bool:
    # Get the important stuff from info
    depth: int = info.get("depth")
    score: str = str(info.get('score').pov(True))

    # Check to see if there is a forced checkmate
    if "#" in score:
        data_dict["Threshold Index"] = 0
        data_dict["Nodes"] = 0
        return True
    # If we haven't explored any new nodes, then we're not making progress and should stop
    if data_dict["Nodes"] >= info.get("nodes"):
        data_dict["Threshold Index"] = 0
        data_dict["Nodes"] = 0
        return True

    # Retrieve the absolute value of the score
    score: int = abs(int(score))

    # If we've reached a depth milestone specified by the breaks
    if depth == data_dict["Breaks"][data_dict["Threshold Index"]][0]:
        # If the score is greater than allowed by the breaks for this depth
        if score > data_dict["Breaks"][data_dict["Threshold Index"]][1]:
            data_dict["Threshold Index"] = 0
            data_dict["Nodes"] = 0
            return True
        else:
            # The evaluation is close enough that we want to analyze deeper
            data_dict["Threshold Index"] += 1

    data_dict["Nodes"] = info.get("nodes")
    return False


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

stockfish_dict = {"Breaks": stockfish_breaks,
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
                #"WeightsFile": "lc0-v0.30.0-windows-gpu-nvidia-cuda/768x15x24h-t82-2-swa-5230000.pb",
                "RamLimitMb": 20000}

leela_dict = {"Breaks": leela_breaks,
              "Threshold Index": 0,
              "Nodes": 0}
# ----- ----- -----

stockfish_yes = True
leela_yes = True

# Start the log
logging.basicConfig(filename="depth_breaks_log.log",
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Create a stop event object, so that we can end the analysis on demand
stop_event = threading.Event()

stockfish_engine = hu.initialize_engine("stockfish", stockfish_config) if stockfish_yes else None
leela_engine = hu.initialize_engine("leela", leela_config) if leela_yes else None

position_lists = dbt.get_slices(db_name="minotaur_data", num_slices=2)

write_queue = queue.Queue()

# Create the threads
stockfish_thread = threading.Thread(target=hu.engine_loop, args=(stockfish_engine, position_lists[0], "Stockfish 17", stockfish_dict, stop_event, stop_conditions, write_queue)) if stockfish_yes else None
leela_thread = threading.Thread(target=hu.engine_loop, args=(leela_engine, position_lists[1], "Leela 0.31.1", leela_dict, stop_event, stop_conditions, write_queue)) if leela_yes else None

thread_list = []
thread_list.append(stockfish_thread) if stockfish_yes else None
thread_list.append(leela_thread) if leela_yes else None
writer_thread = threading.Thread(target=hu.db_writer_loop, args=("minotaur_data", write_queue, thread_list))

# Start the threads
stockfish_thread.start() if stockfish_yes else None
leela_thread.start() if leela_yes else None
writer_thread.start()


# Wait for the user to press the key
keyboard.wait("q")

print(write_queue.qsize())

# Indicate that the stop condition has been met, so the engine loop should finish up
stop_event.set()

print()
print("Stop key detected!")
print("Finishing final calculations...\n")

# Join the threads
stockfish_thread.join() if stockfish_yes else None
leela_thread.join() if leela_yes else None
writer_thread.join()

print(dbt.check_db("minotaur_data"))

# Quit the engines
stockfish_engine.quit() if stockfish_yes else None
leela_engine.quit() if leela_yes else None










