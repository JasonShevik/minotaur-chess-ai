import chess.engine
import chess
import helper_utils as hu
import db_tools as dbt
import threading
import keyboard
import logging
import queue
import csv
from typing import Dict, Tuple, Any


# ---------- ---------- ----------
#   Collect initial positions

def stop_conditions(info: Dict[str, Any], data_dict: Dict[str, Any]) -> Tuple[bool, bool]:
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

# Initialize and configure the engine
stockfish_config = {"Threads": 4,
                    "Hash": 20000}

stockfish_dict = {"Nodes": 0,
                  "Name": "Stockfish 17"}

# ----- -----
# Leela

# Initialize and configure the engine
leela_config = {"Threads": 2,
                "NNCacheSize": 1000000,
                "MinibatchSize": 1024,
                #"WeightsFile": "lc0-v0.30.0-windows-gpu-nvidia-cuda/768x15x24h-t82-2-swa-5230000.pb",
                "RamLimitMb": 20000}

leela_dict = {"Nodes": 0,
              "Name": "Leela 0.31.1"}

# ----- ----- -----

stockfish_yes = True
leela_yes = True

# Start the log
logging.basicConfig(filename="checkmate_finder_log.log",
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Create a stop event object, so that we can end the analysis on demand
stop_event = threading.Event()

# Initialize the engine using the helper function
stockfish_engine = hu.initialize_engine("stockfish", stockfish_config) if stockfish_yes else None
leela_engine = hu.initialize_engine("leela", leela_config) if leela_yes else None

position_lists = dbt.get_slices(db_name="minotaur_data", num_slices=2)

write_queue = queue.Queue()

# Create the threads
stockfish_thread = threading.Thread(target=hu.engine_loop, args=(stockfish_engine, position_lists[0], stockfish_dict, stop_event, stop_conditions, write_queue)) if stockfish_yes else None
leela_thread = threading.Thread(target=hu.engine_loop, args=(leela_engine, position_lists[1], leela_dict, stop_event, stop_conditions, write_queue)) if leela_yes else None

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

# Indicate that the stop condition has been met, so the loops in continuous_analysis should finish current position
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



