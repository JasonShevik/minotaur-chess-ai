import chess.engine
import chess
import helper_utils as hu
import threading
import keyboard
import logging
import csv
from typing import List, Tuple, Callable, Dict, Any


# A function that saves the results of the analysis.
def write_results(position: str, info: Dict[str, Any], data_dict: Dict[str, Any]) -> None:
    # Get the important stuff from info
    best_move: str = info.get("pv")[0]
    depth: int = info.get("depth")

    # Append a new line to the output file
    with open(data_dict["Output Path"], "a") as output_file:
        output_file.write(f"{position.rstrip()},{best_move},{depth},{str(info.get('score').pov(True))}\n")

    # Update our progress. We're adding to the counter associated with this source file.
    data_dict["Progress Dict"][data_dict["Source"]] += 1

    # Create a new progress file to save our new progress
    with open(data_dict["Progress Path"], "w") as new_progress_file:
        # Write the header of the new progress file
        new_progress_file.write("File,Row\n")

        # Iterate over the keys in the progress dictionary
        for key in data_dict["Progress Dict"]:
            # Write the current dictionary entry to the new progress file
            new_progress_file.write(f"{key},{data_dict["Progress Dict"][key]}\n")


# A function that returns True if the engine_loop should stop and False if it should continue.
def stop_conditions(info: Dict[str, Any], data_dict: Dict[str, Any]) -> bool:
    # Get the important stuff from info
    depth: int = info.get("depth")
    score: str = str(info.get('score').pov(True))

    # Check to see if there is a forced checkmate
    if "#" in score:
        return True
    # If we haven't explored any new nodes, then we're not making progress and should stop
    if data_dict["Nodes"] >= info.get("nodes"):
        return True

    # Retrieve the absolute value of the score
    score: int = abs(int(score))

    # If we've reached a depth milestone specified by the breaks
    if depth == data_dict["Breaks"][data_dict["Threshold Index"]][0]:
        # If the score is greater than allowed by the breaks for this depth
        if score > data_dict["Breaks"][data_dict["Threshold Index"]][1]:
            return True
        else:
            # The evaluation is close enough that we want to analyze deeper
            data_dict["Threshold Index"] += 1

    return False


# Start the log
logging.basicConfig(filename="depth_breaks_log.log",
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")

functions_dict = {"Stop": stop_conditions,
                  "Output": write_results}

# Create a stop event object, so that we can end the analysis on demand
stop_event = threading.Event()

threads: List[threading.Thread] = []
engines: List[chess.engine.SimpleEngine] = []

#"""
# Leela

# These breakdowns represent the depth that the engine evaluates to based on the score
# If you reach the move and the evaluation is greater than the score, you stop analyzing
# This is to optimize the data labeling. Positions where one side has a significant advantage are less critical
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

leela_positions: str = "lichess-positions/lichess_positions_part_2.txt"
leela_progress: str = "training-supervised-engines/progress_part_2_leela.csv"
leela_output: str = "training-supervised-engines/results_part_2_leela.csv"

progress_dict: Dict[str, int]
current_row: int
[progress_dict, current_row] = hu.process_progress_file(leela_progress, leela_positions)
leela_engine = hu.initialize_engine("leela", leela_config)

leela_dict = {"Name": "Leela",
              "Source": leela_positions,
              "Row": current_row,
              "Stop": stop_event,

              "Breaks": leela_breaks,
              "Progress Dict": progress_dict,
              "Progress Path": leela_progress,
              "Output Path": leela_output}

with open(leela_positions, "r") as leela_source_file:
    leela_thread = threading.Thread(target=hu.engine_loop, args=(leela_engine, functions_dict, leela_dict))
    threads.append(leela_thread)
    engines.append(leela_engine)
#"""

#"""
# Stockfish

# These breakdowns represent the depth that the engine evaluates to based on the score
# If you reach the move and the evaluation is greater than the score, you stop analyzing
# This is to optimize the data labeling. Positions where one side has a significant advantage are less critical
#                   [move, score (centipawns)]
stockfish_breaks = [[20, 400],
                    [30, 200],
                    [40, 100],
                    [50, -1]]  # Maximum depth

stockfish_config = {"Threads": 6,
                    "Hash": 20000}

stockfish_positions: str = "lichess-positions/lichess_positions_part_1.txt"
stockfish_progress: str = "training-supervised-engines/progress_part_1_stockfish.csv"
stockfish_output: str = "training-supervised-engines/results_part_1_stockfish.csv"

progress_dict: Dict[str, int]
current_row: int
[progress_dict, current_row] = hu.process_progress_file(stockfish_progress, stockfish_positions)

stockfish_engine = hu.initialize_engine("stockfish", stockfish_config)

stockfish_dict = {"Name": "Stockfish",
                  "Source": stockfish_positions,
                  "Row": current_row,
                  "Stop": stop_event,

                  "Breaks": stockfish_breaks,
                  "Progress Dict": progress_dict,
                  "Progress Path": stockfish_progress,
                  "Output Path": stockfish_output}

with open(stockfish_positions, "r") as stockfish_source_file:
    stockfish_thread = threading.Thread(target=hu.engine_loop, args=(stockfish_engine, functions_dict, stockfish_dict))
    threads.append(stockfish_thread)
    engines.append(stockfish_engine)
#"""


# Start the threads
start_thread: threading.Thread
for start_thread in threads:
    start_thread.start()

# Wait for the user to press the key
keyboard.wait("q")

# Indicate that the stop condition has been met, so the engine loop should finish up
stop_event.set()

print()
print("Stop key detected!")
print("Finishing final calculations...\n")

# Join the threads
for thread in threads:
    thread.join()
# Quit the engines
for engine in engines:
    engine.quit()





