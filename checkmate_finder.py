import chess.engine
import chess
import helper_functions as hf
import threading
import keyboard
import logging
import csv


# ---------- ---------- ----------
#   Collect initial positions

def finder_stop_conditions(info, data_dict):
    # Get the important stuff from info
    depth = info.get("depth")
    score = str(info.get('score').pov(True))

    # By default, we assume the position is not a checkmate.
    data_dict["is_checkmate"] = False

    # Check to see if there is a forced checkmate
    if "#" in score:
        # Make a note that this is a checkmate so that when we get to the write_results function, we'll know
        data_dict["is_checkmate"] = True
        # Tell the engine_loop to stop
        return True

    # If we haven't explored any new nodes, then we're not making progress and should stop
    if data_dict["Nodes"] >= info.get("nodes"):
        return True
    # We're making progress, but...
    # If we haven't reached a depth of 15 yet, then lets keep analyzing
    elif depth < 15:
        return False
    # If we've reached a depth of 15, then lets check if it's worth continuing
    elif 15 <= depth < 25:
        # Retrieve the absolute value of the score
        score = abs(int(score))
        # If a side is winning by more than this much, then it's worth continuing
        if score >= 300:
            return False
        # Otherwise, lets stop
        else:
            return True
    # If we're at depth 25, and it's not a forced checkmate, then lets stop
    else:
        return True


def finder_write_results(position, info, data_dict):
    if data_dict["is_checkmate"]:
        with open(data_dict["Output Path"], "a") as output_file:
            # FEN, Score, Best Move
            output_file.write(f"{position.rstrip()},{str(info.get('score').pov(True))},{info.get("pv")[0]}\n")

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


# Start the log
logging.basicConfig(filename="checkmate_finder_log.log",
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Create a stop event object, so that we can end the analysis on demand
stop_event = threading.Event()

# Initialize and configure the engine
stockfish_config = {"Threads": 2,
                    "Hash": 40000,
                    "UCI_Elo": 3190}

# Initialize the engine using the helper function
stockfish_engine = hf.initialize_engine("stockfish", stockfish_config)

# ---------- ---------- ----------
# This implements the finding stuff

# Initialize the relevant filepaths
progress_filepath = "training-supervised-checkmates/progress_part_5_stockfish.csv"
output_filepath = "training-supervised-checkmates/results_part_5_stockfish.csv"
data_filepath = "lichess-positions/lichess_positions_part_5.txt"

# Process the files to find our current progress
[progress_dict, current_row] = hf.process_progress_file(progress_filepath, data_filepath)

# Initialize all of the wrapped up data that will go to the engine_loop
stockfish_dict = {"Name": "Stockfish",
                  "Source": data_filepath,
                  "Row": current_row,
                  "Stop": stop_event,

                  "Progress Dict": progress_dict,
                  "Progress Path": progress_filepath,
                  "Output Path": output_filepath}

# These are the finder
functions_dict = {"Stop": finder_stop_conditions,
                  "Output": finder_write_results}

# Create the thread
checkmate_thread = threading.Thread(target=hf.engine_loop, args=(stockfish_engine, functions_dict, stockfish_dict))

# Start the thread
checkmate_thread.start()

# Wait for the user to press the key
keyboard.wait("q")

# Indicate that the stop condition has been met, so the loops in continuous_analysis should finish current position
stop_event.set()

print()
print("Stop key detected!")
print("Finishing final calculations...\n")

checkmate_thread.join()
stockfish_engine.quit()



