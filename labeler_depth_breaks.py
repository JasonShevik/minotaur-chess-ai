import chess.engine
import chess
import helper_utils as hu
import threading
import keyboard
import logging
import csv


# A function that saves the results of the analysis.
def write_results(position, info, data_dict):
    # Get the important stuff from info
    best_move = info.get("pv")[0]
    depth = info.get("depth")

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
def stop_conditions(info, data_dict):
    # Get the important stuff from info
    depth = info.get("depth")
    score = str(info.get('score').pov(True))

    # Check to see if there is a forced checkmate
    if "#" in score:
        return True
    # If we haven't explored any new nodes, then we're not making progress and should stop
    if data_dict["Nodes"] >= info.get("nodes"):
        return True

    # Retrieve the absolute value of the score
    score = abs(int(score))

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

threads = []
engines = []

#"""
# Leela

# These breakdowns represent the depth that the engine evaluates to based on the score
# If you reach the move and the evaluation is greater than the score, you stop analyzing
# This is to optimize the data labeling. Positions where one side has a significant advantage are less critical
#                    [move, score (centipawns)]
leela_breaks = [[10, 400],
                [13, 200],
                [16, 100],
                [19, -1]]  # Maximum depth

leela_config = {"Threads": 2,
                "NNCacheSize": 1000000,
                "MinibatchSize": 1024,
                #"WeightsFile": "lc0-v0.30.0-windows-gpu-nvidia-cuda/768x15x24h-t82-2-swa-5230000.pb",
                "RamLimitMb": 20000}

leela_positions = "lichess-positions/lichess_positions_part_2.txt"
leela_progress = "training-supervised-engines/progress_part_2_leela.csv"
leela_output = "training-supervised-engines/results_part_2_leela.csv"

leela_engine = hu.initialize_engine("leela", leela_config)
[progress_dict, current_row] = hu.process_progress_file(leela_progress, leela_positions)

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
#                    [move, score (centipawns)]
stockfish_breaks = [[20, 400],
                    [30, 200],
                    [40, 100],
                    [50, -1]]  # Maximum depth

stockfish_config = {"Threads": 2,
                    "Hash": 20000,
                    "UCI_Elo": 3190}

stockfish_positions = "lichess-positions/lichess_positions_part_1.txt"
stockfish_progress = "training-supervised-engines/progress_part_1_stockfish.csv"
stockfish_output = "training-supervised-engines/results_part_1_stockfish.csv"

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


# ----- ----- LEGACY ----- -----

# This code was for fixing labeled positions after I changed methodologies.
# There used to be a 'hopeless' mechanic such that if the score was too high, the position was deemed 'hopeless'
# And the engine wouldn't analyze further. This was to save time and compute resources.
# I ultimately decided that I would prefer that all positions be analyzed to at least depth 20 or forced checkmate.
"""
def analyze_hopeless(engine, analysis_path, depth_score_breaks):
    # This empty set will hold all of the analyzed positions to print later
    file_rows = set()

    # Open the results file that we're re-analyzing
    with open(analysis_path, "r") as analysis_file:
        # Skip the header row
        next(analysis_file)

        foo = 1
        # Go through each line in the file
        for line in analysis_file:
            print(foo)
            foo += 1
            # Split the row into the different columns so that we can look at them individually
            row_elements = line.split(",")

            # If this row was analyzed below the minimum depth and wasn't a forced checkmate
            if int(row_elements[2]) < depth_score_breaks[0][0] and "#" not in row_elements[3]:
                # Set the board object for this position
                board = chess.Board(row_elements[0])

                # Set the index that we use with depth_score_breaks to determine depth based on a score threshold
                threshold_index = 0

                # Begin the analysis of this position
                with engine.analysis(board) as analysis:
                    # Number of nodes explored is 0 at the start of the analysis
                    num_nodes_last = 0
                    # Analyze continuously, depth by depth, until we meet a break condition
                    for info in analysis:
                        # Get the current depth
                        depth = info.get("depth")
                        if depth is None:
                            continue

                        # Get the current score
                        score = info.get("score")
                        # Score can be None when looking at sidelines - Skip those iterations
                        if score is None:
                            continue

                        score = str(score.pov(True))
                        # If we discovered a checkmate before we realized it's hopeless
                        if "#" in score:
                            # Get the first move of the principle variation
                            best_move = info.get("pv")[0]
                            # Add to the file rows set
                            file_rows.add(f"{row_elements[0]},{best_move},{depth},{str(info.get('score').pov(True))}")
                            break

                        # Retrieve the absolute value of the score
                        score = abs(int(score))

                        num_nodes = info.get("nodes")
                        if num_nodes_last >= num_nodes:
                            # Get the first move of the principle variation
                            best_move = info.get("pv")[0]
                            # Add to the file rows set
                            file_rows.add(f"{row_elements[0]},{best_move},{depth},{str(info.get('score').pov(True))}")
                            break
                        num_nodes_last = num_nodes

                        # If we've reached a depth milestone specified by depth_score_breaks
                        if depth == depth_score_breaks[threshold_index][0]:

                            # If the score is greater than allowed by depth_score_breaks for this depth
                            if score > depth_score_breaks[threshold_index][1]:
                                # Get the first move of the principle variation
                                best_move = info.get("pv")[0]
                                # Add to the file rows set
                                file_rows.add(f"{row_elements[0]},{best_move},{depth},{str(info.get('score').pov(True))}")
                                break
                            else:
                                # The evaluation is close enough that we want to analyze deeper
                                threshold_index += 1

            # If this row was analyzed properly, just add it to the list
            else:
                file_rows.add(line)

        # Print file_rows to a new file
        with open("results no hopeless.csv", "w") as output_file:
            output_file.write("Position,Move,Depth,Score\n")
            for row in file_rows:
                output_file.write(row.rstrip() + "\n")

# If you're redoing the analysis from hopeless mechanic
doing_hopeless = False
if doing_hopeless:
    which = "stockfish"
    if which == "leela":
        the_breaks = leela_breaks
        the_config = leela_config
    elif which == "stockfish":
        the_breaks = stockfish_breaks
        the_config = stockfish_config
    else:
        exit(0)
    hopeless_engine = hf.initialize_engine(which, the_config)
    analyze_hopeless(hopeless_engine, f"output-{which}/results.csv", the_breaks)
    hopeless_engine.quit()
    exit(0)
"""



