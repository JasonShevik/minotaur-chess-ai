import chess.engine
import chess
import itertools
import threading
import keyboard
import logging
import time
import csv
import os


# This function creates and returns a chess engine object according to the configuration settings
def initialize_engine(which_engine, configure_options):
    if which_engine == "leela":
        engine_dir = "lc0-v0.30.0-windows-gpu-nvidia-cuda/lc0.exe"
    elif which_engine == "stockfish":
        engine_dir = "stockfish/stockfish-windows-x86-64-avx2.exe"
    else:
        print("Invalid Engine Choice")
        engine_dir = ""

    this_engine = chess.engine.SimpleEngine.popen_uci(engine_dir)
    this_engine.configure(configure_options)
    return this_engine


#
def continuous_analysis(name, engine, depth_score_breaks, positions_filepath, output_filepath, progress_filepath):
    # Initialize a dictionary for the progress associated with each positions file
    progress_dict = {}

    # Check for a progress file
    if os.path.exists(progress_filepath):
        # Open the progress file we found
        with open(progress_filepath, "r") as progress_file:
            progress_reader = csv.reader(progress_file)
            # Skip the header row
            next(progress_reader, None)

            # Read every row of the progress file into a dictionary
            for row in progress_reader:
                progress_dict[row[0]] = int(row[1])

            # Check if the progress file included the positions_filepath
            if positions_filepath in progress_dict:
                # Find the starting row that corresponds to the positions_filepath file for analysis
                current_row = progress_dict[positions_filepath] - 1
            else:
                # Create an empty line for this positions_filepath file
                progress_file.write(f"{positions_filepath},0\n")

                # Add this positions file to the dictionary
                progress_dict[positions_filepath] = 0
                # Set the starting row for the positions_filepath file for analysis
                current_row = -1
    else:
        # Create a new progress file
        with open(progress_filepath, "a") as progress_file:
            # Write the header
            progress_file.write("File,Row\n")
            # Make note that we have made no progress yet on our current file
            progress_file.write(f"{positions_filepath},0\n")

            # Add this positions file to the dictionary
            progress_dict[positions_filepath] = 0
            # Set the starting row for the positions_filepath file for analysis.
            current_row = -1

    # Open the files for positions to analyze and for outputting results
    with (open(positions_filepath, "r") as source_file,
          open(output_filepath, "a") as output_file):

        # If this is the first run, put the header row in the results file
        if current_row == 0:
            output_file.write("Position,Move,Depth,Score\n")

        # Begin looping through each game, starting at current_row
        for position in itertools.islice(source_file, current_row, None):
            if not stop_event.is_set():
                current_row += 1
                print(f"{name}: {current_row}")

                # Set the board object for this position
                board = chess.Board(position)

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
                        # Score can be None when Stockfish looks at sidelines - Skip those iterations
                        if score is None:
                            continue

                        score = str(score.pov(True))
                        # If we discovered a checkmate before we realized it's hopeless
                        if "#" in score:
                            # Get the first move of the principle variation
                            best_move = info.get("pv")[0]
                            # Write to the file
                            output_file.write(f"{position.rstrip()},{best_move},{depth},{str(info.get('score').pov(True))}\n")

                            # Update our progress number, since we finished a position
                            progress_dict[positions_filepath] += 1

                            # Create a new progress file to save our new progress
                            with open(progress_filepath, "w") as new_progress_file:
                                # Write the header of the new progress file
                                new_progress_file.write("File,Row\n")

                                # Iterate over the keys in the progress dictionary
                                for key in progress_dict:
                                    # Write the current dictionary entry to the new progress file
                                    new_progress_file.write(f"{key},{progress_dict[key]}\n")
                            break

                        # Retrieve the absolute value of the score
                        score = abs(int(score))

                        num_nodes = info.get("nodes")
                        if num_nodes_last >= num_nodes:
                            # Get the first move of the principle variation
                            best_move = info.get("pv")[0]
                            # Write to the file
                            output_file.write(f"{position.rstrip()},{best_move},{depth},{str(info.get('score').pov(True))}\n")

                            # Update our progress number, since we finished a position
                            progress_dict[positions_filepath] += 1

                            # Create a new progress file to save our new progress
                            with open(progress_filepath, "w") as new_progress_file:
                                # Write the header of the new progress file
                                new_progress_file.write("File,Row\n")

                                # Iterate over the keys in the progress dictionary
                                for key in progress_dict:
                                    # Write the current dictionary entry to the new progress file
                                    new_progress_file.write(f"{key},{progress_dict[key]}\n")
                            break
                        num_nodes_last = num_nodes

                        # If we've reached a depth milestone specified by depth_score_breaks
                        if depth == depth_score_breaks[threshold_index][0]:

                            # If the score is greater than allowed by depth_score_breaks for this depth
                            if score > depth_score_breaks[threshold_index][1]:
                                # Get the first move of the principle variation
                                best_move = info.get("pv")[0]
                                # Write to the file
                                output_file.write(f"{position.rstrip()},{best_move},{depth},{str(info.get('score').pov(True))}\n")

                                # Update our progress number, since we finished a position
                                progress_dict[positions_filepath] += 1

                                # Create a new progress file to save our new progress
                                with open(progress_filepath, "w") as new_progress_file:
                                    # Write the header of the new progress file
                                    new_progress_file.write("File,Row\n")

                                    # Iterate over the keys in the progress dictionary
                                    for key in progress_dict:
                                        # Write the current dictionary entry to the new progress file
                                        new_progress_file.write(f"{key},{progress_dict[key]}\n")
                                break
                            else:
                                # The evaluation is close enough that we want to analyze deeper
                                threshold_index += 1
            # If the stop_event was set
            else:
                break

    # This will trigger when the stop event is set
    engine.quit()
    print(f"{name} done!")


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


# Start the log
logging.basicConfig(filename="chess_engine.log",
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# These breakdowns represent the depth that the engine evaluates to based on the score
# If you reach the move and the evaluation is greater than the score, you stop analyzing
# This is to optimize the data labeling. Positions where one side has a significant advantage are less critical
#                    [move, score (centipawns)]
leela_breaks = [[10, 400],
                [13, 200],
                [16, 100],
                [19, -1]]  # Maximum depth

stockfish_breaks = [[20, 400],
                    [30, 200],
                    [40, 100],
                    [50, -1]]  # Maximum depth

leela_config = {"Threads": 6,
                "NNCacheSize": 1000000,
                "MinibatchSize": 1024,
                #"WeightsFile": "lc0-v0.30.0-windows-gpu-nvidia-cuda/768x15x24h-t82-2-swa-5230000.pb",
                "RamLimitMb": 20000}

stockfish_config = {"Threads": 8,
                    "Hash": 20000,
                    "UCI_Elo": 3190}


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
    hopeless_engine = initialize_engine(which, the_config)
    analyze_hopeless(hopeless_engine, f"output-{which}/results.csv", the_breaks)
    hopeless_engine.quit()
    exit(0)

# Create a stop event object, so that we can end the analysis on demand
stop_event = threading.Event()

# Create the threads for the engines we're analyzing with
# (name, engine, depth_score_breaks, positions_filepath, output_filepath, progress_filepath)
leela_thread = threading.Thread(target=continuous_analysis, args=("Leela", initialize_engine("leela", leela_config), leela_breaks, "lichess-positions/lichess_positions_part_2.txt", "training-supervised-engines/results_part_2_leela.csv", "training-supervised-engines/progress_part_2_leela.csv"))
stockfish_thread = threading.Thread(target=continuous_analysis, args=("Stockfish", initialize_engine("stockfish", stockfish_config), stockfish_breaks, "lichess-positions/lichess_positions_part_1.txt", "training-supervised-engines/results_part_1_stockfish.csv", "training-supervised-engines/progress_part_1_stockfish.csv"))

# Start the threads
leela_thread.start()
stockfish_thread.start()

# Wait for the user to press the key
keyboard.wait("q")

# Indicate that the stop condition has been met, so the loops in continuous_analysis should finish current position
stop_event.set()

print()
print("Stop key detected!")
print("Finishing final calculations...\n")

leela_thread.join()
stockfish_thread.join()
