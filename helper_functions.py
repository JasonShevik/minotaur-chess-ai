import chess.engine
import chess
import itertools
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


# This function opens the progress file (which has the lookup info) and the positions file (the key for row value)
def process_progress_file(progress_filepath, positions_filepath):
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

    return [progress_dict, current_row]


def engine_loop(engine, functions_dict, data_dict):
    """
    This is a flexible function to continuously analyze positions with user-specified conditions.

    :param engine: The chess.engine object that conducts the analysis.
    :param functions_dict:
        A dictionary with values that are higher-order functions which this function uses.

        Required functions:
        Stop: A function that returns True if the engine_loop should stop and False if it should continue.
        Output: A function that saves the results of the analysis.
    :param data_dict:
        A dictionary containing data used by either this function or those in the functions_dict.
        User can optionally store additional information in this dictionary from within the functions_dict functions.

        User created:
        Name: The name of the engine. Used by functions_dict functions.
        Source: The filepath of the source file (the file containing the FEN codes to analyze).
        Row: The row to start on from the source file.

        Breaks: Not required, but can be used by the Stop function to determine what depth to break at

        Function created:
        Nodes: Number of nodes explored.
        Threshold Index: The index for which depth threshold to use in the optional Breaks value.
            This function only initializes the index. The Stop function may be required to maintain it.
    """

    with open(data_dict["Source"], "r") as source_file:
        # Loop through each position in the source file starting at the current row
        for position in itertools.islice(source_file, data_dict["Row"], None):
            if not data_dict["Stop"].is_set():

                # Update the row number and give the user an update
                data_dict["Row"] += 1
                print(f"{data_dict["Name"]}: {data_dict["Row"]}")

                # Parse the position and make an object
                board = chess.Board(position)

                # Set the index that we use with data_dict["Breaks"] to determine depth based on a score threshold
                data_dict["Threshold Index"] = 0

                # Begin the analysis of this board position
                with engine.analysis(board) as analysis:
                    # Number of nodes explored is 0 at the start of the analysis
                    data_dict["Nodes"] = 0
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

                        # If the user-specified stop conditions are met
                        if functions_dict["Stop"](info, data_dict):
                            # Save the results and clean up the environment
                            functions_dict["Output"](position, info, data_dict)
                            break

                        # Update the number of explored nodes
                        data_dict["Nodes"] = info.get("nodes")

            # If the stop_event was set
            else:
                engine.quit()
                print(f"{data_dict["Name"]} done!")
                break









