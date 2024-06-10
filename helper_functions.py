import chess.engine
import chess
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


#
# <Assumes that the output file already has a header>
def engine_loop(name, engine, source_file, current_row, stop_condition, output_function):
    """

    :param name:
    :param engine:
    :param source_file:
    :param current_row:
    :param stop_condition:
    :param output_function:
    :return:
    """

    for position in itertools.islice(source_file, current_row, None):
        current_row += 1
        print(f"{name}: {current_row}")

        board = chess.Board(position)

        # Begin the analysis of this position
        with engine.analysis(board) as analysis:

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

                #
                if stop_condition(info, name):
                    output_function()























