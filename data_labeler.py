import chess.engine
import chess
import itertools
import logging
import time
import csv
import os


# These breakdowns represent the depth that the engine evaluates to based on the score
# If you reach the move and the evaluation is greater than the score, you stop analyzing
# This is to optimize the data labeling. Positions where one side has a significant advantage are less critical
#                    [move, score (centipawns)]
depth_score_breaks = [[20,  600],
                      [30,  400],
                      [40,  100],
                      [50,  50],
                      [60,  -1]]  # Maximum depth
hopeless = 1500

# Directory for the positions to be labeled
positions_filepath = "lichess-positions/lichess_positions_part_1.txt"
# Directory for chess engines to be used in the analysis
stockfish_dir = "stockfish/stockfish-windows-x86-64-avx2.exe"
# leela_dir = ""
output_filepath = "analysis-output/results.csv"
progress_filepath = "analysis-output/progress.csv"

# Set up the engine, and configure it to utilize a lot of resources
engine = chess.engine.SimpleEngine.popen_uci(stockfish_dir)
engine.configure({"Threads": 15,
                  "Hash": 30000,
                  "UCI_Elo": 3190})

# Start the log
logging.basicConfig(filename="chess_engine.log",
                    level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# The function to analyze a chunk of positions
def analyze_chunk(chunk_start, chunk_length):
    # Start a log for this chunk in case something goes wrong
    logging.basicConfig(filename="chess_engine.log",
                        level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    chunk_place = 0
    # Open the positions file (positions to analyze) and output file
    with (open(positions_filepath, "r") as source_file,
          open(output_filepath, "a") as output_file):

        # If this is the first run, put the header row in the results file
        if chunk_start == 0:
            output_file.write("Position,Move,Depth,Score\n")

        # Begin looping through each game, starting at chunk_start
        for position in itertools.islice(source_file, chunk_start, None):
            time.sleep(2)
            # Set the board object for this position
            board = chess.Board(position)

            # Set the index that we use with depth_score_breaks to determine depth based on a score threshold
            threshold_index = 0

            # Begin the analysis of this position
            with engine.analysis(board) as analysis:
                os.system("cls")
                print(f"Chunk progress: {int((chunk_place/chunk_length)*100)}%")

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

                    # Retrieve the absolute value of the score
                    score = abs(int(str(score.pov(True))))

                    # If the game is completely and utterly lost, stop the eval; there's no need to go deeper
                    if score > hopeless:
                        # Get the first move of the principle variation
                        best_move = info.get("pv")[0]
                        # Write to the file
                        output_file.write(f"{position.rstrip()},{best_move},{depth},{score}\n")
                        break

                    # If we've reached a depth milestone specified by depth_score_breaks
                    if depth == depth_score_breaks[threshold_index][0]:

                        # If the score is greater than allowed by depth_score_breaks for this depth
                        if score > depth_score_breaks[threshold_index][1]:
                            # Get the first move of the principle variation
                            best_move = info.get("pv")[0]
                            # Write to the file
                            output_file.write(f"{position.rstrip()},{best_move},{depth},{score}\n")
                            break
                        else:
                            # The evaluation is close enough that we want to analyze deeper
                            threshold_index += 1

            # Break out once we've finished analyzing all of the positions in this chunk
            chunk_place += 1
            if chunk_place == chunk_length:
                break


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
            starting_row = progress_dict[positions_filepath]
        else:
            # Create an empty line for this positions_filepath file
            progress_file.write(f"{positions_filepath},0\n")

            # Add this positions file to the dictionary
            progress_dict[positions_filepath] = 0
            # Set the starting row for the positions_filepath file for analysis
            starting_row = 0
else:
    # Create a new progress file
    with open(progress_filepath, "a") as progress_file:
        # Write the header
        progress_file.write("File,Row\n")
        # Make note that we have made no progress yet on our current file
        progress_file.write(f"{positions_filepath},0\n")

        # Add this positions file to the dictionary
        progress_dict[positions_filepath] = 0
        # Set the starting row for the positions_filepath file for analysis
        starting_row = 0


chunk_length = 2
for i in range(2):
    # Analyze this chunk
    analyze_chunk(chunk_start=starting_row, chunk_length=chunk_length)

    # Increase the starting row by the chunk_length, so that in the next iteration we start at the right place
    starting_row += chunk_length
    # Update the progress for this positions_filepath
    progress_dict[positions_filepath] += chunk_length

    # Create a new progress file to save our new progress
    with open(progress_filepath, "w") as new_progress_file:
        # Write the header of the new progress file
        new_progress_file.write("File,Row\n")

        # Iterate over the keys in the progress dictionary
        for key in progress_dict:
            # Write the current dictionary entry to the new progress file
            new_progress_file.write(f"{key},{progress_dict[key]}\n")

engine.quit()

