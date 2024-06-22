import chess.engine
import chess
from heapdict import heapdict
import helper_functions
import threading
import itertools
import logging
import csv


# This function will take in a filepath and open the file, then loop through each of the FEN game strings.
# It will analyze each string to a depth of 10-20 to determine if there is a forced checkmate sequence.
# It then collects the positions that are a forced checkmate, along with the length of the sequence.
def collect_positions(data_filepath, output_filepath, progress_filepath, engine, current_row):
    with open(data_filepath) as file:
        for position in itertools.islice(file, current_row, None):
            if not stop_event.is_set():
                current_row += 1
                if current_row % 50 == 0:
                    print(f"Progress: {current_row}\tTotal Checkmates: {}")

                board = chess.Board(position)
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

                        score = str(score.pov(True))
                        # If we discovered a checkmate before we realized it's hopeless
                        if "#" in score:

                            # Then we need to record the position

                        # if the depth is 10 and the relative score is below a certain amount, skip forward
            else:
                break

    engine.quit()
    print("Done!")


#
def expand_position(initial_position):
    to_explore_up = heapdict()
    to_explore_down = heapdict()
    # Add initial_position to to_explore_down

    positions_finished = set()
    # positions_finished.add()
    # "" in positions_finished


    # There are two functions that will make this function work: expand_up and expand_down
    # expand_down finds all the sequences going toward the checkmate.
    # It may take in a #+10 position as input and output a dictionary of FENs and best moves going down to checkmate.
    # Along the way, it also adds every position to the to_explore_up heapdict.
    # Once expand_down is done, you go through every position in to_explore_up and expand_up each one.
    # expand_up looks at all positions that could precede the current position, and checks if any are forced checkmate.
    # If a position is a forced checkmate, it is added to to_explore_down
    # After checking every possible preceding position, that position is popped from to_explore_up.
    # Then you run expand_down on each of positions, if any, in to_explore_down
    # Repeat until to_explore_up and to_explore_down are both empty.

    # Try to use some elegant recursion, but keep it efficient.

    # For maximum speed/efficiency, don't analyze from the losing side.
    # Just find the set of legal responses from losing, then analyze the winning moves (forced checkmate moves).
    # I can make a separate function to deeply analyze losing positions at a later date to improve defensiveness.
    pass

#
def expand_up(initial_position):
    pass

#
def expand_down(initial_position):
    pass



# Initialize and configure the engine
stockfish_engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-avx2.exe")
stockfish_config = {"Threads": 6,
                    "Hash": 30000,
                    "UCI_Elo": 3190}
stockfish_engine.configure(stockfish_config)

progress_filepath = "training-supervised-checkmates/.txt"
output_filepath = "training-supervised-checkmates/.txt"
data_filepath = "lichess-positions/lichess_positions_part_5.txt"

if os.path.exists(progress_filepath):
    pass

# Create a stop event object, so that we can end the analysis on demand
stop_event = threading.Event()

checkmate_analysis_thread = threading.Thread(target=collect_positions, args=(data_filepath, output_filepath, progress_filepath, stockfish_engine, 0))

# Start the threads
checkmate_analysis_thread.start()

# Wait for the user to press the key
keyboard.wait("q")

# Indicate that the stop condition has been met, so the loops in continuous_analysis should finish current position
stop_event.set()

print()
print("Stop key detected!")
print("Finishing final calculations...\n")

checkmate_analysis_thread.join()



