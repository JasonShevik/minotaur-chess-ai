import chess
import chess.engine
import os


# These breakdowns represent the depth that the engine evaluates to based on the score
# If you reach the move and the evaluation is greater than the score, you stop analyzing
# This is to optimize the data labeling. Positions where one side has a significant advantage are less critical
#                    [move, score]
depth_score_breaks = [[20,  8],
                      [30,  4],
                      [40,  2],
                      [50,  1],
                      [60,  -1]]  # Maximum depth
hopeless = 300

# Directory for the positions to be labeled
positions_dir = "lichess-positions/lichess_positions_part_1.txt"
# Directory for chess engines to be used in the analysis
stockfish_dir = "stockfish/stockfish-windows-x86-64-avx2.exe"
# leela_dir = ""
output_filepath = "lichess-positions/results.csv"

# Set up the engine, and configure it to utilize a lot of resources
engine = chess.engine.SimpleEngine.popen_uci(stockfish_dir)
engine.configure({"Threads": 19,
                  "Hash": 45000})

# Check if this is the first time this is being run
if os.path.exists(output_filepath):
    first_run = False
else:
    first_run = True

foobar = 0

# Open the positions file and begin looping through each game (in PGN format)
with (open(positions_dir, "r") as source_file,
      open(output_filepath, "a") as output_file):
    # If this is the first run, put the header row in the results file
    if first_run:
        output_file.write("Position,Move,Depth\n")

    # Begin looping through each game (in PGN format)
    for position in source_file:
        # Set the board object for this position
        board = chess.Board(position)

        # Set the threshold that we use with depth_score_breaks to determine the depth to analyze to
        threshold_index = 0

        # Begin the analysis of this position
        with engine.analysis(board) as analysis:

            print()

            # Analyze continuously, depth by depth, until we meet a break condition
            for info in analysis:
                # Get the current depth
                depth = info.get("depth")
                score = info.get("score")

                if depth is not None:

                    # There is an error causing some positions to give a score of None
                    # This seems to occur when positions are very equal at somewhat high depth (around 27-33)
                    if score is None:
                        print(position)
                        print()
                        print(board)
                        print()
                        print(depth)
                        print()
                        output_file.write(f"{position.rstrip()},--,{depth},ERROR\n")
                        break



                    # Retrieve the absolute value of the score
                    score = abs(int(str(score.pov(True))))

                    print(score)

                    # If the game is completely and utterly lost, stop the eval; there's no need to go deeper
                    if score > hopeless:
                        # Get the first move of the principle variation
                        best_move = info.get("pv")[0]
                        # Write to the file
                        output_file.write(f"{position.rstrip()},{best_move},{depth}\n")
                        break

                    # If we've reached a depth milestone specified by depth_score_breaks
                    if depth == depth_score_breaks[threshold_index][0]:

                        # If the score is greater than allowed by depth_score_breaks for this depth
                        if score > depth_score_breaks[threshold_index][1]:
                            # Get the first move of the principle variation
                            best_move = info.get("pv")[0]
                            # Write to the file
                            output_file.write(f"{position.rstrip()},{best_move},{depth}\n")
                            break
                        else:
                            # The evaluation is close enough that we want to analyze deeper
                            threshold_index += 1

        foobar += 1
        if foobar == 20:
            break
        # break  # Break out of for position in file after 1


engine.quit()

# TODO: add a way to keep track of how far into the source_file you got.
# TODO: fix the error causing Stockfish to refuse to evaluate deeper than around 28-33 depth.
# TODO: change depth_score_breaks thresholds.


