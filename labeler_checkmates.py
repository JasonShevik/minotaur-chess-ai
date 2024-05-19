import chess.engine
import chess
from heapdict import heapdict
import itertools
import logging
import csv


# This function will take in a filepath and open the file, then loop through each of the FEN game strings.
# It will analyze each string to a depth of 10-20 to determine if there is a forced checkmate sequence.
# It then collects the positions that are a forced checkmate, along with the length of the sequence.
def collect_positions(fp, engine, current_row):
    with open(fp) as file:
        for position in itertools.islice(file, current_row, None):
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

                    # if the depth is 10 and the absolute score is below a certain amount, skip forward




#
def expand_position():
    positions_explore = heapdict()

    positions_finished = set()
    # positions_finished.add()
    # "" in positions_finished




# Initialize and configure the engine
stockfish_engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-avx2.exe")
stockfish_config = {"Threads": 8,
                    "Hash": 20000,
                    "UCI_Elo": 3190}
stockfish_engine.configure(stockfish_config)


collect_positions(fp="lichess-positions/lichess_positions_part_5.txt", engine=stockfish_engine, current_row=0)







