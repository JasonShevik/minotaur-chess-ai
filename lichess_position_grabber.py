import chess.pgn
import os
from random import randrange

positions_per_game = 1

# Loop through each of the database files in the lichess-positions directory
for filename in os.listdir("lichess-positions"):
    print("Beginning:\t" + filename)

    # Only proceed if it is a pgn files (Portable Game Notation), and open it
    if filename[-4:].lower() == ".pgn":
        with (open("lichess-positions/" + filename) as pgn):

            # Loop through every game in the pgn file
            for current_game in iter(lambda: chess.pgn.read_game(pgn), None):

                # A list that will hold the indices for the randomly chosen positions
                chosen_positions = []

                # Unfortunately, current_game.mainline_moves() does not support len()
                # So we must iterate through the whole game once to determine the total number of moves
                total_number_moves = sum(1 for _ in current_game.mainline_moves())

                # We will pick a number of random positions from the current_game
                # That number will be the smaller between positions_per_game and total_number_moves - 2
                # It is minus two because we discard the starting position and the final position
                # We discard starting position because there are only 960 of them, and we want millions of positions
                for _ in range(min(positions_per_game, total_number_moves - 2)):

                    # Pick random indices in the range ensuring we do not choose a duplicate
                    index = randrange(start=1, stop=(total_number_moves - 1))
                    while index in chosen_positions:
                        index = randrange(start=1, stop=(total_number_moves - 1))

                    # Add that index to our list
                    chosen_positions.append(index)

                # Sort the chosen_positions so that when we iterate through the game, we only have to do it once
                chosen_positions = sorted(chosen_positions)

                # Iterate through the game to grab the chosen_positions
                board = current_game.board()
                move_num = 0
                for move in first_game.mainline_moves():
                    # If we remove all the indices from the list before the end of the game, we're done early
                    if len(chosen_positions) == 0:
                        break

                    # Make and count the move
                    board.push(move)
                    move_num += 1

                    if move_num == current_game[0]:
                        # Export the current board position to a file
                        # NOT YET DONE ---

                        # Remove that position from chosen_positions
                        del chosen_positions[0]

                break  # Break out of current_game in the file after 1

    break  # Break out of pgn in the directory after 1






