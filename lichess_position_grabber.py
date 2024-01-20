import chess.pgn
import os
from random import randrange

positions_per_game = 1
output_positions = set()

# Loop through each of the database files in the lichess-games directory
for filename in os.listdir("lichess-games"):

    # Only proceed if it is a pgn files (Portable Game Notation), and open it
    if filename[-4:].lower() == ".pgn":
        print("Beginning:\t" + filename)
        with open("lichess-games/" + filename) as pgn:

            # Loop through every game in the pgn file
            for current_game in iter(lambda: chess.pgn.read_game(pgn), None):

                # A list that will hold the indices for the randomly chosen positions
                chosen_positions = []

                # Unfortunately, current_game.mainline_moves() does not support len()
                # So we must iterate through the whole game once to determine the total number of moves
                total_number_moves = sum(1 for _ in current_game.mainline_moves())

                # Chess is solved with 7 or fewer pieces, so we will exclude those positions
                # For those, we will train on endgame databases because it's higher quality data
                # Loop through the game and find out what move number, if any, brought the board to 7 pieces
                board = current_game.board()
                move_num = 0
                upper_limit_rand = total_number_moves - 1
                for move in current_game.mainline_moves():
                    # Make and count the move
                    board.push(move)
                    move_num += 1

                    if len(board.piece_map()) == 7:
                        upper_limit_rand = move_num - 1
                        break

                # We will pick a number of random positions from the current_game
                # That number will be the smaller between positions_per_game and upper_limit_rand - 1
                # It is minus one because we discard the starting position (final position was already excluded)
                # We discard starting position because there are only 960 of them, minimize collisions
                for _ in range(min(positions_per_game, upper_limit_rand - 1)):

                    # Pick random indices in the range ensuring we do not choose a duplicate
                    index = randrange(start=1, stop=upper_limit_rand)
                    while index in chosen_positions:
                        index = randrange(start=1, stop=upper_limit_rand)

                    # Add that index to our list
                    chosen_positions.append(index)

                # Sort the chosen_positions so that, when we iterate through the game, we only have to do it once
                chosen_positions = sorted(chosen_positions)

                # Iterate through the game to grab the chosen_positions
                board = current_game.board()
                move_num = 0
                for move in current_game.mainline_moves():
                    # If we remove all the indices from the list before the end of the game, we're done early
                    if len(chosen_positions) == 0:
                        break

                    # Make and count the move
                    board.push(move)
                    move_num += 1

                    if move_num == chosen_positions[0]:
                        # Add the current board position to the output_positions set in Forsyth–Edwards Notation format
                        output_positions.add(board.fen())

                        # Remove that position from chosen_positions
                        del chosen_positions[0]

                # break  # Break out of for current_game in the file after 1
    # break  # Break out of for pgn in the directory after 1

# Create a file to store the outputs and write to it
print("\nWriting the selected positions to the output file.")
with open("chosen_lichess_positions.txt", "w") as output_file:
    output_file.write("\n".join(output_positions))


# Define a function to split the large file into smaller ones
def split_file(original_file, num_files=5):
    # Open the original file and read the lines
    with open(original_file, 'r') as file:
        lines = file.readlines()

    # Calculate the number of lines per file
    lines_per_file = len(lines) // num_files

    for i in range(num_files):
        with open(f'lichess_positions_part_{i+1}.txt', 'w') as file:
            # Write a segment of lines to each new file
            file.writelines(lines[i*lines_per_file : (i+1)*lines_per_file])

    # Handle any remaining lines for the last file
    if len(lines) % num_files != 0:
        with open(f'lichess_positions_part_{num_files}.txt', 'a') as file:
            file.writelines(lines[num_files*lines_per_file:])


split_file('chosen_lichess_positions.txt')


