import itertools
import pandas as pd
import csv
import os
from typing import List, Dict


# Takes in a FEN string and returns a list of 64 numbers
# https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
def fen_to_vector(fen: str) -> List[float]:
    piece_values: Dict[str, int] = {"p": 1,
                                    "n": 2,
                                    "b": 3,
                                    "r": 4,
                                    "q": 5,
                                    "k": 6}

    # Split the fen by spaces
    fen_parts: List[str] = fen.split(" ")

    # The first part is the board portion. Split it by '/' to get each row
    row_strings: List[str] = fen_parts[0].split("/")

    # Initialize some variables for constructing the vector
    vector_version: List[float] = [0 for _ in range(64)]
    index_buffer_num_rows: int = 0

    # Vectors go a1, b1, ... a2, b2, ..., so if we're white then we need to go through the strings in reverse order
    if fen_parts[1] == "w":
        row_strings = list(reversed(row_strings))

    # If playing as black, then we want to go through the strings in the current order
    # BUT we need to reverse all of the individual strings like we're rotating the board
    elif fen_parts[1] == "b":
        for index, _ in enumerate(row_strings):
            row_strings[index] = str(reversed(row_strings[index]))

    # This value can only be 'w' or 'b'
    else:
        print(f"Invalid FEN: {fen_parts[0]}")
        return False

    # Iterate over the rows backwards (start from row 1 and go up)
    current_row: str
    for current_row in row_strings:
        index_buffer_this_row: int = 0

        # Loop through each character in this row of the chess board
        index: int
        character: str
        for index, character in enumerate(current_row):
            # If the character is numerical...
            if character.isdigit():
                # Record the number of sequential empty squares
                index_buffer_this_row += int(character) - 1

                # For each of the empty squares
                index_to_zero: int
                for index_to_zero in range(int(character)):
                    # Set that index of the vector_version to zero
                    # (index_buffer_num_rows * 8) because we need to offset by the number of rows we've already done
                    # index because we need to offset by the number of characters in this row we've already done
                    # index_to_zero because we need to count up how many zeros we're adding based on the character
                    # (index_buffer_this_row - int(character) + 1) because ...
                    # if not first digit in row, need offset by more
                    vector_version[index + index_to_zero + (index_buffer_num_rows * 8) + (index_buffer_this_row - int(character) + 1)] = 0
            # If the character is alphabetical...
            else:
                # Set the value in the vector to the piece value
                true_index: int = index + (index_buffer_num_rows * 8) + index_buffer_this_row
                vector_version[true_index] = piece_values[character.lower()]

                # If the piece is black
                if character.islower():
                    # And the AI is playing as white
                    if fen_parts[1] == "w":
                        # Then multiply it by -1 because it's on the opponent's team
                        vector_version[true_index] *= -1
                # If the piece is white
                else:
                    # And the AI is playing as black
                    if fen_parts[1] == "b":
                        # Then multiply it by -1 because it's on the opponent's team
                        vector_version[true_index] *= -1

        index_buffer_num_rows += 1

    # ----- Castling -----

    # King value of +/-6 is modified by +/-0.1 and 0.2
    # This means the square can have 4 possible values:
    # 6: May not castle
    # 6.1: May castle king-side
    # 6.2: May castle queen-side
    # 6.3: May castle either side

    # Establish what side we're on so that we know if the king is a positive or negative number
    if fen_parts[1] == "w":
        white_mod: int = 1
        black_mod: int = -1
    else:
        white_mod: int = -1
        black_mod: int = 1

    # Since we may modify kings multiple times in a row...
    # Doing this work on separate list using indices of the old list so .index() works properly
    working_castle_vector = vector_version[:]

    # White may castle king-side
    if "K" in fen_parts[2]:
        working_castle_vector[vector_version.index(white_mod * 6)] += (white_mod * 0.1)
    # White may castle queen-side
    if "Q" in fen_parts[2]:
        working_castle_vector[vector_version.index(white_mod * 6)] += (white_mod * 0.2)
    # Black may castle king-side
    if "k" in fen_parts[2]:
        working_castle_vector[vector_version.index(black_mod * 6)] += (black_mod * 0.1)
    # Black may castle queen-side
    if "q" in fen_parts[2]:
        working_castle_vector[vector_version.index(black_mod * 6)] += (black_mod * 0.2)

    # Save changes to the original list
    vector_version = working_castle_vector[:]

    # ----- En Passant -----

    # If there is no En Passant, finish
    if fen_parts[3][0] == "-":
        return vector_version

    # Based on if the AI's perspective is white or black, define character_values for board orientation
    if white_mod == 1:
        character_values = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
        en_passant_square = (character_values[fen_parts[3][0]] * 8) + int(fen_parts[3][1])
    else:
        character_values = {"a": 7, "b": 6, "c": 5, "d": 4, "e": 3, "f": 2, "g": 1, "h": 0}
        en_passant_square = (character_values[fen_parts[3][0]] * 8) + (8 - int(fen_parts[3][1]))

    # noinspection PyTypeChecker
    vector_version[en_passant_square] = -0.5

    return vector_version






