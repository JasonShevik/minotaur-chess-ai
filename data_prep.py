import itertools
import pandas as pd
import csv
import os


# Takes in a FEN string and returns a list of 64 numbers
# https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
def fen_to_vector(fen):
    piece_values = {"p": 1,
                    "n": 2,
                    "b": 3,
                    "r": 4,
                    "q": 5,
                    "k": 6}

    # Split the fen by spaces
    fen_parts = fen.split(" ")

    # The first part is the board portion. Split it by '/' to get each row
    row_strings = fen_parts[0].split("/")

    # Initialize some variables for constructing the vector
    vector_version = [0 for _ in range(64)]
    index_buffer_num_rows = 0

    # Vectors go a1, b1, ... a2, b2, ..., so if we're white then we need to go through the strings in reverse order
    if fen_parts[1] == "w":
        row_strings = reversed(row_strings)

    # If playing as black, then we want to go through the strings in the current order
    # BUT we need to reverse all of the individual strings like we're rotating the board
    elif fen_parts[1] == "b":
        for index, _ in enumerate(row_strings):
            row_strings[index] = reversed(row_strings[index])

    # This value can only be 'w' or 'b'
    else:
        print(f"Invalid FEN: {fen_parts[0]}")
        return False

    # Iterate over the rows backwards (start from row 1 and go up)
    for current_row in row_strings:
        index_buffer_this_row = 0

        # Loop through each character in this row of the chess board
        for index, character in enumerate(current_row):
            # If the character is numerical...
            if character.isdigit():
                # Record the number of sequential empty squares
                index_buffer_this_row += int(character) - 1

                # For each of the empty squares
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
                true_index = index + (index_buffer_num_rows * 8) + index_buffer_this_row
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
        white_mod = 1
        black_mod = -1
    else:
        white_mod = -1
        black_mod = 1

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


# Take all of the results files in a directory and merge them into a single file with vectorized fen
def merge_and_vectorize(directory, pretraining=False, filter_mates=False):
    # Collect the list of files to merge from the directory (iff they have "results_" in the name)
    files_to_merge = [file_name for file_name in os.listdir(directory) if ("results_" in file_name or pretraining is True)]
    if not files_to_merge:
        print("No files to merge!")
        return False

    # Create a dataframe out of the first file, and remove that file from the list
    merged_df = pd.read_csv(directory + files_to_merge.pop(0))

    # Iterate over the remaining files
    for filename in files_to_merge:
        # Concat each next file to the end of the merged_df
        merged_df = pd.concat([merged_df, pd.read_csv(filename)], ignore_index=True)

    # If the filter_mates flag is set, then we want to remove the forced checkmates from merged_df
    if filter_mates:
        # remove_checkmates function saves the checkmates to a separate file
        merged_df = remove_checkmates(merged_df)

    # Use list comprehensions to create a list of every combination of a1 through h8
    letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
    numbers = [str(x + 1) for x in range(8)]
    df_column_names = [a + x for x in numbers for a in letters]

    # Use a list comprehension to get the list of lists for the dataframe
    list_of_vectors = [fen_to_vector(fen) for fen in merged_df["Position"]]
    converted_positions_df = pd.DataFrame(list_of_vectors, columns=df_column_names)

    # Add the FEN column to the beginning of the dataframe so that we know what FEN the vector refers to
    converted_positions_df.insert(0, "FEN", merged_df["Position"])

    if pretraining:
        # Print the result with 64 input columns to a csv for pretraining
        converted_positions_df.to_csv(f"{directory}merged_for_training.csv", index=False)
    else:
        # Make a dictionary to quickly look up values for move columns/rows
        character_values = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8,
                            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8}

        # Create a list to hold all of our data labels
        label_list = []

        # Go through each row of the input file
        for i in range(len(merged_df.index)):
            # Go through each character in the labeled move string
            columns_and_rows = []
            for character in merged_df.iloc[i]["Move"]:
                # Add the value of that
                columns_and_rows.append(character_values[character])

            # Add those column and row labels to the list of labels
            label_list.append(columns_and_rows)

        # Use list comprehensions to grab the right elements of each label to create a list for each of the columns
        for index, name in enumerate(["s_row", "s_column", "f_row", "f_column"]):
            converted_positions_df[name] = [these_labels[index] for these_labels in label_list]

        # Print the result with 64 input columns and 4 label columns to a csv
        converted_positions_df.to_csv(f"{directory}merged_for_training.csv", index=False)


# Take in a filepath, remove all forced checkmate rows, save the filtered file, return checkmates dataframe
def remove_checkmates(input_df):
    save_path = "training-supervised-checkmates/"

    checkmates_df = input_df[input_df["Score"].str.contains("#")]
    checkmates_df.to_csv(f"{save_path}results_merged_filtered.csv", index=False)

    non_checkmates_df = input_df[~input_df["Score"].str.contains("#")]
    return non_checkmates_df


# Prepares the data for the supervised training phase (including the depth breaks and checkmates data)
def prepare_training_files():
    depth_breaks_dir = "training-supervised-engines/"
    checkmates_dir = "training-supervised-checkmates/"

    # Merge the depth breaks files with filter_mates as true
    merge_and_vectorize(depth_breaks_dir, pretraining=False, filter_mates=True)

    # Merge the checkmates files which now includes those that were filtered from depth breaks
    merge_and_vectorize(checkmates_dir, pretraining=False, filter_mates=False)


# Prepares the data for the pretraining phase
def prepare_pretraining_files():
    pretraining_data_dir = "training-pretraining/"

    merge_and_vectorize(pretraining_data_dir, pretraining=True)


prepare_pretraining_files()

