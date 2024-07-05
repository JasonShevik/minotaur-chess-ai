import itertools
import pandas
import csv
import os


# Takes in a FEN string and returns a list of 64 numbers
def fen_to_vector(fen):

    pass


# Take all of the results files in a directory and merge them into a single file with vectorized fen
def merge_and_vectorize(name, directory):


    results_files = []

    for file in results_files:


    # Use list comprehensions to create a list of every combination of a1 through h8
    letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
    numbers = [str(x + 1) for x in range(8)]
    df_column_names = [a + x for x in numbers for a in letters]

    # Use a list comprehension to get the list of lists for the dataframe
    converted_positions_df = pd.DataFrame([fen_to_vector(fen) for fen in merged_df["Positions"]], columns=df_column_names)

    # Print the merged file
    merged_df.to_csv(f"{name}_merged_for_training.csv", index=False)



# Take in a filepath, remove all forced checkmate rows, save the filtered file, return checkmates dataframe
def remove_checkmates(filepath):
    input_df = pd.read_csv(filepath)

    checkmates_df = input_df[input_df["Score"].str.contains("#")]
    non_checkmates_df = input_df[~input_df["Score"].str.contains("#")]


    # This needs to return a dataframe
    return []


#
def prepare_training_files():
    depth_breaks_dir = ""
    checkmates_dir = ""

    # Merge the depth breaks
    merge_and_vectorize("", "")
    # Remove the checkmates from the merged file and put them in a dataframe
    filtered_checkmates = remove_checkmates("")

    # Print filtered_checkmates to the checkmates_dir
    merge_and_vectorize("", "")


#
def prepare_pretraining_files():
    pass

