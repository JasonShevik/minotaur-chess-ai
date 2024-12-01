import pandas as pd
import sqlite3
from os.path import exists
from typing import List, Callable


# A table for Progress files
# A table for Results files
# Differentiate Checkmate vs Depth Breaks with a new column in Results table

#
def bulk_add_progress() -> None:
    pass


#
def bulk_add_results() -> None:
    pass


# Program Body
if __name__ == "__main__":
    # Initialize database paths and names
    db_path: str = "labeled-data/labeled_chess_data.sqlite"

    # This tool is for initial creation, so check if the dbs exist
    if not os.path.exists(db_path):
        try:
            # Create and connect to the database
            conn: sqlite3.Connection = sqlite3.connect(db_path)

            # Create the tables
            bulk_add_progress()
            bulk_add_results()

            # Close the database
            conn.close()

        except sqlite3.Error as e:
            # Print the error
            print(e)









