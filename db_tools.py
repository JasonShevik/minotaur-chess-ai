import pandas as pd
import sqlite3
from os.path import exists
from typing import List, Callable


def create_db(name: str) -> None:
    # Connect to the database
    conn = sqlite3.connect(f"{name}.db")
    cursor = conn.cursor()

    # Create table with minor improvements
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "960_position_data" (
        fen TEXT PRIMARY KEY,
        is_analyzed INTEGER NOT NULL DEFAULT 0 CHECK(is_analyzed IN (0, 1)),
        engine_name TEXT,
        depth INTEGER,
        score TEXT,
        is_forced_checkmate INTEGER NOT NULL DEFAULT 0 CHECK(is_forced_checkmate IN (0, 1)),
        best_move TEXT
        )
    ''')

    # Commit and close
    conn.commit()
    conn.close()


def add_positions(name: str, filepaths: List[str]) -> None:
    # Connect to the database
    conn = sqlite3.connect(f"{name}.db")
    cursor = conn.cursor()

    filepath: str
    for filepath in filepaths:
        print(f"Adding from {filepath}")
        with open(filepath, 'r') as f:
            fen_list = [(line.strip(),) for line in f]

        cursor.executemany('''
                INSERT INTO "960_position_data" (fen, is_analyzed)
                VALUES (?, 0)
                ON CONFLICT(fen) DO NOTHING
            ''', fen_list)
        conn.commit()
    conn.close()


def add_historical_labels(name: str, filepaths: List[str]) -> None:
    # Connect to the database
    conn = sqlite3.connect(f"{name}.db")
    cursor = conn.cursor()

    BATCH_SIZE = 1000  # Commit after a fixed number of updates
    current_batch = 0

    filepath: str
    for filepath in filepaths:
        print(f"Labeling from {filepath}")
        df = pd.read_csv(filepath)

        for index, (_, row) in enumerate(df.iterrows(), start=1):
            # Identify what engine version I was using based on past commits
            engine_name: str = ""
            if "results_part_1_stockfish" in filepath:
                if index < 3337:
                    engine_name = "Stockfish 16"
                elif index < 8398:
                    engine_name = "Stockfish 16.1"
                else:
                    engine_name = "Stockfish 17"
            elif "results_part_2_leela" in filepath:
                if index < 9125:
                    engine_name = "Leela 0.30.0"
                else:
                    engine_name = "Leela 0.31.1"
            elif "results_part_3_stockfish" in filepath:
                engine_name = "Stockfish 16.1"
            elif "results_part_5_stockfish" in filepath:
                engine_name = "Stockfish 16.1"

            fen = row["Position"]
            is_analyzed = 1
            depth = 25 if "results_part_5_stockfish" in filepath else row["Depth"]
            score = str(row["Score"])
            is_forced_checkmate = 1 if "#" in score else 0
            best_move = str(row["Move"])

            # Update the database
            cursor.execute('''
                            UPDATE "960_position_data" 
                            SET is_analyzed = ?, 
                                engine_name = ?, 
                                depth = ?, 
                                score = ?, 
                                is_forced_checkmate = ?, 
                                best_move = ?
                            WHERE fen = ?
                            ''', (is_analyzed, engine_name, depth, score, is_forced_checkmate, best_move, fen))

            current_batch += 1
            if current_batch >= BATCH_SIZE:
                conn.commit()
                current_batch = 0

    # Final commit for any remaining updates
    conn.commit()

    # Rebuild the index
    cursor.execute('DROP INDEX IF EXISTS idx_is_analyzed')
    cursor.execute('CREATE INDEX idx_is_analyzed ON "960_position_data" (is_analyzed)')
    conn.commit()
    conn.close()


def check_db(name: str) -> None:
    # Connect to the database
    conn = sqlite3.connect(f"{name}.db")
    cursor = conn.cursor()

    print("=== Random Analyzed Positions ===")
    cursor.execute('''
        SELECT fen, engine_name, depth, score, best_move 
        FROM "960_position_data" 
        WHERE is_analyzed = 1 
        ORDER BY RANDOM() 
        LIMIT 5
    ''')
    for row in cursor.fetchall():
        print(f"FEN: {row[0]}")
        print(f"Engine: {row[1]}, Depth: {row[2]}, Score: {row[3]}, Best Move: {row[4]}\n")

    print("=== Random Unanalyzed Positions ===")
    cursor.execute('''
        SELECT fen 
        FROM "960_position_data" 
        WHERE is_analyzed = 0 
        ORDER BY RANDOM() 
        LIMIT 5
    ''')
    for row in cursor.fetchall():
        print(f"FEN: {row[0]}\n")

    print("=== Analysis Statistics ===")
    cursor.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(is_analyzed) as analyzed,
            SUM(is_forced_checkmate) as checkmates
        FROM "960_position_data"
    ''')
    stats = cursor.fetchone()
    print(f"Total positions: {stats[0]}")
    print(f"Analyzed positions: {stats[1]} ({(stats[1]/stats[0]*100):.1f}%)")
    print(f"Forced checkmates found: {stats[2]}")

    print("\n=== Analysis by Engine ===")
    cursor.execute('''
        SELECT engine_name, COUNT(*) as count
        FROM "960_position_data"
        WHERE is_analyzed = 1
        GROUP BY engine_name
        ORDER BY count DESC
    ''')
    for engine, count in cursor.fetchall():
        print(f"{engine}: {count} positions")

    print("\n=== Analysis by Depth ===")
    cursor.execute('''
        SELECT depth, COUNT(*) as count
        FROM "960_position_data"
        WHERE is_analyzed = 1
        GROUP BY depth
        ORDER BY depth DESC
    ''')
    for depth, count in cursor.fetchall():
        print(f"Depth {depth}: {count} positions")

    conn.close()


def export_analyzed_positions(source_db: str, target_db: str) -> None:
    # Connect to the target database
    target_conn = sqlite3.connect(f"{target_db}.db")
    target_cursor = target_conn.cursor()

    # Create the same table structure in the target database
    target_cursor.execute('''
        CREATE TABLE IF NOT EXISTS "960_position_data" (
        fen TEXT PRIMARY KEY,
        is_analyzed INTEGER NOT NULL DEFAULT 0 CHECK(is_analyzed IN (0, 1)),
        engine_name TEXT,
        depth INTEGER,
        score TEXT,
        is_forced_checkmate INTEGER NOT NULL DEFAULT 0 CHECK(is_forced_checkmate IN (0, 1)),
        best_move TEXT
        )
    ''')

    # Attach the source database
    target_cursor.execute(f'ATTACH DATABASE "{source_db}.db" AS source')

    # Copy all analyzed positions
    target_cursor.execute('''
        INSERT INTO "960_position_data"
        SELECT * FROM source."960_position_data"
        WHERE is_analyzed = 1
    ''')

    # Detach the source database
    target_conn.commit()
    target_cursor.execute('DETACH DATABASE source')

    # Create the same index on the new database
    target_cursor.execute('CREATE INDEX idx_is_analyzed ON "960_position_data" (is_analyzed)')

    # Commit changes and close connection
    target_conn.commit()
    target_conn.close()


# Program Body
if __name__ == "__main__":
    # Old: Used to create the database and add labels
    #db_name = "minotaur_data"
    #print("Creating...")
    #create_db(db_name)
    #position_files = ["lichess-positions/lichess_positions_part_1.txt",
    #                  "lichess-positions/lichess_positions_part_2.txt",
    #                  "lichess-positions/lichess_positions_part_3.txt",
    #                  "lichess-positions/lichess_positions_part_4.txt",
    #                  "lichess-positions/lichess_positions_part_5.txt"]
    #add_positions(db_name, position_files)
    #labeled_files = ["training-supervised-engines/results_part_1_stockfish.csv",
    #                 "training-supervised-engines/results_part_2_leela.csv",
    #                 "training-supervised-engines/results_part_3_stockfish.csv",
    #                 "training-supervised-checkmates/results_part_5_stockfish.csv"]
    #add_historical_labels(db_name, labeled_files)
    #print("Checking...\n")
    #check_db(db_name)
    #print("\nExporting...")
    #export_analyzed_positions(db_name, "minotaur_analyzed")
    #print("Checking...\n")
    #check_db("minotaur_analyzed")






