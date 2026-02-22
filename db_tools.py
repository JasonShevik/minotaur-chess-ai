import pandas as pd
import sqlite3
import random
import chess
from os.path import exists
from typing import List, Tuple, Callable, Optional


def create_db(db_name: str) -> None:
    # Connect to the database
    conn: sqlite3.Connection = sqlite3.connect(f"{db_name}.db")
    cursor: sqlite3.Cursor = conn.cursor()

    # Create table
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


def add_positions(db_name: str, filepaths: List[str]) -> None:
    # Connect to the database
    conn: sqlite3.Connection = sqlite3.connect(f"{db_name}.db")
    cursor: sqlite3.Cursor = conn.cursor()

    # Only fill castling_rook_squares if the column already exists (avoid piecemeal: some rows
    # with value, rest empty). To add the column and backfill everything, use backfill_castling_rook_column.
    cursor.execute('PRAGMA table_info("960_position_data")')
    columns = [row[1] for row in cursor.fetchall()]
    has_castling_column = "castling_rook_squares" in columns

    filepath: str
    for filepath in filepaths:
        print(f"Adding from {filepath}")
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f]
            lines = [fen for fen in lines if fen]
        if has_castling_column:
            fen_list = [
                (fen, 0, _castling_rook_squares_to_db_value(_castling_rook_squares_from_fen(fen)[0]))
                for fen in lines
            ]
            cursor.executemany('''
                    INSERT INTO "960_position_data" (fen, is_analyzed, castling_rook_squares)
                    VALUES (?, ?, ?)
                    ON CONFLICT(fen) DO NOTHING
            ''', fen_list)
        else:
            fen_list = [(fen, 0) for fen in lines]
            cursor.executemany('''
                    INSERT INTO "960_position_data" (fen, is_analyzed)
                    VALUES (?, ?)
                    ON CONFLICT(fen) DO NOTHING
            ''', fen_list)
        conn.commit()
    conn.close()


def add_historical_labels(db_name: str, filepaths: List[str]) -> None:
    # Connect to the database
    conn: sqlite3.Connection = sqlite3.connect(f"{db_name}.db")
    cursor: sqlite3.Cursor = conn.cursor()

    BATCH_SIZE: int = 1000  # Commit after a fixed number of updates
    current_batch: int = 0

    filepath: str
    for filepath in filepaths:
        print(f"Labeling from {filepath}")
        df: pd.DataFrame = pd.read_csv(filepath)

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

            fen: str = row["Position"]
            is_analyzed: int = 1
            depth: int = 25 if "results_part_5_stockfish" in filepath else row["Depth"]
            score: str = str(row["Score"])
            is_forced_checkmate: int = 1 if "#" in score else 0
            best_move: str = str(row["Move"])

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


def check_db(db_name: str) -> None:
    # Connect to the database
    conn: sqlite3.Connection = sqlite3.connect(f"{db_name}.db")
    cursor: sqlite3.Cursor = conn.cursor()

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


def print_checkmates_breakdown(db_name: str) -> None:
    # Connect to the database
    with sqlite3.connect(f"{db_name}.db") as conn:
        cursor: sqlite3.Cursor = conn.cursor()

        cursor.execute('''
            SELECT fen, engine_name, depth, score, best_move 
            FROM "960_position_data" 
            WHERE (is_forced_checkmate = 1 AND depth = 25)
            ORDER BY RANDOM() 
            LIMIT 5
        ''')
        for row in cursor.fetchall():
            print(f"FEN: {row[0]}")
            print(f"Engine: {row[1]}, Depth: {row[2]}, Score: {row[3]}, Best Move: {row[4]}\n")

        print("=== Analysis Statistics ===")
        cursor.execute('''
            SELECT 
                SUM(is_forced_checkmate) as checkmates
            FROM "960_position_data"
        ''')
        stats = cursor.fetchone()
        print(f"Total forced checkmates: {stats[0]}")

        print("\n=== Analysis by Engine ===")
        cursor.execute('''
            SELECT engine_name, COUNT(*) as count
            FROM "960_position_data"
            WHERE is_forced_checkmate = 1
            GROUP BY engine_name
            ORDER BY count DESC
        ''')
        for engine, count in cursor.fetchall():
            print(f"{engine}: {count} positions")

        print("\n=== Analysis by Depth ===")
        cursor.execute('''
            SELECT depth, COUNT(*) as count
            FROM "960_position_data"
            WHERE is_forced_checkmate = 1
            GROUP BY depth
            ORDER BY depth DESC
        ''')
        for depth, count in cursor.fetchall():
            print(f"Depth {depth}: {count} positions")


def export_analyzed_positions(source_db: str, target_db: str) -> None:
    """
    Copy all analyzed positions from source_db to target_db. The target table is created
    with the same columns as the source (including castling_rook_squares if present).
    """
    # Check source schema for optional castling_rook_squares column
    with sqlite3.connect(f"{source_db}.db") as src_conn:
        src_cursor = src_conn.cursor()
        src_cursor.execute('PRAGMA table_info("960_position_data")')
        source_columns = [row[1] for row in src_cursor.fetchall()]
    has_castling_column = "castling_rook_squares" in source_columns

    # Connect to the target database
    with sqlite3.connect(f"{target_db}.db") as target_conn:
        target_cursor: sqlite3.Cursor = target_conn.cursor()

        # Create table structure matching source (with or without castling_rook_squares)
        if has_castling_column:
            target_cursor.execute('''
                CREATE TABLE IF NOT EXISTS "960_position_data" (
                fen TEXT PRIMARY KEY,
                is_analyzed INTEGER NOT NULL DEFAULT 0 CHECK(is_analyzed IN (0, 1)),
                engine_name TEXT,
                depth INTEGER,
                score TEXT,
                is_forced_checkmate INTEGER NOT NULL DEFAULT 0 CHECK(is_forced_checkmate IN (0, 1)),
                best_move TEXT,
                castling_rook_squares TEXT
                )
            ''')
        else:
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

        # Copy all analyzed positions (SELECT * so column set matches source/target)
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


def get_slices(db_name: str, num_slices: int) -> List[List[str]]:
    """

    :param db_name:
    :param num_slices:
    :return:
    """
    # Connect to the database
    with sqlite3.connect(f"{db_name}.db") as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute('''
            SELECT fen FROM "960_position_data"
            WHERE is_analyzed = 0
        ''')
        fen_list: List[str] = [fen[0] for fen in cursor.fetchall()]

    random.shuffle(fen_list)
    slice_length = len(fen_list) // num_slices

    return [fen_list[ i * slice_length : (i+1) * slice_length] for i in range(num_slices)]


def backfill_castling_rook_column(db_name: str, batch_size: int = 1000) -> None:
    """
    Add a 'castling_rook_squares' column to the 960_position_data table if it does not exist.
    Then, for every row where that column is NULL or empty, compute the castling rook square
    for each allowed direction (K, Q, k, q): one rook on that side -> use it; multiple -> pick
    one at random. Store as a comma-separated list of 4 integers (K, Q, k, q order; -1 for none).
    """

    conn = sqlite3.connect(f"{db_name}.db")
    cursor = conn.cursor()

    # Check if column exists
    cursor.execute('PRAGMA table_info("960_position_data")')
    columns = [row[1] for row in cursor.fetchall()]
    if "castling_rook_squares" not in columns:
        cursor.execute('ALTER TABLE "960_position_data" ADD COLUMN castling_rook_squares TEXT')
        conn.commit()
        print("Added column castling_rook_squares.")

    # Process in batches by rowid so we never load more than batch_size rows at once
    # (avoids blowing memory on hundreds of millions of rows)
    last_rowid = 0
    total_processed = 0
    ambiguous_fens: List[str] = []
    print("Backfilling castling_rook_squares (batch size %d)..." % batch_size)
    while True:
        cursor.execute('''
            SELECT fen, rowid FROM "960_position_data"
            WHERE (castling_rook_squares IS NULL OR castling_rook_squares = '')
            AND rowid > ?
            ORDER BY rowid
            LIMIT ?
        ''', (last_rowid, batch_size))
        rows = cursor.fetchall()
        if not rows:
            break
        for (fen, rowid) in rows:
            squares, had_ambiguous = _castling_rook_squares_from_fen(fen)
            if had_ambiguous:
                ambiguous_fens.append(fen)
            value = _castling_rook_squares_to_db_value(squares)
            cursor.execute(
                'UPDATE "960_position_data" SET castling_rook_squares = ? WHERE fen = ?',
                (value, fen),
            )
        conn.commit()
        last_rowid = rows[-1][1]
        total_processed += len(rows)
        if total_processed % 50000 == 0 or len(rows) < batch_size:
            print(f"  {total_processed} rows done.")
        if len(rows) < batch_size:
            break

    conn.close()
    print("Backfill complete. Total rows processed: %d" % total_processed)
    if ambiguous_fens:
        print("FENs with two or more rooks on one side (for a castling direction): %d" % len(ambiguous_fens))
        for fen in ambiguous_fens:
            print("  %s" % fen)
    else:
        print("No FENs had two rooks on one side for any castling direction.")


def _castling_rook_squares_from_fen(fen: str, rng: Optional[random.Random] = None) -> Tuple[List[int], bool]:
    """
    For a given FEN, return (squares, had_ambiguous) where squares is a list of 4 square
    indices (0-63) in order K, Q, k, q (-1 for none). had_ambiguous is True if for any
    castling direction there were two or more rooks on that side of the king.
    When multiple rooks exist on one side, pick one at random.
    """
    rand = rng if rng is not None else random
    had_ambiguous = False
    parts = fen.split()

    # If the FEN is invalid, return -1 for all castling squares
    if len(parts) < 3:
        return ([-1, -1, -1, -1], False)
    castling_str = parts[2]
    if castling_str == "-":
        return ([-1, -1, -1, -1], False)
    allowed = set(c for c in castling_str if c in "KQkq")
    if not allowed:
        return ([-1, -1, -1, -1], False)
    try:
        board = chess.Board(fen)
    except (ValueError, TypeError):
        return ([-1, -1, -1, -1], False)

    result: List[int] = [-1, -1, -1, -1]
    # Order: K (white kingside), Q (white queenside), k (black kingside), q (black queenside)
    keys = ["K", "Q", "k", "q"]
    for idx, key in enumerate(keys):
        if key not in allowed:
            continue
        color = chess.WHITE if key in "KQ" else chess.BLACK
        kingside = key in "Kk"
        rank = 0 if color == chess.WHITE else 7
        king_sq = board.king(color)
        if king_sq is None or chess.square_rank(king_sq) != rank:
            continue
        kf = chess.square_file(king_sq)
        rooks_on_rank = [
            s for s in chess.SQUARES
            if chess.square_rank(s) == rank
            and board.piece_at(s) == chess.Piece(chess.ROOK, color)
        ]
        candidates = [
            s for s in rooks_on_rank
            if (chess.square_file(s) > kf if kingside else chess.square_file(s) < kf)
        ]
        if not candidates:
            continue
        if len(candidates) > 1:
            had_ambiguous = True
        chosen = rand.choice(candidates)
        result[idx] = chosen
    return (result, had_ambiguous)


def _castling_rook_squares_to_db_value(squares: List[int]) -> str:
    """Serialize the 4-tuple of square indices for DB storage (comma-separated, -1 for none)."""
    return ",".join(str(s) for s in squares)


# Program Body
if __name__ == "__main__":
    name = "minotaur_data"

    #check_db(name)
    #print_checkmates_breakdown(name)

    #print("\nExporting...")
    #export_analyzed_positions(name, "minotaur_analyzed")

    backfill_castling_rook_column("minotaur_data")


