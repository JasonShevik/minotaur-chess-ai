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
    with the same columns as the source (from PRAGMA table_info), so any columns present
    in the source are copied without code changes.
    """
    with sqlite3.connect(f"{source_db}.db") as src_conn:
        src_cursor = src_conn.cursor()
        src_cursor.execute('PRAGMA table_info("960_position_data")')
        table_info = src_cursor.fetchall()
    # PRAGMA table_info: cid, name, type, notnull, dflt_value, pk
    col_names = [row[1] for row in table_info]
    col_specs: List[str] = []
    for row in table_info:
        cid, name, type_, notnull, dflt_value, pk = row[0], row[1], row[2], row[3], row[4], row[5]
        spec = f'"{name}" {type_ or "TEXT"}'
        if notnull:
            spec += " NOT NULL"
        if dflt_value is not None:
            spec += f" DEFAULT {dflt_value}"
        if pk:
            spec += " PRIMARY KEY"
        col_specs.append(spec)
    create_sql = 'CREATE TABLE IF NOT EXISTS "960_position_data" (' + ", ".join(col_specs) + ")"

    with sqlite3.connect(f"{target_db}.db") as target_conn:
        target_cursor: sqlite3.Cursor = target_conn.cursor()
        target_cursor.execute(create_sql)
        target_cursor.execute(f'ATTACH DATABASE "{source_db}.db" AS source')
        cols = ", ".join(f'"{c}"' for c in col_names)
        target_cursor.execute(f'''
            INSERT OR IGNORE INTO "960_position_data" ({cols})
            SELECT {cols} FROM source."960_position_data"
            WHERE is_analyzed = 1
        ''')
        target_conn.commit()
        target_cursor.execute('DETACH DATABASE source')
        target_cursor.execute('CREATE INDEX IF NOT EXISTS idx_is_analyzed ON "960_position_data" (is_analyzed)')
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


def backfill_castling_rook_column(
    db_name: str, batch_size: int = 1000, enforce_symmetry: bool = True
) -> None:
    """
    Add a 'castling_rook_squares' column to the 960_position_data table if it does not exist.
    Then, for every row where that column is NULL or empty, compute the castling rook square
    for each allowed direction (K, Q, k, q): one rook on that side -> use it; multiple -> pick
    one at random. Store as a comma-separated list of 4 integers (K, Q, k, q order; -1 for none).

    If enforce_symmetry is True, K and k must share the same file, and Q and q must share the
    same file. When both colors can castle to a side but have no rooks on a common file, that
    side's rook slots are set to -1 and an 'inconsistent' column is set to 1. An 'inconsistent'
    column is created if absent. Inconsistent FENs are printed at the end.
    """
    conn = sqlite3.connect(f"{db_name}.db")
    cursor = conn.cursor()

    cursor.execute('PRAGMA table_info("960_position_data")')
    columns = [row[1] for row in cursor.fetchall()]
    if "castling_rook_squares" not in columns:
        cursor.execute('ALTER TABLE "960_position_data" ADD COLUMN castling_rook_squares TEXT')
        conn.commit()
        print("Added column castling_rook_squares.")
    if enforce_symmetry and "inconsistent" not in columns:
        cursor.execute(
            'ALTER TABLE "960_position_data" ADD COLUMN inconsistent INTEGER NOT NULL DEFAULT 0'
        )
        conn.commit()
        print("Added column inconsistent.")

    last_rowid = 0
    total_processed = 0
    ambiguous_fens: List[str] = []
    print(
        "Backfilling castling_rook_squares (batch size %d, enforce_symmetry=%s)..."
        % (batch_size, enforce_symmetry)
    )
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
            if enforce_symmetry:
                squares, had_ambiguous, inconsistent = _castling_rook_squares_from_fen_symmetric(fen)
                if had_ambiguous:
                    ambiguous_fens.append(fen)
                value = _castling_rook_squares_to_db_value(squares)
                cursor.execute(
                    'UPDATE "960_position_data" SET castling_rook_squares = ?, inconsistent = ? WHERE fen = ?',
                    (value, 1 if inconsistent else 0, fen),
                )
            else:
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
        print(
            "FENs with two or more rooks on one side (for a castling direction): %d"
            % len(ambiguous_fens)
        )
        for fen in ambiguous_fens:
            print("  %s" % fen)
    else:
        print("No FENs had two rooks on one side for any castling direction.")

    if enforce_symmetry:
        with sqlite3.connect(f"{db_name}.db") as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT fen FROM "960_position_data" WHERE inconsistent = 1'
            )
            inconsistent_rows = cursor.fetchall()
        if inconsistent_rows:
            print("FENs with inconsistent castling (no shared file for K/k or Q/q): %d" % len(inconsistent_rows))
            for (fen,) in inconsistent_rows:
                print("  %s" % fen)
        else:
            print("No FENs had inconsistent castling.")


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


def _castling_rook_squares_from_fen_symmetric(
    fen: str, rng: Optional[random.Random] = None
) -> Tuple[List[int], bool, bool]:
    """
    Like _castling_rook_squares_from_fen but enforces symmetry: K and k must be on the same
    file, and Q and q must be on the same file. Returns (squares, had_ambiguous, inconsistent).
    When both colors can castle to a side but have no rooks on a common file, that side's
    slots are set to -1 and inconsistent is True.
    """
    rand = rng if rng is not None else random
    had_ambiguous = False
    inconsistent = False
    parts = fen.split()
    if len(parts) < 3:
        return ([-1, -1, -1, -1], False, False)
    castling_str = parts[2]
    if castling_str == "-":
        return ([-1, -1, -1, -1], False, False)
    allowed = set(c for c in castling_str if c in "KQkq")
    if not allowed:
        return ([-1, -1, -1, -1], False, False)
    try:
        board = chess.Board(fen)
    except (ValueError, TypeError):
        return ([-1, -1, -1, -1], False, False)

    result: List[int] = [-1, -1, -1, -1]

    def rooks_on_side(color: bool, kingside: bool) -> List[int]:
        rank = 0 if color == chess.WHITE else 7
        king_sq = board.king(color)
        if king_sq is None or chess.square_rank(king_sq) != rank:
            return []
        kf = chess.square_file(king_sq)
        rooks = [
            s for s in chess.SQUARES
            if chess.square_rank(s) == rank
            and board.piece_at(s) == chess.Piece(chess.ROOK, color)
        ]
        return [
            s for s in rooks
            if (chess.square_file(s) > kf if kingside else chess.square_file(s) < kf)
        ]

    # Kingside: K (idx 0) and k (idx 2)
    if "K" in allowed and "k" in allowed:
        w_rooks = rooks_on_side(chess.WHITE, True)
        b_rooks = rooks_on_side(chess.BLACK, True)
        w_files = {chess.square_file(s) for s in w_rooks}
        b_files = {chess.square_file(s) for s in b_rooks}
        common = w_files & b_files
        if not common:
            inconsistent = True
            # leave result[0] and result[2] as -1
        else:
            chosen_file = rand.choice(list(common))
            ws = [s for s in w_rooks if chess.square_file(s) == chosen_file]
            bs = [s for s in b_rooks if chess.square_file(s) == chosen_file]
            if ws and bs:
                result[0] = ws[0]
                result[2] = bs[0]
            if len(w_rooks) > 1 or len(b_rooks) > 1:
                had_ambiguous = True
    elif "K" in allowed:
        candidates = rooks_on_side(chess.WHITE, True)
        if candidates:
            if len(candidates) > 1:
                had_ambiguous = True
            result[0] = rand.choice(candidates)
    elif "k" in allowed:
        candidates = rooks_on_side(chess.BLACK, True)
        if candidates:
            if len(candidates) > 1:
                had_ambiguous = True
            result[2] = rand.choice(candidates)

    # Queenside: Q (idx 1) and q (idx 3)
    if "Q" in allowed and "q" in allowed:
        w_rooks = rooks_on_side(chess.WHITE, False)
        b_rooks = rooks_on_side(chess.BLACK, False)
        w_files = {chess.square_file(s) for s in w_rooks}
        b_files = {chess.square_file(s) for s in b_rooks}
        common = w_files & b_files
        if not common:
            inconsistent = True
        else:
            chosen_file = rand.choice(list(common))
            ws = [s for s in w_rooks if chess.square_file(s) == chosen_file]
            bs = [s for s in b_rooks if chess.square_file(s) == chosen_file]
            if ws and bs:
                result[1] = ws[0]
                result[3] = bs[0]
            if len(w_rooks) > 1 or len(b_rooks) > 1:
                had_ambiguous = True
    elif "Q" in allowed:
        candidates = rooks_on_side(chess.WHITE, False)
        if candidates:
            if len(candidates) > 1:
                had_ambiguous = True
            result[1] = rand.choice(candidates)
    elif "q" in allowed:
        candidates = rooks_on_side(chess.BLACK, False)
        if candidates:
            if len(candidates) > 1:
                had_ambiguous = True
            result[3] = rand.choice(candidates)

    return (result, had_ambiguous, inconsistent)


def print_head(db_name: str, n: int = 5) -> None:
    with sqlite3.connect(f"{db_name}.db") as conn:
        cursor = conn.cursor()
        cursor.execute('PRAGMA table_info("960_position_data")')
        columns = [row[1] for row in cursor.fetchall()]

        cursor.execute(f'''
            SELECT * FROM "960_position_data"
            LIMIT {n}
        ''')
        rows = cursor.fetchall()

        print(" | ".join(columns))
        print("-" * 80)
        for row in rows:
            print(" | ".join(str(x) for x in row))


# Program Body
if __name__ == "__main__":
    name = "minotaur_data"

    #check_db(name)
    #print_checkmates_breakdown(name)

    #print("\nExporting...")
    #export_analyzed_positions(name, "minotaur_analyzed")

    #backfill_castling_rook_column("minotaur_data")
    
    print_head(name)


"""
FENs with two or more rooks on one side (for a castling direction): 13
  rnr1k1nq/pp6/2p3p1/3P4/1b1PQ3/1PN1P3/P2N1P1P/R3KR2 b KQq - 0 15
  rn3rk1/pR4p1/8/8/2Pp2PB/8/P1nN2P1/R5K1 w q - 1 28
  rn1rk2q/ppp3pp/5n2/4P3/8/5N2/PPP2PPP/R2B1RKQ b q - 0 12
  2r4k/p1N2p1p/1p4pn/8/2P5/1r6/5PP1/2R1R1K1 w Q - 5 30
  rb2krq1/pp2p1pp/2p3n1/5bn1/3P4/2N2pN1/PPP2BPP/RB1RK1Q1 b Qkq - 1 12
  1b1rk3/2rpq2p/1p3p2/3QpPpP/6P1/1NP5/P1P2P2/1BRR1K2 b Q - 4 21
  2kr2rb/pbppq2p/1p1nn1p1/8/P2PP3/1P3Pp1/1B4BP/1KNQRNR1 w K - 0 16
  1r3rk1/p3pp1p/3p1bp1/2p5/P3nP2/1PPNP3/7P/2K1R2R w K - 0 21
  2r2rk1/3pppp1/2b2n1q/p1n1N2p/1pP4P/1P4PQ/PB2P3/1BRRK3 b Q - 3 16
  7r/4kbb1/8/2PPp1p1/2n1P1p1/3N4/P1P3PP/2K2RBR w K - 3 24
  b1rk1rq1/p2p1ppp/1p1b1n2/2pP4/2P2n2/1P1N4/P4PPP/BB1KRRQN b Kkq - 8 10
  1r2q1kr/ppp2pp1/1n3b1p/3p4/6QP/4P1N1/PPPP2P1/1R2BRK1 b Qkq - 5 18
  1kr1r2n/1b3pQ1/4pPpP/1q1p4/p1pP1PP1/R4B2/KPPB4/1R6 w k - 3 27
"""
