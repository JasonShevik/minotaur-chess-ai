import chess.engine
import chess
from typing import Dict, List, Optional, Set, Tuple
import helper_utils as hu

import sys
sys.setrecursionlimit(11000)

def _analyze_position(
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
        attacker: chess.Color,
        *,
        limit_depth: int = 10,
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Analyze a position and return whether it is forced checkmate for the attacker,
    the depth to mate in full moves (if forced), and the best move in UCI.

    :return: (is_forced_mate_for_attacker, depth_to_mate_in_moves, best_move_uci)
    """
    limit = chess.engine.Limit(depth=limit_depth)
    try:
        info = engine.analyse(board, limit)
    except Exception as e:
        print(f"Error analyzing position: {board.fen()}\n Error: {e}")
        return False, None, None

    score_pov = info.get("score")
    if score_pov is None:
        return False, None, None

    score = score_pov.pov(attacker)
    pv = info.get("pv")
    best_move_uci: Optional[str] = str(pv[0]) if pv else None

    if not score.is_mate():
        return False, None, best_move_uci

    mate_plies = score.mate()
    if mate_plies is None or mate_plies < 0:
        return False, None, best_move_uci

    return True, mate_plies, best_move_uci


def _normalize_fen(fen: str) -> str:
    """Return FEN with only the first 4 fields (no halfmove/fullmove) so same position has one key."""
    parts = fen.split()
    return " ".join(parts[:4]) if len(parts) >= 4 else fen


def _ensure_full_fen(fen: str) -> str:
    """Convert a normalized 4-field FEN into a valid 6-field FEN."""
    parts = fen.split()
    if len(parts) == 4:
        return f"{fen} 0 1"
    return fen


def expand_down(
        engine: chess.engine.SimpleEngine,
        position: str,
        path: Optional[Set[str]] = None,
        attacker: Optional[chess.Color] = None,
        verified: Optional[Dict[str, Tuple[int, str]]] = None,
        memo: Optional[Dict[str, Optional[int]]] = None,
) -> Optional[int]:
    """
    Recursively expand the full downward tree of a forced checkmate position.

    Writes only to the shared verified dict (attacker positions). Returns depth on success
    so callers can compute their own depth; no dict merging. Uses path for cycle prevention
    and memo to avoid re-expanding finished positions.

    :param engine: The chess engine to use for analysis.
    :param position: The position (FEN string) to expand down from.
    :param path: Set of FENs on the current path from root (cycle prevention only).
    :param attacker: The side that has the forced checkmate (inferred on first call if None).
    :param verified: Dict to update with attacker positions (position_key -> (depth, best_move)).
    :param memo: Dict of depths for positions we've already finished (depth if success, None if failure).
    :return: Depth to mate from this position on success, or None if not forced mate.
    """
    if path is None: path = set()
    if verified is None: verified = {}
    if memo is None: memo = {}

    global my_count

    my_count += 1
    print(my_count)
    if my_count > 100000:
        return None

    position_key = _normalize_fen(position)

    # 1. Cycle Prevention
    if position_key in path:
        return None

    # 2. Transposition / Cache Check (Handles both sides instantly)
    if position_key in memo:
        return memo[position_key]

    path = set(path)
    path.add(position_key)

    board = chess.Board(position)

    # Infer attacker
    if attacker is None:
        is_mate_white, _, _ = _analyze_position(engine, board, chess.WHITE)
        if is_mate_white:
            attacker = chess.WHITE
        else:
            is_mate_black, _, _ = _analyze_position(engine, board, chess.BLACK)
            if is_mate_black:
                attacker = chess.BLACK
            else:
                memo[position_key] = None
                return None
    assert attacker is not None

    # Engine Oracle Check
    is_forced_mate, mate_plies, best_move_uci = _analyze_position(engine, board, attacker)
    if not is_forced_mate:
        memo[position_key] = None
        return None
    if board.is_checkmate():
        memo[position_key] = 0
        return 0

    is_attacker_turn = board.turn == attacker

    def get_child_depth(child_fen: str) -> Optional[int]:
        """Return cached depth if already explored, else recurse."""
        child_key = _normalize_fen(child_fen)
        if child_key in memo:
            return memo[child_key]
        return expand_down(engine, child_fen, path, attacker, verified, memo)

    if is_attacker_turn:
        if mate_plies == 1 and best_move_uci:
            memo[position_key] = 1
            verified[position_key] = (1, best_move_uci)  # Pure Output Collection
            return 1

        local_candidates: List[Tuple[str, str]] = []
        for move in board.legal_moves:
            board.push(move)
            child_fen = board.fen()
            board.pop()
            if _normalize_fen(child_fen) in path:
                continue
            local_candidates.append((move.uci(), child_fen))

        # Edge Case: All legal moves were loops back into the path
        if not local_candidates and list(board.legal_moves):
            memo[position_key] = None
            return None

        best_depth: Optional[int] = None
        best_move_uci_for_depth: Optional[str] = None

        for move_uci, child_fen in local_candidates:
            child_depth = get_child_depth(child_fen)
            if child_depth is not None:
                our_depth = 1 + child_depth
                if best_depth is None or our_depth < best_depth:
                    best_depth = our_depth
                    best_move_uci_for_depth = move_uci

        if best_depth is None or best_move_uci_for_depth is None:
            memo[position_key] = None
            return None

        memo[position_key] = best_depth
        verified[position_key] = (best_depth, best_move_uci_for_depth)  # Pure Output Collection
        return best_depth

    else:
        # Defender's turn
        local_candidates_defender: List[str] = []
        for move in board.legal_moves:
            board.push(move)
            child_fen = board.fen()
            board.pop()
            if _normalize_fen(child_fen) in path:
                continue
            local_candidates_defender.append(child_fen)

        # Edge Case: All defender moves loop back, so the defender forced a draw
        if not local_candidates_defender and list(board.legal_moves):
            memo[position_key] = None
            return None

        max_child_depth = 0
        for child_fen in local_candidates_defender:
            child_depth = get_child_depth(child_fen)
            if child_depth is None:
                memo[position_key] = None
                return None  # Hidden escape found
            max_child_depth = max(max_child_depth, child_depth)

        depth_defender = 1 + max_child_depth

        # We cache the success in memo for traversal, but DO NOT write to verified!
        memo[position_key] = depth_defender
        return depth_defender


def verify(
        engine: chess.engine.SimpleEngine,
        fens: List[str],
        depth: int = 15,
) -> Tuple[int, int]:
    """
    Verify a list of positions and count how many are forced checkmates.

    Normalized 4-field FENs are expanded to valid 6-field FENs before analysis.
    The first time a position is not a forced mate for the side to move, it is printed.
    """
    checkmates: int = 0
    non_checkmates: int = 0
    printed_first_failure: bool = False

    for fen in fens:
        full_fen = _ensure_full_fen(fen)
        board = chess.Board(full_fen)
        is_forced_mate, mate_plies, best_move_uci = _analyze_position(
            engine,
            board,
            board.turn,
            limit_depth=depth,
        )

        if is_forced_mate:
            checkmates += 1
        else:
            non_checkmates += 1
            if not printed_first_failure:
                print("First non-checkmate found:")
                print(f"  Original FEN: {fen}")
                print(f"  Full FEN: {full_fen}")
                print(f"  Side to move: {'White' if board.turn == chess.WHITE else 'Black'}")
                print(f"  Mate plies: {mate_plies}")
                print(f"  Best move: {best_move_uci}")
                printed_first_failure = True

    return checkmates, non_checkmates


def expand_up() -> None:
    pass


# ##### ##### ##### ##### #####
#       Program body

if __name__ == "__main__":
    verified = {}
    fen = "8/7p/kR2p1p1/5p2/N3RPP1/1P6/1K5P/8 b - - 1 37"
    board = chess.Board(fen)
    print(f"{board}\n\n")

    my_count = 0

    engine = hu.initialize_engine("stockfish", {"Threads": 8, "Hash": 20000})
    expand_down(engine, fen, verified=verified)

    verified_fens = list(verified.keys())
    print(f"Collected {len(verified_fens)} verified attacker positions")

    for fen in verified_fens[:20]:
        print(fen)

    verified_fens = list(verified.keys())
    checkmates, non_checkmates = verify(engine, verified_fens)
    print(f"Verified positions check: {checkmates} checkmates, {non_checkmates} non-checkmates")

    engine.quit()

