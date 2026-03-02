import chess.engine
import chess
from typing import Dict, List, Optional, Set, Tuple


def _analyze_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    attacker: chess.Color,
    *,
    limit_depth: int = 30,
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
    if mate_plies is None or mate_plies <= 0:
        return False, None, best_move_uci
    # Plies to full moves: e.g. mate in 2 plies = 1 full move
    depth_moves = (mate_plies + 1) // 2
    return True, depth_moves, best_move_uci


def _normalize_fen(fen: str) -> str:
    """Return FEN with only the first 4 fields (no halfmove/fullmove) so same position has one key."""
    parts = fen.split()
    return " ".join(parts[:4]) if len(parts) >= 4 else fen


def expand_down(
    engine: chess.engine.SimpleEngine,
    position: str,
    path: Optional[Set[str]] = None,
    attacker: Optional[chess.Color] = None,
    verified: Optional[Dict[str, Tuple[int, str]]] = None,
    explored: Optional[Set[str]] = None,
) -> Optional[Dict[str, Tuple[int, str]]]:
    """
    Recursively expand the full downward tree of a forced checkmate position, caching all verified forced checkmates for the attacker.

    Uses path only for cycle prevention. Explored tracks all positions we've finished
    (success or failure); verified caches forced-mate results. Before recursing into
    a child we check explored and use cached result if present.

    :param engine: The chess engine to use for analysis.
    :param position: The position (FEN string) to expand down from.
    :param path: Set of FENs on the current path from root (cycle prevention only).
    :param attacker: The side that has the forced checkmate (inferred on first call if None).
    :param verified: Dict of positions verified as forced mate (position -> (depth, best_move)).
    :param explored: Set of positions we've already finished (success or failure); avoid re-expanding.
    :return: A local dict of positions verified in this subtree, or None if this branch is not forced mate.
    """
    if path is None:
        path = set()
    if verified is None:
        verified = {}
    if explored is None:
        explored = set()

    position_key = _normalize_fen(position)

    # Cycle: re-entering a position on the current path
    if position_key in path:
        return None
    path = set(path)
    path.add(position_key)

    board = chess.Board(position)

    # Already verified (transposition): return cached success
    if position_key in verified:
        return {position_key: verified[position_key]}

    # Already explored and not verified: previous result was failure
    if position_key in explored:
        return None

    # Infer attacker from first analysis if not provided
    if attacker is None:
        is_mate_white, _, _ = _analyze_position(engine, board, chess.WHITE)
        if is_mate_white:
            attacker = chess.WHITE
        else:
            is_mate_black, _, _ = _analyze_position(engine, board, chess.BLACK)
            if is_mate_black:
                attacker = chess.BLACK
            else:
                explored.add(position_key)
                return None
    assert attacker is not None

    is_forced_mate, depth_moves, best_move_uci = _analyze_position(engine, board, attacker)
    if not is_forced_mate:
        explored.add(position_key)
        return None

    local_verified: Dict[str, Tuple[int, str]] = {}
    is_attacker_turn = board.turn == attacker

    def get_child_result(child_fen: str) -> Optional[Dict[str, Tuple[int, str]]]:
        """Return cached result if child already explored, else recurse."""
        child_key = _normalize_fen(child_fen)
        if child_key in explored:
            if child_key in verified:
                return {child_key: verified[child_key]}
            return None
        return expand_down(engine, child_fen, path, attacker, verified, explored)

    if is_attacker_turn:
        # Checkmate in one: we deliver mate with one move; don't recurse into checkmated position.
        if depth_moves == 1 and best_move_uci:
            assert board.turn == attacker
            entry = (1, best_move_uci)
            verified[position_key] = entry
            local_verified[position_key] = entry
            explored.add(position_key)
            return local_verified

        # Get all legal moves that are also forced checkmates; only skip children on path (cycle).
        local_candidates: List[Tuple[str, str]] = []  # (move_uci, fen_after_move)
        for move in board.legal_moves:
            board.push(move)
            child_fen = board.fen()
            board.pop()
            if _normalize_fen(child_fen) in path:
                continue
            child_board = chess.Board(child_fen)
            child_mate, _, _ = _analyze_position(engine, child_board, attacker)
            if child_mate:
                local_candidates.append((move.uci(), child_fen))

        for move_uci, child_fen in local_candidates:
            child_result = get_child_result(child_fen)
            if child_result is not None:
                local_verified.update(child_result)
                child_key = _normalize_fen(child_fen)
                child_depth = child_result.get(child_key, (0, ""))[0]
                existing = local_verified.get(position_key)
                if existing is None or (1 + child_depth) < existing[0]:
                    assert board.turn == attacker
                    entry = (1 + child_depth, move_uci)
                    verified[position_key] = entry
                    local_verified[position_key] = entry

        if position_key not in local_verified:
            explored.add(position_key)
            return None
        explored.add(position_key)
        return local_verified

    else:
        # Defender's turn: every legal move must lead to forced checkmate; only skip children on path (cycle).
        local_verified_defender: Dict[str, Tuple[int, str]] = {}
        local_candidates_defender: List[str] = []
        for move in board.legal_moves:
            board.push(move)
            child_fen = board.fen()
            board.pop()
            if _normalize_fen(child_fen) in path:
                continue
            child_board = chess.Board(child_fen)
            child_mate, _, _ = _analyze_position(engine, child_board, attacker)
            if not child_mate:
                explored.add(position_key)
                return None  # Defender has an escape
            local_candidates_defender.append(child_fen)

        max_child_depth = 0
        for child_fen in local_candidates_defender:
            child_result = get_child_result(child_fen)
            if child_result is None:
                explored.add(position_key)
                return None  # Hidden escape in this line
            local_verified_defender.update(child_result)
            child_key = _normalize_fen(child_fen)
            child_depth = child_result.get(child_key, (0, ""))[0]
            max_child_depth = max(max_child_depth, child_depth)
        # Defender chooses the move that delays mate longest, so our depth = 1 + max(children).
        depth_defender = 1 + max_child_depth
        local_verified_defender[position_key] = (depth_defender, "")
        explored.add(position_key)
        return local_verified_defender


def expand_up() -> None:
    pass


# ##### ##### ##### ##### #####
#       Program body

if __name__ == "__main__":
    pass

