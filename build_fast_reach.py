"""
Builds fast_reach.c, the compiled search behind piece_reach.py, into the library piece_reach.py loads
(fast_reach.dll on Windows, fast_reach.so elsewhere, next to this file). It then checks the library against
the Python search on a few positions and times both.

    python build_fast_reach.py              build, check, time
    python build_fast_reach.py --no-check   build only

It needs a C compiler: gcc, clang or cc on the PATH, or the one named in the CC environment variable. (On
Windows, gcc comes with MinGW-w64; Strawberry Perl ships one.) The library is made on the machine that uses
it and is not kept in git: build it once on every machine that labels, and again whenever fast_reach.c
changes (piece_reach.py refuses a library built from another version of it). Without the library everything
still works, on the Python search, some hundreds of times slower.

Stop the labeller before rebuilding: Windows does not let a library that is in use be replaced.

The thorough check is tests/test_fast_reach.py.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import Callable, List, Optional

HERE: str = os.path.dirname(os.path.abspath(__file__))
SOURCE: str = os.path.join(HERE, "fast_reach.c")
LIBRARY: str = os.path.join(HERE, "fast_reach.dll" if os.name == "nt" else "fast_reach.so")

Log = Callable[[str], None]

# A few positions with something of everything: castling both ways, Chess960 castling, en passant, promotion,
# checks and pins.
CHECK_FENS: List[str] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "bnrqkrnb/pppppppp/8/8/8/8/PPPPPPPP/BNRQKRNB w KQkq - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
]


def compiler() -> Optional[str]:
    """The C compiler to use, or None."""
    for name in (os.environ.get("CC"), "gcc", "clang", "cc"):
        if name and shutil.which(name):
            return name
    return None


def build(log: Log = print) -> str:
    """Compile fast_reach.c into LIBRARY and return its path. Raises RuntimeError with the reason when it cannot."""
    cc = compiler()
    if cc is None:
        raise RuntimeError("no C compiler found (gcc, clang or cc on the PATH, or name one in CC)")
    made = LIBRARY + ".new"
    # -static-libgcc: on Windows the library must not need gcc's own DLLs, which Python would not find
    command = [cc, "-O3", "-shared", SOURCE, "-o", made] + (["-static-libgcc"] if os.name == "nt" else ["-fPIC"])
    log(" ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(made):
        raise RuntimeError(f"the compiler failed:\n{result.stdout}{result.stderr}")
    try:
        os.replace(made, LIBRARY)
    except OSError as error:
        os.remove(made)
        raise RuntimeError(f"could not replace {LIBRARY}: is a labeller or a Python session still using it? ({error})")
    log(f"built {LIBRARY} ({os.path.getsize(LIBRARY):,} bytes)")
    return LIBRARY


def check(log: Log = print) -> bool:
    """Does the library give the Python search's arrays on CHECK_FENS? Also says how long each takes."""
    import numpy as np
    import piece_reach as pr
    if not pr.use_compiled(True):
        log(f"the library is not in use: {pr.C_UNAVAILABLE}")
        return False
    slow = fast = 0.0
    for fen in CHECK_FENS:
        for search, depth in ((pr.threat_counts, 2), (pr.reach_counts, 3)):
            started = time.perf_counter()
            expected = search(fen, depth, use_c=False)
            slow += time.perf_counter() - started
            started = time.perf_counter()
            got = search(fen, depth)
            fast += time.perf_counter() - started
            if not np.array_equal(got, expected):
                log(f"DIFFERENT from the Python search: {search.__name__}({fen!r}, {depth})")
                return False
    if pr.search_stats()["searches"] < 2 * len(CHECK_FENS):
        log("the library refused positions it should handle")
        return False
    searches = 2 * len(CHECK_FENS)
    log(f"the same arrays as the Python search on {searches} searches; a search took {1000 * slow / searches:.1f} ms in "
        f"Python and {1000 * fast / searches:.3f} ms compiled ({slow / max(fast, 1e-9):,.0f} times faster)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-check", action="store_true", help="build only")
    parser.add_argument("--check-only", action="store_true", help="check the library that is there, without building")
    args = parser.parse_args()
    if args.check_only:
        return 0 if check() else 1
    try:
        build()
    except RuntimeError as error:
        print(error)
        return 1
    if args.no_check:
        return 0
    # In a process of its own, so that this one never has the library loaded and can remove it
    if subprocess.run([sys.executable, os.path.abspath(__file__), "--check-only"]).returncode != 0:
        os.remove(LIBRARY)          # never leave a library that disagrees where piece_reach.py would pick it up
        print("removed the library: labelling stays on the Python search")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
