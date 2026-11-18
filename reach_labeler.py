"""
Fills a database with reach labels (piece_reach.py): for training positions, where every piece can be after
1, 2, ... plies of legal play and in how many ways.

Run it like position_labeler.py: set the options below (or give them on the command line), start it, and press
q in its window to stop. It finishes the positions it has in hand, and the next start carries on from there.

    python reach_labeler.py                         label with the options below
    python reach_labeler.py --depth 4 --workers 16  the same, deeper and on fewer cores
    python reach_labeler.py --check                 what the database holds so far

Which positions. Labels are only useful for positions a training run will see, so the labeler walks the
training split in the order a run with seed ORDER_SEED meets it (the trainer's own shuffle), and with
ROLLOUT_PLIES > 0 also labels the positions after the first moves of each sample's rollout: the positions the
"predicted next z" losses are graded on. Those moves are the trainer's own random draws for that seed and
epoch, so they match what the run will play (as long as it uses the default masks: a sample's mask is drawn
before its moves). Labels are stored under the position itself, so another seed or split reuses whatever
positions it shares.

Depth. A search to DEPTH plies gives the counts for every depth up to it, so a deeper label replaces a
shallower one. Each depth keeps its own place in the walk: after a while at depth 3, a day at depth 4 starts
again from the first position and deepens the labels in training order, and going back to depth 3 carries on
adding new positions where depth 3 left off (skipping any that already have depth 3 or more).

Cost, measured on this machine (python-chess): one core searches a position in about 0.03 s at 3 plies and
0.8 s at 4; 24 workers label about 400 positions a second at 3 plies and about 14 at 4 (a sample with two
rollout positions is three positions). Each further ply costs roughly 25 times more, so 5 plies needs a
compiled search before it is practical. A label takes about 0.9 kB at 3 plies and 1.8 kB at 4. Workers use up to
about 160 MB each at 4 plies; the main process holds the training order (about 3 GB for the full training split).

The table (in its own database file by default, so the 15 GB positions database is never written to and
position_labeler.py can run at the same time):

    reach_counts     position  the position: a FEN without its move counters
                     depth     plies searched
                     counts    piece_reach.pack_counts of the [depth, 64, 65] array
    reach_progress   walk, done: how many positions of each walk are finished
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import logging
import os
import pathlib
import signal
import sqlite3
import sys
import threading
import time
from typing import Callable, Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import piece_reach as pr

# ---------- ---------- ----------
#   Options (the command line overrides them)

DEPTH: int = 3                          # plies searched per position
WORKERS: int = 24                       # worker processes; this machine has 32 logical cores, so 24 leaves 8 free
LOW_PRIORITY: bool = True               # workers run below normal priority: whatever else you do comes first
DB_PATH: str = "minotaur_reach.db"      # the labels database (created if missing)
CACHE_DIR: str = "labyrinth_ssl_cache"  # the position cache the trainer reads
SPLIT: str = "train"
ORDER_SEED: int = 0                     # walk the positions in the order a training run with this seed meets them
ORDER_EPOCH: int = 0                    # ... in this epoch (a run's first epoch is 0)
ROLLOUT_PLIES: int = 2                  # also label the positions after this many moves of each sample's rollout
CHUNK: int = 8                          # cache positions handed to a worker at a time
REPORT_SECONDS: float = 30.0            # a progress line this often
LOG_FILE: str = "reach_labeler_log.log"

Log = Callable[[str], None]


# ---------- ---------- ----------
#   The database


def open_db(path: str) -> sqlite3.Connection:
    """The labels database, with its tables made if they are missing."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")       # readers (a training run) never wait for the writer
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reach_counts (
        position TEXT PRIMARY KEY,
        depth INTEGER NOT NULL,
        counts BLOB NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reach_progress (
        walk TEXT PRIMARY KEY,
        done INTEGER NOT NULL
        )
    """)
    conn.commit()
    return conn


def write_labels(conn: sqlite3.Connection, rows: Sequence[Tuple[str, int, bytes]]) -> None:
    """Add labels, replacing a position's label only by a deeper one. Does not commit."""
    conn.executemany("""
        INSERT INTO reach_counts (position, depth, counts) VALUES (?, ?, ?)
        ON CONFLICT(position) DO UPDATE SET depth = excluded.depth, counts = excluded.counts
        WHERE excluded.depth > reach_counts.depth
    """, rows)


def labelled_depths(conn: sqlite3.Connection, positions: Sequence[str]) -> Dict[str, int]:
    """The depth each of these positions is labelled to (those without a label are left out)."""
    found: Dict[str, int] = {}
    for lo in range(0, len(positions), 500):
        part = list(positions[lo:lo + 500])
        marks = ",".join("?" * len(part))
        found.update(conn.execute(f"SELECT position, depth FROM reach_counts WHERE position IN ({marks})", part))
    return found


class ReachLabels:
    """Read-only access to a labels database, for whatever trains on them."""

    def __init__(self, path: str):
        self.conn = sqlite3.connect(pathlib.Path(path).resolve().as_uri() + "?mode=ro", uri=True)

    def get(self, fen: str):
        """The position's [depth, 64, 65] counts (piece_reach.reach_counts), or None when it has no label."""
        row = self.conn.execute("SELECT counts FROM reach_counts WHERE position = ?", (pr.position_key(fen),)).fetchone()
        return None if row is None else pr.unpack_counts(row[0])

    def close(self) -> None:
        self.conn.close()


def check_db(path: str, log: Log = print) -> None:
    """Print what the labels database holds."""
    if not os.path.exists(path):
        log(f"{path} does not exist yet")
        return
    conn = open_db(path)
    log(f"{path}: {os.path.getsize(path) / 2 ** 20:,.1f} MB")
    for depth, rows, size in conn.execute("SELECT depth, COUNT(*), SUM(LENGTH(counts)) FROM reach_counts GROUP BY depth ORDER BY depth"):
        log(f"  {rows:,} positions labelled to {depth} plies ({size / max(rows, 1):,.0f} bytes each)")
    for walk, done in conn.execute("SELECT walk, done FROM reach_progress ORDER BY walk"):
        log(f"  walk {walk}: {done:,} positions done")
    row = conn.execute("SELECT position, depth, counts FROM reach_counts ORDER BY RANDOM() LIMIT 1").fetchone()
    if row is not None:
        counts = pr.unpack_counts(row[2])
        log(f"  for example {row[0]}")
        for d in range(row[1]):
            log(f"    {int(counts[d].sum(1).max()):,} sequences of {d + 1} plies; (piece, square) pairs with a way: "
                f"{int((counts[d][:, :64] > 0).sum()):,}; pieces that can be captured: {int((counts[d][:, 64] > 0).sum())}")
    conn.close()


# ---------- ---------- ----------
#   Which positions


def walk_name(split: str, seed: int, epoch: int, rollout_plies: int, depth: int) -> str:
    return f"{split} seed {seed} epoch {epoch} rollout {rollout_plies} depth {depth}"


def rollout_fens(fen: str, rng, plies: int, mask_config=None) -> List[str]:
    """
    The positions after the first `plies` moves of the rollout labyrinth_data.build_sample plays from fen with
    this rng (fewer when the game ends): the same draws in the same order, the mask first.
    """
    import chess
    import labyrinth_data as ld
    ld.sample_mask(rng, mask_config if mask_config is not None else ld.MaskConfig())
    board = chess.Board(fen, chess960=True)
    fens = []
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        fens.append(ld.fen_of(board))
    return fens


def walk_positions(cache_dir: str, split: str, seed: int, epoch: int, rollout_plies: int,
                   start: int = 0) -> Iterator[List[str]]:
    """
    For each position of the split, in the order a training run with this seed and epoch meets them and
    starting `start` positions in: the FENs to label (the position, then its rollout positions).
    """
    import labyrinth_data as ld
    dataset = ld.PositionDataset(cache_dir, split, seed=seed)
    dataset.epoch = int(epoch)
    sampler = ld._EpochShuffle(dataset, shuffle=True)       # the trainer's own order
    sampler.start = int(start)
    for i in sampler:
        fen = dataset.fen(i)
        yield [fen] + (rollout_fens(fen, dataset.rng_for(i), rollout_plies, dataset.mask_config) if rollout_plies else [])


# ---------- ---------- ----------
#   Workers and stopping


def _start_worker(low_priority: bool) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)            # Ctrl+C is the main process's to handle
    if not low_priority:
        return
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # below normal
        else:
            os.nice(10)
    except Exception:
        pass                                                # a nicety: never a reason not to label


class _InProcess:
    """Stands in for the process pool when workers = 0: jobs run at once, in this process (tests, debugging)."""

    def submit(self, fn, job):
        future: concurrent.futures.Future = concurrent.futures.Future()
        future.set_result(fn(job))
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        pass


def watch_for_q(stop: threading.Event) -> None:
    """
    Set `stop` when q is pressed in THIS console window. (position_labeler.py listens with the keyboard module,
    which hears a q typed anywhere on the machine; this one only hears its own window, so typing elsewhere
    while it runs is safe.) Without a console (started detached) it never fires: use the stop file instead.
    """
    try:
        import msvcrt
    except ImportError:
        return

    def loop() -> None:
        while not stop.is_set():
            try:
                if msvcrt.kbhit() and msvcrt.getwch() in "qQ":
                    stop.set()
                    return
            except Exception:
                return
            time.sleep(0.2)

    threading.Thread(target=loop, daemon=True).start()


# ---------- ---------- ----------
#   The labeling loop


def label(db_path: str = DB_PATH, cache_dir: str = CACHE_DIR, split: str = SPLIT, depth: int = DEPTH,
          workers: int = WORKERS, order_seed: int = ORDER_SEED, order_epoch: int = ORDER_EPOCH,
          rollout_plies: int = ROLLOUT_PLIES, chunk: int = CHUNK, max_positions: Optional[int] = None,
          stop: Optional[threading.Event] = None, low_priority: bool = LOW_PRIORITY,
          report_seconds: float = REPORT_SECONDS, log: Log = print) -> Dict[str, int]:
    """
    Label positions until `stop` is set, the file <db_path>.stop appears, max_positions cache positions have been
    handled in this call, or the split runs out. Returns {"done": positions of the walk finished in all,
    "handled": cache positions handled in this call, "searched": positions searched, "skipped": positions that
    already had a label this deep}.
    """
    if depth < 1 or workers < 0 or chunk < 1 or rollout_plies < 0:
        raise ValueError("need depth >= 1, workers >= 0, chunk >= 1 and rollout_plies >= 0")
    stop = stop if stop is not None else threading.Event()
    stop_file = db_path + ".stop"
    conn = open_db(db_path)
    walk = walk_name(split, order_seed, order_epoch, rollout_plies, depth)
    row = conn.execute("SELECT done FROM reach_progress WHERE walk = ?", (walk,)).fetchone()
    done = int(row[0]) if row else 0
    log(f"walk '{walk}': {done:,} positions done before this start; {workers} workers")
    positions = walk_positions(cache_dir, split, order_seed, order_epoch, rollout_plies, start=done)
    if workers:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_start_worker,
                                                      initargs=(low_priority,))
    else:
        pool = _InProcess()

    handled = searched = skipped = taken = 0
    exhausted = False
    in_hand: Deque[Tuple[int, int, concurrent.futures.Future]] = collections.deque()   # cache positions, skipped, job

    def hand_out() -> None:
        """One more job for the pool: the next `chunk` cache positions, less whatever is labelled already."""
        nonlocal taken, exhausted
        fens: List[str] = []
        count = 0
        while count < chunk and (max_positions is None or taken < max_positions):
            sample = next(positions, None)
            if sample is None:
                exhausted = True
                break
            fens.extend(sample)
            count += 1
            taken += 1
        if not count:
            return
        keys = {pr.position_key(fen): fen for fen in fens}                  # one search per position
        have = labelled_depths(conn, list(keys))
        todo = [fen for key, fen in keys.items() if have.get(key, 0) < depth]
        in_hand.append((count, len(fens) - len(todo), pool.submit(pr.label_positions, (todo, depth))))

    def finish(count: int, already: int, rows: List[Tuple[str, int, bytes]], advance: bool) -> None:
        nonlocal done, handled, searched, skipped
        write_labels(conn, rows)
        searched += len(rows)
        skipped += already
        if advance:
            done += count
            handled += count
            conn.execute("INSERT INTO reach_progress (walk, done) VALUES (?, ?) "
                         "ON CONFLICT(walk) DO UPDATE SET done = excluded.done", (walk, done))
        conn.commit()

    started = last_report = time.time()
    try:
        while True:
            try:
                wanted = max(2 * workers, 1)
                while len(in_hand) < wanted and not exhausted and (max_positions is None or taken < max_positions):
                    hand_out()
                if not in_hand:
                    break
                count, already, job = in_hand[0]
                rows = job.result()                         # (interrupted while waiting: the job stays in hand)
                in_hand.popleft()
                finish(count, already, rows, advance=True)
            except KeyboardInterrupt:
                stop.set()
            now = time.time()
            if now - last_report >= report_seconds:
                last_report = now
                log(f"{done:,} positions of the walk done; this start: {searched:,} searched "
                    f"({searched / max(now - started, 1e-9):,.1f} a second), {skipped:,} already labelled")
            if os.path.exists(stop_file):
                os.remove(stop_file)
                stop.set()
            if stop.is_set():
                break
        if stop.is_set() and in_hand:
            log("stopping: finishing the positions in hand...")
            in_order = True
            for count, already, job in in_hand:             # those not started are dropped, the rest are kept;
                if job.cancel():                            # the walk's place only moves past an unbroken run
                    in_order = False
                    continue
                finish(count, already, job.result(), advance=in_order)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        conn.commit()
        conn.close()
    log(f"walk '{walk}': {done:,} positions done; this start searched {searched:,} positions in "
        f"{time.time() - started:,.0f} s, {skipped:,} were already labelled")
    return {"done": done, "handled": handled, "searched": searched, "skipped": skipped}


# ---------- ---------- ----------
#   Program body

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill a database with reach labels (see the top of this file).")
    parser.add_argument("--depth", type=int, default=DEPTH, help="plies searched per position")
    parser.add_argument("--workers", type=int, default=WORKERS, help="worker processes (0: none, search in this process)")
    parser.add_argument("--db", default=DB_PATH, help="the labels database")
    parser.add_argument("--cache", default=CACHE_DIR, help="the position cache the trainer reads")
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--seed", type=int, default=ORDER_SEED, help="walk in the order a training run with this seed meets the positions")
    parser.add_argument("--epoch", type=int, default=ORDER_EPOCH)
    parser.add_argument("--rollout-plies", type=int, default=ROLLOUT_PLIES, help="also label this many rollout positions per sample")
    parser.add_argument("--chunk", type=int, default=CHUNK, help="cache positions handed to a worker at a time")
    parser.add_argument("--positions", type=int, default=None, help="stop after this many cache positions (default: run until stopped)")
    parser.add_argument("--normal-priority", action="store_true", help="do not lower the workers' priority")
    parser.add_argument("--check", action="store_true", help="print what the database holds and exit")
    args = parser.parse_args()

    if args.check:
        check_db(args.db)
        sys.exit(0)

    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def say(message: str) -> None:
        print(message, flush=True)
        logging.info(message)

    stop_event = threading.Event()
    watch_for_q(stop_event)
    say(f"Labelling to {args.depth} plies with {args.workers} workers into {args.db}. "
        f"Press q in this window to stop (or create the file {args.db}.stop).")
    label(db_path=args.db, cache_dir=args.cache, split=args.split, depth=args.depth, workers=args.workers,
          order_seed=args.seed, order_epoch=args.epoch, rollout_plies=args.rollout_plies, chunk=args.chunk,
          max_positions=args.positions, stop=stop_event, low_priority=not args.normal_priority, log=say)
    check_db(args.db, say)
