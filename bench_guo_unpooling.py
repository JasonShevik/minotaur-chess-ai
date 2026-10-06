"""
Speed comparison of the unpooling layer, v1 against v2, on chess positions.

    python bench_guo_unpooling.py

Two stacked unpooling layers, sized like the summarizer's defaults (64-wide node features, the
chess edge types in, 16-wide edge features between layers, 128-wide hidden MLPs). Two workloads:

  * rollout: sample the decisions for every position (no gradients), as when collecting data;
  * replay:  replay those decisions with gradients and backpropagate the log-probability, as in a
             PPO update.

v1 runs one position at a time. v2 runs one position at a time (batch=None) and in batches, on
the CPU and, if there is one, the GPU. Times are milliseconds per position.
"""
import random
import time
from typing import Callable, List, Tuple

import chess
import torch

import chess_graph as cg
import guo_et_al_unpooling_v1 as gu1
import guo_et_al_unpooling_v2 as gu2
import labyrinth_dgi_encoder as lde

N_POSITIONS = 256
DX, DU, HIDDEN = 64, 16, 128
BATCH_SIZES = (64, 256)


def random_positions(n: int, seed: int = 0) -> List[str]:
    """Positions from random games: up to 60 random legal moves from the start."""
    rnd = random.Random(seed)
    fens = []
    while len(fens) < n:
        board = chess.Board()
        for _ in range(rnd.randint(4, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rnd.choice(moves))
        fens.append(board.fen())
    return fens


def build_inputs(fens: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    torch.manual_seed(0)
    lift = torch.nn.Linear(cg.NUM_NODE_FEATURES, DX)
    inputs = []
    with torch.no_grad():
        for fen in fens:
            edges, x = cg.create_filled_chess_graphs(fen)
            x, ei, ea = lde.chess_graph_to_single(x, edges)
            inputs.append((lift(x.float()), ei, ea))
    return inputs


def make_layers(module, device):
    torch.manual_seed(1)
    kw = dict(kv=HIDDEN, kia=HIDDEN, kie=HIDDEN, kw=HIDDEN)
    first = module.GuoUnpool(dx=DX, dw=cg.NUM_EDGE_TYPES, dy=DX, du=DU, **kw).to(device).eval()
    second = module.GuoUnpool(dx=DX, dw=DU, dy=DX, du=DU, **kw).to(device).eval()
    return first, second


def timed(fn: Callable[[], None], device, repeats: int = 1) -> float:
    fn()  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def bench_v1(inputs, device) -> Tuple[float, float]:
    l1, l2 = make_layers(gu1, device)
    data = [(x.to(device), ei.to(device), ea.to(device)) for x, ei, ea in inputs]
    records = []

    def rollout():
        records.clear()
        rng = torch.Generator(device=device).manual_seed(0)
        with torch.no_grad():
            for x, ei, ea in data:
                a = l1(x, ei, ea, rng=rng)
                b = l2(a[0], a[1], a[2], rng=rng)
                records.append((a[7], b[7]))

    def replay():
        for (x, ei, ea), (r1, r2) in zip(data, records):
            a = l1(x, ei, ea, actions_to_replay=r1)
            b = l2(a[0], a[1], a[2], actions_to_replay=r2)
            (a[3] + b[3]).backward()

    n = len(data)
    t_roll = timed(rollout, device)
    rollout()
    t_replay = timed(replay, device)
    return 1000 * t_roll / n, 1000 * t_replay / n


def bench_v2(inputs, device, batch_size: int) -> Tuple[float, float]:
    """batch_size 0 = one position at a time with batch=None."""
    l1, l2 = make_layers(gu2, device)
    data = [(x.to(device), ei.to(device), ea.to(device)) for x, ei, ea in inputs]
    if batch_size:
        chunks = [gu2.pack_graphs(*zip(*data[i:i + batch_size])) for i in range(0, len(data), batch_size)]
    else:
        chunks = [(x, ei, ea, None) for x, ei, ea in data]
    records = []

    def rollout():
        records.clear()
        rng = torch.Generator(device=device).manual_seed(0)
        with torch.no_grad():
            for x, ei, ea, batch in chunks:
                a = l1(x, ei, ea, rng=rng, batch=batch)
                b = l2(a[0], a[1], a[2], rng=rng, batch=a.batch if batch is not None else None)
                records.append((a.actions, b.actions))

    def replay():
        for (x, ei, ea, batch), (r1, r2) in zip(chunks, records):
            a = l1(x, ei, ea, actions_to_replay=r1, batch=batch)
            b = l2(a[0], a[1], a[2], actions_to_replay=r2, batch=a.batch if batch is not None else None)
            (a[3] + b[3]).sum().backward()

    n = len(data)
    t_roll = timed(rollout, device)
    rollout()
    t_replay = timed(replay, device)
    return 1000 * t_roll / n, 1000 * t_replay / n


def main() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"Building {N_POSITIONS} chess positions...")
    inputs = build_inputs(random_positions(N_POSITIONS))
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])

    rows = []
    for device in devices:
        rows.append((f"v1, one at a time ({device.type})", *bench_v1(inputs, device)))
        rows.append((f"v2, one at a time ({device.type})", *bench_v2(inputs, device, 0)))
        for bs in BATCH_SIZES:
            rows.append((f"v2, batches of {bs} ({device.type})", *bench_v2(inputs, device, bs)))

    print(f"\n{'':34s}{'rollout':>12s}{'replay+backward':>18s}   (ms per position, 2 layers)")
    for name, roll, rep in rows:
        print(f"{name:34s}{roll:12.2f}{rep:18.2f}")


if __name__ == "__main__":
    main()
