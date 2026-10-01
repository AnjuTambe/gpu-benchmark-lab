"""
train.py - Train a small PyTorch model on fake data and measure how fast it runs.

Two ways to run it:
  python train.py                              -> single process
  torchrun --nproc_per_node=N train.py         -> N processes with DistributedDataParallel (DDP)

Options:
  --device {auto,cpu,mps,cuda}   where to run (default: auto)
  --steps N                      number of timed steps (default: 300)
  --sync-mode {sync,no_sync,no_ddp}
                                 sync:    normal DDP, gradients all-reduced every step (default)
                                 no_sync: DDP wrapper, but gradients are never all-reduced
                                 no_ddp:  no DDP wrapper, each process trains its own model

Device rules for --device auto:
  - NVIDIA GPU (CUDA) first, then Apple GPU (MPS), otherwise CPU.
  - MPS is skipped when running more than 1 process, because it does not
    support distributed training.
The DDP backend follows the device: "nccl" for CUDA, "gloo" for CPU.

At the end, only rank 0 (the first process) prints the timing results and
saves them to results/<device>_p<processes>_bs<batch>_<sync_mode>.json, for
example results/cpu_p2_bs256_sync.json.
"""

import argparse
import contextlib
import json
import os
import platform
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# ---- Settings (big enough that one step takes a few ms on a laptop CPU) ----
BATCH_SIZE = 256     # samples per training step, per process
INPUT_SIZE = 512     # number of features in each fake sample
HIDDEN_SIZE = 2048   # size of each hidden layer
NUM_CLASSES = 10     # number of fake labels
WARMUP_STEPS = 5     # untimed steps first, so one-time setup cost isn't measured
COMM_WARMUP = 5      # untimed all-reduce calls before the communication test
COMM_RUNS = 50       # timed all-reduce calls in the communication test
RESULTS_DIR = "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a small model on fake data.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                        help="where to run (default: auto)")
    parser.add_argument("--steps", type=int, default=300,
                        help="number of timed training steps (default: 300)")
    parser.add_argument("--sync-mode", choices=["sync", "no_sync", "no_ddp"], default="sync",
                        help="how gradients are shared between processes (default: sync)")
    return parser.parse_args()


def read_torchrun_env():
    """
    Read the settings torchrun gives us. Returns (rank, local_rank, world_size).
    With plain `python train.py` this is (0, 0, 1).
    """
    rank = int(os.environ.get("RANK", 0))              # this process's id across all processes
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # this process's id on this machine
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # total number of processes
    return rank, local_rank, world_size


def pick_device(choice, local_rank, world_size):
    """Turn the --device option into a real device, or stop with a clear message."""
    if choice == "auto":
        if torch.cuda.is_available():
            choice = "cuda"
        elif world_size == 1 and torch.backends.mps.is_available():
            choice = "mps"
        else:
            choice = "cpu"

    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda was requested, but no NVIDIA GPU (CUDA) is available.")
        return torch.device(f"cuda:{local_rank}")  # one GPU per process
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("--device mps was requested, but the Apple GPU (MPS) is not available.")
        if world_size > 1:
            raise SystemExit("--device mps cannot be used with more than 1 process "
                             "(MPS does not support distributed training). Use --device cpu.")
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device):
    """GPUs run work in the background. Wait for them to finish so timing is accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def sync(device, world_size):
    """Wait for the device, and with several processes also wait for every process."""
    sync_device(device)
    if world_size > 1:
        dist.barrier()


def time_allreduce(num_values, device, world_size):
    """
    Communication test: average time (ms) of one dist.all_reduce on a float32
    tensor with num_values values, the same size as all the model's gradients.
    Returns None in single-process mode, where there is nothing to communicate.
    """
    if world_size == 1:
        return None
    tensor = torch.ones(num_values, dtype=torch.float32, device=device)
    for _ in range(COMM_WARMUP):
        dist.all_reduce(tensor)
    sync(device, world_size)

    start = time.perf_counter()
    for _ in range(COMM_RUNS):
        dist.all_reduce(tensor)
    sync_device(device)
    return (time.perf_counter() - start) / COMM_RUNS * 1000


def main():
    args = parse_args()
    rank, local_rank, world_size = read_torchrun_env()
    device = pick_device(args.device, local_rank, world_size)
    is_main = rank == 0  # only rank 0 prints and saves

    # With several processes, start DDP. The backend must match the device.
    backend = None
    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    if is_main:
        print(f"Processes: {world_size}  |  Device: {device}  |  Backend: {backend or 'none'}  "
              f"|  CPU threads per process: {torch.get_num_threads()}  |  Sync mode: {args.sync_mode}")

    # Same seed everywhere, so every process builds the same starting model.
    torch.manual_seed(0)

    # A small model: three linear layers with ReLUs in between.
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, NUM_CLASSES),
    ).to(device)
    num_grad_values = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Communication test, before training.
    comm_ms = time_allreduce(num_grad_values, device, world_size)

    # With several processes, wrap the model in DDP so gradients are averaged across processes.
    # --sync-mode no_ddp skips this, so each process trains its own plain model.
    use_ddp = world_size > 1 and args.sync_mode != "no_ddp"
    if use_ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Fake dataset: random inputs and labels. Sized so every process gets
    # exactly WARMUP_STEPS + args.steps batches.
    total_steps = WARMUP_STEPS + args.steps
    num_samples = BATCH_SIZE * total_steps * world_size
    dataset = TensorDataset(
        torch.randn(num_samples, INPUT_SIZE),
        torch.randint(0, NUM_CLASSES, (num_samples,)),
    )

    # DistributedSampler gives each process a different slice of the data.
    # With 1 process it simply returns all the data.
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    def train_step(inputs, labels):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # --sync-mode no_sync: DDP's no_sync() skips the gradient all-reduce in backward().
        if use_ddp and args.sync_mode == "no_sync":
            grad_sync = model.no_sync()
        else:
            grad_sync = contextlib.nullcontext()
        with grad_sync:
            loss = loss_fn(model(inputs), labels)
            loss.backward()  # with DDP (sync mode), gradients are shared between processes here
        optimizer.step()
        return loss

    batches = iter(loader)

    # Warm-up: not timed.
    for _ in range(WARMUP_STEPS):
        train_step(*next(batches))
    sync(device, world_size)

    # Timed run. We also time every step on its own to get p50/p95 latency.
    step_times_ms = []
    start = time.perf_counter()
    for _ in range(args.steps):
        step_start = time.perf_counter()
        loss = train_step(*next(batches))
        sync_device(device)  # make sure this step's GPU work is done before reading the clock
        step_times_ms.append((time.perf_counter() - step_start) * 1000)
    sync(device, world_size)
    total_time = time.perf_counter() - start

    # All numbers below come from the measured run above.
    if is_main:
        total_samples = BATCH_SIZE * args.steps * world_size  # across all processes
        percentiles = statistics.quantiles(step_times_ms, n=100)  # 99 cut points: p1..p99
        results = {
            "device": str(device),
            "world_size": world_size,
            "backend": backend,
            "sync_mode": args.sync_mode,
            "cpu_threads_per_process": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "batch_size_per_process": BATCH_SIZE,
            "timed_steps": args.steps,
            "total_time_sec": total_time,
            "samples_per_sec": total_samples / total_time,
            "time_per_step_ms": (total_time / args.steps) * 1000,
            "step_p50_ms": percentiles[49],  # measured on rank 0
            "step_p95_ms": percentiles[94],  # measured on rank 0
            "grad_values": num_grad_values,
            "comm_ms_per_allreduce": comm_ms,
            "final_loss_rank0": loss.item(),
        }

        print(f"Total time:       {results['total_time_sec']:.4f} s")
        print(f"Samples per sec:  {results['samples_per_sec']:.1f}  (all processes combined)")
        print(f"Time per step:    {results['time_per_step_ms']:.3f} ms  "
              f"(p50 {results['step_p50_ms']:.3f}, p95 {results['step_p95_ms']:.3f})")
        if comm_ms is None:
            print("All-reduce:       n/a (single process)")
        else:
            print(f"All-reduce:       {comm_ms:.3f} ms for {num_grad_values:,} values")
        print(f"Final loss:       {results['final_loss_rank0']:.4f}  (rank 0)")

        # One file per setup, e.g. results/cpu_p2_bs256_sync.json, so runs don't overwrite each other.
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, f"{device.type}_p{world_size}_bs{BATCH_SIZE}_{args.sync_mode}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
