"""
train.py - Train a tiny PyTorch model on fake data and measure how fast it runs.

Two ways to run it:
  python train.py                              -> single process
  torchrun --nproc_per_node=N train.py         -> N processes with DistributedDataParallel (DDP)

Device rules:
  - Single process: NVIDIA GPU (CUDA) first, then Apple GPU (MPS), otherwise CPU.
  - More than 1 process: CUDA with the "nccl" backend if available, otherwise
    CPU with the "gloo" backend. MPS is never used here because it does not
    support distributed training.

At the end, only rank 0 (the first process) prints the timing results and
saves them to results.json.
"""

import json
import os
import platform
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# ---- Settings (small on purpose so it runs fast on a laptop CPU) ----
BATCH_SIZE = 64      # samples per training step, per process
INPUT_SIZE = 128     # number of features in each fake sample
HIDDEN_SIZE = 256    # size of the hidden layer
NUM_CLASSES = 10     # number of fake labels
WARMUP_STEPS = 5     # untimed steps first, so one-time setup cost isn't measured
TIMED_STEPS = 100    # steps we actually time
RESULTS_FILE = "results.json"


def setup_distributed():
    """
    Read the settings torchrun gives us. Returns (rank, local_rank, world_size).
    With plain `python train.py` (or torchrun with 1 process) this is (0, 0, 1)
    and no process group is created.
    """
    rank = int(os.environ.get("RANK", 0))              # this process's id across all processes
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # this process's id on this machine
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # total number of processes

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def pick_device(local_rank, world_size):
    """Choose where to run, following the device rules at the top of this file."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")  # one GPU per process
    if world_size == 1 and torch.backends.mps.is_available():
        return torch.device("mps")                  # Apple GPU, single process only
    return torch.device("cpu")


def sync(device, world_size):
    """
    GPUs run work in the background. Wait for them to finish so timing is accurate.
    With several processes, also wait for every process to reach this point.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    if world_size > 1:
        dist.barrier()


def main():
    rank, local_rank, world_size = setup_distributed()
    device = pick_device(local_rank, world_size)
    is_main = rank == 0  # only rank 0 prints and saves

    if is_main:
        backend = dist.get_backend() if world_size > 1 else "none (single process)"
        print(f"Processes: {world_size}  |  Device: {device}  |  Backend: {backend}")

    # Same seed everywhere, so every process builds the same starting model.
    torch.manual_seed(0)

    # A tiny model: two linear layers with a ReLU in between.
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, NUM_CLASSES),
    ).to(device)

    # With several processes, wrap the model in DDP so gradients are averaged across processes.
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Fake dataset: random inputs and labels. Sized so every process gets
    # exactly WARMUP_STEPS + TIMED_STEPS batches.
    total_steps = WARMUP_STEPS + TIMED_STEPS
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
        loss = loss_fn(model(inputs), labels)
        loss.backward()  # with DDP, gradients are shared between processes here
        optimizer.step()
        return loss

    batches = iter(loader)

    # Warm-up: not timed.
    for _ in range(WARMUP_STEPS):
        train_step(*next(batches))
    sync(device, world_size)

    # Timed run.
    start = time.perf_counter()
    for _ in range(TIMED_STEPS):
        loss = train_step(*next(batches))
    sync(device, world_size)
    total_time = time.perf_counter() - start

    # All numbers below come from the measured run above.
    if is_main:
        total_samples = BATCH_SIZE * TIMED_STEPS * world_size  # across all processes
        results = {
            "device": str(device),
            "world_size": world_size,
            "backend": dist.get_backend() if world_size > 1 else None,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "batch_size_per_process": BATCH_SIZE,
            "timed_steps": TIMED_STEPS,
            "total_time_sec": total_time,
            "samples_per_sec": total_samples / total_time,
            "time_per_step_ms": (total_time / TIMED_STEPS) * 1000,
            "final_loss_rank0": loss.item(),
        }

        print(f"Total time:       {results['total_time_sec']:.4f} s")
        print(f"Samples per sec:  {results['samples_per_sec']:.1f}  (all processes combined)")
        print(f"Time per step:    {results['time_per_step_ms']:.3f} ms")
        print(f"Final loss:       {results['final_loss_rank0']:.4f}  (rank 0)")

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {RESULTS_FILE}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
