"""
train.py - Train a small PyTorch model on fake data and measure how fast it runs.

Two ways to run it:
  python train.py                              -> single process
  torchrun --nproc_per_node=N train.py         -> N processes with DistributedDataParallel (DDP)

Options:
  --device {auto,cpu,mps,cuda}   where to run (default: auto)
  --steps N                      number of timed steps (default: 300)
  --batch-size N                 samples per step, per process (default: 256)
  --out PATH                     where to save the JSON result
  --failure-mode {none,oom,worker-crash,bad-batch,bad-config}
                                 break the run on purpose (default: none), see inject_failure()
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
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# ---- Settings (big enough that one step takes a few ms on a laptop CPU) ----
BATCH_SIZE = 256     # default samples per training step, per process (--batch-size)
INPUT_SIZE = 512     # number of features in each fake sample
HIDDEN_SIZE = 2048   # size of each hidden layer
NUM_CLASSES = 10     # number of fake labels
WARMUP_STEPS = 5     # untimed steps first, so one-time setup cost isn't measured
COMM_WARMUP = 5      # untimed all-reduce calls before the communication test
COMM_RUNS = 50       # timed all-reduce calls in the communication test
RESULTS_DIR = "results"
FAIL_AT_STEP = 3              # --failure-mode oom/worker-crash/bad-batch trigger at this step
BAD_CONFIG_TIMEOUT_SEC = 15   # --failure-mode bad-config gives up connecting after this long


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a small model on fake data.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                        help="where to run (default: auto)")
    parser.add_argument("--steps", type=int, default=300,
                        help="number of timed training steps (default: 300)")
    parser.add_argument("--sync-mode", choices=["sync", "no_sync", "no_ddp"], default="sync",
                        help="how gradients are shared between processes (default: sync)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"samples per step, per process (default: {BATCH_SIZE})")
    parser.add_argument("--out", default=None,
                        help="where to save the JSON result (default: results/<device>_p<N>_bs<B>_<mode>.json)")
    parser.add_argument("--failure-mode", default="none",
                        choices=["none", "oom", "worker-crash", "bad-batch", "bad-config"],
                        help="break the run on purpose to test failure analysis (default: none)")
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


def start_bad_config(backend, rank, world_size):
    """
    --failure-mode bad-config: join the process group claiming one more process
    than was actually started. The missing process never arrives, so connecting
    fails after BAD_CONFIG_TIMEOUT_SEC instead of hanging forever.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")  # needed when run without torchrun
    os.environ.setdefault("MASTER_PORT", "29500")
    wrong_world_size = world_size + 1
    print(f"[failure-mode bad-config] rank {rank}: joining with world_size={wrong_world_size}, "
          f"but only {world_size} process(es) were started", flush=True)
    dist.init_process_group(backend=backend, rank=rank, world_size=wrong_world_size,
                            timeout=timedelta(seconds=BAD_CONFIG_TIMEOUT_SEC))


def inject_failure(mode, step, rank, world_size, device, inputs):
    """
    Break the run on purpose at step FAIL_AT_STEP. Returns the (maybe changed) inputs.
      oom:          a real CUDA out-of-memory error on CUDA; a simulated one on CPU/MPS
      worker-crash: one process exits abruptly (rank 1, or rank 0 with 1 process)
      bad-batch:    the batch gets NaN values; the NaN-loss check in train_step stops the run
    """
    if mode == "none" or step != FAIL_AT_STEP:
        return inputs
    if mode == "oom":
        huge = 1 << 50  # 1 PiB, far more memory than any machine has
        if device.type == "cuda":
            torch.empty(huge, dtype=torch.uint8, device=device)  # raises a real CUDA OOM
        elif device.type == "mps":
            raise RuntimeError(f"[simulated by --failure-mode oom] MPS backend out of memory. "
                               f"Tried to allocate {huge} bytes on private pool.")
        else:
            raise RuntimeError(f"[simulated by --failure-mode oom] DefaultCPUAllocator: can't allocate "
                               f"memory: you tried to allocate {huge} bytes. Error code 12 "
                               f"(Cannot allocate memory)")
    if mode == "worker-crash":
        crash_rank = 1 if world_size > 1 else 0
        if rank == crash_rank:
            os._exit(1)  # exit immediately: no Python traceback, no cleanup, like a real crash
    if mode == "bad-batch":
        inputs = inputs.clone()
        inputs[0] = float("nan")  # one sample full of NaN is enough to make the loss NaN
    return inputs


def main():
    args = parse_args()
    rank, local_rank, world_size = read_torchrun_env()
    device = pick_device(args.device, local_rank, world_size)
    is_main = rank == 0  # only rank 0 prints and saves

    # With several processes, start DDP. The backend must match the device.
    backend = None
    if args.failure_mode == "bad-config":
        backend = "nccl" if device.type == "cuda" else "gloo"
        start_bad_config(backend, rank, world_size)  # fails after a timeout, on purpose
    elif world_size > 1:
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
    num_samples = args.batch_size * total_steps * world_size
    dataset = TensorDataset(
        torch.randn(num_samples, INPUT_SIZE),
        torch.randint(0, NUM_CLASSES, (num_samples,)),
    )

    # DistributedSampler gives each process a different slice of the data.
    # With 1 process it simply returns all the data.
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    def train_step(step, inputs, labels):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inject_failure(args.failure_mode, step, rank, world_size, device, inputs)
        optimizer.zero_grad()
        # --sync-mode no_sync: DDP's no_sync() skips the gradient all-reduce in backward().
        if use_ddp and args.sync_mode == "no_sync":
            grad_sync = model.no_sync()
        else:
            grad_sync = contextlib.nullcontext()
        with grad_sync:
            loss = loss_fn(model(inputs), labels)
            # Stop with a clear error instead of silently training on garbage.
            if not torch.isfinite(loss):
                raise FloatingPointError(f"NaN loss detected on rank {rank} at step {step} "
                                         f"(loss={loss.item()}). Check the input batch for NaN/Inf values.")
            loss.backward()  # with DDP (sync mode), gradients are shared between processes here
        optimizer.step()
        return loss

    batches = iter(loader)

    # Warm-up: not timed.
    for step in range(WARMUP_STEPS):
        train_step(step, *next(batches))
    sync(device, world_size)

    # Timed run. We also time every step on its own to get p50/p95 latency.
    step_times_ms = []
    start = time.perf_counter()
    for step in range(WARMUP_STEPS, WARMUP_STEPS + args.steps):
        step_start = time.perf_counter()
        loss = train_step(step, *next(batches))
        sync_device(device)  # make sure this step's GPU work is done before reading the clock
        step_times_ms.append((time.perf_counter() - step_start) * 1000)
    sync(device, world_size)
    total_time = time.perf_counter() - start

    # All numbers below come from the measured run above.
    if is_main:
        total_samples = args.batch_size * args.steps * world_size  # across all processes
        percentiles = statistics.quantiles(step_times_ms, n=100)  # 99 cut points: p1..p99
        results = {
            "device": str(device),
            "world_size": world_size,
            "backend": backend,
            "sync_mode": args.sync_mode,
            "cpu_threads_per_process": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "batch_size_per_process": args.batch_size,
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
        path = args.out or os.path.join(
            RESULTS_DIR, f"{device.type}_p{world_size}_bs{args.batch_size}_{args.sync_mode}.json")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
