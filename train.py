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
  --profile                      profile a few steps after warmup with torch.profiler (see docs/profiling.md)
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
from torch.profiler import ProfilerActivity, record_function
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
PROFILE_WARMUP = 2            # --profile: steps the profiler runs but throws away (its own startup cost)
PROFILE_STEPS = 10            # --profile: steps that are actually recorded


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
    parser.add_argument("--profile", action="store_true",
                        help=f"profile {PROFILE_STEPS} extra steps after warmup and save a summary and a trace")
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


def is_comm_event(name):
    """Profiler events that are gradient communication (e.g. 'gloo:all_reduce', 'c10d::allreduce_')."""
    name = name.lower()
    return "all_reduce" in name or "allreduce" in name


def merged_length(intervals):
    """Total length covered by a list of (start, end) intervals, counting overlaps once."""
    total, current_end = 0, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            total += end - start
            current_end = end
        elif end > current_end:
            total += end - current_end
            current_end = end
    return total


def allreduce_during_backward(events):
    """
    Measure gradient all-reduce from profiler events. With gloo, DDP *starts* each
    all-reduce on the main thread ('c10d::allreduce_', very short) and the real work
    runs on gloo's own threads ('gloo:all_reduce'), at the same time as backward.
    Returns per-step averages, in ms.
    """
    backward = [(e.time_range.start, e.time_range.end) for e in events if e.name == "backward"]
    comm = [e for e in events if is_comm_event(e.name)]
    launches = [e for e in comm if e.name.startswith("c10d::")]
    work = [e for e in comm if not e.name.startswith("c10d::")]

    # Wall time during which at least one all-reduce was running inside a backward range.
    clipped = []
    for e in work:
        for b_start, b_end in backward:
            start, end = max(e.time_range.start, b_start), min(e.time_range.end, b_end)
            if end > start:
                clipped.append((start, end))
    steps = len(backward) or 1
    return {
        "event_names": sorted({e.name for e in comm}),
        "calls_per_step": len(launches) / steps,
        "launch_ms_per_step": sum(e.cpu_time_total for e in launches) / steps / 1000,
        "work_ms_per_step_summed": sum(e.cpu_time_total for e in work) / steps / 1000,
        "busy_ms_per_step_during_backward": merged_length(clipped) / steps / 1000,
    }


def save_profile(prof, base, rank, device, world_size):
    """
    Save the profiler's summary table and chrome trace, and work out how the
    step time splits into forward / backward / optimizer. Returns a dict for results.
    All numbers are CPU wall time of the recorded ranges, averaged over PROFILE_STEPS.
    """
    trace_file = f"{base}_rank{rank}_trace.json"
    summary_file = f"{base}_rank{rank}_profile.txt"
    prof.export_chrome_trace(trace_file)

    averages = prof.key_averages()
    by_name = {e.key: e for e in averages}
    # The profiler marks each step as "ProfilerStep#N"; together they are the whole profiled time.
    step_us = sum(e.cpu_time_total for e in averages if e.key.startswith("ProfilerStep"))

    def share(name):
        return by_name[name].cpu_time_total / step_us * 100 if name in by_name else 0.0

    info = {
        "rank": rank,
        "profiled_steps": PROFILE_STEPS,
        "step_ms_avg": step_us / PROFILE_STEPS / 1000,
        "forward_pct": share("forward"),
        "backward_pct": share("backward"),
        "optimizer_pct": share("optimizer"),
    }
    info["other_pct"] = 100 - info["forward_pct"] - info["backward_pct"] - info["optimizer_pct"]

    # Gradient all-reduce, if the profiler recorded it.
    events = prof.events()
    if world_size == 1:
        info["allreduce"] = None
        info["allreduce_note"] = "single process: no all-reduce"
    elif not any(is_comm_event(e.name) and not e.name.startswith("c10d::") for e in events):
        info["allreduce"] = None
        info["allreduce_note"] = "the profiler recorded no all-reduce work events, so it cannot be measured here"
    else:
        info["allreduce"] = allreduce_during_backward(events)
        info["allreduce"]["busy_pct_of_step"] = (info["allreduce"]["busy_ms_per_step_during_backward"]
                                                 / info["step_ms_avg"] * 100)
        info["allreduce_note"] = ("busy_ms_per_step_during_backward = wall time while at least one all-reduce "
                                  "ran inside backward; it overlaps with backward compute, so it is not "
                                  "simply added to the step time")

    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    with open(summary_file, "w") as f:
        f.write(f"Profile of rank {rank}, {PROFILE_STEPS} steps, device {device}, {world_size} process(es)\n")
        f.write(f"Average step: {info['step_ms_avg']:.3f} ms  |  forward {info['forward_pct']:.1f}%  "
                f"backward {info['backward_pct']:.1f}%  optimizer {info['optimizer_pct']:.1f}%  "
                f"other {info['other_pct']:.1f}%\n")
        f.write(f"All-reduce: {json.dumps(info['allreduce']) if info['allreduce'] else info['allreduce_note']}\n\n")
        f.write(averages.table(sort_by=sort_by, row_limit=20))
    info["trace_file"] = trace_file
    info["summary_file"] = summary_file
    return info


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
    total_steps = WARMUP_STEPS + args.steps + (PROFILE_WARMUP + PROFILE_STEPS if args.profile else 0)
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
        # --sync-mode no_sync: DDP's no_sync() skips the gradient all-reduce in backward().
        if use_ddp and args.sync_mode == "no_sync":
            grad_sync = model.no_sync()
        else:
            grad_sync = contextlib.nullcontext()
        # The record_function ranges name each part of the step in the profiler output.
        with grad_sync:
            with record_function("forward"):
                loss = loss_fn(model(inputs), labels)
            # Stop with a clear error instead of silently training on garbage.
            if not torch.isfinite(loss):
                raise FloatingPointError(f"NaN loss detected on rank {rank} at step {step} "
                                         f"(loss={loss.item()}). Check the input batch for NaN/Inf values.")
            with record_function("backward"):
                loss.backward()  # with DDP (sync mode), gradients are shared between processes here
        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad()  # clear gradients for the next step
        return loss

    batches = iter(loader)

    # Where results go, e.g. results/cpu_p2_bs256_sync.json. Profile files are saved next to it.
    path = args.out or os.path.join(
        RESULTS_DIR, f"{device.type}_p{world_size}_bs{args.batch_size}_{args.sync_mode}.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Warm-up: not timed.
    for step in range(WARMUP_STEPS):
        train_step(step, *next(batches))
    sync(device, world_size)
    next_step = WARMUP_STEPS

    # Profiling: separate extra steps, so the profiler's overhead does not affect the timed run.
    profile_info = None
    if args.profile:
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
        schedule = torch.profiler.schedule(wait=0, warmup=PROFILE_WARMUP, active=PROFILE_STEPS)
        base = path[:-len(".json")] if path.endswith(".json") else path
        saved = []  # the profiler clears its events after each cycle, so save them in on_trace_ready
        with torch.profiler.profile(activities=activities, schedule=schedule,
                                    on_trace_ready=lambda p: saved.append(
                                        save_profile(p, base, rank, device, world_size))) as prof:
            for step in range(next_step, next_step + PROFILE_WARMUP + PROFILE_STEPS):
                train_step(step, *next(batches))
                sync_device(device)
                prof.step()  # tells the profiler a step has finished
        next_step += PROFILE_WARMUP + PROFILE_STEPS
        profile_info = saved[0]
        sync(device, world_size)

    # Timed run. We also time every step on its own to get p50/p95 latency.
    step_times_ms = []
    start = time.perf_counter()
    for step in range(next_step, next_step + args.steps):
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
            "profile": profile_info,  # None unless --profile; measured on rank 0
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
        if profile_info:
            print(f"Profile (rank 0): forward {profile_info['forward_pct']:.1f}%  "
                  f"backward {profile_info['backward_pct']:.1f}%  "
                  f"optimizer {profile_info['optimizer_pct']:.1f}%  other {profile_info['other_pct']:.1f}%  "
                  f"of {profile_info['step_ms_avg']:.3f} ms/step")
            print(f"Profile files:    {profile_info['summary_file']}, {profile_info['trace_file']}")

        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
