"""
profile_stats.py - Turn torch.profiler results into simple numbers (used by train.py --profile).

This file does not import torch, so it can be tested with small fake events.

Important detail: on a GPU, the profiler records each record_function range twice,
once on the CPU (with the real time) and once as a GPU-side copy with the same name
(CPU time 0). Only the CPU copies are used here, otherwise ranges show 0% and the
step time is counted twice.
"""

RANGES = ("forward", "backward", "optimizer")


def is_cpu(event):
    """True for events recorded on the CPU (device_type prints as 'DeviceType.CPU')."""
    return str(event.device_type).endswith("CPU")


def is_gpu(event):
    return str(event.device_type).endswith("CUDA")


def step_breakdown(averages, profiled_steps):
    """
    From prof.key_averages(): average step time (ms) and the share (%) of it spent
    in each range in RANGES, plus 'other' for the rest.
    """
    cpu = [e for e in averages if is_cpu(e)]
    # The profiler marks each step as "ProfilerStep#N"; together they are the whole profiled time.
    step_us = sum(e.cpu_time_total for e in cpu if e.key.startswith("ProfilerStep"))
    result = {"step_ms_avg": step_us / profiled_steps / 1000 if profiled_steps else 0.0}
    for name in RANGES:
        range_us = sum(e.cpu_time_total for e in cpu if e.key == name)
        result[f"{name}_pct"] = range_us / step_us * 100 if step_us else 0.0
    result["other_pct"] = 100 - sum(result[f"{name}_pct"] for name in RANGES) if step_us else 0.0
    return result


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


def has_allreduce_work(events):
    """True if the profiler recorded actual all-reduce work (not just the call that starts it)."""
    return any(is_comm_event(e.name) and not e.name.startswith("c10d::") for e in events)


def allreduce_during_backward(events):
    """
    Measure gradient all-reduce from prof.events(). DDP *starts* each all-reduce on the
    main thread ('c10d::allreduce_', very short). The real work runs elsewhere:
      - gloo (CPU): on gloo's own threads ('gloo:all_reduce')
      - NCCL (GPU): as a GPU kernel (e.g. 'ncclDevKernel_AllReduce_...'); the CPU-side
        'nccl:all_reduce' only queues it, so GPU kernels are used when present.
    Returns per-step averages, in ms.
    """
    backward = [(e.time_range.start, e.time_range.end) for e in events if e.name == "backward" and is_cpu(e)]
    comm = [e for e in events if is_comm_event(e.name)]
    launches = [e for e in comm if e.name.startswith("c10d::")]
    work = [e for e in comm if not e.name.startswith("c10d::")]
    gpu_work = [e for e in work if is_gpu(e)]
    if gpu_work:
        work = gpu_work

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
        "work_measured_on": "gpu kernels" if gpu_work else "cpu threads",
        "calls_per_step": len(launches) / steps,
        "launch_ms_per_step": sum(e.cpu_time_total for e in launches) / steps / 1000,
        "work_ms_per_step_summed": sum(e.time_range.end - e.time_range.start for e in work) / steps / 1000,
        "busy_ms_per_step_during_backward": merged_length(clipped) / steps / 1000,
    }
