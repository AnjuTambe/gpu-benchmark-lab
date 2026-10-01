"""
Tests for profile_stats, using small fake profiler events (no torch needed).

The GPU-style fakes copy what a real Kaggle T4 run showed: every named range
appears twice, once on the CPU and once as a GPU-side copy with 0 CPU time.

Run with:  pytest
"""

from types import SimpleNamespace as Fake

from profile_stats import allreduce_during_backward, has_allreduce_work, merged_length, step_breakdown

CPU, CUDA = "DeviceType.CPU", "DeviceType.CUDA"


def avg(key, cpu_us, device=CPU):
    """A fake entry from prof.key_averages()."""
    return Fake(key=key, cpu_time_total=cpu_us, device_type=device)


def event(name, start, end, device=CPU, cpu_us=None):
    """A fake entry from prof.events(); times in microseconds."""
    return Fake(name=name, device_type=device, time_range=Fake(start=start, end=end),
                cpu_time_total=(end - start) if cpu_us is None else cpu_us)


def test_step_breakdown_cpu_only():
    averages = [avg("ProfilerStep#2", 50_000), avg("ProfilerStep#3", 50_000),  # 2 steps, 100 ms
                avg("forward", 30_000), avg("backward", 50_000), avg("optimizer", 10_000)]
    result = step_breakdown(averages, profiled_steps=2)
    assert result["step_ms_avg"] == 50.0
    assert result["forward_pct"] == 30.0
    assert result["backward_pct"] == 50.0
    assert result["optimizer_pct"] == 10.0
    assert round(result["other_pct"], 6) == 10.0


def test_step_breakdown_ignores_gpu_copies_of_ranges():
    # On CUDA every range also has a GPU-side copy with 0 CPU time. Before the fix these
    # copies made forward/backward show 0% and doubled the step time.
    averages = [avg("ProfilerStep#2", 40_000), avg("ProfilerStep#2", 0, CUDA),
                avg("forward", 10_000), avg("forward", 0, CUDA),
                avg("backward", 20_000), avg("backward", 0, CUDA),
                avg("optimizer", 4_000), avg("optimizer", 0, CUDA)]
    result = step_breakdown(averages, profiled_steps=1)
    assert result["step_ms_avg"] == 40.0
    assert result["forward_pct"] == 25.0
    assert result["backward_pct"] == 50.0
    assert result["optimizer_pct"] == 10.0


def test_step_breakdown_with_no_steps_recorded():
    result = step_breakdown([], profiled_steps=10)
    assert result["step_ms_avg"] == 0.0
    assert result["forward_pct"] == 0.0


def test_merged_length_counts_overlaps_once():
    assert merged_length([(0, 10), (5, 15), (20, 25)]) == 20
    assert merged_length([]) == 0


def test_allreduce_gloo_uses_background_threads():
    events = [event("backward", 0, 10_000),
              event("c10d::allreduce_", 2_000, 2_020),        # main thread: only starts it
              event("gloo:all_reduce", 2_030, 8_000),         # gloo thread: the real work
              event("gloo:all_reduce", 3_000, 6_000)]
    assert has_allreduce_work(events)
    result = allreduce_during_backward(events)
    assert result["work_measured_on"] == "cpu threads"
    assert result["calls_per_step"] == 1
    assert result["busy_ms_per_step_during_backward"] == (8_000 - 2_030) / 1000  # overlap counted once


def test_allreduce_nccl_uses_gpu_kernels():
    events = [event("backward", 0, 10_000), event("backward", 0, 9_000, CUDA),  # GPU copy is ignored
              event("c10d::allreduce_", 1_000, 1_100),
              event("nccl:all_reduce", 1_100, 1_200),                          # CPU side: only queues it
              event("ncclDevKernel_AllReduce_Sum_f32_RING_LL", 4_000, 7_000, CUDA, cpu_us=0)]
    result = allreduce_during_backward(events)
    assert result["work_measured_on"] == "gpu kernels"
    assert result["work_ms_per_step_summed"] == 3.0
    assert result["busy_ms_per_step_during_backward"] == 3.0


def test_no_allreduce_work_recorded():
    events = [event("backward", 0, 10_000), event("c10d::allreduce_", 1_000, 1_100)]
    assert not has_allreduce_work(events)
