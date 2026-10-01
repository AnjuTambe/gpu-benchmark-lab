"""
Tests for analyzer.classify, using small fake logs shaped like real torchrun output
(the [rankN]: prefixes and the 'Root Cause' block are copied from real failed runs).

Run with:  pytest
"""

from analyzer import classify, format_report


def root_cause(rank, exitcode, traceback="To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html"):
    """The summary torchrun prints when a worker fails."""
    return f"""torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
train.py FAILED
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  rank      : {rank} (local_rank: {rank})
  exitcode  : {exitcode} (pid: 12345)
  error_file: <N/A>
  traceback : {traceback}
============================================================
"""


def test_cuda_out_of_memory():
    log = ("[rank0]: Traceback (most recent call last):\n"
           "[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 TiB. "
           "GPU 0 has a total capacity of 15.77 GiB\n") + root_cause(0, 1)
    info = classify(1, log, False)
    assert info["type"] == "CUDA_OUT_OF_MEMORY"
    assert info["rank"] == 0


def test_cpu_out_of_memory():
    log = ("[rank1]: Traceback (most recent call last):\n"
           "[rank1]: RuntimeError: DefaultCPUAllocator: can't allocate memory: you tried to allocate "
           "1125899906842624 bytes. Error code 12 (Cannot allocate memory)\n") + root_cause(1, 1)
    info = classify(1, log, False)
    assert info["type"] == "OUT_OF_MEMORY"
    assert info["rank"] == 1


def test_mps_out_of_memory():
    log = "RuntimeError: MPS backend out of memory. Tried to allocate 1.00 PiB on private pool.\n"
    info = classify(1, log, False)
    assert info["type"] == "OUT_OF_MEMORY"


def test_worker_crash_without_python_error():
    log = "Processes: 2  |  Device: cpu  |  Backend: gloo\n" + root_cause(1, 1)
    info = classify(1, log, False)
    assert info["type"] == "WORKER_CRASH"
    assert info["rank"] == 1


def test_worker_crash_killed_by_signal():
    log = root_cause(0, -9, traceback="Signal 9 (SIGKILL) received by PID 12345")
    info = classify(1, log, False)
    assert info["type"] == "WORKER_CRASH"
    assert info["rank"] == 0
    assert "SIGKILL" in info["reason"]


def test_nan_loss():
    log = ("[rank1]: Traceback (most recent call last):\n"
           "[rank1]: FloatingPointError: NaN loss detected on rank 1 at step 3 (loss=nan).\n") + root_cause(0, 1)
    info = classify(1, log, False)
    assert info["type"] == "NAN_LOSS"
    assert info["rank"] == 1  # the message names rank 1, even though torchrun's root cause is rank 0


def test_distributed_config_error():
    log = ("[failure-mode bad-config] rank 0: joining with world_size=3, but only 2 process(es) were started\n"
           "torch.distributed.DistStoreError: wait timeout after 15000ms, keys: /default_pg/0//cpu//0/2\n"
           ) + root_cause(0, 1)
    info = classify(1, log, False)
    assert info["type"] == "DISTRIBUTED_CONFIG_ERROR"
    assert info["rank"] == 0


def test_distributed_config_error_wins_over_timeout():
    # A hang we stopped ourselves, but the log says why: the hostname could not be resolved.
    log = ("[c10d] The IPv6 network addresses of (my-mac.local, 54517) cannot be retrieved "
           "(gai error: 8 - nodename nor servname provided, or not known).\n")
    info = classify(None, log, True)
    assert info["type"] == "DISTRIBUTED_CONFIG_ERROR"


def test_timeout():
    log = "Processes: 2  |  Device: cpu  |  Backend: gloo\n"
    info = classify(None, log, True)
    assert info["type"] == "TIMEOUT"
    assert info["rank"] is None


def test_unknown_error_is_not_guessed():
    log = ("[rank0]: Traceback (most recent call last):\n"
           "[rank0]: ValueError: something nobody wrote a rule for\n") + root_cause(0, 1)
    info = classify(1, log, False)
    assert info["type"] == "UNKNOWN"
    assert "No rule matched" in info["reason"]
    assert "ValueError: something nobody wrote a rule for" in info["reason"]


def test_unknown_with_empty_log():
    info = classify(1, "", False)
    assert info["type"] == "UNKNOWN"
    assert info["rank"] is None


def test_format_report():
    report = format_report(classify(None, "", True))
    lines = report.splitlines()
    assert lines[0] == "FAILURE DETECTED"
    assert lines[1] == "Type: TIMEOUT"
    assert lines[2] == "Rank: unknown"
    assert lines[3].startswith("Reason: ")
    assert lines[4].startswith("Suggested action: ")
