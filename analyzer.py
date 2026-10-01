"""
analyzer.py - Explain why a training run failed, using its exit code and log text.

    from analyzer import classify, format_report
    result = classify(exit_code=1, log_text=open("run.log").read(), timed_out=False)
    print(format_report(result))

Or from the command line:
    python analyzer.py path/to/run.log --exit-code 1
    python analyzer.py path/to/run.log --timed-out

The rules are simple text and exit-code checks, written from real torchrun logs
(see tests/test_analyzer.py). Rules are checked in order and the first match wins.
If no rule matches, the result is UNKNOWN: the analyzer never guesses.
"""

import argparse
import re

# Lines like "RuntimeError: ..." or "[rank1]: FloatingPointError: ..." (torchrun adds the [rankN] prefix).
ERROR_LINE = re.compile(r"^(?:\[rank(\d+)\]:\s*)?((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception)):\s?(.*)$",
                        re.MULTILINE)

# Text rules: (type, list of regexes). Checked top to bottom.
TEXT_RULES = [
    ("CUDA_OUT_OF_MEMORY", [r"CUDA out of memory", r"CUDA error: out of memory"]),
    ("OUT_OF_MEMORY", [r"can't allocate memory", r"Cannot allocate memory", r"MPS backend out of memory",
                       r"DefaultCPUAllocator"]),
    ("NAN_LOSS", [r"NaN loss detected", r"[Ll]oss (?:is|=) ?nan\b"]),
    ("DISTRIBUTED_CONFIG_ERROR", [r"DistStoreError", r"client socket has timed out",
                                  r"Connection refused", r"nodename nor servname provided",
                                  r"[Aa]ddress already in use", r"Unsupported nproc_per_node"]),
]

REASONS = {
    "CUDA_OUT_OF_MEMORY": "The GPU ran out of memory.",
    "OUT_OF_MEMORY": "The process ran out of memory (CPU or Apple GPU).",
    "NAN_LOSS": "The loss became NaN, so training cannot continue.",
    "DISTRIBUTED_CONFIG_ERROR": "The processes could not connect to each other.",
}

ACTIONS = {
    "CUDA_OUT_OF_MEMORY": "Reduce --batch-size or the model size. Check that no other process is using "
                          "the GPU (nvidia-smi). Mixed precision or gradient checkpointing also save memory.",
    "OUT_OF_MEMORY": "Reduce --batch-size or the model size, run fewer processes on this machine, "
                     "or close other memory-heavy programs.",
    "WORKER_CRASH": "Look at the failed rank's own output and the system logs (dmesg on Linux, "
                    "Console.app on macOS). A process killed by SIGKILL is often the OS out-of-memory killer. "
                    "Rerun to see if it repeats; torchrun --max-restarts can restart failed workers.",
    "NAN_LOSS": "Check the input data for NaN/Inf values, lower the learning rate, "
                "or add gradient clipping.",
    "DISTRIBUTED_CONFIG_ERROR": "Check that --nproc_per_node matches the world size every process expects, "
                                "that MASTER_ADDR is reachable (use 127.0.0.1 on one machine) and the port is free. "
                                "On macOS set GLOO_SOCKET_IFNAME=lo0.",
    "TIMEOUT": "Look at the last lines of the log to see where it stopped. If the run is just slow, "
               "raise the timeout. If one rank is stuck, rerun with TORCH_DISTRIBUTED_DEBUG=DETAIL.",
    "UNKNOWN": "Read the full log. If this failure happens again, add a rule for it to analyzer.py.",
}


def _root_cause(log_text):
    """
    Read torchrun's 'Root Cause (first observed failure)' block.
    Returns (rank, exitcode, signal_name); each is None if not found.
    """
    match = re.search(r"Root Cause \(first observed failure\):(.*?)(?:={10,}|$)", log_text, re.DOTALL)
    if not match:
        return None, None, None
    block = match.group(1)
    rank = re.search(r"rank\s*:\s*(\d+)", block)
    code = re.search(r"exitcode\s*:\s*(-?\d+)", block)
    signal = re.search(r"Signal \d+ \((SIG\w+)\)", block)
    return (int(rank.group(1)) if rank else None,
            int(code.group(1)) if code else None,
            signal.group(1) if signal else None)


def _worker_errors(log_text):
    """Python exceptions raised inside the training processes (not torchrun's own ChildFailedError)."""
    return [m for m in ERROR_LINE.finditer(log_text) if not m.group(2).endswith("ChildFailedError")]


def _rank_from_line(line, prefix_rank, fallback):
    """Prefer an explicit 'rank N' in the message, then the [rankN] prefix, then the fallback."""
    explicit = re.search(r"\brank (\d+)\b", line)
    if explicit:
        return int(explicit.group(1))
    if prefix_rank is not None:
        return int(prefix_rank)
    return fallback


def classify(exit_code, log_text, timed_out):
    """
    Classify a failed run.
      exit_code: the launcher's exit code (None if the run was killed by our timeout)
      log_text:  everything the run printed (stdout + stderr)
      timed_out: True if the run was stopped because it took too long
    Returns a dict with: type, rank (None if unknown), reason, suggested_action.
    """
    log_text = log_text or ""
    root_rank, root_code, root_signal = _root_cause(log_text)
    errors = _worker_errors(log_text)

    def result(kind, rank, reason):
        return {"type": kind, "rank": rank, "reason": reason, "suggested_action": ACTIONS[kind]}

    # 1. Text rules: a known error message in the log. Error lines ("XError: ...") are checked
    #    first, so the reason shows the error itself rather than a line of traceback source code.
    candidate_lines = [m.group(0) for m in errors] + log_text.splitlines()
    for kind, patterns in TEXT_RULES:
        for line in candidate_lines:
            if any(re.search(p, line) for p in patterns):
                prefix = re.match(r"^\[rank(\d+)\]:", line)
                rank = _rank_from_line(line, prefix.group(1) if prefix else None, root_rank)
                return result(kind, rank, f"{REASONS[kind]} Log line: {line.strip()[:300]}")

    # 2. Worker crash: torchrun reports a failed rank, and either it was killed by a signal
    #    or no Python exception was printed (the process just disappeared).
    if root_rank is not None and root_code not in (None, 0):
        if root_signal:
            return result("WORKER_CRASH", root_rank,
                          f"Rank {root_rank} was killed by {root_signal} (exit code {root_code}).")
        if not errors:
            return result("WORKER_CRASH", root_rank,
                          f"Rank {root_rank} exited with code {root_code} without printing a Python error, "
                          f"so it stopped abruptly.")

    # 3. Timeout: the run was stopped by us and nothing above explains why.
    if timed_out:
        return result("TIMEOUT", None, "The run did not finish before the timeout and the log shows "
                                       "no known error, so it was stopped. It may be hung or just slow.")

    # 4. No rule matched: say so, and show what we saw.
    if errors:
        last = errors[-1]
        rank = _rank_from_line(last.group(0), last.group(1), root_rank)
        seen = f"Last error in log: {last.group(2)}: {last.group(3).strip()[:300]}"
    else:
        rank = root_rank
        seen = "No error message found in the log."
    return result("UNKNOWN", rank, f"No rule matched (exit code {exit_code}). {seen}")


def format_report(info):
    """Format a classify() result for printing."""
    return "\n".join([
        "FAILURE DETECTED",
        f"Type: {info['type']}",
        f"Rank: {info['rank'] if info['rank'] is not None else 'unknown'}",
        f"Reason: {info['reason']}",
        f"Suggested action: {info['suggested_action']}",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain why a training run failed.")
    parser.add_argument("log", help="path to the run's log file")
    parser.add_argument("--exit-code", type=int, default=None)
    parser.add_argument("--timed-out", action="store_true")
    args = parser.parse_args()
    with open(args.log) as f:
        print(format_report(classify(args.exit_code, f.read(), args.timed_out)))
