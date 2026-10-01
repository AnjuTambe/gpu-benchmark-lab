"""
benchmark.py - Run train.py over many settings with one command and summarize the results.

Example:
  python benchmark.py --procs 1,2 --batch-sizes 128,256 --device cpu

For every combination of (procs, batch size, sync mode) it runs train.py through
torchrun --repeats times, saves every run in a new folder results/<timestamp>/,
and writes results.csv there with one row per combination (median of the repeats).
"""

import argparse
import csv
import json
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
from datetime import datetime

import torch

CSV_COLUMNS = ["device", "procs", "batch_size", "sync_mode", "samples_per_sec", "step_p50_ms",
               "step_p95_ms", "allreduce_ms", "scaling_efficiency", "status"]


def int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def str_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run train.py over many settings.")
    parser.add_argument("--procs", type=int_list, default=[1, 2], help="process counts, e.g. 1,2")
    parser.add_argument("--batch-sizes", type=int_list, default=[256], help="per-process batch sizes, e.g. 128,256")
    parser.add_argument("--sync-modes", type=str_list, default=["sync"], help="e.g. sync,no_sync,no_ddp")
    parser.add_argument("--repeats", type=int, default=3, help="runs per combination (default: 3)")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=300, help="timed steps per run (default: 300)")
    parser.add_argument("--threads", type=int, default=2, help="OMP_NUM_THREADS per process (default: 2)")
    parser.add_argument("--timeout", type=int, default=120, help="seconds before a run counts as failed (default: 120)")
    args = parser.parse_args()
    for mode in args.sync_modes:
        if mode not in ("sync", "no_sync", "no_ddp"):
            parser.error(f"unknown sync mode: {mode}")
    return args


def resolve_device(choice, procs):
    """Turn 'auto' into one real device for the whole benchmark, so all rows are comparable."""
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if max(procs) == 1 and torch.backends.mps.is_available():
        return "mps"  # only usable when every run is single-process
    return "cpu"


def skip_reason(device, procs):
    """Return why this machine can't run a combination, or None if it can."""
    if device == "cuda":
        if not torch.cuda.is_available():
            return "no CUDA GPU available"
        if procs > torch.cuda.device_count():
            return f"needs {procs} GPUs, only {torch.cuda.device_count()} available"
    if device == "mps":
        if not torch.backends.mps.is_available():
            return "MPS not available"
        if procs > 1:
            return "MPS does not support more than 1 process"
    return None


def free_port():
    """Ask the OS for an unused port, so every run gets its own master port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_once(args, device, procs, batch_size, sync_mode, out_json, log_path):
    """Run train.py once through torchrun. Returns (result dict or None, status text)."""
    cmd = [
        sys.executable, "-m", "torch.distributed.run",  # same as `torchrun`, using this venv's Python
        "--nnodes=1", f"--nproc_per_node={procs}",
        "--master_addr=127.0.0.1", f"--master_port={free_port()}",
        "train.py",
        "--device", device, "--steps", str(args.steps),
        "--batch-size", str(batch_size), "--sync-mode", sync_mode,
        "--out", out_json,
    ]
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
    if platform.system() == "Darwin":
        env["GLOO_SOCKET_IFNAME"] = "lo0"  # macOS: avoid hostname lookup problems in gloo

    # start_new_session puts torchrun and its worker processes in their own group,
    # so on timeout we can stop all of them, not just torchrun.
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            code = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return None, f"timeout after {args.timeout}s"

    if code != 0:
        return None, f"exit code {code}"
    if not os.path.exists(out_json):
        return None, "no result file"
    with open(out_json) as f:
        return json.load(f), "ok"


def median_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value, digits):
    return "" if value is None else f"{value:.{digits}f}"


def main():
    args = parse_args()
    device = resolve_device(args.device, args.procs)
    out_dir = os.path.join("results", datetime.now().strftime("%Y%m%d-%H%M%S"))
    runs_dir = os.path.join(out_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    print(f"Device: {device}  |  OMP_NUM_THREADS: {args.threads}  |  steps: {args.steps}  "
          f"|  repeats: {args.repeats}  |  timeout: {args.timeout}s")
    print(f"Saving to {out_dir}/\n")

    rows = []
    for sync_mode in args.sync_modes:
        for batch_size in args.batch_sizes:
            for procs in args.procs:
                row = {"device": device, "procs": procs, "batch_size": batch_size, "sync_mode": sync_mode}
                name = f"{device}_p{procs}_bs{batch_size}_{sync_mode}"

                reason = skip_reason(device, procs)
                if reason:
                    print(f"SKIP {name}: {reason}")
                    rows.append({**row, "status": f"skipped: {reason}"})
                    continue

                results, failures = [], []
                for r in range(1, args.repeats + 1):
                    base = os.path.join(runs_dir, f"{name}_r{r}")
                    result, status = run_once(args, device, procs, batch_size, sync_mode,
                                              base + ".json", base + ".log")
                    if result:
                        results.append(result)
                        print(f"  {name} run {r}: {result['samples_per_sec']:.1f} samples/s, "
                              f"p50 {result['step_p50_ms']:.3f} ms")
                    else:
                        failures.append(status)
                        print(f"  {name} run {r}: FAILED ({status}), see {base}.log")

                # Median of the successful repeats. Failed repeats are counted in the status.
                row["samples_per_sec"] = median_or_none([x["samples_per_sec"] for x in results])
                row["step_p50_ms"] = median_or_none([x["step_p50_ms"] for x in results])
                row["step_p95_ms"] = median_or_none([x["step_p95_ms"] for x in results])
                row["allreduce_ms"] = median_or_none([x["comm_ms_per_allreduce"] for x in results])
                if not failures:
                    row["status"] = "ok"
                elif results:
                    row["status"] = f"partial: {len(results)}/{args.repeats} ok ({'; '.join(failures)})"
                else:
                    row["status"] = f"failed: {'; '.join(failures)}"
                rows.append(row)

    # scaling_efficiency = samples/sec / (procs x 1-process samples/sec), same device, batch size, sync mode.
    for row in rows:
        base = next((b for b in rows if b["procs"] == 1 and b["batch_size"] == row["batch_size"]
                     and b["sync_mode"] == row["sync_mode"]), None)
        if row.get("samples_per_sec") and base and base.get("samples_per_sec"):
            row["scaling_efficiency"] = row["samples_per_sec"] / (row["procs"] * base["samples_per_sec"])
        else:
            row["scaling_efficiency"] = None

    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in CSV_COLUMNS})

    # Clean table at the end.
    header = ["device", "procs", "batch", "sync_mode", "samples/s", "p50 ms", "p95 ms",
              "allreduce ms", "scaling", "status"]
    table = [[r["device"], str(r["procs"]), str(r["batch_size"]), r["sync_mode"],
              fmt(r.get("samples_per_sec"), 1), fmt(r.get("step_p50_ms"), 3), fmt(r.get("step_p95_ms"), 3),
              fmt(r.get("allreduce_ms"), 3),
              "" if r.get("scaling_efficiency") is None else f"{r['scaling_efficiency']:.1%}",
              r["status"]] for r in rows]
    widths = [max(len(h), *(len(t[i]) for t in table)) for i, h in enumerate(header)]
    print()
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for t in table:
        print("  ".join(v.ljust(w) for v, w in zip(t, widths)))
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
