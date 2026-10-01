"""
regression.py - Compare a sweep's results.csv with a saved baseline CSV.

Each row is matched to the baseline row with the same device, procs, batch_size
and sync_mode. A row regresses when its samples_per_sec is more than
tolerance_pct percent below the baseline. gate_failures() decides whether the
whole comparison passes.

Rows are only compared when both say which machine they ran on (system, cpu_name,
gpu_name) and the machines match, so e.g. Mac and Kaggle results are never mixed.

This file does not import torch, so it can be tested without running anything.
"""

import csv

KEY_COLUMNS = ("device", "procs", "batch_size", "sync_mode")
MACHINE_COLUMNS = ("system", "cpu_name", "gpu_name")  # must match between a row and its baseline


def load_csv(path):
    """Read a results.csv file into a list of dicts (all values are strings)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def row_key(row):
    """The config that identifies a row, e.g. ('cpu', 2, 256, 'sync')."""
    return (row["device"], int(row["procs"]), int(row["batch_size"]), row["sync_mode"])


def describe(key):
    return " ".join(f"{name}={value}" for name, value in zip(KEY_COLUMNS, key))


def throughput(row):
    """samples_per_sec as a float, or None if the row has no number (failed or skipped run)."""
    value = (row.get("samples_per_sec") or "").strip()
    return float(value) if value else None


def machine(row):
    """The machine identity of a row, or None if the row has no hardware info."""
    values = tuple((row.get(c) or "").strip() for c in MACHINE_COLUMNS)
    return values if any(values) else None


def compare(current_rows, baseline_rows, tolerance_pct):
    """
    Compare every current row with its baseline row.
    Returns one dict per current row with:
      config, outcome, actual, baseline, expected_min, change_pct, detail
    outcome is one of:
      "ok"           within tolerance
      "regression"   more than tolerance_pct percent below the baseline
      "failed"       this run did not finish cleanly (status is not "ok"), so it can't be checked
      "skipped"      this config was skipped on this machine
      "no_baseline"  no usable baseline row for this config
      "hardware_mismatch"  the baseline was measured on a different (or unknown) machine
    """
    baseline = {row_key(row): row for row in baseline_rows}
    findings = []
    for row in current_rows:
        key = row_key(row)
        status = (row.get("status") or "").strip()
        actual = throughput(row)
        base_row = baseline.get(key)
        base = throughput(base_row) if base_row else None
        finding = {"config": describe(key), "actual": actual, "baseline": base,
                   "expected_min": None, "change_pct": None, "detail": ""}

        if status.startswith("skipped"):
            finding.update(outcome="skipped", detail=status)
        elif status != "ok" or actual is None:
            failure = (row.get("failure_type") or "").strip()
            finding.update(outcome="failed", detail=status + (f" ({failure})" if failure else ""))
        elif base_row is None:
            finding.update(outcome="no_baseline", detail="no row with this config in the baseline")
        elif base is None:
            finding.update(outcome="no_baseline",
                           detail=f"baseline row has no throughput (baseline status: {base_row.get('status')})")
        elif machine(row) is None or machine(base_row) is None:
            finding.update(outcome="hardware_mismatch",
                           detail="hardware unknown for " + ("this run" if machine(row) is None else "the baseline")
                                  + "; re-create the baseline on this machine")
        elif machine(row) != machine(base_row):
            finding.update(outcome="hardware_mismatch",
                           detail=f"this run: {' / '.join(machine(row))}; baseline: {' / '.join(machine(base_row))}")
        else:
            expected_min = base * (1 - tolerance_pct / 100)
            finding.update(expected_min=expected_min, change_pct=(actual - base) / base * 100,
                           outcome="regression" if actual < expected_min else "ok")
        findings.append(finding)
    return findings


def format_findings(findings, tolerance_pct):
    """Turn compare() results into the text to print."""
    lines = []
    for f in findings:
        if f["outcome"] == "regression":
            lines += ["PERFORMANCE REGRESSION",
                      f"Config: {f['config']}",
                      f"Expected: >= {f['expected_min']:.1f} samples/sec "
                      f"(baseline {f['baseline']:.1f}, tolerance {tolerance_pct:g}%)",
                      f"Actual: {f['actual']:.1f} samples/sec ({f['change_pct']:+.1f}% vs baseline)",
                      ""]
        elif f["outcome"] == "ok":
            lines.append(f"OK           {f['config']}: {f['actual']:.1f} samples/sec "
                         f"({f['change_pct']:+.1f}% vs baseline {f['baseline']:.1f})")
        else:
            label = {"failed": "NOT CHECKED  (run failed)", "skipped": "NOT CHECKED  (skipped)",
                     "no_baseline": "NOT CHECKED  (no baseline)",
                     "hardware_mismatch": "NOT CHECKED  (different hardware)"}[f["outcome"]]
            lines.append(f"{label} {f['config']}: {f['detail']}")

    failures = gate_failures(findings)
    if failures:
        lines.append("GATE FAILED: " + "; ".join(failures))
    else:
        unchecked = sum(f["outcome"] in ("skipped", "no_baseline", "hardware_mismatch") for f in findings)
        lines.append("No regression" + (f" ({unchecked} of {len(findings)} rows could not be checked, see above)"
                                         if unchecked else ""))
    return "\n".join(lines)


def gate_failures(findings):
    """
    Reasons the gate fails; an empty list means it passes. The gate fails when:
      - any row regressed,
      - any row's run failed (a crash must not pass as "no regression"),
      - no row could be checked at all (nothing was actually compared).
    Skipped rows (the machine cannot run that config), rows without a baseline and
    rows whose baseline came from different hardware are reported but do not fail
    the gate on their own.
    """
    reasons = []
    regressed = sum(f["outcome"] == "regression" for f in findings)
    failed = sum(f["outcome"] == "failed" for f in findings)
    checked = sum(f["outcome"] in ("ok", "regression") for f in findings)
    if regressed:
        reasons.append(f"{regressed} of {len(findings)} rows regressed")
    if failed:
        reasons.append(f"{failed} of {len(findings)} rows failed to run")
    if checked == 0:
        reasons.append("no row could be checked against the baseline")
    return reasons
