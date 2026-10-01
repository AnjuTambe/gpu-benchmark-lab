"""
Tests for regression.compare and the gate (gate_failures), using small fake CSV data.
No torch or torchrun needed.

Run with:  pytest
"""

import csv
import io

from regression import compare, format_findings, gate_failures

HEADER = ("device,procs,batch_size,sync_mode,samples_per_sec,step_p50_ms,step_p95_ms,"
          "allreduce_ms,scaling_efficiency,status,failure_type\n")

BASELINE = HEADER + """\
cpu,1,256,sync,30000.0,7.4,10.0,,1.0,ok,
cpu,2,256,sync,20000.0,22.4,23.8,3.6,0.33,ok,
"""

OK_ROW = "cpu,1,256,sync,30000.0,7.4,10.0,,1.0,ok,\n"  # matches the baseline exactly


def rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def run_gate(current, baseline=BASELINE, tolerance=15):
    findings = compare(rows(current), rows(baseline), tolerance_pct=tolerance)
    return findings, gate_failures(findings), format_findings(findings, tolerance)


def test_pass_within_tolerance():
    findings, failures, text = run_gate(HEADER + """\
cpu,1,256,sync,27000.0,8.0,10.5,,1.0,ok,
cpu,2,256,sync,21000.0,21.0,23.0,3.5,0.39,ok,
""")
    assert [f["outcome"] for f in findings] == ["ok", "ok"]  # -10% and +5% are both allowed
    assert failures == []
    assert text.splitlines()[-1] == "No regression"


def test_regression_below_tolerance():
    findings, failures, text = run_gate(HEADER + OK_ROW + "cpu,2,256,sync,16000.0,28.0,30.0,3.6,0.27,ok,\n")
    f = findings[1]
    assert f["outcome"] == "regression"  # -20% is more than the 15% allowed
    assert f["expected_min"] == 17000.0
    assert failures == ["1 of 2 rows regressed"]
    assert "PERFORMANCE REGRESSION" in text
    assert "Config: device=cpu procs=2 batch_size=256 sync_mode=sync" in text
    assert "Expected: >= 17000.0 samples/sec" in text
    assert "Actual: 16000.0 samples/sec" in text
    assert text.splitlines()[-1].startswith("GATE FAILED")


def test_exactly_at_threshold_is_not_a_regression():
    findings, failures, _ = run_gate(HEADER + "cpu,2,256,sync,17000.0,,,,,ok,\n")
    assert findings[0]["outcome"] == "ok"
    assert failures == []


def test_missing_baseline_is_reported_but_passes_if_other_rows_checked():
    # batch 128 is not in the baseline; the batch 256 row is checked and fine.
    findings, failures, text = run_gate(HEADER + OK_ROW + "cpu,2,128,sync,15000.0,18.0,19.0,3.5,0.3,ok,\n")
    assert findings[1]["outcome"] == "no_baseline"
    assert failures == []
    assert "NOT CHECKED  (no baseline) device=cpu procs=2 batch_size=128 sync_mode=sync" in text
    assert "1 of 2 rows could not be checked" in text


def test_missing_baseline_for_every_row_fails_the_gate():
    findings, failures, _ = run_gate(HEADER + "cpu,2,128,sync,15000.0,18.0,19.0,3.5,0.3,ok,\n")
    assert findings[0]["outcome"] == "no_baseline"
    assert failures == ["no row could be checked against the baseline"]


def test_failed_run_fails_the_gate():
    findings, failures, text = run_gate(HEADER + OK_ROW + "cpu,2,256,sync,,,,,,failed: timeout after 120s,TIMEOUT\n")
    assert findings[1]["outcome"] == "failed"
    assert "TIMEOUT" in findings[1]["detail"]
    assert failures == ["1 of 2 rows failed to run"]
    assert "NOT CHECKED  (run failed)" in text


def test_partial_run_counts_as_failed():
    findings, failures, _ = run_gate(HEADER + OK_ROW +
                                     "cpu,2,256,sync,21000.0,,,,,partial: 2/3 ok (exit code 1),WORKER_CRASH\n")
    assert findings[1]["outcome"] == "failed"
    assert failures == ["1 of 2 rows failed to run"]


def test_skipped_row_does_not_fail_the_gate():
    baseline = BASELINE + "cuda,2,256,sync,90000.0,,,,,ok,\n"
    findings, failures, _ = run_gate(HEADER + OK_ROW + "cuda,2,256,sync,,,,,,skipped: no CUDA GPU available,\n",
                                     baseline=baseline)
    assert findings[1]["outcome"] == "skipped"
    assert failures == []


def test_only_skipped_rows_fails_the_gate():
    findings, failures, _ = run_gate(HEADER + "cuda,2,256,sync,,,,,,skipped: no CUDA GPU available,\n")
    assert findings[0]["outcome"] == "skipped"
    assert failures == ["no row could be checked against the baseline"]


def test_baseline_row_without_throughput():
    baseline = HEADER + "cpu,2,256,sync,,,,,,failed: exit code 1,UNKNOWN\n"
    findings, _, _ = run_gate(HEADER + "cpu,2,256,sync,20000.0,,,,,ok,\n", baseline=baseline)
    assert findings[0]["outcome"] == "no_baseline"
