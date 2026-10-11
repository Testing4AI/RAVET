#!/usr/bin/env python3
"""Export audited retrieval-change results without altering paper evidence."""

import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import run_retrieval_deployment_changes as study
from run_partition_confirmation import file_hash, read, save


ROOT, OUT = study.ROOT, study.OUT
PAIR_KEYS = ("block", "retriever", "model")
METRICS = study.METRICS
TEST_FILES = (
    "test_closeout.py", "test_closeout_extensions.py", "test_closeout_repetition.py",
    "test_summary_extension.py", "test_orthogonal_policy.py", "test_sentinel_paraphrase.py",
    "test_corpus_retrieval.py", "test_density_curve.py", "test_adaptive_theft.py",
    "test_adaptive_clean_null.py", "test_clean_control.py", "test_migration_confirmation.py",
    "test_diagnostic_request_cache.py", "test_model_oracle_ensemble.py",
    "test_representation_attacks.py", "test_retrieval_deployment.py",
    "test_retrieval_request_audit.py",
)


def csv_export(name, rows):
    assert rows and all(row.keys() == rows[0].keys() for row in rows)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first_crossing_metrics(curve):
    crosses = [row["budget"] for row in curve if row["anytime_verified"]]
    first = min(crosses, default=31)
    return {**{f"vsr{b}": float(first <= b) for b in (6, 15, 30)},
            **{f"mq{b}": float(min(first, b)) for b in (15, 30)}}


def paired_comparisons(conditions, scenarios):
    comparisons = []
    for scenario in scenarios:
        index = {(r["method"], *(r[k] for k in PAIR_KEYS)): r for r in conditions
                 if r["scenario"] == scenario and r["state"] == "watermarked"}
        keys = sorted({key[1:] for key in index})
        assert len(keys) == 8
        for key in keys:
            baseline, ours = (index[(method, *key)] for method in ("SentinelRAG", "RAVET"))
            row = {"scenario": scenario, **dict(zip(PAIR_KEYS, key, strict=True))}
            for metric in METRICS:
                row[f"sentinelrag_{metric}"] = baseline[metric]
                row[f"ravet_{metric}"] = ours[metric]
                row[f"delta_{metric}"] = ours[metric] - baseline[metric]
            comparisons.append(row)
    return comparisons


def summarize_pairs(pairs, scenarios):
    summary = []
    for scenario, metric in itertools.product(scenarios, METRICS):
        rows = [r for r in pairs if r["scenario"] == scenario]
        delta = np.array([r[f"delta_{metric}"] for r in rows])
        benefit = delta if metric.startswith("vsr") else -delta
        summary.append({"scenario": scenario, "metric": metric, "paired_configurations": len(rows),
                        "ours_better": int(np.sum(benefit > 1e-10)),
                        "equal": int(np.sum(np.abs(benefit) <= 1e-10)),
                        "ours_worse": int(np.sum(benefit < -1e-10)),
                        "mean_delta": float(np.mean(delta))})
    return summary


def main():
    protocol, _, lock = study.freeze()
    result, trajectories = read(OUT / "results.json"), read(OUT / "trajectories.json")
    provenance, rankings = read(OUT / "provenance_audit_v2.json"), read(OUT / "ranking_audit.json")
    assert result["complete"] and provenance["status"] == rankings["status"] == "pass"
    assert result["logical_answer_records"] == provenance["checked_answer_records"] == 2880
    assert rankings["reconstructed_contexts"] == 1440

    # Check first-crossing metrics directly, not through the report helper.
    replayed = {}
    for trace in trajectories:
        metrics = first_crossing_metrics(trace["curve"])
        key = (trace["scenario"], trace["state"], trace["method"], *(trace[k] for k in PAIR_KEYS))
        replayed.setdefault(key, []).append(metrics)
    for row in result["conditions"]:
        key = (row["scenario"], row["state"], row["method"], *(row[k] for k in PAIR_KEYS))
        assert all(np.isclose(row[metric], np.mean([r[metric] for r in replayed[key]]), atol=1e-12)
                   for metric in METRICS)

    pairs = paired_comparisons(result["conditions"], protocol["scenarios"])
    pair_summary = summarize_pairs(pairs, protocol["scenarios"])
    reference_deltas = []
    for method, state in itertools.product(protocol["methods"], ("watermarked", "clean")):
        rows = [r for r in result["conditions"] if (r["method"], r["state"]) == (method, state)]
        reference = {tuple(r[k] for k in PAIR_KEYS): r for r in rows if r["scenario"] == "top5_reference"}
        for row in rows:
            if row["scenario"] == "top5_reference":
                continue
            before = reference[tuple(row[k] for k in PAIR_KEYS)]
            reference_deltas.append({"scenario": row["scenario"], "state": state, "method": method,
                **{k: row[k] for k in PAIR_KEYS},
                **{f"delta_{m}": row[m] - before[m] for m in METRICS},
                "delta_answer_positive_count": row["answer_positive_count"] - before["answer_positive_count"],
                "delta_own_carrier_retrieval_fraction": row["own_carrier_retrieval_fraction"] - before["own_carrier_retrieval_fraction"],
                "delta_own_literal_support_fraction": row["own_literal_support_fraction"] - before["own_literal_support_fraction"]})

    analysis = {"complete": True, "pair_keys": PAIR_KEYS,
                "configurations_per_scenario_and_method": 8,
                "baseline_policy_seeds_averaged_within_configuration": 3,
                "metrics_independently_reproduced": True, "paired_comparisons": pairs,
                "paired_direction_summary": pair_summary, "changes_from_top5_reference": reference_deltas,
                "new_cohort_not_pooled_with_144_conditions": True, **study.verify(lock)}
    save(OUT / "paired_analysis.json", analysis)
    csv_export("summary.csv", result["summary"])
    csv_export("conditions.csv", result["conditions"])
    csv_export("paired_comparisons.csv", pairs)
    csv_export("paired_direction_summary.csv", pair_summary)
    csv_export("changes_from_top5_reference.csv", reference_deltas)

    command = [sys.executable, "-m", "pytest", *[f"tests/{name}" for name in TEST_FILES], "-q"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    save(OUT / "test_receipt.json", {"command": command, "exit_code": completed.returncode,
         "stdout": completed.stdout, "stderr": completed.stderr})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt_paths = [OUT / name for name in (
        "protocol_lock.json", "frozen_contexts.json", "execution_status.json", "results.json",
        "trajectories.json", "provenance_audit_v2.json", "ranking_audit.json", "paired_analysis.json",
        "summary.csv", "conditions.csv", "paired_comparisons.csv", "paired_direction_summary.csv",
        "changes_from_top5_reference.csv", "test_receipt.json",
    )]
    receipt_paths.extend((Path(__file__).resolve(), ROOT / "scripts/audit_retrieval_deployment_rankings.py",
                          ROOT / "scripts/audit_retrieval_deployment_requests_v2.py"))
    report = ROOT / "docs/RETRIEVAL_DEPLOYMENT_CHANGES_20261003.zh-CN.md"
    if report.exists():
        receipt_paths.append(report)
    receipt = {"status": "complete", "logical_answer_records": 2880,
               "new_setting_logical_records": 1920, "ranked_contexts_audited": 1440,
               "independent_first_crossing_reproduction": True,
               "execution": read(OUT / "execution_status.json"),
               "fresh_unique_requests_by_preset": provenance["fresh_unique_execution_requests_by_preset"],
               "tests": {"exit_code": completed.returncode, "summary": completed.stdout.strip().splitlines()[-1]},
               "files_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in receipt_paths},
               **study.verify(lock)}
    save(OUT / "completion_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "paired_direction_summary": pair_summary,
                      "tests": receipt["tests"]}), flush=True)


if __name__ == "__main__":
    main()
