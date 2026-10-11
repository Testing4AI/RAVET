#!/usr/bin/env python3
"""Offline, outcome-preserving replay. Python standard library only."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def e_value(successes, trials, p0):
    if not 0 < p0 < 1 or not 0 <= successes <= trials:
        raise ValueError("Invalid Bernoulli counts or background bound")
    epsilon = min(1e-6, (1 - p0) / 201)
    start, end = p0 + epsilon, 1 - epsilon
    logs = []
    for i in range(200):
        theta = start + (end - start) * i / 199
        logs.append(successes * math.log(theta / p0)
                    + (trials - successes) * math.log((1 - theta) / (1 - p0)))
    maximum = max(logs)
    return math.exp(min(700, maximum + math.log(sum(math.exp(x - maximum) for x in logs) / 200)))


def replay(row):
    labels, p0, alpha = row["labels"], row["p0"], row["alpha"]
    if not labels or any(type(x) is not bool for x in labels):
        raise ValueError("A trajectory needs boolean oracle labels")
    if len(labels) != len(row["unit_order"]) or len(set(row["unit_order"])) != len(labels):
        raise ValueError("Every observation must have a distinct evidence identity")
    total, crossed, values = 0, False, []
    for t, x in enumerate(labels, 1):
        total += x
        crossed = crossed or e_value(total, t, p0) >= 1 / alpha
        values.append(int(crossed))
    return values


def condition_rows(traces):
    grouped = defaultdict(list)
    for row in traces:
        curve = replay(row)
        if curve != row["archived_curve"]:
            raise ValueError("Replay disagrees with archived crossing: " + row["source_id"])
        grouped[(row["study"], row["condition_id"], row["method"])].append((row, curve))
    result = []
    for (study, identity, method), group in sorted(grouped.items()):
        source = group[0][0]
        metadata = ("scenario", "dataset", "density", "retriever", "target", "state", "replicate",
                    "expansion_kind", "expansion_factor", "p0", "alpha")
        for row, curve in group:
            if any(row[k] != source[k] for k in metadata):
                raise ValueError("Conflicting metadata in a configuration")
        n = len(group[0][1])
        if any(len(curve) != n for _, curve in group):
            raise ValueError("Different observation windows in one configuration")
        result.append({"study": study, "condition_id": identity, "method": method,
                       **{k: source[k] for k in metadata}, "ordering_replicates": len(group),
                       "curve": [sum(curve[t] for _, curve in group) / len(group) for t in range(n)]})
    methods = defaultdict(set)
    for r in result:
        methods[(r["study"], r["condition_id"])].add(r["method"])
    if any(m != {"SentinelRAG", "RAVET"} for m in methods.values()):
        raise ValueError("An unpaired configuration was exported")
    return result


def summarize(rows, budget):
    if not rows or any(len(r["curve"]) < budget for r in rows):
        raise ValueError("Budget exceeds archived observation window")
    n = len(rows)
    vsr = sum(r["curve"][budget - 1] for r in rows) / n
    mq = sum(sum(1 - (r["curve"][t - 1] if t else 0) for t in range(budget)) for r in rows) / n
    return {"conditions": n, "vsr": vsr, "mq": mq}


def verify_files(root):
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for record in manifest["files"]:
        path = root / record["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Integrity mismatch: " + record["path"])
    return len(manifest["files"])


def report(rows):
    cells = []
    keys = sorted({(r["study"], r["scenario"], r["state"]) for r in rows})
    for study, scenario, state in keys:
        for method in ("SentinelRAG", "RAVET"):
            subset = [r for r in rows if (r["study"], r["scenario"], r["state"], r["method"])
                      == (study, scenario, state, method)]
            for budget in (6, 15, 30):
                if budget <= min(len(r["curve"]) for r in subset):
                    cells.append({"study": study, "scenario": scenario, "state": state,
                                  "method": method, "budget": budget, **summarize(subset, budget)})
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Check packaged file hashes before replay")
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced")
    args = parser.parse_args()
    if args.verify:
        print(f"Integrity: {verify_files(ROOT)} packaged files match")
    payload = json.loads((ROOT / "results/trajectories.json").read_text())
    rows = condition_rows(payload["traces"])
    expected = json.loads((ROOT / "results/conditions.json").read_text())
    if rows != expected:
        raise ValueError("Condition-level aggregation differs from the public snapshot")
    cells = report(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "conditions.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.output / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    print(f"Replay: {len(payload['traces'])} archived trajectories; {len(rows)} method-configuration rows")
    print("Primary study (144 paired protected configurations; p0=0.15)")
    for method in ("SentinelRAG", "RAVET"):
        subset = [r for r in rows if r["study"] == "primary" and r["state"] == "watermarked" and r["method"] == method]
        a, b = summarize(subset, 6), summarize(subset, 15)
        print(f"  {method:12} VSR(6)={a['vsr']:.1%}  VSR(15)={b['vsr']:.1%}  MQ(15)={b['mq']:.2f}")
    print("No network requests or new model judgments. Output:", args.output)


if __name__ == "__main__":
    main()
