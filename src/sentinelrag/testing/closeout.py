"""Dependency-aware diagnostics for frozen ownership-verification studies."""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from sentinelrag.testing.ragwm import evidence_curve


def static_initial_order(selected, prior=0.5):
    """Freeze the adaptive policy's initial utility without posterior updates."""
    ids = [p.profile.candidate.canary_id for p in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Static control requires one selected test per sentinel")
    return sorted(selected, key=lambda p: (
        prior * p.retained_success_probability,
        p.profile.proposed_score,
        p.profile.candidate.probe_id,
    ), reverse=True)


def cumulative_metrics(labels, p0, alpha=0.01, cap=15):
    if len(labels) < cap or any(type(x) is not bool for x in labels):
        raise ValueError("Complete binary labels are required")
    trace = evidence_curve(labels[:cap], p0, alpha)
    first = trace["first_anytime"]
    return {
        "vsr6": float(first is not None and first <= min(6, cap)),
        "vsr15": float(first is not None),
        "mq15": float(first if first is not None else cap),
        "first_crossing": first,
    }


def paired_point(rows):
    if not rows:
        raise ValueError("Empty paired comparison")
    result = {}
    for metric in ("vsr6", "vsr15", "mq15"):
        a = float(np.mean([r["baseline"][metric] for r in rows]))
        b = float(np.mean([r["candidate"][metric] for r in rows]))
        result[metric] = {"baseline": a, "candidate": b, "difference": b - a}
    result["query_savings"] = (
        1 - result["mq15"]["candidate"] / result["mq15"]["baseline"]
    )
    result["conditions"] = len(rows)
    return result


def draw_sensitivity(rows):
    """Enumerate a small stratified bootstrap without splitting dependent cells."""
    draws = defaultdict(list)
    strata = defaultdict(set)
    for row in rows:
        draw = row["block"], row["replicate_seed"]
        draws[draw].append(row)
        strata[row["block"]].add(draw)
    if len(strata) != 2 or sorted(len(x) for x in strata.values()) != [3, 3]:
        raise ValueError("Expected two corpora with three draws each")
    sizes = {len(x) for x in draws.values()}
    if len(sizes) != 1:
        raise ValueError("Unbalanced draw sizes")
    choices = [list(itertools.product(sorted(s), repeat=len(s)))
               for _, s in sorted(strata.items())]
    samples = []
    for choices_by_stratum in itertools.product(*choices):
        sample = [r for choice in choices_by_stratum for draw in choice for r in draws[draw]]
        samples.append(paired_point(sample))
    intervals = {}
    for metric in ("vsr6", "vsr15", "mq15"):
        values = [x[metric]["difference"] for x in samples]
        intervals[metric] = np.quantile(values, [0.025, 0.975]).tolist()
    intervals["query_savings"] = np.quantile(
        [x["query_savings"] for x in samples], [0.025, 0.975]
    ).tolist()
    return {
        "draw_count": len(draws),
        "bootstrap_enumerations": len(samples),
        "point": paired_point(rows),
        "descriptive_percentile_intervals": intervals,
        "by_draw": [{"block": k[0], "replicate_seed": k[1], **paired_point(v)}
                    for k, v in sorted(draws.items())],
        "leave_one_draw_out": [
            {"omitted_draw": list(k), **paired_point(
                [r for other, values in draws.items() if other != k for r in values]
            )} for k in sorted(draws)
        ],
        "interpretation": "Conditional dependence-aware sensitivity, not an independent-corpus population confidence guarantee",
    }
