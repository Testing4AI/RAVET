"""Separate candidate selection from scheduling in migration diagnostics."""

from collections import defaultdict

from sentinelrag.testing.adaptive import (
    adaptive_bayes_ordering,
    cluster_thompson_ordering,
    failure_repair_ordering,
    selected_random_ordering,
    sentinelrag_original_ordering,
    source_similarity_ordering,
)

SELECTORS = ("q0", "failure_repair", "retrieval_probability")
SCHEDULERS = ("random", "bayes", "thompson")


def select_candidates(probes, selector):
    if selector == "q0":
        selected = sentinelrag_original_ordering(probes, seed=0)
    elif selector == "failure_repair":
        selected = failure_repair_ordering(probes)
    elif selector == "retrieval_probability":
        selected = selected_random_ordering(probes, seed=0)
    elif selector == "source_similarity":
        selected = source_similarity_ordering(probes)
    else:
        raise ValueError(f"Unknown selector: {selector}")
    selected = sorted(selected, key=lambda p: p.profile.candidate.canary_id)
    ids = [p.profile.candidate.canary_id for p in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("Selection must contain exactly one probe per sentinel")
    return selected


def schedule_selected(selected, clusters, scheduler, labels, *, seed, p0, prior=0.5):
    """Existing schedulers receive only the one preselected probe per sentinel."""
    ids = {p.profile.candidate.canary_id for p in selected}
    if len(ids) != len(selected):
        raise ValueError("Scheduler must not receive alternative probes")
    if not ids <= clusters.keys():
        raise ValueError("Missing owner cluster")
    if scheduler == "random":
        return sentinelrag_original_ordering(selected, seed=seed)
    if scheduler == "static":
        return source_similarity_ordering(selected)
    feedback = defaultdict(bool, labels)
    kwargs = {
        "retained_success_by_probe": feedback,
        "removed_success_by_probe": feedback,
    }
    if scheduler == "bayes":
        return adaptive_bayes_ordering(
            selected,
            clusters,
            ids,
            prior_retention=prior,
            absent_success_probability=p0,
            **kwargs,
        )
    if scheduler == "thompson":
        return cluster_thompson_ordering(selected, clusters, ids, seed=seed, **kwargs)
    raise ValueError(f"Unknown scheduler: {scheduler}")


def next_selected_probe(selected, clusters, scheduler, prefix, *, seed, p0, prior=0.5):
    ids = [r["probe_id"] for r in prefix]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated probe in prefix")
    ordering = schedule_selected(
        selected,
        clusters,
        scheduler,
        {r["probe_id"]: r["is_correct"] for r in prefix},
        seed=seed,
        p0=p0,
        prior=prior,
    )
    ordered_ids = [p.profile.candidate.probe_id for p in ordering]
    if ordered_ids[: len(ids)] != ids:
        raise ValueError("Observed prefix differs from the frozen scheduler")
    return ordering[len(ids)] if len(ids) < len(ordering) else None
