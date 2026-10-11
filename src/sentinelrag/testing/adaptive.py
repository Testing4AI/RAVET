"""Cluster-correlated theft simulation and adaptive probe prioritization."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from sentinelrag.testing.artifacts import CanaryArtifact
from sentinelrag.testing.matrix import (
    CALIBRATION_VARIANTS,
    HELDOUT_VARIANT,
    RetrievalScoreCache,
    observe_candidate,
)
from sentinelrag.testing.prioritization import (
    ProbeProfile,
    rank_probes,
    score_profiles,
)


@dataclass(frozen=True)
class PreparedProbe:
    """A scored probe and its held-out retrieval outcome."""

    profile: ProbeProfile
    heldout_success: bool
    retained_success_probability: float


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    # NumPy/scikit-learn RandomState consumers require an unsigned 32-bit seed.
    return int.from_bytes(digest[:4], "big")


def cluster_artifacts(
    artifacts: Sequence[CanaryArtifact],
    *,
    num_clusters: int = 6,
    seed: int = 20260820,
) -> dict[str, int]:
    """Cluster complete artifacts using deterministic owner-side text features."""

    if not artifacts:
        return {}
    cluster_count = min(num_clusters, len(artifacts))
    texts = ["\n".join(artifact.documents) for artifact in artifacts]
    matrix = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=20_000,
    ).fit_transform(texts)
    if matrix.shape[1] > 2 and len(artifacts) > 2:
        components = min(32, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if components >= 2:
            matrix = TruncatedSVD(
                n_components=components, random_state=seed
            ).fit_transform(matrix)
            matrix = normalize(matrix)
    labels = KMeans(
        n_clusters=cluster_count,
        n_init=20,
        random_state=seed,
    ).fit_predict(matrix)
    return {
        artifact.canary_id: int(label)
        for artifact, label in zip(artifacts, labels)
    }


def prepare_probes(
    cache: RetrievalScoreCache,
    artifacts: Sequence[CanaryArtifact],
    *,
    top_k: int,
) -> list[PreparedProbe]:
    """Build frozen calibration profiles and held-out retrieval labels."""

    selected_ids = {artifact.canary_id for artifact in artifacts}
    profiles: list[ProbeProfile] = []
    heldout: list[bool] = []
    retained_probabilities: list[float] = []
    for candidate_index, candidate in enumerate(cache.candidates):
        observations = [
            observe_candidate(
                cache,
                candidate_index=candidate_index,
                selected_canary_ids=selected_ids,
                variant_name=variant,
            )
            for variant in CALIBRATION_VARIANTS
        ]
        profile = ProbeProfile(candidate=candidate, observations=observations)
        profiles.append(profile)
        calibration_hits = sum(
            observation.target_rank is not None
            and observation.target_rank <= top_k
            for observation in observations
        )
        # Beta(1, 1) smoothing keeps one failed query from collapsing a
        # cluster's posterior to zero in the adaptive policy.
        retained_probabilities.append(
            (calibration_hits + 1.0) / (len(observations) + 2.0)
        )
        heldout_observation = observe_candidate(
            cache,
            candidate_index=candidate_index,
            selected_canary_ids=selected_ids,
            variant_name=HELDOUT_VARIANT,
        )
        heldout.append(
            heldout_observation.target_rank is not None
            and heldout_observation.target_rank <= top_k
        )
    score_profiles(profiles)
    return [
        PreparedProbe(profile, success, probability)
        for profile, success, probability in zip(
            profiles, heldout, retained_probabilities
        )
    ]


def _prepared_by_probe(
    probes: Sequence[PreparedProbe],
) -> dict[str, PreparedProbe]:
    return {probe.profile.candidate.probe_id: probe for probe in probes}


def _static_ordering(
    probes: Sequence[PreparedProbe], method: str, *, seed: int
) -> list[PreparedProbe]:
    by_id = _prepared_by_probe(probes)
    profiles = [probe.profile for probe in probes]
    if method == "random":
        ordering = rank_probes(profiles, "random", seed=seed)
    elif method == "similarity":
        ordering = rank_probes(profiles, "similarity")
    elif method == "composite_v1":
        ordering = rank_probes(profiles, "proposed")
    else:
        raise ValueError(f"Unknown static policy: {method}")
    return [by_id[profile.candidate.probe_id] for profile in ordering]


def sentinelrag_original_ordering(
    probes: Sequence[PreparedProbe], *, seed: int
) -> list[PreparedProbe]:
    """Reproduce SentinelRAG's one-pre-generated-probe-per-sentinel policy."""

    first_by_canary: dict[str, PreparedProbe] = {}
    for probe in probes:
        candidate = probe.profile.candidate
        metadata = getattr(candidate, "metadata", {})
        question_index = int(metadata.get("question_index", 0))
        incumbent = first_by_canary.get(candidate.canary_id)
        if incumbent is None:
            first_by_canary[candidate.canary_id] = probe
            continue
        incumbent_metadata = getattr(incumbent.profile.candidate, "metadata", {})
        incumbent_key = (
            int(incumbent_metadata.get("question_index", 0)),
            incumbent.profile.candidate.probe_id,
        )
        if (question_index, candidate.probe_id) < incumbent_key:
            first_by_canary[candidate.canary_id] = probe

    ordering = list(first_by_canary.values())
    random.Random(seed).shuffle(ordering)
    return ordering


def selected_random_ordering(
    probes: Sequence[PreparedProbe], *, seed: int
) -> list[PreparedProbe]:
    """Select the strongest candidate per sentinel, then randomize sentinels."""

    ordering = list(
        _best_by_canary(probes, score="retained_probability").values()
    )
    random.Random(seed).shuffle(ordering)
    return ordering


def random_candidate_ordering(
    probes: Sequence[PreparedProbe], *, seed: int
) -> list[PreparedProbe]:
    """Select one uniformly randomized candidate per sentinel and randomize order."""

    return _static_ordering(probes, "random", seed=seed)


def source_similarity_ordering(
    probes: Sequence[PreparedProbe],
) -> list[PreparedProbe]:
    """Select and order probes using owner-side source similarity only."""

    return _static_ordering(probes, "similarity", seed=0)


def _best_by_canary(
    probes: Sequence[PreparedProbe],
    *,
    score: str,
) -> dict[str, PreparedProbe]:
    selected: dict[str, PreparedProbe] = {}
    for probe in probes:
        canary_id = probe.profile.candidate.canary_id
        if score == "proposed":
            value = probe.profile.proposed_score
        elif score == "retained_probability":
            value = probe.retained_success_probability
        else:
            raise ValueError(f"Unknown probe score: {score}")
        incumbent = selected.get(canary_id)
        if incumbent is None:
            selected[canary_id] = probe
            continue
        incumbent_value = (
            incumbent.profile.proposed_score
            if score == "proposed"
            else incumbent.retained_success_probability
        )
        if (value, probe.profile.proposed_score, probe.profile.candidate.probe_id) > (
            incumbent_value,
            incumbent.profile.proposed_score,
            incumbent.profile.candidate.probe_id,
        ):
            selected[canary_id] = probe
    return selected


def failure_repair_ordering(
    probes: Sequence[PreparedProbe], *, seed: int | None = None
) -> list[PreparedProbe]:
    """Select only substantial shadow-failure repairs, with optional random order."""
    by_id = _prepared_by_probe(probes)
    selected = [
        by_id[profile.candidate.probe_id]
        for profile in rank_probes(
            [probe.profile for probe in probes], "failure_repair"
        )
    ]
    if seed is not None:
        random.Random(seed).shuffle(selected)
    return selected


def cluster_diverse_ordering(
    probes: Sequence[PreparedProbe],
    cluster_by_canary: Mapping[str, int],
) -> list[PreparedProbe]:
    """Round-robin clusters while keeping the strongest probe inside each."""

    selected = _best_by_canary(probes, score="proposed")
    queues: dict[int, list[PreparedProbe]] = defaultdict(list)
    for canary_id, probe in selected.items():
        queues[cluster_by_canary[canary_id]].append(probe)
    for queue in queues.values():
        queue.sort(
            key=lambda probe: (
                probe.profile.proposed_score,
                probe.profile.candidate.probe_id,
            ),
            reverse=True,
        )
    cluster_order = sorted(
        queues,
        key=lambda cluster: (
            queues[cluster][0].profile.proposed_score,
            -cluster,
        ),
        reverse=True,
    )
    ordering = []
    while any(queues.values()):
        for cluster in cluster_order:
            if queues[cluster]:
                ordering.append(queues[cluster].pop(0))
    return ordering


def _is_success(
    probe: PreparedProbe,
    retained_canaries: set[str],
    *,
    retained_success_by_probe: Mapping[str, bool] | None = None,
    removed_success_by_probe: Mapping[str, bool] | None = None,
) -> bool:
    candidate = probe.profile.candidate
    if candidate.canary_id in retained_canaries:
        if retained_success_by_probe is not None:
            return bool(retained_success_by_probe[candidate.probe_id])
        return probe.heldout_success
    if removed_success_by_probe is not None:
        return bool(removed_success_by_probe[candidate.probe_id])
    return False


def adaptive_bayes_ordering(
    probes: Sequence[PreparedProbe],
    cluster_by_canary: Mapping[str, int],
    retained_canaries: set[str],
    *,
    prior_retention: float,
    retained_success_by_probe: Mapping[str, bool] | None = None,
    removed_success_by_probe: Mapping[str, bool] | None = None,
    absent_success_probability: float = 0.0,
    probe_selector: str = "retained_probability",
) -> list[PreparedProbe]:
    """Choose probes online using a posterior over retained topic clusters."""

    if probe_selector == "retained_probability":
        remaining = _best_by_canary(probes, score="retained_probability")
    elif probe_selector == "first":
        remaining = {
            probe.profile.candidate.canary_id: probe
            for probe in sentinelrag_original_ordering(probes, seed=0)
        }
    elif probe_selector == "failure_repair":
        remaining = {
            probe.profile.candidate.canary_id: probe
            for probe in failure_repair_ordering(probes)
        }
    else:
        raise ValueError(f"Unknown probe selector: {probe_selector}")
    clusters = sorted(set(cluster_by_canary.values()))
    posterior = {cluster: float(prior_retention) for cluster in clusters}
    ordering: list[PreparedProbe] = []
    while remaining:
        def utility(item: tuple[str, PreparedProbe]):
            canary_id, probe = item
            cluster = cluster_by_canary[canary_id]
            return (
                posterior[cluster] * probe.retained_success_probability,
                probe.profile.proposed_score,
                probe.profile.candidate.probe_id,
            )

        canary_id, probe = max(remaining.items(), key=utility)
        del remaining[canary_id]
        ordering.append(probe)
        cluster = cluster_by_canary[canary_id]
        probability = probe.retained_success_probability
        success = _is_success(
            probe,
            retained_canaries,
            retained_success_by_probe=retained_success_by_probe,
            removed_success_by_probe=removed_success_by_probe,
        )
        current = posterior[cluster]
        if success:
            numerator = current * probability
            denominator = numerator + (
                (1.0 - current) * absent_success_probability
            )
        else:
            numerator = current * (1.0 - probability)
            denominator = numerator + (
                (1.0 - current) * (1.0 - absent_success_probability)
            )
        posterior[cluster] = (
            numerator / denominator if denominator > 1e-12 else 0.0
        )
    return ordering


def cluster_thompson_ordering(
    probes: Sequence[PreparedProbe],
    cluster_by_canary: Mapping[str, int],
    retained_canaries: set[str],
    *,
    seed: int,
    retained_success_by_probe: Mapping[str, bool] | None = None,
    removed_success_by_probe: Mapping[str, bool] | None = None,
) -> list[PreparedProbe]:
    """Generic cluster bandit using only observed binary verification rewards."""

    selected = _best_by_canary(probes, score="retained_probability")
    queues: dict[int, list[PreparedProbe]] = defaultdict(list)
    for canary_id, probe in selected.items():
        queues[cluster_by_canary[canary_id]].append(probe)
    for queue in queues.values():
        queue.sort(
            key=lambda probe: (
                probe.retained_success_probability,
                probe.profile.proposed_score,
                probe.profile.candidate.probe_id,
            ),
            reverse=True,
        )

    generator = np.random.default_rng(seed)
    alpha = {cluster: 1.0 for cluster in queues}
    beta = {cluster: 1.0 for cluster in queues}
    ordering = []
    while any(queues.values()):
        available = [cluster for cluster, queue in queues.items() if queue]
        sampled = {
            cluster: float(generator.beta(alpha[cluster], beta[cluster]))
            for cluster in available
        }
        cluster = max(available, key=lambda value: (sampled[value], -value))
        probe = queues[cluster].pop(0)
        ordering.append(probe)
        success = _is_success(
            probe,
            retained_canaries,
            retained_success_by_probe=retained_success_by_probe,
            removed_success_by_probe=removed_success_by_probe,
        )
        alpha[cluster] += float(success)
        beta[cluster] += float(not success)
    return ordering


def evaluate_policy(
    ordering: Sequence[PreparedProbe],
    retained_canaries: set[str],
    cluster_by_canary: Mapping[str, int],
    *,
    budgets: Sequence[int],
    retained_success_by_probe: Mapping[str, bool] | None = None,
    removed_success_by_probe: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Evaluate one policy without assuming within-cluster independence."""

    curve = []
    successes = 0
    first_success = None
    cumulative_detection = 0.0
    discovered_clusters: set[int] = set()
    for query_index, probe in enumerate(ordering, start=1):
        canary_id = probe.profile.candidate.canary_id
        cluster = cluster_by_canary[canary_id]
        success = _is_success(
            probe,
            retained_canaries,
            retained_success_by_probe=retained_success_by_probe,
            removed_success_by_probe=removed_success_by_probe,
        )
        successes += int(success)
        if success:
            discovered_clusters.add(cluster)
            if first_success is None:
                first_success = query_index
        cumulative_detection += successes / query_index
        curve.append(
            {
                "query": query_index,
                "probe_id": probe.profile.candidate.probe_id,
                "canary_id": canary_id,
                "cluster": cluster,
                "success": success,
                "cumulative_successes": successes,
                "discovered_clusters": len(discovered_clusters),
            }
        )
    budget_metrics = {}
    for budget in budgets:
        usable = min(budget, len(curve))
        row = curve[usable - 1] if usable else None
        budget_metrics[str(budget)] = {
            "successes": 0 if row is None else row["cumulative_successes"],
            "success_rate": (
                0.0 if row is None else row["cumulative_successes"] / usable
            ),
            "retained_cluster_discovery": (
                0 if row is None else row["discovered_clusters"]
            ),
        }
    return {
        "num_queries": len(curve),
        "total_successes": successes,
        "final_success_rate": successes / len(curve) if curve else 0.0,
        "first_success_queries": first_success,
        "normalized_cumulative_detection": (
            cumulative_detection / len(curve) if curve else 0.0
        ),
        "budgets": budget_metrics,
        "curve": curve,
    }


def simulate_attack(
    probes: Sequence[PreparedProbe],
    cluster_by_canary: Mapping[str, int],
    *,
    retained_cluster_count: int,
    attack_seed: int,
    budgets: Sequence[int] = (5, 10, 20, 60),
    scenario: str = "cluster",
    retained_success_by_probe: Mapping[str, bool] | None = None,
    removed_success_by_probe: Mapping[str, bool] | None = None,
    absent_success_probability: float = 0.0,
) -> dict[str, Any]:
    """Run all policies against one paired partial-theft realization."""

    clusters = sorted(set(cluster_by_canary.values()))
    randomizer = random.Random(attack_seed)
    if scenario == "cluster":
        retained_clusters = set(
            randomizer.sample(clusters, min(retained_cluster_count, len(clusters)))
        )
        retained_canaries = {
            canary_id
            for canary_id, cluster in cluster_by_canary.items()
            if cluster in retained_clusters
        }
    elif scenario == "artifact_random":
        retained_clusters = set()
        canary_ids = sorted(cluster_by_canary)
        retained_count = round(
            len(canary_ids) * retained_cluster_count / len(clusters)
        )
        retained_canaries = set(randomizer.sample(canary_ids, retained_count))
    else:
        raise ValueError(f"Unknown attack scenario: {scenario}")

    orderings = {
        "sentinelrag_original": sentinelrag_original_ordering(
            probes, seed=stable_seed("sentinelrag-original-order", attack_seed)
        ),
        "random": _static_ordering(
            probes,
            "random",
            seed=stable_seed("random-multiprobe-order", attack_seed),
        ),
        "similarity": _static_ordering(probes, "similarity", seed=attack_seed),
        "composite_v1": _static_ordering(
            probes, "composite_v1", seed=attack_seed
        ),
        "selected_random": selected_random_ordering(
            probes,
            seed=stable_seed("selected-random-order", attack_seed),
        ),
        "failure_repair_random": failure_repair_ordering(
            probes,
            seed=stable_seed("failure-repair-random-order", attack_seed),
        ),
    }
    orderings["cluster_diverse"] = cluster_diverse_ordering(
        probes, cluster_by_canary
    )
    orderings["adaptive_bayes"] = adaptive_bayes_ordering(
        probes,
        cluster_by_canary,
        retained_canaries,
        prior_retention=retained_cluster_count / len(clusters),
        retained_success_by_probe=retained_success_by_probe,
        removed_success_by_probe=removed_success_by_probe,
        absent_success_probability=absent_success_probability,
    )
    orderings["cluster_thompson"] = cluster_thompson_ordering(
        probes,
        cluster_by_canary,
        retained_canaries,
        seed=stable_seed("cluster-thompson", attack_seed),
        retained_success_by_probe=retained_success_by_probe,
        removed_success_by_probe=removed_success_by_probe,
    )
    orderings["q0_adaptive_bayes"] = adaptive_bayes_ordering(
        probes,
        cluster_by_canary,
        retained_canaries,
        prior_retention=retained_cluster_count / len(clusters),
        retained_success_by_probe=retained_success_by_probe,
        removed_success_by_probe=removed_success_by_probe,
        absent_success_probability=absent_success_probability,
        probe_selector="first",
    )
    orderings["failure_repair_adaptive"] = adaptive_bayes_ordering(
        probes,
        cluster_by_canary,
        retained_canaries,
        prior_retention=retained_cluster_count / len(clusters),
        retained_success_by_probe=retained_success_by_probe,
        removed_success_by_probe=removed_success_by_probe,
        absent_success_probability=absent_success_probability,
        probe_selector="failure_repair",
    )
    return {
        "scenario": scenario,
        "attack_seed": attack_seed,
        "retained_cluster_count": retained_cluster_count,
        "retained_clusters": sorted(retained_clusters),
        "retained_canaries": sorted(retained_canaries),
        "retained_artifact_fraction": len(retained_canaries) / len(cluster_by_canary),
        "policies": {
            method: evaluate_policy(
                ordering,
                retained_canaries,
                cluster_by_canary,
                budgets=budgets,
                retained_success_by_probe=retained_success_by_probe,
                removed_success_by_probe=removed_success_by_probe,
            )
            for method, ordering in orderings.items()
        },
    }


def mean_ci(values: Sequence[float], *, seed: int = 20260820) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"mean": None, "ci95": [None, None]}
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(10_000, len(values)), replace=True)
    means = samples.mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def paired_signflip_pvalue(
    values: Sequence[float], *, seed: int = 20260820, draws: int = 20_000
) -> float | None:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    observed = abs(float(values.mean()))
    generator = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    chunk_size = 2_000
    while completed < draws:
        current = min(chunk_size, draws - completed)
        signs = generator.choice((-1.0, 1.0), size=(current, len(values)))
        permuted = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
        completed += current
    return (extreme + 1) / (draws + 1)


def paired_summary(values: Sequence[float]) -> dict[str, Any]:
    result = mean_ci(values)
    result["paired_signflip_p"] = paired_signflip_pvalue(values)
    return result


def summarize_attacks(
    attacks: Sequence[Mapping[str, Any]],
    *,
    budgets: Sequence[int] = (5, 10, 20, 60),
) -> dict[str, Any]:
    """Aggregate paired attack seeds and compare policies to composite_v1."""

    summaries = {}
    policy_names = list(attacks[0]["policies"]) if attacks else []
    for policy in policy_names:
        rows = [attack["policies"][policy] for attack in attacks]
        summaries[policy] = {
            "first_success_queries_capped": mean_ci(
                [
                    row["first_success_queries"]
                    if row["first_success_queries"] is not None
                    else row["num_queries"] + 1
                    for row in rows
                ]
            ),
            "normalized_cumulative_detection": mean_ci(
                [row["normalized_cumulative_detection"] for row in rows]
            ),
            "budgets": {
                str(budget): {
                    "success_rate": mean_ci(
                        [row["budgets"][str(budget)]["success_rate"] for row in rows]
                    ),
                    "successes": mean_ci(
                        [row["budgets"][str(budget)]["successes"] for row in rows]
                    ),
                    "retained_cluster_discovery": mean_ci(
                        [
                            row["budgets"][str(budget)][
                                "retained_cluster_discovery"
                            ]
                            for row in rows
                        ]
                    ),
                }
                for budget in budgets
            },
        }
    paired = {}
    if attacks and "composite_v1" in policy_names:
        for policy in ("cluster_diverse", "adaptive_bayes"):
            paired[policy] = {
                f"success_at_{budget}_difference": paired_summary(
                    [
                        attack["policies"][policy]["budgets"][str(budget)][
                            "successes"
                        ]
                        - attack["policies"]["composite_v1"]["budgets"][
                            str(budget)
                        ]["successes"]
                        for attack in attacks
                    ]
                )
                for budget in budgets
            }
            paired[policy]["cumulative_detection_difference"] = paired_summary(
                [
                    attack["policies"][policy][
                        "normalized_cumulative_detection"
                    ]
                    - attack["policies"]["composite_v1"][
                        "normalized_cumulative_detection"
                    ]
                    for attack in attacks
                ]
            )
    return {
        "num_attack_seeds": len(attacks),
        "mean_retained_artifact_fraction": (
            float(np.mean([row["retained_artifact_fraction"] for row in attacks]))
            if attacks
            else None
        ),
        "policies": summaries,
        "paired_vs_composite_v1": paired,
    }
