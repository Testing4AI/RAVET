"""Probe prioritization independent of any particular canary construction scheme."""

from __future__ import annotations

import math
import random
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sentinelrag.utils.io import load_json
from sentinelrag.utils.stats import (
    bernoulli_mixture_e_value,
    binomial_test_greater,
)

MIN_FAILURE_REPAIR_HIT_GAIN = 0.5


@dataclass(frozen=True)
class ProbeCandidate:
    """A black-box test and its oracle, grouped by independent canary unit."""

    probe_id: str
    canary_id: str
    question: str
    expected_answer: str
    canary_text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalObservation:
    """Retrieval evidence for one shadow-system configuration."""

    variant: str
    clean_best_distance: float
    target_distance: float | None
    target_rank: int | None

    @property
    def hit_at_5(self) -> bool:
        return self.target_rank is not None and self.target_rank <= 5

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.target_rank is None else 1.0 / self.target_rank

    @property
    def separation_margin(self) -> float:
        if self.target_distance is None:
            return -1.0
        return self.clean_best_distance - self.target_distance


@dataclass
class ProbeProfile:
    """Calibration measurements and ranker scores for one probe."""

    candidate: ProbeCandidate
    observations: list[RetrievalObservation]
    similarity_score: float = 0.0
    proposed_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["features"] = self.features()
        return payload

    def features(self) -> dict[str, float]:
        if not self.observations:
            return {
                "original_target_similarity": 0.0,
                "robust_margin": -1.0,
                "hit_rate_at_5": 0.0,
                "mean_reciprocal_rank": 0.0,
            }

        original = next(
            (item for item in self.observations if item.variant == "original"),
            self.observations[0],
        )
        margins = sorted(item.separation_margin for item in self.observations)
        robust_index = max(0, math.ceil(0.2 * len(margins)) - 1)
        return {
            "original_target_similarity": (
                0.0
                if original.target_distance is None
                else 1.0 - original.target_distance
            ),
            "robust_margin": margins[robust_index],
            "hit_rate_at_5": sum(item.hit_at_5 for item in self.observations)
            / len(self.observations),
            "mean_reciprocal_rank": sum(
                item.reciprocal_rank for item in self.observations
            )
            / len(self.observations),
        }


def load_sentinelrag_candidates(path: str | Path) -> list[ProbeCandidate]:
    """Adapt a SentinelRAG injection result to the generic candidate contract."""

    payload = load_json(path)
    texts = payload["watermark_texts"]
    question_groups = payload["watermark_questions"]
    candidates: list[ProbeCandidate] = []

    for group_index, group in enumerate(question_groups):
        canary_index = int(group.get("watermark_index", group_index))
        questions = group.get("questions", [])
        answers = group.get("correct_answers", [])
        if canary_index >= len(texts):
            raise ValueError(f"Canary index {canary_index} has no watermark text")
        if len(questions) != len(answers):
            raise ValueError(f"Canary {canary_index} has mismatched questions and answers")

        for question_index, (question, answer) in enumerate(zip(questions, answers)):
            candidates.append(
                ProbeCandidate(
                    probe_id=f"c{canary_index:03d}-p{question_index:02d}",
                    canary_id=f"c{canary_index:03d}",
                    question=question.strip(),
                    expected_answer=answer.strip(),
                    canary_text=texts[canary_index].strip(),
                    metadata={
                        "adapter": "sentinelrag",
                        "canary_index": canary_index,
                        "question_index": question_index,
                    },
                )
            )

    if not candidates:
        raise ValueError("Injection result contains no probe candidates")
    return candidates


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def score_profiles(profiles: Sequence[ProbeProfile]) -> None:
    """Assign a similarity baseline and robust clean-separation score in place."""

    feature_rows = [profile.features() for profile in profiles]
    similarities = _minmax(
        [row["original_target_similarity"] for row in feature_rows]
    )
    margins = _minmax([row["robust_margin"] for row in feature_rows])
    hit_rates = _minmax([row["hit_rate_at_5"] for row in feature_rows])
    reciprocal_ranks = _minmax(
        [row["mean_reciprocal_rank"] for row in feature_rows]
    )

    for index, profile in enumerate(profiles):
        profile.similarity_score = similarities[index]
        profile.proposed_score = (
            0.50 * margins[index]
            + 0.30 * hit_rates[index]
            + 0.20 * reciprocal_ranks[index]
        )


def rank_probes(
    profiles: Sequence[ProbeProfile],
    method: str,
    *,
    seed: int = 0,
) -> list[ProbeProfile]:
    """Select one probe per canary and order independent canary tests."""

    if method not in {"random", "similarity", "proposed", "failure_repair"}:
        raise ValueError(f"Unknown prioritization method: {method}")

    if method == "random":
        shuffled = list(profiles)
        random.Random(seed).shuffle(shuffled)
        selected: list[ProbeProfile] = []
        seen: set[str] = set()
        for profile in shuffled:
            if profile.candidate.canary_id not in seen:
                selected.append(profile)
                seen.add(profile.candidate.canary_id)
        return selected

    if method == "failure_repair":
        grouped: dict[str, list[ProbeProfile]] = {}
        for profile in profiles:
            grouped.setdefault(profile.candidate.canary_id, []).append(profile)
        selected = []
        for group in grouped.values():
            if len(group) == 1:
                selected.append(group[0])
                continue
            first = min(
                group,
                key=lambda profile: (
                    int(profile.candidate.metadata.get("question_index", 0)),
                    profile.candidate.probe_id,
                ),
            )
            best = max(
                group,
                key=lambda profile: (
                    profile.features()["hit_rate_at_5"],
                    profile.features()["robust_margin"],
                    profile.features()["mean_reciprocal_rank"],
                    profile.proposed_score,
                    profile.candidate.probe_id,
                ),
            )
            if (
                best.features()["hit_rate_at_5"]
                - first.features()["hit_rate_at_5"]
                >= MIN_FAILURE_REPAIR_HIT_GAIN
            ):
                selected.append(best)
            else:
                selected.append(first)
        return sorted(
            selected,
            key=lambda profile: (
                profile.proposed_score,
                profile.candidate.probe_id,
            ),
            reverse=True,
        )

    score_name = "similarity_score" if method == "similarity" else "proposed_score"
    best_by_canary: dict[str, ProbeProfile] = {}
    for profile in profiles:
        canary_id = profile.candidate.canary_id
        incumbent = best_by_canary.get(canary_id)
        if incumbent is None or (
            getattr(profile, score_name), profile.candidate.probe_id
        ) > (getattr(incumbent, score_name), incumbent.candidate.probe_id):
            best_by_canary[canary_id] = profile

    return sorted(
        best_by_canary.values(),
        key=lambda profile: (
            getattr(profile, score_name),
            profile.candidate.probe_id,
        ),
        reverse=True,
    )


def normalized_answer_match(answer: str, expected: str) -> bool:
    """Conservative answer oracle used by the initial end-to-end experiment."""

    def normalize(value: str) -> str:
        canonical = unicodedata.normalize("NFKC", value).casefold()
        return " ".join(re.findall(r"[^\W_]+", canonical))

    normalized_answer = normalize(answer)
    normalized_expected = normalize(expected)
    return bool(normalized_expected) and (
        f" {normalized_expected} " in f" {normalized_answer} "
    )


def answer_tokens_supported(text: str, expected: str) -> bool:
    """Check literal evidence while tolerating punctuation and simple plurals."""

    def evidence_tokens(value: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", value.casefold())
        return {
            token[:-1] if len(token) > 3 and token.endswith("s") else token
            for token in tokens
        }

    expected_tokens = evidence_tokens(expected)
    return bool(expected_tokens) and expected_tokens <= evidence_tokens(text)


def evaluate_ordering(
    ordering: Sequence[ProbeProfile],
    outcomes: Mapping[str, bool],
    *,
    p0: float = 0.02,
    alpha: float = 0.01,
) -> dict[str, Any]:
    """Evaluate a distinct-canary ordering under a sequential query budget."""

    curve = []
    successes = 0
    first_significant: int | None = None
    first_anytime_significant: int | None = None
    cumulative_area = 0.0

    for budget, profile in enumerate(ordering, start=1):
        success = bool(outcomes.get(profile.candidate.probe_id, False))
        successes += int(success)
        p_value = binomial_test_greater(successes, budget, p0=p0)
        e_value = bernoulli_mixture_e_value(successes, budget, p0=p0)
        if first_significant is None and p_value <= alpha:
            first_significant = budget
        if first_anytime_significant is None and e_value >= 1.0 / alpha:
            first_anytime_significant = budget
        detection_rate = successes / budget
        cumulative_area += detection_rate
        curve.append(
            {
                "budget": budget,
                "probe_id": profile.candidate.probe_id,
                "canary_id": profile.candidate.canary_id,
                "success": success,
                "cumulative_successes": successes,
                "detection_rate": detection_rate,
                "p_value": p_value,
                "significant": p_value <= alpha,
                "e_value": e_value,
                "anytime_significant": first_anytime_significant is not None,
            }
        )

    total = len(ordering)
    return {
        "num_queries": total,
        "num_successes": successes,
        "final_detection_rate": successes / total if total else 0.0,
        "queries_to_significance": first_significant,
        "queries_to_anytime_significance": first_anytime_significant,
        "normalized_cumulative_detection": cumulative_area / total if total else 0.0,
        "curve": curve,
    }
