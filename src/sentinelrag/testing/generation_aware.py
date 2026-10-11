"""Generation-aware features and frozen probe prioritization models."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sentinelrag.testing.adaptive import PreparedProbe

RETRIEVAL_FEATURES = (
    "original_target_similarity",
    "robust_margin",
    "hit_rate_at_1",
    "hit_rate_at_5",
    "mean_reciprocal_rank",
    "mean_target_similarity",
    "min_target_similarity",
    "mean_clean_best_similarity",
    "mean_separation_margin",
    "rank_log_mean",
    "rank_log_std",
    "retained_success_probability",
    "composite_v1_score",
)

SURFACE_FEATURES = (
    "question_tokens",
    "answer_tokens",
    "canary_tokens",
    "required_support_count",
    "question_canary_containment",
    "answer_canary_containment",
    "question_answer_overlap",
    "question_digit_fraction",
    "answer_digit_fraction",
    "answer_upper_fraction",
    "answer_code_like",
    "answer_short_entity",
)

ALL_FEATURES = RETRIEVAL_FEATURES + SURFACE_FEATURES


@dataclass(frozen=True)
class LabeledProbe:
    dataset: str
    probe_id: str
    canary_id: str
    features: Mapping[str, float]
    label_semantic: bool
    label_exact: bool
    clean_semantic: bool
    clean_exact: bool


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _containment(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    if not left_set:
        return 0.0
    return len(left_set & set(right)) / len(left_set)


def extract_features(probe: PreparedProbe) -> dict[str, float]:
    """Extract only features available before querying the target RAG system."""

    profile = probe.profile
    base = profile.features()
    observations = profile.observations
    target_similarities = [
        0.0 if row.target_distance is None else 1.0 - row.target_distance
        for row in observations
    ]
    clean_similarities = [1.0 + row.clean_best_distance for row in observations]
    margins = [row.separation_margin for row in observations]
    capped_log_ranks = [
        math.log1p(100 if row.target_rank is None else min(row.target_rank, 100))
        for row in observations
    ]
    candidate = profile.candidate
    question_tokens = _tokens(candidate.question)
    answer_tokens = _tokens(candidate.expected_answer)
    canary_tokens = _tokens(candidate.canary_text)
    answer_letters = [char for char in candidate.expected_answer if char.isalpha()]
    answer_upper_fraction = (
        sum(char.isupper() for char in answer_letters) / len(answer_letters)
        if answer_letters
        else 0.0
    )
    answer_digit_fraction = (
        sum(char.isdigit() for char in candidate.expected_answer)
        / max(1, len(candidate.expected_answer))
    )
    question_digit_fraction = (
        sum(char.isdigit() for char in candidate.question)
        / max(1, len(candidate.question))
    )
    required_support = candidate.metadata.get("required_document_indices", [0])
    features = {
        "original_target_similarity": base["original_target_similarity"],
        "robust_margin": base["robust_margin"],
        "hit_rate_at_1": (
            sum(row.target_rank is not None and row.target_rank <= 1 for row in observations)
            / len(observations)
            if observations
            else 0.0
        ),
        "hit_rate_at_5": base["hit_rate_at_5"],
        "mean_reciprocal_rank": base["mean_reciprocal_rank"],
        "mean_target_similarity": float(np.mean(target_similarities)),
        "min_target_similarity": float(np.min(target_similarities)),
        "mean_clean_best_similarity": float(np.mean(clean_similarities)),
        "mean_separation_margin": float(np.mean(margins)),
        "rank_log_mean": float(np.mean(capped_log_ranks)),
        "rank_log_std": float(np.std(capped_log_ranks)),
        "retained_success_probability": probe.retained_success_probability,
        "composite_v1_score": profile.proposed_score,
        "question_tokens": float(len(question_tokens)),
        "answer_tokens": float(len(answer_tokens)),
        "canary_tokens": float(len(canary_tokens)),
        "required_support_count": float(len(required_support)),
        "question_canary_containment": _containment(
            question_tokens, canary_tokens
        ),
        "answer_canary_containment": _containment(answer_tokens, canary_tokens),
        "question_answer_overlap": _containment(answer_tokens, question_tokens),
        "question_digit_fraction": question_digit_fraction,
        "answer_digit_fraction": answer_digit_fraction,
        "answer_upper_fraction": answer_upper_fraction,
        "answer_code_like": float(
            bool(re.search(r"[A-Z]", candidate.expected_answer))
            and bool(re.search(r"\d", candidate.expected_answer))
        ),
        "answer_short_entity": float(len(answer_tokens) <= 4),
    }
    return features


def build_labeled_probes(
    dataset: str,
    prepared: Sequence[PreparedProbe],
    answers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[LabeledProbe]:
    """Join frozen pre-query features with existing black-box answer labels."""

    prepared_by_id = {
        row.profile.candidate.probe_id: row for row in prepared
    }
    clean_by_id = {row["probe_id"]: row for row in answers["clean"]}
    watermarked_by_id = {
        row["probe_id"]: row for row in answers["watermarked"]
    }
    shared = sorted(prepared_by_id.keys() & watermarked_by_id.keys() & clean_by_id.keys())
    return [
        LabeledProbe(
            dataset=dataset,
            probe_id=probe_id,
            canary_id=prepared_by_id[probe_id].profile.candidate.canary_id,
            features=extract_features(prepared_by_id[probe_id]),
            label_semantic=bool(watermarked_by_id[probe_id]["is_correct"]),
            label_exact=bool(watermarked_by_id[probe_id]["is_correct_exact"]),
            clean_semantic=bool(clean_by_id[probe_id]["is_correct"]),
            clean_exact=bool(clean_by_id[probe_id]["is_correct_exact"]),
        )
        for probe_id in shared
    ]


def matrix_from_rows(
    rows: Sequence[LabeledProbe], feature_names: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [[row.features[name] for name in feature_names] for row in rows],
        dtype=float,
    )


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    random_state=20260820,
                    max_iter=2_000,
                ),
            ),
        ]
    )


def fit_model(
    rows: Sequence[LabeledProbe],
    feature_names: Sequence[str],
    *,
    label_name: str = "label_semantic",
) -> Pipeline:
    labels = np.asarray([int(getattr(row, label_name)) for row in rows])
    if len(np.unique(labels)) < 2:
        raise ValueError("Generation-aware training requires both label classes")
    return make_model().fit(matrix_from_rows(rows, feature_names), labels)


def prediction_metrics(labels: Sequence[bool], scores: Sequence[float]) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return {
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "brier": float("nan"),
            "log_loss": float("nan"),
        }
    clipped = np.clip(scores, 1e-8, 1 - 1e-8)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, clipped)),
        "log_loss": float(log_loss(labels, clipped)),
    }


def evaluate_transfer(
    train_rows: Sequence[LabeledProbe],
    test_rows: Sequence[LabeledProbe],
    *,
    label_name: str,
) -> dict[str, Any]:
    """Evaluate all frozen feature groups and retrieval-score baselines."""

    labels = [bool(getattr(row, label_name)) for row in test_rows]
    groups = {
        "retrieval_only": RETRIEVAL_FEATURES,
        "surface_only": SURFACE_FEATURES,
        "retrieval_surface": ALL_FEATURES,
    }
    result = {}
    for group_name, feature_names in groups.items():
        model = fit_model(train_rows, feature_names, label_name=label_name)
        scores = model.predict_proba(matrix_from_rows(test_rows, feature_names))[:, 1]
        result[group_name] = prediction_metrics(labels, scores)
    for baseline, feature in {
        "original_similarity": "original_target_similarity",
        "robust_margin": "robust_margin",
        "composite_v1": "composite_v1_score",
    }.items():
        result[baseline] = prediction_metrics(
            labels, [row.features[feature] for row in test_rows]
        )
    return {
        "train_dataset": sorted({row.dataset for row in train_rows}),
        "test_dataset": sorted({row.dataset for row in test_rows}),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "test_positive_rate": float(np.mean(labels)),
        "models": result,
    }


def score_prepared(
    model: Pipeline,
    prepared: Sequence[PreparedProbe],
    feature_names: Sequence[str],
) -> dict[str, float]:
    features = [extract_features(row) for row in prepared]
    matrix = np.asarray(
        [[row[name] for name in feature_names] for row in features], dtype=float
    )
    probabilities = model.predict_proba(matrix)[:, 1]
    return {
        row.profile.candidate.probe_id: float(probability)
        for row, probability in zip(prepared, probabilities)
    }


def select_and_order(
    prepared: Sequence[PreparedProbe],
    scores: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Select one probe per independent canary and order by frozen score."""

    selected: dict[str, PreparedProbe] = {}
    for row in prepared:
        candidate = row.profile.candidate
        incumbent = selected.get(candidate.canary_id)
        if incumbent is None or (
            scores[candidate.probe_id], candidate.probe_id
        ) > (
            scores[incumbent.profile.candidate.probe_id],
            incumbent.profile.candidate.probe_id,
        ):
            selected[candidate.canary_id] = row
    ordered = sorted(
        selected.values(),
        key=lambda row: (
            scores[row.profile.candidate.probe_id],
            row.profile.candidate.probe_id,
        ),
        reverse=True,
    )
    return [
        {
            "rank": rank,
            "probe_id": row.profile.candidate.probe_id,
            "canary_id": row.profile.candidate.canary_id,
            "score": scores[row.profile.candidate.probe_id],
        }
        for rank, row in enumerate(ordered, start=1)
    ]
