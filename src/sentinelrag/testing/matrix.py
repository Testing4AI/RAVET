"""Cached retrieval simulation for multi-seed canary experiments."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sentinelrag.testing.artifacts import CanaryArtifact
from sentinelrag.testing.prioritization import (
    ProbeCandidate,
    ProbeProfile,
    RetrievalObservation,
    evaluate_ordering,
    normalized_answer_match,
    rank_probes,
    score_profiles,
)

CALIBRATION_VARIANTS = (
    "original",
    "reverse_sentences",
    "boilerplate",
    "paired_chunks",
)
HELDOUT_VARIANT = "single_sentence_chunks"
PARTIAL_THEFT_VARIANT = "partial_theft_0.5"


@dataclass(frozen=True)
class PhysicalDocument:
    canary_id: str
    logical_index: int
    text: str


@dataclass
class VariantScores:
    variant: str
    documents: list[PhysicalDocument]
    clean_top_scores: np.ndarray
    canary_scores: np.ndarray


@dataclass
class RetrievalScoreCache:
    candidates: list[ProbeCandidate]
    variants: dict[str, VariantScores]
    clean_depth: int
    retriever: str


def split_sentences(text: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]
    return sentences or [text.strip()]


def transform_document(text: str, variant: str) -> list[str]:
    if variant == "original":
        return [text]
    if variant == "reverse_sentences":
        return [" ".join(reversed(split_sentences(text)))]
    if variant == "boilerplate":
        return [
            (
                "This archived record is retained for routine evidence review. "
                f"{text} The record is maintained for future audit and replication."
            )
        ]
    if variant == "paired_chunks":
        sentences = split_sentences(text)
        return [
            " ".join(sentences[index : index + 2])
            for index in range(0, len(sentences), 2)
        ]
    if variant == "single_sentence_chunks":
        return split_sentences(text)
    raise ValueError(f"Unknown document variant: {variant}")


def flatten_variant_documents(
    artifacts: Sequence[CanaryArtifact], variant: str
) -> list[PhysicalDocument]:
    documents = []
    for artifact in artifacts:
        for logical_index, logical_document in enumerate(artifact.documents):
            documents.extend(
                PhysicalDocument(artifact.canary_id, logical_index, chunk)
                for chunk in transform_document(logical_document, variant)
            )
    return documents


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Embedding matrix contains non-finite values")
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left = left / np.maximum(left_norm, 1e-12)
    right = right / np.maximum(right_norm, 1e-12)
    # Accelerate may emit spurious floating-point warnings for finite normalized
    # operands; the explicit postcondition below remains the correctness guard.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = left @ right.T
    if not np.isfinite(scores).all():
        raise ValueError("Cosine score matrix contains non-finite values")
    return scores.astype(np.float32)


def build_dense_score_cache(
    vectorstore,
    artifacts: Sequence[CanaryArtifact],
    *,
    clean_depth: int = 50,
) -> RetrievalScoreCache:
    """Cache Contriever query scores against clean and canary documents."""

    candidates = [probe for artifact in artifacts for probe in artifact.probes]
    questions = [candidate.question for candidate in candidates]
    query_embeddings = np.asarray(
        vectorstore.get_embeddings_batch(questions), dtype=np.float32
    )
    clean_result = vectorstore.collection.query(
        query_embeddings=query_embeddings.tolist(),
        n_results=min(clean_depth, vectorstore.collection.count()),
        include=["distances"],
    )
    clean_top_scores = np.asarray(
        [[1.0 - float(distance) for distance in row] for row in clean_result["distances"]],
        dtype=np.float32,
    )

    variants = {}
    for variant in (*CALIBRATION_VARIANTS, HELDOUT_VARIANT):
        documents = flatten_variant_documents(artifacts, variant)
        document_embeddings = np.asarray(
            vectorstore.get_embeddings_batch([document.text for document in documents]),
            dtype=np.float32,
        )
        variants[variant] = VariantScores(
            variant=variant,
            documents=documents,
            clean_top_scores=clean_top_scores,
            canary_scores=_cosine_matrix(query_embeddings, document_embeddings),
        )
    return RetrievalScoreCache(
        candidates=candidates,
        variants=variants,
        clean_depth=clean_depth,
        retriever="contriever",
    )


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def sparse_bm25_scores(
    documents: Sequence[str],
    queries: Sequence[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    epsilon: float = 0.25,
):
    """Return a sparse score matrix equivalent to rank_bm25.BM25Okapi."""

    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-z0-9]+\b",
        dtype=np.float64,
    )
    counts = vectorizer.fit_transform(documents).tocsr()
    corpus_size = counts.shape[0]
    document_frequency = np.asarray(counts.getnnz(axis=0), dtype=np.float64)
    idf = np.log(corpus_size - document_frequency + 0.5) - np.log(
        document_frequency + 0.5
    )
    average_idf = float(idf.mean())
    idf[idf < 0] = epsilon * average_idf

    document_lengths = np.asarray(counts.sum(axis=1)).ravel()
    average_length = float(document_lengths.mean())
    weighted = counts.copy()
    row_indices = np.repeat(
        np.arange(corpus_size), np.diff(weighted.indptr)
    )
    term_frequencies = weighted.data
    length_norm = k1 * (
        1.0 + b * (document_lengths[row_indices] / average_length - 1.0)
    )
    weighted.data = (
        idf[weighted.indices]
        * term_frequencies
        * (k1 + 1.0)
        / (term_frequencies + length_norm)
    )
    query_counts = vectorizer.transform(queries)
    return (query_counts @ weighted.T).tocsr()


def build_bm25_score_cache(
    clean_documents: Sequence[str],
    artifacts: Sequence[CanaryArtifact],
    *,
    clean_depth: int = 50,
) -> RetrievalScoreCache:
    """Cache BM25 scores; selected seeds later hide unselected canary documents."""

    candidates = [probe for artifact in artifacts for probe in artifact.probes]
    questions = [candidate.question for candidate in candidates]
    variants = {}

    for variant in (*CALIBRATION_VARIANTS, HELDOUT_VARIANT):
        documents = flatten_variant_documents(artifacts, variant)
        all_documents = list(clean_documents) + [document.text for document in documents]
        score_matrix = sparse_bm25_scores(all_documents, questions)
        clean_rows = []
        canary_rows = []
        clean_count = len(clean_documents)
        canary_count = len(documents)
        for query_index in range(len(questions)):
            sparse_row = score_matrix.getrow(query_index)
            scores = np.zeros(clean_count + canary_count, dtype=np.float32)
            scores[sparse_row.indices] = sparse_row.data
            clean_scores = scores[:clean_count]
            depth = min(clean_depth, clean_count)
            top_indices = np.argpartition(clean_scores, -depth)[-depth:]
            clean_rows.append(np.sort(clean_scores[top_indices])[::-1])
            canary_rows.append(scores[clean_count:])
        variants[variant] = VariantScores(
            variant=variant,
            documents=documents,
            clean_top_scores=np.asarray(clean_rows, dtype=np.float32),
            canary_scores=np.asarray(canary_rows, dtype=np.float32),
        )
    return RetrievalScoreCache(
        candidates=candidates,
        variants=variants,
        clean_depth=clean_depth,
        retriever="bm25",
    )


def _candidate_support_groups(
    candidate: ProbeCandidate,
    variant: VariantScores,
) -> list[list[int]]:
    required_indices = candidate.metadata.get("required_document_indices", [0])
    required_evidence = candidate.metadata.get(
        "required_evidence", [candidate.expected_answer]
    )
    groups = []
    for logical_index, evidence in zip(required_indices, required_evidence):
        matching = [
            index
            for index, document in enumerate(variant.documents)
            if document.canary_id == candidate.canary_id
            and document.logical_index == logical_index
            and normalized_answer_match(document.text, evidence)
        ]
        if not matching:
            matching = [
                index
                for index, document in enumerate(variant.documents)
                if document.canary_id == candidate.canary_id
                and document.logical_index == logical_index
            ]
        groups.append(matching)
    return groups


def observe_candidate(
    cache: RetrievalScoreCache,
    *,
    candidate_index: int,
    selected_canary_ids: set[str],
    variant_name: str,
) -> RetrievalObservation:
    variant = cache.variants[variant_name]
    candidate = cache.candidates[candidate_index]
    selected_document_indices = [
        index
        for index, document in enumerate(variant.documents)
        if document.canary_id in selected_canary_ids
    ]
    clean_scores = variant.clean_top_scores[candidate_index]
    selected_scores = variant.canary_scores[
        candidate_index, selected_document_indices
    ]
    combined_scores = np.concatenate([clean_scores, selected_scores])

    support_ranks = []
    support_scores = []
    for group in _candidate_support_groups(candidate, variant):
        available_group = [
            document_index
            for document_index in group
            if variant.documents[document_index].canary_id in selected_canary_ids
        ]
        if not available_group:
            return RetrievalObservation(
                variant=variant_name,
                clean_best_distance=-float(clean_scores[0]),
                target_distance=None,
                target_rank=None,
            )
        best_score = max(
            float(variant.canary_scores[candidate_index, index])
            for index in available_group
        )
        rank = 1 + int(np.count_nonzero(combined_scores > best_score + 1e-12))
        support_ranks.append(rank)
        support_scores.append(best_score)

    return RetrievalObservation(
        variant=variant_name,
        clean_best_distance=-float(clean_scores[0]),
        target_distance=-min(support_scores),
        target_rank=max(support_ranks),
    )


def _minmax(values: Sequence[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def rank_margin_only(profiles: Sequence[ProbeProfile]) -> list[ProbeProfile]:
    margins = _minmax([profile.features()["robust_margin"] for profile in profiles])
    saved = [profile.proposed_score for profile in profiles]
    try:
        for profile, margin in zip(profiles, margins):
            profile.proposed_score = margin
        return rank_probes(profiles, "proposed")
    finally:
        for profile, score in zip(profiles, saved):
            profile.proposed_score = score


def run_seed(
    cache: RetrievalScoreCache,
    artifacts: Sequence[CanaryArtifact],
    *,
    seed: int,
    sample_size: int,
    top_k: int,
) -> dict[str, Any]:
    randomizer = random.Random(seed)
    selected_artifacts = randomizer.sample(
        list(artifacts), min(sample_size, len(artifacts))
    )
    selected_ids = {artifact.canary_id for artifact in selected_artifacts}
    retained_artifacts = randomizer.sample(
        selected_artifacts, math.ceil(0.5 * len(selected_artifacts))
    )
    partial_theft_ids = {artifact.canary_id for artifact in retained_artifacts}
    candidate_indices = [
        index
        for index, candidate in enumerate(cache.candidates)
        if candidate.canary_id in selected_ids
    ]
    profiles = []
    heldout_outcomes = {
        HELDOUT_VARIANT: {},
        PARTIAL_THEFT_VARIANT: {},
    }
    for candidate_index in candidate_indices:
        candidate = cache.candidates[candidate_index]
        observations = [
            observe_candidate(
                cache,
                candidate_index=candidate_index,
                selected_canary_ids=selected_ids,
                variant_name=variant,
            )
            for variant in CALIBRATION_VARIANTS
        ]
        profiles.append(ProbeProfile(candidate, observations))
        heldout = observe_candidate(
            cache,
            candidate_index=candidate_index,
            selected_canary_ids=selected_ids,
            variant_name=HELDOUT_VARIANT,
        )
        support_retrieved = (
            heldout.target_rank is not None and heldout.target_rank <= top_k
        )
        heldout_outcomes[HELDOUT_VARIANT][candidate.probe_id] = support_retrieved
        heldout_outcomes[PARTIAL_THEFT_VARIANT][candidate.probe_id] = (
            candidate.canary_id in partial_theft_ids and support_retrieved
        )

    score_profiles(profiles)
    orderings = {
        "random": rank_probes(profiles, "random", seed=seed),
        "similarity": rank_probes(profiles, "similarity"),
        "margin_only": rank_margin_only(profiles),
        "composite_v1": rank_probes(profiles, "proposed"),
    }
    heldout_methods = {
        variant: {
            method: evaluate_ordering(ordering, outcomes)
            for method, ordering in orderings.items()
        }
        for variant, outcomes in heldout_outcomes.items()
    }
    return {
        "seed": seed,
        "selected_canaries": sorted(selected_ids),
        "partial_theft_retained_canaries": sorted(partial_theft_ids),
        "methods": heldout_methods[HELDOUT_VARIANT],
        "heldout_methods": heldout_methods,
    }


def aggregate_seed_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods = runs[0]["methods"].keys() if runs else []
    aggregate = {}
    for method in methods:
        method_runs = [run["methods"][method] for run in runs]
        significance_queries = [
            run["queries_to_significance"]
            for run in method_runs
            if run["queries_to_significance"] is not None
        ]
        aggregate[method] = {
            "mean_detection_rate": float(
                np.mean([run["final_detection_rate"] for run in method_runs])
            ),
            "std_detection_rate": float(
                np.std(
                    [run["final_detection_rate"] for run in method_runs], ddof=1
                )
            )
            if len(method_runs) > 1
            else 0.0,
            "mean_cumulative_detection": float(
                np.mean(
                    [run["normalized_cumulative_detection"] for run in method_runs]
                )
            ),
            "mean_queries_to_significance": (
                float(np.mean(significance_queries))
                if significance_queries
                else None
            ),
            "significant_fraction": float(
                np.mean(
                    [run["queries_to_significance"] is not None for run in method_runs]
                )
            ),
        }
    return aggregate


def aggregate_heldout_seed_runs(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate every held-out transformation while preserving old run files."""

    if not runs or "heldout_methods" not in runs[0]:
        return {HELDOUT_VARIANT: aggregate_seed_runs(runs)}
    variants = runs[0]["heldout_methods"].keys()
    return {
        variant: aggregate_seed_runs(
            [{"methods": run["heldout_methods"][variant]} for run in runs]
        )
        for variant in variants
    }


def stable_condition_seed(dataset: str, retriever: str, scheme: str, offset: int):
    digest = hashlib.sha256(
        f"{dataset}|{retriever}|{scheme}|{offset}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")
