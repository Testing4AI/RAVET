"""Retrieval helpers for uniformly migrated clean and sentinel corpora."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IndexedMigrationDocument:
    source_type: str
    source_id: str
    chunk_index: int
    text: str


def top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    """Return descending top-score indices with deterministic tie breaking."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("Expected one-dimensional scores")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain non-finite values")
    count = min(max(0, count), len(scores))
    if count == 0:
        return np.asarray([], dtype=np.int64)
    candidate_indices = np.argpartition(scores, -count)[-count:]
    return np.asarray(
        sorted(candidate_indices, key=lambda index: (-scores[index], int(index))),
        dtype=np.int64,
    )


def target_outcomes(
    scores: np.ndarray,
    documents: list[IndexedMigrationDocument],
    canary_by_probe: list[str],
    *,
    top_k: int,
) -> dict[int, bool]:
    """Check whether each probe retrieves a chunk from its own sentinel."""

    scores = np.asarray(scores)
    if scores.shape != (len(canary_by_probe), len(documents)):
        raise ValueError("Score matrix shape does not match probes and documents")
    outcomes = {}
    for probe_index, canary_id in enumerate(canary_by_probe):
        ranked = top_indices(scores[probe_index], top_k)
        outcomes[probe_index] = any(
            documents[index].source_type == "sentinel"
            and documents[index].source_id == canary_id
            for index in ranked
        )
    return outcomes


def context_rows(
    scores: np.ndarray,
    documents: list[IndexedMigrationDocument],
    *,
    top_k: int,
    retained_canaries: set[str] | None = None,
    clean_only: bool = False,
) -> list[dict[str, object]]:
    """Materialize one ranked RAG context under optional partial retention."""

    eligible = []
    for index, document in enumerate(documents):
        include_sentinel = not clean_only and (
            retained_canaries is None or document.source_id in retained_canaries
        )
        if document.source_type == "clean" or include_sentinel:
            eligible.append(index)
    eligible_array = np.asarray(eligible, dtype=np.int64)
    if not len(eligible_array):
        return []
    local_top = top_indices(np.asarray(scores)[eligible_array], top_k)
    ranked = eligible_array[local_top]
    return [
        {
            "source": documents[index].source_type,
            "id": documents[index].source_id,
            "chunk_index": documents[index].chunk_index,
            "text": documents[index].text,
            "score": float(np.asarray(scores)[index]),
        }
        for index in ranked
    ]
