import numpy as np

from sentinelrag.testing.corpus_retrieval import (
    IndexedMigrationDocument,
    context_rows,
    target_outcomes,
    top_indices,
)


def documents():
    return [
        IndexedMigrationDocument("clean", "d0", 0, "clean zero"),
        IndexedMigrationDocument("sentinel", "c0", 0, "sentinel zero"),
        IndexedMigrationDocument("sentinel", "c1", 0, "sentinel one"),
    ]


def test_top_indices_are_descending_and_deterministic_for_ties():
    assert top_indices(np.asarray([0.2, 0.9, 0.9]), 2).tolist() == [1, 2]


def test_target_outcomes_require_the_matching_sentinel():
    scores = np.asarray([[0.1, 0.9, 0.8], [0.1, 0.9, 0.8]])
    outcomes = target_outcomes(scores, documents(), ["c0", "c1"], top_k=1)
    assert outcomes == {0: True, 1: False}


def test_context_rows_support_clean_and_partial_contexts():
    scores = np.asarray([0.5, 0.9, 0.8])
    clean = context_rows(scores, documents(), top_k=2, clean_only=True)
    assert [row["id"] for row in clean] == ["d0"]
    partial = context_rows(
        scores, documents(), top_k=2, retained_canaries={"c1"}
    )
    assert [row["id"] for row in partial] == ["c1", "d0"]
