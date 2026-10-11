"""Outcome-blind partitions and prefix-only execution for migration confirmation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from sentinelrag.testing.adaptive import (
    adaptive_bayes_ordering,
    cluster_thompson_ordering,
    failure_repair_ordering,
    random_candidate_ordering,
    sentinelrag_original_ordering,
    source_similarity_ordering,
)
from sentinelrag.testing.corpus_retrieval import IndexedMigrationDocument
from sentinelrag.testing.prioritization import normalized_answer_match

POLICIES = (
    "sentinelrag_original",
    "random_candidate",
    "source_similarity",
    "failure_repair_random",
    "q0_adaptive_bayes",
    "failure_repair_adaptive",
    "cluster_thompson",
    "adaptive_bayes",
)


def explicit_abstention(answer, expected):
    text = answer.strip().casefold().rstrip(".!\u3002")
    expected_text = expected.strip().casefold().rstrip(".!\u3002")
    return text == "cannot answer" and expected_text not in {
        "",
        "cannot answer",
        "unknown",
        "unavailable",
        "not known",
        "not applicable",
    }


def independent_corpus_partitions(records, *, num_clusters=8, seed=20260906):
    """Fit topics on ordinary documents, then assign original sentinels unchanged."""
    clean = [row for row in records if row.source_type == "clean"]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 1),
        max_features=15000,
        sublinear_tf=True,
    )
    clean_features = vectorizer.fit_transform(
        "\n".join(row.variants["original"]) for row in clean
    )
    components = min(48, clean_features.shape[0] - 1, clean_features.shape[1] - 1)
    reducer = TruncatedSVD(n_components=components, random_state=seed)
    clean_vectors = normalize(reducer.fit_transform(clean_features))
    model = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=seed,
        n_init=20,
        batch_size=256,
    ).fit(clean_vectors)
    vectors = normalize(
        reducer.transform(
            vectorizer.transform("\n".join(row.variants["original"]) for row in records)
        )
    )
    labels = model.predict(vectors)
    return {
        f"{row.source_type}:{row.source_id}": int(label)
        for row, label in zip(records, labels, strict=True)
    }


def materialize_partition(
    records, assignment, retained_topics, *, variant, clean_only=False
):
    """Physically select ordinary and sentinel documents using one partition rule."""
    documents = []
    for row in records:
        if clean_only and row.source_type != "clean":
            continue
        if assignment[f"{row.source_type}:{row.source_id}"] not in retained_topics:
            continue
        documents.extend(
            IndexedMigrationDocument(row.source_type, row.source_id, index, text)
            for index, text in enumerate(row.variants[variant])
        )
    return documents


def context_diagnostics(candidate, context):
    own = [
        row
        for row in context
        if row["source"] == "sentinel" and row["id"] == candidate.canary_id
    ]
    return {
        "retrieved_sentinel": bool(own),
        "own_chunk_literal_support": any(
            normalized_answer_match(row["text"], candidate.expected_answer)
            for row in own
        ),
        "context_literal_support": normalized_answer_match(
            "\n".join(row["text"] for row in context), candidate.expected_answer
        ),
    }


def policy_ordering(probes, clusters, policy, labels, *, seed, p0, prior=0.5):
    """Only a policy's observed prefix is supplied as target feedback."""
    if policy == "sentinelrag_original":
        return sentinelrag_original_ordering(probes, seed=seed)
    if policy == "random_candidate":
        return random_candidate_ordering(probes, seed=seed)
    if policy == "source_similarity":
        return source_similarity_ordering(probes)
    if policy == "failure_repair_random":
        return failure_repair_ordering(probes, seed=seed)
    # Missing labels only affect the speculative tail; next_probe returns its
    # first unobserved item, before any such label can influence a decision.
    feedback = defaultdict(bool, labels)
    common = {
        "retained_success_by_probe": feedback,
        "removed_success_by_probe": feedback,
    }
    canaries = set(clusters)
    if policy == "cluster_thompson":
        return cluster_thompson_ordering(
            probes, clusters, canaries, seed=seed, **common
        )
    selector = {
        "q0_adaptive_bayes": "first",
        "failure_repair_adaptive": "failure_repair",
        "adaptive_bayes": "retained_probability",
    }[policy]
    return adaptive_bayes_ordering(
        probes,
        clusters,
        canaries,
        prior_retention=prior,
        absent_success_probability=p0,
        probe_selector=selector,
        **common,
    )


def next_probe(probes, clusters, policy, prefix, *, seed, p0, prior=0.5):
    ids = [row["probe_id"] for row in prefix]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated probe in observed prefix")
    ordering = policy_ordering(
        probes,
        clusters,
        policy,
        {row["probe_id"]: row["is_correct"] for row in prefix},
        seed=seed,
        p0=p0,
        prior=prior,
    )
    generated_ids = [row.profile.candidate.probe_id for row in ordering]
    if generated_ids[: len(ids)] != ids:
        raise ValueError("Observed prefix differs from the frozen policy")
    return ordering[len(ids)] if len(ids) < len(ordering) else None


def document_identity(document):
    return (
        document.source_type,
        document.source_id,
        document.chunk_index,
        document.text,
    )


def dense_subset_scores(full_scores, full_documents, subset):
    """Exact subset ranking for a fixed dense encoder, with content checks."""
    positions = {
        document_identity(row): index for index, row in enumerate(full_documents)
    }
    indices = [positions[document_identity(row)] for row in subset]
    return np.asarray(full_scores)[:, indices]


def serialize_documents(documents):
    return [asdict(row) for row in documents]
