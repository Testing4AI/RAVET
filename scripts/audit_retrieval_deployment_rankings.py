#!/usr/bin/env python3
"""Independently reconstruct chunk coverage and all changed retrieval contexts."""

import itertools
import json

import numpy as np

import run_retrieval_deployment_changes as study
from run_partition_confirmation import ROOT, digest, file_hash, read, save
from sentinelrag.testing.corpus_retrieval import IndexedMigrationDocument, context_rows
from sentinelrag.testing.matrix import sparse_bm25_scores
from sentinelrag.testing.retrieval_deployment import rechunk_documents


def checked_vectors(texts, path, source):
    metadata = read(path.with_suffix(".json"))
    assert metadata["signature"] == digest({"texts": texts, "model": source["dense_model"],
                                            "revision": source["dense_revision"]})
    vectors = np.load(path)["vectors"]
    assert len(vectors) == len(texts) and np.isfinite(vectors).all()
    return vectors


def main():
    lock = read(study.OUT / "protocol_lock.json")
    protocol, source = lock["protocol"], lock["source_protocol"]
    banks, corpora = study.source_banks(protocol), {}
    logical, physical = 0, 0
    for block, method, state in itertools.product(protocol["blocks"], protocol["methods"], ("watermarked", "clean")):
        raw = banks[block]["watermarked"][method] if state == "watermarked" else banks[block]["clean"]
        documents = [IndexedMigrationDocument(**r) for r in raw]
        chunks = rechunk_documents(documents, size=protocol["chunk_size_words"], overlap=protocol["chunk_overlap_words"])
        for original in documents:
            own = [r for r in chunks if (r.source_type, r.source_id) == (original.source_type, original.source_id)]
            words = original.text.split()
            if len(words) <= protocol["chunk_size_words"]:
                assert own == [original]
            else:
                recovered = own[0].text.split()
                for row in own[1:]:
                    previous = recovered[-protocol["chunk_overlap_words"]:]
                    assert previous == row.text.split()[:protocol["chunk_overlap_words"]]
                    recovered.extend(row.text.split()[protocol["chunk_overlap_words"]:])
                assert recovered == words
            logical += 1
        physical += len(chunks)
        corpora[block, method, state] = chunks
    texts = sorted({r.text for rows in corpora.values() for r in rows})
    vectors = checked_vectors(texts, study.OUT / "dense/shared_chunks.npz", source)
    positions = {text: i for i, text in enumerate(texts)}
    frozen = read(study.OUT / "frozen_contexts.json")
    checked = 0
    for block, method, state, retriever in itertools.product(
            protocol["blocks"], protocol["methods"], ("watermarked", "clean"), protocol["target_retrievers"]):
        tests, _, _ = study.shared.suite(protocol, source, block, method)
        documents = corpora[block, method, state]
        if retriever == "bm25":
            scores = sparse_bm25_scores([d.text for d in documents], [t["question"] for t in tests]).toarray()
        else:
            queries = checked_vectors([t["question"] for t in tests],
                                      study.OUT / "dense/queries" / f"{digest(tests)}.npz", source)
            scores = np.einsum("ij,kj->ik", queries, vectors[[positions[d.text] for d in documents]],
                               optimize=True).astype(np.float32)
        for scenario in protocol["scenarios"]:
            job = f"{block}/{method}/{scenario}/{retriever}/{state}"
            path = study.OUT / "contexts" / job / "contexts.json"
            assert file_hash(path) == frozen["contexts_sha256"][job]
            context = read(path)
            reference = read(study.archived_path(protocol, block, method, retriever, state))
            for i, test in enumerate(tests):
                expected = (context_rows(scores[i], documents, top_k=5) if scenario == "rechunk_120_30" else
                            reference["contexts"][test["probe_id"]][:protocol["scenario_top_k"][scenario]])
                actual = context["contexts"][test["unit_id"]]
                assert len(actual) == len(expected)
                for a, b in zip(actual, expected, strict=True):
                    assert all(a[k] == b[k] for k in ("source", "id", "chunk_index", "text"))
                    assert np.isclose(a["score"], b["score"], rtol=1e-6, atol=1e-7)
                checked += 1
    assert checked == 1440
    result = {"status": "pass", "reconstructed_contexts": checked,
              "logical_document_occurrences_checked": logical, "physical_chunk_occurrences_checked": physical,
              "unique_chunk_texts": len(texts), "word_order_and_overlap_preserved": True,
              "new_api_requests": 0, **study.verify(lock)}
    save(study.OUT / "ranking_audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
