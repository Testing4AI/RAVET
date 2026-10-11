"""Content-preserving index changes with logical evidence identities intact."""

from dataclasses import replace
from math import ceil
from statistics import median

from sentinelrag.testing.corpus_retrieval import IndexedMigrationDocument
from sentinelrag.testing.representation_attacks import word_chunks


def rechunk_documents(documents, *, size=120, overlap=30):
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Invalid chunk window or overlap")
    result = []
    for document in documents:
        words = document.text.split()
        if len(words) <= size:
            result.append(replace(document, chunk_index=0))
            continue
        count = 1 + ceil((len(words) - size) / (size - overlap))
        chunks = word_chunks(document.text, size=size, overlap=overlap)[:count]
        result.extend(replace(document, chunk_index=i, text=text) for i, text in enumerate(chunks))
    identities = [(d.source_type, d.source_id, d.chunk_index) for d in result]
    if len(identities) != len(set(identities)):
        raise ValueError("Nonunique physical chunk identities")
    return result


def chunk_inventory(documents, *, size=120, overlap=30):
    chunks = rechunk_documents(documents, size=size, overlap=overlap)
    rows = []
    for source_type in ("clean", "sentinel"):
        selected = [d for d in documents if d.source_type == source_type]
        lengths = [len(d.text.split()) for d in selected]
        rows.append({"source_type": source_type, "logical_documents": len(selected),
                     "physical_chunks": sum(d.source_type == source_type for d in chunks),
                     "split_documents": sum(length > size for length in lengths),
                     "split_fraction": sum(length > size for length in lengths) / len(lengths) if lengths else 0,
                     "median_words": median(lengths) if lengths else 0,
                     "min_words": min(lengths) if lengths else 0,
                     "max_words": max(lengths) if lengths else 0})
    return rows


def context_diagnostics(test, context, *, carrier_present):
    own = [row for row in context if row["source"] == "sentinel" and row["id"] == test["unit_id"]]
    expected = " ".join(test["expected_answer"].split()).casefold()
    return {"carrier_present": carrier_present, "own_carrier_retrieved": bool(own),
            "own_answer_literal_support": any(expected in " ".join(r["text"].split()).casefold() for r in own),
            "retrieved_own_chunks": len(own)}
