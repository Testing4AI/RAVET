"""Load immutable, explicitly identified ordinary-document KB blocks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from sentinelrag.testing.matrix import _cosine_matrix


def text_fingerprint(text):
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()


def load_corpus_subset(path, *, dataset=None):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if dataset is not None and payload["dataset"] != dataset:
        raise ValueError("Corpus subset dataset mismatch")
    corpus = {}
    fingerprints = set()
    for row in payload["documents"]:
        key, text = row["document_id"], row["text"]
        fingerprint = text_fingerprint(text)
        if not text.strip() or key in corpus or fingerprint in fingerprints:
            raise ValueError("Empty or duplicate corpus subset document")
        if fingerprint != row["text_fingerprint"]:
            raise ValueError("Corpus subset content hash mismatch")
        corpus[key] = {"title": row.get("title", ""), "text": text}
        fingerprints.add(fingerprint)
    if len(corpus) != payload["document_count"]:
        raise ValueError("Corpus subset count mismatch")
    return corpus


class ExactSubsetCollection:
    """Read-only clean-background ranking on explicitly selected saved vectors."""

    def __init__(self, embeddings):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if (
            self.embeddings.ndim != 2
            or not len(self.embeddings)
            or not np.isfinite(self.embeddings).all()
        ):
            raise ValueError("Invalid clean subset embeddings")

    def count(self):
        return len(self.embeddings)

    def query(self, *, query_embeddings, n_results, include):
        if include != ["distances"] or n_results < 1:
            raise ValueError("Only positive-depth distance queries are supported")
        scores = _cosine_matrix(query_embeddings, self.embeddings)
        top = np.argsort(-scores, axis=1, kind="stable")[:, :n_results]
        return {"distances": (1 - np.take_along_axis(scores, top, axis=1)).tolist()}
