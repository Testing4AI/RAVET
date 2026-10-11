"""Utilities for full-retention RAG representation-attack experiments."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentinelrag.testing.artifacts import CanaryArtifact
from sentinelrag.testing.matrix import PhysicalDocument, split_sentences


@dataclass(frozen=True)
class MaterializedTransformation:
    """One transformed representation of an intact logical canary artifact."""

    canary_id: str
    source_sha256: str
    variants: Mapping[str, tuple[str, ...]]
    metadata: Mapping[str, Any]


def normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def answers_preserved(texts: Sequence[str], answers: Sequence[str]) -> bool:
    """Require every answer string to remain present after normalization."""

    joined = normalized_text("\n".join(texts))
    return all(normalized_text(answer) in joined for answer in answers if answer.strip())


def word_chunks(text: str, *, size: int, overlap: int = 0) -> tuple[str, ...]:
    if size < 1:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")
    words = text.split()
    if not words:
        return ("",)
    stride = size - overlap
    return tuple(
        " ".join(words[start : start + size])
        for start in range(0, len(words), stride)
    )


def deterministic_variants(
    artifact: CanaryArtifact,
) -> dict[str, tuple[str, ...]]:
    """Represent intact knowledge with common storage/migration formats."""

    source = "\n".join(artifact.documents)
    fact_rows = [
        f"Question: {probe.question}\nAnswer: {probe.expected_answer}"
        for probe in artifact.probes
    ]
    return {
        "original": tuple(artifact.documents),
        "sentence_rechunk": tuple(split_sentences(source)),
        "word_rechunk_40": word_chunks(source, size=40, overlap=8),
        "fact_table": tuple(fact_rows),
    }


def source_sha256(artifact: CanaryArtifact) -> str:
    return hashlib.sha256("\n".join(artifact.documents).encode("utf-8")).hexdigest()


def load_materialized_transformations(
    path: str | Path,
) -> dict[str, MaterializedTransformation]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = {}
    for row in payload["artifacts"]:
        record = MaterializedTransformation(
            canary_id=row["canary_id"],
            source_sha256=row["source_sha256"],
            variants={
                name: tuple(texts) for name, texts in row["variants"].items()
            },
            metadata=row.get("metadata", {}),
        )
        records[record.canary_id] = record
    return records


def flatten_materialized_documents(
    artifacts: Sequence[CanaryArtifact],
    records: Mapping[str, MaterializedTransformation],
    variant: str,
) -> list[PhysicalDocument]:
    documents = []
    for artifact in artifacts:
        record = records[artifact.canary_id]
        if record.source_sha256 != source_sha256(artifact):
            raise ValueError(f"Source hash mismatch for {artifact.canary_id}")
        try:
            chunks = record.variants[variant]
        except KeyError as error:
            raise ValueError(
                f"Transformation {variant!r} missing for {artifact.canary_id}"
            ) from error
        documents.extend(
            PhysicalDocument(
                canary_id=artifact.canary_id,
                logical_index=0,
                text=chunk,
            )
            for chunk in chunks
        )
    return documents

