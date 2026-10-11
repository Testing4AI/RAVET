"""Uniform corpus-migration records for realistic RAG preprocessing tests."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from sentinelrag.testing.representation_attacks import word_chunks


@dataclass(frozen=True)
class CorpusMigrationRecord:
    """Uniformly transformed clean or sentinel source document."""

    source_type: str
    source_id: str
    source_sha256: str
    variants: Mapping[str, tuple[str, ...]]
    metadata: Mapping[str, object]


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_corpus_variants(text: str) -> dict[str, tuple[str, ...]]:
    """Apply common deterministic ingestion policies without content targeting."""

    return {
        "original": (text,),
        "word_rechunk_120_30": word_chunks(text, size=120, overlap=30),
    }


def parse_migration_response(content: str) -> tuple[str, str]:
    """Extract paraphrase and summary blocks from a model response."""

    paraphrase_match = re.search(
        r"<PARAPHRASE>\s*(.*?)\s*</PARAPH[^>]*>",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    summary_match = re.search(
        r"<COMPRESSED_SUMMARY>\s*(.*?)\s*</COMPRESSED_SUMMARY>",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not paraphrase_match or not summary_match:
        return "", ""
    return paraphrase_match.group(1).strip(), summary_match.group(1).strip()


def load_corpus_migration_manifest(
    path: str | Path,
) -> list[CorpusMigrationRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for row in payload["documents"]:
        records.append(
            CorpusMigrationRecord(
                source_type=row["source_type"],
                source_id=str(row["source_id"]),
                source_sha256=row["source_sha256"],
                variants={
                    name: tuple(texts)
                    for name, texts in row["variants"].items()
                },
                metadata=row.get("metadata", {}),
            )
        )
    return records


def validate_uniform_manifest(
    records: Sequence[CorpusMigrationRecord],
    clean_sources: Mapping[str, str],
    sentinel_sources: Mapping[str, str],
) -> None:
    expected = {
        **{("clean", str(key)): value for key, value in clean_sources.items()},
        **{
            ("sentinel", str(key)): value
            for key, value in sentinel_sources.items()
        },
    }
    observed = {(record.source_type, record.source_id): record for record in records}
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise ValueError(
            f"Migration manifest identity mismatch: missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    for identity, source in expected.items():
        if observed[identity].source_sha256 != text_sha256(source):
            raise ValueError(f"Migration source hash mismatch for {identity}")
