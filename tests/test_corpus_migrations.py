import json

import pytest

from sentinelrag.testing.corpus_migrations import (
    deterministic_corpus_variants,
    load_corpus_migration_manifest,
    parse_migration_response,
    text_sha256,
    validate_uniform_manifest,
)


def test_deterministic_corpus_variants_preserve_source_words():
    text = " ".join(f"word{index}" for index in range(250))
    variants = deterministic_corpus_variants(text)
    assert variants["original"] == (text,)
    assert len(variants["word_rechunk_120_30"]) == 3
    assert variants["word_rechunk_120_30"][0].split()[:2] == ["word0", "word1"]


def test_uniform_manifest_validates_identity_and_hash(tmp_path):
    clean = {"d0": "clean text"}
    sentinel = {"c0": "sentinel text"}
    rows = [
        {
            "source_type": source_type,
            "source_id": source_id,
            "source_sha256": text_sha256(text),
            "variants": {"original": [text]},
        }
        for source_type, source_id, text in (
            ("clean", "d0", clean["d0"]),
            ("sentinel", "c0", sentinel["c0"]),
        )
    ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"documents": rows}), encoding="utf-8")
    records = load_corpus_migration_manifest(path)
    validate_uniform_manifest(records, clean, sentinel)

    with pytest.raises(ValueError, match="identity mismatch"):
        validate_uniform_manifest(records[:-1], clean, sentinel)


def test_parse_migration_response_extracts_tagged_blocks():
    paraphrase, summary = parse_migration_response(
        "<PARAPHRASE>Long rewrite.</PARAPHRASE>\n"
        "<COMPRESSED_SUMMARY>Short facts.</COMPRESSED_SUMMARY>"
    )
    assert paraphrase == "Long rewrite."
    assert summary == "Short facts."


def test_parse_migration_response_tolerates_misspelled_paraphrase_close_tag():
    paraphrase, summary = parse_migration_response(
        "<PARAPHRASE>Long rewrite.</PARAPHRESE>\n"
        "<COMPRESSED_SUMMARY>Short facts.</COMPRESSED_SUMMARY>"
    )
    assert paraphrase == "Long rewrite."
    assert summary == "Short facts."
