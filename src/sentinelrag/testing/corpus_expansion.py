"""Outcome-blind, nested corpus expansion with disjoint calibration samples."""

import random
import re
from dataclasses import replace

from sentinelrag.testing.corpus_migrations import text_sha256


def normalized_text_hash(text):
    return text_sha256(re.sub(r"\s+", " ", text).strip().casefold())


def donor_split(documents, excluded_texts, *, donor, seed, evaluation_count, calibration_count):
    excluded = {normalized_text_hash(text) for text in excluded_texts}
    seen, eligible = set(excluded), []
    for document in sorted(documents, key=lambda d: d.source_id):
        key = normalized_text_hash(document.text)
        if not document.text.strip() or key in seen:
            continue
        seen.add(key)
        eligible.append(replace(document, source_id=f"external:{donor}:{document.source_id}"))
    random.Random(seed).shuffle(eligible)
    if len(eligible) < evaluation_count + calibration_count:
        raise ValueError("Not enough unique, disjoint donor documents")
    return (eligible[:evaluation_count],
            eligible[evaluation_count:evaluation_count + calibration_count])


def expanded_documents(base, donors, factor, *, ordinary_count):
    if factor < 1 or factor != int(factor):
        raise ValueError("Expansion factor must be a positive integer")
    if sum(d.source_type == "clean" for d in base) != ordinary_count:
        raise ValueError("Unexpected frozen ordinary-document count")
    count = (factor - 1) * ordinary_count
    if len(donors) < count:
        raise ValueError("Incomplete expansion pool")
    result = list(base) + list(donors[:count])
    if len({(d.source_type, d.source_id, d.chunk_index) for d in result}) != len(result):
        raise ValueError("Document identity collision")
    return result
