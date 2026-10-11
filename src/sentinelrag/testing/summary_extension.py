"""Blind summary validation and prefix-only replay for frozen ownership tests."""

import json
import math

from sentinelrag.testing.corpus_migrations import text_sha256
from sentinelrag.testing.orthogonal_policy import next_selected_probe


def summary_prompt(documents, ratio):
    rows = [{"id": text_sha256(text), "max_words": max(1, math.ceil(len(text.split()) * ratio)),
             "text": text} for text in documents]
    return (
        "Apply routine summary preprocessing independently to each source document below. "
        "Retain concrete entities, relations, numbers, qualifications and conclusions where possible; "
        "remove repetition and rhetorical framing. Do not add facts, explanations or labels. "
        "Each summary MUST stay within its specified max_words (whitespace-separated words). "
        "Return only a JSON object mapping each source id to its nonempty summary string.\n"
        "Source documents:\n" + json.dumps(rows, ensure_ascii=True)
    )


def validate_summaries(value, documents, ratio):
    expected = {text_sha256(text): text for text in documents}
    if not isinstance(value, dict) or value.keys() != expected.keys():
        raise ValueError("Summary document identities differ from frozen inputs")
    for key, source in expected.items():
        result = value[key]
        limit = max(1, math.ceil(len(source.split()) * ratio))
        if not isinstance(result, str) or not result.strip() or len(result.split()) > limit:
            raise ValueError("Empty or insufficiently compressed summary")
    return {key: text.strip() for key, text in value.items()}


def concise_summary_prompt(document, ratio, target_words):
    row = {"id": text_sha256(document),
           "max_words": max(1, math.ceil(len(document.split()) * ratio)), "text": document}
    return (
        f"Write a concise factual summary in at most {target_words} whitespace-separated words. "
        "This is summarization, not sentence-by-sentence paraphrasing. Focus on the main finding; "
        "omit secondary details, introductory framing, lists and repetition. Do not invent facts. "
        "Return only a JSON object mapping the source id to a summary string.\n"
        "Source documents:\n" + json.dumps([row], ensure_ascii=True)
    )


def causal_replay(selected, clusters, records, *, seed, p0, limit):
    by_id = {r["probe_id"]: r for r in records}
    expected = {p.profile.candidate.probe_id for p in selected}
    if len(by_id) != len(records) or by_id.keys() != expected:
        raise ValueError("Replay requires one complete answer per frozen test")
    prefix = []
    for _ in range(min(limit, len(selected))):
        probe = next_selected_probe(selected, clusters, "bayes", prefix, seed=seed, p0=p0)
        prefix.append(by_id[probe.profile.candidate.probe_id])
    return prefix
