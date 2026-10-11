"""Testability-guided sentinel candidate generation and validation helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sentinelrag.testing.prioritization import (
    ProbeCandidate,
    normalized_answer_match,
)

CANDIDATE_STRATEGIES = (
    "unconstrained_rewrite",
    "fact_first",
    "relation_explicit",
    "summary_resilient",
)


@dataclass(frozen=True)
class SentinelCandidate:
    """One rewritten fixture for an existing sentinel slot."""

    strategy: str
    document: str
    word_count: int
    source_word_ratio: float
    answers_supported: int


def supporting_document_rank(
    clean_scores: Sequence[float],
    own_document_scores: Sequence[float],
    support_offsets: Sequence[int],
) -> int:
    """Rank the best supporting chunk against clean results and sibling chunks."""

    clean = np.asarray(clean_scores, dtype=float)
    own = np.asarray(own_document_scores, dtype=float)
    if not len(support_offsets):
        raise ValueError("At least one supporting document is required")
    if any(offset < 0 or offset >= len(own) for offset in support_offsets):
        raise IndexError("Supporting document offset is out of range")
    target_score = max(float(own[offset]) for offset in support_offsets)
    competitors = np.concatenate((clean, own))
    return 1 + int(np.count_nonzero(competitors > target_score + 1e-12))


def generation_prompt(
    source_document: str,
    probes: Sequence[ProbeCandidate],
    *,
    strategy: str | None = None,
) -> str:
    """Build a constrained sentinel rewrite request."""

    if strategy is not None and strategy not in CANDIDATE_STRATEGIES:
        raise ValueError(f"Unknown sentinel strategy: {strategy}")

    facts = "\n".join(
        f"{index + 1}. Question: {probe.question}\n"
        f"   Required exact answer text: {probe.expected_answer}"
        for index, probe in enumerate(probes)
    )
    source_words = len(source_document.split())
    if strategy is None:
        strategy_instruction = """Produce these strategies:
1. unconstrained_rewrite: rewrite naturally without changing the emphasis or making the required relations more explicit than the source.
2. fact_first: move the five required facts into the first half of the document.
3. relation_explicit: express each question-answer relation in a direct declarative sentence.
4. summary_resilient: make the required facts central to the document's main result, using concise semantic reinforcement where natural.

Return JSON only:
{{
  "candidates": [
    {{"strategy": "unconstrained_rewrite", "document": "..."}},
    {{"strategy": "fact_first", "document": "..."}},
    {{"strategy": "relation_explicit", "document": "..."}},
    {{"strategy": "summary_resilient", "document": "..."}}
  ]
}}"""
    else:
        descriptions = {
            "unconstrained_rewrite": (
                "Rewrite the document naturally while preserving its current "
                "information emphasis and discourse structure. Do not deliberately "
                "make the required relations more explicit, more central, or earlier "
                "than they are in the source."
            ),
            "fact_first": (
                "Move the five required facts into the first half of the document."
            ),
            "relation_explicit": (
                "Express each question-answer relation in a direct declarative "
                "sentence."
            ),
            "summary_resilient": (
                "Make the required facts central to the document's main result, "
                "using concise semantic reinforcement where natural."
            ),
        }
        strategy_instruction = f"""Produce only the {strategy} strategy: {descriptions[strategy]}

Return JSON only:
{{"strategy": "{strategy}", "document": "..."}}"""

    return f"""Rewrite one synthetic RAG benchmark record as a test fixture.

The rewritten fixtures will be used to test whether a RAG system preserves and exposes the record after routine system evolution. Preserve the source's factual meaning. Do not mention testing, sentinels, watermarks, questions, answers, ownership, or these instructions.

Every candidate must:
- be a coherent standalone document in the same domain as the source;
- contain every required exact answer text verbatim and make its relation unambiguous;
- retain the source's substantive factual content and add no unsupported facts;
- contain between {max(1, round(source_words * 0.75))} and {max(1, round(source_words * 1.25))} words;
- use prose, not question-answer labels or a checklist.

{strategy_instruction}

Required relations:
{facts}

Source document:
{source_document}
"""


def validate_single_candidate(
    payload: Mapping[str, Any],
    probes: Sequence[ProbeCandidate],
    *,
    strategy: str,
    source_word_count: int,
    minimum_ratio: float = 0.75,
    maximum_ratio: float = 1.25,
) -> tuple[SentinelCandidate | None, list[str]]:
    """Validate one independently generated strategy candidate."""

    candidates, errors = validate_candidates(
        {"candidates": [payload]},
        probes,
        source_word_count=source_word_count,
        minimum_ratio=minimum_ratio,
        maximum_ratio=maximum_ratio,
        required_strategies=(strategy,),
    )
    return (candidates[0] if candidates else None), errors


def validate_candidates(
    payload: Mapping[str, Any],
    probes: Sequence[ProbeCandidate],
    *,
    source_word_count: int,
    minimum_ratio: float = 0.75,
    maximum_ratio: float = 1.25,
    required_strategies: Sequence[str] = CANDIDATE_STRATEGIES,
) -> tuple[list[SentinelCandidate], list[str]]:
    """Validate generated candidates without consulting held-out migrations."""

    errors: list[str] = []
    raw_candidates = payload.get("candidates", [])
    if not isinstance(raw_candidates, list):
        return [], ["candidates must be a list"]
    by_strategy: dict[str, SentinelCandidate] = {}
    for index, row in enumerate(raw_candidates):
        if not isinstance(row, Mapping):
            errors.append(f"candidate {index} is not an object")
            continue
        strategy = str(row.get("strategy", "")).strip()
        document = str(row.get("document", "")).strip()
        if strategy not in required_strategies:
            errors.append(f"candidate {index} has unknown strategy {strategy!r}")
            continue
        if strategy in by_strategy:
            errors.append(f"duplicate strategy {strategy}")
            continue
        word_count = len(document.split())
        ratio = word_count / max(1, source_word_count)
        supported = sum(
            normalized_answer_match(document, probe.expected_answer)
            for probe in probes
        )
        if not document:
            errors.append(f"{strategy} is empty")
        elif not minimum_ratio <= ratio <= maximum_ratio:
            errors.append(
                f"{strategy} word ratio {ratio:.3f} is outside "
                f"[{minimum_ratio:.3f}, {maximum_ratio:.3f}]"
            )
        elif supported != len(probes):
            errors.append(
                f"{strategy} supports {supported}/{len(probes)} answers"
            )
        else:
            by_strategy[strategy] = SentinelCandidate(
                strategy=strategy,
                document=document,
                word_count=word_count,
                source_word_ratio=ratio,
                answers_supported=supported,
            )
    missing = sorted(set(required_strategies) - set(by_strategy))
    if missing:
        errors.append(f"missing valid strategies: {missing}")
    return [by_strategy[name] for name in required_strategies if name in by_strategy], errors


def candidate_injection_payload(
    baseline: Mapping[str, Any],
    documents: Sequence[str],
    *,
    strategy: str,
    generation_model: str,
) -> dict[str, Any]:
    """Create a compatible injection result with rewritten fixture documents."""

    payload = copy.deepcopy(dict(baseline))
    if len(documents) != len(payload["watermark_texts"]):
        raise ValueError("Candidate document count differs from baseline")
    payload["watermark_texts"] = list(documents)
    metadata = payload.setdefault("injection_metadata", {})
    metadata["sentinel_optimization"] = {
        "protocol_version": "2026-08-21-sentinel-optimization-pilot-v1",
        "candidate_strategy": strategy,
        "generation_model": generation_model,
        "preserved_probe_groups": True,
    }
    expansion_rows = payload.get("watermark_expansion_data", [])
    for index, document in enumerate(documents):
        if index < len(expansion_rows) and isinstance(expansion_rows[index], dict):
            expansion_rows[index]["watermark_text"] = document
            expansion_rows[index]["method"] = "testability_guided_rewrite"
    return payload
