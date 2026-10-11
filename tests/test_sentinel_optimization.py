import copy

from sentinelrag.testing.prioritization import ProbeCandidate
from sentinelrag.testing.sentinel_optimization import (
    CANDIDATE_STRATEGIES,
    candidate_injection_payload,
    generation_prompt,
    supporting_document_rank,
    validate_candidates,
    validate_single_candidate,
)


def probes():
    return [
        ProbeCandidate(
            probe_id="p0",
            canary_id="c0",
            question="What code is recorded?",
            expected_answer="Luma Crest",
            canary_text="",
        ),
        ProbeCandidate(
            probe_id="p1",
            canary_id="c0",
            question="Which year is recorded?",
            expected_answer="2047",
            canary_text="",
        ),
    ]


def test_validate_candidates_requires_all_answers_and_strategies():
    document = "Luma Crest is the recorded code. The recorded year is 2047."
    payload = {
        "candidates": [
            {"strategy": strategy, "document": document}
            for strategy in CANDIDATE_STRATEGIES
        ]
    }
    candidates, errors = validate_candidates(
        payload,
        probes(),
        source_word_count=len(document.split()),
    )
    assert not errors
    assert [candidate.strategy for candidate in candidates] == list(
        CANDIDATE_STRATEGIES
    )


def test_validate_candidates_rejects_missing_answer():
    document = "Luma Crest is the recorded code."
    payload = {
        "candidates": [
            {"strategy": strategy, "document": document}
            for strategy in CANDIDATE_STRATEGIES
        ]
    }
    candidates, errors = validate_candidates(
        payload,
        probes(),
        source_word_count=len(document.split()),
    )
    assert not candidates
    assert any("supports 1/2 answers" in error for error in errors)


def test_validate_single_candidate_and_prompt_strategy():
    document = "Luma Crest is the recorded code. The recorded year is 2047."
    candidate, errors = validate_single_candidate(
        {"strategy": "fact_first", "document": document},
        probes(),
        strategy="fact_first",
        source_word_count=len(document.split()),
    )
    assert candidate is not None
    assert not errors
    prompt = generation_prompt(document, probes(), strategy="fact_first")
    assert "Produce only the fact_first strategy" in prompt


def test_unconstrained_rewrite_prompt_avoids_relation_optimization():
    document = "Luma Crest is the recorded code. The recorded year is 2047."
    prompt = generation_prompt(
        document,
        probes(),
        strategy="unconstrained_rewrite",
    )
    assert "Produce only the unconstrained_rewrite strategy" in prompt
    assert "Do not deliberately make the required relations more explicit" in prompt


def test_candidate_injection_payload_does_not_mutate_baseline():
    baseline = {
        "watermark_texts": ["old"],
        "watermark_expansion_data": [{"watermark_text": "old"}],
        "injection_metadata": {},
    }
    original = copy.deepcopy(baseline)
    rewritten = candidate_injection_payload(
        baseline,
        ["new"],
        strategy="fact_first",
        generation_model="model",
    )
    assert baseline == original
    assert rewritten["watermark_texts"] == ["new"]
    assert rewritten["watermark_expansion_data"][0]["watermark_text"] == "new"


def test_supporting_document_rank_uses_best_support_and_sibling_chunks():
    rank = supporting_document_rank(
        clean_scores=[0.9, 0.7, 0.4],
        own_document_scores=[0.8, 0.6, 0.95],
        support_offsets=[0, 1],
    )
    assert rank == 3
