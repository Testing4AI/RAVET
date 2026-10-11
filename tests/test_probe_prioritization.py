import json

import pytest

from sentinelrag.testing.prioritization import (
    ProbeCandidate,
    ProbeProfile,
    RetrievalObservation,
    answer_tokens_supported,
    evaluate_ordering,
    load_sentinelrag_candidates,
    normalized_answer_match,
    rank_probes,
    score_profiles,
)
from sentinelrag.utils.stats import bernoulli_mixture_e_value


def test_answer_token_support_tolerates_hyphens_plurals_and_intervening_words():
    assert answer_tokens_supported(
        "The 5.7-month outcome increased lactate threshold and lean body mass.",
        "5.7 months",
    )
    assert answer_tokens_supported(
        "The regimen increased lactate threshold and lean body mass.",
        "increased lean body mass",
    )
    assert not answer_tokens_supported("The response was 70%.", "5.7 months")


def _profile(probe_id, canary_id, target_distance, clean_distance, target_rank):
    return ProbeProfile(
        candidate=ProbeCandidate(
            probe_id=probe_id,
            canary_id=canary_id,
            question=f"Question {probe_id}?",
            expected_answer="answer",
            canary_text="Canary text with answer.",
        ),
        observations=[
            RetrievalObservation(
                variant="original",
                clean_best_distance=clean_distance,
                target_distance=target_distance,
                target_rank=target_rank,
            )
        ],
    )


def test_sentinelrag_adapter_emits_generic_candidates(tmp_path):
    source = tmp_path / "injection.json"
    source.write_text(
        json.dumps(
            {
                "watermark_texts": ["The hidden threshold is 42."],
                "watermark_questions": [
                    {
                        "watermark_index": 0,
                        "questions": ["What is the hidden threshold?"],
                        "correct_answers": ["42"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_sentinelrag_candidates(source)

    assert len(candidates) == 1
    assert candidates[0].canary_id == "c000"
    assert candidates[0].expected_answer == "42"


def test_proposed_score_rewards_clean_separation():
    cleanly_separated = _profile("good", "c0", 0.10, 0.80, 1)
    clean_collision = _profile("collision", "c1", 0.10, 0.12, 1)
    profiles = [cleanly_separated, clean_collision]

    score_profiles(profiles)

    assert cleanly_separated.similarity_score == pytest.approx(
        clean_collision.similarity_score
    )
    assert cleanly_separated.proposed_score > clean_collision.proposed_score


def test_ranking_keeps_only_one_probe_per_canary():
    profiles = [
        _profile("c0-low", "c0", 0.4, 0.8, 3),
        _profile("c0-high", "c0", 0.1, 0.8, 1),
        _profile("c1", "c1", 0.2, 0.8, 1),
    ]
    score_profiles(profiles)

    ordered = rank_probes(profiles, "proposed")

    assert len(ordered) == 2
    assert {item.candidate.canary_id for item in ordered} == {"c0", "c1"}
    assert any(item.candidate.probe_id == "c0-high" for item in ordered)


def test_failure_repair_keeps_first_probe_when_shadow_coverage_ties():
    first = _profile("c0-p00", "c0", 0.3, 0.8, 2)
    first.candidate.metadata["question_index"] = 0
    alternative = _profile("c0-p01", "c0", 0.1, 0.8, 1)
    alternative.candidate.metadata["question_index"] = 1
    profiles = [first, alternative]
    score_profiles(profiles)

    ordered = rank_probes(profiles, "failure_repair")

    assert [row.candidate.probe_id for row in ordered] == ["c0-p00"]


def test_failure_repair_switches_when_candidate_repairs_shadow_failure():
    first = _profile("c0-p00", "c0", None, 0.8, None)
    first.candidate.metadata["question_index"] = 0
    alternative = _profile("c0-p01", "c0", 0.1, 0.8, 1)
    alternative.candidate.metadata["question_index"] = 1
    profiles = [first, alternative]
    score_profiles(profiles)

    ordered = rank_probes(profiles, "failure_repair")

    assert [row.candidate.probe_id for row in ordered] == ["c0-p01"]


def test_failure_repair_ignores_single_variant_gain():
    def profile(probe_id, question_index, ranks):
        return ProbeProfile(
            candidate=ProbeCandidate(
                probe_id=probe_id,
                canary_id="c0",
                question="Question?",
                expected_answer="answer",
                canary_text="Canary text with answer.",
                metadata={"question_index": question_index},
            ),
            observations=[
                RetrievalObservation(
                    variant=f"v{index}",
                    clean_best_distance=0.8,
                    target_distance=0.1 if rank is not None else None,
                    target_rank=rank,
                )
                for index, rank in enumerate(ranks)
            ],
        )

    first = profile("c0-p00", 0, [1, 1, 1, None])
    alternative = profile("c0-p01", 1, [1, 1, 1, 1])
    profiles = [first, alternative]
    score_profiles(profiles)

    ordered = rank_probes(profiles, "failure_repair")

    assert [row.candidate.probe_id for row in ordered] == ["c0-p00"]


def test_sequential_evaluation_uses_distinct_ordered_outcomes():
    profiles = [_profile(f"p{i}", f"c{i}", 0.1, 0.8, 1) for i in range(3)]
    score_profiles(profiles)
    outcomes = {"p0": True, "p1": True, "p2": False}

    result = evaluate_ordering(profiles, outcomes, p0=0.02, alpha=0.01)

    assert result["num_successes"] == 2
    assert result["queries_to_significance"] == 2
    assert result["curve"][1]["p_value"] == pytest.approx(0.0004)
    assert result["queries_to_anytime_significance"] is not None
    assert result["curve"][1]["e_value"] > 0


def test_anytime_rejection_remains_true_after_e_value_falls():
    profiles = [_profile(f"p{i}", f"c{i}", 0.1, 0.8, 1) for i in range(5)]
    score_profiles(profiles)
    outcomes = {"p0": True, "p1": True}

    result = evaluate_ordering(profiles, outcomes, p0=0.02, alpha=0.01)

    assert result["curve"][1]["e_value"] >= 100.0
    assert result["curve"][4]["e_value"] < 100.0
    assert result["curve"][4]["anytime_significant"] is True


def test_bernoulli_mixture_e_value_is_anytime_evidence():
    assert bernoulli_mixture_e_value(0, 0, 0.02) == pytest.approx(1.0)
    assert bernoulli_mixture_e_value(0, 5, 0.02) < 1.0
    assert bernoulli_mixture_e_value(5, 5, 0.02) > 100.0


@pytest.mark.parametrize(
    "successes,trials,p0",
    [(-1, 1, 0.02), (2, 1, 0.02), (1, 1, 0.0), (1, 1, 1.0)],
)
def test_bernoulli_mixture_e_value_rejects_invalid_inputs(
    successes, trials, p0
):
    with pytest.raises(ValueError):
        bernoulli_mixture_e_value(successes, trials, p0)


def test_normalized_answer_oracle():
    assert normalized_answer_match("The threshold is > 50 ng/mL.", "> 50 ng/mL")
    assert not normalized_answer_match("Cannot answer", "> 50 ng/mL")
    assert normalized_answer_match("The graph has n vertices.", "n")
    assert not normalized_answer_match("Which function is exponential?", "n")
    assert normalized_answer_match("The complexity is Õ(n²).", "Õ(n²)")
    assert not normalized_answer_match(
        "Does the graph contain a planted clique on n vertices?", "Õ(n²)"
    )
