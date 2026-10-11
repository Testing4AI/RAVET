import pytest

from sentinelrag.testing.corpus_migrations import text_sha256
from sentinelrag.testing.sentinel_paraphrase import cumulative_metrics, paraphrase_prompt, validate_paraphrases


def test_paraphraser_gets_source_text_only():
    prompt = paraphrase_prompt(["Axon-17 inhibits RQX-4."])
    assert "Axon-17 inhibits RQX-4." in prompt
    assert "expected_answer" not in prompt and "probe" not in prompt
    assert "Do not summarize" in prompt


def test_validation_does_not_require_secret_answer_strings():
    source = "The policy requires twenty participants."
    value = {text_sha256(source): "Participation by 20 people is mandatory under the policy."}
    assert validate_paraphrases(value, [source]) == value
    with pytest.raises(ValueError):
        validate_paraphrases({text_sha256(source): source}, [source])
    with pytest.raises(ValueError):
        validate_paraphrases({"wrong": "Something else."}, [source])


def test_success_and_cost_use_same_cumulative_event():
    metrics, curve = cumulative_metrics([True] * 10 + [False] * 20, .15, .01)
    assert metrics["vsr15"] == metrics["vsr30"] == 1
    assert metrics["mq15"] == metrics["mq30"]
    assert metrics["mq15"] == sum(1 - (curve[i - 1]["anytime_verified"] if i else 0) for i in range(15))
    with pytest.raises(ValueError):
        cumulative_metrics([True] * 15, .15, .01)
