import json

from sentinelrag.testing.artifacts import build_artifact_pools


def test_all_schemes_share_the_generic_probe_contract(tmp_path):
    source = tmp_path / "injection.json"
    source.write_text(
        json.dumps(
            {
                "watermark_texts": ["Trial Orion reported a threshold of 42 units."],
                "watermark_questions": [
                    {
                        "watermark_index": 0,
                        "questions": ["What threshold did Trial Orion report?"],
                        "correct_answers": ["42 units"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    pools = build_artifact_pools(source)

    assert set(pools) == {
        "factual",
        "lexical",
        "compositional",
        "lexical_v2",
        "compositional_v2",
    }
    assert all(len(artifacts) == 1 for artifacts in pools.values())
    assert len(pools["lexical"][0].probes) == 5
    assert len(pools["compositional"][0].documents) == 2
    compositional_probe = pools["compositional"][0].probes[0]
    assert compositional_probe.metadata["required_document_indices"] == [0, 1]
    assert len(compositional_probe.metadata["required_evidence"]) == 2
    lexical_v2 = pools["lexical_v2"][0]
    assert len(lexical_v2.probes) == 5
    assert all(
        probe.expected_answer not in probe.question for probe in lexical_v2.probes
    )
    compositional_v2 = pools["compositional_v2"][0]
    assert len(compositional_v2.documents) == 4
    assert len(compositional_v2.probes) == 5
    for probe in compositional_v2.probes:
        assert probe.metadata["required_document_indices"] == [0, 1]
        assert probe.metadata["distractor_document_indices"] == [2, 3]
        assert probe.expected_answer in compositional_v2.documents[1]
        assert probe.expected_answer not in compositional_v2.documents[2]
        assert probe.expected_answer not in compositional_v2.documents[3]
