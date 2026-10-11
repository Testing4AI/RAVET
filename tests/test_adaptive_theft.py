from types import SimpleNamespace

from sentinelrag.testing.adaptive import (
    PreparedProbe,
    adaptive_bayes_ordering,
    cluster_diverse_ordering,
    cluster_thompson_ordering,
    evaluate_policy,
    paired_signflip_pvalue,
    random_candidate_ordering,
    selected_random_ordering,
    sentinelrag_original_ordering,
    simulate_attack,
    source_similarity_ordering,
    stable_seed,
)


def make_probe(
    canary, probe, proposed, probability, success=True, question_index=0
):
    profile = SimpleNamespace(
        candidate=SimpleNamespace(
            canary_id=canary,
            probe_id=probe,
            metadata={"question_index": question_index},
        ),
        similarity_score=proposed,
        proposed_score=proposed,
    )
    return PreparedProbe(profile, success, probability)


def test_stable_seed_fits_sklearn_random_state():
    seed = stable_seed("v2", "nfcorpus", "factual")
    assert 0 <= seed <= 2**32 - 1
    assert seed == stable_seed("v2", "nfcorpus", "factual")


def test_paired_signflip_detects_consistent_direction():
    assert paired_signflip_pvalue([1.0] * 20, draws=5_000) < 0.001
    assert paired_signflip_pvalue([1.0, -1.0] * 10, draws=5_000) > 0.5


def test_cluster_diverse_round_robins_clusters():
    probes = [
        make_probe("a", "a1", 0.9, 0.8),
        make_probe("b", "b1", 0.8, 0.8),
        make_probe("c", "c1", 0.7, 0.8),
        make_probe("d", "d1", 0.6, 0.8),
    ]
    clusters = {"a": 0, "b": 0, "c": 1, "d": 1}
    ordering = cluster_diverse_ordering(probes, clusters)
    assert [row.profile.candidate.canary_id for row in ordering] == [
        "a",
        "c",
        "b",
        "d",
    ]


def test_sentinelrag_original_uses_first_question_per_canary():
    probes = [
        make_probe("a", "a-p2", 0.9, 0.9, question_index=2),
        make_probe("a", "a-p0", 0.1, 0.1, question_index=0),
        make_probe("b", "b-p1", 0.8, 0.8, question_index=1),
        make_probe("b", "b-p0", 0.2, 0.2, question_index=0),
    ]
    ordering = sentinelrag_original_ordering(probes, seed=7)
    assert {row.profile.candidate.probe_id for row in ordering} == {
        "a-p0",
        "b-p0",
    }
    assert [row.profile.candidate.probe_id for row in ordering] == [
        row.profile.candidate.probe_id
        for row in sentinelrag_original_ordering(probes, seed=7)
    ]


def test_selected_random_chooses_best_candidate_but_randomizes_artifacts():
    probes = [
        make_probe("a", "a-low", 0.1, 0.1, question_index=0),
        make_probe("a", "a-high", 0.9, 0.9, question_index=1),
        make_probe("b", "b-low", 0.2, 0.2, question_index=0),
        make_probe("b", "b-high", 0.8, 0.8, question_index=1),
    ]
    ordering = selected_random_ordering(probes, seed=7)
    assert {row.profile.candidate.probe_id for row in ordering} == {
        "a-high",
        "b-high",
    }


def test_simple_main_baselines_select_one_probe_per_canary():
    probes = [
        make_probe("a", "a-low", 0.1, 0.1, question_index=0),
        make_probe("a", "a-high", 0.9, 0.9, question_index=1),
        make_probe("b", "b-low", 0.2, 0.2, question_index=0),
        make_probe("b", "b-high", 0.8, 0.8, question_index=1),
    ]

    random_order = random_candidate_ordering(probes, seed=7)
    similarity_order = source_similarity_ordering(probes)

    assert len(random_order) == len(similarity_order) == 2
    assert len({row.profile.candidate.canary_id for row in random_order}) == 2
    assert {row.profile.candidate.probe_id for row in similarity_order} == {
        "a-high",
        "b-high",
    }


def test_adaptive_policy_exploits_cluster_after_success():
    probes = [
        make_probe("a", "a1", 0.9, 0.9),
        make_probe("b", "b1", 0.8, 0.8),
        make_probe("c", "c1", 0.7, 0.85),
        make_probe("d", "d1", 0.6, 0.7),
    ]
    clusters = {"a": 0, "b": 0, "c": 1, "d": 1}
    ordering = adaptive_bayes_ordering(
        probes, clusters, {"a", "b"}, prior_retention=0.5
    )
    assert [row.profile.candidate.canary_id for row in ordering[:2]] == ["a", "b"]


def test_cluster_thompson_is_seeded_and_uses_one_probe_per_artifact():
    probes = [
        make_probe("a", "a1", 0.9, 0.9),
        make_probe("b", "b1", 0.8, 0.8),
        make_probe("c", "c1", 0.7, 0.7),
        make_probe("d", "d1", 0.6, 0.6),
    ]
    clusters = {"a": 0, "b": 0, "c": 1, "d": 1}
    first = cluster_thompson_ordering(
        probes, clusters, {"a", "b"}, seed=7
    )
    second = cluster_thompson_ordering(
        probes, clusters, {"a", "b"}, seed=7
    )
    assert [row.profile.candidate.probe_id for row in first] == [
        row.profile.candidate.probe_id for row in second
    ]
    assert len({row.profile.candidate.canary_id for row in first}) == 4


def test_evaluation_counts_distinct_discovered_clusters():
    probes = [
        make_probe("a", "a1", 0.9, 0.9),
        make_probe("b", "b1", 0.8, 0.8),
        make_probe("c", "c1", 0.7, 0.7),
    ]
    clusters = {"a": 0, "b": 0, "c": 1}
    result = evaluate_policy(
        probes, {"a", "b", "c"}, clusters, budgets=[2, 3]
    )
    assert result["budgets"]["2"]["successes"] == 2
    assert result["budgets"]["2"]["retained_cluster_discovery"] == 1
    assert result["budgets"]["3"]["retained_cluster_discovery"] == 2


def test_external_target_labels_override_retrieval_proxy_and_keep_clean_hits():
    probes = [
        make_probe("a", "a1", 0.9, 0.9, success=True),
        make_probe("b", "b1", 0.8, 0.8, success=False),
    ]
    clusters = {"a": 0, "b": 1}
    result = evaluate_policy(
        probes,
        {"a"},
        clusters,
        budgets=[2],
        retained_success_by_probe={"a1": False, "b1": False},
        removed_success_by_probe={"a1": False, "b1": True},
    )
    assert [row["success"] for row in result["curve"]] == [False, True]
    assert result["budgets"]["2"]["successes"] == 1


def test_adaptive_policy_updates_from_external_target_labels():
    probes = [
        make_probe("a", "a1", 0.9, 0.9, success=False),
        make_probe("b", "b1", 0.8, 0.8, success=False),
        make_probe("c", "c1", 0.7, 0.85, success=True),
    ]
    clusters = {"a": 0, "b": 0, "c": 1}
    ordering = adaptive_bayes_ordering(
        probes,
        clusters,
        {"a", "b"},
        prior_retention=0.5,
        retained_success_by_probe={"a1": True, "b1": True, "c1": False},
        removed_success_by_probe={"a1": False, "b1": False, "c1": False},
    )
    assert [row.profile.candidate.canary_id for row in ordering[:2]] == ["a", "b"]


def test_q0_adaptive_keeps_first_candidate_per_artifact():
    probes = [
        make_probe("a", "a-p0", 0.1, 0.1, question_index=0),
        make_probe("a", "a-p1", 0.9, 0.9, question_index=1),
        make_probe("b", "b-p0", 0.2, 0.2, question_index=0),
        make_probe("b", "b-p1", 0.8, 0.8, question_index=1),
    ]
    clusters = {"a": 0, "b": 1}
    ordering = adaptive_bayes_ordering(
        probes,
        clusters,
        {"a"},
        prior_retention=0.5,
        probe_selector="first",
    )
    assert {row.profile.candidate.probe_id for row in ordering} == {
        "a-p0",
        "b-p0",
    }


def test_paired_attack_uses_same_retained_set_for_all_policies():
    probes = [
        make_probe("a", "a1", 0.9, 0.9),
        make_probe("b", "b1", 0.8, 0.8),
        make_probe("c", "c1", 0.7, 0.7),
        make_probe("d", "d1", 0.6, 0.6),
    ]
    clusters = {"a": 0, "b": 0, "c": 1, "d": 1}
    result = simulate_attack(
        probes,
        clusters,
        retained_cluster_count=1,
        attack_seed=7,
        budgets=[2, 4],
    )
    retained = set(result["retained_canaries"])
    assert retained in ({"a", "b"}, {"c", "d"})
    assert set(result["policies"]) == {
        "sentinelrag_original",
        "random",
        "similarity",
        "composite_v1",
        "selected_random",
        "failure_repair_random",
        "cluster_diverse",
        "adaptive_bayes",
        "cluster_thompson",
        "q0_adaptive_bayes",
        "failure_repair_adaptive",
    }
