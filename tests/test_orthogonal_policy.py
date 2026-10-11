import pytest

from sentinelrag.testing.adaptive import PreparedProbe
from sentinelrag.testing.orthogonal_policy import (
    SCHEDULERS,
    SELECTORS,
    next_selected_probe,
    schedule_selected,
    select_candidates,
)
from sentinelrag.testing.prioritization import (
    ProbeCandidate,
    ProbeProfile,
    RetrievalObservation,
)


def candidates():
    result = []
    for i in range(8):
        for j in range(2):
            candidate = ProbeCandidate(
                f"c{i}-p{j}",
                f"c{i}",
                "question",
                "answer",
                "text",
                {"question_index": j},
            )
            profile = ProbeProfile(
                candidate,
                [RetrievalObservation("original", 0.5, 0.1, 1 + j)] * 4,
                similarity_score=i + j,
                proposed_score=2 * i + j,
            )
            result.append(PreparedProbe(profile, False, 0.4 + 0.3 * j))
    return result


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_scheduler_preserves_exact_selected_set_and_only_uses_prefix(
    selector, scheduler
):
    selected = select_candidates(candidates(), selector)
    clusters = {f"c{i}": i // 2 for i in range(8)}
    labels = {p.profile.candidate.probe_id: i % 3 == 0 for i, p in enumerate(selected)}
    expected = schedule_selected(
        selected, clusters, scheduler, labels, seed=31, p0=0.15
    )
    prefix = []
    for item in expected:
        actual = next_selected_probe(
            selected,
            clusters,
            scheduler,
            prefix,
            seed=31,
            p0=0.15,
        )
        assert actual.profile.candidate.probe_id == item.profile.candidate.probe_id
        prefix.append(
            {
                "probe_id": actual.profile.candidate.probe_id,
                "is_correct": labels[actual.profile.candidate.probe_id],
            }
        )
    assert {r["probe_id"] for r in prefix} == {
        p.profile.candidate.probe_id for p in selected
    }
    assert (
        next_selected_probe(selected, clusters, scheduler, prefix, seed=31, p0=0.15)
        is None
    )


def test_selection_is_reproducible_and_independent_of_scheduler():
    assert all(
        p.profile.candidate.probe_id.endswith("p0")
        for p in select_candidates(candidates(), "q0")
    )
    assert all(
        p.profile.candidate.probe_id.endswith("p1")
        for p in select_candidates(candidates(), "retrieval_probability")
    )
    assert [
        p.profile.candidate.probe_id for p in select_candidates(candidates(), "q0")
    ] == [
        p.profile.candidate.probe_id
        for p in select_candidates(list(reversed(candidates())), "q0")
    ]


def test_alternative_candidates_are_not_allowed_inside_scheduler():
    with pytest.raises(ValueError, match="alternative"):
        schedule_selected(
            candidates(), {f"c{i}": 0 for i in range(8)}, "bayes", {}, seed=1, p0=0.15
        )


def test_bad_or_repeated_prefix_is_rejected():
    selected = select_candidates(candidates(), "q0")
    clusters = {f"c{i}": 0 for i in range(8)}
    with pytest.raises(ValueError, match="prefix differs"):
        next_selected_probe(
            selected,
            clusters,
            "bayes",
            [{"probe_id": "absent", "is_correct": False}],
            seed=1,
            p0=0.15,
        )

    with pytest.raises(ValueError, match="Repeated"):
        next_selected_probe(
            selected,
            clusters,
            "bayes",
            [{"probe_id": "x", "is_correct": False}] * 2,
            seed=1,
            p0=0.15,
        )


def test_source_similarity_control_is_static_and_keeps_one_probe_per_sentinel():
    from sentinelrag.testing.adaptive import source_similarity_ordering

    probes = candidates()
    selected = select_candidates(probes, "source_similarity")
    clusters = {f"c{i}": i // 2 for i in range(8)}
    expected = source_similarity_ordering(probes)
    prefix = []
    for probe in expected:
        actual = next_selected_probe(
            selected, clusters, "static", prefix, seed=31, p0=0.15
        )
        assert actual.profile.candidate.probe_id == probe.profile.candidate.probe_id
        prefix.append(
            {"probe_id": actual.profile.candidate.probe_id, "is_correct": True}
        )
    assert len({p.profile.candidate.canary_id for p in selected}) == 8


def test_bayes_order_is_independent_of_seed_and_input_permutation():
    selected = select_candidates(candidates(), "retrieval_probability")
    for probe in selected:
        probe.profile.proposed_score = 1.0
    clusters = {f"c{i}": i // 2 for i in range(8)}
    labels = {p.profile.candidate.probe_id: i % 3 == 0 for i, p in enumerate(selected)}
    orders = []
    for seed in (1, 31, 999):
        for inputs in (selected, list(reversed(selected))):
            order = schedule_selected(inputs, clusters, "bayes", labels, seed=seed, p0=.15)
            orders.append([p.profile.candidate.probe_id for p in order])
    assert all(order == orders[0] for order in orders)
    assert orders[0][0] == "c7-p1"


def test_deterministic_bayes_policy_still_adapts_to_answer_feedback():
    selected = select_candidates(candidates(), "retrieval_probability")
    clusters = {f"c{i}": i // 2 for i in range(8)}
    first = next_selected_probe(selected, clusters, "bayes", [], seed=1, p0=.15)
    assert first.profile.candidate.probe_id == "c7-p1"
    next_ids = {}
    for answer in (False, True):
        prefix = [{"probe_id": first.profile.candidate.probe_id, "is_correct": answer}]
        second = next_selected_probe(selected, clusters, "bayes", prefix, seed=999, p0=.15)
        next_ids[answer] = second.profile.candidate.probe_id
    assert next_ids == {False: "c5-p1", True: "c6-p1"}
