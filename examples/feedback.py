"""Small synthetic example using RAVET's actual prefix-only scheduler."""
from sentinelrag.testing.adaptive import PreparedProbe
from sentinelrag.testing.orthogonal_policy import next_selected_probe, select_candidates
from sentinelrag.testing.prioritization import ProbeCandidate, ProbeProfile, RetrievalObservation


def main():
    probes = []
    for i in range(8):
        for j in range(2):
            candidate = ProbeCandidate(f"c{i}-p{j}", f"c{i}", "Which code is recorded?",
                                       f"code-{i}", f"Synthetic record c{i} has code-{i}.", {"question_index": j})
            observations = [RetrievalObservation(f"shadow-{k}", .5, .1 if j or k<2 else None,
                                                 1 if j or k<2 else None) for k in range(4)]
            profile = ProbeProfile(candidate, observations, proposed_score=2*i+j)
            exposure = (1+sum(o.hit_at_5 for o in observations))/6
            probes.append(PreparedProbe(profile, False, exposure))
    selected = select_candidates(probes, "failure_repair")
    groups = {f"c{i}": i//2 for i in range(8)}
    first = next_selected_probe(selected, groups, "bayes", [], seed=0, p0=.15)
    print("Synthetic mechanism example, not a paper result")
    print("First probe:", first.profile.candidate.probe_id)
    for positive in (True, False):
        prefix = [{"probe_id": first.profile.candidate.probe_id, "is_correct": positive}]
        second = next_selected_probe(selected, groups, "bayes", prefix, seed=0, p0=.15)
        print(f"After {'positive' if positive else 'negative'} feedback: {second.profile.candidate.probe_id}")


if __name__ == "__main__":
    main()
