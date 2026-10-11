"""Exact finite-bank random-order and assumption-labeled cluster diagnostics."""

import itertools

import numpy as np

from sentinelrag.utils.stats import bernoulli_mixture_e_value


def exact_random_first_crossing(n, positives, p0=0.15, alpha=0.01, cap=None):
    if not 0 <= positives <= n or n <= 0:
        raise ValueError("Invalid finite answer bank")
    cap = n if cap is None else cap
    if not 1 <= cap <= n:
        raise ValueError("Observation window exceeds finite bank")
    alive = {0: 1.0}
    survival = [1.0]
    mq = 0.0
    first_mass = []
    for t in range(cap):
        mq += sum(alive.values())
        after = {}
        crossed = 0.0
        for s, mass in alive.items():
            probability = (positives - s) / (n - t)
            for outcome, branch_probability in ((1, probability), (0, 1 - probability)):
                branch = mass * branch_probability
                if branch <= 0:
                    continue
                successes = s + outcome
                if bernoulli_mixture_e_value(successes, t + 1, p0) >= 1 / alpha:
                    crossed += branch
                else:
                    after[successes] = after.get(successes, 0.0) + branch
        first_mass.append(crossed)
        alive = after
        survival.append(sum(alive.values()))
    if not np.isclose(survival[-1] + sum(first_mass), 1.0):
        raise ValueError("Finite-bank probability mass lost")
    return {"vsr6": 1 - survival[min(6, cap)], "vsr15": 1 - survival[-1],
            "mq15": mq, "first_crossing_mass": first_mass, "bank_size": n,
            "bank_positives": positives, "cap": cap}


def descriptive_sign_flips(draws):
    if len(draws) != 6:
        raise ValueError("Expected six dependency clusters")
    result = {}
    for metric in ("vsr6", "vsr15", "mq15"):
        deltas = np.asarray([d[metric]["difference"] for d in draws], dtype=float)
        observed = abs(float(deltas.mean()))
        flipped = [abs(float(np.mean(deltas * signs)))
                   for signs in itertools.product((-1, 1), repeat=len(draws))]
        result[metric] = {"mean_delta": float(deltas.mean()),
                          "two_sided_sign_flip_tail": sum(x >= observed - 1e-12 for x in flipped) / len(flipped),
                          "patterns": len(flipped), "positive_draws": int(sum(deltas > 1e-12)),
                          "zero_draws": int(sum(abs(deltas) <= 1e-12)),
                          "negative_draws": int(sum(deltas < -1e-12))}
    return {"metrics": result,
            "interpretation": "Assumption-dependent descriptive sign-flip diagnostic; not randomized or independent-corpus inference"}
