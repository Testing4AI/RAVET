"""Shared helpers for blinded human-oracle analysis."""

POSITIVE_LABELS = {"1", "true", "yes", "y"}
NEGATIVE_LABELS = {"0", "false", "no", "n"}


def parse_binary_label(value):
    normalized = value.strip().casefold()
    if normalized in POSITIVE_LABELS:
        return True
    if normalized in NEGATIVE_LABELS:
        return False
    return None


def cohen_kappa(labels_a, labels_b):
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("Kappa requires two equally sized nonempty label vectors")
    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / len(
        labels_a
    )
    a_rate = sum(labels_a) / len(labels_a)
    b_rate = sum(labels_b) / len(labels_b)
    expected = a_rate * b_rate + (1 - a_rate) * (1 - b_rate)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)
