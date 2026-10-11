"""Validation helpers for semantic fixture audits."""

VALID_ENTAILMENT_LABELS = {"entailed", "contradicted", "unknown"}


def normalize_entailment_judgments(payload, expected_count):
    """Validate, normalize, and order one fixture's relation judgments."""
    if not isinstance(payload, dict) or not isinstance(
        payload.get("judgments"), list
    ):
        raise TypeError("Missing judgments list")
    by_index = {}
    for row in payload["judgments"]:
        if not isinstance(row, dict):
            raise TypeError("Judgment is not an object")
        index = int(row.get("relation_index"))
        label = str(row.get("label", "")).strip().casefold()
        if label not in VALID_ENTAILMENT_LABELS:
            raise ValueError(f"Invalid label {label!r}")
        confidence = float(row.get("confidence"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid confidence {confidence}")
        if index in by_index:
            raise ValueError(f"Duplicate relation index {index}")
        by_index[index] = {
            "relation_index": index,
            "label": label,
            "confidence": confidence,
            "reason": str(row.get("reason", "")).strip(),
        }
    expected = set(range(expected_count))
    if set(by_index) != expected:
        raise ValueError(
            f"Expected relation indices {sorted(expected)}, got {sorted(by_index)}"
        )
    return [by_index[index] for index in range(expected_count)]
