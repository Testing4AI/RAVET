"""Agreement analysis for independent model-based oracle audits."""

from __future__ import annotations

from collections import Counter, defaultdict

from sentinelrag.testing.manual_oracle import cohen_kappa


def verdict_to_binary(verdict):
    """Map a model verdict to a binary label, retaining ambiguity as missing."""
    normalized = str(verdict).strip().casefold()
    if normalized == "correct":
        return True
    if normalized == "incorrect":
        return False
    if normalized == "ambiguous":
        return None
    raise ValueError(f"Unknown oracle verdict: {verdict!r}")


def agreement_summary(labels_a, labels_b):
    """Summarize agreement after removing pairs with a missing label."""
    paired = [
        (left, right)
        for left, right in zip(labels_a, labels_b, strict=True)
        if left is not None and right is not None
    ]
    if not paired:
        return {
            "comparable": 0,
            "agreements": 0,
            "disagreements": 0,
            "agreement_rate": None,
            "cohen_kappa": None,
        }
    left, right = zip(*paired, strict=True)
    agreements = sum(a == b for a, b in paired)
    return {
        "comparable": len(paired),
        "agreements": agreements,
        "disagreements": len(paired) - agreements,
        "agreement_rate": agreements / len(paired),
        "cohen_kappa": cohen_kappa(list(left), list(right)),
    }


def fleiss_kappa(label_rows):
    """Compute Fleiss' kappa for complete binary-label rows."""
    if not label_rows:
        return None
    raters = len(label_rows[0])
    if raters < 2 or any(len(row) != raters for row in label_rows):
        raise ValueError("Fleiss' kappa requires equally sized rows with >=2 raters")
    if any(label is None for row in label_rows for label in row):
        raise ValueError("Fleiss' kappa requires complete labels")

    item_agreement = []
    positives = 0
    for row in label_rows:
        positive = sum(bool(label) for label in row)
        negative = raters - positive
        positives += positive
        item_agreement.append(
            (positive * positive + negative * negative - raters)
            / (raters * (raters - 1))
        )
    observed = sum(item_agreement) / len(item_agreement)
    positive_rate = positives / (len(label_rows) * raters)
    expected = positive_rate**2 + (1 - positive_rate) ** 2
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def _model_summary(sample_ids, judgments, key_by_id):
    labels = [judgments[sample_id]["label"] for sample_id in sample_ids]
    semantic = [bool(key_by_id[sample_id]["automatic_semantic"]) for sample_id in sample_ids]
    exact = [bool(key_by_id[sample_id]["automatic_exact"]) for sample_id in sample_ids]
    decided_confidences = [
        judgments[sample_id]["confidence"]
        for sample_id in sample_ids
        if judgments[sample_id]["label"] is not None
    ]
    operational_missing = sum(
        judgments[sample_id]["operational_failure"] for sample_id in sample_ids
    )
    return {
        "samples": len(sample_ids),
        "correct": sum(label is True for label in labels),
        "incorrect": sum(label is False for label in labels),
        "ambiguous": sum(label is None for label in labels),
        "operational_missing": operational_missing,
        "semantic_ambiguous": sum(label is None for label in labels)
        - operational_missing,
        "mean_confidence_decided": sum(decided_confidences)
        / len(decided_confidences)
        if decided_confidences
        else None,
        "vs_automatic_semantic": agreement_summary(labels, semantic),
        "vs_automatic_exact": agreement_summary(labels, exact),
    }


def analyze_model_oracle_ensemble(key_rows, audits):
    """Analyze primary independent model judges against each other and references."""
    if len(audits) < 2:
        raise ValueError("At least two independent model audits are required")
    key_by_id = {row["sample_id"]: row for row in key_rows}
    if len(key_by_id) != len(key_rows):
        raise ValueError("Audit key contains duplicate sample IDs")
    sample_ids = sorted(key_by_id)

    model_rows = {}
    model_metadata = {}
    for label, payload in audits.items():
        judgment_by_id = {
            row["sample_id"]: {
                "label": verdict_to_binary(row["verdict"]),
                "verdict": row["verdict"],
                "confidence": float(row["confidence"]),
                "attempts": int(row.get("attempts", 1)),
                "operational_failure": str(row.get("reason", "")).startswith(
                    "provider_or_parse_failure:"
                ),
            }
            for row in payload["judgments"]
        }
        if set(judgment_by_id) != set(sample_ids):
            missing = sorted(set(sample_ids) - set(judgment_by_id))
            extra = sorted(set(judgment_by_id) - set(sample_ids))
            raise ValueError(
                f"Audit {label!r} has mismatched sample IDs; missing={missing}, extra={extra}"
            )
        model_rows[label] = judgment_by_id
        model_metadata[label] = {
            "model": payload.get("model", label),
            "model_preset": payload.get("model_preset"),
        }

    model_labels = sorted(model_rows)
    per_model = {}
    for label in model_labels:
        per_model[label] = model_metadata[label] | _model_summary(
            sample_ids, model_rows[label], key_by_id
        )

    pairwise = []
    for index, left in enumerate(model_labels):
        for right in model_labels[index + 1 :]:
            summary = agreement_summary(
                [model_rows[left][sample_id]["label"] for sample_id in sample_ids],
                [model_rows[right][sample_id]["label"] for sample_id in sample_ids],
            )
            pairwise.append({"model_a": left, "model_b": right, **summary})

    complete_rows = []
    unanimous_ids = []
    split_ids = []
    any_ambiguous_ids = []
    majority_rows = []
    for sample_id in sample_ids:
        labels = [model_rows[label][sample_id]["label"] for label in model_labels]
        available = [label for label in labels if label is not None]
        if len(available) == len(model_labels):
            complete_rows.append(labels)
            if len(set(available)) == 1:
                unanimous_ids.append(sample_id)
            else:
                split_ids.append(sample_id)
        else:
            any_ambiguous_ids.append(sample_id)
        counts = Counter(available)
        if (
            len(available) >= 2
            and counts.most_common(2)[0][1] > len(available) / 2
        ):
            majority_rows.append((sample_id, counts.most_common(1)[0][0]))

    majority_labels = [label for _, label in majority_rows]
    majority_ids = [sample_id for sample_id, _ in majority_rows]
    semantic = [bool(key_by_id[sample_id]["automatic_semantic"]) for sample_id in majority_ids]
    exact = [bool(key_by_id[sample_id]["automatic_exact"]) for sample_id in majority_ids]
    majority = {
        "samples": len(majority_rows),
        "coverage": len(majority_rows) / len(sample_ids),
        "correct": sum(majority_labels),
        "incorrect": len(majority_labels) - sum(majority_labels),
        "vs_automatic_semantic": agreement_summary(majority_labels, semantic),
        "vs_automatic_exact": agreement_summary(majority_labels, exact),
    }

    strata = defaultdict(list)
    for sample_id in sample_ids:
        row = key_by_id[sample_id]
        strata[(row["section"], row["rag_model"], row["target_retriever"])].append(
            sample_id
        )
    stratum_summaries = []
    majority_by_id = dict(majority_rows)
    for (section, target_model, retriever), ids in sorted(strata.items()):
        complete = [
            sample_id
            for sample_id in ids
            if all(model_rows[label][sample_id]["label"] is not None for label in model_labels)
        ]
        unanimous = sum(
            len({model_rows[label][sample_id]["label"] for label in model_labels}) == 1
            for sample_id in complete
        )
        vote_ids = [sample_id for sample_id in ids if sample_id in majority_by_id]
        vote_labels = [majority_by_id[sample_id] for sample_id in vote_ids]
        references = [bool(key_by_id[sample_id]["automatic_semantic"]) for sample_id in vote_ids]
        stratum_summaries.append(
            {
                "section": section,
                "target_model": target_model,
                "retriever": retriever,
                "samples": len(ids),
                "fully_comparable": len(complete),
                "unanimous": unanimous,
                "unanimity_rate": unanimous / len(complete) if complete else None,
                "majority_vs_automatic_semantic": agreement_summary(vote_labels, references),
            }
        )

    decisions = []
    for sample_id in sample_ids:
        labels = {
            label: model_rows[label][sample_id]["verdict"] for label in model_labels
        }
        row = key_by_id[sample_id]
        decisions.append(
            {
                "sample_id": sample_id,
                "section": row["section"],
                "target_model": row["rag_model"],
                "retriever": row["target_retriever"],
                "automatic_semantic": bool(row["automatic_semantic"]),
                "automatic_exact": bool(row["automatic_exact"]),
                "model_verdicts": labels,
                "operationally_missing_models": [
                    label
                    for label in model_labels
                    if model_rows[label][sample_id]["operational_failure"]
                ],
                "majority": majority_by_id.get(sample_id),
                "unanimous": sample_id in unanimous_ids,
            }
        )

    return {
        "protocol_version": "2026-08-22-model-oracle-ensemble-v1",
        "samples": len(sample_ids),
        "models": model_labels,
        "per_model": per_model,
        "pairwise": pairwise,
        "all_model_summary": {
            "fully_comparable": len(complete_rows),
            "any_ambiguous": len(any_ambiguous_ids),
            "unanimous": len(unanimous_ids),
            "unanimity_rate": len(unanimous_ids) / len(complete_rows)
            if complete_rows
            else None,
            "split": len(split_ids),
            "fleiss_kappa": fleiss_kappa(complete_rows),
        },
        "majority_vote": majority,
        "strata": stratum_summaries,
        "decisions": decisions,
        "interpretation_note": (
            "These are independent model-judge consistency results, not human labels."
        ),
    }
