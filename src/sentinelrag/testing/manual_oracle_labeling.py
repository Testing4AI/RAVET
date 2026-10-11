"""Safe persistence helpers for the blinded manual-oracle labeling sheet."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "sample_id",
    "question",
    "expected_answer",
    "target_answer",
    "retrieved_context",
    "manual_correct_yes_no",
    "manual_confidence_1_5",
    "manual_notes",
)


def load_blinded_sheet(path: Path, *, expected_rows: int = 100) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Blinded sheet is missing fields: {sorted(missing)}")
        rows = list(reader)

    sample_ids = [row["sample_id"] for row in rows]
    if len(rows) != expected_rows or len(set(sample_ids)) != expected_rows:
        raise ValueError(
            f"Expected exactly {expected_rows} rows with unique sample IDs"
        )
    return rows


def normalize_label(payload: dict[str, Any]) -> dict[str, str]:
    sample_id = str(payload.get("sample_id", "")).strip()
    decision = str(payload.get("manual_correct_yes_no", "")).strip().lower()
    confidence = str(payload.get("manual_confidence_1_5", "")).strip()
    notes = str(payload.get("manual_notes", "")).strip()
    if not sample_id:
        raise ValueError("sample_id is required")
    if decision not in {"yes", "no"}:
        raise ValueError("manual_correct_yes_no must be yes or no")
    if confidence not in {"1", "2", "3", "4", "5"}:
        raise ValueError("manual_confidence_1_5 must be 1 through 5")
    if confidence in {"1", "2"} and not notes:
        raise ValueError("manual_notes is required for confidence 1 or 2")
    return {
        "sample_id": sample_id,
        "manual_correct_yes_no": decision,
        "manual_confidence_1_5": confidence,
        "manual_notes": notes,
    }


def save_label(path: Path, payload: dict[str, Any]) -> tuple[list[dict[str, str]], int]:
    normalized = normalize_label(payload)
    rows = load_blinded_sheet(path)
    matches = [row for row in rows if row["sample_id"] == normalized["sample_id"]]
    if len(matches) != 1:
        raise ValueError("sample_id is not present exactly once in the blinded sheet")
    target = matches[0]
    target.update(normalized)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    completed = sum(
        row["manual_correct_yes_no"] in {"yes", "no"}
        and row["manual_confidence_1_5"] in {"1", "2", "3", "4", "5"}
        for row in rows
    )
    return rows, completed
