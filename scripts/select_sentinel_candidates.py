#!/usr/bin/env python3
"""Select sentinel fixtures using only frozen shadow retrieval environments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from sentinelrag.testing.artifacts import build_artifact_pools
from sentinelrag.testing.matrix import CALIBRATION_VARIANTS
from sentinelrag.testing.prioritization import normalized_answer_match
from sentinelrag.testing.sentinel_optimization import (
    candidate_injection_payload,
    supporting_document_rank,
)
from sentinelrag.utils.io import save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline-injection", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--original-matrix-root", required=True)
    parser.add_argument("--candidate-matrix-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--retrievers", nargs="+", default=["contriever", "bm25"]
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--selection-mode",
        choices=["document_average", "joint_probe"],
        default="document_average",
    )
    return parser.parse_args()


def load_condition(condition_dir: Path):
    metadata = json.loads(
        (condition_dir / "retrieval_cache_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    arrays = np.load(condition_dir / "retrieval_scores.npz")
    return metadata, arrays


def score_slot(metadata, arrays, artifact, *, top_k):
    candidate_index = {
        probe_id: index for index, probe_id in enumerate(metadata["candidate_ids"])
    }
    environments = {}
    for variant_name in CALIBRATION_VARIANTS:
        documents = metadata["variants"][variant_name]["documents"]
        own_indices = [
            index
            for index, document in enumerate(documents)
            if document["canary_id"] == artifact.canary_id
        ]
        ranks = []
        for probe in artifact.probes:
            row_index = candidate_index[probe.probe_id]
            support_offsets = [
                offset
                for offset, document_index in enumerate(own_indices)
                if normalized_answer_match(
                    documents[document_index]["text"], probe.expected_answer
                )
            ]
            if not support_offsets:
                support_offsets = list(range(len(own_indices)))
            ranks.append(
                supporting_document_rank(
                    arrays[f"{variant_name}__clean"][row_index],
                    arrays[f"{variant_name}__canary"][row_index, own_indices],
                    support_offsets,
                )
            )
        environments[variant_name] = {
            "ranks": ranks,
            "hit_rate": sum(rank <= top_k for rank in ranks) / len(ranks),
            "mean_reciprocal_rank": float(
                np.mean([1.0 / rank for rank in ranks])
            ),
        }
    return environments


def summarize_candidate(environment_rows, *, word_ratio, prefer_original):
    hit_rates = [row["hit_rate"] for row in environment_rows.values()]
    reciprocal_ranks = [
        row["mean_reciprocal_rank"] for row in environment_rows.values()
    ]
    selection_key = (
        min(hit_rates),
        float(np.mean(hit_rates)),
        float(np.mean(reciprocal_ranks)),
        -abs(word_ratio - 1.0),
        int(prefer_original),
    )
    return {
        "worst_environment_hit_rate": selection_key[0],
        "mean_hit_rate": selection_key[1],
        "mean_reciprocal_rank": selection_key[2],
        "word_ratio_deviation": abs(word_ratio - 1.0),
        "selection_key": list(selection_key),
        "environments": environment_rows,
    }


def summarize_probe_pair(
    environment_rows, *, probe_index, word_ratio, prefer_original, top_k
):
    ranks = [row["ranks"][probe_index] for row in environment_rows.values()]
    hits = [rank <= top_k for rank in ranks]
    selection_key = (
        min(hits),
        float(np.mean(hits)),
        float(np.mean([1.0 / rank for rank in ranks])),
        -abs(word_ratio - 1.0),
        int(prefer_original),
    )
    return {
        "probe_index": probe_index,
        "ranks": ranks,
        "worst_environment_hit": bool(selection_key[0]),
        "mean_hit_rate": selection_key[1],
        "mean_reciprocal_rank": selection_key[2],
        "word_ratio_deviation": abs(word_ratio - 1.0),
        "selection_key": list(selection_key),
    }


def main():
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    baseline_path = Path(args.baseline_injection)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        Path(args.candidate_manifest).read_text(encoding="utf-8")
    )
    candidate_rows = manifest["records"]
    labels = ["original", *manifest["candidate_strategies"]]
    if len(candidate_rows) != len(baseline["watermark_texts"]):
        raise ValueError("Candidate manifest is incomplete")

    artifacts = {}
    conditions = {}
    injection_paths = {"original": baseline_path}
    candidate_root = Path(args.candidate_manifest).parent
    for label in labels[1:]:
        injection_paths[label] = candidate_root / label / "injection_result.json"
    for label, injection_path in injection_paths.items():
        artifacts[label] = build_artifact_pools(injection_path)["factual"]
        for retriever in args.retrievers:
            if label == "original":
                root = Path(args.original_matrix_root)
            else:
                root = Path(args.candidate_matrix_root) / label
            condition_dir = root / args.dataset / retriever / "factual"
            conditions[(label, retriever)] = load_condition(condition_dir)

    selections = []
    optimized_documents = []
    for slot_index, candidate_row in enumerate(candidate_rows):
        candidate_scores = {}
        candidate_documents = {"original": candidate_row["source_document"]}
        candidate_documents.update(
            {
                row["strategy"]: row["document"]
                for row in candidate_row["candidates"]
            }
        )
        word_ratios = {"original": 1.0}
        word_ratios.update(
            {
                row["strategy"]: row["source_word_ratio"]
                for row in candidate_row["candidates"]
            }
        )
        for label in labels:
            environment_rows = {}
            for retriever in args.retrievers:
                metadata, arrays = conditions[(label, retriever)]
                scores = score_slot(
                    metadata,
                    arrays,
                    artifacts[label][slot_index],
                    top_k=args.top_k,
                )
                environment_rows.update(
                    {f"{retriever}:{name}": row for name, row in scores.items()}
                )
            candidate_scores[label] = summarize_candidate(
                environment_rows,
                word_ratio=word_ratios[label],
                prefer_original=label == "original",
            )
        selected_probe_index = None
        selected_pair_score = None
        if args.selection_mode == "joint_probe":
            pair_scores = {}
            for label in labels:
                pair_scores[label] = [
                    summarize_probe_pair(
                        candidate_scores[label]["environments"],
                        probe_index=probe_index,
                        word_ratio=word_ratios[label],
                        prefer_original=label == "original",
                        top_k=args.top_k,
                    )
                    for probe_index in range(
                        len(artifacts[label][slot_index].probes)
                    )
                ]
            selected, selected_pair_score = max(
                (
                    (label, score)
                    for label, scores in pair_scores.items()
                    for score in scores
                ),
                key=lambda row: tuple(row[1]["selection_key"]),
            )
            selected_probe_index = selected_pair_score["probe_index"]
        else:
            pair_scores = None
            selected = max(
                labels,
                key=lambda label: tuple(
                    candidate_scores[label]["selection_key"]
                ),
            )
        optimized_documents.append(candidate_documents[selected])
        selections.append(
            {
                "slot_index": slot_index,
                "canary_id": candidate_row["canary_id"],
                "selected": selected,
                "selected_probe_index": selected_probe_index,
                "selected_probe_id": (
                    None
                    if selected_probe_index is None
                    else artifacts[selected][slot_index]
                    .probes[selected_probe_index]
                    .probe_id
                ),
                "selected_pair_score": selected_pair_score,
                "candidate_scores": candidate_scores,
                "pair_scores": pair_scores,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution = Counter(row["selected"] for row in selections)
    selection_manifest = {
        "protocol_version": "2026-08-21-sentinel-optimization-pilot-v1",
        "dataset": args.dataset,
        "top_k": args.top_k,
        "selection_mode": args.selection_mode,
        "retrievers": args.retrievers,
        "calibration_variants": list(CALIBRATION_VARIANTS),
        "heldout_data_used_for_selection": False,
        "selection_order": (
            [
                "worst_environment_probe_hit",
                "mean_probe_hit_rate",
                "mean_probe_reciprocal_rank",
                "minimum_word_ratio_deviation",
                "prefer_original_on_exact_tie",
            ]
            if args.selection_mode == "joint_probe"
            else [
                "worst_environment_hit_rate",
                "mean_hit_rate",
                "mean_reciprocal_rank",
                "minimum_word_ratio_deviation",
                "prefer_original_on_exact_tie",
            ]
        ),
        "selection_distribution": dict(sorted(distribution.items())),
        "selections": selections,
    }
    save_json(selection_manifest, output_dir / "selection_manifest.json")
    optimized = candidate_injection_payload(
        baseline,
        optimized_documents,
        strategy="shadow_selected",
        generation_model=manifest["generation_model"],
    )
    optimized["injection_metadata"]["sentinel_optimization"].update(
        {
            "selection_manifest": str(
                (output_dir / "selection_manifest.json").resolve()
            ),
            "selection_distribution": dict(sorted(distribution.items())),
            "heldout_data_used_for_selection": False,
        }
    )
    save_json(optimized, output_dir / "injection_result.json")
    print(json.dumps(dict(sorted(distribution.items()))), flush=True)
    print(output_dir / "selection_manifest.json", flush=True)


if __name__ == "__main__":
    main()
