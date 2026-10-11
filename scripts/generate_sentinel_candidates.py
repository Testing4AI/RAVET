#!/usr/bin/env python3
"""Generate checkpointed sentinel rewrite candidates for fixed probe groups."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from sentinelrag.testing.artifacts import build_artifact_pools
from sentinelrag.testing.generation_requests import BoundedGenerationClient
from sentinelrag.testing.sentinel_optimization import (
    CANDIDATE_STRATEGIES,
    candidate_injection_payload,
    generation_prompt,
    validate_single_candidate,
)
from sentinelrag.utils.io import save_json
from sentinelrag.utils.llm_client import parse_json_content
from sentinelrag.utils.model_registry import create_llm_client_from_preset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--injection-result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="api-deepseek-v4-flash-0731")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-slots", type=int)
    parser.add_argument("--bounded-requests", action="store_true")
    parser.add_argument("--fallback-original", action="store_true")
    parser.add_argument(
        "--native-json",
        action="store_true",
        help="Request provider-native JSON output instead of parsing text locally",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=CANDIDATE_STRATEGIES,
        default=list(CANDIDATE_STRATEGIES),
        help="Candidate strategies to generate (defaults to all)",
    )
    return parser.parse_args()


async def generate(args):
    if not os.getenv("RAVET_API_KEY"):
        raise RuntimeError("RAVET_API_KEY must be set")
    baseline_path = Path(args.injection_result)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    artifacts = build_artifact_pools(baseline_path)["factual"]
    output_dir = Path(args.output_dir)
    record_dir = output_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    client, preset = create_llm_client_from_preset(args.model)
    if args.bounded_requests:
        if args.native_json:
            raise ValueError("Bounded generation uses locally parsed text JSON")
        client = BoundedGenerationClient(
            client, preset, output_dir / "requests", timeout=args.timeout
        )
    semaphore = asyncio.Semaphore(args.batch_size)
    strategies = list(dict.fromkeys(args.strategies))

    async def generate_strategy(index, artifact, strategy):
        checkpoint = record_dir / f"{index:03d}-{strategy}.json"
        if checkpoint.exists():
            row = json.loads(checkpoint.read_text(encoding="utf-8"))
            retry_service_failure = row.get("fallback_original") and any(
                "Bounded generation failed:" in error
                for error in row.get("validation_errors", [])
            )
            if retry_service_failure:
                save_json(
                    row,
                    output_dir / "service_failure_fallback_archive" / checkpoint.name,
                )
            candidate, errors = validate_single_candidate(
                row,
                artifact.probes,
                strategy=strategy,
                source_word_count=len(artifact.documents[0].split()),
            )
            if (
                candidate is not None
                and not errors
                and row.get("generation_model") == preset.model
                and not retry_service_failure
            ):
                return row

        prompt = generation_prompt(
            artifact.documents[0], artifact.probes, strategy=strategy
        )
        last_errors = []
        for attempt in range(args.max_attempts):
            try:
                async with semaphore:
                    response = await client.ask_llm_async(
                        prompt, is_json=args.native_json, timeout=args.timeout
                    )
                payload = (
                    response
                    if isinstance(response, dict)
                    else parse_json_content(response)
                )
            except (json.JSONDecodeError, TimeoutError) as error:
                last_errors = [f"response_error: {error}"]
                continue
            except RuntimeError as error:
                if not args.fallback_original or "token cap" not in str(error):
                    raise
                last_errors = [f"bounded_request_failure: {error}"]
                break
            candidate, errors = validate_single_candidate(
                payload,
                artifact.probes,
                strategy=strategy,
                source_word_count=len(artifact.documents[0].split()),
            )
            if candidate is not None and not errors:
                row = {
                    "slot_index": index,
                    "canary_id": artifact.canary_id,
                    "strategy": candidate.strategy,
                    "document": candidate.document,
                    "word_count": candidate.word_count,
                    "source_word_ratio": candidate.source_word_ratio,
                    "answers_supported": candidate.answers_supported,
                    "generation_model": preset.model,
                    "attempts": attempt + 1,
                }
                save_json(row, checkpoint)
                return row
            last_errors = errors
            missing_answers = (
                [
                    probe.expected_answer
                    for probe in artifact.probes
                    if probe.expected_answer.lower() not in candidate.document.lower()
                ]
                if candidate is not None
                else [
                    probe.expected_answer
                    for probe in artifact.probes
                    if probe.expected_answer.lower()
                    not in str(payload.get("document", "")).lower()
                ]
            )
            prompt += (
                "\nThe previous response failed validation. Correct every issue: "
                f"{errors}. Missing required exact answer strings: "
                f"{missing_answers}"
            )
        failure = {
            "canary_id": artifact.canary_id,
            "strategy": strategy,
            "attempts_this_run": attempt + 1,
            "errors": last_errors,
            "fallback_original": args.fallback_original,
        }
        save_json(failure, output_dir / "failures" / checkpoint.name)
        if args.fallback_original:
            candidate, errors = validate_single_candidate(
                {"strategy": strategy, "document": artifact.documents[0]},
                artifact.probes,
                strategy=strategy,
                source_word_count=len(artifact.documents[0].split()),
            )
            if candidate is None or errors:
                raise ValueError(f"Original fixture fails fallback contract: {errors}")
            row = {
                "slot_index": index,
                "canary_id": artifact.canary_id,
                "strategy": strategy,
                "document": candidate.document,
                "word_count": candidate.word_count,
                "source_word_ratio": 1.0,
                "answers_supported": candidate.answers_supported,
                "generation_model": preset.model,
                "attempts": attempt + 1,
                "fallback_original": True,
                "validation_errors": last_errors,
            }
            save_json(row, checkpoint)
            return row
        raise ValueError(
            f"Could not generate {strategy} for {artifact.canary_id}: {last_errors}"
        )

    async def generate_one(index, artifact):
        candidates = await asyncio.gather(
            *(generate_strategy(index, artifact, strategy) for strategy in strategies)
        )
        return {
            "slot_index": index,
            "canary_id": artifact.canary_id,
            "source_document": artifact.documents[0],
            "source_word_count": len(artifact.documents[0].split()),
            "generation_model": preset.model,
            "native_json_requested": args.native_json,
            "attempts": max(row["attempts"] for row in candidates),
            "candidates": candidates,
        }

    if args.max_slots is not None:
        artifacts = artifacts[: args.max_slots]
    rows = []
    for start in range(0, len(artifacts), args.batch_size):
        batch = list(enumerate(artifacts))[start : start + args.batch_size]
        rows.extend(await asyncio.gather(*(generate_one(*row) for row in batch)))
        print(f"completed {len(rows)}/{len(artifacts)} slots", flush=True)

    rows.sort(key=lambda row: row["slot_index"])
    save_json(
        {
            "protocol_version": "2026-08-21-sentinel-optimization-pilot-v1",
            "baseline_injection": str(baseline_path.resolve()),
            "generation_model_preset": args.model,
            "generation_model": preset.model,
            "slot_count": len(rows),
            "candidate_strategies": strategies,
            "records": rows,
        },
        output_dir / "sentinel_candidates.json",
    )
    if len(rows) != len(baseline["watermark_texts"]):
        print(
            "partial candidate manifest complete; skipping strategy injections",
            flush=True,
        )
        return
    for strategy in strategies:
        documents = [
            next(
                candidate["document"]
                for candidate in row["candidates"]
                if candidate["strategy"] == strategy
            )
            for row in rows
        ]
        strategy_dir = output_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        injection = candidate_injection_payload(
            baseline, documents, strategy=strategy, generation_model=preset.model
        )
        injection["injection_metadata"]["sentinel_optimization"][
            "identity_fallback_slots"
        ] = [
            row["slot_index"]
            for row in rows
            if any(
                c["strategy"] == strategy and c.get("fallback_original")
                for c in row["candidates"]
            )
        ]
        save_json(injection, strategy_dir / "injection_result.json")
    print(output_dir / "sentinel_candidates.json", flush=True)


def main():
    asyncio.run(generate(parse_args()))


if __name__ == "__main__":
    main()
