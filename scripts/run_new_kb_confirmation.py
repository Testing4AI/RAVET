#!/usr/bin/env python3
"""Frozen, prefix-only end-to-end confirmation on new target KB blocks."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import itertools
import platform
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_corpus_migration_retrieval import flatten_variant, load_owner_cache
from run_orthogonal_policy_diagnostic import CachedOnly, model_settings, run_trace
from run_partition_confirmation import (
    PAPER_FILES,
    ROOT,
    DenseBank,
    RequestCache,
    digest,
    file_hash,
    read,
    save,
)

from sentinelrag.testing.adaptive import cluster_artifacts, prepare_probes, stable_seed
from sentinelrag.testing.artifacts import build_artifact_pools
from sentinelrag.testing.corpus_migrations import (
    load_corpus_migration_manifest,
    validate_uniform_manifest,
)
from sentinelrag.testing.corpus_retrieval import context_rows
from sentinelrag.testing.corpus_subset import load_corpus_subset
from sentinelrag.testing.matrix import sparse_bm25_scores
from sentinelrag.testing.migration_confirmation import (
    context_diagnostics,
    dense_subset_scores,
    independent_corpus_partitions,
    serialize_documents,
)
from sentinelrag.testing.orthogonal_policy import select_candidates

OUT = ROOT / "output/fse-new-block-confirmation-20260906"
PROTOCOL = ROOT / "experiments/new_block_confirmation_20260907.json"
CODE = (
    Path(__file__),
    ROOT / "scripts/audit_new_kb_confirmation.py",
    ROOT / "scripts/audit_new_kb_preparation.py",
    ROOT / "scripts/summarize_new_kb_confirmation.py",
    ROOT / "scripts/execute_new_kb_study.py",
    ROOT / "scripts/run_orthogonal_policy_diagnostic.py",
    ROOT / "scripts/run_partition_confirmation.py",
    ROOT / "src/sentinelrag/testing/orthogonal_policy.py",
    ROOT / "src/sentinelrag/testing/adaptive.py",
    ROOT / "src/sentinelrag/testing/prioritization.py",
    ROOT / "src/sentinelrag/testing/migration_confirmation.py",
    ROOT / "src/sentinelrag/testing/artifacts.py",
    ROOT / "src/sentinelrag/testing/matrix.py",
    ROOT / "src/sentinelrag/testing/corpus_migrations.py",
    ROOT / "src/sentinelrag/testing/corpus_retrieval.py",
    ROOT / "src/sentinelrag/testing/corpus_subset.py",
    ROOT / "scripts/run_corpus_migration_retrieval.py",
    ROOT / "src/sentinelrag/utils/model_registry.py",
    ROOT / "src/sentinelrag/utils/stats.py",
    ROOT / "src/sentinelrag/prompts/templates.py",
)


def artifact_path(block, fixture):
    root = OUT / "blocks" / block
    if fixture == "original":
        return root / "injection/injection_result.json"
    return root / "candidates" / fixture / "injection_result.json"


def manifest_path(block, fixture):
    return (
        OUT
        / "blocks"
        / block
        / "migrations"
        / fixture
        / "corpus_migration_manifest.json"
    )


def owner_probes(block, fixture, protocol):
    artifacts = build_artifact_pools(artifact_path(block, fixture))["factual"]
    root = OUT / "blocks" / block
    subset = root / "corpus_subset.json"
    provenance = read(root / "owner" / fixture / "owner_subset_provenance.json")
    assert provenance["sha256"] == file_hash(subset)
    cache = load_owner_cache(
        root / "owner" / fixture / block.split("-block-")[0] / "contriever/factual",
        artifacts,
    )
    assert len(artifacts) == 60 and len(cache.candidates) == 300
    return artifacts, prepare_probes(cache, artifacts, top_k=protocol["top_k"])


def topic_combinations(protocol, block):
    combinations = list(
        itertools.combinations(
            range(protocol["migration_partition_count"]),
            protocol["retained_partition_count"],
        )
    )
    random.Random(
        stable_seed(protocol["protocol_version"], block, "topic-combinations")
    ).shuffle(combinations)
    return dict(zip(protocol["migration_seeds"], combinations, strict=False))


def scenario_documents(
    records, assignment, topics, scenario, *, seed, clean_only=False
):
    variant = "original" if scenario in {"S0", "S2"} else "compressed_summary"
    all_ids = {f"{r.source_type}:{r.source_id}" for r in records}
    retained = {key for key in all_ids if assignment[key] in topics}
    if scenario in {"S0", "S1"}:
        retained = all_ids
    elif scenario == "S4":
        matched = set()
        for kind in ("clean", "sentinel"):
            candidates = sorted(key for key in all_ids if key.startswith(f"{kind}:"))
            count = sum(key in retained for key in candidates)
            matched.update(
                random.Random(stable_seed(seed, kind, "random-retention")).sample(
                    candidates, count
                )
            )
        retained = matched
    elif scenario not in {"S2", "S3"}:
        raise ValueError(f"Unknown scenario: {scenario}")
    return [
        d
        for d in flatten_variant(records, variant)
        if f"{d.source_type}:{d.source_id}" in retained
        and (not clean_only or d.source_type == "clean")
    ]


class BlockDenseBank(DenseBank):
    def scores(self, block, fixture, documents, candidates):
        signature = digest(
            {
                "documents": serialize_documents(documents),
                "questions": [p.question for p in candidates],
                "model": self.protocol["dense_model"],
                "revision": self.protocol["dense_revision"],
            }
        )
        path = OUT / "dense" / block / fixture / f"{signature}.npz"
        if path.exists():
            result = np.load(path)["scores"]
            assert result.shape == (len(candidates), len(documents))
            assert np.isfinite(result).all()
            return result
        scores = (
            self.encode([p.question for p in candidates])
            @ self.encode([d.text for d in documents]).T
        )
        assert np.isfinite(scores).all()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, scores=np.asarray(scores, dtype=np.float32))
        save(
            path.with_suffix(".json"),
            {
                "signature": signature,
                "shape": list(scores.shape),
                "mode": "new fixed-encoder exact scores",
            },
        )
        return scores


def stage_cases(protocol, stage):
    if stage == "controls":
        return [(s, protocol["migration_seeds"][0]) for s in ("S0", "S1", "S2", "S4")]
    seeds = (
        protocol["migration_seeds"]
        if stage == "confirm"
        else protocol["migration_seeds"][:1]
    )
    return [("S3", seed) for seed in seeds]


def job_names(protocol, stage):
    return [
        f"{block}/{fixture}/{scenario}/{retriever}/{seed}/{state}"
        for block in protocol["blocks"]
        for fixture in protocol["fixtures"]
        for scenario, seed in stage_cases(protocol, stage)
        for retriever in protocol["target_retrievers"]
        for state in ("watermarked", "clean")
    ]


def prepare(stage, blocks=None):
    protocol = read(PROTOCOL)
    bank = BlockDenseBank(protocol)
    for block in blocks or protocol["blocks"]:
        assert block in protocol["blocks"]
        dataset = block.split("-block-")[0]
        root = OUT / "blocks" / block
        clean = load_corpus_subset(root / "corpus_subset.json", dataset=dataset)
        original = load_corpus_migration_manifest(manifest_path(block, "original"))
        assert {
            r.source_id: r.variants["original"][0]
            for r in original
            if r.source_type == "clean"
        } == {k: v["text"] for k, v in clean.items()}
        artifacts, _ = owner_probes(block, "original", protocol)
        partition_path = root / "partitions.json"
        signature = digest(
            {
                "records": file_hash(manifest_path(block, "original")),
                "protocol": protocol,
            }
        )
        if partition_path.exists():
            partition = read(partition_path)
            assert partition["signature"] == signature
        else:
            partition = {
                "signature": signature,
                "assignment": independent_corpus_partitions(
                    original,
                    num_clusters=protocol["migration_partition_count"],
                    seed=stable_seed(block, protocol["partition_seed"]),
                ),
                "owner_clusters": cluster_artifacts(
                    artifacts,
                    num_clusters=protocol["owner_cluster_count"],
                    seed=stable_seed(block, "owner-clusters"),
                ),
            }
            save(partition_path, partition)
        combinations = topic_combinations(protocol, block)
        for fixture in protocol["fixtures"]:
            records = load_corpus_migration_manifest(manifest_path(block, fixture))
            assert {
                r.source_id: r.variants for r in records if r.source_type == "clean"
            } == {r.source_id: r.variants for r in original if r.source_type == "clean"}
            fixture_artifacts, _ = owner_probes(block, fixture, protocol)
            validate_uniform_manifest(
                records,
                {key: value["text"] for key, value in clean.items()},
                {a.canary_id: "\n".join(a.documents) for a in fixture_artifacts},
            )
            candidates = [p for a in fixture_artifacts for p in a.probes]
            assert [
                (p.probe_id, p.question, p.expected_answer) for p in candidates
            ] == [
                (p.probe_id, p.question, p.expected_answer)
                for a in artifacts
                for p in a.probes
            ]
            dense = {}
            for scenario, seed in stage_cases(protocol, stage):
                variant = (
                    "original" if scenario in {"S0", "S2"} else "compressed_summary"
                )
                full = flatten_variant(records, variant)
                if variant not in dense:
                    dense[variant] = bank.scores(block, fixture, full, candidates)
                for retriever in protocol["target_retrievers"]:
                    for state in ("watermarked", "clean"):
                        documents = scenario_documents(
                            records,
                            partition["assignment"],
                            set(combinations[seed]),
                            scenario,
                            seed=stable_seed(block, seed),
                            clean_only=state == "clean",
                        )
                        assert documents
                        scores = (
                            sparse_bm25_scores(
                                [d.text for d in documents],
                                [p.question for p in candidates],
                            ).toarray()
                            if retriever == "bm25"
                            else dense_subset_scores(dense[variant], full, documents)
                        )
                        assert np.isfinite(scores).all()
                        contexts = {
                            p.probe_id: context_rows(
                                scores[i], documents, top_k=protocol["top_k"]
                            )
                            for i, p in enumerate(candidates)
                        }
                        condition = {
                            "block": block,
                            "dataset": dataset,
                            "fixture": fixture,
                            "scenario": scenario,
                            "retriever": retriever,
                            "seed": seed,
                            "state": state,
                            "top_k": protocol["top_k"],
                            "retained_topics": list(combinations[seed])
                            if scenario in {"S2", "S3"}
                            else None,
                            "retained_canaries": sorted(
                                {
                                    d.source_id
                                    for d in documents
                                    if d.source_type == "sentinel"
                                }
                            ),
                            "num_retained_clean": len(
                                {
                                    d.source_id
                                    for d in documents
                                    if d.source_type == "clean"
                                }
                            ),
                            "index_mode": "BM25 rebuilt on actual subset"
                            if retriever == "bm25"
                            else "BGE-M3 exact actual subset",
                        }
                        payload = {
                            "condition": condition,
                            "signature": digest(
                                {
                                    "condition": condition,
                                    "documents": serialize_documents(documents),
                                    "protocol": protocol,
                                }
                            ),
                            "documents_sha256": digest(serialize_documents(documents)),
                            "contexts": contexts,
                            "diagnostics": {
                                p.probe_id: context_diagnostics(p, contexts[p.probe_id])
                                for p in candidates
                            },
                        }
                        job = f"{block}/{fixture}/{scenario}/{retriever}/{seed}/{state}"
                        path = OUT / "contexts" / job / "contexts.json"
                        if path.exists():
                            assert read(path) == payload
                        else:
                            save(path, payload)
                        print(f"[prepared] {job}", flush=True)


def freeze(stage):
    protocol = read(PROTOCOL)
    audit = OUT / "preparation_audit.json"
    assert read(audit)["status"] == "pass"
    inputs = [
        PROTOCOL,
        OUT / "block_inventory.json",
        ROOT / protocol["generation_amendment"],
        ROOT / "experiments/new_kb_reporting_20260907.json",
        audit,
    ]
    for block in protocol["blocks"]:
        inputs += [
            OUT / "blocks" / block / p
            for p in (
                "corpus_subset.json",
                "partitions.json",
                "ko-pool/ko_pool.json",
                "injection/injection_checkpoint.json",
            )
        ]
        for fixture in protocol["fixtures"]:
            base = OUT / "blocks" / block / "owner" / fixture
            inputs += [
                artifact_path(block, fixture),
                manifest_path(block, fixture),
                base / "owner_subset_provenance.json",
            ]
            inputs += list(
                (base / block.split("-block-")[0] / "contriever/factual").glob(
                    "retrieval_*.*"
                )
            )
    frozen = {
        "protocol": protocol,
        "input_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in inputs},
        "code_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in CODE},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "torch",
                    "sentence-transformers",
                    "transformers",
                    "chromadb",
                    "openai",
                )
            },
        },
        "model_settings": {
            m: model_settings(m)
            for m in protocol["target_models"] + [protocol["verifier_model"]]
        },
        "paper_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in PAPER_FILES},
    }
    path = OUT / "frozen_protocol.json"
    if path.exists():
        assert read(path) == frozen
    else:
        save(path, frozen)
    jobs = job_names(protocol, stage)
    contexts = {j: file_hash(OUT / "contexts" / j / "contexts.json") for j in jobs}
    stage_path = OUT / f"{stage}_freeze.json"
    payload = {"frozen_protocol_sha256": file_hash(path), "context_sha256": contexts}
    if stage_path.exists():
        assert read(stage_path) == payload
    else:
        save(stage_path, payload)
    return protocol, jobs


def policy_specs(protocol):
    return [
        (p["selector"], p["scheduler"], seed)
        for p in protocol["policies"]
        for seed in (protocol["policy_seed_ids"] if p["stochastic"] else [0])
    ]


def seed_for(protocol, condition, seed_id):
    return stable_seed(
        protocol["protocol_version"], condition["block"], condition["seed"], seed_id
    )


def prepared_sets(protocol):
    result = {}
    for block in protocol["blocks"]:
        clusters = read(OUT / "blocks" / block / "partitions.json")["owner_clusters"]
        for fixture in protocol["fixtures"]:
            _, probes = owner_probes(block, fixture, protocol)
            for selector in {p["selector"] for p in protocol["policies"]}:
                result[(block, fixture, selector)] = (
                    select_candidates(probes, selector),
                    clusters,
                )
    return result


async def execute(stage, concurrency, online):
    protocol, jobs = freeze(stage)
    sets = prepared_sets(protocol)
    requests = (
        RequestCache(protocol, concurrency, output_root=OUT)
        if online
        else CachedOnly((OUT,))
    )
    limiter = asyncio.Semaphore(4)

    async def group(job, model):
        async with limiter:
            c = read(OUT / "contexts" / job / "contexts.json")["condition"]
            await asyncio.gather(
                *(
                    run_trace(
                        protocol,
                        requests,
                        job,
                        model,
                        selector,
                        scheduler,
                        *sets[(c["block"], c["fixture"], selector)],
                        output_root=OUT / f"seed-{seed}",
                        context_root=OUT / "contexts",
                        seed_override=seed_for(protocol, c, seed),
                    )
                    for selector, scheduler, seed in policy_specs(protocol)
                )
            )
            print(f"[group complete] {job}/{model}", flush=True)

    save(
        OUT / f"{stage}_status.json",
        {"status": "running", "mode": "online" if online else "cache_only"},
    )
    try:
        await asyncio.gather(
            *(group(job, model) for job in jobs for model in protocol["target_models"])
        )
    finally:
        inventory = report(stage)
        save(
            OUT / f"{stage}_status.json",
            {
                **inventory,
                "new_requests_this_run": getattr(requests, "new_requests", 0),
                "cache_hits_this_run": getattr(requests, "reused_requests", 0),
            },
        )


def report(stage):
    protocol = read(PROTOCOL)
    rows = []
    expected = 0
    for job in job_names(protocol, stage):
        for model in protocol["target_models"]:
            for selector, scheduler, seed in policy_specs(protocol):
                expected += 1
                path = (
                    OUT
                    / f"seed-{seed}"
                    / "traces"
                    / job
                    / model
                    / f"{selector}--{scheduler}.json"
                )
                if path.exists() and read(path)["complete"]:
                    row = read(path)
                    c, e = row["condition"], row["evaluation"]
                    metrics = {
                        f"vsr_{b}": e["curve"][b - 1]["significant"]
                        for b in range(1, protocol["max_queries"] + 1)
                    }
                    for b in range(1, protocol["max_queries"] + 1):
                        t = e["queries_to_anytime_significance"]
                        metrics[f"anytime_{b}"] = t is not None and t <= b
                        metrics[f"restricted_{b}"] = min(t if t is not None else b, b)
                    rows.append(
                        {
                            **c,
                            "model": model,
                            "selector": selector,
                            "scheduler": scheduler,
                            "policy_seed_id": seed,
                            **metrics,
                        }
                    )
    save(OUT / f"{stage}_raw_metrics.json", rows)
    inventory = {
        "status": "complete" if len(rows) == expected else "incomplete_resumable",
        "completed_traces": len(rows),
        "expected_traces": expected,
    }
    save(OUT / f"{stage}_inventory.json", inventory)
    if len(rows) != expected:
        return inventory
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(
            row[k]
            for k in (
                "block",
                "fixture",
                "scenario",
                "retriever",
                "seed",
                "state",
                "model",
                "selector",
                "scheduler",
            )
        )
        grouped[key].append(row)
    metric_names = [
        k for k in rows[0] if k.startswith(("vsr_", "anytime_", "restricted_"))
    ]
    condition_means = []
    for items in grouped.values():
        condition_means.append(
            {
                **{
                    k: v
                    for k, v in items[0].items()
                    if k not in metric_names and k != "policy_seed_id"
                },
                **{k: float(np.mean([r[k] for r in items])) for k in metric_names},
            }
        )
    save(OUT / f"{stage}_condition_means.json", condition_means)
    summary = defaultdict(list)
    for row in condition_means:
        summary[
            tuple(
                row[k]
                for k in ("fixture", "scenario", "state", "selector", "scheduler")
            )
        ].append(row)
    save(
        OUT / f"{stage}_summary.json",
        [
            {
                **dict(
                    zip(
                        ("fixture", "scenario", "state", "selector", "scheduler"),
                        key,
                        strict=True,
                    )
                ),
                "paired_conditions": len(items),
                **{k: float(np.mean([r[k] for r in items])) for k in metric_names},
            }
            for key, items in sorted(summary.items())
        ],
    )
    print(inventory, flush=True)
    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "freeze", "run", "replay", "report")
    )
    parser.add_argument(
        "--stage", choices=("pilot", "confirm", "controls"), default="pilot"
    )
    parser.add_argument("--blocks", nargs="+")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.stage, args.blocks)
    elif args.action == "freeze":
        freeze(args.stage)
    elif args.action == "report":
        report(args.stage)
    else:
        asyncio.run(execute(args.stage, args.concurrency, args.action == "run"))
