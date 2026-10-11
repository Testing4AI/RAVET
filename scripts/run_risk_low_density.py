#!/usr/bin/env python3
"""Run low-injection-density retrieval and end-to-end migration sensitivity."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import itertools
import json
import platform
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_new_kb_confirmation import (
    OUT as SOURCE,
)
from run_new_kb_confirmation import (
    PAPER_FILES,
    artifact_path,
    owner_probes,
)
from run_orthogonal_policy_diagnostic import CachedOnly, run_trace
from run_partition_confirmation import ROOT, RequestCache, digest, file_hash, read, save

from sentinelrag.testing.adaptive import stable_seed
from sentinelrag.testing.artifacts import build_artifact_pools
from sentinelrag.testing.corpus_migrations import CorpusMigrationRecord, text_sha256
from sentinelrag.testing.corpus_retrieval import (
    IndexedMigrationDocument,
    context_rows,
    target_outcomes,
)
from sentinelrag.testing.matrix import sparse_bm25_scores
from sentinelrag.testing.migration_confirmation import (
    context_diagnostics,
    independent_corpus_partitions,
)
from sentinelrag.testing.orthogonal_policy import (
    next_selected_probe,
    schedule_selected,
    select_candidates,
)
from sentinelrag.testing.prioritization import evaluate_ordering

PROTOCOL = ROOT / "experiments/risk_low_density_20260922.json"
OUT = ROOT / "output/risk-low-density-20260922-v4"
CODE = (
    Path(__file__),
    ROOT / "scripts/run_new_kb_confirmation.py",
    ROOT / "scripts/run_orthogonal_policy_diagnostic.py",
    ROOT / "src/sentinelrag/testing/adaptive.py",
    ROOT / "src/sentinelrag/testing/orthogonal_policy.py",
    ROOT / "src/sentinelrag/testing/corpus_retrieval.py",
    ROOT / "src/sentinelrag/testing/migration_confirmation.py",
    ROOT / "src/sentinelrag/testing/prioritization.py",
    ROOT / "src/sentinelrag/utils/stats.py",
)


def dataset_for(block):
    return block.split("-block-")[0]


def load_full_corpus(dataset):
    rows = []
    seen = set()
    path = ROOT / "datasets" / dataset / "corpus.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source_id = str(row["_id"])
            text = str(row.get("text", "")).strip()
            if not text:
                text = str(row.get("title", "")).strip()
            if not text or source_id in seen:
                raise ValueError(f"Invalid full-corpus row: {dataset}/{source_id}")
            seen.add(source_id)
            rows.append(IndexedMigrationDocument("clean", source_id, 0, text))
    return rows


def sentinel_documents(block, fixture):
    artifacts = build_artifact_pools(artifact_path(block, fixture))["factual"]
    return [
        IndexedMigrationDocument(
            "sentinel", artifact.canary_id, 0, "\n".join(artifact.documents)
        )
        for artifact in artifacts
    ]


def migration_records(clean_documents, sentinel_rows):
    return [
        CorpusMigrationRecord(
            row.source_type,
            row.source_id,
            text_sha256(row.text),
            {"original": (row.text,)},
            {},
        )
        for row in [*clean_documents, *sentinel_rows]
    ]


def lock():
    protocol = read(PROTOCOL)
    source_protocol = read(SOURCE / "frozen_protocol.json")["protocol"]
    for field in (
        "blocks",
        "fixtures",
        "target_retrievers",
        "target_models",
        "verifier_model",
        "top_k",
        "max_queries",
        "alpha",
        "p0",
        "prior_retention",
        "dense_model",
        "dense_revision",
    ):
        assert protocol[field] == source_protocol[field]
    inputs = [
        PROTOCOL,
        SOURCE / "frozen_protocol.json",
        SOURCE / "preparation_audit.json",
    ]
    for dataset in sorted({dataset_for(block) for block in protocol["blocks"]}):
        inputs.append(ROOT / "datasets" / dataset / "corpus.jsonl")
    for block in protocol["blocks"]:
        inputs.append(SOURCE / "blocks" / block / "partitions.json")
        for fixture in protocol["fixtures"]:
            inputs.append(artifact_path(block, fixture))
    payload = {
        "protocol": protocol,
        "protocol_sha256": file_hash(PROTOCOL),
        "input_sha256": {
            str(path.relative_to(ROOT)): file_hash(path) for path in inputs
        },
        "code_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in CODE},
        "paper_sha256": {
            str(path.relative_to(ROOT)): file_hash(path) for path in PAPER_FILES
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "sentence-transformers",
                    "torch",
                    "openai",
                )
            },
        },
    }
    path = OUT / "protocol_lock.json"
    if path.exists():
        assert read(path) == payload
    else:
        save(path, payload)
    return payload


def density_subsets(protocol, block, canary_ids):
    dataset = dataset_for(block)
    result = []
    for label, count in protocol["density_counts"][dataset].items():
        if count > len(canary_ids):
            raise ValueError(f"Density {label} exceeds sentinel pool")
        seeds = [0] if count == len(canary_ids) else protocol["density_subset_seeds"]
        for seed in seeds:
            selected = sorted(
                canary_ids
                if count == len(canary_ids)
                else random.Random(
                    stable_seed(protocol["protocol_version"], block, label, seed)
                ).sample(canary_ids, count)
            )
            result.append((label, seed, selected))
    return result


def retained_topics(protocol, block):
    combinations = list(
        itertools.combinations(
            range(protocol["migration_partition_count"]),
            protocol["retained_partition_count"],
        )
    )
    random.Random(stable_seed(protocol["protocol_version"], block, "topics")).shuffle(
        combinations
    )
    return dict(zip(protocol["migration_seeds"], combinations, strict=False))


class DenseScores:
    def __init__(self, protocol):
        self.protocol = protocol
        self.model = None

    def encode(self, texts, path):
        signature = digest(
            {
                "texts": texts,
                "model": self.protocol["dense_model"],
                "revision": self.protocol["dense_revision"],
            }
        )
        metadata = path.with_suffix(".json")
        if (
            path.exists()
            and metadata.exists()
            and read(metadata)["signature"] == signature
        ):
            return np.load(path)["vectors"]
        if self.model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(4)
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model = SentenceTransformer(
                self.protocol["dense_model"],
                revision=self.protocol["dense_revision"],
                device=device,
                local_files_only=True,
            )
            self.model.max_seq_length = self.protocol["dense_max_seq_length"]
        vectors = self.model.encode(
            texts,
            batch_size=self.protocol["dense_embedding_batch_size"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=vectors)
        save(metadata, {"signature": signature, "shape": list(vectors.shape)})
        return vectors

    def full_scores(self, dataset, block, fixture, documents, probes):
        clean_count = sum(row.source_type == "clean" for row in documents)
        clean = documents[:clean_count]
        sentinels = documents[clean_count:]
        root = OUT / "dense" / block / fixture
        queries = self.encode(
            [probe.question for probe in probes], root / "queries.npz"
        )
        clean_vectors = self.encode(
            [row.text for row in clean], OUT / "dense" / dataset / "clean.npz"
        )
        sentinel_vectors = self.encode(
            [row.text for row in sentinels], root / "sentinels.npz"
        )
        scores = queries @ np.concatenate((clean_vectors, sentinel_vectors), axis=0).T
        assert scores.shape == (len(probes), len(documents))
        return np.asarray(scores, dtype=np.float32)


def partition_for(protocol, block, clean_documents, original_sentinels):
    path = OUT / "partitions" / f"{block}.json"
    signature = digest(
        {
            "clean": [
                (row.source_id, text_sha256(row.text)) for row in clean_documents
            ],
            "sentinels": [
                (row.source_id, text_sha256(row.text)) for row in original_sentinels
            ],
            "partition_seed": protocol["partition_seed"],
            "count": protocol["migration_partition_count"],
        }
    )
    if path.exists():
        payload = read(path)
        assert payload["signature"] == signature
        return payload["assignment"]
    assignment = independent_corpus_partitions(
        migration_records(clean_documents, original_sentinels),
        num_clusters=protocol["migration_partition_count"],
        seed=stable_seed(protocol["partition_seed"], block),
    )
    save(path, {"signature": signature, "assignment": assignment})
    return assignment


def subset_documents(documents, assignment, topics, injected, *, clean_only=False):
    return [
        row
        for row in documents
        if assignment[f"{row.source_type}:{row.source_id}"] in topics
        and (
            row.source_type == "clean" or (not clean_only and row.source_id in injected)
        )
    ]


def select_score_columns(full_documents, selected_documents, full_scores):
    positions = {
        (row.source_type, row.source_id, row.chunk_index): index
        for index, row in enumerate(full_documents)
    }
    indices = [
        positions[(row.source_type, row.source_id, row.chunk_index)]
        for row in selected_documents
    ]
    return full_scores[:, indices]


def selected_for(probes, selector, injected):
    return [
        probe
        for probe in select_candidates(probes, selector)
        if probe.profile.candidate.canary_id in injected
    ]


def retrieval_summary(
    protocol,
    selected,
    clusters,
    labels,
    scheduler,
    seed,
):
    ordering = schedule_selected(
        selected,
        clusters,
        scheduler,
        labels,
        seed=seed,
        p0=protocol["p0"],
        prior=protocol["prior_retention"],
    )
    capped = ordering[: min(protocol["max_queries"], len(ordering))]
    evaluation = evaluate_ordering(
        [probe.profile for probe in capped],
        labels,
        p0=protocol["p0"],
        alpha=protocol["alpha"],
    )
    return {
        "queries": len(capped),
        "retrieval_hit_rate": evaluation["final_detection_rate"],
        "hits": evaluation["num_successes"],
        "vsr_6": len(capped) >= 6 and evaluation["curve"][5]["significant"],
        "anytime_within_cap": evaluation["queries_to_anytime_significance"] is not None,
        "queries_to_anytime": evaluation["queries_to_anytime_significance"],
    }


def prepare():
    locked = lock()
    protocol = locked["protocol"]
    dense = DenseScores(protocol)
    retrieval_rows = []
    end_jobs = []
    full_corpora = {
        dataset: load_full_corpus(dataset)
        for dataset in sorted({dataset_for(block) for block in protocol["blocks"]})
    }
    end_lookup = {(row["block"], row["density"]) for row in protocol["end_to_end"]}
    for block in protocol["blocks"]:
        dataset = dataset_for(block)
        clean = full_corpora[dataset]
        original_sentinels = sentinel_documents(block, "original")
        assignment = partition_for(protocol, block, clean, original_sentinels)
        topics_by_seed = retained_topics(protocol, block)
        clusters = read(SOURCE / "blocks" / block / "partitions.json")["owner_clusters"]
        canary_ids = sorted(row.source_id for row in original_sentinels)
        density_sets = density_subsets(protocol, block, canary_ids)
        for fixture in protocol["fixtures"]:
            sentinel_rows = sentinel_documents(block, fixture)
            full_documents = [*clean, *sentinel_rows]
            artifacts, probes = owner_probes(block, fixture, protocol)
            candidates = [probe for artifact in artifacts for probe in artifact.probes]
            assert [p.profile.candidate.probe_id for p in probes] == [
                candidate.probe_id for candidate in candidates
            ]
            dense_full = dense.full_scores(
                dataset, block, fixture, full_documents, candidates
            )
            for density, density_seed, injected_list in density_sets:
                injected = set(injected_list)
                density_ratio = len(injected) / (len(clean) + len(injected))
                intact_documents = [
                    row
                    for row in full_documents
                    if row.source_type == "clean" or row.source_id in injected
                ]
                for retriever in protocol["target_retrievers"]:
                    scores = (
                        sparse_bm25_scores(
                            [row.text for row in intact_documents],
                            [candidate.question for candidate in candidates],
                        )
                        .toarray()
                        .astype(np.float32)
                        if retriever == "bm25"
                        else select_score_columns(
                            full_documents, intact_documents, dense_full
                        )
                    )
                    indexed = target_outcomes(
                        scores,
                        intact_documents,
                        [candidate.canary_id for candidate in candidates],
                        top_k=protocol["top_k"],
                    )
                    labels = {
                        candidate.probe_id: indexed[index]
                        for index, candidate in enumerate(candidates)
                    }
                    for selector in ("q0", "failure_repair"):
                        selected = selected_for(probes, selector, injected)
                        summary = retrieval_summary(
                            protocol,
                            selected,
                            clusters,
                            labels,
                            "random",
                            stable_seed(
                                protocol["protocol_version"],
                                block,
                                density,
                                density_seed,
                                fixture,
                                retriever,
                            ),
                        )
                        retrieval_rows.append(
                            {
                                "block": block,
                                "dataset": dataset,
                                "fixture": fixture,
                                "density": density,
                                "density_subset_seed": density_seed,
                                "retriever": retriever,
                                "clean_documents": len(clean),
                                "injected_sentinels": len(injected),
                                "injection_ratio": density_ratio,
                                "selector": selector,
                                **summary,
                            }
                        )
                if (block, density) not in end_lookup or density_seed != protocol[
                    "density_subset_seeds"
                ][0]:
                    continue
                for migration_seed, topics in topics_by_seed.items():
                    for retriever in protocol["target_retrievers"]:
                        for state in ("watermarked", "clean"):
                            documents = subset_documents(
                                full_documents,
                                assignment,
                                set(topics),
                                injected,
                                clean_only=state == "clean",
                            )
                            scores = (
                                sparse_bm25_scores(
                                    [row.text for row in documents],
                                    [candidate.question for candidate in candidates],
                                )
                                .toarray()
                                .astype(np.float32)
                                if retriever == "bm25"
                                else select_score_columns(
                                    full_documents, documents, dense_full
                                )
                            )
                            contexts, diagnostics = {}, {}
                            for index, candidate in enumerate(candidates):
                                context = context_rows(
                                    scores[index], documents, top_k=protocol["top_k"]
                                )
                                contexts[candidate.probe_id] = context
                                diagnostics[candidate.probe_id] = context_diagnostics(
                                    candidate, context
                                )
                            condition = {
                                "block": block,
                                "dataset": dataset,
                                "fixture": fixture,
                                "scenario": "full_corpus_topic_subset",
                                "density": density,
                                "density_subset_seed": density_seed,
                                "retriever": retriever,
                                "seed": migration_seed,
                                "state": state,
                                "injected_sentinels": len(injected),
                                "injected_ids": sorted(injected),
                                "injection_ratio": density_ratio,
                                "retained_topics": list(topics),
                                "retained_clean": sum(
                                    row.source_type == "clean" for row in documents
                                ),
                                "retained_sentinels": sum(
                                    row.source_type == "sentinel" for row in documents
                                ),
                            }
                            signature = digest(
                                {
                                    "condition": condition,
                                    "contexts": contexts,
                                    "diagnostics": diagnostics,
                                }
                            )
                            job = (
                                f"{block}/{fixture}/{density}/{retriever}/"
                                f"{migration_seed}/{state}"
                            )
                            payload = {
                                "condition": condition,
                                "contexts": contexts,
                                "diagnostics": diagnostics,
                                "signature": signature,
                            }
                            path = OUT / "contexts" / job / "contexts.json"
                            if path.exists():
                                assert read(path) == payload
                            else:
                                save(path, payload)
                            end_jobs.append(job)
                            print(
                                f"[density prepared] {job} clean={condition['retained_clean']} "
                                f"sentinel={condition['retained_sentinels']}",
                                flush=True,
                            )
    save(OUT / "retrieval_rows.json", retrieval_rows)
    save(OUT / "jobs.json", {"jobs": sorted(set(end_jobs))})
    print(
        f"[density prepared] retrieval_rows={len(retrieval_rows)} "
        f"end_to_end_jobs={len(set(end_jobs))}",
        flush=True,
    )


def freeze():
    locked = lock()
    protocol = locked["protocol"]
    jobs = read(OUT / "jobs.json")["jobs"]
    assert (
        len(jobs)
        == len(protocol["end_to_end"])
        * len(protocol["fixtures"])
        * len(protocol["migration_seeds"])
        * len(protocol["target_retrievers"])
        * 2
    )
    payload = {
        "lock_sha256": file_hash(OUT / "protocol_lock.json"),
        "retrieval_rows_sha256": file_hash(OUT / "retrieval_rows.json"),
        "context_sha256": {
            job: file_hash(OUT / "contexts" / job / "contexts.json") for job in jobs
        },
    }
    path = OUT / "frozen_protocol.json"
    if path.exists():
        assert read(path) == payload
    else:
        save(path, payload)
    return protocol, jobs


def policy_specs(protocol):
    return [
        (policy["selector"], policy["scheduler"], seed)
        for policy in protocol["policies"]
        for seed in (protocol["policy_seed_ids"] if policy["stochastic"] else [0])
    ]


def policy_seed(protocol, condition, seed_id):
    return stable_seed(
        protocol["protocol_version"],
        condition["block"],
        condition["density"],
        condition["seed"],
        seed_id,
    )


def prepared_set(protocol, condition, selector):
    _, probes = owner_probes(condition["block"], condition["fixture"], protocol)
    selected = selected_for(probes, selector, set(condition["injected_ids"]))
    clusters = read(SOURCE / "blocks" / condition["block"] / "partitions.json")[
        "owner_clusters"
    ]
    assert len(selected) == condition["injected_sentinels"]
    return selected, clusters


async def execute(*, online, concurrency):
    protocol, jobs = freeze()
    requests = (
        RequestCache(protocol, concurrency, output_root=OUT, readonly_roots=(SOURCE,))
        if online
        else CachedOnly((OUT, SOURCE))
    )
    limiter = asyncio.Semaphore(4)

    async def group(job, model):
        async with limiter:
            condition = read(OUT / "contexts" / job / "contexts.json")["condition"]
            await asyncio.gather(
                *(
                    run_trace(
                        protocol,
                        requests,
                        job,
                        model,
                        selector,
                        scheduler,
                        *prepared_set(protocol, condition, selector),
                        output_root=OUT / f"seed-{seed}",
                        context_root=OUT / "contexts",
                        seed_override=policy_seed(protocol, condition, seed),
                    )
                    for selector, scheduler, seed in policy_specs(protocol)
                )
            )
            print(f"[density group] {job}/{model}", flush=True)

    save(OUT / "status.json", {"status": "running", "online": online})
    try:
        await asyncio.gather(
            *(group(job, model) for job in jobs for model in protocol["target_models"])
        )
    finally:
        inventory = report()
        save(
            OUT / "status.json",
            {
                **inventory,
                "online": online,
                "new_requests_this_run": getattr(requests, "new_requests", 0),
                "cache_hits_this_run": getattr(requests, "reused_requests", 0),
            },
        )


def trace_path(job, model, selector, scheduler, seed):
    return (
        OUT / f"seed-{seed}" / "traces" / job / model / f"{selector}--{scheduler}.json"
    )


def endpoint(evaluation, max_queries):
    stop = evaluation["queries_to_anytime_significance"]
    result = {}
    for budget in range(1, max_queries + 1):
        result[f"vsr_{budget}"] = bool(evaluation["curve"][budget - 1]["significant"])
        result[f"anytime_{budget}"] = stop is not None and stop <= budget
        result[f"restricted_{budget}"] = min(
            stop if stop is not None else budget, budget
        )
    return result


def report():
    protocol, jobs = freeze()
    expected, rows = 0, []
    for job in jobs:
        for model in protocol["target_models"]:
            for selector, scheduler, seed in policy_specs(protocol):
                expected += 1
                path = trace_path(job, model, selector, scheduler, seed)
                if path.exists() and read(path)["complete"]:
                    payload = read(path)
                    rows.append(
                        {
                            **payload["condition"],
                            "model": model,
                            "selector": selector,
                            "scheduler": scheduler,
                            "policy_seed_id": seed,
                            **endpoint(payload["evaluation"], protocol["max_queries"]),
                        }
                    )
    inventory = {
        "status": "complete" if len(rows) == expected else "incomplete_resumable",
        "completed_traces": len(rows),
        "expected_traces": expected,
        "completed_logical_queries": len(rows) * protocol["max_queries"],
    }
    save(OUT / "raw_metrics.json", rows)
    save(OUT / "inventory.json", inventory)
    if len(rows) != expected:
        return inventory
    metric_names = [
        key for key in rows[0] if key.startswith(("vsr_", "anytime_", "restricted_"))
    ]
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(
            row[name]
            for name in (
                "block",
                "fixture",
                "density",
                "retriever",
                "seed",
                "state",
                "model",
                "selector",
                "scheduler",
            )
        )
        grouped[key].append(row)
    means = []
    for items in grouped.values():
        means.append(
            {
                **{
                    key: value
                    for key, value in items[0].items()
                    if key not in metric_names and key != "policy_seed_id"
                },
                **{
                    key: float(np.mean([row[key] for row in items]))
                    for key in metric_names
                },
            }
        )
    save(OUT / "condition_means.json", means)
    return inventory


def audit():
    protocol, jobs = freeze()
    failures, checked = [], 0
    for job in jobs:
        condition = read(OUT / "contexts" / job / "contexts.json")["condition"]
        for model in protocol["target_models"]:
            for selector, scheduler, seed in policy_specs(protocol):
                path = trace_path(job, model, selector, scheduler, seed)
                if not path.exists():
                    failures.append(f"missing:{path.relative_to(ROOT)}")
                    continue
                payload = read(path)
                records = payload["records"]
                if len(records) != protocol["max_queries"]:
                    failures.append(f"length:{path.relative_to(ROOT)}")
                    continue
                if len({row["canary_id"] for row in records}) != len(records):
                    failures.append(f"repeated-canary:{path.relative_to(ROOT)}")
                selected, clusters = prepared_set(protocol, condition, selector)
                for index, row in enumerate(records):
                    expected = next_selected_probe(
                        selected,
                        clusters,
                        scheduler,
                        records[:index],
                        seed=policy_seed(protocol, condition, seed),
                        p0=protocol["p0"],
                        prior=protocol["prior_retention"],
                    )
                    if expected.profile.candidate.probe_id != row["probe_id"]:
                        failures.append(f"prefix:{path.relative_to(ROOT)}:{index}")
                        break
                profiles = {p.profile.candidate.probe_id: p.profile for p in selected}
                evaluation = evaluate_ordering(
                    [profiles[row["probe_id"]] for row in records],
                    {row["probe_id"]: row["is_correct"] for row in records},
                    p0=protocol["p0"],
                    alpha=protocol["alpha"],
                )
                if evaluation != payload["evaluation"]:
                    failures.append(f"evaluation:{path.relative_to(ROOT)}")
                checked += 1
    locked = read(OUT / "protocol_lock.json")
    unchanged = all(
        file_hash(ROOT / path) == sha for path, sha in locked["paper_sha256"].items()
    )
    result = {
        "status": "pass" if not failures and unchanged else "fail",
        "checked_traces": checked,
        "failures": failures,
        "paper_unchanged": unchanged,
    }
    save(OUT / "audit.json", result)
    if result["status"] != "pass":
        raise RuntimeError(f"Low-density audit failed: {failures[:3]}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("lock", "prepare", "freeze", "run", "replay", "report", "audit"),
    )
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.action == "lock":
        lock()
    elif args.action == "prepare":
        prepare()
    elif args.action == "freeze":
        freeze()
    elif args.action == "report":
        print(report())
    elif args.action == "audit":
        print(audit())
    else:
        asyncio.run(execute(online=args.action == "run", concurrency=args.concurrency))
