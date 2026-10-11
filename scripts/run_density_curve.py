#!/usr/bin/env python3
"""Run a development-confirmation injection-density curve experiment."""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.metadata
import itertools
import json
import platform
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_orthogonal_policy_diagnostic import run_trace
from run_partition_confirmation import RequestCache, digest, file_hash, read, save
from run_risk_low_density import (
    PAPER_FILES,
    DenseScores,
    context_diagnostics,
    context_rows,
    dataset_for,
    load_full_corpus,
    owner_probes,
    partition_for,
    retrieval_summary,
    select_score_columns,
    selected_for,
    sentinel_documents,
    sparse_bm25_scores,
)
from run_risk_low_density import (
    SOURCE as OWNER_SOURCE,
)

from sentinelrag.testing.adaptive import stable_seed
from sentinelrag.testing.corpus_retrieval import target_outcomes
from sentinelrag.testing.orthogonal_policy import next_selected_probe
from sentinelrag.testing.prioritization import evaluate_ordering

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/density_curve_20260922.json"
OUT = ROOT / "output/density-curve-20260922-v3"
OLD_DENSITY = ROOT / "output/risk-low-density-20260922-v4"
CODE = (
    Path(__file__),
    ROOT / "scripts/run_risk_low_density.py",
    ROOT / "scripts/run_orthogonal_policy_diagnostic.py",
    ROOT / "src/sentinelrag/testing/adaptive.py",
    ROOT / "src/sentinelrag/testing/orthogonal_policy.py",
    ROOT / "src/sentinelrag/testing/prioritization.py",
    ROOT / "src/sentinelrag/testing/corpus_retrieval.py",
    ROOT / "src/sentinelrag/utils/stats.py",
)


def source_protocol():
    return read(OWNER_SOURCE / "frozen_protocol.json")["protocol"]


def density_source_protocol():
    return read(OLD_DENSITY / "protocol_lock.json")["protocol"]


def phase_for(block, protocol):
    if block in protocol["development_blocks"]:
        return "development"
    if block in protocol["confirmation_blocks"]:
        return "confirmation"
    raise ValueError(f"Unassigned block: {block}")


def lock():
    protocol = read(PROTOCOL)
    source = source_protocol()
    for field in (
        "fixtures",
        "target_retrievers",
        "target_models",
        "verifier_model",
        "top_k",
        "alpha",
        "p0",
        "prior_retention",
        "dense_model",
        "dense_revision",
    ):
        assert protocol[field] == source[field], field
    blocks = protocol["development_blocks"] + protocol["confirmation_blocks"]
    assert len(set(blocks)) == len(blocks)
    assert protocol["max_queries"] <= min(protocol["density_counts"].values())
    inputs = [
        PROTOCOL,
        OWNER_SOURCE / "frozen_protocol.json",
        OLD_DENSITY / "protocol_lock.json",
    ]
    for dataset in sorted({dataset_for(block) for block in blocks}):
        inputs.append(ROOT / "datasets" / dataset / "corpus.jsonl")
    for block in blocks:
        inputs.append(OWNER_SOURCE / "blocks" / block / "partitions.json")
        for fixture in protocol["fixtures"]:
            from run_new_kb_confirmation import artifact_path

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


def clean_subset(protocol, block, documents, replicate_seed):
    count = protocol["clean_sample_count"]
    if len(documents) < count:
        raise ValueError(f"Insufficient clean corpus for {block}")
    indices = random.Random(
        stable_seed(protocol["protocol_version"], block, "clean", replicate_seed)
    ).sample(range(len(documents)), count)
    return [documents[index] for index in sorted(indices)]


def nested_density_subsets(protocol, block, canary_ids, replicate_seed):
    if max(protocol["density_counts"].values()) > len(canary_ids):
        raise ValueError(f"Insufficient sentinels for {block}")
    ordered = list(canary_ids)
    random.Random(
        stable_seed(protocol["protocol_version"], block, "sentinels", replicate_seed)
    ).shuffle(ordered)
    return {
        label: set(ordered[:count])
        for label, count in protocol["density_counts"].items()
    }


def topics_for_replicates(protocol, block):
    combinations = list(
        itertools.combinations(
            range(protocol["migration_partition_count"]),
            protocol["retained_partition_count"],
        )
    )
    random.Random(
        stable_seed(protocol["protocol_version"], block, "topics")
    ).shuffle(combinations)
    seeds = protocol["replicate_seeds"]
    return dict(zip(seeds, combinations[: len(seeds)], strict=True))


def scenario_documents(clean, sentinels, assignment, scenario, topics, injected, state):
    rows = [
        *clean,
        *(row for row in sentinels if state == "watermarked" and row.source_id in injected),
    ]
    if scenario == "intact":
        return rows
    if scenario != "topic_subset":
        raise ValueError(f"Unknown scenario: {scenario}")
    return [
        row
        for row in rows
        if assignment[f"{row.source_type}:{row.source_id}"] in topics
    ]


def selected_score_rows(candidates, active_candidates, full_scores):
    positions = {
        candidate.probe_id: index for index, candidate in enumerate(candidates)
    }
    return full_scores[[positions[candidate.probe_id] for candidate in active_candidates]]


def audited_dense_full_scores(
    dense, protocol, dataset, block, fixture, documents, candidates
):
    clean_count = sum(row.source_type == "clean" for row in documents)
    clean = documents[:clean_count]
    sentinels = documents[clean_count:]
    root = OLD_DENSITY / "dense" / block / fixture
    queries = dense.encode(
        [candidate.question for candidate in candidates], root / "queries.npz"
    )
    clean_vectors = dense.encode(
        [row.text for row in clean], OLD_DENSITY / "dense" / dataset / "clean.npz"
    )
    sentinel_vectors = dense.encode(
        [row.text for row in sentinels], root / "sentinels.npz"
    )
    document_vectors = np.concatenate((clean_vectors, sentinel_vectors), axis=0)
    if not np.isfinite(queries).all() or not np.isfinite(document_vectors).all():
        raise RuntimeError(f"Non-finite BGE-M3 vector: {block}/{fixture}")
    scores = np.einsum(
        "ij,kj->ik", queries, document_vectors, optimize=True
    ).astype(np.float32)
    if not np.isfinite(scores).all():
        raise RuntimeError(f"Non-finite BGE-M3 score: {block}/{fixture}")
    assert scores.shape == (len(candidates), len(documents))
    return scores, {
        "block": block,
        "fixture": fixture,
        "shape": list(scores.shape),
        "minimum": float(scores.min()),
        "maximum": float(scores.max()),
        "all_finite": True,
        "implementation": "numpy.einsum",
        "model": protocol["dense_model"],
        "revision": protocol["dense_revision"],
    }


def prepare():
    locked = lock()
    protocol = locked["protocol"]
    source = source_protocol()
    blocks = protocol["development_blocks"] + protocol["confirmation_blocks"]
    datasets = sorted({dataset_for(block) for block in blocks})
    corpora = {dataset: load_full_corpus(dataset) for dataset in datasets}
    dense = DenseScores(source)
    jobs = {"development": [], "confirmation": []}
    retrieval_rows = []
    manifests = []
    dense_audit = []

    for block in blocks:
        dataset = dataset_for(block)
        full_clean = corpora[dataset]
        original_sentinels = sentinel_documents(block, "original")
        assignment = partition_for(
            density_source_protocol(), block, full_clean, original_sentinels
        )
        owner_clusters = read(OWNER_SOURCE / "blocks" / block / "partitions.json")[
            "owner_clusters"
        ]
        topics_by_seed = topics_for_replicates(protocol, block)
        phase = phase_for(block, protocol)
        canary_ids = sorted(row.source_id for row in original_sentinels)

        fixture_payloads = {}
        for fixture in protocol["fixtures"]:
            sentinels = sentinel_documents(block, fixture)
            artifacts, probes = owner_probes(block, fixture, source)
            candidates = [probe for artifact in artifacts for probe in artifact.probes]
            full_documents = [*full_clean, *sentinels]
            dense_scores, score_audit = audited_dense_full_scores(
                dense,
                protocol,
                dataset,
                block,
                fixture,
                full_documents,
                candidates,
            )
            dense_audit.append(score_audit)
            fixture_payloads[fixture] = (
                sentinels,
                probes,
                candidates,
                full_documents,
                dense_scores,
            )

        for replicate_seed in protocol["replicate_seeds"]:
            sampled_clean = clean_subset(
                protocol, block, full_clean, replicate_seed
            )
            density_sets = nested_density_subsets(
                protocol, block, canary_ids, replicate_seed
            )
            topics = set(topics_by_seed[replicate_seed])
            for density, injected in density_sets.items():
                expected_count = protocol["density_counts"][density]
                assert len(injected) == expected_count
                manifests.append(
                    {
                        "phase": phase,
                        "block": block,
                        "dataset": dataset,
                        "replicate_seed": replicate_seed,
                        "density": density,
                        "clean_ids": [row.source_id for row in sampled_clean],
                        "injected_ids": sorted(injected),
                        "retained_topics": sorted(topics),
                    }
                )
                for fixture, payload in fixture_payloads.items():
                    sentinels, probes, candidates, full_documents, dense_full = payload
                    active_candidates = [
                        candidate
                        for candidate in candidates
                        if candidate.canary_id in injected
                    ]
                    dense_active = selected_score_rows(
                        candidates, active_candidates, dense_full
                    )
                    selected_sets = {
                        selector: selected_for(probes, selector, injected)
                        for selector in ("q0", "failure_repair")
                    }
                    for scenario in ("intact", "topic_subset"):
                        states = ["watermarked"]
                        if density in protocol["clean_control_densities"]:
                            states.append("clean")
                        for state in states:
                            documents = scenario_documents(
                                sampled_clean,
                                sentinels,
                                assignment,
                                scenario,
                                topics,
                                injected,
                                state,
                            )
                            for retriever in protocol["target_retrievers"]:
                                scores = (
                                    sparse_bm25_scores(
                                        [row.text for row in documents],
                                        [candidate.question for candidate in active_candidates],
                                    )
                                    .toarray()
                                    .astype(np.float32)
                                    if retriever == "bm25"
                                    else select_score_columns(
                                        full_documents, documents, dense_active
                                    )
                                )
                                contexts, diagnostics = {}, {}
                                for index, candidate in enumerate(active_candidates):
                                    context = context_rows(
                                        scores[index], documents, top_k=protocol["top_k"]
                                    )
                                    contexts[candidate.probe_id] = context
                                    diagnostics[candidate.probe_id] = context_diagnostics(
                                        candidate, context
                                    )
                                condition = {
                                    "phase": phase,
                                    "block": block,
                                    "dataset": dataset,
                                    "fixture": fixture,
                                    "density": density,
                                    "density_label": protocol["density_labels"][density],
                                    "replicate_seed": replicate_seed,
                                    "scenario": scenario,
                                    "retriever": retriever,
                                    "state": state,
                                    "seed": replicate_seed,
                                    "clean_documents": len(sampled_clean),
                                    "injected_sentinels": len(injected),
                                    "injected_ids": sorted(injected),
                                    "injection_ratio": len(injected)
                                    / (len(sampled_clean) + len(injected)),
                                    "retained_topics": sorted(topics)
                                    if scenario == "topic_subset"
                                    else list(range(protocol["migration_partition_count"])),
                                    "retained_clean": sum(
                                        row.source_type == "clean" for row in documents
                                    ),
                                    "retained_sentinels": sum(
                                        row.source_type == "sentinel" for row in documents
                                    ),
                                }
                                payload_out = {
                                    "condition": condition,
                                    "contexts": contexts,
                                    "diagnostics": diagnostics,
                                }
                                payload_out["signature"] = digest(payload_out)
                                job = (
                                    f"{block}/{fixture}/{density}/{replicate_seed}/"
                                    f"{scenario}/{retriever}/{state}"
                                )
                                path = OUT / "contexts" / job / "contexts.json"
                                if path.exists():
                                    assert read(path) == payload_out
                                else:
                                    save(path, payload_out)
                                jobs[phase].append(job)

                                if state != "watermarked":
                                    continue
                                indexed_outcomes = target_outcomes(
                                    scores,
                                    documents,
                                    [
                                        candidate.canary_id
                                        for candidate in active_candidates
                                    ],
                                    top_k=protocol["top_k"],
                                )
                                labels = {
                                    candidate.probe_id: indexed_outcomes[index]
                                    for index, candidate in enumerate(active_candidates)
                                }
                                for selector, selected in selected_sets.items():
                                    result = retrieval_summary(
                                        protocol,
                                        selected,
                                        owner_clusters,
                                        labels,
                                        "random",
                                        stable_seed(
                                            protocol["protocol_version"],
                                            block,
                                            replicate_seed,
                                            density,
                                            scenario,
                                            fixture,
                                            retriever,
                                            selector,
                                        ),
                                    )
                                    retrieval_rows.append(
                                        {
                                            **condition,
                                            "selector": selector,
                                            **result,
                                        }
                                    )
                                    assert result["hits"] <= condition["retained_sentinels"]
            print(
                f"[density curve prepared] {block} seed={replicate_seed}",
                flush=True,
            )

    jobs = {key: sorted(set(values)) for key, values in jobs.items()}
    save(OUT / "jobs.json", jobs)
    save(OUT / "subset_manifest.json", manifests)
    save(OUT / "retrieval_rows.json", retrieval_rows)
    save(OUT / "dense_numerical_audit.json", dense_audit)
    print(
        f"[density curve prepared] development={len(jobs['development'])} "
        f"confirmation={len(jobs['confirmation'])} retrieval={len(retrieval_rows)}",
        flush=True,
    )


def freeze():
    locked = lock()
    jobs = read(OUT / "jobs.json")
    payload = {
        "lock_sha256": file_hash(OUT / "protocol_lock.json"),
        "subset_manifest_sha256": file_hash(OUT / "subset_manifest.json"),
        "retrieval_rows_sha256": file_hash(OUT / "retrieval_rows.json"),
        "dense_numerical_audit_sha256": file_hash(
            OUT / "dense_numerical_audit.json"
        ),
        "context_sha256": {
            job: file_hash(OUT / "contexts" / job / "contexts.json")
            for phase in ("development", "confirmation")
            for job in jobs[phase]
        },
    }
    path = OUT / "frozen_protocol.json"
    if path.exists():
        assert read(path) == payload
    else:
        save(path, payload)
    return locked["protocol"], jobs


def policy_specs(protocol, phase, fixture, selected_scheduler=None):
    seeds = protocol["policy_seed_ids"]
    if fixture == "original":
        return [("q0", "random", seed) for seed in seeds]
    if fixture != "relation_explicit":
        raise ValueError(f"Unexpected fixture: {fixture}")
    schedulers = (
        protocol["candidate_schedulers"]
        if phase == "development"
        else [selected_scheduler]
    )
    result = []
    for scheduler in schedulers:
        for seed in seeds if scheduler == "random" else [0]:
            result.append(("failure_repair", scheduler, seed))
    return result


def policy_seed(protocol, condition, selector, scheduler, seed_id):
    return stable_seed(
        protocol["protocol_version"],
        condition["block"],
        condition["density"],
        condition["replicate_seed"],
        condition["scenario"],
        selector,
        scheduler,
        seed_id,
    )


def prepared_set(protocol, condition, selector):
    _, probes = owner_probes(condition["block"], condition["fixture"], source_protocol())
    selected = selected_for(probes, selector, set(condition["injected_ids"]))
    clusters = read(OWNER_SOURCE / "blocks" / condition["block"] / "partitions.json")[
        "owner_clusters"
    ]
    assert len(selected) == condition["injected_sentinels"]
    return selected, clusters


def selection_receipt():
    path = OUT / "selected_scheduler.json"
    if not path.exists():
        raise RuntimeError("Run the development phase and select a scheduler first")
    return read(path)


async def execute_phase(phase, *, concurrency):
    protocol, jobs = freeze()
    selected_scheduler = (
        selection_receipt()["selected_scheduler"]
        if phase == "confirmation"
        else None
    )
    requests = RequestCache(
        protocol,
        concurrency,
        output_root=OUT,
        readonly_roots=(OLD_DENSITY, OWNER_SOURCE),
    )
    limiter = asyncio.Semaphore(4)

    async def group(job, model):
        async with limiter:
            condition = read(OUT / "contexts" / job / "contexts.json")["condition"]
            assert condition["phase"] == phase
            specs = policy_specs(
                protocol, phase, condition["fixture"], selected_scheduler
            )
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
                        output_root=OUT / phase / f"seed-{seed_id}",
                        context_root=OUT / "contexts",
                        seed_override=policy_seed(
                            protocol, condition, selector, scheduler, seed_id
                        ),
                    )
                    for selector, scheduler, seed_id in specs
                )
            )
            print(f"[density curve {phase}] {job}/{model}", flush=True)

    save(
        OUT / phase / "status.json",
        {"status": "running", "phase": phase},
    )
    try:
        await asyncio.gather(
            *(
                group(job, model)
                for job in jobs[phase]
                for model in protocol["target_models"]
            )
        )
    finally:
        inventory = report_phase(phase)
        save(
            OUT / phase / "status.json",
            {
                **inventory,
                "phase": phase,
                "new_requests_this_run": requests.new_requests,
                "cache_hits_this_run": requests.reused_requests,
            },
        )


def trace_path(phase, job, model, selector, scheduler, seed_id):
    return (
        OUT
        / phase
        / f"seed-{seed_id}"
        / "traces"
        / job
        / model
        / f"{selector}--{scheduler}.json"
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


def report_phase(phase):
    protocol, jobs = freeze()
    selected_scheduler = (
        selection_receipt()["selected_scheduler"]
        if phase == "confirmation"
        else None
    )
    expected = 0
    rows = []
    for job in jobs[phase]:
        condition = read(OUT / "contexts" / job / "contexts.json")["condition"]
        specs = policy_specs(protocol, phase, condition["fixture"], selected_scheduler)
        for model in protocol["target_models"]:
            for selector, scheduler, seed_id in specs:
                expected += 1
                path = trace_path(
                    phase, job, model, selector, scheduler, seed_id
                )
                if path.exists() and read(path)["complete"]:
                    payload = read(path)
                    rows.append(
                        {
                            **payload["condition"],
                            "model": model,
                            "selector": selector,
                            "scheduler": scheduler,
                            "policy_seed_id": seed_id,
                            **endpoint(payload["evaluation"], protocol["max_queries"]),
                        }
                    )
    inventory = {
        "status": "complete" if len(rows) == expected else "incomplete_resumable",
        "completed_traces": len(rows),
        "expected_traces": expected,
        "completed_logical_queries": len(rows) * protocol["max_queries"],
    }
    save(OUT / phase / "raw_metrics.json", rows)
    save(OUT / phase / "inventory.json", inventory)
    if len(rows) != expected:
        return inventory

    metrics = [
        key for key in rows[0] if key.startswith(("vsr_", "anytime_", "restricted_"))
    ]
    grouped = defaultdict(list)
    group_keys = (
        "phase",
        "block",
        "dataset",
        "fixture",
        "density",
        "density_label",
        "replicate_seed",
        "scenario",
        "retriever",
        "state",
        "model",
        "selector",
        "scheduler",
    )
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    means = []
    for key, items in grouped.items():
        means.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "injection_ratio": items[0]["injection_ratio"],
                "injected_sentinels": items[0]["injected_sentinels"],
                "retained_clean": items[0]["retained_clean"],
                "retained_sentinels": items[0]["retained_sentinels"],
                **{
                    metric: float(np.mean([row[metric] for row in items]))
                    for metric in metrics
                },
            }
        )
    save(OUT / phase / "condition_means.json", means)
    return inventory


def condition_key(row):
    return tuple(
        row[key]
        for key in (
            "block",
            "dataset",
            "density",
            "replicate_seed",
            "scenario",
            "retriever",
            "state",
            "model",
        )
    )


def compare_rows(candidate, baseline, max_queries):
    anytime = f"anytime_{max_queries}"
    restricted = f"restricted_{max_queries}"
    left = (candidate[anytime], -candidate[restricted], candidate["vsr_6"])
    right = (baseline[anytime], -baseline[restricted], baseline["vsr_6"])
    return 1 if left > right else -1 if left < right else 0


def select_scheduler():
    inventory = report_phase("development")
    if inventory["status"] != "complete":
        raise RuntimeError(f"Development phase incomplete: {inventory}")
    protocol = read(PROTOCOL)
    rows = [
        row
        for row in read(OUT / "development/condition_means.json")
        if row["state"] == "watermarked"
    ]
    baseline = {
        condition_key(row): row
        for row in rows
        if row["fixture"] == "original"
        and row["selector"] == "q0"
        and row["scheduler"] == "random"
    }
    scores = []
    cap = protocol["max_queries"]
    for scheduler in protocol["candidate_schedulers"]:
        candidate = {
            condition_key(row): row
            for row in rows
            if row["fixture"] == "relation_explicit"
            and row["selector"] == "failure_repair"
            and row["scheduler"] == scheduler
        }
        if candidate.keys() != baseline.keys():
            raise RuntimeError(f"Unpaired development scheduler: {scheduler}")
        comparisons = [
            compare_rows(candidate[key], baseline[key], cap) for key in baseline
        ]
        scores.append(
            {
                "scheduler": scheduler,
                "conditions": len(comparisons),
                "wins": comparisons.count(1),
                "ties": comparisons.count(0),
                "losses": comparisons.count(-1),
                "net_wins": comparisons.count(1) - comparisons.count(-1),
                "mean_anytime_delta": float(
                    np.mean(
                        [
                            candidate[key][f"anytime_{cap}"]
                            - baseline[key][f"anytime_{cap}"]
                            for key in baseline
                        ]
                    )
                ),
                "mean_restricted_delta": float(
                    np.mean(
                        [
                            candidate[key][f"restricted_{cap}"]
                            - baseline[key][f"restricted_{cap}"]
                            for key in baseline
                        ]
                    )
                ),
                "mean_vsr_delta": float(
                    np.mean(
                        [
                            candidate[key]["vsr_6"] - baseline[key]["vsr_6"]
                            for key in baseline
                        ]
                    )
                ),
            }
        )
    preference = {
        name: len(protocol["selection_rule"]["tie_preference"]) - index
        for index, name in enumerate(protocol["selection_rule"]["tie_preference"])
    }
    selected = max(
        scores,
        key=lambda row: (
            row["net_wins"],
            row["mean_anytime_delta"],
            -row["mean_restricted_delta"],
            row["mean_vsr_delta"],
            preference[row["scheduler"]],
        ),
    )
    receipt = {
        "protocol_version": protocol["protocol_version"],
        "selection_rule": protocol["selection_rule"],
        "development_metrics_sha256": file_hash(
            OUT / "development/condition_means.json"
        ),
        "selected_scheduler": selected["scheduler"],
        "scores": scores,
    }
    path = OUT / "selected_scheduler.json"
    if path.exists():
        assert read(path) == receipt
    else:
        save(path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def paired_bootstrap(deltas, seed):
    values = np.asarray(deltas, dtype=float)
    if not len(values):
        return [None, None]
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def paired_group_summary(pairs, protocol, label):
    cap = protocol["max_queries"]
    comparisons = [compare_rows(method, baseline, cap) for baseline, method in pairs]
    metrics = ("vsr_6", f"anytime_{cap}", f"restricted_{cap}")
    result = {
        "group": label,
        "conditions": len(pairs),
        "wins": comparisons.count(1),
        "ties": comparisons.count(0),
        "losses": comparisons.count(-1),
    }
    for metric in metrics:
        baseline_values = [left[metric] for left, _ in pairs]
        method_values = [right[metric] for _, right in pairs]
        deltas = [right - left for left, right in zip(baseline_values, method_values)]
        result[metric] = {
            "baseline": float(np.mean(baseline_values)),
            "method": float(np.mean(method_values)),
            "method_minus_baseline": float(np.mean(deltas)),
            "paired_bootstrap_95ci": paired_bootstrap(
                deltas, stable_seed(protocol["protocol_version"], label, metric)
            ),
        }
    return result


def summarize():
    inventory = report_phase("confirmation")
    if inventory["status"] != "complete":
        raise RuntimeError(f"Confirmation phase incomplete: {inventory}")
    protocol = read(PROTOCOL)
    receipt = selection_receipt()
    scheduler = receipt["selected_scheduler"]
    rows = read(OUT / "confirmation/condition_means.json")
    watermarked = [row for row in rows if row["state"] == "watermarked"]
    baseline = {
        condition_key(row): row
        for row in watermarked
        if row["fixture"] == "original"
        and row["selector"] == "q0"
        and row["scheduler"] == "random"
    }
    method = {
        condition_key(row): row
        for row in watermarked
        if row["fixture"] == "relation_explicit"
        and row["selector"] == "failure_repair"
        and row["scheduler"] == scheduler
    }
    if baseline.keys() != method.keys():
        raise RuntimeError("Unpaired confirmation results")
    all_pairs = [(baseline[key], method[key]) for key in baseline]
    tables = [paired_group_summary(all_pairs, protocol, "overall")]
    for density in protocol["density_counts"]:
        for scenario in ("intact", "topic_subset"):
            pairs = [
                (baseline[key], method[key])
                for key in baseline
                if baseline[key]["density"] == density
                and baseline[key]["scenario"] == scenario
            ]
            tables.append(
                paired_group_summary(pairs, protocol, f"{density}/{scenario}")
            )
    for dataset in ("scidocs", "arguana"):
        pairs = [
            (baseline[key], method[key])
            for key in baseline
            if baseline[key]["dataset"] == dataset
        ]
        tables.append(paired_group_summary(pairs, protocol, f"dataset/{dataset}"))

    clean = [row for row in rows if row["state"] == "clean"]
    cap = protocol["max_queries"]
    clean_summary = {
        "conditions": len(clean),
        "fixed_positive_conditions": sum(bool(row["vsr_6"]) for row in clean),
        "anytime_positive_conditions": sum(
            bool(row[f"anytime_{cap}"]) for row in clean
        ),
    }
    retrieval = read(OUT / "retrieval_rows.json")
    retrieval_grouped = defaultdict(list)
    for row in retrieval:
        retrieval_grouped[
            (
                row["phase"],
                row["density"],
                row["scenario"],
                row["fixture"],
                row["retriever"],
                row["selector"],
            )
        ].append(row)
    retrieval_summary_rows = []
    for key, items in sorted(retrieval_grouped.items()):
        retrieval_summary_rows.append(
            {
                **dict(
                    zip(
                        (
                            "phase",
                            "density",
                            "scenario",
                            "fixture",
                            "retriever",
                            "selector",
                        ),
                        key,
                        strict=True,
                    )
                ),
                "rows": len(items),
                "retrieval_hit_rate": float(
                    np.mean([row["retrieval_hit_rate"] for row in items])
                ),
                "vsr_6": float(np.mean([row["vsr_6"] for row in items])),
                "anytime_within_cap": float(
                    np.mean([row["anytime_within_cap"] for row in items])
                ),
            }
        )
    summary = {
        "protocol_version": protocol["protocol_version"],
        "selected_scheduler": scheduler,
        "development_selection": receipt,
        "confirmation_inventory": inventory,
        "confirmation": tables,
        "clean_controls": clean_summary,
        "retrieval_sensitivity": retrieval_summary_rows,
        "full_corpus_stress_anchor": read(
            ROOT / "output/risk-experiments-20260922/summary.json"
        )["low_density"],
    }
    save(OUT / "summary.json", summary)

    with (OUT / "density_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group",
                "conditions",
                "wins",
                "ties",
                "losses",
                "baseline_anytime",
                "method_anytime",
                "delta_anytime",
                "baseline_queries",
                "method_queries",
                "delta_queries",
                "baseline_vsr6",
                "method_vsr6",
                "delta_vsr6",
            ]
        )
        for row in tables:
            writer.writerow(
                [
                    row["group"],
                    row["conditions"],
                    row["wins"],
                    row["ties"],
                    row["losses"],
                    row[f"anytime_{cap}"]["baseline"],
                    row[f"anytime_{cap}"]["method"],
                    row[f"anytime_{cap}"]["method_minus_baseline"],
                    row[f"restricted_{cap}"]["baseline"],
                    row[f"restricted_{cap}"]["method"],
                    row[f"restricted_{cap}"]["method_minus_baseline"],
                    row["vsr_6"]["baseline"],
                    row["vsr_6"]["method"],
                    row["vsr_6"]["method_minus_baseline"],
                ]
            )
    write_report(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def percent(value):
    return f"{100 * value:.1f}%"


def write_report(summary):
    protocol = read(PROTOCOL)
    cap = protocol["max_queries"]
    rows = summary["confirmation"]
    overall = rows[0]
    lines = [
        "# RAGTest 密度曲线实验报告",
        "",
        "状态：确认实验完成；论文正文未修改",
        "",
        "## 结论先行",
        "",
        f"开发集按冻结规则选择 `{summary['selected_scheduler']}` scheduler。",
        (
            f"未见确认集总体为 {overall['wins']} 胜 / {overall['ties']} 平 / {overall['losses']} 负，"
            f"Anytime@{cap} 从 {percent(overall[f'anytime_{cap}']['baseline'])} 变为 "
            f"{percent(overall[f'anytime_{cap}']['method'])}，平均截断查询从 "
            f"{overall[f'restricted_{cap}']['baseline']:.2f} 变为 "
            f"{overall[f'restricted_{cap}']['method']:.2f}。"
        ),
        "",
        "## 确认集密度曲线",
        "",
        f"| 密度/场景 | 条件 | 胜/平/负 | Baseline Anytime@{cap} | RAGTest | Δ | Baseline queries | RAGTest | Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[1:7]:
        lines.append(
            f"| {row['group']} | {row['conditions']} | {row['wins']}/{row['ties']}/{row['losses']} | "
            f"{percent(row[f'anytime_{cap}']['baseline'])} | {percent(row[f'anytime_{cap}']['method'])} | "
            f"{100 * row[f'anytime_{cap}']['method_minus_baseline']:+.1f} pp | "
            f"{row[f'restricted_{cap}']['baseline']:.2f} | {row[f'restricted_{cap}']['method']:.2f} | "
            f"{row[f'restricted_{cap}']['method_minus_baseline']:+.2f} |"
        )
    clean = summary["clean_controls"]
    lines.extend(
        [
            "",
            "## Clean controls",
            "",
            (
                f"共 {clean['conditions']} 个 clean policy conditions；固定检验阳性 "
                f"{clean['fixed_positive_conditions']}，anytime 阳性 "
                f"{clean['anytime_positive_conditions']}。"
            ),
            "",
            "## Claim 边界",
            "",
            (
                "主张以未见 block-2 的配对结果为准。开发集只用于选择一个全局 scheduler。"
                "所有负场景均计入分母；完整语料 0.1% 压力锚点单列，不与 3,000-document 主曲线混算。"
            ),
        ]
    )
    (OUT / "report.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_phase(phase):
    protocol, jobs = freeze()
    selected_scheduler = (
        selection_receipt()["selected_scheduler"]
        if phase == "confirmation"
        else None
    )
    failures = []
    checked = 0
    for job in jobs[phase]:
        condition = read(OUT / "contexts" / job / "contexts.json")["condition"]
        specs = policy_specs(protocol, phase, condition["fixture"], selected_scheduler)
        for model in protocol["target_models"]:
            for selector, scheduler, seed_id in specs:
                path = trace_path(phase, job, model, selector, scheduler, seed_id)
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
                seed = policy_seed(protocol, condition, selector, scheduler, seed_id)
                for index, row in enumerate(records):
                    expected = next_selected_probe(
                        selected,
                        clusters,
                        scheduler,
                        records[:index],
                        seed=seed,
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
    return failures, checked


def audit():
    failures = []
    checked = 0
    for phase in ("development", "confirmation"):
        phase_failures, phase_checked = audit_phase(phase)
        failures.extend(phase_failures)
        checked += phase_checked
    locked = read(OUT / "protocol_lock.json")
    paper_unchanged = all(
        file_hash(ROOT / path) == sha for path, sha in locked["paper_sha256"].items()
    )
    receipt = selection_receipt()
    development_hash_matches = receipt["development_metrics_sha256"] == file_hash(
        OUT / "development/condition_means.json"
    )
    result = {
        "status": "pass"
        if not failures and paper_unchanged and development_hash_matches
        else "fail",
        "checked_traces": checked,
        "failures": failures,
        "paper_unchanged": paper_unchanged,
        "development_selection_hash_matches": development_hash_matches,
    }
    save(OUT / "audit.json", result)
    if result["status"] != "pass":
        raise RuntimeError(f"Density curve audit failed: {result}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "lock",
            "prepare",
            "freeze",
            "run-dev",
            "select",
            "run-confirm",
            "report-dev",
            "report-confirm",
            "summary",
            "audit",
        ),
    )
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    if args.action == "lock":
        lock()
    elif args.action == "prepare":
        prepare()
    elif args.action == "freeze":
        freeze()
    elif args.action == "run-dev":
        asyncio.run(execute_phase("development", concurrency=args.concurrency))
    elif args.action == "select":
        select_scheduler()
    elif args.action == "run-confirm":
        asyncio.run(execute_phase("confirmation", concurrency=args.concurrency))
    elif args.action == "report-dev":
        print(report_phase("development"))
    elif args.action == "report-confirm":
        print(report_phase("confirmation"))
    elif args.action == "summary":
        summarize()
    else:
        audit()
