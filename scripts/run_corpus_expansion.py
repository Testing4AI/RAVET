#!/usr/bin/env python3
"""Prepare and execute a fixed-evidence, paired knowledge-base expansion study."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

import run_summary_density as summary
from run_partition_confirmation import ROOT, digest, file_hash, read, save
from sentinelrag.testing.adaptive import stable_seed
from sentinelrag.testing.corpus_expansion import donor_split, expanded_documents, normalized_text_hash
from sentinelrag.testing.corpus_retrieval import IndexedMigrationDocument, context_rows
from sentinelrag.testing.matrix import sparse_bm25_scores
from sentinelrag.testing.sentinel_paraphrase import cumulative_metrics
from sentinelrag.testing.summary_extension import causal_replay
from sentinelrag.utils.model_registry import load_model_preset


PROTOCOL = ROOT / "experiments/corpus_expansion_20261002_v2.json"
OUT = ROOT / "output/corpus-expansion-20261002-v2"


def scenarios(protocol):
    return [("reference", 1)] + [(kind, factor) for kind in ("related", "cross")
                                 for factor in protocol["expansion_factors"] if factor > 1]


def freeze():
    protocol = read(PROTOCOL)
    source = read(ROOT / protocol["source_density"] / "protocol_lock.json")["protocol"]
    paths = [PROTOCOL, ROOT / protocol["source_corpora"], Path(__file__),
             ROOT / "src/sentinelrag/testing/corpus_expansion.py",
             ROOT / "src/sentinelrag/testing/sentinel_paraphrase.py",
             ROOT / "src/sentinelrag/testing/summary_extension.py",
             ROOT / "scripts/run_summary_density.py", ROOT / "scripts/run_ragwm_pilot.py"]
    for donor, metadata in protocol["donor_sources"].items():
        archive = ROOT / "datasets" / f"{donor}.zip"
        if hashlib.md5(archive.read_bytes()).hexdigest() != metadata["md5"]:
            raise ValueError(f"Official donor checksum mismatch: {donor}")
        paths += [archive, ROOT / "datasets" / donor / "corpus.jsonl"]
    for block in protocol["blocks"]:
        for fixture in ("original", "relation_explicit"):
            paths.append(summary.risk.artifact_path(block, fixture))
            paths.append(ROOT / protocol["source_density"] / "contexts" / block / fixture /
                         "d1" / str(protocol["replicate_seed"]) / "intact/bm25/watermarked/contexts.json")
    lock = {"protocol": protocol, "source_protocol": source,
            "input_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
            "model_settings": {name: {"model": load_model_preset(name, ROOT / "models").model,
                                      "args": load_model_preset(name, ROOT / "models").llm_arg}
                               for name in [*protocol["target_models"], protocol["verifier_model"]]}}
    path = OUT / "protocol_lock.json"
    if path.exists() and read(path) != lock:
        raise ValueError("Frozen expansion study changed; create a new version")
    if not path.exists():
        save(path, lock)
    return protocol, source, read(ROOT / protocol["source_corpora"])


def cache_for(protocol, concurrency):
    return summary.ragwm.AuditCache(protocol, concurrency, output_root=OUT,
                                   readonly_roots=(ROOT / "output/sentinel-paraphrase-20261002-v1",
                                                   ROOT / "output/summary-density-20261001-v6",
                                                   ROOT / protocol["source_ragwm"],
                                                   ROOT / protocol["source_density"]))


async def preflight(protocol, concurrency):
    cache = cache_for(protocol, concurrency)
    try:
        for name in sorted(cache.clients):
            row = await cache.ask(name, f"Connectivity check for {protocol['protocol_version']}. Reply only READY.")
            save(OUT / "preflight" / f"{name}.json", {k: row[k] for k in
                 ("model", "returned_model", "key", "finish_reason")})
            print(f"[preflight] {row['model']}: complete", flush=True)
    finally:
        await summary.close_cache(cache)


def donor_banks(protocol, banks):
    path = OUT / "donor_samples.json"
    if path.exists():
        return read(path)
    excluded = [d["text"] for bank in banks["blocks"].values()
                for docs in [bank["clean"], bank["calibration"], *bank["watermarked"].values()] for d in docs]
    maximum = max(protocol["expansion_factors"]) - 1
    result = {}
    for donor in protocol["donor_sources"]:
        original, empty = [], 0
        with (ROOT / "datasets" / donor / "corpus.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                text = str(row.get("text", "")).strip() or str(row.get("title", "")).strip()
                if not text:
                    empty += 1
                    continue
                original.append(IndexedMigrationDocument("clean", str(row["_id"]), 0, text))
        evaluation, calibration = donor_split(
            original, excluded, donor=donor,
            seed=stable_seed(protocol["expansion_seed"], donor),
            evaluation_count=maximum * protocol["clean_count"],
            calibration_count=maximum * protocol["calibration_count"])
        result[donor] = {"evaluation": [asdict(d) for d in evaluation],
                         "calibration": [asdict(d) for d in calibration],
                         "full_corpus_count": len(original) + empty, "empty_records_skipped": empty}
        print(f"[sampled] {donor}: {len(evaluation)} evaluation + {len(calibration)} calibration", flush=True)
    save(path, result)
    return result


def prepare(protocol, source, banks, retriever):
    donors = donor_banks(protocol, banks)
    dense = summary.risk.DenseScores(source)
    dense_vectors = {}
    if retriever == "bge-m3":
        for donor, payload in donors.items():
            texts = [d["text"] for group in ("evaluation", "calibration") for d in payload[group]]
            # Chunk checkpoints prevent a long encoding job from losing completed batches.
            chunks = [dense.encode(texts[i:i + 1000], OUT / "dense" / donor / f"{i:05d}.npz")
                      for i in range(0, len(texts), 1000)]
            dense_vectors[donor] = np.concatenate(chunks)
    for block in protocol["blocks"]:
        bank = banks["blocks"][block]
        for method in protocol["methods"]:
            tests, _, _ = summary.suite(protocol, source, block, method)
            if retriever == "bge-m3":
                questions = [t["question"] for t in tests]
                qv = dense.encode(questions, OUT / "dense/queries" / f"{digest(questions)}.npz")
            for kind, factor in scenarios(protocol):
                donor = protocol["donor_mapping"][block].get(kind)
                for state in ("watermarked", "clean", "calibration"):
                    job = f"{block}/{method}/{kind}-x{factor}/{retriever}/{state}"
                    path = OUT / "contexts" / job / "contexts.json"
                    if path.exists():
                        continue
                    key = "calibration" if state == "calibration" else "evaluation"
                    raw = bank["watermarked"][method] if state == "watermarked" else bank[state]
                    base = [IndexedMigrationDocument(**d) for d in raw]
                    extra = [IndexedMigrationDocument(**d) for d in donors[donor][key]] if donor else []
                    count = protocol["calibration_count"] if state == "calibration" else protocol["clean_count"]
                    docs = expanded_documents(base, extra, factor, ordinary_count=count)
                    if retriever == "bm25":
                        scores = sparse_bm25_scores([d.text for d in docs], [t["question"] for t in tests]).toarray()
                    else:
                        base_texts = [d.text for d in base]
                        bv = dense.encode(base_texts, OUT / "dense/base" / f"{digest(base_texts)}.npz")
                        if donor:
                            offset = len(donors[donor]["evaluation"]) if key == "calibration" else 0
                            ev = dense_vectors[donor][offset:offset + (factor - 1) * count]
                            bv = np.concatenate([bv, ev])
                        scores = np.einsum("ij,kj->ik", qv, bv, optimize=True).astype(np.float32)
                    contexts, diagnostics = {}, {}
                    for i, unit in enumerate(tests):
                        ctx = context_rows(scores[i], docs, top_k=protocol["top_k"])
                        contexts[unit["unit_id"]] = ctx
                        diagnostics[unit["unit_id"]] = {
                            "carrier_present": state == "watermarked",
                            "own_carrier_retrieved": any(r["source"] == "sentinel" and r["id"] == unit["unit_id"] for r in ctx),
                            "external_context_count": sum(str(r["id"]).startswith("external:") for r in ctx)}
                    payload = {"condition": {"block": block, "method": method,
                               "scenario": "calibration" if state == "calibration" else "corpus_expansion",
                               "expansion_kind": kind, "expansion_factor": factor, "donor": donor,
                               "retriever": retriever, "state": "clean" if state == "calibration" else state,
                               "replicate_seed": protocol["replicate_seed"], "documents": len(docs),
                               "ordinary_documents": count * factor,
                               "sentinel_count": protocol["unit_count"] if state == "watermarked" else 0},
                               "tests": tests, "contexts": contexts, "diagnostics": diagnostics,
                               "document_signature": digest([asdict(d) for d in docs])}
                    payload["signature"] = digest(payload)
                    save(path, payload)
                    print(f"[prepared] {job} ({len(docs)} documents)", flush=True)
    paths = sorted((OUT / "contexts").glob("*/*/*/*/*/contexts.json"))
    jobs = [str(p.parent.relative_to(OUT / "contexts")) for p in paths]
    save(OUT / "frozen_contexts.json", {"jobs": jobs, "contexts_sha256": {
         j: file_hash(OUT / "contexts" / j / "contexts.json") for j in jobs}})


def summarize(protocol, source):
    frozen = read(OUT / "frozen_contexts.json")
    groups = [read(OUT / "execution" / j / m / "answers.json")
              for j, m in itertools.product(frozen["jobs"], protocol["target_models"])]
    expected = len(protocol["blocks"]) * len(protocol["methods"]) * len(scenarios(protocol)) * len(protocol["target_retrievers"]) * 3 * len(protocol["target_models"])
    if len(groups) != expected or not all(g["complete"] and len(g["records"]) == 30 for g in groups):
        raise ValueError("Incomplete expanded target, clean, or calibration collection")
    calibration = []
    cal = [g for g in groups if g["condition"]["scenario"] == "calibration"]
    tail = (1 - protocol["calibration_confidence"]) / len(cal)
    for g in cal:
        x, n = sum(r["is_correct"] for r in g["records"]), len(g["records"])
        upper = float(beta.ppf(1 - tail, x + 1, n - x)) if x < n else 1.0
        calibration.append({**g["condition"], "model": g["model"], "positives": x, "n": n, "upper": upper})
    p0 = max(protocol["minimum_common_p0"], max(r["upper"] for r in calibration))
    if p0 >= 1:
        raise ValueError("Uninformative expanded clean calibration")
    save(OUT / "calibration.json", {"common_p0": p0, "groups": calibration,
                                   "scope": "empirical marginal bound, not a conditional-null guarantee"})
    grouped, trajectories = defaultdict(list), []
    for g in groups:
        c = g["condition"]
        if c["scenario"] == "calibration":
            continue
        method = c["method"]
        _, selected, clusters = summary.suite(protocol, source, c["block"], method)
        for seed_id in ([0] if method == "RAVET" else protocol["policy_seeds"]):
            seed = stable_seed(protocol["protocol_version"], c["block"], method, seed_id)
            if method == "RAVET":
                order = causal_replay(selected, clusters, g["records"], seed=seed,
                                      p0=protocol["scheduler_p0"], limit=30)
            else:
                order = list(g["records"])
                random.Random(seed).shuffle(order)
            labels = [r["is_correct"] for r in order]
            metrics, curve = cumulative_metrics(labels, p0, protocol["alpha"])
            metrics["answer_rate"] = sum(labels) / 30
            metrics["retrieval_rate"] = sum(r["own_carrier_retrieved"] for r in order) / 30
            metrics["external_context_share"] = sum(r["external_context_count"] for r in order) / (30 * protocol["top_k"])
            for i, point in enumerate(curve, 1):
                metrics[f"curve{i}"] = int(point["anytime_verified"])
            key = (c["block"], method, c["retriever"], c["state"], g["model"], c["expansion_kind"], c["expansion_factor"])
            grouped[key].append(metrics)
            trajectories.append({**c, "model": g["model"], "policy_seed": seed_id,
                                 "probe_order": [r["probe_id"] for r in order], "labels": labels, "curve": curve})
    names = ("block", "method", "retriever", "state", "model", "expansion_kind", "expansion_factor")
    conditions = [{**dict(zip(names, key, strict=True)),
                   **{m: float(np.mean([r[m] for r in rows])) for m in rows[0]}} for key, rows in grouped.items()]
    cells = []
    for kind, factor in scenarios(protocol):
        for method, state in itertools.product(protocol["methods"], ("watermarked", "clean")):
            rows = [r for r in conditions if (r["method"], r["state"], r["expansion_kind"], r["expansion_factor"]) == (method, state, kind, factor)]
            cells.append({"method": method, "state": state, "expansion_kind": kind,
                          "expansion_factor": factor, "configurations": len(rows),
                          **{m: float(np.mean([r[m] for r in rows])) for m in grouped[next(iter(grouped))][0]}})
    lock = read(OUT / "protocol_lock.json")
    donors = read(OUT / "donor_samples.json")
    audit = {"complete": True, "paired_protected_configurations": 56,
             "answer_records": sum(len(g["records"]) for g in groups), "calibration_groups": len(cal),
             "frozen_inputs_valid": all(file_hash(ROOT / p) == h for p, h in lock["input_sha256"].items()),
             "contexts_valid": all(file_hash(OUT / "contexts" / j / "contexts.json") == h for j, h in frozen["contexts_sha256"].items()),
             "donor_text_disjoint": all(not {normalized_text_hash(d["text"]) for d in v["evaluation"]} &
                                       {normalized_text_hash(d["text"]) for d in v["calibration"]} for v in donors.values()),
             "all_original_carriers_retained": all(g["condition"]["sentinel_count"] == 30 for g in groups if g["condition"]["state"] == "watermarked")}
    save(OUT / "results.json", {"common_p0": p0, "summary": cells, "conditions": conditions})
    save(OUT / "trajectories.json", trajectories)
    save(OUT / "audit.json", audit)
    print({"common_p0": p0, "audit": audit, "summary": [{k: c[k] for k in
          ("method", "state", "expansion_kind", "expansion_factor", "vsr6", "vsr15", "mq15", "vsr30", "mq30", "retrieval_rate")} for c in cells]}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "prepare", "execute", "summarize"))
    parser.add_argument("--retriever", choices=("bm25", "bge-m3"), default="bm25")
    parser.add_argument("--concurrency", type=int, default=24)
    args = parser.parse_args()
    protocol, source, banks = freeze()
    summary.OUT, summary.cache_for = OUT, cache_for
    if args.stage == "preflight":
        asyncio.run(preflight(protocol, args.concurrency))
    elif args.stage == "prepare":
        prepare(protocol, source, banks, args.retriever)
    elif args.stage == "execute":
        asyncio.run(summary.execute(protocol, args.concurrency))
    else:
        summarize(protocol, source)


if __name__ == "__main__":
    main()
