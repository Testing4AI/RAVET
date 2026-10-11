#!/usr/bin/env python3
"""Run a frozen carrier-only paraphrase study without changing archived studies."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
from collections import defaultdict
from dataclasses import replace

import numpy as np
from scipy.stats import beta

import run_summary_density as summary
from run_partition_confirmation import ROOT, digest, file_hash, read, save
from sentinelrag.testing.adaptive import stable_seed
from sentinelrag.testing.corpus_migrations import text_sha256
from sentinelrag.testing.corpus_retrieval import IndexedMigrationDocument, context_rows
from sentinelrag.testing.matrix import sparse_bm25_scores
from sentinelrag.testing.ragwm import yes_no
from sentinelrag.testing.sentinel_paraphrase import cumulative_metrics, paraphrase_prompt, validate_paraphrases
from sentinelrag.testing.summary_extension import causal_replay
from sentinelrag.utils.llm_client import parse_json_content
from sentinelrag.utils.model_registry import load_model_preset


PROTOCOL = ROOT / "experiments/sentinel_paraphrase_20261002.json"
OUT = ROOT / "output/sentinel-paraphrase-20261002-v1"


def freeze():
    protocol = read(PROTOCOL)
    source = read(ROOT / protocol["source_density"] / "protocol_lock.json")["protocol"]
    files = [PROTOCOL, ROOT / protocol["source_corpora"]]
    for block in protocol["blocks"]:
        for fixture in ("original", "relation_explicit"):
            files.append(summary.risk.artifact_path(block, fixture))
        files.extend((ROOT / protocol["source_density"] / "contexts" / block / fixture / "d1" /
                      str(protocol["replicate_seed"]) / "intact/bm25/watermarked/contexts.json")
                     for fixture in ("original", "relation_explicit"))
    files.extend([ROOT / "scripts/run_sentinel_paraphrase.py",
                  ROOT / "src/sentinelrag/testing/sentinel_paraphrase.py",
                  ROOT / "scripts/run_summary_density.py"])
    names = set([*protocol["target_models"], protocol["verifier_model"], protocol["paraphrase_model"]])
    lock = {"protocol": protocol, "source_protocol": source,
            "input_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in files},
            "model_settings": {name: {"model": load_model_preset(name, ROOT / "models").model,
                                      "args": load_model_preset(name, ROOT / "models").llm_arg}
                               for name in sorted(names)}}
    path = OUT / "protocol_lock.json"
    if path.exists() and read(path) != lock:
        raise ValueError("Frozen paraphrase study changed; create a new version")
    if not path.exists():
        save(path, lock)
    banks = read(ROOT / protocol["source_corpora"])
    save(OUT / "source_corpora.json", banks)
    return protocol, source, banks


def cache_for(protocol, concurrency):
    return summary.ragwm.AuditCache(protocol, concurrency, output_root=OUT,
                                   readonly_roots=(ROOT / protocol["source_ragwm"],
                                                   ROOT / "output/summary-density-20261001-v6",
                                                   ROOT / protocol["source_density"]))


async def preflight(protocol, concurrency):
    cache = cache_for(protocol, concurrency)
    try:
        for name in sorted(cache.clients):
            record = await cache.ask(name, "Reply with the single word READY.")
            save(OUT / "preflight" / f"{name}.json", {k: record[k] for k in
                 ("model", "returned_model", "key", "finish_reason")})
            print(f"[preflight] {record['model']}: complete", flush=True)
    finally:
        await summary.close_cache(cache)


async def transform(protocol, banks, concurrency):
    texts = {text_sha256(d["text"]): d["text"] for bank in banks["blocks"].values()
             for method in protocol["methods"] for d in bank["watermarked"][method]
             if d["source_type"] == "sentinel"}
    ordered = [texts[key] for key in sorted(texts)]
    size = protocol["paraphrase_batch_size"]
    cache = cache_for(protocol, concurrency)
    limiter = asyncio.Semaphore(concurrency)

    async def batch(documents):
        keys = [text_sha256(text) for text in documents]
        if all((OUT / "paraphrases" / f"{key}.json").exists() for key in keys):
            return
        async with limiter:
            base = paraphrase_prompt(documents)
            for attempt in range(4):
                prompt = base + ("" if not attempt else
                                f"\nFormatting retry {attempt}: return all exact ids with nonempty rewritten strings.")
                row = await cache.ask(protocol["paraphrase_model"], prompt)
                try:
                    values = validate_paraphrases(parse_json_content(row["content"]), documents)
                    break
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
            else:
                raise ValueError("No valid blind paraphrase after preserved formatting attempts")
            for key, text in zip(keys, documents, strict=True):
                save(OUT / "paraphrases" / f"{key}.json", {
                    "source_sha256": key, "source_words": len(text.split()),
                    "paraphrase": values[key], "paraphrase_words": len(values[key].split()),
                    "request": row["key"], "attempt": attempt + 1})
            print(f"[paraphrase] saved {len(documents)} carriers", flush=True)
    try:
        await asyncio.gather(*(batch(ordered[i:i + size]) for i in range(0, len(ordered), size)))
    finally:
        await summary.close_cache(cache)
    save(OUT / "paraphrase_status.json", {"complete": True, "unique_carriers": len(texts),
         "ordinary_documents_transformed": 0, "source_corpora_sha256": file_hash(OUT / "source_corpora.json")})


def prepare(protocol, source, banks):
    if not read(OUT / "paraphrase_status.json")["complete"]:
        raise ValueError("Incomplete paraphrase bank")
    dense = summary.risk.DenseScores(source)
    all_texts = sorted({d["text"] for bank in banks["blocks"].values()
                        for docs in (bank["clean"], bank["calibration"]) for d in docs} |
                       {read(p)["paraphrase"] for p in (OUT / "paraphrases").glob("*.json")})
    positions = {text: i for i, text in enumerate(all_texts)}
    vectors = dense.encode(all_texts, OUT / "dense/shared_documents.npz")
    jobs = []
    for block in protocol["blocks"]:
        bank = banks["blocks"][block]
        for method in protocol["methods"]:
            tests, _, _ = summary.suite(protocol, source, block, method)
            for scenario, state in [("calibration", "clean"), ("sentinel_paraphrase", "clean"),
                                    ("sentinel_paraphrase", "watermarked")]:
                raw = bank["calibration"] if scenario == "calibration" else (
                    bank["clean"] if state == "clean" else bank["watermarked"][method])
                docs = [IndexedMigrationDocument(**d) for d in raw]
                docs = [replace(d, text=read(OUT / "paraphrases" / f"{text_sha256(d.text)}.json")["paraphrase"])
                        if d.source_type == "sentinel" else d for d in docs]
                for retriever in protocol["target_retrievers"]:
                    job = f"{block}/{method}/{scenario}/{retriever}/{state}"
                    jobs.append(job)
                    path = OUT / "contexts" / job / "contexts.json"
                    if path.exists():
                        continue
                    if retriever == "bm25":
                        scores = sparse_bm25_scores([d.text for d in docs], [u["question"] for u in tests]).toarray()
                    else:
                        dv = vectors[[positions[d.text] for d in docs]]
                        qv = dense.encode([u["question"] for u in tests], OUT / "dense/queries" / f"{digest([u['question'] for u in tests])}.npz")
                        scores = np.einsum("ij,kj->ik", qv, dv, optimize=True).astype(np.float32)
                    contexts, diagnostics = {}, {}
                    for i, unit in enumerate(tests):
                        key = unit["unit_id"]
                        ctx = context_rows(scores[i], docs, top_k=protocol["top_k"])
                        contexts[key] = ctx
                        diagnostics[key] = {"carrier_present": state == "watermarked",
                                            "own_carrier_retrieved": any(r["source"] == "sentinel" and r["id"] == key for r in ctx)}
                    payload = {"condition": {"block": block, "method": method, "scenario": scenario,
                               "retriever": retriever, "state": state, "replicate_seed": protocol["replicate_seed"],
                               "documents": len(docs)}, "tests": tests, "contexts": contexts, "diagnostics": diagnostics}
                    payload["signature"] = digest(payload)
                    save(path, payload)
                    print(f"[prepared] {job}", flush=True)
    save(OUT / "frozen_contexts.json", {"jobs": jobs, "contexts_sha256": {
         job: file_hash(OUT / "contexts" / job / "contexts.json") for job in jobs}})


async def audit_fidelity(protocol, banks, concurrency):
    texts = {text_sha256(d["text"]): d["text"] for bank in banks["blocks"].values()
             for method in protocol["methods"] for d in bank["watermarked"][method] if d["source_type"] == "sentinel"}
    cache = cache_for(protocol, concurrency)
    limiter = asyncio.Semaphore(concurrency)

    async def audit(key, text):
        path = OUT / "fidelity" / f"{key}.json"
        if path.exists():
            return
        rewritten = read(OUT / "paraphrases" / f"{key}.json")["paraphrase"]
        prompt = ("Do these two documents express the same factual content without adding, deleting or altering factual "
                  "claims, entities, relations, numerical values or qualifications? Wording and sentence order may differ. "
                  "Reply only Yes or No.\nSource:\n" + text + "\nParaphrase:\n" + rewritten)
        async with limiter:
            for attempt in range(3):
                row = await cache.ask(protocol["verifier_model"], prompt + ("" if not attempt else
                                      f"\nFormatting retry {attempt}: only Yes or No."))
                try:
                    same = yes_no(row["content"])
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("Malformed fidelity judgment")
        save(path, {"source_sha256": key, "same_facts_model_judgment": same, "request": row["key"],
                    "included_in_target_evaluation": True})
    try:
        await asyncio.gather(*(audit(key, text) for key, text in texts.items()))
    finally:
        await summary.close_cache(cache)


def summarize(protocol, source):
    frozen = read(OUT / "frozen_contexts.json")
    groups = [read(OUT / "execution" / job / model / "answers.json")
              for job, model in itertools.product(frozen["jobs"], protocol["target_models"])]
    if not all(g["complete"] and len(g["records"]) == protocol["unit_count"] for g in groups):
        raise ValueError("Incomplete target collection")
    cal = [g for g in groups if g["condition"]["scenario"] == "calibration"]
    tail = (1 - protocol["calibration_confidence"]) / len(cal)
    calibration = []
    for g in cal:
        x, n = sum(r["is_correct"] for r in g["records"]), len(g["records"])
        upper = float(beta.ppf(1 - tail, x + 1, n - x)) if x < n else 1.0
        calibration.append({**g["condition"], "model": g["model"], "positives": x, "n": n, "upper": upper})
    p0 = max(protocol["minimum_common_p0"], max(r["upper"] for r in calibration))
    if p0 >= 1:
        raise ValueError("Uninformative clean calibration")
    save(OUT / "calibration.json", {"common_p0": p0, "groups": calibration,
                                    "scope": "empirical marginal bound, not a conditional-null guarantee"})
    trajectories, grouped = [], defaultdict(list)
    for g in groups:
        c = g["condition"]
        if c["scenario"] == "calibration":
            continue
        method = c["method"]
        _, selected, clusters = summary.suite(protocol, source, c["block"], method)
        for seed_id in ([0] if method == "RAVET" else protocol["policy_seeds"]):
            seed = stable_seed(protocol["protocol_version"], c["block"], c["scenario"], method, seed_id)
            if method == "RAVET":
                order = causal_replay(selected, clusters, g["records"], seed=seed,
                                      p0=protocol["scheduler_p0"], limit=protocol["collection_horizon"])
            else:
                order = list(g["records"])
                random.Random(seed).shuffle(order)
            labels = [r["is_correct"] for r in order]
            metrics, curve = cumulative_metrics(labels, p0, protocol["alpha"])
            metrics["answer_rate"] = sum(labels) / len(labels)
            metrics["retrieval_rate"] = sum(r["own_carrier_retrieved"] for r in order) / len(order)
            for i, point in enumerate(curve, 1):
                metrics[f"curve{i}"] = int(point["anytime_verified"])
            key = (c["block"], method, c["retriever"], c["state"], g["model"])
            grouped[key].append(metrics)
            trajectories.append({**c, "model": g["model"], "policy_seed": seed_id,
                                 "probe_order": [r["probe_id"] for r in order], "labels": labels, "curve": curve})
    names = ("block", "method", "retriever", "state", "model")
    conditions = [{**dict(zip(names, key, strict=True)),
                   **{m: float(np.mean([r[m] for r in rows])) for m in rows[0]}} for key, rows in grouped.items()]
    cells = []
    for method, state in itertools.product(protocol["methods"], ("watermarked", "clean")):
        rows = [r for r in conditions if (r["method"], r["state"]) == (method, state)]
        cells.append({"method": method, "state": state, "configurations": len(rows),
                      **{m: float(np.mean([r[m] for r in rows])) for m in grouped[next(iter(grouped))][0]}})
    fidelity = [read(p) for p in (OUT / "fidelity").glob("*.json")]
    lock = read(OUT / "protocol_lock.json")
    audit = {"complete": True, "answer_records": sum(len(g["records"]) for g in groups),
             "calibration_groups": len(cal), "positive_configurations_per_method": 8,
             "clean_configurations_per_method": 8, "fidelity_audited": len(fidelity),
             "fidelity_model_passed": sum(r["same_facts_model_judgment"] for r in fidelity),
             "ordinary_documents_transformed": 0,
             "frozen_inputs_valid": all(file_hash(ROOT / p) == h for p, h in lock["input_sha256"].items()),
             "contexts_valid": all(file_hash(OUT / "contexts" / j / "contexts.json") == h for j, h in frozen["contexts_sha256"].items())}
    save(OUT / "results.json", {"common_p0": p0, "summary": cells, "conditions": conditions})
    save(OUT / "trajectories.json", trajectories)
    save(OUT / "audit.json", audit)
    lines = ["# Sentinel-only LLM paraphrase", "", f"Common clean-calibrated p0={p0:.6f}; alpha=0.01.",
             "Eight paired deployments per method; one corpus/identity replicate. Only carrier texts are paraphrased.",
             "The paraphraser sees source text only, not probes, expected answers or method identities.",
             "", "| Method | State | VSR(6) | VSR(15) | MQ(15) | VSR(30) | MQ(30) |", "|---|---|---:|---:|---:|---:|---:|"]
    for c in cells:
        lines.append(f"| {c['method']} | {c['state']} | {c['vsr6']:.1%} | {c['vsr15']:.1%} | {c['mq15']:.2f} | {c['vsr30']:.1%} | {c['mq30']:.2f} |")
    lines += ["", "FPR uses the same crossing rule on clean deployments. Random seeds are averaged within each deployment.",
              f"Fidelity model judgments: {audit['fidelity_model_passed']}/{len(fidelity)}; no failed audit is excluded.",
              "This is not whole-corpus paraphrase, not a removal attack, and not human-validated semantic equivalence.",
              "The old NFCorpus pilot is not pooled into these results.", "", "```json", json.dumps(audit, indent=2), "```", ""]
    save(OUT / "report_lines.json", lines)
    (ROOT / "docs/SENTINEL_PARAPHRASE_RESULTS_20261002.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"common_p0": p0, "summary": cells, "audit": audit}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "transform", "prepare", "execute", "audit", "summarize", "all"))
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    protocol, source, banks = freeze()
    summary.OUT = OUT
    summary.cache_for = cache_for
    stages = ("preflight", "transform", "prepare", "execute", "audit", "summarize") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "preflight":
            asyncio.run(preflight(protocol, args.concurrency))
        elif stage == "transform":
            asyncio.run(transform(protocol, banks, args.concurrency))
        elif stage == "prepare":
            prepare(protocol, source, banks)
        elif stage == "execute":
            asyncio.run(summary.execute(protocol, args.concurrency))
        elif stage == "audit":
            asyncio.run(audit_fidelity(protocol, banks, args.concurrency))
        else:
            summarize(protocol, source)


if __name__ == "__main__":
    main()
