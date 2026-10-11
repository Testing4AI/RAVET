#!/usr/bin/env python3
"""Frozen 1% summary extension comparing three ownership-verification methods."""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import run_density_curve as density
import run_ragwm_pilot as ragwm
import run_risk_low_density as risk
from run_partition_confirmation import ROOT, digest, file_hash, read, save
from scipy.stats import beta

from sentinelrag.prompts.templates import PromptTemplates
from sentinelrag.testing.adaptive import stable_seed
from sentinelrag.testing.corpus_migrations import text_sha256
from sentinelrag.testing.corpus_retrieval import IndexedMigrationDocument, context_rows, top_indices
from sentinelrag.testing.matrix import sparse_bm25_scores
from sentinelrag.testing.migration_confirmation import explicit_abstention
from sentinelrag.testing.ragwm import evidence_curve, yes_no
from sentinelrag.testing.summary_extension import causal_replay, concise_summary_prompt, summary_prompt, validate_summaries
from sentinelrag.utils.llm_client import parse_json_content
from sentinelrag.utils.model_registry import load_model_preset

PROTOCOL = ROOT / "experiments/summary_density_20261001.json"
OUT = ROOT / "output/summary-density-20261001-v6"
METHOD_FIXTURES = {"SentinelRAG": ("original", "q0"), "RAVET": ("relation_explicit", "failure_repair")}


def freeze():
    protocol = read(PROTOCOL)
    source = read(ROOT / protocol["source_density"] / "protocol_lock.json")["protocol"]
    baseline = ROOT / protocol["source_ragwm"]
    inputs = [PROTOCOL, baseline / "protocol_lock.json", baseline / "frozen_contexts.json"]
    for block in protocol["blocks"]:
        inputs.extend([baseline / "construction" / block / "carriers.json",
                       baseline / "injection" / block / str(protocol["replicate_seed"]) / "manifest.json",
                       risk.OUT / "partitions" / f"{block}.json",
                       ROOT / "datasets" / risk.dataset_for(block) / "corpus.jsonl"])
        for fixture, _ in METHOD_FIXTURES.values():
            inputs.append(risk.artifact_path(block, fixture))
    code = [Path(__file__), ROOT / "src/sentinelrag/testing/summary_extension.py",
            ROOT / "scripts/run_ragwm_pilot.py", ROOT / "src/sentinelrag/testing/orthogonal_policy.py"]
    lock = {"protocol": protocol, "source_protocol": source,
            "input_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in inputs},
            "code_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in code},
            "paper_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in density.PAPER_FILES},
            "model_settings": {name: {"model": load_model_preset(name, ROOT / "models").model,
                                      "args": load_model_preset(name, ROOT / "models").llm_arg}
                               for name in [*protocol["target_models"], protocol["verifier_model"]]}}
    path = OUT / "protocol_lock.json"
    if path.exists():
        old = read(path)
        # Papers may be independently edited by the user; frozen research inputs must not change.
        if {k: v for k, v in old.items() if k != "paper_sha256"} != {
                k: v for k, v in lock.items() if k != "paper_sha256"}:
            raise ValueError("Frozen research inputs changed; create a new protocol version")
    else:
        save(path, lock)
    return protocol, source


def cache_for(protocol, concurrency):
    return ragwm.AuditCache(protocol, concurrency, output_root=OUT,
                            readonly_roots=(ROOT / "output/summary-density-20261001-v5",
                                            ROOT / "output/summary-density-20261001-v4",
                                            ROOT / "output/summary-density-20261001-v3",
                                            ROOT / "output/summary-density-20261001-v1",
                                            ROOT / protocol["source_ragwm"]))


async def close_cache(cache):
    await asyncio.gather(*(client.async_client.close() for client, _ in cache.clients.values()))


def suite(protocol, source, block, method):
    seed = protocol["replicate_seed"]
    if method == "RAG-WM":
        carriers = read(ROOT / protocol["source_ragwm"] / "construction" / block / "carriers.json")["records"]
        return [{"unit_id": c["unit"]["unit_id"], "probe_id": c["unit"]["unit_id"],
                 "question": c["question"], "expected_answer":
                 f"{c['unit']['e1']} {c['unit']['relation'].replace('_', ' ')} {c['unit']['e2']}.",
                 "relation_unit": c["unit"]} for c in carriers], None, None
    fixture, selector = METHOD_FIXTURES[method]
    condition = read(ROOT / protocol["source_density"] / "contexts" / block / fixture / "d1" /
                     str(seed) / "intact/bm25/watermarked/contexts.json")["condition"]
    selected, clusters = density.prepared_set(source, condition, selector)
    return [{"unit_id": p.profile.candidate.canary_id, "probe_id": p.profile.candidate.probe_id,
             "question": p.profile.candidate.question, "expected_answer": p.profile.candidate.expected_answer}
            for p in selected], selected, clusters


def source_corpora(protocol, source):
    path = OUT / "source_corpora.json"
    if path.exists():
        return read(path)
    blocks = {}
    for block in protocol["blocks"]:
        seed = protocol["replicate_seed"]
        baseline = ROOT / protocol["source_ragwm"]
        manifest = read(baseline / "injection" / block / str(seed) / "manifest.json")
        full, clean = ragwm.evaluation_clean(source, block, seed)
        by_id = {d.source_id: d for d in full}
        calibration = [by_id[key] for key in manifest["calibration_ids"]]
        if set(manifest["clean_ids"]) != {d.source_id for d in clean}:
            raise ValueError("Baseline clean sample mismatch")
        if len(clean) != protocol["clean_count"] or len(calibration) != protocol["calibration_count"]:
            raise ValueError("Unexpected corpus size")
        modified = list(clean)
        carriers = read(baseline / "construction" / block / "carriers.json")["records"]
        anchors = defaultdict(list)
        for carrier in carriers:
            variants = carrier["selected_variants"]
            scores = sparse_bm25_scores([d.text for d in modified], [carrier["question"]]).toarray()[0]
            for position, variant in zip(top_indices(scores, len(variants)), variants, strict=True):
                index = int(position)
                doc = modified[index]
                modified[index] = replace(doc, text=doc.text + " " + variant["text"])
                anchors[carrier["unit"]["unit_id"]].append(doc.source_id)
        if dict(anchors) != manifest["anchor_ids"]:
            raise ValueError("RAG-WM injection replay differs from frozen manifest")
        marked = {"RAG-WM": modified}
        for method, (fixture, _) in METHOD_FIXTURES.items():
            tests, _, _ = suite(protocol, source, block, method)
            ids = {u["unit_id"] for u in tests}
            marked[method] = clean + [d for d in risk.sentinel_documents(block, fixture) if d.source_id in ids]
        blocks[block] = {"clean": [asdict(d) for d in clean], "calibration": [asdict(d) for d in calibration],
                         "watermarked": {m: [asdict(d) for d in docs] for m, docs in marked.items()},
                         "anchors": dict(anchors), "retained_topics": manifest["retained_topics"]}
    payload = {"blocks": blocks}
    payload["signature"] = digest(payload)
    save(path, payload)
    return payload


async def compress(protocol, source, concurrency):
    banks = source_corpora(protocol, source)
    texts = {}
    for block in banks["blocks"].values():
        for docs in [block["clean"], block["calibration"], *block["watermarked"].values()]:
            for doc in docs:
                texts[text_sha256(doc["text"])] = doc["text"]
    ordered = [texts[key] for key in sorted(texts)]
    size = protocol["summary_batch_size"]
    batches = [ordered[i:i + size] for i in range(0, len(ordered), size)]
    cache = cache_for(protocol, concurrency)
    cache.model_semaphores[protocol["summary_model"]] = asyncio.Semaphore(protocol["summary_request_concurrency"])
    limiter = asyncio.Semaphore(concurrency)
    completed = 0

    async def batch(index, documents):
        nonlocal completed
        keys = [text_sha256(text) for text in documents]
        for key, text in zip(keys, documents, strict=True):
            current = OUT / "summaries" / f"{key}.json"
            if len(text.split()) <= protocol["preserve_short_words"]:
                save(current, {"source_sha256": key, "summary": text, "request": None,
                               "source_words": len(text.split()), "summary_words": len(text.split()),
                               "batch": index, "attempt": 0, "policy": "preserve_short_document"})
                continue
            legacy = ROOT / "output/summary-density-20261001-v5/summaries" / f"{key}.json"
            if not legacy.exists():
                legacy = ROOT / "output/summary-density-20261001-v4/summaries" / f"{key}.json"
            if not legacy.exists():
                legacy = ROOT / "output/summary-density-20261001-v3/summaries" / f"{key}.json"
            if not legacy.exists():
                legacy = ROOT / "output/summary-density-20261001-v1/summaries" / f"{key}.json"
            if not current.exists() and legacy.exists():
                row = read(legacy)
                validate_summaries({key: row["summary"]}, [text], protocol["summary_ratio"])
                save(current, {**row, "reused_from": str(legacy.relative_to(ROOT))})
        if all((OUT / "summaries" / f"{key}.json").exists() for key in keys):
            completed += 1
            return
        async with limiter:
            documents = [t for t in documents if len(t.split()) > protocol["preserve_short_words"]]
            keys = [text_sha256(text) for text in documents]
            prompt = summary_prompt(documents, protocol["summary_ratio"])
            values, requests, invalid = {}, {}, {}
            for attempt in range(4):
                suffix = "" if attempt == 0 else (
                    f"\nFormatting retry {attempt}: use exact ids and respect EVERY max_words. "
                    "Count whitespace-separated words; return only the required JSON object.")
                record = await cache.ask(protocol["summary_model"], prompt + suffix)
                try:
                    parsed = parse_json_content(record["content"])
                    if not isinstance(parsed, dict) or parsed.keys() != set(keys):
                        raise ValueError("Wrong summary ids")
                    for key, text in zip(keys, documents, strict=True):
                        try:
                            result = validate_summaries({key: parsed[key]}, [text], protocol["summary_ratio"])
                            values[key], requests[key] = result[key], record["key"]
                        except ValueError:
                            if isinstance(parsed[key], str):
                                invalid[key] = parsed[key]
                    break
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
            async def repair(key, text):
                if key in values:
                    return
                base = summary_prompt([text], protocol["summary_ratio"])
                maximum = max(1, math.ceil(len(text.split()) * protocol["summary_ratio"]))
                previous = invalid.get(key, "")
                for retry in range(6):
                    target = max(1, int(maximum * max(.3, .8 - .1 * retry)))
                    feedback = (f"\nPrevious invalid summary ({len(previous.split())} words):\n" +
                                json.dumps(previous, ensure_ascii=True)) if previous else ""
                    row = await cache.ask(protocol["summary_model"], base + (
                        f"\nLength-validation repair {retry}: aim for no more than {target} words "
                        f"to leave a counting margin below the mandatory {maximum}-word limit. "
                        "Shorten the previous summary by merging sentences and omitting secondary detail; "
                        "do not add information; return only the one-id JSON object." + feedback))
                    try:
                        parsed = parse_json_content(row["content"])
                        if isinstance(parsed, dict) and isinstance(parsed.get(key), str):
                            previous = parsed[key]
                        result = validate_summaries(parsed, [text], protocol["summary_ratio"])
                        values[key], requests[key] = result[key], row["key"]
                        return
                    except (ValueError, TypeError, json.JSONDecodeError):
                        continue
                for retry in range(8):
                    target = max(1, int(maximum * max(.25, .7 - .1 * retry)))
                    model = protocol["summary_model"] if retry < 2 else protocol["summary_recovery_model"]
                    row = await cache.ask(model, concise_summary_prompt(text, protocol["summary_ratio"], target))
                    try:
                        result = validate_summaries(parse_json_content(row["content"]), [text], protocol["summary_ratio"])
                        values[key], requests[key] = result[key], row["key"]
                        return
                    except (ValueError, TypeError, json.JSONDecodeError):
                        continue
                raise ValueError(f"Summary {key} failed preserved blind length recovery")
            await asyncio.gather(*(repair(key, text) for key, text in zip(keys, documents, strict=True)))
            for key, text in zip(keys, documents, strict=True):
                save(OUT / "summaries" / f"{key}.json", {
                    "source_sha256": key, "summary": values[key], "request": requests[key],
                    "source_words": len(text.split()), "summary_words": len(values[key].split()),
                    "batch": index, "attempt": attempt + 1})
            completed += 1
            if completed % 20 == 0:
                print(f"[summary] batches={completed}/{len(batches)}, documents={min(completed * size, len(texts))}", flush=True)
    print(f"[summary] unique documents={len(texts)}, batches={len(batches)}", flush=True)
    try:
        await asyncio.gather(*(batch(i, docs) for i, docs in enumerate(batches)))
    finally:
        await close_cache(cache)
    save(OUT / "summary_status.json", {"complete": True, "unique_documents": len(texts), "batches": len(batches),
                                        "new_requests": cache.new_requests, "cache_hits": cache.reused_requests,
                                        "source_corpora_sha256": file_hash(OUT / "source_corpora.json")})


def prepare(protocol, source):
    if not read(OUT / "summary_status.json")["complete"]:
        raise ValueError("Incomplete summary bank")
    banks = source_corpora(protocol, source)
    dense = risk.DenseScores(source)
    jobs = []
    for block, bank in banks["blocks"].items():
        assignment = read(risk.OUT / "partitions" / f"{block}.json")["assignment"]
        topics = set(bank["retained_topics"])
        for method in protocol["methods"]:
            tests, _, _ = suite(protocol, source, block, method)
            for scenario, state in [("calibration", "clean"), *itertools.product(protocol["scenarios"], ("watermarked", "clean"))]:
                raw = bank["calibration"] if scenario == "calibration" else (
                    bank["watermarked"][method] if state == "watermarked" else bank["clean"])
                if scenario == "summary_topic":
                    raw = [d for d in raw if assignment[f"{d['source_type']}:{d['source_id']}"] in topics]
                docs = [replace(IndexedMigrationDocument(**d), text=read(
                    OUT / "summaries" / f"{text_sha256(d['text'])}.json")["summary"]) for d in raw]
                for retriever in protocol["target_retrievers"]:
                    job = f"{block}/{method}/{scenario}/{retriever}/{state}"
                    path = OUT / "contexts" / job / "contexts.json"
                    jobs.append(job)
                    if path.exists():
                        continue
                    # Corpus encoding is shared across both retrievers, methods and deletion subsets.
                    dv = None
                    if retriever != "bm25":
                        dv = dense.encode([d.text for d in docs], OUT / "dense/documents" / f"{digest([d.text for d in docs])}.npz")
                        qv = dense.encode([u["question"] for u in tests], OUT / "dense/queries" / f"{digest([u['question'] for u in tests])}.npz")
                        scores = np.einsum("ij,kj->ik", qv, dv, optimize=True).astype(np.float32)
                    else:
                        scores = sparse_bm25_scores([d.text for d in docs], [u["question"] for u in tests]).toarray()
                    contexts, diagnostics = {}, {}
                    available = {(d.source_type, d.source_id) for d in docs}
                    for i, unit in enumerate(tests):
                        key = unit["unit_id"]
                        ctx = context_rows(scores[i], docs, top_k=protocol["top_k"])
                        anchors = set(bank["anchors"][key]) if method == "RAG-WM" else {key}
                        kind = "clean" if method == "RAG-WM" else "sentinel"
                        contexts[key] = ctx
                        diagnostics[key] = {
                            "carrier_present": state == "watermarked" and any((kind, key) in available for key in anchors),
                            "own_carrier_retrieved": state == "watermarked" and any(r["source"] == kind and r["id"] in anchors for r in ctx)}
                    payload = {"condition": {"block": block, "method": method, "scenario": scenario,
                                              "retriever": retriever, "state": state, "replicate_seed": protocol["replicate_seed"],
                                              "documents": len(docs), "retained_topics": sorted(topics)},
                               "tests": tests, "contexts": contexts, "diagnostics": diagnostics}
                    payload["signature"] = digest(payload)
                    save(path, payload)
                    print(f"[prepared] {job}: documents={len(docs)}", flush=True)
    save(OUT / "frozen_contexts.json", {"jobs": jobs, "contexts_sha256": {
        job: file_hash(OUT / "contexts" / job / "contexts.json") for job in jobs}})


async def execute(protocol, concurrency):
    frozen = read(OUT / "frozen_contexts.json")
    cache = cache_for(protocol, concurrency)
    prompts = read(ROOT / protocol["source_ragwm"] / "upstream_prompts.json")
    limiter = asyncio.Semaphore(concurrency)

    async def group(job, model):
        async with limiter:
            context_path = OUT / "contexts" / job / "contexts.json"
            if file_hash(context_path) != frozen["contexts_sha256"][job]:
                raise ValueError("Frozen contexts changed")
            payload = read(context_path)
            path = OUT / "execution" / job / model / "answers.json"
            data = read(path) if path.exists() else {"condition": payload["condition"], "model": model,
                                                    "signature": payload["signature"], "records": [], "complete": False}
            if data["signature"] != payload["signature"]:
                raise ValueError("Answer checkpoint mismatches frozen contexts")
            for unit in payload["tests"][len(data["records"]):]:
                key, question, expected = unit["unit_id"], unit["question"], unit["expected_answer"]
                answer = await cache.ask(model, PromptTemplates.answer_with_rag(
                    "\n".join(r["text"] for r in payload["contexts"][key]), question))
                if payload["condition"]["method"] == "RAG-WM":
                    correct, judge_key, _ = await ragwm.check_relation(cache, protocol, prompts, unit["relation_unit"], answer["content"])
                    kind = "ragwm_upstream_relation_oracle"
                elif explicit_abstention(answer["content"], expected):
                    correct, judge_key, kind = False, None, "exact_abstention_rule_v1"
                else:
                    for attempt in range(3):
                        prompt = PromptTemplates.judge_answer_correctness(question, answer["content"], expected)
                        verdict = await cache.ask(protocol["verifier_model"], prompt + (
                            "" if attempt == 0 else f"\nFormatting retry {attempt}: only Yes or No."))
                        try:
                            correct = yes_no(verdict["content"])
                            break
                        except ValueError:
                            continue
                    else:
                        raise ValueError("Malformed answer oracle")
                    judge_key, kind = verdict["key"], "glm_semantic"
                data["records"].append({**unit, "answer": answer["content"], "is_correct": correct,
                                         "answer_request": answer["key"], "judge_request": judge_key,
                                         "judge_kind": kind, **payload["diagnostics"][key]})
                data["complete"] = len(data["records"]) == protocol["unit_count"]
                save(path, data)
            print(f"[executed] {job}/{model}: {len(data['records'])}/30", flush=True)
    await asyncio.gather(*(group(job, model) for job in frozen["jobs"] for model in protocol["target_models"]))
    await close_cache(cache)
    save(OUT / "execution_status.json", {"complete": True, "answer_groups": len(frozen["jobs"]) * len(protocol["target_models"]),
                                         "new_requests": cache.new_requests, "cache_hits": cache.reused_requests})


def summarize(protocol, source):
    frozen = read(OUT / "frozen_contexts.json")
    groups = []
    for job, model in itertools.product(frozen["jobs"], protocol["target_models"]):
        data = read(OUT / "execution" / job / model / "answers.json")
        if not data["complete"] or len({r["unit_id"] for r in data["records"]}) != protocol["unit_count"]:
            raise ValueError("Incomplete answer group")
        groups.append(data)
    calibration = []
    calgroups = [g for g in groups if g["condition"]["scenario"] == "calibration"]
    tail = (1 - protocol["calibration_confidence"]) / len(calgroups)
    for group in calgroups:
        x, n = sum(r["is_correct"] for r in group["records"]), len(group["records"])
        upper = float(beta.ppf(1 - tail, x + 1, n - x)) if x < n else 1.0
        calibration.append({**group["condition"], "model": group["model"], "positives": x, "n": n, "rate": x / n, "upper": upper})
    p0 = max(protocol["minimum_common_p0"], max(r["upper"] for r in calibration))
    if p0 >= 1:
        raise ValueError("Calibration does not permit informative verification")
    save(OUT / "calibration.json", {"common_p0": p0, "groups": calibration,
                                     "scope": "empirical marginal-rate upper bound, not conditional-null proof"})
    rows, trajectories = [], []
    for group in groups:
        condition, model, records = group["condition"], group["model"], group["records"]
        if condition["scenario"] == "calibration":
            continue
        method = condition["method"]
        seeds = [0] if method == "RAVET" else protocol["policy_seeds"]
        tests, selected, clusters = suite(protocol, source, condition["block"], method)
        for seed_id in seeds:
            seed = stable_seed(protocol["protocol_version"], condition["block"], condition["scenario"], method, seed_id)
            if method == "RAVET":
                order = causal_replay(selected, clusters, records, seed=seed,
                                      p0=protocol["scheduler_p0"], limit=protocol["collection_horizon"])
            else:
                order = list(records)
                random.Random(seed).shuffle(order)
            labels = [r["is_correct"] for r in order]
            ev = evidence_curve(labels[:protocol["comparison_horizon"]], p0, protocol["alpha"])
            full = evidence_curve(labels, p0, protocol["alpha"])
            rows.append({**condition, "model": model, "policy_seed": seed_id, "p0": p0,
                         "edr15": sum(labels[:15]) / 15, "edr30": sum(labels) / len(labels),
                         "anytime15": ev["curve"][-1]["anytime_verified"], "queries15": ev["restricted_queries"],
                         "carrier_retrieval15": sum(r["own_carrier_retrieved"] for r in order[:15]) / 15,
                         "carrier_presence30": sum(r["carrier_present"] for r in order) / len(order),
                         "vsr30": full["curve"][-1]["fixed_verified"],
                         **{f"vsr{b}": ev["curve"][b - 1]["fixed_verified"] for b in (4, 6, 10, 15)}})
            trajectories.append({**condition, "model": model, "policy_seed": seed_id,
                                 "probe_order": [r["probe_id"] for r in order], "labels": labels,
                                 "curve": full["curve"]})
    metrics = ("edr15", "edr30", "vsr4", "vsr6", "vsr10", "vsr15", "vsr30", "anytime15", "queries15",
               "carrier_retrieval15", "carrier_presence30")
    keys = ("block", "method", "scenario", "retriever", "state", "model")
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    conditions = [{**dict(zip(keys, key, strict=True)), **{m: float(np.mean([v[m] for v in values])) for m in metrics}}
                  for key, values in grouped.items()]
    summary = []
    for method, scenario, state in itertools.product(protocol["methods"], protocol["scenarios"], ("watermarked", "clean")):
        values = [r for r in conditions if (r["method"], r["scenario"], r["state"]) == (method, scenario, state)]
        summary.append({"method": method, "scenario": scenario, "state": state, "configurations": len(values),
                        **{m: float(np.mean([v[m] for v in values])) for m in metrics}})
    save(OUT / "results.json", {"common_p0": p0, "summary": summary, "conditions": conditions})
    save(OUT / "trajectories.json", trajectories)
    with (OUT / "trajectory_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lock = read(OUT / "protocol_lock.json")
    summaries = [read(p) for p in (OUT / "summaries").glob("*.json")]
    audit = {"complete": True, "answer_records": sum(len(g["records"]) for g in groups),
             "answer_groups": len(groups), "positive_configurations_per_method": 16,
             "clean_configurations_per_method": 16, "calibration_groups": len(calgroups),
             "summary_documents": len(summaries), "mean_summary_word_ratio": float(np.mean([
                 s["summary_words"] / s["source_words"] for s in summaries])),
             "short_documents_preserved": sum(s.get("policy") == "preserve_short_document" for s in summaries),
             "contexts_valid": all(file_hash(OUT / "contexts" / j / "contexts.json") == h for j, h in frozen["contexts_sha256"].items()),
             "paper_unchanged": all(file_hash(ROOT / p) == h for p, h in lock["paper_sha256"].items())}
    save(OUT / "audit.json", audit)
    report(protocol, summary, conditions, p0, audit)
    print(json.dumps({"audit": audit, "p0": p0, "summary": summary}, indent=2), flush=True)


def report(protocol, summary, conditions, p0, audit):
    lines = ["# 1% 摘要压缩扩展实验", "", "## 范围与处理", "",
             "已冻结的单重复配对扩展：SciDocs、ArguAna；BM25、BGE-M3；DeepSeek-V4-Flash、Kimi-K3；top-k=5。",
             "3000 篇普通文档、30 个证据身份；独立无标记校准使用每语料另外 1000 篇文档。",
             "先插入，再对整个文档做盲摘要；正文均使用同一个 55% 词数上限（向上取整），不超过20词的短记录原样保留。",
             "摘要不接收测试问题、预期答案或标记身份。主题删除沿用压缩前冻结的 4/8 主题，不重新分组。",
             "RAG-WM 沿用已有原文内追加载体，不能把它称为与独立 sentinel 相同的文档注入密度。",
             "SentinelRAG/RAVET 保持已有测试；RAG-WM 使用官方关系判定 prompt，而非通用答案 oracle。", "",
             "## 主结果", "", "固定预算成功率与提前停止成功率分开报告。每个方法、每场景有 8 个正例部署；随机顺序种子先在部署内平均。", "",
             "| 场景 | 方法 | EDR@15 | VSR@6 | VSR@15 | 提前验证@15 | 截断查询数@15 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        if row["state"] == "watermarked":
            lines.append(f"| {row['scenario']} | {row['method']} | {row['edr15']:.1%} | {row['vsr6']:.1%} | {row['vsr15']:.1%} | {row['anytime15']:.1%} | {row['queries15']:.2f} |")
    lines += ["", "## 背景误报与完整性", "", f"独立压缩后背景校准得到统一 p0={p0:.6f}，alpha=0.01；方法排序仍使用已冻结的 scheduler p0=0.15。",
              "这是经验性边际背景校准，不是自适应条件零假设的数学证明。模型判定不等同于人工真值。", ""]
    for row in summary:
        if row["state"] == "clean":
            lines.append(f"- {row['scenario']} / {row['method']}：固定15次误报 {row['vsr15']:.1%}，提前停止误报 {row['anytime15']:.1%}。")
    lines += ["", f"完整采集 {audit['answer_records']} 个回答，{audit['answer_groups']} 个组，{audit['summary_documents']} 个不同原文摘要。",
              f"平均摘要/原文词数比 {audit['mean_summary_word_ratio']:.3f}；冻结 context 哈希核验 {audit['contexts_valid']}；论文未修改 {audit['paper_unchanged']}。",
              "每组离线采集30个不同证据测试，再仅用前缀回答因果重放调度；采集/摘要API费用不等同于部署查询数。", "",
              "## 配对结果与失败配置", ""]
    wins = losses = ties = 0
    for comparator in ("SentinelRAG", "RAG-WM"):
        w = l = t = 0
        bad = []
        for row in [r for r in conditions if r["method"] == "RAVET" and r["state"] == "watermarked"]:
            other = next(r for r in conditions if r["method"] == comparator and all(r[k] == row[k] for k in ("block", "scenario", "retriever", "state", "model")))
            delta = row["anytime15"] - other["anytime15"]
            qdelta = other["queries15"] - row["queries15"]
            if delta > 1e-9 or (abs(delta) <= 1e-9 and qdelta > 1e-9): w += 1
            elif delta < -1e-9 or qdelta < -1e-9:
                l += 1; bad.append(f"- {row['block']} / {row['scenario']} / {row['retriever']} / {row['model']}：RAVET 提前验证 {row['anytime15']:.1%}，{comparator} {other['anytime15']:.1%}；查询 {row['queries15']:.2f} vs {other['queries15']:.2f}。")
            else: t += 1
        lines += [f"对 {comparator}，按提前验证率优先、查询数其次比较：胜 {w}，负 {l}，平 {t}（共16配置）。", *bad, ""]
    lines += ["## 解释边界", "", "这是一个冻结重复上的必要扩展，不替代主实验三重复密度曲线。",
              "结果不能预设为更好：摘要可能保留、丢弃或改写任何方法的关系事实；完整失败配置与干净对照都已保留。",
              "该轮没有优化 sentinel、重选问题或按目标回答调参；如果出现优势，它是在既有方法上的部署变化迁移效果。", "",
              "## 文件", "", f"- 原始与汇总数据：`{OUT.relative_to(ROOT)}`", "- 冻结设置：`experiments/summary_density_20261001.json`",
              "- `results.json`、`trajectory_metrics.csv`、`trajectories.json`、`calibration.json`、`audit.json`。", ""]
    path = ROOT / "docs/SUMMARY_DENSITY_RESULTS_20261001.zh-CN.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "compress", "prepare", "execute", "summarize", "all"), default="all")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    protocol, source = freeze()
    if args.stage in ("preflight", "all"):
        async def preflight():
            cache = cache_for(protocol, args.concurrency)
            for name in [*protocol["target_models"], protocol["verifier_model"]]:
                record = await cache.ask(name, "Reply with the single word READY. Summary extension preflight.")
                print(f"[preflight] {record['model']}: complete response", flush=True)
            await close_cache(cache)
        asyncio.run(preflight())
    if args.stage in ("compress", "all"):
        asyncio.run(compress(protocol, source, args.concurrency))
    if args.stage in ("prepare", "all"):
        prepare(protocol, source)
    if args.stage in ("execute", "all"):
        asyncio.run(execute(protocol, args.concurrency))
    if args.stage in ("summarize", "all"):
        summarize(protocol, source)


if __name__ == "__main__":
    main()
