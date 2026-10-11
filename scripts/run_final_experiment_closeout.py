#!/usr/bin/env python3
"""Close experimental gaps using frozen archives and a matched static control."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

import run_density_curve as density
import run_ragwm_pilot as cache_api
from run_partition_confirmation import ROOT, digest, file_hash, read, save
from sentinelrag.prompts.templates import PromptTemplates
from sentinelrag.testing.closeout import (
    cumulative_metrics, draw_sensitivity, paired_point, static_initial_order,
)
from sentinelrag.testing.migration_confirmation import explicit_abstention
from sentinelrag.testing.ragwm import yes_no
from sentinelrag.utils.model_registry import load_model_preset


PROTOCOL = ROOT / "experiments/final_experiment_closeout_20261003.json"
OUT = ROOT / "output/final-experiment-closeout-20261003-v1"
SOURCE = density.OUT
RECEIPT = ROOT / "output/two-method-paper-revision-20261003-v6/revision_receipt.json"


def frozen(protocol):
    receipt = read(RECEIPT)
    before = dict(receipt["source_sha256"])
    for name in ("paper/fse2027/main.tex", "paper/fse2027/main.pdf",
                 "docs/FSE_2027_PAPER_DRAFT.md", "paper/fse2027/supplement.tex"):
        before[name] = file_hash(ROOT / name)
    for name, sha in receipt["source_sha256"].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    files = [PROTOCOL, Path(__file__), ROOT / "src/sentinelrag/testing/closeout.py",
             SOURCE / "jobs.json", SOURCE / "frozen_protocol.json",
             SOURCE / "protocol_lock.json", SOURCE / "confirmation/raw_metrics.json"]
    lock = {
        "protocol": protocol,
        "frozen_source_sha256": before,
        "execution_inputs_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in files},
        "model_settings": {
            name: {"model": load_model_preset(name, ROOT / "models").model,
                   "args": load_model_preset(name, ROOT / "models").llm_arg}
            for name in read(SOURCE / "protocol_lock.json")["protocol"]["target_models"]
            + [read(SOURCE / "protocol_lock.json")["protocol"]["verifier_model"]]
        },
    }
    path = OUT / "protocol_lock.json"
    if path.exists() and read(path) != lock:
        raise ValueError("Closeout protocol changed; preserve old version")
    if not path.exists():
        save(path, lock)
    return lock


def verify_unchanged(lock):
    changes = [name for name, sha in lock["frozen_source_sha256"].items()
               if file_hash(ROOT / name) != sha]
    if changes:
        raise ValueError(f"Original evidence/manuscript changed: {changes}")
    return {"frozen_files_checked": len(lock["frozen_source_sha256"]),
            "original_evidence_and_manuscript_unchanged": True}


def key(row):
    return tuple(row[k] for k in (
        "block", "density", "replicate_seed", "scenario", "retriever", "state", "model",
    ))


def combine(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[key(row), row["method"]].append(row)
    paired = []
    for condition in sorted({k[0] for k in grouped}):
        methods = {}
        metadata = None
        for method in ("SentinelRAG", "RAVET"):
            items = grouped[condition, method]
            expected = 3 if method == "SentinelRAG" else 1
            if len(items) != expected:
                raise ValueError("Incomplete primary method pair")
            metadata = items[0]
            methods[method] = {metric: float(np.mean([r[metric] for r in items]))
                               for metric in ("vsr6", "vsr15", "mq15")}
        paired.append({k: metadata[k] for k in (
            "block", "density", "replicate_seed", "scenario", "retriever", "state", "model",
        )} | {"baseline": methods["SentinelRAG"], "candidate": methods["RAVET"]})
    return paired


def primary_analysis(protocol, lock):
    raw = read(SOURCE / "confirmation/raw_metrics.json")
    archive_protocol = read(SOURCE / "protocol_lock.json")["protocol"]
    assert len(raw) == 768 and archive_protocol["p0"] == protocol["decision_p0"]
    trace_hashes, labels = {}, []
    for row in raw:
        job = "/".join(str(row[k]) for k in (
            "block", "fixture", "density", "replicate_seed", "scenario", "retriever", "state",
        ))
        path = density.trace_path("confirmation", job, row["model"], row["selector"],
                                  row["scheduler"], row["policy_seed_id"])
        trace = read(path)
        if not trace["complete"] or len(trace["records"]) != 15:
            raise ValueError("Primary trace incomplete")
        if len({r["canary_id"] for r in trace["records"]}) != 15:
            raise ValueError("Non-distinct primary evidence")
        outcomes = [r["is_correct"] for r in trace["records"]]
        metrics = cumulative_metrics(outcomes, protocol["decision_p0"])
        for metric, original in (("vsr6", "anytime_6"), ("vsr15", "anytime_15"),
                                 ("mq15", "restricted_15")):
            if metrics[metric] != row[original]:
                raise ValueError("Cumulative raw result does not replay")
        labels.append({**row, "method": "SentinelRAG" if row["fixture"] == "original" else "RAVET",
                       "labels": outcomes, **metrics})
        trace_hashes[str(path.relative_to(ROOT))] = file_hash(path)
    positives = combine([r for r in labels if r["state"] == "watermarked"])
    assert len(positives) == 144
    result = {"kind": "frozen-archive statistical analysis; no fresh calls",
              "uncertainty": draw_sensitivity(positives)}
    result["by_scenario"] = [
        {"scenario": scenario, **draw_sensitivity([r for r in positives if r["scenario"] == scenario])}
        for scenario in ("intact", "topic_subset")
    ]
    seeds = []
    for seed in protocol["ordering_seed_sensitivity"]:
        selected = [r for r in labels if r["state"] == "watermarked"
                    and (r["method"] == "RAVET" or r["policy_seed_id"] == seed)]
        grouped = defaultdict(dict)
        for row in selected:
            grouped[key(row)][row["method"]] = {k: row[k] for k in ("vsr6", "vsr15", "mq15")}
        pairs = [{"baseline": m["SentinelRAG"], "candidate": m["RAVET"]} for m in grouped.values()]
        assert len(pairs) == 144
        seeds.append({"ordering_seed": seed, **paired_point(pairs)})
    result["baseline_ordering_seed_sensitivity"] = seeds
    sensitivity = []
    for p0 in protocol["decision_sensitivity_grid"]:
        rescored = [{**r, **cumulative_metrics(r["labels"], p0)} for r in labels]
        positive = combine([r for r in rescored if r["state"] == "watermarked"])
        clean = combine([r for r in rescored if r["state"] == "clean"])
        sensitivity.append({"decision_p0": p0, "scheduler_p0_unchanged": 0.15,
                            **paired_point(positive), "clean_configurations_per_method": len(clean),
                            "clean_fpr15": {
                                "SentinelRAG": float(np.mean([r["baseline"]["vsr15"] for r in clean])),
                                "RAVET": float(np.mean([r["candidate"]["vsr15"] for r in clean])),
                            }})
    result["fixed_order_decision_sensitivity"] = sensitivity
    result["source_trace_sha256"] = trace_hashes
    result.update(verify_unchanged(lock))
    save(OUT / "primary_analysis.json", result)
    save(OUT / "primary_pairs.json", positives)
    print(json.dumps({k: v for k, v in result.items() if k not in ("uncertainty", "source_trace_sha256", "by_scenario")}, ensure_ascii=False), flush=True)


@lru_cache(maxsize=2)
def selected_pool(block):
    _, probes = density.owner_probes(block, "relation_explicit", density.source_protocol())
    return density.selected_for(probes, "failure_repair", {p.profile.candidate.canary_id for p in probes})


def static_jobs():
    return [j for j in read(SOURCE / "jobs.json")["confirmation"] if "/relation_explicit/" in j]


def static_order(payload):
    condition = payload["condition"]
    selected = [p for p in selected_pool(condition["block"])
                if p.profile.candidate.canary_id in set(condition["injected_ids"])]
    if len(selected) != condition["injected_sentinels"]:
        raise ValueError("Frozen selected-suite mismatch")
    return static_initial_order(selected)[:15]


def static_path(job, model):
    return OUT / "static_control" / "traces" / job / model / "answers.json"


async def static_collection(protocol, lock, online):
    source_protocol = read(SOURCE / "protocol_lock.json")["protocol"]
    cache = cache_api.AuditCache(source_protocol, 8, output_root=OUT,
        readonly_roots=(SOURCE, density.OLD_DENSITY, density.OWNER_SOURCE,
                        ROOT / "output/summary-density-20261001-v6"))
    progress = {"completed": 0, "expected": 192, "new_requests": 0, "cache_hits": 0}
    limiter = asyncio.Semaphore(8)
    try:
        if online:
            for name, (client, preset) in cache.clients.items():
                answer = await client.async_client.with_options(max_retries=0).chat.completions.create(
                    model=preset.model, messages=[{"role": "user", "content": "Reply only OK."}],
                    timeout=60, max_tokens=128)
                if not answer.choices[0].message.content:
                    raise ValueError(f"Preflight empty: {name}")
                save(OUT / "static_control" / "preflight" / f"{name}.json",
                     {"requested_model": preset.model, "returned_model": answer.model,
                      "finish_reason": answer.choices[0].finish_reason,
                      "content": answer.choices[0].message.content})
        async def collect(job, model):
            async with limiter:
                payload = read(SOURCE / "contexts" / job / "contexts.json")
                ordered = static_order(payload)
                adaptive_path = density.trace_path("confirmation", job, model, "failure_repair", "bayes", 0)
                adaptive = read(adaptive_path)
                if adaptive["context_signature"] != payload["signature"]:
                    raise ValueError("Adaptive and static contexts differ")
                recorded = {r["probe_id"]: r for r in adaptive["records"]}
                path = static_path(job, model)
                rows = []
                if path.exists():
                    existing = read(path)
                    if existing["context_signature"] != payload["signature"]:
                        raise ValueError("Static checkpoint context changed")
                    rows = existing["records"]
                def checkpoint():
                    save(path, {"condition": payload["condition"], "model": model,
                                "context_signature": payload["signature"],
                                "selected_probe_ids": [p.profile.candidate.probe_id for p in ordered],
                                "complete": len(rows) == 15, "records": rows})
                for index, probe in enumerate(ordered):
                    candidate = probe.profile.candidate
                    if index < len(rows):
                        if rows[index]["probe_id"] != candidate.probe_id:
                            raise ValueError("Static checkpoint order changed")
                        continue
                    if candidate.probe_id in recorded:
                        row = {**recorded[candidate.probe_id], "collection_origin": "archived_adaptive_exact_context"}
                    else:
                        prompt = PromptTemplates.answer_with_rag(
                            "\n".join(r["text"] for r in payload["contexts"][candidate.probe_id]), candidate.question)
                        if not online:
                            client, preset = cache.clients[model]
                            cache_key = digest({"preset": model, "model": preset.model,
                                                "args": preset.llm_arg, "prompt": prompt})
                            if not any((root / "requests" / cache_key[:2] / f"{cache_key}.json").exists()
                                       for root in (OUT, *cache.readonly_roots)):
                                checkpoint()
                                return
                        answer = await cache.ask(model, prompt)
                        if explicit_abstention(answer["content"], candidate.expected_answer):
                            verdict, judge_kind = {"key": None, "content": "No"}, "exact_abstention_rule_v1"
                        else:
                            oracle = PromptTemplates.judge_answer_correctness(
                                candidate.question, answer["content"], candidate.expected_answer)
                            if not online:
                                client, preset = cache.clients[source_protocol["verifier_model"]]
                                cache_key = digest({"preset": source_protocol["verifier_model"], "model": preset.model,
                                                    "args": preset.llm_arg, "prompt": oracle})
                                if not any((root / "requests" / cache_key[:2] / f"{cache_key}.json").exists()
                                           for root in (OUT, *cache.readonly_roots)):
                                    checkpoint()
                                    return
                            verdict = await cache.ask(source_protocol["verifier_model"], oracle)
                            judge_kind = "glm_semantic"
                        row = {"probe_id": candidate.probe_id, "canary_id": candidate.canary_id,
                               "question": candidate.question, "expected_answer": candidate.expected_answer,
                               "answer": answer["content"], "is_correct": yes_no(verdict["content"]),
                               "answer_request": answer["key"], "judge_request": verdict["key"],
                               "judge_kind": judge_kind, "collection_origin": "exact_request_cache_or_fresh",
                               **payload["diagnostics"][candidate.probe_id]}
                    rows.append(row)
                    checkpoint()
                progress["completed"] += 1
                progress.update(new_requests=cache.new_requests, cache_hits=cache.reused_requests)
                save(OUT / "static_control" / "status.json", {"status": "running", **progress})
                if progress["completed"] % 16 == 0:
                    print(json.dumps(progress), flush=True)
        await asyncio.gather(*(collect(job, model) for job in static_jobs()
                               for model in source_protocol["target_models"]))
        progress.update(new_requests=cache.new_requests, cache_hits=cache.reused_requests)
        save(OUT / "static_control" / "status.json", {
            "status": "complete" if progress["completed"] == progress["expected"] else "cache_only_partial",
            "mode": "online_completion" if online else "cache_only", **progress,
            **verify_unchanged(lock)})
        print(json.dumps(progress), flush=True)
    finally:
        for client, _ in cache.clients.values():
            await client.async_client.close()


def static_analysis(protocol, lock):
    source_protocol = read(SOURCE / "protocol_lock.json")["protocol"]
    paired, clean, hashes = [], [], {}
    for job in static_jobs():
        for model in source_protocol["target_models"]:
            path = static_path(job, model)
            static = read(path)
            adaptive_path = density.trace_path("confirmation", job, model, "failure_repair", "bayes", 0)
            adaptive = read(adaptive_path)
            if not static["complete"] or not adaptive["complete"]:
                raise ValueError("Static/adaptive pair incomplete")
            assert static["context_signature"] == adaptive["context_signature"]
            left = cumulative_metrics([r["is_correct"] for r in static["records"]], protocol["decision_p0"])
            right = cumulative_metrics([r["is_correct"] for r in adaptive["records"]], protocol["decision_p0"])
            row = {**static["condition"], "model": model, "baseline": left, "candidate": right}
            (paired if row["state"] == "watermarked" else clean).append(row)
            for file in (path, adaptive_path):
                hashes[str(file.relative_to(ROOT))] = file_hash(file)
    assert len(paired) == 144 and len(clean) == 48
    result = {"comparison": "Frozen initial utility versus answer-feedback updates, same selected RAVET suite",
              "offline_acquisition_separate_from_simulated_query_cost": True,
              "point": paired_point(paired), "uncertainty": draw_sensitivity(paired),
              "by_scenario": [{"scenario": s, **paired_point([r for r in paired if r["scenario"] == s])}
                              for s in ("intact", "topic_subset")],
              "by_density_scenario": [{"density": d, "scenario": s,
                                       **paired_point([r for r in paired if (r["density"], r["scenario"]) == (d, s)])}
                                      for d in ("d0p5", "d1", "d2") for s in ("intact", "topic_subset")],
              "mq_wins_ties_losses": {
                  "adaptive_lower": sum(r["candidate"]["mq15"] < r["baseline"]["mq15"] for r in paired),
                  "tied": sum(r["candidate"]["mq15"] == r["baseline"]["mq15"] for r in paired),
                  "adaptive_higher": sum(r["candidate"]["mq15"] > r["baseline"]["mq15"] for r in paired)},
              "clean_fpr15": {"static": float(np.mean([r["baseline"]["vsr15"] for r in clean])),
                              "adaptive": float(np.mean([r["candidate"]["vsr15"] for r in clean]))},
              "source_sha256": hashes, **verify_unchanged(lock)}
    save(OUT / "static_control" / "analysis.json", result)
    save(OUT / "static_control" / "pairs.json", paired)
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_sha256", "uncertainty")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("analysis", "cache", "online", "static-report"))
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    lock = frozen(protocol)
    if args.command == "analysis":
        primary_analysis(protocol, lock)
    elif args.command in ("cache", "online"):
        asyncio.run(static_collection(protocol, lock, args.command == "online"))
    else:
        static_analysis(protocol, lock)


if __name__ == "__main__":
    main()
