#!/usr/bin/env python3
"""Parallel matched retrieval/index changes, isolated from frozen paper results."""

import argparse
import asyncio
import itertools
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

import run_final_experiment_closeout as core
import run_ragwm_pilot as request_api
import run_summary_density as shared
from run_partition_confirmation import ROOT, digest, file_hash, read, save
from sentinelrag.prompts.templates import PromptTemplates
from sentinelrag.testing.adaptive import stable_seed
from sentinelrag.testing.corpus_retrieval import IndexedMigrationDocument, context_rows
from sentinelrag.testing.matrix import sparse_bm25_scores
from sentinelrag.testing.migration_confirmation import explicit_abstention
from sentinelrag.testing.ragwm import yes_no
from sentinelrag.testing.retrieval_deployment import (
    chunk_inventory, context_diagnostics, rechunk_documents,
)
from sentinelrag.testing.sentinel_paraphrase import cumulative_metrics
from sentinelrag.testing.summary_extension import causal_replay


PROTOCOL = ROOT / "experiments/retrieval_deployment_changes_20261003.json"
OUT = ROOT / "output/retrieval-deployment-changes-20261003-v1"
FIXTURES = {"SentinelRAG": "original", "RAVET": "relation_explicit"}
METRICS = ("vsr6", "vsr15", "mq15", "vsr30", "mq30")


def archived_path(protocol, block, method, retriever, state):
    density = "d1" if state == "watermarked" else "d2"
    return ROOT / protocol["source_density"] / "contexts" / block / FIXTURES[method] / density / str(
        protocol["replicate_seed"]) / "intact" / retriever / state / "contexts.json"


def freeze():
    protocol = read(PROTOCOL)
    source = read(ROOT / protocol["source_density"] / "protocol_lock.json")["protocol"]
    files = [PROTOCOL, Path(__file__).resolve(), ROOT / "src/sentinelrag/testing/retrieval_deployment.py",
             ROOT / "src/sentinelrag/testing/representation_attacks.py", ROOT / "scripts/run_summary_density.py",
             ROOT / "scripts/run_density_curve.py", ROOT / "scripts/run_ragwm_pilot.py",
             ROOT / "scripts/run_partition_confirmation.py", ROOT / "scripts/run_risk_low_density.py",
             ROOT / "src/sentinelrag/testing/summary_extension.py", ROOT / "src/sentinelrag/testing/adaptive.py",
             ROOT / "src/sentinelrag/prompts/templates.py", ROOT / protocol["source_corpora"],
             ROOT / protocol["source_corpora_lock"], ROOT / protocol["source_density"] / "protocol_lock.json"]
    files.extend(archived_path(protocol, *args) for args in itertools.product(
        protocol["blocks"], protocol["methods"], protocol["target_retrievers"], ("watermarked", "clean")))
    for block in protocol["blocks"]:
        files.extend(shared.risk.artifact_path(block, fixture) for fixture in FIXTURES.values())
    settings = {name: {"model": shared.load_model_preset(name, ROOT / "models").model,
                       "args": shared.load_model_preset(name, ROOT / "models").llm_arg}
                for name in [*protocol["target_models"], protocol["verifier_model"]]}
    prior = read(core.OUT / "protocol_lock.json")
    assert all(settings[k] == prior["model_settings"][k] for k in settings)
    lock = {"protocol": protocol, "source_protocol": source, "model_settings": settings,
            "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in files},
            "frozen_source_sha256": prior["frozen_source_sha256"]}
    path = OUT / "protocol_lock.json"
    if path.exists():
        if read(path) != lock:
            raise ValueError("Frozen retrieval-change study changed; preserve and version")
    else:
        save(path, lock)
    verify(lock)
    return protocol, source, lock


def verify(lock):
    assert all(file_hash(ROOT / name) == sha for name, sha in lock["source_sha256"].items())
    return core.verify_unchanged(lock)


def source_banks(protocol):
    banks = read(ROOT / protocol["source_corpora"])
    assert banks["signature"] == digest(banks["blocks"])
    return banks["blocks"]


def inventory(protocol, source):
    rows = []
    for block, bank in source_banks(protocol).items():
        assert len(bank["clean"]) == 3000
        identities = None
        for method in protocol["methods"]:
            tests, _, _ = shared.suite(protocol, source, block, method)
            own = {d["source_id"] for d in bank["watermarked"][method] if d["source_type"] == "sentinel"}
            assert own == {t["unit_id"] for t in tests} and len(own) == 30
            assert identities is None or own == identities
            identities = own
            docs = [IndexedMigrationDocument(**d) for d in bank["watermarked"][method]]
            for row in chunk_inventory(docs, size=protocol["chunk_size_words"], overlap=protocol["chunk_overlap_words"]):
                rows.append({"block": block, "method": method, **row})
    assert any(r["split_documents"] for r in rows if r["source_type"] == "sentinel")
    payload = {"outcome_blind": True, "chunk_size_words": protocol["chunk_size_words"],
               "chunk_overlap_words": protocol["chunk_overlap_words"], "rows": rows}
    save(OUT / "chunk_inventory.json", payload)
    save(OUT / "source_audit.json", {"source": protocol["source_corpora"],
         "sampling_repetition_not_independent_kb": True, "source_density": 30 / 3030,
         "density_unit": "logical source documents, not physical chunks", "logical_sentinels_per_method": 30})
    print(json.dumps(payload), flush=True)


def prepare(protocol, source, lock):
    inventory(protocol, source)
    banks = source_banks(protocol)
    corpus_chunks = {}
    for block, bank in banks.items():
        for state, method in itertools.product(("watermarked", "clean"), protocol["methods"]):
            raw = bank["watermarked"][method] if state == "watermarked" else bank["clean"]
            docs = [IndexedMigrationDocument(**d) for d in raw]
            corpus_chunks[block, method, state] = rechunk_documents(
                docs, size=protocol["chunk_size_words"], overlap=protocol["chunk_overlap_words"])
    dense = shared.risk.DenseScores(source)
    texts = sorted({d.text for docs in corpus_chunks.values() for d in docs})
    vectors = dense.encode(texts, OUT / "dense/shared_chunks.npz")
    positions = {text: i for i, text in enumerate(texts)}
    jobs = []
    for block, method, state, retriever in itertools.product(
            protocol["blocks"], protocol["methods"], ("watermarked", "clean"), protocol["target_retrievers"]):
        tests, _, _ = shared.suite(protocol, source, block, method)
        reference = read(archived_path(protocol, block, method, retriever, state))
        raw = banks[block]["watermarked"][method] if state == "watermarked" else banks[block]["clean"]
        original = {(d["source_type"], d["source_id"], d["chunk_index"]): d["text"] for d in raw}
        original_contexts = {t["unit_id"]: reference["contexts"][t["probe_id"]] for t in tests}
        assert all(len(rows) == 5 for rows in original_contexts.values())
        assert all(original[r["source"], r["id"], r["chunk_index"]] == r["text"]
                   for rows in original_contexts.values() for r in rows)
        docs = corpus_chunks[block, method, state]
        if retriever == "bm25":
            scores = sparse_bm25_scores([d.text for d in docs], [t["question"] for t in tests]).toarray()
        else:
            queries = dense.encode([t["question"] for t in tests], OUT / "dense/queries" / f"{digest(tests)}.npz")
            scores = np.einsum("ij,kj->ik", queries, vectors[[positions[d.text] for d in docs]],
                               optimize=True).astype(np.float32)
        for scenario in protocol["scenarios"]:
            job = f"{block}/{method}/{scenario}/{retriever}/{state}"
            jobs.append(job)
            path = OUT / "contexts" / job / "contexts.json"
            if path.exists():
                continue
            contexts, diagnostics = {}, {}
            for i, test in enumerate(tests):
                identity = test["unit_id"]
                ctx = (context_rows(scores[i], docs, top_k=5) if scenario == "rechunk_120_30" else
                       original_contexts[identity][:protocol["scenario_top_k"][scenario]])
                contexts[identity] = ctx
                diagnostics[identity] = context_diagnostics(test, ctx, carrier_present=state == "watermarked")
            payload = {"condition": {"block": block, "method": method, "scenario": scenario,
                       "retriever": retriever, "state": state, "replicate_seed": protocol["replicate_seed"],
                       "logical_documents": len(raw), "physical_chunks": len(docs) if scenario == "rechunk_120_30" else len(raw),
                       "top_k": protocol["scenario_top_k"][scenario], "injected_sentinels": 30 if state == "watermarked" else 0},
                       "tests": tests, "contexts": contexts, "diagnostics": diagnostics}
            payload["signature"] = digest(payload)
            save(path, payload)
            print(f"[prepared] {job}", flush=True)
    save(OUT / "frozen_contexts.json", {"jobs": jobs, "contexts_sha256": {
        job: file_hash(OUT / "contexts" / job / "contexts.json") for job in jobs}})
    verify(lock)


def cache_roots(protocol):
    return (core.OUT, ROOT / protocol["source_density"], shared.risk.SOURCE,
            ROOT / "output/summary-density-20261001-v6", ROOT / "output/closeout-summary-repetition-20261003-v1",
            ROOT / "output/closeout-wrong-owner-20261003-v1")


async def execute(protocol, lock, concurrency, *, preflight_only=False):
    cache = request_api.AuditCache(protocol, concurrency, output_root=OUT, readonly_roots=cache_roots(protocol))
    try:
        for name in cache.clients:
            row = await cache.ask(name, "Reply with the single word READY.")
            assert row["returned_model"] == lock["model_settings"][name]["model"]
            save(OUT / "preflight" / f"{name}.json", {k: row[k] for k in ("model", "returned_model", "key", "finish_reason")})
            print(f"[preflight] {row['model']}: complete", flush=True)
        if preflight_only:
            return
        frozen = read(OUT / "frozen_contexts.json")
        groups = asyncio.Semaphore(concurrency)
        async def collect(job, model):
            async with groups:
                path = OUT / "contexts" / job / "contexts.json"
                assert file_hash(path) == frozen["contexts_sha256"][job]
                context = read(path)
                destination = OUT / "execution" / job / model / "answers.json"
                data = read(destination) if destination.exists() else {
                    "condition": context["condition"], "model": model, "signature": context["signature"],
                    "records": [], "complete": False}
                assert data["signature"] == context["signature"]
                for test in context["tests"][len(data["records"]):]:
                    prompt = PromptTemplates.answer_with_rag("\n".join(r["text"] for r in context["contexts"][test["unit_id"]]), test["question"])
                    answer = await cache.ask(model, prompt)
                    assert answer["returned_model"] == lock["model_settings"][model]["model"]
                    if explicit_abstention(answer["content"], test["expected_answer"]):
                        correct, judge_key, kind = False, None, "exact_abstention_rule_v1"
                    else:
                        base = PromptTemplates.judge_answer_correctness(test["question"], answer["content"], test["expected_answer"])
                        for attempt in range(3):
                            verdict = await cache.ask(protocol["verifier_model"], base + (
                                "" if attempt == 0 else f"\nFormatting retry {attempt}: only Yes or No."))
                            try:
                                correct = yes_no(verdict["content"])
                                break
                            except ValueError:
                                continue
                        else:
                            raise ValueError("Malformed oracle; no negative label imputation")
                        judge_key, kind = verdict["key"], "glm_semantic"
                    data["records"].append({**test, "answer": answer["content"], "is_correct": correct,
                        "answer_request": answer["key"], "judge_request": judge_key, "judge_kind": kind,
                        **context["diagnostics"][test["unit_id"]]})
                    data["complete"] = len(data["records"]) == 30
                    save(destination, data)
                print(f"[executed] {job}/{model}: 30/30", flush=True)
        # Jobs interleave top-3, rechunking and reference under one shared limiter/cache.
        await asyncio.gather(*(collect(job, model) for job in frozen["jobs"] for model in protocol["target_models"]))
        verify(lock)
        save(OUT / "execution_status.json", {"complete": True, "answer_groups": len(frozen["jobs"]) * 2,
             "new_requests": cache.new_requests, "cache_hits": cache.reused_requests,
             "parallel_group_limit": concurrency, "shared_request_limiter": True})
    finally:
        await shared.close_cache(cache)


def replay(protocol, source, decision_p0):
    frozen = read(OUT / "frozen_contexts.json")
    values, trajectories = [], []
    for job, model in itertools.product(frozen["jobs"], protocol["target_models"]):
        data = read(OUT / "execution" / job / model / "answers.json")
        assert data["complete"] and len({r["unit_id"] for r in data["records"]}) == 30
        condition = data["condition"]
        _, selected, clusters = shared.suite(protocol, source, condition["block"], condition["method"])
        runs = []
        for seed_id in ((0,) if condition["method"] == "RAVET" else protocol["policy_seeds"]):
            if condition["method"] == "RAVET":
                ordered = causal_replay(selected, clusters, data["records"], seed=0,
                                       p0=protocol["scheduler_p0"], limit=30)
            else:
                ordered = sorted(data["records"], key=lambda r: r["unit_id"])
                seed = stable_seed(protocol["protocol_version"], condition["block"], protocol["replicate_seed"], seed_id)
                random.Random(seed).shuffle(ordered)
            metrics, curve = cumulative_metrics([r["is_correct"] for r in ordered], decision_p0, protocol["alpha"])
            runs.append(metrics)
            trajectories.append({**condition, "model": model, "policy_seed": seed_id,
                                 "probe_order": [r["probe_id"] for r in ordered], "unit_order": [r["unit_id"] for r in ordered],
                                 "labels": [r["is_correct"] for r in ordered], "curve": curve})
        values.append({**condition, "model": model, **{m: float(np.mean([r[m] for r in runs])) for m in METRICS},
                       "answer_positive_count": sum(r["is_correct"] for r in data["records"]),
                       "own_carrier_retrieval_fraction": np.mean([r["own_carrier_retrieved"] for r in data["records"]]),
                       "own_literal_support_fraction": np.mean([r["own_answer_literal_support"] for r in data["records"]])})
    cells = []
    for scenario, state, method in itertools.product(protocol["scenarios"], ("watermarked", "clean"), protocol["methods"]):
        rows = [r for r in values if (r["scenario"], r["state"], r["method"]) == (scenario, state, method)]
        assert len(rows) == 8
        cells.append({"scenario": scenario, "state": state, "method": method, "conditions": 8,
                      **{m: float(np.mean([r[m] for r in rows])) for m in (*METRICS, "answer_positive_count",
                          "own_carrier_retrieval_fraction", "own_literal_support_fraction")}})
    return values, cells, trajectories


def report(protocol, source, lock):
    values, cells, trajectories = replay(protocol, source, protocol["decision_p0"])
    contrast = []
    for scenario in protocol["scenarios"]:
        methods = {r["method"]: r for r in cells if r["scenario"] == scenario and r["state"] == "watermarked"}
        baseline, ours = methods["SentinelRAG"], methods["RAVET"]
        contrast.append({"scenario": scenario, "vsr6_gain_pp": 100 * (ours["vsr6"] - baseline["vsr6"]),
                         "vsr15_gain_pp": 100 * (ours["vsr15"] - baseline["vsr15"]),
                         "vsr30_gain_pp": 100 * (ours["vsr30"] - baseline["vsr30"]),
                         "mq15_saving_percent": 100 * (1 - ours["mq15"] / baseline["mq15"]),
                         "mq30_saving_percent": 100 * (1 - ours["mq30"] / baseline["mq30"])})
    sensitivity = [{"decision_p0": p0, "scheduler_p0_unchanged": protocol["scheduler_p0"],
                    "summary": replay(protocol, source, p0)[1]} for p0 in (.15, .2884, .3004, .3906)]
    result = {"complete": True, "decision_p0": protocol["decision_p0"],
              "logical_answer_records": 30 * len(values), "new_setting_logical_records": 30 * sum(
                  r["scenario"] != "top5_reference" for r in values),
              "summary": cells, "conditions": values, "contrasts": contrast,
              "decision_sensitivity_diagnostic": sensitivity, "chunk_inventory": read(OUT / "chunk_inventory.json"),
              "source_audit": read(OUT / "source_audit.json"), **verify(lock)}
    save(OUT / "trajectories.json", trajectories)
    save(OUT / "results.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("conditions", "decision_sensitivity_diagnostic", "chunk_inventory")}), flush=True)


def audit(protocol, source, lock):
    frozen = read(OUT / "frozen_contexts.json")
    roots = (OUT, *cache_roots(protocol))
    accepted, fresh, returned = {}, Counter(), Counter()
    def check(key, model, prompts, content=None):
        settings = lock["model_settings"][model]
        allowed = {digest({"preset": model, "model": settings["model"], "args": settings["args"], "prompt": p}): p for p in prompts}
        assert key in allowed
        path = next((root / "requests" / key[:2] / f"{key}.json" for root in roots
                     if (root / "requests" / key[:2] / f"{key}.json").exists()), None)
        assert path is not None
        request = read(path)
        assert (request["key"], request["model"], request["prompt"], request["args"]) == (key, settings["model"], allowed[key], settings["args"])
        assert request["returned_model"] == settings["model"] and request["content"] and request["finish_reason"] != "length"
        assert content is None or content == request["content"]
        if key not in accepted:
            accepted[key] = {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)}
            returned[model, request["returned_model"]] += 1
            if path.is_relative_to(OUT):
                fresh[model] += 1
        return request
    checked = 0
    banks = source_banks(protocol)
    for job in frozen["jobs"]:
        path = OUT / "contexts" / job / "contexts.json"
        assert file_hash(path) == frozen["contexts_sha256"][job]
        context = read(path)
        assert context["signature"] == digest({k: v for k, v in context.items() if k != "signature"})
        condition = context["condition"]
        block, method, state = (condition[k] for k in ("block", "method", "state"))
        expected_tests, _, _ = shared.suite(protocol, source, block, method)
        assert context["tests"] == expected_tests
        raw = banks[block]["watermarked"][method] if state == "watermarked" else banks[block]["clean"]
        documents = [IndexedMigrationDocument(**d) for d in raw]
        if condition["scenario"] == "rechunk_120_30":
            documents = rechunk_documents(documents, size=protocol["chunk_size_words"], overlap=protocol["chunk_overlap_words"])
        lookup = {(d.source_type, d.source_id, d.chunk_index): d.text for d in documents}
        assert len(lookup) == len(documents)
        reference = read(archived_path(protocol, block, method, condition["retriever"], state))
        for test in context["tests"]:
            rows = context["contexts"][test["unit_id"]]
            assert len(rows) == condition["top_k"]
            assert all(lookup[r["source"], r["id"], r["chunk_index"]] == r["text"] for r in rows)
            if condition["scenario"] != "rechunk_120_30":
                assert rows == reference["contexts"][test["probe_id"]][:condition["top_k"]]
            assert context["diagnostics"][test["unit_id"]] == context_diagnostics(test, rows, carrier_present=state == "watermarked")
        for model in protocol["target_models"]:
            data = read(OUT / "execution" / job / model / "answers.json")
            assert data["complete"] and data["signature"] == context["signature"] and len(data["records"]) == 30
            for test, row in zip(context["tests"], data["records"], strict=True):
                assert all(row[k] == v for k, v in test.items())
                assert all(row[k] == v for k, v in context["diagnostics"][test["unit_id"]].items())
                prompt = PromptTemplates.answer_with_rag("\n".join(r["text"] for r in context["contexts"][test["unit_id"]]), test["question"])
                check(row["answer_request"], model, [prompt], row["answer"])
                if explicit_abstention(row["answer"], test["expected_answer"]):
                    assert not row["is_correct"] and row["judge_request"] is None
                else:
                    base = PromptTemplates.judge_answer_correctness(test["question"], row["answer"], test["expected_answer"])
                    verdict = check(row["judge_request"], protocol["verifier_model"], [base] + [
                        base + f"\nFormatting retry {i}: only Yes or No." for i in (1, 2)])
                    assert row["is_correct"] == yes_no(verdict["content"])
                checked += 1
    assert checked == 2880
    baseline_orders = {}
    values, cells, trajectories = replay(protocol, source, protocol["decision_p0"])
    assert read(OUT / "trajectories.json") == trajectories
    assert read(OUT / "results.json")["summary"] == cells
    for trace in trajectories:
        assert len(set(trace["unit_order"])) == 30
        if trace["method"] == "SentinelRAG":
            key = trace["block"], trace["policy_seed"]
            if key in baseline_orders:
                assert baseline_orders[key] == trace["unit_order"]
            baseline_orders[key] = trace["unit_order"]
    result = {"status": "pass", "checked_answer_records": checked, "unique_verified_requests": len(accepted),
              "fresh_unique_execution_requests_by_preset": dict(fresh),
              "returned_models": [{"preset": k[0], "returned_model": k[1], "unique_requests": v} for k, v in returned.items()],
              "baseline_orders_paired_across_scenarios": True, "accepted_requests": accepted, **verify(lock)}
    save(OUT / "provenance_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "accepted_requests"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("inventory", "prepare", "preflight", "execute", "report", "audit", "all"))
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    protocol, source, lock = freeze()
    if args.stage == "inventory":
        inventory(protocol, source)
    if args.stage in ("prepare", "all"):
        prepare(protocol, source, lock)
    if args.stage in ("preflight", "execute", "all"):
        asyncio.run(execute(protocol, lock, args.concurrency, preflight_only=args.stage == "preflight"))
    if args.stage in ("report", "execute", "all"):
        report(protocol, source, lock)
    if args.stage in ("audit", "execute", "all"):
        audit(protocol, source, lock)


if __name__ == "__main__":
    main()
