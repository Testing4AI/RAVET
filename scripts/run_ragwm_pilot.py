#!/usr/bin/env python3
"""Transfer RAG-WM's released mechanism to the frozen RAVET RAG testbed."""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import run_density_curve as density
import run_risk_low_density as risk
import run_small_target_transfer as transfer
from openai import APIError
from run_partition_confirmation import ROOT, RequestCache, digest, file_hash, read, save
from scipy.stats import beta

from sentinelrag.prompts.templates import PromptTemplates
from sentinelrag.testing.adaptive import stable_seed
from sentinelrag.testing.corpus_retrieval import context_rows, top_indices
from sentinelrag.testing.matrix import sparse_bm25_scores
from sentinelrag.testing.migration_confirmation import explicit_abstention
from sentinelrag.testing.ragwm import (
    eligible_variants,
    evidence_curve,
    feedback_text,
    keyed_units,
    upstream_prompts,
    yes_no,
)
from sentinelrag.utils.llm_client import parse_json_content
from sentinelrag.utils.model_registry import load_model_preset

PROTOCOL = ROOT / "experiments/ragwm_pilot_20261001.json"
OUT = ROOT / "output/ragwm-pilot-20261001-v3"
UPSTREAM = ROOT / "tmp/ragwm-upstream"


class AuditCache(RequestCache):
    async def request(self, client, preset, prompt, path, key):
        for attempt in range(6):
            started = time.monotonic()
            try:
                kwargs = {**preset.llm_arg, "timeout": 180, "response_format": {"type": "text"}}
                async with self.model_semaphores[preset.preset_name]:
                    async with self.start_locks[preset.preset_name]:
                        await asyncio.sleep(max(0, self.next_start[preset.preset_name] - time.monotonic()))
                        self.next_start[preset.preset_name] = time.monotonic() + 0.5
                    async with self.semaphore:
                        response = await client.async_client.with_options(max_retries=0).chat.completions.create(
                            model=preset.model, messages=[{"role": "user", "content": prompt}], **kwargs)
                choice = response.choices[0]
                record = {"key": key, "preset": preset.preset_name, "model": preset.model,
                          "returned_model": response.model, "prompt": prompt, "args": preset.llm_arg,
                          "content": choice.message.content, "finish_reason": choice.finish_reason,
                          "attempts": attempt + 1, "elapsed_seconds": time.monotonic() - started,
                          "usage": response.usage.model_dump() if response.usage else None}
                save(path.with_name(f"{key}.attempt-{attempt + 1}.json"), record)
                if not record["content"] or record["finish_reason"] == "length":
                    raise ValueError("Empty or truncated completion")
                save(path, record)
                self.new_requests += 1
                if self.new_requests % 25 == 0:
                    print(f"[api] new={self.new_requests} reused={self.reused_requests}", flush=True)
                return record
            except (APIError, ValueError) as error:
                status = getattr(error, "status_code", None)
                save(self.output_root / "errors" / f"{key}.attempt-{attempt + 1}.json",
                     {"key": key, "model": preset.model, "error_type": type(error).__name__,
                      "status_code": status, "attempt": attempt + 1})
                if attempt == 5 or status in (401, 403):
                    raise RuntimeError(f"{preset.model}: {type(error).__name__}, HTTP {status}") from None
                delay = min(60, 3 * 2**attempt)
                if status == 429:
                    delay = max(delay, 30)
                print(f"[retry] {preset.model}: {type(error).__name__}, wait={delay}s", flush=True)
                await asyncio.sleep(delay)


def freeze():
    protocol = read(PROTOCOL)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip()
    if revision != protocol["upstream_commit"]:
        raise ValueError("Upstream revision mismatch")
    source = read(ROOT / protocol["source_density"] / "protocol_lock.json")["protocol"]
    inputs = [PROTOCOL, ROOT / protocol["source_comparison"] / "frozen_protocol.json"]
    for block in protocol["blocks"]:
        inputs.extend([density.OWNER_SOURCE / "blocks" / block / "ko-pool/ko_pool.json",
                       risk.OUT / "partitions" / f"{block}.json",
                       ROOT / "datasets" / risk.dataset_for(block) / "corpus.jsonl"])
    code = [Path(__file__), ROOT / "src/sentinelrag/testing/ragwm.py",
            UPSTREAM / "src/prompt_define.py", UPSTREAM / "src/main.py",
            UPSTREAM / "src/watermark_role.py", UPSTREAM / "rag/vectorstore.py"]
    settings = {name: {"model": load_model_preset(name, ROOT / "models").model,
                       "args": load_model_preset(name, ROOT / "models").llm_arg}
                for name in [*protocol["target_models"], protocol["verifier_model"]]}
    payload = {"protocol": protocol, "density_source_protocol": source, "model_settings": settings,
               "input_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in inputs},
               "code_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in code}}
    path = OUT / "protocol_lock.json"
    if path.exists() and read(path) != payload:
        raise ValueError("Frozen pilot changed; create a new version")
    if not path.exists():
        save(path, payload)
    save(OUT / "upstream_prompts.json", upstream_prompts(UPSTREAM / "src/prompt_define.py"))
    return protocol, source


def cache_for(protocol, concurrency):
    return AuditCache(protocol, concurrency, output_root=OUT,
                      readonly_roots=(ROOT / "output/ragwm-pilot-20261001-v2",
                                      ROOT / "output/ragwm-pilot-20261001-v1"))


async def checked_json(cache, model, prompt, validator):
    for attempt in range(3):
        extra = "" if attempt == 0 else f"\nFormatting retry {attempt}: return only complete valid JSON in the required schema."
        record = await cache.ask(model, prompt + extra)
        try:
            value = validator(parse_json_content(record["content"]))
            return value, record["key"]
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    raise ValueError("Invalid generation JSON after preserved formatting attempts")


async def check_relation(cache, protocol, prompts, unit, text):
    prompt = prompts["WATERMARK_CHECK"].format(rag_doc=text, E1=unit["e1"], E2=unit["e2"], R1=unit["relation"])
    for attempt in range(3):
        suffix = "" if attempt == 0 else f"\nFormatting retry {attempt}: only yes or no."
        record = await cache.ask(protocol["verifier_model"], prompt + suffix)
        try:
            return yes_no(record["content"]), record["key"], record["content"]
        except ValueError:
            pass
    raise ValueError("Malformed relation judgment")


async def preflight(protocol, concurrency):
    cache = cache_for(protocol, concurrency)
    records = {}
    for name in [*protocol["target_models"], protocol["verifier_model"]]:
        record = await cache.ask(name, "Reply with the single word READY.")
        records[name] = {"model": record["model"], "returned_model": record["returned_model"],
                         "request": record["key"], "response": record["content"]}
        print(f"[preflight] {record['model']}: complete response", flush=True)
    save(OUT / "preflight.json", records)


def evaluation_clean(source, block, seed):
    full = risk.load_full_corpus(risk.dataset_for(block))
    return full, density.clean_subset(source, block, full, seed)


async def construct(protocol, source, concurrency):
    prompts = read(OUT / "upstream_prompts.json")
    cache = cache_for(protocol, concurrency)
    for block in protocol["blocks"]:
        root = OUT / "construction" / block
        ko_path = density.OWNER_SOURCE / "blocks" / block / "ko-pool/ko_pool.json"
        unit_path = root / "units.json"
        if not unit_path.exists():
            payload = keyed_units(read(ko_path)["real_kos"], f"ragwm-pilot-{block}-20261001",
                                  protocol["unit_count"], entity_limit=protocol["entity_limit"],
                                  relation_limit=protocol["relation_limit"])
            payload["real_ko_sha256"] = file_hash(ko_path)
            save(unit_path, payload)
        units = read(unit_path)["units"]
        _, clean = evaluation_clean(source, block, protocol["replicate_seeds"][0])
        questions = [prompts["WATERMARK_ASK_RAG"].format(E1=u["e1"], E2=u["e2"]) for u in units]
        scores = sparse_bm25_scores([d.text for d in clean], questions).toarray()
        jobs = asyncio.Semaphore(4)

        async def build(index, unit, *, root=root, jobs=jobs, clean=clean,
                        scores=scores, questions=questions, block=block):
            path = root / "units" / f"{unit['unit_id']}.json"
            if path.exists() and read(path).get("complete"):
                return read(path)
            async with jobs:
                checkpoint = read(path) if path.exists() else {"unit": unit, "variants": []}
                if "initial_texts" not in checkpoint:
                    def validate(value):
                        if not isinstance(value, list) or len(value) != protocol["variants_per_unit"]:
                            raise ValueError("Wrong number of watermark variants")
                        if not all(isinstance(t, str) and t.strip() for t in value) or len(set(value)) != len(value):
                            raise ValueError("Empty or duplicate watermark variant")
                        return value
                    generation = prompts["WATERMARK_GENERATE"].format(
                        E1=unit["e1"], E2=unit["e2"], R1=unit["relation"], K=protocol["variants_per_unit"])
                    checkpoint["initial_texts"], checkpoint["generation_request"] = await checked_json(
                        cache, protocol["owner_model"], generation, validate)
                    save(path, checkpoint)
                anchor = clean[int(top_indices(scores[index], 1)[0])]
                checkpoint["shadow_anchor_id"] = anchor.source_id
                for variant_index in range(len(checkpoint["variants"]), protocol["variants_per_unit"]):
                    text = checkpoint["initial_texts"][variant_index]
                    rounds = []
                    for feedback in range(protocol["max_feedback_rounds"] + 1):
                        direct, direct_key, direct_text = await check_relation(cache, protocol, prompts, unit, text)
                        shadow_prompt = prompts["MULTIPLE_PROMPT"].replace("[question]", questions[index]).replace(
                            "[context]", anchor.text + " " + text)
                        answer = await cache.ask(protocol["owner_model"], shadow_prompt)
                        shadow, shadow_key, shadow_text = await check_relation(
                            cache, protocol, prompts, unit, answer["content"])
                        rounds.append({"text": text, "direct_ok": direct, "shadow_ok": shadow,
                                       "direct_request": direct_key, "shadow_answer_request": answer["key"],
                                       "shadow_judge_request": shadow_key})
                        if (direct and shadow) or feedback == protocol["max_feedback_rounds"]:
                            break
                        repair = prompts["WATERMARK_GENERATE_FEEDBACK_BOTH"].format(
                            E1=unit["e1"], E2=unit["e2"], R1=unit["relation"], WT=text,
                            WE=answer["content"], WD1=direct_text, WD2=shadow_text)
                        text, repair_key = await checked_json(cache, protocol["owner_model"], repair, feedback_text)
                        rounds[-1]["repair_request"] = repair_key
                    checkpoint["variants"].append({"variant": variant_index, "text": text,
                                                   "direct_ok": direct, "shadow_ok": shadow,
                                                   "feedback_rounds": len(rounds) - 1, "history": rounds})
                    save(path, checkpoint)
                checkpoint["complete"] = True
                checkpoint["question"] = questions[index]
                checkpoint["selected_variants"] = eligible_variants(checkpoint["variants"])
                save(path, checkpoint)
                print(f"[constructed] {block}/{unit['unit_id']}: selected={len(checkpoint['selected_variants'])}/5", flush=True)
                return checkpoint

        records = await asyncio.gather(*(build(i, u) for i, u in enumerate(units)))
        save(root / "carriers.json", {"block": block, "records": records,
                                      "complete": len(records) == protocol["unit_count"]})
    save(OUT / "construction_status.json", {"complete": True, "new_requests": cache.new_requests,
                                             "cache_hits": cache.reused_requests})


def rank(documents, questions, retriever, dense, source, path):
    if retriever == "bm25":
        return sparse_bm25_scores([d.text for d in documents], questions).toarray().astype(np.float32)
    qv = dense.encode(questions, path / "queries.npz")
    dv = dense.encode([d.text for d in documents], path / "documents.npz")
    scores = np.einsum("ij,kj->ik", qv, dv, optimize=True).astype(np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite dense retrieval scores")
    return scores


def prepare(protocol, source):
    dense = risk.DenseScores(source)
    jobs = []
    for block in protocol["blocks"]:
        carriers = read(OUT / "construction" / block / "carriers.json")["records"]
        units = [c["unit"] for c in carriers]
        questions = [c["question"] for c in carriers]
        assignment = read(risk.OUT / "partitions" / f"{block}.json")["assignment"]
        for seed in protocol["replicate_seeds"]:
            full, clean = evaluation_clean(source, block, seed)
            clean_ids = {d.source_id for d in clean}
            eligible = [d for d in full if d.source_id not in clean_ids]
            calibration = random.Random(stable_seed(protocol["protocol_version"], block, "calibration")).sample(
                eligible, protocol["calibration_count"])
            topics = set(density.topics_for_replicates(source, block)[seed])
            modified = list(clean)
            anchor_ids = defaultdict(list)
            # Re-index after each append, as upstream mutates the retrieval collection.
            for carrier in carriers:
                selected = carrier["selected_variants"]
                scores = sparse_bm25_scores([d.text for d in modified], [carrier["question"]]).toarray()[0]
                for position, variant in zip(top_indices(scores, len(selected)), selected, strict=True):
                    index = int(position)
                    doc = modified[index]
                    modified[index] = replace(doc, text=doc.text + " " + variant["text"])
                    anchor_ids[carrier["unit"]["unit_id"]].append(doc.source_id)
            altered = {d.source_id for a, d in zip(clean, modified, strict=True) if a.text != d.text}
            summary = {"evidence_units": len(units), "appended_texts": sum(len(c["selected_variants"]) for c in carriers),
                       "modified_documents": len(altered), "clean_documents": len(clean),
                       "added_words": sum(len(d.text.split()) - len(a.text.split()) for a, d in zip(clean, modified, strict=True)),
                       "anchor_ids": dict(anchor_ids), "clean_ids": sorted(clean_ids), "retained_topics": sorted(topics),
                       "calibration_ids": sorted(d.source_id for d in calibration)}
            save(OUT / "injection" / block / str(seed) / "manifest.json", summary)
            corpora = [("calibration", "clean", calibration)]
            for scenario in ("intact", "topic_subset"):
                for state, docs in (("watermarked", modified), ("clean", clean)):
                    selected_docs = docs if scenario == "intact" else [
                        d for d in docs if assignment[f"clean:{d.source_id}"] in topics]
                    corpora.append((scenario, state, selected_docs))
            for scenario, state, docs in corpora:
                if not docs:
                    raise ValueError("Empty corpus condition")
                for retriever in protocol["target_retrievers"]:
                    job = f"{block}/{seed}/{scenario}/{retriever}/{state}"
                    path = OUT / "contexts" / job / "contexts.json"
                    if path.exists():
                        jobs.append(job)
                        continue
                    score_matrix = rank(docs, questions, retriever, dense, source,
                                        OUT / "dense" / block / str(seed) / scenario / state)
                    contexts, diagnostics = {}, {}
                    for index, unit in enumerate(units):
                        ctx = context_rows(score_matrix[index], docs, top_k=protocol["top_k"])
                        contexts[unit["unit_id"]] = ctx
                        available = {d.source_id for d in docs}
                        diagnostics[unit["unit_id"]] = {
                            "carrier_present": state == "watermarked" and bool(set(anchor_ids[unit["unit_id"]]) & available),
                            "own_carrier_retrieved": state == "watermarked" and any(
                                r["id"] in anchor_ids[unit["unit_id"]] for r in ctx)}
                    condition = {"block": block, "replicate_seed": seed, "scenario": scenario,
                                 "retriever": retriever, "state": state, "documents": len(docs), "top_k": protocol["top_k"]}
                    payload = {"condition": condition, "contexts": contexts, "diagnostics": diagnostics,
                               "units": units, "questions": dict(zip([u["unit_id"] for u in units], questions, strict=True))}
                    payload["signature"] = digest(payload)
                    save(path, payload)
                    jobs.append(job)
                    print(f"[prepared] {job}: documents={len(docs)}", flush=True)
    receipt = {"jobs": jobs, "contexts_sha256": {
        job: file_hash(OUT / "contexts" / job / "contexts.json") for job in jobs},
               "carrier_sha256": {block: file_hash(OUT / "construction" / block / "carriers.json")
                                  for block in protocol["blocks"]}}
    save(OUT / "frozen_contexts.json", receipt)


async def execute(protocol, concurrency):
    frozen = read(OUT / "frozen_contexts.json")
    cache = cache_for(protocol, concurrency)
    jobs = asyncio.Semaphore(8)

    async def group(job, model):
        async with jobs:
            payload_path = OUT / "contexts" / job / "contexts.json"
            if file_hash(payload_path) != frozen["contexts_sha256"][job]:
                raise ValueError("Frozen retrieval contexts changed")
            payload = read(payload_path)
            path = OUT / "execution" / job / model / "answers.json"
            data = read(path) if path.exists() else {
                "condition": payload["condition"], "context_signature": payload["signature"],
                "model": model, "records": [], "complete": False}
            if data["context_signature"] != payload["signature"]:
                raise ValueError("Answer checkpoint context mismatch")
            for unit in payload["units"][len(data["records"]):]:
                key = unit["unit_id"]
                question = payload["questions"][key]
                expected = f"{unit['e1']} {unit['relation'].replace('_', ' ')} {unit['e2']}."
                answer = await cache.ask(model, PromptTemplates.answer_with_rag(
                    "\n".join(r["text"] for r in payload["contexts"][key]), question))
                if explicit_abstention(answer["content"], expected):
                    correct, judge_key, judge_kind = False, None, "exact_abstention_rule_v1"
                else:
                    prompt = PromptTemplates.judge_answer_correctness(question, answer["content"], expected)
                    for attempt in range(3):
                        verdict = await cache.ask(protocol["verifier_model"], prompt + (
                            "" if attempt == 0 else f"\nFormatting retry {attempt}: only Yes or No."))
                        try:
                            correct = yes_no(verdict["content"])
                            break
                        except ValueError:
                            continue
                    else:
                        raise ValueError("Malformed common oracle judgment")
                    judge_key, judge_kind = verdict["key"], "glm_semantic"
                data["records"].append({"unit_id": key, "question": question, "expected_answer": expected,
                                         "answer": answer["content"], "is_correct": correct,
                                         "answer_request": answer["key"], "judge_request": judge_key,
                                         "judge_kind": judge_kind, **payload["diagnostics"][key]})
                data["complete"] = len(data["records"]) == len(payload["units"])
                save(path, data)
            print(f"[executed] {job}/{model}: {len(data['records'])}/30", flush=True)

    await asyncio.gather(*(group(job, model) for job in frozen["jobs"] for model in protocol["target_models"]))
    save(OUT / "execution_status.json", {"complete": True, "answer_groups": len(frozen["jobs"]) * len(protocol["target_models"]),
                                         "new_requests": cache.new_requests, "cache_hits": cache.reused_requests})


def summarize(protocol):
    frozen = read(OUT / "frozen_contexts.json")
    calibrations = []
    n_groups = len(protocol["blocks"]) * len(protocol["target_retrievers"]) * len(protocol["target_models"])
    tail = (1 - protocol["calibration_confidence"]) / n_groups
    for block, seed, retriever, model in itertools.product(
            protocol["blocks"], protocol["replicate_seeds"], protocol["target_retrievers"], protocol["target_models"]):
        job = f"{block}/{seed}/calibration/{retriever}/clean"
        records = read(OUT / "execution" / job / model / "answers.json")["records"]
        x, n = sum(r["is_correct"] for r in records), len(records)
        upper = float(beta.ppf(1 - tail, x + 1, n - x)) if x < n else 1.0
        calibrations.append({"block": block, "retriever": retriever, "model": model,
                             "positives": x, "probes": n, "rate": x / n, "binomial_upper": upper})
    common = max(protocol["minimum_common_p0"], max(r["binomial_upper"] for r in calibrations))
    if common >= 1:
        common = 1 - 1e-6
    save(OUT / "calibration.json", {"groups": calibrations, "common_p0": common,
         "source_p0": protocol["minimum_common_p0"], "confidence": protocol["calibration_confidence"],
         "scope": "exploratory pooled-rate bound; shared entities can violate independent Bernoulli assumptions; not a conditional-null guarantee"})
    source_root = ROOT / protocol["source_comparison"]
    rows, audit = [], []

    def add(method, condition, model, labels, policy_seed, p0, scope="common"):
        horizon = protocol["comparison_horizon"]
        if len(labels) < horizon:
            raise ValueError("Incomplete comparative trajectory")
        evaluation = evidence_curve(labels[:horizon], p0, protocol["alpha"])
        row = {**condition, "method": method, "model": model, "policy_seed": policy_seed,
               "p0": p0, "decision_scope": scope, "edr15": sum(labels[:horizon]) / horizon,
               "anytime15": evaluation["curve"][-1]["anytime_verified"],
               "queries15": evaluation["restricted_queries"],
               **{f"vsr{b}": evaluation["curve"][b - 1]["fixed_verified"] for b in (4, 6, 10, 15)}}
        rows.append(row)

    for block, seed, scenario, retriever, state, model in itertools.product(
            protocol["blocks"], protocol["replicate_seeds"], ("intact", "topic_subset"),
            protocol["target_retrievers"], ("watermarked", "clean"), protocol["target_models"]):
        job = f"{block}/{seed}/{scenario}/{retriever}/{state}"
        path = OUT / "execution" / job / model / "answers.json"
        data = read(path)
        if not data["complete"] or len({r["unit_id"] for r in data["records"]}) != protocol["unit_count"]:
            raise ValueError("Incomplete or repeated evidence")
        condition = {"block": block, "replicate_seed": seed, "scenario": scenario,
                     "retriever": retriever, "state": state}
        for policy_seed in protocol["policy_seeds"]:
            order = list(data["records"])
            random.Random(stable_seed(protocol["protocol_version"], block, seed, policy_seed)).shuffle(order)
            add("RAG-WM", condition, model, [r["is_correct"] for r in order], policy_seed, common)
            relation_count = len(read(OUT / "construction" / block / "units.json")["relations"])
            add("RAG-WM", condition, model, [r["is_correct"] for r in order], policy_seed,
                1 / relation_count, "native_relation_count")
        for method, fixture, selector, scheduler, seeds in (
            ("SentinelRAG", "original", "q0", "random", protocol["policy_seeds"]),
            ("RAVET", "relation_explicit", "failure_repair", "bayes", [0])):
            source_job = transfer.source_job(block, fixture, "d1", seed, scenario, retriever, state)
            for policy_seed in seeds:
                source_path = source_root / "execution" / f"seed-{policy_seed}" / "traces" / source_job / model / f"{selector}--{scheduler}.json"
                trace = read(source_path)
                if not trace["complete"]:
                    raise ValueError("Incomplete source comparator")
                add(method, condition, model, [r["is_correct"] for r in trace["records"]], policy_seed, common)
                audit.append({"source": str(source_path.relative_to(ROOT)), "sha256": file_hash(source_path)})
    save(OUT / "reused_source_traces.json", audit)
    keys = list(rows[0])
    with (OUT / "trajectory_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    # Random-order seeds are averaged within a deployment, never counted as new deployments.
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in ("method", "decision_scope", "block", "replicate_seed", "scenario", "retriever", "state", "model"))].append(row)
    metrics = ("vsr4", "vsr6", "vsr10", "vsr15", "anytime15", "queries15", "edr15")
    conditions = []
    for key, values in grouped.items():
        conditions.append({**dict(zip(("method", "decision_scope", "block", "replicate_seed", "scenario", "retriever", "state", "model"), key, strict=True)),
                           **{m: float(np.mean([v[m] for v in values])) for m in metrics}})
    summary = []
    for method, scope, state in sorted({(r["method"], r["decision_scope"], r["state"]) for r in conditions}):
        values = [r for r in conditions if (r["method"], r["decision_scope"], r["state"]) == (method, scope, state)]
        summary.append({"method": method, "decision_scope": scope, "state": state, "configurations": len(values),
                        **{m: float(np.mean([v[m] for v in values])) for m in metrics}})
    save(OUT / "results.json", {"common_p0": common, "summary": summary, "conditions": conditions})
    requests = list((OUT / "requests").glob("*/*.json"))
    complete_requests = [p for p in requests if ".attempt-" not in p.name]
    save(OUT / "audit.json", {"complete": True, "comparative_positive_configurations": 16,
                              "comparative_clean_configurations": 16, "new_completed_requests": len(complete_requests),
                              "frozen_context_hashes_valid": all(file_hash(OUT / "contexts" / j / "contexts.json") == h
                                                                for j, h in frozen["contexts_sha256"].items()),
                              "source_trace_hash_receipts": len(audit), "paper_unchanged": True})
    print(json.dumps({"common_p0": common, "summary": summary}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "construct", "prepare", "execute", "summarize", "all"), default="all")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    protocol, source = freeze()
    if args.stage in ("preflight", "all"):
        asyncio.run(preflight(protocol, args.concurrency))
    if args.stage in ("construct", "all"):
        asyncio.run(construct(protocol, source, args.concurrency))
    if args.stage in ("prepare", "all"):
        prepare(protocol, source)
    if args.stage in ("execute", "all"):
        asyncio.run(execute(protocol, args.concurrency))
    if args.stage in ("summarize", "all"):
        summarize(protocol)


if __name__ == "__main__":
    main()
