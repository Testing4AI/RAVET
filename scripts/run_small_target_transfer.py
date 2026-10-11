#!/usr/bin/env python3
"""Transfer frozen ownership tests to the existing Qwen target endpoint."""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import run_density_curve as density
from run_orthogonal_policy_diagnostic import model_settings, run_trace
from run_partition_confirmation import RequestCache, digest, file_hash, read, save

from sentinelrag.testing.orthogonal_policy import next_selected_probe
from sentinelrag.testing.prioritization import evaluate_ordering

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/small_target_transfer_20261001.json"
OUT = ROOT / "output/small-target-transfer-20261001-v4"
HISTORY_ROOTS = (ROOT / "output/small-target-transfer-20261001-v3",
                 ROOT / "output/small-target-transfer-20261001-v2")


def final_answer(content):
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
    if re.search(r"</?think>", text, flags=re.IGNORECASE) or not text:
        raise ValueError("Missing final answer or incomplete reasoning block")
    return text


class FinalAnswerRequestCache(RequestCache):
    async def ask(self, model_name, prompt):
        row = await super().ask(model_name, prompt)
        if model_name in self.protocol["target_models"]:
            return {**row, "content": final_answer(row["content"])}
        return row


def condition_key(row):
    return tuple(row[k] for k in (
        "block", "density", "replicate_seed", "scenario", "retriever", "state"
    ))


def source_job(block, fixture, label, seed, scenario, retriever, state):
    return "/".join(map(str, (block, fixture, label, seed, scenario, retriever, state)))


def policy_specs(protocol, fixture):
    if fixture == "original":
        return [("q0", "random", seed) for seed in protocol["policy_seed_ids"]]
    if fixture == "relation_explicit":
        return [("failure_repair", protocol["selected_scheduler"], 0)]
    raise ValueError(f"Unknown fixture: {fixture}")


def clean_payload(watermarked, clean):
    for key in ("block", "fixture", "replicate_seed", "scenario", "retriever"):
        if watermarked["condition"][key] != clean["condition"][key]:
            raise ValueError(f"Unmatched clean context: {key}")
    for key in ("retained_topics", "retained_clean", "clean_documents"):
        if watermarked["condition"][key] != clean["condition"][key]:
            raise ValueError(f"Clean corpus mismatch: {key}")
    ids = set(watermarked["contexts"])
    if not ids <= clean["contexts"].keys():
        raise ValueError("Missing clean probe contexts")
    condition = {**watermarked["condition"], "state": "clean", "retained_sentinels": 0}
    payload = {
        "condition": condition,
        "contexts": {key: clean["contexts"][key] for key in sorted(ids)},
        "diagnostics": {key: clean["diagnostics"][key] for key in sorted(ids)},
        "provenance": {
            "derivation": "unchanged clean retrieval restricted to the frozen d1 suite",
            "source_clean_signature": clean["signature"],
            "source_suite_signature": watermarked["signature"],
        },
    }
    payload["signature"] = digest(payload)
    return payload


def freeze():
    protocol = read(PROTOCOL)
    source = ROOT / protocol["source_density"]
    original = read(source / "protocol_lock.json")["protocol"]
    source_frozen = read(source / "frozen_protocol.json")
    for field in ("confirmation_blocks", "target_retrievers", "replicate_seeds",
                  "fixtures", "policy_seed_ids", "verifier_model", "top_k",
                  "max_queries", "alpha", "p0", "prior_retention"):
        if protocol[field] != original[field]:
            raise ValueError(f"Changed frozen source setting: {field}")
    if read(source / "selected_scheduler.json")["selected_scheduler"] != protocol["selected_scheduler"]:
        raise ValueError("Scheduler differs from the source selection")
    inputs = {str(PROTOCOL.relative_to(ROOT)): file_hash(PROTOCOL)}
    for name in ("protocol_lock.json", "frozen_protocol.json", "selected_scheduler.json",
                 "confirmation/condition_means.json"):
        path = source / name
        inputs[str(path.relative_to(ROOT))] = file_hash(path)
    owner_lock = density.OWNER_SOURCE / "frozen_protocol.json"
    inputs[str(owner_lock.relative_to(ROOT))] = file_hash(owner_lock)
    for block in protocol["confirmation_blocks"]:
        root = density.OWNER_SOURCE / "blocks" / block
        for suffix in ("partitions.json", "injection/injection_result.json",
                       "candidates/relation_explicit/injection_result.json",
                       "owner/original/scidocs/contriever/factual/retrieval_cache_metadata.json"):
            path = root / suffix
            if path.exists():
                inputs[str(path.relative_to(ROOT))] = file_hash(path)
        for fixture in protocol["fixtures"]:
            dataset = block.split("-block-")[0]
            cache = root / "owner" / fixture / dataset / "contriever/factual"
            for path in sorted(cache.glob("*")):
                if path.is_file():
                    inputs[str(path.relative_to(ROOT))] = file_hash(path)
    jobs, context_hashes = [], {}
    combinations = itertools.product(protocol["confirmation_blocks"], protocol["fixtures"],
                                     protocol["replicate_seeds"], protocol["scenarios"],
                                     protocol["target_retrievers"])
    for block, fixture, seed, scenario, retriever in combinations:
        water_job = source_job(block, fixture, protocol["density"], seed, scenario, retriever, "watermarked")
        clean_job = source_job(block, fixture, "d2", seed, scenario, retriever, "clean")
        wp, cp = source / "contexts" / water_job / "contexts.json", source / "contexts" / clean_job / "contexts.json"
        for path in (wp, cp):
            inputs[str(path.relative_to(ROOT))] = file_hash(path)
            relative = str(path.relative_to(source / "contexts"))
            if source_frozen["context_sha256"][relative.removesuffix("/contexts.json")] != file_hash(path):
                raise ValueError("Source context is no longer frozen")
        watermarked, clean = read(wp), read(cp)
        for payload in (watermarked, clean_payload(watermarked, clean)):
            condition = payload["condition"]
            job = source_job(block, fixture, protocol["density"], seed, scenario, retriever, condition["state"])
            path = OUT / "contexts" / job / "contexts.json"
            if path.exists() and read(path) != payload:
                raise ValueError("Supplement context changed")
            if not path.exists():
                save(path, payload)
            jobs.append(job)
            context_hashes[job] = file_hash(path)
    code = [Path(__file__), ROOT / "scripts/run_density_curve.py",
            ROOT / "scripts/run_orthogonal_policy_diagnostic.py",
            ROOT / "scripts/run_partition_confirmation.py",
            ROOT / "scripts/run_risk_low_density.py",
            ROOT / "src/sentinelrag/testing/orthogonal_policy.py",
            ROOT / "src/sentinelrag/testing/prioritization.py",
            ROOT / "src/sentinelrag/prompts/templates.py",
            ROOT / "src/sentinelrag/utils/stats.py"]
    frozen = {
        "protocol": protocol,
        "source_protocol": original,
        "input_sha256": inputs,
        "context_sha256": context_hashes,
        "code_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in code},
        "model_settings": {m: model_settings(m) for m in protocol["target_models"] + [protocol["verifier_model"]]},
        "paper_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in density.PAPER_FILES},
        "jobs": sorted(jobs),
    }
    path = OUT / "frozen_protocol.json"
    if path.exists() and read(path) != frozen:
        raise ValueError("Supplement freeze mismatch: do not overwrite prior evidence")
    if not path.exists():
        save(path, frozen)
    return frozen


async def preflight(frozen):
    protocol = frozen["protocol"]
    cache = RequestCache(protocol, 2, output_root=OUT)
    rows = []
    for name, (client, preset) in cache.clients.items():
        try:
            catalog = await client.async_client.with_options(max_retries=0).models.list(timeout=30)
            present = preset.model in {entry.id for entry in catalog.data}
            response = await client.async_client.with_options(max_retries=0).chat.completions.create(
                model=preset.model,
                messages=[{"role": "user", "content": "Reply with only the word OK."}],
                **preset.llm_arg, timeout=60, max_tokens=512,
            )
            content = response.choices[0].message.content
            if not content or response.choices[0].finish_reason == "length":
                raise ValueError("Incomplete invocation preflight")
            rows.append({"preset": name, "requested_model": preset.model,
                         "catalog_present": present, "returned_model": response.model,
                         "finish_reason": response.choices[0].finish_reason,
                         "content": content, "status": "pass"})
        except Exception as error:
            save(OUT / "preflight.json", {"status": "failed", "models": rows,
                 "failed_model": name, "error_type": type(error).__name__,
                 "status_code": getattr(error, "status_code", None)})
            raise RuntimeError(f"Preflight failed: {name} {type(error).__name__}") from None
        finally:
            await client.async_client.close()
            client.client.close()
    receipt = {"status": "pass", "models": rows}
    save(OUT / "preflight.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def trace_path(job, model, selector, scheduler, seed):
    return OUT / "execution" / f"seed-{seed}" / "traces" / job / model / f"{selector}--{scheduler}.json"


def inventory(frozen):
    protocol = frozen["protocol"]
    complete, records, expected = 0, 0, 0
    for job in frozen["jobs"]:
        condition = read(OUT / "contexts" / job / "contexts.json")["condition"]
        for model in protocol["target_models"]:
            for selector, scheduler, seed in policy_specs(protocol, condition["fixture"]):
                expected += 1
                path = trace_path(job, model, selector, scheduler, seed)
                if path.exists():
                    payload = read(path)
                    complete += bool(payload["complete"])
                    records += len(payload["records"])
    return {"status": "complete" if complete == expected else "incomplete_resumable",
            "completed_traces": complete, "expected_traces": expected,
            "logical_queries": records}


async def execute(frozen, concurrency):
    protocol = frozen["protocol"]
    if read(OUT / "preflight.json")["status"] != "pass":
        raise RuntimeError("Passing preflight required")
    requests = FinalAnswerRequestCache(protocol, concurrency, output_root=OUT,
                            readonly_roots=(*HISTORY_ROOTS, ROOT / protocol["source_density"], density.OWNER_SOURCE))
    requests.model_semaphores = {name: asyncio.Semaphore(protocol["transport_model_concurrency"][name])
                                for name in requests.clients}
    limiters = {model: asyncio.Semaphore(protocol["transport_group_concurrency_per_target"])
                for model in protocol["target_models"]}
    source_protocol = frozen["source_protocol"]
    prepared = {}
    for job in frozen["jobs"]:
        condition = read(OUT / "contexts" / job / "contexts.json")["condition"]
        for selector, _, _ in policy_specs(protocol, condition["fixture"]):
            key = (condition["block"], condition["fixture"], condition["replicate_seed"], selector)
            if key not in prepared:
                prepared[key] = density.prepared_set(source_protocol, condition, selector)

    async def group(job, model):
        async with limiters[model]:
            condition = read(OUT / "contexts" / job / "contexts.json")["condition"]
            for selector, scheduler, seed in policy_specs(protocol, condition["fixture"]):
                key = (condition["block"], condition["fixture"], condition["replicate_seed"], selector)
                await run_trace(protocol, requests, job, model, selector, scheduler,
                                *prepared[key], output_root=OUT / "execution" / f"seed-{seed}",
                                context_root=OUT / "contexts",
                                seed_override=density.policy_seed(source_protocol, condition, selector, scheduler, seed))
            state = {**inventory(frozen), "status": "running",
                     "new_requests_this_run": requests.new_requests,
                     "cache_hits_this_run": requests.reused_requests}
            save(OUT / "execution_status.json", state)
            print(f"[small target] {state['completed_traces']}/{state['expected_traces']} traces; {job}", flush=True)

    save(OUT / "execution_status.json", {**inventory(frozen), "status": "running"})
    try:
        await asyncio.gather(*(group(job, model) for job in frozen["jobs"] for model in protocol["target_models"]))
    finally:
        save(OUT / "execution_status.json", {**inventory(frozen),
             "new_requests_this_run": requests.new_requests,
             "cache_hits_this_run": requests.reused_requests})
        for client, _ in requests.clients.values():
            await client.async_client.close()
            client.client.close()


def collect(frozen):
    protocol = frozen["protocol"]
    rows, traces = [], []
    grouped = defaultdict(list)
    for job in frozen["jobs"]:
        condition = read(OUT / "contexts" / job / "contexts.json")["condition"]
        for model in protocol["target_models"]:
            for selector, scheduler, seed in policy_specs(protocol, condition["fixture"]):
                payload = read(trace_path(job, model, selector, scheduler, seed))
                if not payload["complete"]:
                    raise RuntimeError("Incomplete trajectory cannot enter summary")
                row = {**condition, "model": model, "selector": selector,
                       "scheduler": scheduler, "policy_seed_id": seed,
                       **density.endpoint(payload["evaluation"], protocol["max_queries"])}
                rows.append(row)
                grouped[(condition_key(row), model, condition["fixture"])].append(row)
                traces.append(payload)
    metrics = [key for key in rows[0] if key.startswith(("vsr_", "anytime_", "restricted_"))]
    means = [{**items[0], "policy_seed_id": None, "policy_replicates": len(items),
              **{metric: float(np.mean([r[metric] for r in items])) for metric in metrics}}
             for items in grouped.values()]
    save(OUT / "raw_metrics.json", rows)
    save(OUT / "condition_means.json", means)
    return means, traces


def summarize(frozen):
    protocol = frozen["protocol"]
    means, traces = collect(frozen)
    baseline = {(r["model"], condition_key(r)): r for r in means if r["state"] == "watermarked" and r["fixture"] == "original"}
    method = {(r["model"], condition_key(r)): r for r in means if r["state"] == "watermarked" and r["fixture"] == "relation_explicit"}
    if baseline.keys() != method.keys() or len(baseline) != protocol["expected_watermarked_pairs"] * len(protocol["target_models"]):
        raise ValueError("Watermarked pairing/count mismatch")
    summaries = [density.paired_group_summary([(baseline[k], method[k]) for k in baseline], protocol, "overall")]
    for field in ("model", "scenario", "dataset", "retriever"):
        for value in sorted({r[field] for r in baseline.values()}):
            pairs = [(baseline[k], method[k]) for k in baseline if baseline[k][field] == value]
            summaries.append(density.paired_group_summary(pairs, protocol, f"{field}/{value}"))
    clean = [r for r in means if r["state"] == "clean"]
    diagnostic = defaultdict(dict)
    diagnostic_rows = defaultdict(list)
    for trace in traces:
        if trace["condition"]["state"] == "watermarked":
            # Deduplicate policy-seed overlap within each deployment/fixture, not across deployments.
            key = (trace["model"], condition_key(trace["condition"]), trace["condition"]["fixture"])
            diagnostic[key].update({r["probe_id"]: r for r in trace["records"]})
    for (model, _, fixture), items in diagnostic.items():
        diagnostic_rows[f"{model}/{fixture}"].extend(items.values())
    diagnostics = {}
    for fixture, items in diagnostic_rows.items():
        retrieved = [r for r in items if r["retrieved_sentinel"]]
        diagnostics[fixture] = {
            "selected_unique_probe_deployment_pairs": len(items),
            "not_retrieved": sum(not r["retrieved_sentinel"] for r in items),
            "retrieved_but_answer_failed": sum(not r["is_correct"] for r in retrieved),
            "retrieved_and_answer_correct": sum(r["is_correct"] for r in retrieved),
            "answer_correct_without_own_sentinel": sum(r["is_correct"] and not r["retrieved_sentinel"] for r in items),
            "semantic_success_given_retrieved": float(np.mean([r["is_correct"] for r in retrieved])) if retrieved else None,
        }
    older = read(ROOT / protocol["source_density"] / "confirmation/condition_means.json")
    selected_conditions = {condition_key(r) for r in baseline.values()}
    cross_model = []
    for model in sorted({r["model"] for r in older}):
        old = [r for r in older if r["model"] == model and condition_key(r) in selected_conditions]
        left = {condition_key(r): r for r in old if r["fixture"] == "original"}
        right = {condition_key(r): r for r in old if r["fixture"] == "relation_explicit"}
        if left.keys() != selected_conditions or right.keys() != selected_conditions:
            raise ValueError("Historical target comparison not on identical conditions")
        cross_model.append({"model": model, **density.paired_group_summary([(left[k], right[k]) for k in left], protocol, "same-24-conditions")})
    for model in protocol["target_models"]:
        row = next(r for r in summaries if r["group"] == f"model/{model}")
        cross_model.append({"model": model, **row})
    result = {"status": "complete", "protocol_version": protocol["protocol_version"],
              "inventory": inventory(frozen), "summaries": summaries,
              "clean_controls": {"paired_deployments": len(clean) // 2, "policy_conditions": len(clean),
                    "fixed_positive_conditions": sum(bool(r["vsr_6"]) for r in clean),
                    "anytime_positive_conditions": sum(bool(r["anytime_15"]) for r in clean)},
              "clean_controls_by_model": {model: {
                    "paired_deployments": len([r for r in clean if r["model"] == model]) // 2,
                    "policy_conditions": len([r for r in clean if r["model"] == model]),
                    "fixed_positive_conditions": sum(bool(r["vsr_6"]) for r in clean if r["model"] == model),
                    "anytime_positive_conditions": sum(bool(r["anytime_15"]) for r in clean if r["model"] == model)}
                    for model in protocol["target_models"]},
              "diagnostics": diagnostics, "same_condition_target_comparison": cross_model}
    save(OUT / "summary.json", result)
    lines = ["# Qwen target transfer results", "", "Post-hoc target transfer; original primary aggregate unchanged.", "",
             "| Group | Conditions | Wins/ties/losses | SentinelRAG VSR@6 | RAVET VSR@6 | SentinelRAG Anytime@15 | RAVET Anytime@15 | SentinelRAG calls | RAVET calls |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['group']} | {row['conditions']} | {row['wins']}/{row['ties']}/{row['losses']} | "
                     f"{row['vsr_6']['baseline']:.1%} | {row['vsr_6']['method']:.1%} | "
                     f"{row['anytime_15']['baseline']:.1%} | {row['anytime_15']['method']:.1%} | "
                     f"{row['restricted_15']['baseline']:.2f} | {row['restricted_15']['method']:.2f} |")
    lines.extend(["", "Clean controls: " + json.dumps(result["clean_controls"]), "",
                  "Diagnostics are descriptive for the policy-selected union, not an unbiased all-probe population.",
                  "Paired bootstrap intervals resample the 24 deployment conditions; conditions share source blocks.",
                  "Qwen3.5-35B-A3B is a 35B-total/3B-active MoE. Smaller active size is not evidence of weaker capability."])
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fields = ["model", "group", "conditions", "wins", "ties", "losses"] + [f"{m}_{side}" for m in ("vsr_6", "anytime_15", "restricted_15") for side in ("baseline", "method", "method_minus_baseline")]
    with (OUT / "same_condition_target_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in cross_model:
            writer.writerow({**{k: row[k] for k in fields[:6]},
                **{f"{m}_{side}": row[m][side] for m in ("vsr_6", "anytime_15", "restricted_15") for side in ("baseline", "method", "method_minus_baseline")}})
    print(json.dumps(summaries[0], ensure_ascii=False), flush=True)
    return result


def audit(frozen):
    protocol, source_protocol = frozen["protocol"], frozen["source_protocol"]
    failures, checked, request_records = [], 0, {}
    for group in ("input_sha256", "code_sha256", "paper_sha256"):
        for name, expected in frozen[group].items():
            if file_hash(ROOT / name) != expected:
                failures.append(f"hash:{group}:{name}")
    for job in frozen["jobs"]:
        context_path = OUT / "contexts" / job / "contexts.json"
        if file_hash(context_path) != frozen["context_sha256"][job]:
            failures.append(f"context-hash:{job}")
        context = read(context_path)
        condition = context["condition"]
        for model in protocol["target_models"]:
            for selector, scheduler, seed_id in policy_specs(protocol, condition["fixture"]):
                path = trace_path(job, model, selector, scheduler, seed_id)
                if not path.exists():
                    failures.append(f"missing:{job}:{seed_id}")
                    continue
                payload = read(path)
                records = payload["records"]
                if not payload["complete"] or len(records) != protocol["max_queries"]:
                    failures.append(f"incomplete:{job}:{seed_id}")
                    continue
                if len({r["canary_id"] for r in records}) != len(records):
                    failures.append(f"repeated-carrier:{job}:{seed_id}")
                selected, clusters = density.prepared_set(source_protocol, condition, selector)
                profiles = {p.profile.candidate.probe_id: p.profile for p in selected}
                seed = density.policy_seed(source_protocol, condition, selector, scheduler, seed_id)
                for index, row in enumerate(records):
                    expected = next_selected_probe(selected, clusters, scheduler, records[:index], seed=seed,
                                                   p0=protocol["p0"], prior=protocol["prior_retention"])
                    if expected.profile.candidate.probe_id != row["probe_id"]:
                        failures.append(f"prefix:{job}:{seed_id}:{index}")
                    for kind in ("answer_request", "judge_request"):
                        key = row[kind]
                        if key is None:
                            if kind != "judge_request" or row["judge_kind"] != "exact_abstention_rule_v1":
                                failures.append(f"missing-key:{job}:{kind}")
                            continue
                        if key not in request_records:
                            for root in (OUT, *HISTORY_ROOTS, ROOT / protocol["source_density"], density.OWNER_SOURCE):
                                request_path = root / "requests" / key[:2] / f"{key}.json"
                                if request_path.exists():
                                    request_records[key] = read(request_path)
                                    break
                            else:
                                failures.append(f"missing-request:{key}")
                                continue
                        record = request_records[key]
                        expected_model = model_settings(model if kind == "answer_request" else protocol["verifier_model"])["model"]
                        if record["model"] != expected_model or record["finish_reason"] == "length" or not record["content"]:
                            failures.append(f"request-model-or-completion:{key}")
                        if kind == "answer_request" and final_answer(record["content"]) != row["answer"]:
                            failures.append(f"final-answer:{key}")
                evaluation = evaluate_ordering([profiles[r["probe_id"]] for r in records],
                    {r["probe_id"]: r["is_correct"] for r in records}, p0=protocol["p0"], alpha=protocol["alpha"])
                if evaluation != payload["evaluation"]:
                    failures.append(f"evaluation:{job}:{seed_id}")
                checked += 1
    result = {"status": "pass" if not failures and checked == protocol["expected_traces"] else "fail",
              "checked_traces": checked, "failures": failures,
              "unique_referenced_requests": len(request_records),
              "model_identities": sorted({(r["model"], r.get("returned_model")) for r in request_records.values()})}
    save(OUT / "audit.json", result)
    if result["status"] != "pass":
        raise RuntimeError(f"Audit failed with {len(failures)} errors")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


async def main(action, concurrency):
    frozen = freeze()
    if action == "freeze":
        print(json.dumps({"jobs": len(frozen["jobs"]), **inventory(frozen)}))
    elif action == "preflight":
        await preflight(frozen)
    elif action == "run":
        await execute(frozen, concurrency)
    elif action == "summary":
        summarize(frozen)
    elif action == "audit":
        audit(frozen)
    elif action == "all":
        await preflight(frozen)
        await execute(frozen, concurrency)
        summarize(frozen)
        audit(frozen)
    else:
        print(json.dumps(inventory(frozen)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "preflight", "run", "summary", "audit", "status", "all"))
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(main(args.action, args.concurrency))
