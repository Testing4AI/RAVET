#!/usr/bin/env python3
"""Append Top-1 to an audited deployment study without changing its evidence."""

import argparse
import asyncio
import csv
import itertools
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_retrieval_deployment_changes as previous
import summarize_retrieval_deployment_changes as exports
from audit_retrieval_deployment_rankings import checked_vectors
from audit_retrieval_deployment_requests_v2 import validate_request
from run_partition_confirmation import ROOT, digest, file_hash, read, save
from sentinelrag.prompts.templates import PromptTemplates
from sentinelrag.testing.corpus_retrieval import IndexedMigrationDocument, context_rows
from sentinelrag.testing.matrix import sparse_bm25_scores
from sentinelrag.testing.migration_confirmation import explicit_abstention
from sentinelrag.testing.ragwm import yes_no
from sentinelrag.testing.retrieval_deployment import context_diagnostics, rechunk_documents


PROTOCOL = ROOT / "experiments/retrieval_deployment_completion_20261003.json"
OUT = ROOT / "output/retrieval-deployment-completion-20261003-v2"
PRIOR = ROOT / "output/retrieval-deployment-changes-20261003-v1"
BASE_REPLAY = previous.replay
BASE_CACHE_ROOTS = previous.cache_roots
ALLOWED_PAPER_REVISIONS = {
    "paper/fse2027/main.tex", "paper/fse2027/supplement.tex", "docs/FSE_2027_PAPER_DRAFT.md",
}


def prefix_rows(rows, count):
    if type(count) is not int or count not in (1, 3, 5) or len(rows) != 5:
        raise ValueError("Only a fixed Top-1/3/5 prefix of an archived Top-5 is allowed")
    return rows[:count]


def baseline_protocol(protocol):
    return {**protocol, "protocol_version": protocol["baseline_order_version"]}


def replay(protocol, source, p0):
    return BASE_REPLAY(baseline_protocol(protocol), source, p0)


def cache_roots(protocol):
    return (PRIOR, *BASE_CACHE_ROOTS(protocol))


def configure():
    # Reuse the frozen collector and metric implementation in this process only.
    previous.OUT = OUT
    previous.replay = replay
    previous.cache_roots = cache_roots


def freeze():
    protocol = read(PROTOCOL)
    prior = read(PRIOR / "protocol_lock.json")
    receipt = read(PRIOR / "completion_receipt.json")
    assert receipt["status"] == "complete"
    assert read(PRIOR / "provenance_audit_v2.json")["status"] == "pass"
    assert read(PRIOR / "ranking_audit.json")["status"] == "pass"
    assert protocol["baseline_order_version"] == prior["protocol"]["protocol_version"]
    for key in ("blocks", "replicate_seed", "methods", "target_retrievers", "target_models", "verifier_model",
                "source_density", "source_corpora", "source_corpora_lock", "chunk_size_words", "chunk_overlap_words",
                "unit_count", "collection_horizon", "decision_p0", "scheduler_p0", "alpha", "policy_seeds"):
        assert protocol[key] == prior["protocol"][key]
    assert protocol["scenarios"] == ["top1", *prior["protocol"]["scenarios"]]
    assert protocol["skipped_experiments"] == ["hybrid_retrieval"]
    assert all(file_hash(ROOT / name) == sha for name, sha in prior["source_sha256"].items())
    assert all(file_hash(ROOT / name) == sha for name, sha in receipt["files_sha256"].items())

    revision = {name: {"prior_sha256": sha, "current_sha256": file_hash(ROOT / name)}
                for name, sha in prior["frozen_source_sha256"].items() if file_hash(ROOT / name) != sha}
    assert set(revision) <= ALLOWED_PAPER_REVISIONS
    snapshot = {name: file_hash(ROOT / name) for name in prior["frozen_source_sha256"]}
    inputs = {**prior["source_sha256"], **receipt["files_sha256"]}
    for path in [PROTOCOL, Path(__file__).resolve(), ROOT / "tests/test_retrieval_deployment_completion.py",
                 PRIOR / "protocol_lock.json", PRIOR / "completion_receipt.json",
                 *sorted((PRIOR / "contexts").rglob("contexts.json")),
                 *sorted((PRIOR / "execution").rglob("answers.json")),
                 *sorted((PRIOR / "dense").rglob("*.npz")), *sorted((PRIOR / "dense").rglob("*.json"))]:
        inputs[str(path.relative_to(ROOT))] = file_hash(path)
    settings = {name: {"model": previous.shared.load_model_preset(name, ROOT / "models").model,
                       "args": previous.shared.load_model_preset(name, ROOT / "models").llm_arg}
                for name in [*protocol["target_models"], protocol["verifier_model"]]}
    assert settings == prior["model_settings"]
    lock = {"protocol": protocol, "source_protocol": prior["source_protocol"], "model_settings": settings,
            "source_sha256": inputs, "frozen_source_sha256": snapshot,
            "authorized_model_naming_revisions_since_prior_study": revision,
            "historical_prior_lock_not_modified": True}
    path = OUT / "protocol_lock.json"
    if path.exists():
        assert read(path) == lock, "Frozen completion changed; preserve and version"
    else:
        save(path, lock)
    previous.verify(lock)
    return protocol, prior["source_protocol"], lock


def copy_exact(source, destination):
    if destination.exists():
        assert file_hash(destination) == file_hash(source)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def prepare(protocol, source, lock):
    jobs, reused = [], {}
    for name in ("chunk_inventory.json", "source_audit.json"):
        copy_exact(PRIOR / name, OUT / name)
    for block, method, state, retriever in itertools.product(
            protocol["blocks"], protocol["methods"], ("watermarked", "clean"), protocol["target_retrievers"]):
        reference = read(PRIOR / "contexts" / block / method / "top5_reference" / retriever / state / "contexts.json")
        expected_tests, _, _ = previous.shared.suite(protocol, source, block, method)
        assert reference["tests"] == expected_tests
        for scenario in protocol["scenarios"]:
            job = f"{block}/{method}/{scenario}/{retriever}/{state}"
            jobs.append(job)
            path = OUT / "contexts" / job / "contexts.json"
            if scenario != "top1":
                src = PRIOR / "contexts" / job / "contexts.json"
                copy_exact(src, path)
                reused[str(path.relative_to(OUT))] = {"source": str(src.relative_to(ROOT)), "sha256": file_hash(src)}
                for model in protocol["target_models"]:
                    src = PRIOR / "execution" / job / model / "answers.json"
                    dst = OUT / "execution" / job / model / "answers.json"
                    assert read(src)["complete"]
                    copy_exact(src, dst)
                    reused[str(dst.relative_to(OUT))] = {"source": str(src.relative_to(ROOT)), "sha256": file_hash(src)}
                continue
            contexts = {test["unit_id"]: prefix_rows(reference["contexts"][test["unit_id"]], 1) for test in expected_tests}
            payload = {"condition": {**reference["condition"], "scenario": "top1", "top_k": 1},
                       "tests": expected_tests, "contexts": contexts,
                       "diagnostics": {test["unit_id"]: context_diagnostics(
                           test, contexts[test["unit_id"]], carrier_present=state == "watermarked") for test in expected_tests}}
            payload["signature"] = digest(payload)
            if path.exists():
                assert read(path) == payload
            else:
                save(path, payload)
            print(f"[prepared] {job}", flush=True)
    assert len(jobs) == 64 and len(reused) == 144
    save(OUT / "frozen_contexts.json", {"jobs": jobs, "contexts_sha256": {
        job: file_hash(OUT / "contexts" / job / "contexts.json") for job in jobs}})
    save(OUT / "reused_input_manifest.json", {"files": reused, "logical_answer_records": 2880,
         "new_top1_groups": 32, "new_top1_logical_answer_records": 960})
    previous.verify(lock)


def audit_rankings(protocol, source, lock):
    banks, corpora, logical, physical = previous.source_banks(protocol), {}, 0, 0
    for block, method, state in itertools.product(protocol["blocks"], protocol["methods"], ("watermarked", "clean")):
        raw = banks[block]["watermarked"][method] if state == "watermarked" else banks[block]["clean"]
        documents = [IndexedMigrationDocument(**r) for r in raw]
        chunks = rechunk_documents(documents, size=120, overlap=30)
        by_source = defaultdict(list)
        for row in chunks:
            by_source[row.source_type, row.source_id].append(row)
        for document in documents:
            own, words = by_source[document.source_type, document.source_id], document.text.split()
            if len(words) <= 120:
                assert own == [document]
            else:
                recovered = own[0].text.split()
                for row in own[1:]:
                    assert recovered[-30:] == row.text.split()[:30]
                    recovered.extend(row.text.split()[30:])
                assert recovered == words
            logical += 1
        physical += len(chunks)
        corpora[block, method, state] = chunks
    texts = sorted({r.text for docs in corpora.values() for r in docs})
    vectors = checked_vectors(texts, PRIOR / "dense/shared_chunks.npz", source)
    positions = {text: i for i, text in enumerate(texts)}
    frozen, checked = read(OUT / "frozen_contexts.json"), 0
    for block, method, state, retriever in itertools.product(
            protocol["blocks"], protocol["methods"], ("watermarked", "clean"), protocol["target_retrievers"]):
        tests, _, _ = previous.shared.suite(protocol, source, block, method)
        docs = corpora[block, method, state]
        if retriever == "bm25":
            scores = sparse_bm25_scores([d.text for d in docs], [t["question"] for t in tests]).toarray()
        else:
            queries = checked_vectors([t["question"] for t in tests], PRIOR / "dense/queries" / f"{digest(tests)}.npz", source)
            scores = np.einsum("ij,kj->ik", queries, vectors[[positions[d.text] for d in docs]], optimize=True).astype(np.float32)
        reference = read(previous.archived_path(protocol, block, method, retriever, state))
        for scenario in protocol["scenarios"]:
            job = f"{block}/{method}/{scenario}/{retriever}/{state}"
            path = OUT / "contexts" / job / "contexts.json"
            assert file_hash(path) == frozen["contexts_sha256"][job]
            payload = read(path)
            assert payload["signature"] == digest({k: v for k, v in payload.items() if k != "signature"})
            assert payload["tests"] == tests
            for i, test in enumerate(tests):
                expected = (context_rows(scores[i], docs, top_k=5) if scenario == "rechunk_120_30" else
                            prefix_rows(reference["contexts"][test["probe_id"]], protocol["scenario_top_k"][scenario]))
                actual = payload["contexts"][test["unit_id"]]
                assert len(actual) == len(expected)
                for a, b in zip(actual, expected, strict=True):
                    assert all(a[k] == b[k] for k in ("source", "id", "chunk_index", "text"))
                    assert np.isclose(a["score"], b["score"], rtol=1e-6, atol=1e-7)
                assert payload["diagnostics"][test["unit_id"]] == context_diagnostics(test, actual, carrier_present=state == "watermarked")
                checked += 1
    assert checked == 1920
    result = {"status": "pass", "reconstructed_contexts": checked, "new_top1_contexts": 480,
              "logical_document_occurrences_checked": logical, "physical_chunk_occurrences_checked": physical,
              "word_order_and_overlap_preserved": True, "top1_and_top3_exact_prefixes": True,
              "new_api_requests": 0, **previous.verify(lock)}
    save(OUT / "ranking_audit.json", result)
    print(json.dumps(result), flush=True)


def report(protocol, source, lock):
    previous.report(protocol, source, lock)
    result = read(OUT / "results.json")
    old = read(PRIOR / "results.json")
    assert [r for r in result["summary"] if r["scenario"] != "top1"] == old["summary"]
    old_traces = read(PRIOR / "trajectories.json")
    assert [r for r in read(OUT / "trajectories.json") if r["scenario"] != "top1"] == old_traces
    result.update({"prior_results_exactly_reproduced": True, "new_top1_logical_records": 960,
                   "reused_prior_logical_records": 2880, "hybrid_retrieval_not_run": True})
    save(OUT / "results.json", result)


def audit_requests(protocol, source, lock):
    frozen = read(OUT / "frozen_contexts.json")
    roots = (OUT, *cache_roots(protocol))
    accepted, fresh, models, schema = {}, Counter(), Counter(), Counter()
    def check(key, model, prompts, content=None):
        path = next((root / "requests" / key[:2] / f"{key}.json" for root in roots
                     if (root / "requests" / key[:2] / f"{key}.json").exists()), None)
        assert path is not None
        record = read(path)
        kind = validate_request(record, key, model, lock["model_settings"][model], prompts, content)
        if key not in accepted:
            accepted[key] = {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path), "parameter_validation": kind}
            models[model, record["returned_model"]] += 1
            schema[kind] += 1
            if path.is_relative_to(OUT):
                fresh[model] += 1
        return record
    checked, origins = 0, Counter()
    for job, model in itertools.product(frozen["jobs"], protocol["target_models"]):
        path = OUT / "contexts" / job / "contexts.json"
        assert file_hash(path) == frozen["contexts_sha256"][job]
        context = read(path)
        data = read(OUT / "execution" / job / model / "answers.json")
        assert data["complete"] and data["signature"] == context["signature"]
        assert data["condition"] == context["condition"] and data["model"] == model
        assert len(data["records"]) == len(context["tests"]) == 30
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
            origins["top1" if context["condition"]["scenario"] == "top1" else "prior"] += 1
    assert checked == 3840 and origins == {"top1": 960, "prior": 2880}
    for name, row in read(OUT / "reused_input_manifest.json")["files"].items():
        assert file_hash(OUT / name) == file_hash(ROOT / row["source"]) == row["sha256"]
    _, cells, traces = replay(protocol, source, protocol["decision_p0"])
    assert read(OUT / "results.json")["summary"] == cells and read(OUT / "trajectories.json") == traces
    orders = {}
    for trace in traces:
        assert len(set(trace["unit_order"])) == 30
        if trace["method"] == "SentinelRAG":
            key = trace["block"], trace["policy_seed"]
            assert key not in orders or orders[key] == trace["unit_order"]
            orders[key] = trace["unit_order"]
    assert read(OUT / "execution_status.json")["answer_groups"] == 128
    result = {"status": "pass", "checked_answer_records": checked, "record_origin": dict(origins),
              "unique_verified_requests": len(accepted), "fresh_unique_execution_requests_by_preset": dict(fresh),
              "returned_models": [{"preset": k[0], "returned_model": k[1], "unique_requests": v} for k, v in models.items()],
              "request_parameter_validation_counts": dict(schema), "accepted_requests": accepted,
              "reused_prior_files_byte_identical": True, "prior_metrics_and_orders_unchanged": True,
              **previous.verify(lock)}
    save(OUT / "provenance_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "accepted_requests"}), flush=True)


def summarize(protocol, source, lock):
    result, trajectories = read(OUT / "results.json"), read(OUT / "trajectories.json")
    provenance, rankings = read(OUT / "provenance_audit.json"), read(OUT / "ranking_audit.json")
    assert provenance["status"] == rankings["status"] == "pass"
    replayed = defaultdict(list)
    for trace in trajectories:
        key = tuple(trace[k] for k in ("scenario", "state", "method", *exports.PAIR_KEYS))
        replayed[key].append(exports.first_crossing_metrics(trace["curve"]))
    for row in result["conditions"]:
        key = tuple(row[k] for k in ("scenario", "state", "method", *exports.PAIR_KEYS))
        assert all(np.isclose(row[m], np.mean([r[m] for r in replayed[key]]), atol=1e-12) for m in previous.METRICS)
    pairs = exports.paired_comparisons(result["conditions"], protocol["scenarios"])
    directions = exports.summarize_pairs(pairs, protocol["scenarios"])
    depth = []
    for scenario, count in (("top1", 1), ("top3", 3), ("top5_reference", 5)):
        for method in protocol["methods"]:
            positive = next(r for r in result["summary"] if (r["scenario"], r["method"], r["state"]) == (scenario, method, "watermarked"))
            clean = next(r for r in result["summary"] if (r["scenario"], r["method"], r["state"]) == (scenario, method, "clean"))
            depth.append({"top_k": count, "method": method, "conditions": 8,
                          **{m: positive[m] for m in previous.METRICS}, "fpr15": clean["vsr15"], "fpr30": clean["vsr30"]})
    for name, rows in (("summary.csv", result["summary"]), ("conditions.csv", result["conditions"]),
                       ("paired_comparisons.csv", pairs), ("paired_direction_summary.csv", directions),
                       ("retrieval_depth.csv", depth)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save(OUT / "paired_analysis.json", {"complete": True, "paired_direction_summary": directions,
         "paired_comparisons": pairs, "metrics_independently_reproduced": True, "not_pooled_with_144_conditions": True})
    command = [sys.executable, "-m", "pytest", *[f"tests/{name}" for name in exports.TEST_FILES],
               "tests/test_retrieval_deployment_completion.py", "-q"]
    tests = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    save(OUT / "test_receipt.json", {"command": command, "exit_code": tests.returncode,
                                   "stdout": tests.stdout, "stderr": tests.stderr})
    assert tests.returncode == 0, tests.stdout + tests.stderr
    files = [OUT / name for name in ("protocol_lock.json", "frozen_contexts.json", "reused_input_manifest.json",
        "execution_status.json", "results.json", "trajectories.json", "ranking_audit.json", "provenance_audit.json",
        "summary.csv", "conditions.csv", "paired_comparisons.csv", "paired_direction_summary.csv",
        "retrieval_depth.csv", "paired_analysis.json", "test_receipt.json")]
    report_path = ROOT / "docs/RETRIEVAL_DEPLOYMENT_COMPLETION_20261003.zh-CN.md"
    if report_path.exists():
        files.append(report_path)
    receipt = {"status": "complete", "logical_answer_records": 3840, "new_top1_logical_records": 960,
               "reused_prior_logical_records": 2880, "prior_metrics_and_orders_unchanged": True,
               "hybrid_retrieval_not_run": True, "independent_first_crossing_reproduction": True,
               "tests": {"exit_code": 0, "summary": tests.stdout.strip().splitlines()[-1]},
               "execution": read(OUT / "execution_status.json"),
               "fresh_unique_execution_requests_by_preset": provenance["fresh_unique_execution_requests_by_preset"],
               "files_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in files}, **previous.verify(lock)}
    save(OUT / "completion_receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "files_sha256"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "preflight", "execute", "report", "audit", "summarize", "all"))
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    configure()
    protocol, source, lock = freeze()
    if args.stage in ("prepare", "all"):
        prepare(protocol, source, lock)
        audit_rankings(protocol, source, lock)
    if args.stage in ("preflight", "execute", "all"):
        asyncio.run(previous.execute(protocol, lock, args.concurrency, preflight_only=args.stage == "preflight"))
    if args.stage in ("report", "execute", "all"):
        report(protocol, source, lock)
    if args.stage in ("audit", "execute", "all"):
        if args.stage == "audit":
            audit_rankings(protocol, source, lock)
        audit_requests(protocol, source, lock)
    if args.stage in ("summarize", "execute", "all"):
        summarize(protocol, source, lock)


if __name__ == "__main__":
    main()
