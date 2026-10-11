#!/usr/bin/env python3
"""Audit frozen answer banks, including historical parameter-committed caches."""

import itertools
import json
from collections import Counter
from pathlib import Path

import run_retrieval_deployment_changes as study
from run_partition_confirmation import digest, file_hash, read, save
from sentinelrag.prompts.templates import PromptTemplates
from sentinelrag.testing.corpus_retrieval import IndexedMigrationDocument
from sentinelrag.testing.migration_confirmation import explicit_abstention
from sentinelrag.testing.ragwm import yes_no
from sentinelrag.testing.retrieval_deployment import context_diagnostics, rechunk_documents


def validate_request(request, key, model, settings, prompts, content=None):
    allowed = {digest({"preset": model, "model": settings["model"], "args": settings["args"], "prompt": p}): p
               for p in prompts}
    assert key in allowed
    assert (request["key"], request["model"], request["prompt"]) == (key, settings["model"], allowed[key])
    assert request["returned_model"] == settings["model"]
    assert request["content"] and request["finish_reason"] != "length"
    assert content is None or content == request["content"]
    if "preset" in request:
        assert request["preset"] == model
    if "args" in request:
        assert request["args"] == settings["args"]
        return "explicit_parameter_field_and_reconstructed_key"
    # Legacy RequestCache omitted this field, but its request key committed it.
    return "legacy_parameters_committed_by_reconstructed_request_key"


def main():
    protocol, source, lock = study.freeze()
    out, root = study.OUT, study.ROOT
    roots = (out, *study.cache_roots(protocol))
    frozen = read(out / "frozen_contexts.json")
    accepted, fresh, returned, schema = {}, Counter(), Counter(), Counter()
    legacy_roots = Counter()

    def check(key, model, prompts, content=None):
        path = next((base / "requests" / key[:2] / f"{key}.json" for base in roots
                     if (base / "requests" / key[:2] / f"{key}.json").exists()), None)
        assert path is not None
        request = read(path)
        kind = validate_request(request, key, model, lock["model_settings"][model], prompts, content)
        if key not in accepted:
            accepted[key] = {"path": str(path.relative_to(root)), "sha256": file_hash(path), "parameter_validation": kind}
            returned[model, request["returned_model"]] += 1
            schema[kind] += 1
            if kind.startswith("legacy_"):
                legacy_roots[str(path.parents[2].relative_to(root))] += 1
            if path.is_relative_to(out):
                fresh[model] += 1
        return request

    checked = 0
    banks = study.source_banks(protocol)
    for job in frozen["jobs"]:
        path = out / "contexts" / job / "contexts.json"
        assert file_hash(path) == frozen["contexts_sha256"][job]
        context = read(path)
        assert context["signature"] == digest({k: v for k, v in context.items() if k != "signature"})
        condition = context["condition"]
        block, method, state = (condition[k] for k in ("block", "method", "state"))
        expected_tests, _, _ = study.shared.suite(protocol, source, block, method)
        assert context["tests"] == expected_tests
        raw = banks[block]["watermarked"][method] if state == "watermarked" else banks[block]["clean"]
        documents = [IndexedMigrationDocument(**d) for d in raw]
        if condition["scenario"] == "rechunk_120_30":
            documents = rechunk_documents(documents, size=protocol["chunk_size_words"], overlap=protocol["chunk_overlap_words"])
        lookup = {(d.source_type, d.source_id, d.chunk_index): d.text for d in documents}
        assert len(lookup) == len(documents)
        reference = read(study.archived_path(protocol, block, method, condition["retriever"], state))
        for test in context["tests"]:
            rows = context["contexts"][test["unit_id"]]
            assert len(rows) == condition["top_k"]
            assert all(lookup[r["source"], r["id"], r["chunk_index"]] == r["text"] for r in rows)
            if condition["scenario"] != "rechunk_120_30":
                assert rows == reference["contexts"][test["probe_id"]][:condition["top_k"]]
            assert context["diagnostics"][test["unit_id"]] == context_diagnostics(test, rows, carrier_present=state == "watermarked")
        for model in protocol["target_models"]:
            data = read(out / "execution" / job / model / "answers.json")
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
    _, cells, trajectories = study.replay(protocol, source, protocol["decision_p0"])
    assert read(out / "trajectories.json") == trajectories
    assert read(out / "results.json")["summary"] == cells
    baseline_orders = {}
    for trace in trajectories:
        assert len(set(trace["unit_order"])) == 30
        if trace["method"] == "SentinelRAG":
            key = trace["block"], trace["policy_seed"]
            if key in baseline_orders:
                assert baseline_orders[key] == trace["unit_order"]
            baseline_orders[key] = trace["unit_order"]
    execution = read(out / "execution_status.json")
    assert execution["complete"] and execution["answer_groups"] == 96
    result = {"status": "pass", "audit_revision": "v2_legacy_cache_compatibility",
              "audit_script_sha256": file_hash(Path(__file__).resolve()),
              "frozen_execution_script_not_modified": True, "answer_banks_not_modified": True,
              "initial_audit_issue": "Legacy cache records omitted args; original inline audit raised KeyError after complete collection",
              "checked_answer_records": checked, "unique_verified_requests": len(accepted),
              "fresh_unique_execution_requests_by_preset": dict(fresh),
              "request_parameter_validation_counts": dict(schema), "legacy_cache_roots": dict(legacy_roots),
              "legacy_caches_not_backfilled_or_rewritten": True,
              "returned_models": [{"preset": k[0], "returned_model": k[1], "unique_requests": v} for k, v in returned.items()],
              "baseline_orders_paired_across_scenarios": True, "accepted_requests": accepted, **study.verify(lock)}
    save(out / "provenance_audit_v2.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "accepted_requests"}), flush=True)


if __name__ == "__main__":
    main()
