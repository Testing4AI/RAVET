#!/usr/bin/env python3
"""Run selector x scheduler experiments without changing either paper or old data."""

from __future__ import annotations

import argparse
import asyncio
import csv
from collections import defaultdict

import numpy as np
from openai import APIError
from run_partition_confirmation import (
    OUT as SOURCE,
)
from run_partition_confirmation import (
    PAPER_FILES,
    ROOT,
    RequestCache,
    digest,
    file_hash,
    owner_probes,
    read,
    save,
)

from sentinelrag.prompts.templates import PromptTemplates
from sentinelrag.testing.adaptive import stable_seed
from sentinelrag.testing.migration_confirmation import explicit_abstention
from sentinelrag.testing.orthogonal_policy import next_selected_probe, select_candidates
from sentinelrag.testing.prioritization import (
    evaluate_ordering,
    normalized_answer_match,
)
from sentinelrag.utils.model_registry import load_model_preset

OUT = ROOT / "output/orthogonal-policy-diagnostic-20260906"
PROTOCOL = ROOT / "experiments/orthogonal_policy_diagnostic_20260906.json"


def model_settings(name):
    preset = load_model_preset(name, models_dir=ROOT / "models")
    return {"model": preset.model, "llm_url": preset.llm_url, "args": preset.llm_arg}


def request_key(name, prompt):
    settings = model_settings(name)
    return digest(
        {
            "preset": name,
            "model": settings["model"],
            "args": settings["args"],
            "prompt": prompt,
        }
    )


def cached_request(key, *, roots=None):
    for root in (OUT, SOURCE) if roots is None else roots:
        path = root / "requests" / key[:2] / f"{key}.json"
        if path.exists():
            row = read(path)
            assert row["key"] == key
            return row
    return None


class MissingRequest(Exception):
    def __init__(self, model, key):
        super().__init__(f"Missing cached request: {model} {key}")
        self.model = model
        self.key = key


class CachedOnly:
    def __init__(self, roots=None):
        self.roots = roots

    async def ask(self, model, prompt):
        key = request_key(model, prompt)
        record = cached_request(key, roots=self.roots)
        if record is None:
            raise MissingRequest(model, key)
        assert record["model"] == model_settings(model)["model"]
        assert record["prompt"] == prompt
        return record


def freeze():
    protocol = read(PROTOCOL)
    source = read(SOURCE / "frozen_protocol.json")
    for field in (
        "datasets",
        "fixtures",
        "target_retrievers",
        "migration_seeds",
        "target_models",
        "verifier_model",
        "top_k",
        "p0",
        "alpha",
    ):
        assert protocol[field] == source["protocol"][field]
    frozen_path = OUT / "frozen_protocol.json"
    context_hashes = {
        job: file_hash(SOURCE / job / "contexts.json")
        for job in read(SOURCE / "jobs.json")["jobs"]
    }
    public_settings = {
        name: model_settings(name)
        for name in protocol["target_models"] + [protocol["verifier_model"]]
    }
    if frozen_path.exists():
        frozen = read(frozen_path)
        assert frozen["protocol"] == protocol
        assert frozen["source_context_hashes"] == context_hashes
        assert frozen["model_settings"] == public_settings
    else:
        save(
            frozen_path,
            {
                "protocol": protocol,
                "protocol_sha256": file_hash(PROTOCOL),
                "source_protocol_sha256": file_hash(SOURCE / "frozen_protocol.json"),
                "source_context_hashes": context_hashes,
                "model_settings": public_settings,
                "paper_sha256": {
                    str(path.relative_to(ROOT)): file_hash(path) for path in PAPER_FILES
                },
            },
        )
    return protocol


def policy_seed(protocol, condition):
    return stable_seed(
        protocol["protocol_version"],
        condition["dataset"],
        condition["seed"],
        protocol["policy_seed"],
    )


def prepared_sets(protocol, *, output_root=None):
    sets = {}
    metadata = {}
    for dataset in protocol["datasets"]:
        clusters = read(SOURCE / dataset / "partitions.json")["owner_clusters"]
        for fixture in protocol["fixtures"]:
            _, probes = owner_probes(dataset, fixture, protocol["top_k"])
            for selector in protocol["selectors"]:
                selected = select_candidates(probes, selector)
                key = (dataset, fixture, selector)
                sets[key] = (selected, clusters)
                metadata["/".join(key)] = [
                    {
                        "probe_id": p.profile.candidate.probe_id,
                        "canary_id": p.profile.candidate.canary_id,
                        "retrieval_probability": p.retained_success_probability,
                    }
                    for p in selected
                ]
    save(
        (OUT if output_root is None else output_root) / "selected_candidates.json",
        metadata,
    )
    return sets


async def run_trace(
    protocol,
    requests,
    job,
    model,
    selector,
    scheduler,
    selected,
    clusters,
    *,
    output_root=None,
    context_root=None,
    seed_override=None,
):
    destination = OUT if output_root is None else output_root
    source = SOURCE if context_root is None else context_root
    payload = read(source / job / "contexts.json")
    condition = payload["condition"]
    path = destination / "traces" / job / model / f"{selector}--{scheduler}.json"
    ids = [p.profile.candidate.probe_id for p in selected]
    selection_signature = digest(
        [
            (
                p.profile.candidate.probe_id,
                p.retained_success_probability,
                p.profile.proposed_score,
            )
            for p in selected
        ]
    )
    seed = policy_seed(protocol, condition) if seed_override is None else seed_override
    rows = []
    if path.exists():
        checkpoint = read(path)
        assert checkpoint["selection_signature"] == selection_signature
        assert checkpoint["context_signature"] == payload["signature"]
        assert checkpoint["policy_seed"] == seed
        rows = checkpoint["records"]
        if checkpoint["complete"]:
            return
    profiles = {p.profile.candidate.probe_id: p.profile for p in selected}

    def checkpoint(missing=None):
        evaluation = evaluate_ordering(
            [profiles[r["probe_id"]] for r in rows],
            {r["probe_id"]: r["is_correct"] for r in rows},
            p0=protocol["p0"],
            alpha=protocol["alpha"],
        )
        save(
            path,
            {
                "condition": condition,
                "model": model,
                "selector": selector,
                "scheduler": scheduler,
                "policy_seed": seed,
                "selected_probe_ids": ids,
                "selection_signature": selection_signature,
                "context_signature": payload["signature"],
                "complete": len(rows) == protocol["max_queries"],
                "records": rows,
                "evaluation": evaluation,
                "missing": missing,
            },
        )

    while len(rows) < protocol["max_queries"]:
        probe = next_selected_probe(
            selected,
            clusters,
            scheduler,
            rows,
            seed=seed,
            p0=protocol["p0"],
            prior=protocol["prior_retention"],
        )
        candidate = probe.profile.candidate
        context = payload["contexts"][candidate.probe_id]
        try:
            answer = await requests.ask(
                model,
                PromptTemplates.answer_with_rag(
                    "\n".join(r["text"] for r in context),
                    candidate.question,
                ),
            )
            if explicit_abstention(answer["content"], candidate.expected_answer):
                verdict = {"key": None, "content": "No"}
                judge_kind = "exact_abstention_rule_v1"
            else:
                verdict = await requests.ask(
                    protocol["verifier_model"],
                    PromptTemplates.judge_answer_correctness(
                        candidate.question,
                        answer["content"],
                        candidate.expected_answer,
                    ),
                )
                judge_kind = "glm_semantic"
        except MissingRequest as error:
            checkpoint(
                {"model": error.model, "key": error.key, "probe_id": candidate.probe_id}
            )
            return
        decision = verdict["content"].strip().casefold()
        if not decision.startswith(("yes", "no")):
            raise ValueError("Unparseable oracle response, not a negative label")
        rows.append(
            {
                "probe_id": candidate.probe_id,
                "canary_id": candidate.canary_id,
                "question": candidate.question,
                "expected_answer": candidate.expected_answer,
                "answer": answer["content"],
                "is_correct": decision.startswith("yes"),
                "is_correct_exact": normalized_answer_match(
                    answer["content"], candidate.expected_answer
                ),
                "answer_request": answer["key"],
                "judge_request": verdict["key"],
                "judge_kind": judge_kind,
                "owner_retrieval_probability": probe.retained_success_probability,
                **payload["diagnostics"][candidate.probe_id],
            }
        )
        checkpoint()


async def execute(online, concurrency):
    protocol = freeze()
    sets = prepared_sets(protocol)
    requests = (
        RequestCache(protocol, concurrency, output_root=OUT, readonly_roots=(SOURCE,))
        if online
        else CachedOnly()
    )
    limiter = asyncio.Semaphore(4)
    completed_groups = 0
    save(
        OUT / "execution_status.json",
        {"status": "running", "mode": "online" if online else "cache_only"},
    )
    if online:
        for name, (client, preset) in requests.clients.items():
            try:
                await client.async_client.with_options(
                    max_retries=0
                ).chat.completions.create(
                    model=preset.model,
                    messages=[{"role": "user", "content": "Reply only OK."}],
                    timeout=30,
                    max_tokens=64,
                )
            except APIError as error:
                save(
                    OUT / "execution_status.json",
                    {
                        "status": "blocked_preflight",
                        "model": name,
                        "error_type": type(error).__name__,
                    },
                )
                raise RuntimeError(
                    f"Preflight failed: {name} {type(error).__name__}"
                ) from None
            print(f"[preflight] {name}: connected", flush=True)

    async def group(job, model):
        nonlocal completed_groups
        async with limiter:
            condition = read(SOURCE / job / "contexts.json")["condition"]
            await asyncio.gather(
                *(
                    run_trace(
                        protocol,
                        requests,
                        job,
                        model,
                        selector,
                        scheduler,
                        *sets[(condition["dataset"], condition["fixture"], selector)],
                    )
                    for selector in protocol["selectors"]
                    for scheduler in protocol["schedulers"]
                )
            )
            completed_groups += 1
            if completed_groups % 8 == 0:
                print(f"[groups] {completed_groups}/96", flush=True)

    await asyncio.gather(
        *(
            group(job, model)
            for job in read(SOURCE / "jobs.json")["jobs"]
            for model in protocol["target_models"]
        )
    )
    summary = report()
    save(
        OUT / "execution_status.json",
        {
            "status": "complete"
            if summary["completed_traces"] == protocol["expected_traces"]
            else "needs_missing_requests",
            "mode": "online" if online else "cache_only",
            "new_requests_this_run": getattr(requests, "new_requests", 0),
            "reused_requests_this_run": getattr(requests, "reused_requests", None),
            **summary,
        },
    )


def endpoint(row):
    evaluation = row["evaluation"]
    stop = evaluation["queries_to_anytime_significance"]
    return {
        **{
            f"vsr_at_{b}": int(evaluation["curve"][b - 1]["significant"])
            for b in (2, 4, 6, 8, 10)
        },
        "anytime_at_10": int(stop is not None),
        "restricted_queries_10": stop if stop is not None else 10,
    }


def report():
    protocol = read(OUT / "frozen_protocol.json")["protocol"]
    rows = [read(path) for path in (OUT / "traces").glob("*/*/*/*/*/api-*/*.json")]
    complete = [r for r in rows if r["complete"]]
    missing = {r["missing"]["key"]: r["missing"] for r in rows if r.get("missing")}
    selected_ids = {}
    for row in rows:
        c = row["condition"]
        key = (c["dataset"], c["fixture"], row["selector"])
        current = tuple(row["selected_probe_ids"])
        assert selected_ids.setdefault(key, current) == current
        assert len({r["canary_id"] for r in row["records"]}) == len(row["records"])
    frozen = read(OUT / "frozen_protocol.json")
    unchanged = all(
        file_hash(ROOT / name) == value
        for name, value in frozen["paper_sha256"].items()
    )
    inventory = {
        "started_traces": len(rows),
        "completed_traces": len(complete),
        "expected_traces": protocol["expected_traces"],
        "observed_logical_queries": sum(len(r["records"]) for r in rows),
        "distinct_next_missing_requests": len(missing),
        "paper_unchanged": unchanged,
        "all_schedulers_share_selected_sets": True,
    }
    save(OUT / "inventory.json", inventory)
    save(OUT / "next_missing_requests.json", list(missing.values()))
    summaries = defaultdict(list)
    data = []
    for row in complete:
        c = row["condition"]
        entry = {
            **{k: c[k] for k in ("dataset", "fixture", "retriever", "seed", "state")},
            "model": row["model"],
            "selector": row["selector"],
            "scheduler": row["scheduler"],
            **endpoint(row),
        }
        data.append(entry)
        summaries[(c["fixture"], row["selector"], row["scheduler"], c["state"])].append(
            entry
        )
    if data:
        with (OUT / "completed_traces.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    text = [
        "# Probe 选择与调度正交诊断",
        "",
        f"完成 {len(complete)}/{protocol['expected_traces']} 条轨迹；已观察逻辑查询 {inventory['observed_logical_queries']}；论文未变：{unchanged}。",
        "本轮使用此前已观察的迁移实现，属于机制诊断，不是新方法的独立确认。",
        "同一 selector 在三个 scheduler 上使用完全相同的候选集合；策略种子与迁移种子分开。",
        "主预算为 4/6，10 次为次要预算；anytime 为共同 e-process 的首次越界，未检出消耗记满 10 次。",
        "",
    ]
    table = []
    if len(complete) != protocol["expected_traces"]:
        text += [
            f"当前仍有 {len(missing)} 个不同的下一步请求缺失；后续还可能出现其他未观察请求。",
            "缺失处暂停，不把未知结果当作答错；暂不汇总不平衡完成子集的性能排名。",
        ]
    else:
        text += [
            "| Fixture | Selector | Scheduler | 条件 | VSR@4 | VSR@6 | VSR@10 | Anytime@10 | E[min(T,10)] |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for (fixture, selector, scheduler, state), group in sorted(summaries.items()):
            values = {
                metric: float(np.mean([r[metric] for r in group]))
                for metric in endpoint(complete[0])
            }
            table.append(
                {
                    "fixture": fixture,
                    "selector": selector,
                    "scheduler": scheduler,
                    "state": state,
                    "conditions": len(group),
                    **values,
                }
            )
            if state == "watermarked":
                text.append(
                    f"| {fixture} | {selector} | {scheduler} | {len(group)} | "
                    + " | ".join(
                        f"{values[m]:.3f}"
                        for m in ("vsr_at_4", "vsr_at_6", "vsr_at_10", "anytime_at_10")
                    )
                    + f" | {values['restricted_queries_10']:.2f} |"
                )
        negatives = [r for r in data if r["state"] == "clean"]
        text += [
            "",
            f"Clean：{len(negatives)} 条；anytime 错误验证 {sum(r['anytime_at_10'] for r in negatives)} 次。",
            "共享语料及缓存，不是独立部署样本；p0=.15 的有效性仍依赖条件 null 上界，不由零误报自动证明。",
        ]
    save(OUT / "summary_table.json", table)
    (OUT / "report.zh-CN.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(inventory, flush=True)
    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "replay", "run", "report"))
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepared_sets(freeze())
    elif args.stage == "report":
        report()
    else:
        asyncio.run(execute(args.stage == "run", args.concurrency))
