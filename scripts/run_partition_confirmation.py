#!/usr/bin/env python3
"""Prepare, execute, and summarize the frozen full-corpus migration supplement."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from openai import APIError
from run_corpus_migration_retrieval import flatten_variant, load_owner_cache
from sklearn.metrics import adjusted_rand_score

from sentinelrag.prompts.templates import PromptTemplates
from sentinelrag.testing.adaptive import cluster_artifacts, prepare_probes, stable_seed
from sentinelrag.testing.artifacts import build_artifact_pools
from sentinelrag.testing.corpus_migrations import load_corpus_migration_manifest
from sentinelrag.testing.corpus_retrieval import IndexedMigrationDocument, context_rows
from sentinelrag.testing.matrix import sparse_bm25_scores
from sentinelrag.testing.migration_confirmation import (
    context_diagnostics,
    dense_subset_scores,
    explicit_abstention,
    independent_corpus_partitions,
    materialize_partition,
    next_probe,
    policy_ordering,
    serialize_documents,
)
from sentinelrag.testing.prioritization import (
    answer_tokens_supported,
    evaluate_ordering,
    normalized_answer_match,
)
from sentinelrag.utils.model_registry import create_llm_client_from_preset

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/partition_confirmation_20260906.json"
OUT = ROOT / "output/fse-partition-confirmation-20260906"
FRESH = ROOT / "output/fse-fresh-holdout"
PAPER_FILES = (ROOT / "docs/FSE_2027_PAPER_DRAFT.md", ROOT / "paper/fse2027/main.tex")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_path(dataset, fixture):
    if fixture == "original":
        return FRESH / "checkpointed" / dataset / "injection/injection_result.json"
    return FRESH / "candidates" / dataset / fixture / "injection_result.json"


def fixture_records(dataset, fixture):
    return load_corpus_migration_manifest(
        FRESH
        / "corpus-migrations"
        / dataset
        / fixture
        / "corpus_migration_manifest.json"
    )


def owner_probes(dataset, fixture, top_k):
    artifacts = build_artifact_pools(artifact_path(dataset, fixture))["factual"]
    cache = load_owner_cache(
        FRESH / "matrices" / fixture / dataset / "contriever/factual", artifacts
    )
    return artifacts, prepare_probes(cache, artifacts, top_k=top_k)


class DenseBank:
    def __init__(self, protocol):
        self.protocol = protocol
        self.model = None
        self.embeddings = {}

    def encode(self, texts):
        missing = list(
            dict.fromkeys(text for text in texts if text not in self.embeddings)
        )
        if missing:
            if self.model is None:
                import torch
                from sentence_transformers import SentenceTransformer

                torch.set_num_threads(4)
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                self.model = SentenceTransformer(
                    self.protocol["dense_model"],
                    revision=self.protocol["dense_revision"],
                    device=device,
                    local_files_only=True,
                )
            vectors = self.model.encode(
                missing,
                batch_size=8,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            self.embeddings.update(zip(missing, vectors, strict=True))
        return np.asarray([self.embeddings[text] for text in texts], dtype=np.float32)

    def scores(self, dataset, fixture, documents, candidates):
        signature = digest(
            {
                "documents": serialize_documents(documents),
                "questions": [c.question for c in candidates],
                "revision": self.protocol["dense_revision"],
            }
        )
        path = OUT / "dense" / dataset / fixture / "scores.npz"
        if path.exists() and read(path.with_suffix(".json"))["signature"] == signature:
            return np.load(path)["scores"]
        upstream = ROOT / "output/posthoc-modern-retriever" / dataset / fixture
        source = upstream / "scores-bge-m3-compressed_summary.npz"
        if source.exists():
            metadata = read(upstream / "retrieval_cache_manifest.json")
            if metadata["candidate_ids"] != [c.probe_id for c in candidates]:
                raise ValueError("Dense cache candidate mismatch")
            old_documents = [
                IndexedMigrationDocument(**row)
                for row in read(upstream / "documents-compressed_summary.json")
            ]
            scores = dense_subset_scores(
                np.load(source)["scores"], old_documents, documents
            )
            provenance = str(source.relative_to(ROOT))
        else:
            queries = self.encode([c.question for c in candidates])
            encoded = self.encode([d.text for d in documents])
            scores = queries @ encoded.T
            provenance = "new fixed-encoder BGE-M3 embeddings"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, scores=np.asarray(scores, dtype=np.float32))
        save(
            path.with_suffix(".json"),
            {
                "signature": signature,
                "provenance": provenance,
                "shape": list(scores.shape),
            },
        )
        return scores


def prepare():
    protocol = read(PROTOCOL)
    frozen = OUT / "frozen_protocol.json"
    if frozen.exists() and read(frozen)["protocol"] != protocol:
        raise ValueError("Protocol changed: use a new version/output directory")
    if not frozen.exists():
        save(
            frozen,
            {
                "protocol": protocol,
                "protocol_sha256": file_hash(PROTOCOL),
                "paper_sha256": {
                    str(p.relative_to(ROOT)): file_hash(p) for p in PAPER_FILES
                },
            },
        )
    bank = DenseBank(protocol)
    jobs = []
    for dataset in protocol["datasets"]:
        original_records = fixture_records(dataset, "original")
        original_artifacts, _ = owner_probes(dataset, "original", protocol["top_k"])
        owner_clusters = cluster_artifacts(
            original_artifacts,
            num_clusters=protocol["owner_cluster_count"],
            seed=stable_seed("2026-08-20-v2.0-frozen", dataset, "factual", "clusters"),
        )
        partition_path = OUT / dataset / "partitions.json"
        if partition_path.exists():
            assignment = read(partition_path)["assignment"]
        else:
            assignment = independent_corpus_partitions(
                original_records,
                num_clusters=protocol["migration_partition_count"],
                seed=protocol["partition_seed"],
            )
            ids = sorted(owner_clusters)
            save(
                partition_path,
                {
                    "assignment": assignment,
                    "owner_clusters": owner_clusters,
                    "adjusted_rand_index_sentinels": adjusted_rand_score(
                        [owner_clusters[c] for c in ids],
                        [assignment[f"sentinel:{c}"] for c in ids],
                    ),
                    "partition_counts": dict(Counter(assignment.values())),
                },
            )
        for fixture in protocol["fixtures"]:
            records = fixture_records(dataset, fixture)
            original_clean = {
                r.source_id: r.variants
                for r in original_records
                if r.source_type == "clean"
            }
            if original_clean != {
                r.source_id: r.variants for r in records if r.source_type == "clean"
            }:
                raise ValueError(
                    "Ordinary-document migrations differ across fixture policies"
                )
            artifacts, probes = owner_probes(dataset, fixture, protocol["top_k"])
            candidates = [p for a in artifacts for p in a.probes]
            if [(p.probe_id, p.question, p.expected_answer) for p in candidates] != [
                (p.probe_id, p.question, p.expected_answer)
                for a in original_artifacts
                for p in a.probes
            ]:
                raise ValueError("Fixture rewrite changed the test contract")
            full_documents = flatten_variant(records, protocol["variant"])
            full_dense = bank.scores(dataset, fixture, full_documents, candidates)
            migrated_texts = {
                r.source_id: "\n".join(r.variants[protocol["variant"]])
                for r in records
                if r.source_type == "sentinel"
            }
            for seed in protocol["migration_seeds"]:
                retained_topics = sorted(
                    random.Random(seed).sample(
                        range(protocol["migration_partition_count"]),
                        protocol["retained_partition_count"],
                    )
                )
                for retriever in protocol["target_retrievers"]:
                    for state in ("watermarked", "clean"):
                        directory = (
                            OUT / dataset / fixture / retriever / str(seed) / state
                        )
                        documents = materialize_partition(
                            records,
                            assignment,
                            set(retained_topics),
                            variant=protocol["variant"],
                            clean_only=state == "clean",
                        )
                        if not documents:
                            raise ValueError("Empty retained corpus")
                        if retriever == "bm25":
                            scores = (
                                sparse_bm25_scores(
                                    [d.text for d in documents],
                                    [c.question for c in candidates],
                                )
                                .toarray()
                                .astype(np.float32)
                            )
                            index_mode = "BM25 rebuilt on physically retained documents"
                        else:
                            scores = dense_subset_scores(
                                full_dense, full_documents, documents
                            )
                            index_mode = (
                                "fixed BGE-M3 exact subset inner-product ranking"
                            )
                        contexts = {}
                        diagnostics = {}
                        for index, candidate in enumerate(candidates):
                            ctx = context_rows(
                                scores[index], documents, top_k=protocol["top_k"]
                            )
                            contexts[candidate.probe_id] = ctx
                            diagnostics[candidate.probe_id] = context_diagnostics(
                                candidate, ctx
                            )
                            diagnostics[candidate.probe_id][
                                "legacy_whole_document_proxy"
                            ] = bool(
                                diagnostics[candidate.probe_id]["retrieved_sentinel"]
                                and answer_tokens_supported(
                                    migrated_texts[candidate.canary_id],
                                    candidate.expected_answer,
                                )
                            )
                        retained = sorted(
                            {
                                d.source_id
                                for d in documents
                                if d.source_type == "sentinel"
                            }
                        )
                        condition = {
                            "dataset": dataset,
                            "fixture": fixture,
                            "retriever": retriever,
                            "seed": seed,
                            "state": state,
                            "retained_topics": retained_topics,
                            "retained_canaries": retained,
                            "num_retained_sentinels": len(retained),
                            "num_retained_clean": len(
                                {
                                    d.source_id
                                    for d in documents
                                    if d.source_type == "clean"
                                }
                            ),
                            "index_mode": index_mode,
                            "top_k": protocol["top_k"],
                        }
                        signature = digest(
                            {
                                "condition": condition,
                                "documents": serialize_documents(documents),
                                "protocol": protocol,
                            }
                        )
                        save(
                            directory / "contexts.json",
                            {
                                "condition": condition,
                                "signature": signature,
                                "contexts": contexts,
                                "diagnostics": diagnostics,
                            },
                        )
                        selection = {}
                        for policy in (
                            "sentinelrag_original",
                            "failure_repair_random",
                            "source_similarity",
                        ):
                            order = policy_ordering(
                                probes,
                                owner_clusters,
                                policy,
                                {},
                                seed=seed,
                                p0=protocol["p0"],
                            )
                            rows = [
                                diagnostics[p.profile.candidate.probe_id] for p in order
                            ]
                            selection[policy] = {
                                key: float(np.mean([r[key] for r in rows]))
                                for key in rows[0]
                            }
                        save(
                            directory / "retrieval_summary.json",
                            {
                                "condition": condition,
                                "all_probes": {
                                    key: float(
                                        np.mean([r[key] for r in diagnostics.values()])
                                    )
                                    for key in next(iter(diagnostics.values()))
                                },
                                "selectors": selection,
                                "proxy_disagreement_count": sum(
                                    r["legacy_whole_document_proxy"]
                                    != r["own_chunk_literal_support"]
                                    for r in diagnostics.values()
                                ),
                            },
                        )
                        jobs.append(str(directory.relative_to(OUT)))
                        print(
                            f"[prepared] {dataset}/{fixture}/{retriever}/{seed}/{state} "
                            f"clean={condition['num_retained_clean']} sentinel={len(retained)}",
                            flush=True,
                        )
    save(OUT / "jobs.json", {"jobs": jobs, "protocol_sha256": file_hash(PROTOCOL)})
    print(f"[prepared] {len(jobs)} actual corpus conditions", flush=True)


class RequestCache:
    def __init__(self, protocol, concurrency, *, output_root=None, readonly_roots=()):
        self.protocol = protocol
        self.output_root = Path(output_root) if output_root is not None else OUT
        self.readonly_roots = tuple(Path(root) for root in readonly_roots)
        names = protocol["target_models"] + [protocol["verifier_model"]]
        self.clients = {
            name: create_llm_client_from_preset(name, models_dir=ROOT / "models")
            for name in names
        }
        self.semaphore = asyncio.Semaphore(concurrency)
        self.model_semaphores = {name: asyncio.Semaphore(4) for name in names}
        self.start_locks = {name: asyncio.Lock() for name in names}
        self.next_start = defaultdict(float)
        self.inflight = {}
        self.new_requests = 0
        self.reused_requests = 0

    async def ask(self, model_name, prompt):
        client, preset = self.clients[model_name]
        key = digest(
            {
                "preset": model_name,
                "model": preset.model,
                "args": preset.llm_arg,
                "prompt": prompt,
            }
        )
        path = self.output_root / "requests" / key[:2] / f"{key}.json"
        for root in (self.output_root, *self.readonly_roots):
            cached = root / "requests" / key[:2] / f"{key}.json"
            if cached.exists():
                self.reused_requests += 1
                return read(cached)
        if key not in self.inflight:
            self.inflight[key] = asyncio.create_task(
                self.request(client, preset, prompt, path, key)
            )
        return await self.inflight[key]

    async def request(self, client, preset, prompt, path, key):
        for attempt in range(6):
            try:
                started = time.monotonic()
                kwargs = {
                    **preset.llm_arg,
                    "timeout": 120,
                    "response_format": {"type": "text"},
                }
                async with self.model_semaphores[preset.preset_name]:
                    async with self.start_locks[preset.preset_name]:
                        await asyncio.sleep(
                            max(
                                0.0,
                                self.next_start[preset.preset_name] - time.monotonic(),
                            )
                        )
                        self.next_start[preset.preset_name] = time.monotonic() + 1.0
                    async with self.semaphore:
                        response = await client.async_client.with_options(
                            max_retries=0
                        ).chat.completions.create(
                            model=preset.model,
                            messages=[{"role": "user", "content": prompt}],
                            **kwargs,
                        )
                content = response.choices[0].message.content
                if not content or response.choices[0].finish_reason == "length":
                    raise ValueError("Empty or truncated model completion")
                record = {
                    "key": key,
                    "model": preset.model,
                    "returned_model": response.model,
                    "prompt": prompt,
                    "content": content,
                    "attempts": attempt + 1,
                    "elapsed_seconds": time.monotonic() - started,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "finish_reason": response.choices[0].finish_reason,
                }
                save(path, record)
                self.new_requests += 1
                if self.new_requests % 50 == 0:
                    print(
                        f"[api] new={self.new_requests} reused={self.reused_requests}",
                        flush=True,
                    )
                return record
            except (APIError, ValueError) as error:
                if attempt == 5:
                    save(
                        self.output_root / "errors" / f"{key}.json",
                        {
                            "model": preset.model,
                            "error_type": type(error).__name__,
                            "status_code": getattr(error, "status_code", None),
                        },
                    )
                    raise RuntimeError(
                        f"{preset.model}: {type(error).__name__}"
                    ) from None
                delay = min(60, 5 * 2**attempt)
                if getattr(error, "status_code", None) == 429:
                    delay = max(30, delay)
                    self.next_start[preset.preset_name] = time.monotonic() + delay
                print(
                    f"[retry] {preset.model} {type(error).__name__}; wait={delay}s",
                    flush=True,
                )
                await asyncio.sleep(delay)


async def run_policy(protocol, requests, job, model_name, probes, clusters, policy):
    directory = OUT / job
    payload = read(directory / "contexts.json")
    path = directory / model_name / f"{policy}.json"
    rows = []
    if path.exists():
        checkpoint = read(path)
        if checkpoint["context_signature"] != payload["signature"]:
            raise ValueError("Stale policy checkpoint")
        rows = checkpoint["records"]
    by_id = {p.profile.candidate.probe_id: p.profile for p in probes}
    condition = payload["condition"]
    while len(rows) < protocol["max_queries_per_policy"]:
        probe = next_probe(
            probes,
            clusters,
            policy,
            rows,
            seed=condition["seed"],
            p0=protocol["p0"],
            prior=protocol["owner_prior_retention"],
        )
        if probe is None:
            break
        candidate = probe.profile.candidate
        context = payload["contexts"][candidate.probe_id]
        answer = await requests.ask(
            model_name,
            PromptTemplates.answer_with_rag(
                "\n".join(row["text"] for row in context), candidate.question
            ),
        )
        if explicit_abstention(answer["content"], candidate.expected_answer):
            verdict = {"content": "No", "key": None}
            judge_kind = "exact_abstention_rule_v1"
        else:
            verdict = await requests.ask(
                protocol["verifier_model"],
                PromptTemplates.judge_answer_correctness(
                    candidate.question, answer["content"], candidate.expected_answer
                ),
            )
            judge_kind = "glm_semantic"
        decision = verdict["content"].strip().casefold()
        if not decision.startswith(("yes", "no")):
            raise ValueError("Unparseable oracle response; do not count as negative")
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
                **payload["diagnostics"][candidate.probe_id],
            }
        )
        evaluation = evaluate_ordering(
            [by_id[r["probe_id"]] for r in rows],
            {r["probe_id"]: r["is_correct"] for r in rows},
            p0=protocol["p0"],
            alpha=protocol["alpha"],
        )
        save(
            path,
            {
                "condition": condition,
                "model": model_name,
                "policy": policy,
                "context_signature": payload["signature"],
                "p0": protocol["p0"],
                "complete": len(rows) == protocol["max_queries_per_policy"],
                "records": rows,
                "evaluation": evaluation,
            },
        )
    print(f"[complete] {job}/{model_name}/{policy}", flush=True)


async def execute(concurrency):
    protocol = read(OUT / "frozen_protocol.json")["protocol"]
    if protocol != read(PROTOCOL):
        raise ValueError("Frozen protocol mismatch")
    requests = RequestCache(protocol, concurrency)
    # Check endpoint connectivity before scheduling experimental requests.
    for name, (client, preset) in requests.clients.items():
        try:
            response = await client.async_client.with_options(
                max_retries=0
            ).chat.completions.create(
                model=preset.model,
                messages=[{"role": "user", "content": "Reply only OK."}],
                timeout=30,
                max_tokens=64,
            )
            print(
                f"[preflight] {name}: {response.choices[0].message.content!r}",
                flush=True,
            )
        except APIError as error:
            save(
                OUT / "execution_status.json",
                {
                    "status": "blocked_preflight",
                    "model": name,
                    "error_type": type(error).__name__,
                    "status_code": getattr(error, "status_code", None),
                    "new_experimental_requests": 0,
                },
            )
            raise RuntimeError(
                f"Endpoint preflight failed: {name} {type(error).__name__}"
            ) from None
    save(
        OUT / "execution_status.json", {"status": "running", "concurrency": concurrency}
    )
    cache = {}
    for dataset in protocol["datasets"]:
        clusters = read(OUT / dataset / "partitions.json")["owner_clusters"]
        for fixture in protocol["fixtures"]:
            _, probes = owner_probes(dataset, fixture, protocol["top_k"])
            cache[(dataset, fixture)] = (probes, clusters)
    limiter = asyncio.Semaphore(4)

    async def group(job, model):
        async with limiter:
            parts = Path(job).parts
            probes, clusters = cache[(parts[0], parts[1])]
            await asyncio.gather(
                *(
                    run_policy(protocol, requests, job, model, probes, clusters, policy)
                    for policy in protocol["policies"]
                )
            )

    await asyncio.gather(
        *(
            group(job, model)
            for job in read(OUT / "jobs.json")["jobs"]
            for model in protocol["target_models"]
        )
    )
    save(
        OUT / "execution_status.json",
        {
            "status": "complete",
            "new_requests": requests.new_requests,
            "reused_requests": requests.reused_requests,
        },
    )


def report():
    protocol = read(OUT / "frozen_protocol.json")["protocol"]
    jobs = read(OUT / "jobs.json")["jobs"]
    rows = []
    grouped = defaultdict(list)
    for job in jobs:
        summary = read(OUT / job / "retrieval_summary.json")
        condition = summary["condition"]
        if condition["state"] == "watermarked":
            grouped[
                (condition["dataset"], condition["retriever"], condition["fixture"])
            ].append(summary)
        for model in protocol["target_models"]:
            for policy in protocol["policies"]:
                path = OUT / job / model / f"{policy}.json"
                if path.exists() and read(path).get("complete"):
                    rows.append(read(path))
    expected = len(jobs) * len(protocol["target_models"]) * len(protocol["policies"])
    unchanged = all(
        file_hash(ROOT / name) == value
        for name, value in read(OUT / "frozen_protocol.json")["paper_sha256"].items()
    )
    save(
        OUT / "result_inventory.json",
        {
            "completed_policy_traces": len(rows),
            "expected_policy_traces": expected,
            "paper_unchanged": unchanged,
        },
    )
    text = [
        "# 2026-09-06 分区迁移补充实验",
        "",
        f"真实语料/检索条件：{len(jobs)}；完整 target-policy traces：{len(rows)}/{expected}。",
        f"论文 Markdown/LaTeX 哈希未改变：{unchanged}。",
        "",
        "## 设置与边界",
        "",
        "原始/关系显式化 fixture 使用相同问题、答案、普通文档和独立分区。每个实际索引保留 8 个 clean-corpus topics 中的 4 个。BM25 重建；BGE-M3 固定编码器按真实子集检索。",
        "owner 的 6 个 sentinel clusters 与迁移的 8 个 clean-corpus topics 分开构建；相同分区应用到两种 fixture。未筛掉不利分区。",
        "阈值在本次 target 结果前固定为 p0=.15、alpha=.01；每策略最多 10 次，重点比较预算 4/6。没有达到阈值的 trace 保留为未检出，不当作第 10 次成功。",
        "这些是已有数据集上的 post-review confirmation，不是全新数据集 holdout；3 个保留种子不足以推断广泛部署总体。",
        "SentinelRAG-style 使用原始 fixture、首 probe 和随机顺序，统一使用本实验验证器；不等同于原文全部设置的严格复跑。",
        "主判定由 GLM-5.1 完成；运行中启用精确 Cannot answer 的确定性 No 规则，启用前的 516 个此类回答均获 GLM 的 No。其他回答继续由 GLM 判断。该 operational amendment 不改变 target 调用、阈值、候选或比较范围，也不将节省的 judge 调用计为 probe 效率收益。",
        "",
        "## 真实分区",
        "",
        "| 数据集 | 种子 | 保留普通文档 | 保留 sentinel | Owner/迁移分区 ARI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in protocol["datasets"]:
        partition = read(OUT / dataset / "partitions.json")
        for seed in protocol["migration_seeds"]:
            c = read(
                OUT
                / dataset
                / "original/bm25"
                / str(seed)
                / "watermarked/contexts.json"
            )["condition"]
            text.append(
                f"| {dataset} | {seed} | {c['num_retained_clean']} | {c['num_retained_sentinels']} | "
                f"{partition['adjusted_rand_index_sentinels']:.3f} |"
            )
    text += [
        "",
        "## 实际上下文检索结果",
        "",
        "以下是检索/字面支持，不是 target LLM 成功率。普通文档和 sentinel 均已实际按分区过滤。",
        "",
        "| 数据集 | Retriever | Fixture | 首 probe 命中 | 首 probe chunk 字面支持 | Repair 命中 | Repair chunk 字面支持 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for (dataset, retriever, fixture), group in sorted(grouped.items()):
        values = [
            np.mean([r["selectors"][policy][key] for r in group])
            for policy in ("sentinelrag_original", "failure_repair_random")
            for key in ("retrieved_sentinel", "own_chunk_literal_support")
        ]
        text.append(
            f"| {dataset} | {retriever} | {fixture} | "
            + " | ".join(f"{v:.3f}" for v in values)
            + " |"
        )
    text += ["", "## Target 端到端结果", ""]
    if not rows:
        text.append(
            "尚无完成的新增 target trace。不可把上面的检索结果写成模型回答或端到端收益。"
        )
    else:
        text += [
            "| Fixture | Policy | 完成正控 traces | VSR@4 | VSR@6 | VSR@10 | Anytime 检出@10 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        groups = defaultdict(list)
        for row in rows:
            if row["condition"]["state"] == "watermarked":
                groups[(row["condition"]["fixture"], row["policy"])].append(row)
        for (fixture, policy), group in sorted(groups.items()):
            rates = [
                np.mean([r["evaluation"]["curve"][b - 1]["significant"] for r in group])
                for b in (4, 6, 10)
            ]
            anytime = np.mean(
                [
                    r["evaluation"]["queries_to_anytime_significance"] is not None
                    for r in group
                ]
            )
            text.append(
                f"| {fixture} | {policy} | {len(group)} | "
                + " | ".join(f"{v:.3f}" for v in rates + [anytime])
                + " |"
            )
        negative = [r for r in rows if r["condition"]["state"] == "clean"]
        text += [
            "",
            f"已完成 clean traces：{len(negative)}。它们共享语料和部分请求缓存，不能当作全部独立统计样本。",
        ]
        for budget in (4, 6, 10):
            hits = sum(
                r["evaluation"]["curve"][budget - 1]["significant"] for r in negative
            )
            text.append(
                f"- Clean fixed-budget {budget}：{hits}/{len(negative)} 次拒绝 H0。"
            )
        text.append(
            "- 校准阈值未使用本次 clean 输出更新；该负控不自动证明每步自适应条件 null bound。"
        )
    text += [
        "",
        "## 数据与执行",
        "",
        "- frozen_protocol.json：预声明设置及论文文件哈希。",
        "- 每个 corpus condition 的 contexts.json：真实 top-k 上下文及索引构造记录。",
        "- retrieval_summary.json：新 chunk literal 指标与旧整篇文档代理的差异。",
        "- requests/：按模型配置和实际 prompt 哈希归档响应、用量和延迟，不保存 API key。",
        "- 每个模型/策略 JSON：仅依据该策略已观察前缀执行，逐次 checkpoint。",
        "",
    ]
    (OUT / "report.zh-CN.md").write_text("\n".join(text), encoding="utf-8")
    print(OUT / "report.zh-CN.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "run":
        asyncio.run(execute(args.concurrency))
    else:
        report()
