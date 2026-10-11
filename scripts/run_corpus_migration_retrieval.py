#!/usr/bin/env python3
"""Build reusable retrieval caches for uniformly migrated RAG corpora."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sentinelrag.cli.detect_watermark import setup_vectorstore
from sentinelrag.testing.artifacts import build_artifact_pools
from sentinelrag.testing.corpus_migrations import (
    load_corpus_migration_manifest,
    validate_uniform_manifest,
)
from sentinelrag.testing.corpus_retrieval import (
    IndexedMigrationDocument,
    target_outcomes,
)
from sentinelrag.testing.matrix import (
    PhysicalDocument,
    RetrievalScoreCache,
    VariantScores,
    _cosine_matrix,
    observe_candidate,
    sparse_bm25_scores,
)
from sentinelrag.testing.prioritization import (
    ProbeProfile,
    answer_tokens_supported,
    evaluate_ordering,
    rank_probes,
    score_profiles,
)
from sentinelrag.utils.datasets import load_beir_datasets
from sentinelrag.utils.io import save_json

BUDGETS = (2, 4, 6, 8, 10)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus-subset", help="Immutable ordinary-document KB block")
    parser.add_argument("--injection-result", required=True)
    parser.add_argument("--migration-manifest", required=True)
    parser.add_argument("--matrix-root", default="output/overnight-matrix")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--retrievers", nargs="+", default=["contriever", "bm25"])
    parser.add_argument(
        "--source-retrievers",
        nargs="+",
        help="Owner-side retrievers used to construct frozen probe profiles.",
    )
    parser.add_argument(
        "--target-retrievers",
        nargs="+",
        help="Deployment retrievers used to score the migrated corpus.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Optional subset of migration variants to score.",
    )
    parser.add_argument(
        "--bge-model",
        default="BAAI/bge-m3",
        help="Sentence-Transformers model used by the bge-m3 target retriever.",
    )
    parser.add_argument(
        "--bge-revision",
        default="5617a9f61b028005a4858fdac845db406aefb181",
        help="Pinned Hugging Face revision for the BGE-M3 model.",
    )
    parser.add_argument(
        "--dense-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-clean-docs",
        type=int,
        help="Use the same deterministic clean-corpus prefix as a sliced migration manifest.",
    )
    return parser.parse_args()


class SentenceTransformerEncoder:
    """Lazy normalized dense encoder for modern retrieval extensions."""

    def __init__(self, model_name, revision, device):
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.model = SentenceTransformer(model_name, revision=revision, device=device)

    def encode(self, texts, batch_size):
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )


def load_owner_cache(path: Path, artifacts) -> RetrievalScoreCache:
    metadata = json.loads(
        (path / "retrieval_cache_metadata.json").read_text(encoding="utf-8")
    )
    arrays = np.load(path / "retrieval_scores.npz")
    candidates = [probe for artifact in artifacts for probe in artifact.probes]
    if [candidate.probe_id for candidate in candidates] != metadata["candidate_ids"]:
        raise ValueError(f"Candidate mismatch in owner cache {path}")
    variants = {
        name: VariantScores(
            variant=name,
            documents=[PhysicalDocument(**row) for row in details["documents"]],
            clean_top_scores=arrays[f"{name}__clean"],
            canary_scores=arrays[f"{name}__canary"],
        )
        for name, details in metadata["variants"].items()
    }
    return RetrievalScoreCache(
        candidates=candidates,
        variants=variants,
        clean_depth=int(metadata["clean_depth"]),
        retriever=metadata["retriever"],
    )


def profiles_for_cache(cache, canary_ids):
    profiles = []
    for index, candidate in enumerate(cache.candidates):
        observations = [
            observe_candidate(
                cache,
                candidate_index=index,
                selected_canary_ids=canary_ids,
                variant_name=variant,
            )
            for variant in (
                "original",
                "reverse_sentences",
                "boilerplate",
                "paired_chunks",
            )
        ]
        profiles.append(ProbeProfile(candidate, observations))
    score_profiles(profiles)
    return profiles


def first_probe_ordering(profiles, seed):
    first = {}
    for profile in profiles:
        index = int(profile.candidate.metadata.get("question_index", 0))
        incumbent = first.get(profile.candidate.canary_id)
        if incumbent is None or (
            index,
            profile.candidate.probe_id,
        ) < (
            int(incumbent.candidate.metadata.get("question_index", 0)),
            incumbent.candidate.probe_id,
        ):
            first[profile.candidate.canary_id] = profile
    ordering = list(first.values())
    random.Random(seed).shuffle(ordering)
    return ordering


def selected_random_ordering(profiles, seed):
    ordering = list(rank_probes(profiles, "proposed"))
    random.Random(seed).shuffle(ordering)
    return ordering


def flatten_variant(records, variant):
    documents = []
    for record in records:
        for chunk_index, text in enumerate(record.variants[variant]):
            documents.append(
                IndexedMigrationDocument(
                    record.source_type,
                    record.source_id,
                    chunk_index,
                    text,
                )
            )
    return documents


def batch_embeddings(vectorstore, texts, batch_size):
    rows = []
    for start in range(0, len(texts), batch_size):
        rows.extend(vectorstore.get_embeddings_batch(texts[start : start + batch_size]))
        if start and start % (batch_size * 20) == 0:
            print(f"embedded {start}/{len(texts)}", flush=True)
    return np.asarray(rows, dtype=np.float32)


def cached_scores(
    *,
    retriever,
    variant,
    documents,
    questions,
    vectorstore,
    dense_encoder,
    output_dir,
    embedding_batch_size,
):
    score_path = output_dir / f"scores-{retriever}-{variant}.npz"
    if score_path.exists():
        return np.load(score_path)["scores"]
    texts = [document.text for document in documents]
    if retriever == "contriever":
        query_embeddings = batch_embeddings(
            vectorstore, questions, embedding_batch_size
        )
        document_embeddings = batch_embeddings(vectorstore, texts, embedding_batch_size)
        scores = _cosine_matrix(query_embeddings, document_embeddings)
    elif retriever == "bm25":
        scores = sparse_bm25_scores(texts, questions).toarray().astype(np.float32)
    elif retriever == "bge-m3":
        if dense_encoder is None:
            raise ValueError("bge-m3 requires a dense encoder")
        query_embeddings = dense_encoder.encode(questions, embedding_batch_size)
        document_embeddings = dense_encoder.encode(texts, embedding_batch_size)
        scores = _cosine_matrix(query_embeddings, document_embeddings)
    else:
        raise ValueError(f"Unknown retriever: {retriever}")
    np.savez_compressed(score_path, scores=scores)
    return scores


def oracle_ordering(profiles, outcome_by_probe):
    selected = {}
    for profile in profiles:
        canary_id = profile.candidate.canary_id
        incumbent = selected.get(canary_id)
        key = (
            outcome_by_probe[profile.candidate.probe_id],
            profile.proposed_score,
            profile.candidate.probe_id,
        )
        if incumbent is None or key > (
            outcome_by_probe[incumbent.candidate.probe_id],
            incumbent.proposed_score,
            incumbent.candidate.probe_id,
        ):
            selected[canary_id] = profile
    return sorted(
        selected.values(),
        key=lambda row: (
            outcome_by_probe[row.candidate.probe_id],
            row.proposed_score,
            row.candidate.probe_id,
        ),
        reverse=True,
    )


def summarize_ordering(ordering, outcome_by_probe, support_by_probe=None):
    result = evaluate_ordering(ordering, outcome_by_probe)
    curve = {row["budget"]: row for row in result["curve"]}
    payload = {
        "hit_rate": result["final_detection_rate"],
        "queries_to_significance_descriptive": result["queries_to_significance"],
        "queries_to_anytime_significance": result["queries_to_anytime_significance"],
        "successes_at_budget": {
            str(budget): curve[budget]["cumulative_successes"]
            for budget in BUDGETS
            if budget in curve
        },
        "probe_ids": [row.candidate.probe_id for row in ordering],
        "probe_outcomes": [
            {
                "probe_id": row.candidate.probe_id,
                "question_index": int(row.candidate.metadata.get("question_index", 0)),
                "shadow_features": row.features(),
                "retrieval_hit": bool(outcome_by_probe[row.candidate.probe_id]),
                "literal_support": (
                    bool(support_by_probe[row.candidate.probe_id])
                    if support_by_probe is not None
                    else None
                ),
                "strict_support": (
                    bool(
                        outcome_by_probe[row.candidate.probe_id]
                        and support_by_probe[row.candidate.probe_id]
                    )
                    if support_by_probe is not None
                    else None
                ),
            }
            for row in ordering
        ],
    }
    if support_by_probe is not None:
        supported_outcomes = {
            probe_id: bool(outcome_by_probe[probe_id] and support_by_probe[probe_id])
            for probe_id in outcome_by_probe
        }
        supported = evaluate_ordering(ordering, supported_outcomes)
        supported_curve = {row["budget"]: row for row in supported["curve"]}
        payload.update(
            {
                "literal_answer_support_rate": sum(
                    bool(support_by_probe[row.candidate.probe_id]) for row in ordering
                )
                / len(ordering),
                "retrieval_and_literal_support_rate": supported["final_detection_rate"],
                "retrieval_and_literal_support_at_budget": {
                    str(budget): supported_curve[budget]["cumulative_successes"]
                    for budget in BUDGETS
                    if budget in supported_curve
                },
            }
        )
    return payload


def main():
    args = parse_args()
    source_retrievers = args.source_retrievers or args.retrievers
    target_retrievers = args.target_retrievers or args.retrievers
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = build_artifact_pools(args.injection_result)["factual"]
    candidates = [probe for artifact in artifacts for probe in artifact.probes]
    candidate_ids = [candidate.probe_id for candidate in candidates]
    canary_ids = {artifact.canary_id for artifact in artifacts}

    if args.corpus_subset:
        from sentinelrag.testing.corpus_subset import load_corpus_subset

        corpus = load_corpus_subset(args.corpus_subset, dataset=args.dataset)
    else:
        corpus, _, _ = load_beir_datasets(args.dataset, "test")
    clean_rows = [
        (str(document_id), str(item["text"]))
        for document_id, item in sorted(corpus.items())
    ]
    if args.max_clean_docs is not None:
        clean_rows = clean_rows[: args.max_clean_docs]
    clean_sources = dict(clean_rows)
    sentinel_sources = {
        artifact.canary_id: "\n".join(artifact.documents) for artifact in artifacts
    }
    records = load_corpus_migration_manifest(args.migration_manifest)
    validate_uniform_manifest(records, clean_sources, sentinel_sources)
    variants = sorted(set.intersection(*(set(row.variants) for row in records)))
    sentinel_records = {
        record.source_id: record
        for record in records
        if record.source_type == "sentinel"
    }

    owner_profiles = {}
    for retriever in source_retrievers:
        owner_cache = load_owner_cache(
            Path(args.matrix_root) / args.dataset / retriever / "factual",
            artifacts,
        )
        owner_profiles[retriever] = profiles_for_cache(owner_cache, canary_ids)

    setup_args = SimpleNamespace(
        eval_dataset=args.dataset,
        eval_model_code="contriever",
        score_function="cosine",
        split="test",
        gpu_id=0,
    )
    vectorstore = None
    if "contriever" in target_retrievers:
        vectorstore, _, _ = setup_vectorstore(setup_args)
    dense_encoder = None
    if "bge-m3" in target_retrievers:
        dense_encoder = SentenceTransformerEncoder(
            args.bge_model, args.bge_revision, args.dense_device
        )
    questions = [candidate.question for candidate in candidates]
    conditions = []
    cache_manifest = {
        "protocol_version": "2026-08-20-corpus-migration-retrieval-v1",
        "dataset": args.dataset,
        "clean_document_count": len(clean_sources),
        "max_clean_docs": args.max_clean_docs,
        "candidate_ids": candidate_ids,
        "canary_by_probe": [candidate.canary_id for candidate in candidates],
        "variants": {},
    }
    if args.variants:
        unknown_variants = sorted(set(args.variants) - set(variants))
        if unknown_variants:
            raise ValueError(f"Unknown migration variants: {unknown_variants}")
        variants = args.variants
    for variant in variants:
        documents = flatten_variant(records, variant)
        support_by_probe = {
            candidate.probe_id: answer_tokens_supported(
                "\n".join(sentinel_records[candidate.canary_id].variants[variant]),
                candidate.expected_answer,
            )
            for candidate in candidates
        }
        document_path = output_dir / f"documents-{variant}.json"
        save_json([asdict(document) for document in documents], document_path)
        cache_manifest["variants"][variant] = {
            "documents": str(document_path.resolve()),
            "document_count": len(documents),
        }
        for target_retriever in target_retrievers:
            print(f"[{variant}/{target_retriever}] scoring", flush=True)
            scores = cached_scores(
                retriever=target_retriever,
                variant=variant,
                documents=documents,
                questions=questions,
                vectorstore=vectorstore,
                dense_encoder=dense_encoder,
                output_dir=output_dir,
                embedding_batch_size=args.embedding_batch_size,
            )
            for top_k in args.top_k:
                outcomes_index = target_outcomes(
                    scores,
                    documents,
                    [candidate.canary_id for candidate in candidates],
                    top_k=top_k,
                )
                outcome_by_probe = {
                    candidate.probe_id: outcomes_index[index]
                    for index, candidate in enumerate(candidates)
                }
                for source_retriever, profiles in owner_profiles.items():
                    orderings = {
                        "sentinelrag_original": first_probe_ordering(
                            profiles, seed=20260820
                        ),
                        "random_candidate": rank_probes(
                            profiles, "random", seed=20260820
                        ),
                        "selected_random": selected_random_ordering(
                            profiles, seed=20260820
                        ),
                        "source_similarity": rank_probes(profiles, "similarity"),
                        "source_composite": rank_probes(profiles, "proposed"),
                        "failure_repair": rank_probes(profiles, "failure_repair"),
                    }
                    orderings["candidate_oracle"] = oracle_ordering(
                        profiles, outcome_by_probe
                    )
                    conditions.append(
                        {
                            "variant": variant,
                            "source_retriever": source_retriever,
                            "target_retriever": target_retriever,
                            "top_k": top_k,
                            "methods": {
                                method: summarize_ordering(
                                    ordering,
                                    outcome_by_probe,
                                    support_by_probe,
                                )
                                for method, ordering in orderings.items()
                            },
                        }
                    )
    save_json(cache_manifest, output_dir / "retrieval_cache_manifest.json")
    save_json(
        {
            "protocol_version": cache_manifest["protocol_version"],
            "dataset": args.dataset,
            "clean_document_count": len(clean_sources),
            "max_clean_docs": args.max_clean_docs,
            "pool_size": len(artifacts),
            "candidate_count": len(candidates),
            "variants": variants,
            "retrievers": sorted(set(source_retrievers + target_retrievers)),
            "source_retrievers": source_retrievers,
            "target_retrievers": target_retrievers,
            "dense_model": (
                {
                    "name": dense_encoder.model_name,
                    "revision": dense_encoder.revision,
                    "device": dense_encoder.device,
                    "normalized": True,
                }
                if dense_encoder is not None
                else None
            ),
            "top_k": args.top_k,
            "conditions": conditions,
        },
        output_dir / "corpus_migration_retrieval_results.json",
    )
    print(output_dir / "corpus_migration_retrieval_results.json", flush=True)


if __name__ == "__main__":
    main()
