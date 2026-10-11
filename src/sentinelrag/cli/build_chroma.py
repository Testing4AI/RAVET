#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a ChromaDB collection for a supported RAG dataset."""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset

from sentinelrag.rag import VectorStore, check_collection
from sentinelrag.utils import load_beir_datasets, load_models
from sentinelrag.utils.paths import default_chroma_dir


def _load_corpus(eval_dataset: str, split: str):
    if eval_dataset == "closed_qa":
        train_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        closed_qa_dataset = train_dataset.filter(lambda example: example["category"] == "closed_qa")
        corpus = {}
        for i, item in enumerate(closed_qa_dataset):
            corpus[str(i)] = {
                "text": f"{item['instruction']}. {item['context']}",
                "title": f"QA_{i}",
            }
        return corpus

    dataset_split = "train" if eval_dataset in {"msmarco", "msmarco_200k"} else split
    corpus, _, _ = load_beir_datasets(eval_dataset, dataset_split)
    return corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SentinelRAG ChromaDB vector collection.")
    parser.add_argument("--eval_dataset", type=str, default="nfcorpus", help="Dataset name")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "dev"])
    parser.add_argument(
        "--eval_model_code",
        type=str,
        default="contriever",
        choices=["contriever", "contriever-msmarco", "ance"],
        help="Embedding/retrieval model",
    )
    parser.add_argument(
        "--score_function",
        type=str,
        default="cosine",
        choices=["cosine", "l2", "ip"],
        help="ChromaDB distance metric",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Embedding batch size")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use when CUDA is available")
    parser.add_argument("--chroma_dir", type=str, default=str(default_chroma_dir()), help="ChromaDB directory")
    parser.add_argument("--force", action="store_true", help="Recreate the collection if it already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = f"cuda:{args.gpu_id}"

    collection_name = f"{args.eval_dataset}_{args.eval_model_code}_{args.score_function}"
    corpus = _load_corpus(args.eval_dataset, args.split)
    expected_size = len(corpus)

    exists, current_size = check_collection(collection_name, chroma_path=args.chroma_dir)
    if exists and current_size == expected_size and not args.force:
        print(f"Collection {collection_name} already has {current_size} documents at {args.chroma_dir}.")
        return

    print(f"Loading embedding model: {args.eval_model_code}")
    model, _, tokenizer, get_emb = load_models(args.eval_model_code)

    use_local = exists and not args.force
    vectorstore = VectorStore(
        model,
        tokenizer,
        get_emb,
        corpus,
        device,
        collection_name,
        use_local=use_local,
        distance=args.score_function,
        chroma_path=args.chroma_dir,
    )
    vectorstore.populate_vectors(batch_size=args.batch_size)

    final_count = vectorstore.collection.count()
    print(f"Collection {collection_name} ready at {args.chroma_dir} ({final_count} documents).")


if __name__ == "__main__":
    main()
