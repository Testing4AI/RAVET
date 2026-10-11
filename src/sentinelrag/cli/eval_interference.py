#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI entry point for Watermark Interference Evaluation (Retrieval Interference and Answer Interference)

Usage:
python -m sentinelrag.cli.eval_interference --injection_result_path ./output/watermark_injections/injection_*/injection_result.json --eval_model_code contriever --eval_dataset nfcorpus
"""

import argparse
import os
import random
from datetime import datetime

import torch
from dotenv import load_dotenv

from sentinelrag.utils import (
    Log,
    add_model_preset_arg,
    create_llm_client_from_preset,
    file_exist,
    save_json,
    load_json,
    find_latest_injection_result,
)
from sentinelrag.core import (
    InterferenceEvaluator,
    setup_vectorstore,
    load_main_questions,
)

# Load environment variables
load_dotenv()

# --- Configuration ---
LLM_PRESET = "gpt-5-mini"
EVAL_LLM_KWARGS = {}

RLLM_PRESET = "gpt-5-mini"
RAG_LLM_KWARGS = {}


# Default seed
DEFAULT_SEED = 633


def load_injection_result(injection_result_path):
    """Load injection results from file"""
    if not os.path.exists(injection_result_path):
        raise FileNotFoundError(f"Injection result file does not exist: {injection_result_path}")
    
    injection_result = load_json(injection_result_path)
    print("Injection results loaded successfully")
    return injection_result


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate retrieval interference and answer interference after watermark injection")
    add_model_preset_arg(parser, flag='llm', default_preset=LLM_PRESET, help_text='Answer-evaluation LLM preset')
    add_model_preset_arg(parser, flag='rllm', default_preset=RLLM_PRESET, help_text='RAG answer-generation LLM preset')
    parser.add_argument("--injection_result_path", type=str, default=None, help="Path to injection_result.json")
    parser.add_argument(
        "--eval_model_code",
        type=str,
        default='contriever',
        choices=["contriever", "contriever-msmarco", "ance"],
        help="Retrieval model code",
    )
    parser.add_argument('--num_select_kos', type=int, default=None, 
                        help='Number of KOs (used for auto-selecting injection result)')
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default=None,
        help="Dataset name (defaults to injection metadata)",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--score_function",
        type=str,
        default="cosine",
        choices=["cosine", "l2", "ip"],
        help="Similarity calculation function",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--num_questions", type=int, default=50, help="Number of main-task questions to sample (0 for all questions)")
    parser.add_argument("--gpu_id", type=int, default=1, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = f"cuda:{args.gpu_id}"
    else:
        device = "cpu"
        print("CUDA is not available; using CPU.")

    # Auto-select injection result if not provided
    if args.injection_result_path is None:
        if args.eval_dataset and args.num_select_kos:
            try:
                # args.output_dir default is "./output", which matches common basepath
                print(f"Auto-selecting injection result for {args.eval_dataset} k={args.num_select_kos}...")
                args.injection_result_path = find_latest_injection_result(args.output_dir, args.eval_dataset, args.num_select_kos)
            except Exception as e:
                raise ValueError(f"Could not auto-select injection result: {e}")
        else:
            raise ValueError("Either --injection_result_path OR (--eval_dataset AND --num_select_kos) must be provided.")

    injection_result = load_injection_result(args.injection_result_path)
    if args.eval_dataset is None:
        args.eval_dataset = injection_result["injection_metadata"]["ko_pool_source"].get("dataset_used")
    if args.eval_dataset is None:
        raise ValueError("Cannot infer dataset from injection results, please set --eval_dataset explicitly")

    num_ko = injection_result["injection_metadata"].get("num_selected_kos", "unknown")
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, "watermark_interference", args.eval_dataset, f"k{num_ko}", base_timestamp)
    log_file = os.path.join(output_dir, "interference.log")
    file_exist(log_file)
    logger = Log(log_file=log_file).get(__file__)

    print("=" * 60)
    print("Retrieval Interference and Answer Interference Evaluation")
    print("=" * 60)
    print(f"Dataset: {args.eval_dataset}")
    print(f"Top-k: {args.top_k}")
    print(f"Questions sampled: {args.num_questions}")

    # Setup two separate LLM clients
    rag_llm_client, rllm_preset = create_llm_client_from_preset(args.rllm, include_async=False)
    eval_llm_client, llm_preset = create_llm_client_from_preset(args.llm, include_async=False)
    logger.info(f"RAG LLM preset: {rllm_preset.preset_name} -> {rllm_preset.model} ({rllm_preset.llm_url})")
    logger.info(f"Eval LLM preset: {llm_preset.preset_name} -> {llm_preset.model} ({llm_preset.llm_url})")
    vectorstore, collection_name, collection_len = setup_vectorstore(
        eval_dataset=args.eval_dataset,
        eval_model_code=args.eval_model_code,
        score_function=args.score_function,
        split=args.split,
        gpu_id=args.gpu_id,
        device=device,
    )
    logger.info(f"Collection: {collection_name} (len={collection_len})")


    questions = load_main_questions(args.eval_dataset, args.split, args.num_questions, args.seed)
    if not questions:
        raise ValueError("No questions available for evaluation")

    # Create evaluator with separate LLM clients for RAG and evaluation
    evaluator = InterferenceEvaluator(
        rag_llm_client, eval_llm_client, vectorstore, args.top_k,
        collection_len=collection_len,
        rag_llm_kwargs=RAG_LLM_KWARGS, eval_llm_kwargs=EVAL_LLM_KWARGS
    )

    # Phase 1: Clean retrieval only (no answer generation yet)
    print("\n" + "=" * 60)
    clean_retrievals, clean_lookup = evaluator.evaluate_fully_optimized(questions, None)

    # Inject watermarks
    watermark_texts = injection_result.get("watermark_texts", [])
    watermark_ids = evaluator.inject_watermarks(watermark_texts)

    # Phase 2-4: Watermarked retrieval, comparison, and strategic answer generation
    clean_runs, watermarked_runs = evaluator.complete_evaluation_after_injection(
        questions, clean_retrievals, watermark_ids
    )
    
    def format_run_for_output(run):
        """Format a run for JSON output: remove doc_ids and rag_document, convert retrieved_documents to dict"""
        doc_ids = run.get("doc_ids") or []
        retrieved_docs = run.get("retrieved_documents") or []
        # Build dict with id as key and doc as value
        documents_dict = {}
        for doc_id, doc in zip(doc_ids, retrieved_docs):
            documents_dict[doc_id] = doc
        
        return {
            "question": run.get("question"),
            "question_id": run.get("question_id"),
            "distances": run.get("distances"),
            "retrieved_documents": documents_dict,
            "answer": run.get("answer"),
            "watermark_rank": run.get("watermark_rank"),
            "watermark_distance": run.get("watermark_distance"),
        }
    
    # Save final runs with answers (formatted)
    formatted_clean_runs = [format_run_for_output(run) for run in clean_runs]
    clean_runs_path = os.path.join(output_dir, "clean_runs.json")
    save_json(formatted_clean_runs, clean_runs_path)
    logger.info(f"Clean runs saved to: {clean_runs_path}")
    
    formatted_watermarked_runs = [format_run_for_output(run) for run in watermarked_runs]
    watermarked_runs_path = os.path.join(output_dir, "watermarked_runs.json")
    save_json(formatted_watermarked_runs, watermarked_runs_path)
    logger.info(f"Watermarked runs saved to: {watermarked_runs_path}")

    answer_interference, retrieval_interference, details = evaluator.compute_interference(clean_runs, watermarked_runs, watermark_ids)
    print(f"\nAnswer Interference: {answer_interference:.3f}")
    print(f"Retrieval Interference: {retrieval_interference:.3f}")
    
    # Count skipped answer checks
    skipped_count = sum(1 for d in details if d.get("answer_check_skipped", False))
    if skipped_count > 0:
        print(f"Answer checks skipped (consistent retrieval, no watermarks): {skipped_count}/{len(details)}")

    if watermark_ids:
        success = evaluator.cleanup_watermarks(watermark_ids)
        if not success:
            logger.error("Failed to clean up watermarks")

    question_ids = [qid for qid, _ in questions]
    clean_lookup = {run.get("question_id"): run for run in clean_runs}
    watermarked_lookup = {run.get("question_id"): run for run in watermarked_runs}

    def retrieved_texts(run):
        docs = run.get("retrieved_documents") or []
        if docs:
            return docs
        rag_doc = run.get("rag_document") or ""
        return [segment for segment in rag_doc.split("\n") if segment.strip()]

    different_responses = []
    different_retrievals = []

    for detail in details:
        qid = detail.get("question_id")
        clean_run = clean_lookup.get(qid, {})
        wm_run = watermarked_lookup.get(qid, {})
        question_text = clean_run.get("question") or detail.get("question")

        if detail.get("answer_interfered"):
            different_responses.append(
                {
                    "question_id": qid,
                    "question": question_text,
                    "clean_answer": clean_run.get("answer", ""),
                    "watermarked_answer": wm_run.get("answer", ""),
                    "watermark": None,
                    "clean_retrieved_text": retrieved_texts(clean_run),
                    "watermarked_retrieved_text": retrieved_texts(wm_run),
                }
            )

        if detail.get("retrieval_interfered"):
            different_retrievals.append(
                {
                    "question_id": qid,
                    "question": question_text,
                    "clean_doc_ids": clean_run.get("doc_ids", []),
                    "watermarked_doc_ids": wm_run.get("doc_ids", []),
                    "clean_retrieved_text": retrieved_texts(clean_run),
                    "watermarked_retrieved_text": retrieved_texts(wm_run),
                    "watermark": None,
                }
            )

    matching_answers = len(details) - len(different_responses)
    matching_retrievals = len(details) - len(different_retrievals)
    answer_consistency = 1 - answer_interference
    retrieval_consistency = 1 - retrieval_interference

    results = {
        "metrics": {
            "answer_interference": answer_interference,
            "retrieval_interference": retrieval_interference,
            "answer_consistency": answer_consistency,
            "retrieval_consistency": retrieval_consistency,
        },
        "answer_interference": {
            "score": answer_interference,
            "matching_answers": matching_answers,
            "evaluated_questions": len(details),
            "interfered_responses": different_responses,
            "num_interfered_responses": len(different_responses),
        },
        "retrieval_interference": {
            "score": retrieval_interference,
            "matching_retrievals": matching_retrievals,
            "evaluated_questions": len(details),
            "interfered_retrievals": different_retrievals,
            "num_interfered_retrievals": len(different_retrievals),
        },
        "overall": {
            "watermarked_answers": len(watermarked_runs),
            "clean_answers": len(clean_runs),
            "watermarked_retrievals": len(watermarked_runs),
            "clean_retrievals": len(clean_runs),
            "question_ids": question_ids,
        },
    }

    result_path = os.path.join(output_dir, "interference_results.json")
    save_json(results, result_path)
    logger.info(f"Results saved to: {result_path}")
    print(f"\nResults saved to {result_path}")
    print(f"Log file: {log_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
