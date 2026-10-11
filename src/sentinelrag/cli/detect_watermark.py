#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI entry point for Watermark Detection

Usage:
python -m sentinelrag.cli.detect_watermark --injection_result_path ./output/watermark_injections/injection_*/injection_result.json --eval_model_code contriever
"""

import argparse
import os
import random
import re
from datetime import datetime

import numpy as np
import torch
from dotenv import load_dotenv
from datasets import load_dataset

from sentinelrag.utils import (
    add_model_preset_arg,
    create_llm_client_from_preset,
    load_beir_datasets,
    load_json,
    Log,
    file_exist,
    load_models,
    load_dataset_for_watermark_generation,
    find_latest_injection_result,
)
from sentinelrag.rag import VectorStore, check_collection, check_collection_exists, check_and_clean_existing_watermarks
from sentinelrag.core.detector import (
    RAGWatermarkDetector,
    perform_watermark_detection,
    save_detection_results,
)

# Load environment variables
load_dotenv()

# --- Configuration ---
RLLM_PRESET = "gpt-5-mini"
RLLM_KWARGS = {}

DLLM_PRESET = "gpt-5-nano"
DLLM_KWARGS = {}


# Set random seed
SEED = 633


def setup_seeds(seed):
    """Setup random seeds for reproducibility"""
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import asyncio


def load_injection_result(injection_result_path):
    """Load injection results from file"""
    if not os.path.exists(injection_result_path):
        raise FileNotFoundError(f"Injection result file does not exist: {injection_result_path}")
    return load_json(injection_result_path)


def setup_vectorstore(args, eval_dataset=None, device=None, logger=None):
    """Setup vector store"""
    if eval_dataset is None:
        eval_dataset = args.eval_dataset
        
    collection_name = f"{eval_dataset}_{args.eval_model_code}_{args.score_function}"

    # Load retrieval model
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    
    # Check if vector database exists
    collection_exist = check_collection_exists(collection_name)
    
    if not collection_exist:
        raise ValueError(
            f"Vector database does not exist. Please run first: "
            f"sentinelrag-build-chroma --eval_dataset {eval_dataset} --eval_model_code {args.eval_model_code}"
        )

    # Load dataset once and prepare corpus data
    if eval_dataset == 'closed_qa':
        train_dataset = load_dataset("databricks/databricks-dolly-15k", split='train')
        closed_qa_dataset = train_dataset.filter(lambda example: example['category'] == 'closed_qa')
        datalen = len(closed_qa_dataset)
        corpus_for_vectorstore = {}
        for i, item in enumerate(closed_qa_dataset):
            corpus_for_vectorstore[str(i)] = {
                'text': f"{item['instruction']}. {item['context']}",
                'title': f"QA_{i}"
            }
    else:
        corpus, _, _ = load_beir_datasets(eval_dataset, args.split)
        datalen = len(corpus)
        corpus_for_vectorstore = corpus

    if device is None:
        device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else "cpu"
        
    vectorstore = VectorStore(model, tokenizer, get_emb, corpus_for_vectorstore, device, 
                             collection_name, use_local=True)
    
    # Check and clean any existing watermarks
    collection_len = check_and_clean_existing_watermarks(vectorstore, datalen, logger)

    return vectorstore, collection_name, collection_len


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Watermark Detection CLI')
    add_model_preset_arg(parser, flag='rllm', default_preset=RLLM_PRESET, help_text='RAG answer-generation LLM preset')
    add_model_preset_arg(parser, flag='dllm', default_preset=DLLM_PRESET, help_text='Detection/verification LLM preset')
    
    # Required parameters
    parser.add_argument('--injection_result_path', type=str, default=None,
                        help='Injection result file path')
    parser.add_argument('--eval_model_code', type=str, default='contriever',
                        choices=['contriever', 'contriever-msmarco', 'ance'],
                        help='Retrieval model')
    
    # Optional parameters
    parser.add_argument('--num_select_kos', type=int, default=None, 
                        help='Number of KOs (used for auto-selecting injection result)')
    parser.add_argument('--eval_dataset', type=str, default=None,
                        help='Dataset name (if not specified, inferred from injection results)')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--score_function', type=str, default='cosine', 
                        choices=['cosine', 'l2', 'ip'], help='Similarity function')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top-k documents to retrieve')
    parser.add_argument('--questions_per_ko', type=int, default=1, help='Number of questions per KO')
    parser.add_argument('--watermark_query_count', type=int, default=None, help='Number of watermark queries (default: use all injected watermarks)')
    parser.add_argument('--alpha', type=float, default=0.01, help='Significance level')
    parser.add_argument('--p0', type=float, default=0.02, help='Null hypothesis accuracy')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for async operations (RAG generation and detection)')
    parser.add_argument('--paraphrase_batch_size', type=int, default=30, help='Batch size for paraphrase attack (default: 30)')
    
    # Other parameters
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU device ID')
    parser.add_argument('--basepath', type=str, default='./output', help='Output path')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--paraphrase', action='store_true',
                        help='Perform paraphrase attack')
    parser.add_argument('--skip_detection', action='store_true',
                        help='Skip original watermark detection, only run attack(s) specified by --paraphrase and/or --translate')
    parser.add_argument('--translate', type=str, nargs='?', const='zh-cn', default=None,
                        help='Perform translation attack on the retrieved corpus. Optionally specify target language (default: zh-cn)')
    parser.add_argument('--test_clean', action='store_true',
                        help='Also test on clean database')
    parser.add_argument('--tran_on_docs', action='store_true',
                        help='If set, perform translation on retrieved documents (legacy behavior). Default is to translate the RAG response.')
    parser.add_argument('--disable_generation', action='store_true',
                        help='Skip RAG response generation and verification, only do retrieval')
    parser.add_argument('--hard', action='store_true',
                        help='Use hardened RAG generation prompt with defensive anti-injection cue')
    parser.add_argument('--xhard', action='store_true',
                        help='Use extra-hard RAG defense prompt (includes all --hard defenses plus stricter refusal rules)')
    parser.add_argument('--verification_mode', type=str, default='correct_answer_based',
                        choices=['ko_based', 'correct_answer_based'],
                        help='Answer verification mode: ko_based uses KO facts, correct_answer_based uses the generated correct_answer (default: correct_answer_based)')
    
    # Partial theft attack arguments
    parser.add_argument('--partial_theft', action='store_true',
                        help='Simulate partial theft attack by randomly sampling a subset of the corpus')
    parser.add_argument('--theft_ratio', type=float, nargs='+', default=[0.5],
                        help='Ratio(s) of corpus to keep in partial theft attack. Can specify multiple values, e.g., --theft_ratio 0.1 0.3 0.5 (default: 0.5)')
    parser.add_argument('--search_k', type=int, default=10000,
                        help='Candidate pool size for retrieval before filtering in partial theft mode (default: 100)')
    
    return parser.parse_args()


async def run_async_main():
    args = parse_args()
    
    # Setup seeds
    setup_seeds(args.seed)
    
    # Select device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = f'cuda:{args.gpu_id}'
    else:
        device = "cpu"
        print("CUDA is not available; using CPU for detection.")
    
    # Auto-select injection result if not provided
    if args.injection_result_path is None:
        if args.eval_dataset and args.num_select_kos:
            try:
                print(f"Auto-selecting injection result for {args.eval_dataset} k={args.num_select_kos}...")
                args.injection_result_path = find_latest_injection_result(
                    args.basepath, args.eval_dataset, args.num_select_kos
                )
            except Exception as e:
                raise ValueError(f"Could not auto-select injection result: {e}")
        else:
            raise ValueError("Either --injection_result_path OR (--eval_dataset AND --num_select_kos) must be provided.")

    # Validate injection result file
    if not os.path.exists(args.injection_result_path):
        raise FileNotFoundError(f"Injection result file does not exist: {args.injection_result_path}")
    
    # Create detection result folder
    print(f"Starting watermark detection...")
    
    try:
        # 1. Load injection results
        injection_result = load_injection_result(args.injection_result_path)
        
        # 2. Infer dataset (if not specified)
        if args.eval_dataset is None:
            args.eval_dataset = injection_result['injection_metadata']['ko_pool_source'].get('dataset_used')
            if args.eval_dataset is None:
                raise ValueError("Cannot infer dataset, please specify --eval_dataset")

        # Create detection result folder (now that we have the dataset)
        # Create detection result folder with structure: .../{dataset}/k{knum}/{timestamp}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add top_k to timestamp if it's not the default value
        if args.top_k != 5:
            timestamp = f"{timestamp}_topk{args.top_k}"

        # Mark hardened runs in output directory names.
        if getattr(args, "hard", False):
            timestamp = f"{timestamp}_hard"
        if getattr(args, "xhard", False):
            timestamp = f"{timestamp}_xhard"
        
        num_kos = injection_result.get('injection_metadata', {}).get('num_selected_kos', 'unknown')
        k_dir = f"k{num_kos}"
        
        # Use 'attacks' folder if --translate or --paraphrase or --partial_theft is specified, otherwise 'watermark_detections'
        if args.translate is not None or args.paraphrase or getattr(args, "partial_theft", False):
            output_folder = 'attacks'
        else:
            output_folder = 'watermark_detections'
            
        sub_path = k_dir
        if getattr(args, "partial_theft", False):
            theft_ratios = getattr(args, "theft_ratio", [0.5])
            # For output path, use first ratio or 'multi' if multiple ratios
            if len(theft_ratios) == 1:
                sub_path = os.path.join(k_dir, "partial_theft", str(theft_ratios[0]))
            else:
                ratios_str = "_".join(str(r) for r in theft_ratios)
                sub_path = os.path.join(k_dir, "partial_theft", f"multi_{ratios_str}")
            
        basepath = os.path.join(args.basepath, output_folder, args.eval_dataset, sub_path, timestamp)

        # Setup logging
        log_file = os.path.join(basepath, 'detection.log')
        file_exist(log_file)
        logger = Log(log_file=log_file).get(__file__)
        logger.info(f'Parameters: {args}')
        logger.info(f'Injection result file: {args.injection_result_path}')
        logger.info(f'Detection result folder: {basepath}')
        logger.info(f'Using device: {device}')
        
        print(f"Detection result folder: {basepath}")
        
        # 3. Load dataset
        sampled_dataset, full_dataset, ndataset = load_dataset_for_watermark_generation(
            args.eval_dataset, args.split, 0)
        logger.info(f'Successfully loaded dataset, total size: {len(full_dataset)}')
        
        # 4. Setup LLM clients
        rllm_client, rllm_preset = create_llm_client_from_preset(args.rllm)
        dllm_client, dllm_preset = create_llm_client_from_preset(args.dllm)
        logger.info(f'Using RLLM preset for RAG answer generation: {rllm_preset.preset_name} -> {rllm_preset.model} ({rllm_preset.llm_url})')
        logger.info(f'Using DLLM preset for watermark detection/verification: {dllm_preset.preset_name} -> {dllm_preset.model} ({dllm_preset.llm_url})')
        logger.info(f'Hard mode: {args.hard}')
        logger.info(f'XHard mode: {args.xhard}')
        
        # 5. Setup RAG system
        vectorstore, collection_name, collection_len = setup_vectorstore(args, args.eval_dataset, device=device, logger=logger)
        vectorstore_info = {
            "collection_name": collection_name,
            "collection_length": collection_len,
            "eval_model_code": args.eval_model_code,
            "score_function": args.score_function,
            "eval_dataset": args.eval_dataset
        }
        logger.info(f'Dataset length: {len(ndataset)}, collection length: {collection_len}')
        
        # Auto-enable paraphrase if paraphrase_only is set
        # (Removed: --skip_detection no longer requires --paraphrase)
        
        # 6. Perform watermark detection
        detection_results = await perform_watermark_detection(
            full_dataset, vectorstore, rllm_client, injection_result,
            top_k=args.top_k,
            questions_per_ko=args.questions_per_ko,
            p0=args.p0,
            alpha=args.alpha,
            test_clean=args.test_clean,
            do_paraphrase=getattr(args, "paraphrase", False),
            cleanup_watermarks_flag=True,
            basepath=basepath,
            logger=logger,
            rllm_kwargs=RLLM_KWARGS,
            dllm_client=dllm_client,
            dllm_kwargs=DLLM_KWARGS,
            watermark_query_count=args.watermark_query_count,
            skip_detection=getattr(args, "skip_detection", False),
            batch_size=args.batch_size,
            disable_generation=getattr(args, "disable_generation", False),
            hard_mode=(getattr(args, "hard", False) or getattr(args, "xhard", False)),
            xhard_mode=getattr(args, "xhard", False),
            verification_mode=getattr(args, "verification_mode", "correct_answer_based"),
            do_translate=args.translate is not None,
            translate_lang=args.translate if args.translate else "zh-cn",
            do_partial_theft=getattr(args, "partial_theft", False),
            theft_ratios=getattr(args, "theft_ratio", [0.5]),
            search_k=getattr(args, "search_k", 100),
        )
        
        (result_original, result_watermarked, detector_original, detector_watermarked,
         documents_original_path, documents_watermarked_path, watermark_ids,
         paraphrase_result, paraphrase_results_path,
         translate_result, translate_results_path) = detection_results
        
        # 7. Save detection results
        result_path, db_info_path = save_detection_results(
            basepath, injection_result, full_dataset, result_original, result_watermarked,
            detector_original, detector_watermarked,
            documents_original_path, documents_watermarked_path, vectorstore_info, watermark_ids,
            paraphrase_result, paraphrase_results_path,
            eval_dataset=args.eval_dataset,
            split=args.split,
            top_k=args.top_k,
            questions_per_ko=args.questions_per_ko,
            alpha=args.alpha,
            p0=args.p0,
            cleanup_watermarks_flag=True,
            logger=logger,
            rllm_model=rllm_preset.model,
            dllm_model=dllm_preset.model,
            translate_result=translate_result,
            translate_results_path=translate_results_path,
        )
        
        print(f"Detection complete. Results saved to: {basepath}")
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f'Watermark detection failed: {e}')
        print(f"Watermark detection failed: {e}")
        import traceback
        traceback.print_exc()
        return


def main():
    """Entry point for the CLI."""
    asyncio.run(run_async_main())


if __name__ == "__main__":
    main()
