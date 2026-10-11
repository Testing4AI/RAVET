#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI entry point for Watermark Injection

Usage:
python -m sentinelrag.cli.inject_watermark --ko_pool_path ./output/ko_pools/ko_pool_*/ko_pool.json --secret_key user123_secret --eval_dataset nfcorpus --eval_model_code contriever
"""

import argparse
import asyncio
import os
import sys
import random
import re

import numpy as np
import torch
from datetime import datetime
from dotenv import load_dotenv
from datasets import load_dataset

from sentinelrag.utils import (
    add_model_preset_arg,
    create_llm_client_from_preset,
    load_beir_datasets,
    load_json,
    save_json,
    Log,
    file_exist,
    load_models,
    load_dataset_for_watermark_generation,
)
from sentinelrag.rag import VectorStore, check_collection, check_collection_exists, check_and_clean_existing_watermarks
from sentinelrag.core.injector import WatermarkInjector, inject_to_vectorstore
from sentinelrag.prompts import PromptTemplates

# Load environment variables
load_dotenv()

# --- Configuration ---
LLM_PRESET = "qwen3.5-2b-local"
LLM_KWARGS = {}

# Set random seed
SEED = 633


def get_generation_model_suffix(model: str) -> str:
    """Return a short generation-model suffix for output folder names."""
    # Common aliases for readability in output paths.
    alias_map = {
        "gpt-5-mini": "g5m",
        "gpt-5": "g5",
        "gpt-4o-mini": "g4om",
        "gpt-4o": "g4o",
    }
    lowered = (model or "").strip().lower()
    if lowered in alias_map:
        return alias_map[lowered]

    # Fallback: keep only alphanumerics and cap length to keep paths short.
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    return compact[:8] if compact else "gen"


def get_stealth_suffix(stealth_mode: str) -> str:
    """Return a short stealth-mode suffix for output folder names."""
    mode = (stealth_mode or "normal").strip().lower()
    alias_map = {
        "normal": "stn",
        "mild": "stm",
        "strong": "sts",
    }
    return alias_map.get(mode, "stn")


def setup_seeds(seed):
    """Setup random seeds for reproducibility"""
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


async def generate_questions_for_watermarks(llm_client, watermark_texts, questions_per_ko, logger, batch_size=5):
    """Generate verification questions for all watermark texts
    
    Args:
        llm_client: LLM client with async support
        watermark_texts: List of watermark texts
        questions_per_ko: Number of questions to generate per watermark
        logger: Logger instance
        batch_size: Number of concurrent requests
        
    Returns:
        List of question data for each watermark, where each item contains:
        - questions: List of question strings
        - correct_answers: List of corresponding correct answers
    """
    semaphore = asyncio.Semaphore(batch_size)
    
    async def generate_questions_for_text(idx, watermark_text):
        async with semaphore:
            prompt = PromptTemplates.generate_simple_verification_questions(
                watermark_text, num_questions=questions_per_ko
            )
            
            for attempt in range(2):  # max_attempts = 2
                try:
                    kwargs = {**LLM_KWARGS, "is_json": True}
                    result = await llm_client.ask_llm_async(prompt, **kwargs)
                    questions = result.get("questions", [])
                    
                    if not questions:
                        logger.warning(f"Watermark #{idx+1}: Attempt {attempt+1} - No questions generated, retrying...")
                        continue
                    
                    # Extract questions and correct_answers
                    question_list = []
                    answer_list = []
                    for q in questions[:questions_per_ko]:
                        if isinstance(q, dict):
                            question_list.append(q.get("question", ""))
                            answer_list.append(q.get("correct_answer", ""))
                        else:
                            question_list.append(q)
                            answer_list.append("")
                    
                    logger.info(f"Watermark #{idx+1}: Generated {len(question_list)} questions")
                    return {
                        "watermark_index": idx,
                        "questions": question_list,
                        "correct_answers": answer_list
                    }
                    
                except Exception as e:
                    logger.error(f"Watermark #{idx+1}: Error generating questions - {e}")
                    if attempt == 1:  # Last attempt
                        return {
                            "watermark_index": idx,
                            "questions": [],
                            "correct_answers": [],
                            "error": str(e)
                        }
            
            return {
                "watermark_index": idx,
                "questions": [],
                "correct_answers": []
            }
    
    tasks = [generate_questions_for_text(idx, text) for idx, text in enumerate(watermark_texts)]
    results = await asyncio.gather(*tasks)
    
    # Sort by watermark_index to maintain order
    results.sort(key=lambda x: x["watermark_index"])
    
    return results


def load_ko_pool(ko_pool_path):
    """Load KO pool from file"""
    if not os.path.exists(ko_pool_path):
        raise FileNotFoundError(f"KO pool file does not exist: {ko_pool_path}")
    return load_json(ko_pool_path)


def setup_vectorstore(args, logger=None):
    """Setup vector store"""
    collection_name = f"{args.eval_dataset}_{args.eval_model_code}_{args.score_function}"

    # Load retrieval model
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    
    # Check if vector database exist
    collection_exist = check_collection_exists(collection_name)
    
    if not collection_exist:
        raise ValueError(
            f"Vector database does not exist. Please run first: "
            f"sentinelrag-build-chroma --eval_dataset {args.eval_dataset} --eval_model_code {args.eval_model_code}"
        )

    # Load dataset once and prepare corpus for vector store
    if args.eval_dataset == 'closed_qa':
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
        corpus, _, _ = load_beir_datasets(args.eval_dataset, args.split)
        datalen = len(corpus)
        corpus_for_vectorstore = corpus
    
    # Use existing vector store
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    vectorstore = VectorStore(model, tokenizer, get_emb, corpus_for_vectorstore, device, 
                             collection_name, use_local=True)
    
    # Clean vector store before injection (in case of previous injections)
    collection_len = check_and_clean_existing_watermarks(vectorstore, datalen, logger)
    return vectorstore, collection_name, collection_len


def save_injection_results(basepath, injection_result, vectorstore_info, watermark_ids, logger):
    """Save injection results to files"""
    
    # Save complete injection results
    injection_result_path = os.path.join(basepath, 'injection_result.json')
    save_json(injection_result, injection_result_path)
    logger.info(f'Injection results saved to: {injection_result_path}')
    
    # Save selected KOs (for compatibility)
    selected_kos_path = os.path.join(basepath, 'selected_kos.json')
    save_json({"selected_kos": injection_result["selected_kos"]}, selected_kos_path)
    logger.info(f'Selected KOs saved to: {selected_kos_path}')
    
    # Save watermark texts
    watermark_texts_path = os.path.join(basepath, 'watermark_texts.json')
    save_json({"watermark_texts": injection_result["watermark_texts"]}, watermark_texts_path)
    logger.info(f'Watermark texts saved to: {watermark_texts_path}')
    
    # Save questions and answers
    questions_answers_path = os.path.join(basepath, 'questions_answers.json')
    qa_data = {
        "all_questions": injection_result.get("all_questions", []),
        "all_correct_answers": injection_result.get("all_correct_answers", []),
        "watermark_questions": injection_result.get("watermark_questions", [])
    }
    save_json(qa_data, questions_answers_path)
    logger.info(f'Questions and answers saved to: {questions_answers_path}')
    
    # Save vector database information
    db_injection_info = {
        **vectorstore_info,
        "watermark_ids": watermark_ids,
        "injection_timestamp": datetime.now().isoformat(),
        "num_watermarks_injected": len(watermark_ids)
    }
    db_info_path = os.path.join(basepath, 'vectorstore_injection_info.json')
    save_json(db_injection_info, db_info_path)
    logger.info(f'Vector database injection info saved to: {db_info_path}')
    
    return injection_result_path, selected_kos_path, watermark_texts_path, db_info_path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Watermark Injection CLI')
    add_model_preset_arg(parser, flag='llm', default_preset=LLM_PRESET, help_text='Watermark-generation LLM preset')
    
    # Required arguments
    parser.add_argument('--ko_pool_path', type=str, required=True,
                        help='KO pool file path')
    parser.add_argument('--secret_key', type=str, required=True,
                        help='User secret key (for deterministic KO selection)')
    parser.add_argument('--eval_dataset', type=str, required=False, default=None,
                        help='Dataset name (optional, can be inferred from KO pool)')
    parser.add_argument('--eval_model_code', type=str, default='contriever',
                        choices=['contriever', 'contriever-msmarco', 'ance'],
                        help='Retrieval model')
    
    # Optional arguments
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--num_select_kos', type=int, default=3, 
                        help='Number of KOs to select from pool')
    parser.add_argument('--score_function', type=str, default='cosine', 
                        choices=['cosine', 'l2', 'ip'], help='Similarity function')
    
    # Other arguments
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU device ID')
    parser.add_argument('--basepath', type=str, default='./output', help='Output path')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--inject_to_db', action='store_true', 
                        help='Whether to inject watermarks to vector database (default: no)')
    parser.add_argument('--stealth_mode', type=str, default='normal',
                        choices=['normal', 'mild', 'strong'],
                        help='Stealth control for LLM watermark expansion prompts')
    parser.add_argument('--questions_per_ko', type=int, default=1,
                        help='Number of verification questions to generate per watermark')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size for async question generation')
    
    return parser.parse_args()


async def async_main():
    args = parse_args()
    
    # Setup seeds
    setup_seeds(args.seed)
    
    # Setup GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    
    # Infer dataset from KO pool if not provided
    if args.eval_dataset is None:
        try:
            print(f"Inferring dataset from KO pool: {args.ko_pool_path}")
            # We load the KO pool here to check metadata
            pool_data = load_ko_pool(args.ko_pool_path)
            args.eval_dataset = pool_data.get("metadata", {}).get("dataset_used")
            if args.eval_dataset is None:
                raise ValueError("Could not infer 'dataset_used' from KO pool metadata.")
            print(f"Inferred dataset: {args.eval_dataset}")
        except Exception as e:
            print(f"Error inferring dataset: {e}")
            print("Please provide --eval_dataset explicitly.")
            sys.exit(1)

    # Create timestamped injection result folder with generation model suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _, llm_preset = create_llm_client_from_preset(args.llm)
    gen_model_suffix = get_generation_model_suffix(llm_preset.model)
    stealth_suffix = get_stealth_suffix(args.stealth_mode)
    timestamp_with_suffix = f"{timestamp}_{gen_model_suffix}_{stealth_suffix}"
    k_dir = f"k{args.num_select_kos}"
    basepath = os.path.join(args.basepath, "watermark_injections", args.eval_dataset, k_dir, timestamp_with_suffix)
    
    # Setup logging
    log_file = os.path.join(basepath, 'injection.log')
    file_exist(log_file)
    logger = Log(log_file=log_file).get(__file__)
    logger.info(f'Arguments: {args}')
    logger.info(f'Injection result folder: {basepath}')
    
    print(f"Injection started: {basepath}")
    
    try:
        # 1. Load KO pool
        ko_pool_data = load_ko_pool(args.ko_pool_path)
        logger.info(f'Loaded KO pool with {len(ko_pool_data["ko_pool"])} KOs')
        
        # 2. Load dataset (for style sampling)
        sampled_dataset, full_dataset, ndataset = load_dataset_for_watermark_generation(
            args.eval_dataset, args.split, 0)  # Use complete dataset
        logger.info(f'Successfully loaded dataset, total size: {len(full_dataset)}')
        
        # 3. Setup LLM
        llm_client, llm_preset = create_llm_client_from_preset(args.llm)
        logger.info(
            f'LLM preset created successfully: {llm_preset.preset_name} '
            f'-> {llm_preset.model} ({llm_preset.llm_url})'
        )
        
        # 4. Execute watermark injection (use async version since we're in async context)
        injector = WatermarkInjector(
            llm_client, 
            args.secret_key, 
            llm_kwargs=LLM_KWARGS,
            stealth_mode=args.stealth_mode,
        )
        injection_result = await injector.inject_watermarks_async(
            ko_pool_data, full_dataset, args.num_select_kos
        )
        
        # 5. Generate verification questions for each watermark text
        logger.info(f'Generating {args.questions_per_ko} verification question(s) per watermark...')
        print(f"Generating verification questions for {len(injection_result['watermark_texts'])} watermarks...")
        
        question_data = await generate_questions_for_watermarks(
            llm_client,
            injection_result['watermark_texts'],
            args.questions_per_ko,
            logger,
            batch_size=args.batch_size
        )
        
        # Add questions to injection result
        injection_result['watermark_questions'] = question_data
        
        # Also create flattened lists for convenience (matching detector format)
        all_questions = []
        all_correct_answers = []
        for qd in question_data:
            all_questions.extend(qd.get("questions", []))
            all_correct_answers.extend(qd.get("correct_answers", []))
        
        injection_result['all_questions'] = all_questions
        injection_result['all_correct_answers'] = all_correct_answers
        
        logger.info(f'Generated {len(all_questions)} total questions for {len(question_data)} watermarks')
        print(f"Generated {len(all_questions)} total questions")
        
        # 6. Optional: inject to vector database
        watermark_ids = []
        vectorstore_info = {}
        
        if args.inject_to_db:
            # Setup vector store
            vectorstore, collection_name, collection_len = setup_vectorstore(args, logger)
            vectorstore_info = {
                "collection_name": collection_name,
                "collection_length": collection_len,
                "eval_model_code": args.eval_model_code,
                "score_function": args.score_function,
                "eval_dataset": args.eval_dataset
            }
            
            # Inject watermarks to vector database
            watermark_ids = inject_to_vectorstore(
                vectorstore, injection_result["watermark_texts"], logger
            )
        
        # 7. Save injection results
        result_paths = save_injection_results(
            basepath, injection_result, vectorstore_info, watermark_ids, logger
        )
        injection_result_path, selected_kos_path, watermark_texts_path, db_info_path = result_paths
        
        print(f"Injection complete. Results saved to: {injection_result_path}")
        
    except Exception as e:
        logger.error(f'Watermark injection failed: {e}')
        print(f"Watermark injection failed: {e}")
        import traceback
        traceback.print_exc()
        return


def main():
    """Entry point for the CLI."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
