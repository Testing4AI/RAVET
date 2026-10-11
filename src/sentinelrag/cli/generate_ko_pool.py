#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI entry point for KO Pool Generation

Usage:
python -m sentinelrag.cli.generate_ko_pool --eval_dataset nfcorpus --target_ko_count 50 --num_examples 10
"""

import argparse
import os
import sys
import random
import concurrent.futures
import math

import numpy as np
import torch
from datetime import datetime
from dotenv import load_dotenv

from sentinelrag.utils import (
    add_model_preset_arg,
    create_llm_client_from_preset,
    save_json,
    Log,
    file_exist,
    load_dataset_for_watermark_generation,
)
from sentinelrag.core.ko_pool import KOPoolGenerator

# Load environment variables
load_dotenv()

# --- Configuration ---
KO_GENERATION_PRESET = "qwen3.5-2b-local"

KO_GENERATION_LLM_KWARGS = {
    "is_json": True,   
    # "extra_body":{
    #     "chat_template_kwargs": {"enable_thinking": False},
    # },
}

ABSTRACT_TO_KO_PRESET = "gpt-5-nano"

ABSTRACT_TO_KO_LLM_KWARGS = {
    "is_json": True,
    "reasoning_effort": "low",
}

# Set random seed
SEED = 633


def setup_seeds(seed):
    """Setup random seeds for reproducibility"""
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='KO Pool Generation CLI')
    add_model_preset_arg(parser, flag='ko-generation-llm', default_preset=KO_GENERATION_PRESET, help_text='KO generation LLM preset')
    add_model_preset_arg(parser, flag='abstract-llm', default_preset=ABSTRACT_TO_KO_PRESET, help_text='Abstract-to-KO LLM preset')
    
    parser.add_argument('--eval_dataset', type=str, default='hotpotqa', help='Dataset to use')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    
    # KO generation parameters
    parser.add_argument('--target_ko_count', type=int, default=100, 
                        help='Target number of KOs to generate')
    parser.add_argument('--num_examples', type=int, default=10,
                        help='Number of real samples for style learning')
    
    # Other parameters
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU device ID')
    parser.add_argument('--basepath', type=str, default='./output', help='Output path')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--abstract_workers', type=int, default=10,
                        help='Worker threads for parallel abstract_to_ko conversion')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup seeds
    setup_seeds(args.seed)
    
    # Setup GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    
    # Create timestamped KO pool folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ko_pool_name = f"ko_pool_{args.eval_dataset}_n{args.num_examples}_t{args.target_ko_count}_{timestamp}"
    # Group generated pools by model under output/ko_pools/{model}/
    ko_generation_llm_client, ko_generation_preset = create_llm_client_from_preset(args.ko_generation_llm, include_async=False)
    abstract_to_ko_llm_client, abstract_preset = create_llm_client_from_preset(args.abstract_llm, include_async=False)
    basepath = os.path.join(args.basepath, 'ko_pools', ko_generation_preset.preset_name, ko_pool_name)
    
    # Setup logging
    log_file = os.path.join(basepath, 'ko_generation.log')
    file_exist(log_file)
    logger = Log(log_file=log_file).get(__file__)
    logger.info(f'Arguments: {args}')
    logger.info(f'KO pool folder: {basepath}')
    
    print(f"KO pool generation started: {basepath}")
    
    try:
        # 1. Load dataset (full dataset, sampling happens internally)
        _, full_dataset, _ = load_dataset_for_watermark_generation(
            args.eval_dataset, args.split, sample_size=0)
        logger.info(f'Successfully loaded dataset, size: {len(full_dataset)}')

        # 2. Setup stage-specific LLMs
        logger.info(
            f'LLM presets created successfully: '
            f'abstract_to_ko={abstract_preset.preset_name}->{abstract_preset.model} '
            f'({abstract_preset.llm_url}), '
            f'ko_generation={ko_generation_preset.preset_name}->{ko_generation_preset.model} '
            f'({ko_generation_preset.llm_url})'
        )
        
        # 3. Generate KO pool with potential parallelism
        generator = KOPoolGenerator(
            full_dataset,
            ko_generation_llm_client,
            llm_kwargs=KO_GENERATION_LLM_KWARGS,
            abstract_llm_client=abstract_to_ko_llm_client,
            abstract_llm_kwargs=ABSTRACT_TO_KO_LLM_KWARGS,
            abstract_workers=args.abstract_workers,
        )
        
        ko_pool_data = {
            "metadata": {},
            "real_kos": [],
            "ko_pool": []
        }
        
        # Determine how many chunks we need
        chunk_size = 50
        num_chunks = math.ceil(args.target_ko_count / chunk_size)
        
        if num_chunks > 1:
            logger.info(f"Target count {args.target_ko_count} > {chunk_size}. Splitting into {num_chunks} parallel chunks.")
            print(f"Target count {args.target_ko_count} > {chunk_size}. Splitting into {num_chunks} parallel chunks.")
            
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_chunks, 10)) as executor:
                for i in range(num_chunks):
                    # Calculate target for this chunk
                    # Last chunk gets the remainder
                    if i == num_chunks - 1:
                        chunk_target = args.target_ko_count - (i * chunk_size)
                    else:
                        chunk_target = chunk_size
                    
                    # Offset ensures unique salts for each chunk
                    # Each chunk roughly consumes 1 batch index per 50 items, but to be safe and avoid collision
                    # we can space them out. Since each call manages its own batch_idx locally starting from offset,
                    # we just need to ensure offsets don't overlap. 
                    # KOPoolGenerator increments batch_idx by 1 for every 50 items.
                    # So passing i as offset is NOT enough if one chunk needs multiple batches (unlikely if chunk_size=50).
                    # Actually, since we force chunk_size=50, each chunk will strictly use 1 batch (batch_idx = offset).
                    # So simply passing i as offset is sufficient.
                     
                    futures.append(
                        executor.submit(
                            generator.generate_ko_pool,
                            target_ko_count=chunk_target,
                            num_examples=args.num_examples,
                            batch_offset=i
                        )
                    )
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    ko_pool_data["ko_pool"].extend(result["ko_pool"])
                    ko_pool_data["real_kos"].extend(result["real_kos"])
                except Exception as exc:
                    logger.error(f"Chunk generation generated an exception: {exc}")
                    print(f"Chunk generation generated an exception: {exc}")
                    
        else:
            # Single chunk handling
            result = generator.generate_ko_pool(
                target_ko_count=args.target_ko_count,
                num_examples=args.num_examples
            )
            ko_pool_data = result

        # Retry loop: keep generating if we still don't have enough (serially)
        # This handles cases where some chunks might have partially failed or returned fewer items
        iteration = 1
        # Calculate current max batch offset used so we can continue from there
        current_max_offset = num_chunks 
        
        while len(ko_pool_data["ko_pool"]) < args.target_ko_count:
            remaining = args.target_ko_count - len(ko_pool_data["ko_pool"])
            logger.info(f'Retry iteration {iteration}: Have {len(ko_pool_data["ko_pool"])} KOs, need {remaining} more')
            print(f"Retry iteration {iteration}: Have {len(ko_pool_data['ko_pool'])} KOs, need {remaining} more")
            
            # Generate additional KOs with remaining count as target
            additional_data = generator.generate_ko_pool(
                target_ko_count=remaining,
                num_examples=args.num_examples,
                batch_offset=current_max_offset + iteration # Ensure new seeds
            )
            
            # Append new KOs to existing pool
            ko_pool_data["ko_pool"].extend(additional_data["ko_pool"])
            # Also extend real_kos if new ones were generated
            ko_pool_data["real_kos"].extend(additional_data["real_kos"])
            
            iteration += 1
        
        logger.info(f'Final KO count: {len(ko_pool_data["ko_pool"])}')
        print(f"Final KO count: {len(ko_pool_data['ko_pool'])}")
        
        # Update metadata
        # Create base metadata structure if it doesn't exist (from manual aggregation)
        if "metadata" not in ko_pool_data:
             ko_pool_data["metadata"] = {}
             
        ko_pool_data["metadata"]["actual_ko_count"] = len(ko_pool_data["ko_pool"])
        ko_pool_data["metadata"]["retry_iterations"] = iteration - 1 # iteration starts at 1
        ko_pool_data["metadata"]["dataset_used"] = args.eval_dataset
        ko_pool_data["metadata"]["dataset_size"] = len(full_dataset)
        ko_pool_data["metadata"]["abstract_model"] = abstract_preset.model
        ko_pool_data["metadata"]["ko_generation_model"] = ko_generation_preset.model
        
        # 4. Save KO pool data
        ko_pool_path = os.path.join(basepath, 'ko_pool.json')
        save_json(ko_pool_data, ko_pool_path)
        logger.info(f'KO pool saved to: {ko_pool_path}')
        
        # 5. Save configuration info
        config_info = {
            "generation_config": vars(args),
            "ko_pool_path": ko_pool_path,
            "generation_timestamp": datetime.now().isoformat(),
            "total_kos": len(ko_pool_data["ko_pool"]),
            "models": {
                "abstract_to_ko": abstract_preset.model,
                "ko_generation": ko_generation_preset.model,
            },
            "usage_instructions": {
                "injection_script": (
                    f"python -m sentinelrag.cli.inject_watermark --ko_pool_path {ko_pool_path} "
                    f"--secret_key YOUR_SECRET_KEY --eval_dataset {args.eval_dataset} "
                    "--eval_model_code contriever --stealth_mode strong"
                ),
            }
        }
        config_path = os.path.join(basepath, 'config.json')
        save_json(config_info, config_path)
        
        print(f"KO pool generation complete. Saved to: {ko_pool_path}")
        
    except Exception as e:
        logger.error(f'KO pool generation failed: {e}')
        print(f"KO pool generation failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
