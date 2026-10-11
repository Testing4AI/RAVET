#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KO Pool Generator

Generates Knowledge Object (KO) pools for watermark injection.
"""

import json
import concurrent.futures
from datetime import datetime

from sentinelrag.utils import deterministic_sample
from sentinelrag.prompts import PromptTemplates


# Default key for KO pool generation (can be overridden)
DEFAULT_KO_GENERATION_SECRET = "ko_pool_generation_master_key_2024"


class KOPoolGenerator:
    """Knowledge Object pool generator."""

    def __init__(
        self,
        dataset,
        llm_client,
        generation_secret=None,
        llm_kwargs: dict = None,
        abstract_llm_client=None,
        abstract_llm_kwargs: dict = None,
        abstract_workers: int = 1,
    ):
        """
        Initialize KO Pool Generator
        
        Args:
            dataset: Dataset to sample from for style learning
            llm_client: LLM client for fake KO generation
            generation_secret: Secret key for deterministic sampling (optional)
            llm_kwargs: Additional kwargs for fake KO generation LLM calls
            abstract_llm_client: Optional separate LLM client for abstract_to_ko
            abstract_llm_kwargs: Optional kwargs for abstract_to_ko LLM calls
            abstract_workers: Number of worker threads for parallel abstract_to_ko calls
        """
        self.dataset = dataset
        self.llm_client = llm_client
        self.abstract_llm_client = abstract_llm_client or llm_client
        self.generation_secret = generation_secret or DEFAULT_KO_GENERATION_SECRET
        self.llm_kwargs = llm_kwargs or {}
        self.abstract_llm_kwargs = abstract_llm_kwargs or self.llm_kwargs
        self.abstract_workers = max(1, int(abstract_workers))

    def _abstract_to_ko(self, text_document):
        """Convert abstract document to Knowledge Object (KO)"""
        prompt = PromptTemplates.abstract_to_ko(text_document)
        return self.abstract_llm_client.ask_llm(prompt, **self.abstract_llm_kwargs)

    def _generate_fake_kos(self, real_kos_examples, num_to_generate):
        """Generate fake KOs based on real examples"""
        examples_str = json.dumps(real_kos_examples, indent=2, ensure_ascii=False)
        prompt = PromptTemplates.generate_fake_kos(examples_str, num_to_generate)
        result = self.llm_client.ask_llm(prompt, **self.llm_kwargs)
        return result.get("fake_kos", [])

    def generate_ko_pool(self, target_ko_count: int = 50, num_examples: int = 10, batch_offset: int = 0):
        """
        Generate KO pool
        
        Args:
            target_ko_count: Target number of KOs to generate
            num_examples: Number of real samples for style learning
            batch_offset: Offset for starting batch index (for parallel generation)
            
        Returns:
            dict: KO pool data containing metadata, real_kos, and ko_pool
        """
        print(f"--- Starting KO pool generation (target: {target_ko_count}, offset: {batch_offset}) ---")
        
        fake_kos = []
        all_real_kos = []
        # Smaller structured generations are much more reliable for local 2B models.
        MAX_KOS_PER_CALL = 10
        batch_idx = batch_offset

        while len(fake_kos) < target_ko_count:
            remaining = target_ko_count - len(fake_kos)
            batch_size = min(MAX_KOS_PER_CALL, remaining)
            
            # 1. Sample documents for THIS batch
            # Vary salt per batch to ensure different documents are used
            # We use generation_secret + batch_idx to create a unique deterministic salt
            batch_salt = f'style-analysis-batch-{batch_idx}'
            print(f"  Batch {batch_idx+1}: Sampling {num_examples} documents...")
            
            ko_style_samples = deterministic_sample(
                self.dataset, n=num_examples, 
                secret_key=self.generation_secret, 
                salt=batch_salt
            )
            
            # 2. Convert to real KOs (parallelized to reduce LLM wall time)
            total_docs = len(ko_style_samples)
            if self.abstract_workers == 1 or total_docs <= 1:
                batch_real_kos = []
                for idx, doc in enumerate(ko_style_samples, start=1):
                    print(f"    abstract_to_ko progress: {idx}/{total_docs}")
                    batch_real_kos.append(self._abstract_to_ko(doc))
            else:
                worker_count = min(self.abstract_workers, total_docs)
                print(f"    Converting {total_docs} docs with {worker_count} workers...")
                batch_real_kos = [None] * total_docs
                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_to_idx = {
                        executor.submit(self._abstract_to_ko, doc): idx
                        for idx, doc in enumerate(ko_style_samples)
                    }
                    completed = 0
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        batch_real_kos[idx] = future.result()
                        completed += 1
                        print(f"    abstract_to_ko progress: {completed}/{total_docs}")
            
            all_real_kos.extend(batch_real_kos)

            # 3. Generate fake KOs
            print(f"    Generating {batch_size} fake KOs (current total: {len(fake_kos)})...")
            batch_result = self._generate_fake_kos(batch_real_kos, num_to_generate=batch_size)
            
            if not batch_result:
                print("    Warning: No KOs generated in this batch. Stopping.")
                break
                
            fake_kos.extend(batch_result)
            batch_idx += 1
        
        # 4. Prepare KO pool data
        ko_pool_data = {
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "generation_secret": self.generation_secret,
                "dataset_used": "unknown",  # Will be set by caller
                "num_examples": num_examples,
                "target_ko_count": target_ko_count,
                "actual_ko_count": len(fake_kos),
                "abstract_model": getattr(self.abstract_llm_client, "model", "unknown"),
                "ko_generation_model": getattr(self.llm_client, "model", "unknown"),
                "abstract_workers": self.abstract_workers,
            },
            "real_kos": all_real_kos,
            "ko_pool": fake_kos
        }
        
        print(f"--- KO pool generation complete: {len(fake_kos)} KOs generated ---")
        return ko_pool_data
