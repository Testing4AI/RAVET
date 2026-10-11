#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watermark Injector

Selects KOs from a pool and expands them into watermark texts for injection.
"""

import json
import asyncio
from datetime import datetime

from sentinelrag.utils import deterministic_sample, deterministic_select_kos
from sentinelrag.prompts import PromptTemplates


class WatermarkInjector:
    """Watermark injector based on pre-generated KO pool"""

    def __init__(
        self,
        llm_client,
        secret_key,
        llm_kwargs: dict = None,
        stealth_mode: str = "normal",
    ):
        """
        Initialize watermark injector
        
        Args:
            llm_client: LLM client for text generation
            secret_key: User secret key for deterministic KO selection
            llm_kwargs: Additional kwargs to pass to llm_client.ask_llm()
            stealth_mode: Prompt style control for LLM expansion: normal|mild|strong
        """
        self.llm_client = llm_client
        self.secret_key = secret_key
        self.llm_kwargs = llm_kwargs or {}
        self.stealth_mode = (stealth_mode or "normal").strip().lower()
        if self.stealth_mode not in {"normal", "mild", "strong"}:
            raise ValueError("stealth_mode must be one of: normal, mild, strong")

    def _expand_ko_to_text(self, fake_ko, style_examples):
        """Expand fake KO to long text watermark"""
        fake_ko_str = json.dumps(fake_ko, indent=2, ensure_ascii=False)

        # Build style sample string
        examples_str = ""
        for i, example in enumerate(style_examples, 1):
            examples_str += f"--- Writing Example {i} ---\n{example.strip()}\n\n"

        prompt = PromptTemplates.expand_ko_to_text(
            fake_ko_str,
            examples_str,
            stealth_mode=self.stealth_mode,
        )
        return self.llm_client.ask_llm(prompt, **self.llm_kwargs)

    async def _expand_ko_to_text_async(self, ko, style_examples):
        """Async wrapper for expanding fake KO to long text watermark"""
        return await asyncio.to_thread(self._expand_ko_to_text, ko, style_examples)

    def _sample_style_example_groups(self, dataset, num_groups: int, sample_count: int = 3):
        """Sample style examples once and split into groups.

        This performs a single deterministic sample of `num_groups * sample_count` items,
        then splits the sampled items into `num_groups` groups.
        """
        total_needed = num_groups * sample_count
        if total_needed <= 0:
            return []

        dataset_size = len(dataset)
        if dataset_size == 0:
            return [[] for _ in range(num_groups)]

        sampled_count = min(total_needed, dataset_size)
        sampled_items = deterministic_sample(
            dataset,
            n=sampled_count,
            secret_key=self.secret_key,
            salt='style-for-writing-batch'
        )

        groups = []
        for i in range(num_groups):
            start = i * sample_count
            end = start + sample_count
            group = sampled_items[start:end]

            # If dataset is smaller than total_needed, deterministically wrap around.
            while len(group) < sample_count:
                wrap_idx = (start + len(group)) % len(sampled_items)
                group.append(sampled_items[wrap_idx])

            groups.append(group)

        return groups

    async def _inject_watermarks_async(self, ko_pool_data, dataset, num_select_kos: int = 3):
        """
        Async version of inject_watermarks
        """
        print("--- Starting KO pool-based watermark injection (Async) ---")
        
        # 1. Deterministically select KOs from pool based on secret key
        ko_pool = ko_pool_data['ko_pool']
        
        selected_kos = deterministic_select_kos(
            ko_pool, num_select_kos, self.secret_key, salt='watermark-injection'
        )
        print(f"  Selected {len(selected_kos)} KOs from pool")
        
        # 2. Expand each KO to watermark text - ASYNC
        print(f"  Expanding {len(selected_kos)} KOs to watermark text in parallel...")

        style_examples_for_writing = await asyncio.to_thread(
            self._sample_style_example_groups,
            dataset,
            len(selected_kos),
        )

        async def _expand_one_ko(index, ko_item, ko_style_examples):
            result = await self._expand_ko_to_text_async(ko_item, ko_style_examples)
            return index, result

        tasks = [
            asyncio.create_task(_expand_one_ko(i, ko, style_examples_for_writing[i]))
            for i, ko in enumerate(selected_kos)
        ]

        total_tasks = len(tasks)
        completed_tasks = 0
        expansion_start = datetime.now()
        results_by_index = [None] * total_tasks

        for completed_task in asyncio.as_completed(tasks):
            idx, result = await completed_task
            results_by_index[idx] = result
            completed_tasks += 1
            elapsed = (datetime.now() - expansion_start).total_seconds()
            progress_pct = (completed_tasks / total_tasks * 100) if total_tasks else 100.0
            print(
                f"    Progress: {completed_tasks}/{total_tasks} "
                f"({progress_pct:.1f}%) completed, elapsed {elapsed:.1f}s"
            )
        
        watermark_texts = []
        watermark_expansion_data = []

        for i, (ko, result) in enumerate(zip(selected_kos, results_by_index)):
            wm_text = result
            print(f"    KO {i+1}: Generated text with LLM")

            expansion_record = {
                "ko_index": i,
                "ko": ko,
                "watermark_text": wm_text,
                "expansion_timestamp": datetime.now().isoformat(),
                "method": "llm_expansion"
            }
            watermark_expansion_data.append(expansion_record)
            watermark_texts.append(wm_text)
        
        # 4. Prepare injection result data
        injection_metadata = {
            "secret_key_hash": hash(self.secret_key),  # Don't save raw key, only hash
            "injection_timestamp": datetime.now().isoformat(),
            "ko_pool_source": ko_pool_data.get('metadata', {}),
            "num_selected_kos": len(selected_kos),
            "num_final_watermarks": len(watermark_texts),
            "watermark_method": "llm_expansion",
            "stealth_mode": self.stealth_mode,
        }
        
        injection_result = {
            "injection_metadata": injection_metadata,
            "selected_kos": selected_kos,
            "watermark_texts": watermark_texts,
            "style_examples_for_writing": style_examples_for_writing,
            "watermark_expansion_data": watermark_expansion_data,
            "injection_stats": {
                "kos_selected": len(selected_kos),
                "watermarks_generated": len(watermark_texts),
                "success_rate": len(watermark_texts) / len(selected_kos) if selected_kos else 0
            }
        }
        
        print(f"--- Watermark injection complete: {len(watermark_texts)} watermarks generated ---")
        return injection_result

    async def inject_watermarks_async(self, ko_pool_data, dataset, num_select_kos: int = 3):
        """
        Async version of inject_watermarks - use when already in an async context
        
        Args:
            ko_pool_data: KO pool data (from KOPoolGenerator)
            dataset: Dataset (for style sampling)
            num_select_kos: Number of KOs to select
            
        Returns:
            dict: Injection results containing selected KOs, watermark texts, etc.
        """
        return await self._inject_watermarks_async(ko_pool_data, dataset, num_select_kos)

    def inject_watermarks(self, ko_pool_data, dataset, num_select_kos: int = 3):
        """
        Select and inject watermarks from KO pool (Wrapper for async execution)
        
        Args:
            ko_pool_data: KO pool data (from KOPoolGenerator)
            dataset: Dataset (for style sampling)
            num_select_kos: Number of KOs to select
            
        Returns:
            dict: Injection results containing selected KOs, watermark texts, etc.
        """
        return asyncio.run(self._inject_watermarks_async(ko_pool_data, dataset, num_select_kos))


def inject_to_vectorstore(vectorstore, watermark_texts, logger=None):
    """
    Inject watermarks into vector database
    
    Args:
        vectorstore: VectorStore instance
        watermark_texts: List of watermark text strings
        logger: Optional logger instance
        
    Returns:
        list: IDs of injected watermark documents
    """
    watermark_ids = []
    
    for i, wm_text in enumerate(watermark_texts):
        try:
            wid = vectorstore.inject_direct(wm_text)
            watermark_ids.append(wid)
            if logger:
                logger.info(f'Watermark document #{i+1} added, ID: {wid}')
        except Exception as e:
            if logger:
                logger.error(f"Error adding watermark document: {e}")
    
    return watermark_ids
