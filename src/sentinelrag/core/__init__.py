#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core business logic for watermark generation, injection, and detection.
"""

from sentinelrag.core.ko_pool import KOPoolGenerator
from sentinelrag.core.injector import WatermarkInjector
from sentinelrag.core.detector import (
    RAGWatermarkDetector,
    perform_watermark_detection,
    save_detection_results,
    generate_all_questions,
    test_questions_on_database,
    calculate_overall_results,
    inject_watermarks,
    cleanup_watermarks,
    paraphrase_text,
    run_paraphrase_attack,
    run_translate_attack,
    answer_contains_entity_pair,
)
from sentinelrag.core.interference import (
    InterferenceEvaluator,
    setup_llm_client,
    setup_vectorstore,
    load_main_questions,
)
from sentinelrag.core.attack import (
    remove_unrelated_content,
    remove_unrelated_content_sync,
    perform_unrelated_content_attack,
    perform_knowledge_expansion_attack,
)

__all__ = [
    "KOPoolGenerator",
    "WatermarkInjector",
    "RAGWatermarkDetector",
    "perform_watermark_detection",
    "save_detection_results",
    "generate_all_questions",
    "test_questions_on_database",
    "calculate_overall_results",
    "inject_watermarks",
    "cleanup_watermarks",
    "paraphrase_text",
    "run_paraphrase_attack",
    "run_translate_attack",
    "answer_contains_entity_pair",
    "InterferenceEvaluator",
    "setup_llm_client",
    "setup_vectorstore",
    "load_main_questions",
    # Attack functions
    "remove_unrelated_content",
    "remove_unrelated_content_sync",
    "perform_unrelated_content_attack",
    "perform_knowledge_expansion_attack",
]
