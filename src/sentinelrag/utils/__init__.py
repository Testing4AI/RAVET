#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified utility exports for watermark processing."""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULES = {
    # I/O
    "save_json": "sentinelrag.utils.io",
    "load_json": "sentinelrag.utils.io",
    "save_results": "sentinelrag.utils.io",
    "load_results": "sentinelrag.utils.io",
    "file_exist": "sentinelrag.utils.io",
    "create_file_if_not_exists": "sentinelrag.utils.io",
    "find_latest_injection_result": "sentinelrag.utils.io",
    "NpEncoder": "sentinelrag.utils.io",
    # LLM
    "LLMClient": "sentinelrag.utils.llm_client",
    "ModelPreset": "sentinelrag.utils.model_registry",
    "add_model_preset_arg": "sentinelrag.utils.model_registry",
    "list_model_presets": "sentinelrag.utils.model_registry",
    "load_model_preset": "sentinelrag.utils.model_registry",
    "create_llm_client_from_preset": "sentinelrag.utils.model_registry",
    # Embedding and model utilities
    "load_models": "sentinelrag.utils.embeddings",
    "contriever_get_emb": "sentinelrag.utils.embeddings",
    "dpr_get_emb": "sentinelrag.utils.embeddings",
    "ance_get_emb": "sentinelrag.utils.embeddings",
    "model_code_to_qmodel_name": "sentinelrag.utils.embeddings",
    "model_code_to_cmodel_name": "sentinelrag.utils.embeddings",
    # Dataset utilities
    "load_beir_datasets": "sentinelrag.utils.datasets",
    "load_dataset_for_watermark_generation": "sentinelrag.utils.datasets",
    "data_prepare": "sentinelrag.utils.datasets",
    # Sampling utilities
    "deterministic_sample": "sentinelrag.utils.sampling",
    "deterministic_select_kos": "sentinelrag.utils.sampling",
    "setup_seeds": "sentinelrag.utils.sampling",
    # Statistical utilities
    "binomial_test_greater": "sentinelrag.utils.stats",
    "f1_score": "sentinelrag.utils.stats",
    # Text processing utilities
    "clean_str": "sentinelrag.utils.text",
    "is_valid_json": "sentinelrag.utils.text",
    "extract_doc": "sentinelrag.utils.text",
    "extract_doc_list": "sentinelrag.utils.text",
    "find_substrings_containing_sentences": "sentinelrag.utils.text",
    "documents_hash": "sentinelrag.utils.text",
    "remove_duplicates_with_indices": "sentinelrag.utils.text",
    # Logging utilities
    "Log": "sentinelrag.utils.logging",
    # Paths
    "repo_root": "sentinelrag.utils.paths",
    "default_chroma_dir": "sentinelrag.utils.paths",
    "default_datasets_dir": "sentinelrag.utils.paths",
    "default_models_dir": "sentinelrag.utils.paths",
    "default_output_dir": "sentinelrag.utils.paths",
}

__all__ = [*_EXPORT_MODULES]


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module 'sentinelrag.utils' has no attribute '{name}'")

    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
