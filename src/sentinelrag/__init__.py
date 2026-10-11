#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SentinelRAG: watermark generation, injection, and detection for RAG corpora."""

from __future__ import annotations

from importlib import import_module


__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "KOPoolGenerator": "sentinelrag.core.ko_pool",
    "WatermarkInjector": "sentinelrag.core.injector",
    "RAGWatermarkDetector": "sentinelrag.core.detector",
}

__all__ = [*_LAZY_EXPORTS]


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'sentinelrag' has no attribute '{name}'")

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
