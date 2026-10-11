#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Path helpers for the SentinelRAG package."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent


def repo_root() -> Path:
    """Return the repository root used for local data and generated artifacts."""
    return Path(os.getenv("SENTINELRAG_ROOT", str(REPO_ROOT))).expanduser().resolve()


def default_datasets_dir() -> Path:
    return Path(os.getenv("SENTINELRAG_DATASETS_DIR", str(repo_root() / "datasets"))).expanduser().resolve()


def default_output_dir() -> Path:
    return Path(os.getenv("SENTINELRAG_OUTPUT_DIR", str(repo_root() / "output"))).expanduser().resolve()


def default_models_dir() -> Path:
    return Path(os.getenv("SENTINELRAG_MODELS_DIR", str(repo_root() / "models"))).expanduser().resolve()


def default_chroma_dir() -> Path:
    return Path(os.getenv("SENTINELRAG_CHROMA_DIR", str(repo_root() / "chromadb_db"))).expanduser().resolve()

