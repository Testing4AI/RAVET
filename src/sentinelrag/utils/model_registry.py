#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model preset registry for OpenAI-compatible LLM endpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai

from sentinelrag.utils.llm_client import LLMClient
from sentinelrag.utils.paths import default_models_dir


MODELS_DIR = default_models_dir()


@dataclass(frozen=True)
class ModelPreset:
    """Named preset describing one OpenAI-compatible model endpoint."""

    preset_name: str
    llm_url: str
    api_key: str
    model: str
    llm_arg: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], fallback_name: str) -> "ModelPreset":
        llm_url = data.get("llm_url")
        model = data.get("model")
        if not llm_url or not model:
            raise ValueError(
                f"Invalid model preset '{fallback_name}': 'llm_url' and 'model' are required."
            )
        llm_arg = data.get("llm_arg") or {}
        if not isinstance(llm_arg, dict):
            raise ValueError(
                f"Invalid model preset '{fallback_name}': 'llm_arg' must be an object."
            )
        return cls(
            preset_name=data.get("preset_name", fallback_name),
            llm_url=llm_url,
            api_key=data.get("api_key", ""),
            model=model,
            llm_arg=llm_arg,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelPreset":
        preset_path = Path(path)
        with preset_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        fallback_name = preset_path.stem
        return cls.from_dict(data, fallback_name=fallback_name)


def _preset_path(preset_name: str, models_dir: str | Path | None = None) -> Path:
    base_dir = Path(models_dir) if models_dir else MODELS_DIR
    return base_dir / f"{preset_name}.json"


def list_model_presets(models_dir: str | Path | None = None) -> list[str]:
    """Return all preset names available in the models directory."""
    base_dir = Path(models_dir) if models_dir else MODELS_DIR
    if not base_dir.exists():
        return []
    return sorted(path.stem for path in base_dir.glob("*.json"))


def load_model_preset(preset_name: str, models_dir: str | Path | None = None) -> ModelPreset:
    """Load a preset by name."""
    preset_path = _preset_path(preset_name, models_dir=models_dir)
    if not preset_path.exists():
        available = ", ".join(list_model_presets(models_dir=models_dir)) or "none"
        raise ValueError(
            f"Unknown model preset '{preset_name}'. Expected one of: {available}"
        )
    return ModelPreset.from_file(preset_path)


_ENV_API_KEY_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _resolve_api_key(raw_api_key: str | None) -> str | None:
    """Resolve literal or ${ENV_VAR} API keys from model presets."""
    if raw_api_key is None:
        return None

    api_key = raw_api_key.strip()
    if not api_key:
        return None

    match = _ENV_API_KEY_PATTERN.match(api_key)
    if not match:
        return api_key

    env_name = match.group(1)
    resolved = os.getenv(env_name)
    if resolved is None:
        raise ValueError(
            f"Model preset API key references '${{{env_name}}}', but that environment variable is not set."
        )
    return resolved


def create_llm_client_from_preset(
    preset_name: str,
    *,
    models_dir: str | Path | None = None,
    api_key: str | None = None,
    include_async: bool = True,
) -> tuple[LLMClient, ModelPreset]:
    """Build an LLMClient and return it together with the resolved preset."""
    preset = load_model_preset(preset_name, models_dir=models_dir)
    resolved_api_key = api_key if api_key is not None else _resolve_api_key(preset.api_key)
    client = openai.OpenAI(api_key=resolved_api_key, base_url=preset.llm_url)
    async_client = None
    if include_async:
        async_client = openai.AsyncOpenAI(api_key=resolved_api_key, base_url=preset.llm_url)
    return LLMClient(client, preset.model, async_client=async_client, llm_arg=preset.llm_arg), preset


def add_model_preset_arg(
    parser: argparse.ArgumentParser,
    *,
    flag: str,
    default_preset: str,
    help_text: str,
    models_dir: str | Path | None = None,
) -> None:
    """Add a standard preset-selection argument."""
    choices = list_model_presets(models_dir=models_dir)
    parser.add_argument(
        f"--{flag}",
        type=str,
        default=default_preset,
        choices=choices if choices else None,
        help=f"{help_text} (preset name in models/, default: {default_preset}; examples in models.example/)",
    )
