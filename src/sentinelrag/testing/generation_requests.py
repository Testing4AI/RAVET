"""Bounded, durable requests for expensive structured fixture generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

from openai import APIError

from sentinelrag.utils.llm_client import parse_json_content


class BoundedGenerationClient:
    def __init__(self, client, preset, output_dir, *, timeout=120, max_tokens=8192):
        self.client = client.client.with_options(max_retries=0)
        self.model = preset.model
        self.args = {
            **preset.llm_arg,
            "response_format": {"type": "text"},
            "max_tokens": max_tokens,
        }
        self.timeout = timeout
        self.output_dir = Path(output_dir)

    def ask_llm(self, prompt, is_json=False, **kwargs):
        effective = {**self.args, **kwargs}
        timeout = effective.pop("timeout", self.timeout)
        key = hashlib.sha256(
            json.dumps(
                {"model": self.model, "args": effective, "prompt": prompt},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        path = self.output_dir / key[:2] / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            assert record["key"] == key and record["prompt"] == prompt
            return (
                parse_json_content(record["content"]) if is_json else record["content"]
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        for previous in path.parent.glob(f"{key}.attempt-*.json"):
            record = json.loads(previous.read_text(encoding="utf-8"))
            if record["finish_reason"] == "length":
                raise RuntimeError(
                    "Prior request hit the token cap; retain the failure instead of retrying truncated generation on resume"
                )
        for attempt in range(3):
            started = time.monotonic()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout,
                    **effective,
                )
                choice = response.choices[0]
                record = {
                    "key": key,
                    "model": self.model,
                    "returned_model": response.model,
                    "prompt": prompt,
                    "args": effective,
                    "content": choice.message.content,
                    "finish_reason": choice.finish_reason,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "elapsed_seconds": time.monotonic() - started,
                    "attempt": attempt + 1,
                }
                attempt_path = path.with_name(f"{key}.attempt-{attempt + 1}.json")
                attempt_path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                if choice.finish_reason == "length":
                    raise RuntimeError(
                        "Structured generation hit the token cap; raw response retained, do not accept truncation"
                    )
                if not choice.message.content:
                    raise ValueError("Empty structured generation")
                value = (
                    parse_json_content(choice.message.content)
                    if is_json
                    else choice.message.content
                )
                temporary = path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(path)
                print(
                    f"[generation request] {self.model} {key[:10]} complete", flush=True
                )
                return value
            except (APIError, ValueError) as error:
                error_path = path.with_name(f"{key}.error-{attempt + 1}.json")
                error_path.write_text(
                    json.dumps(
                        {
                            "key": key,
                            "model": self.model,
                            "attempt": attempt + 1,
                            "error_type": type(error).__name__,
                            "status_code": getattr(error, "status_code", None),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
                    + "\n"
                )
                print(
                    f"[generation retry] {self.model} {type(error).__name__} {attempt + 1}/3",
                    flush=True,
                )
                if attempt == 2:
                    raise RuntimeError(
                        f"Bounded generation failed: {self.model} {key[:10]} {type(error).__name__}"
                    ) from None
                delay = 5 * 2**attempt
                if getattr(error, "status_code", None) == 429:
                    delay = max(30, delay)
                    response = getattr(error, "response", None)
                    if response is not None:
                        retry_after = response.headers.get("retry-after", "")
                        if retry_after.replace(".", "", 1).isdigit():
                            delay = max(delay, float(retry_after))
                time.sleep(delay)
        raise AssertionError("Unreachable")

    async def ask_llm_async(self, prompt, is_json=False, **kwargs):
        return await asyncio.to_thread(self.ask_llm, prompt, is_json=is_json, **kwargs)
