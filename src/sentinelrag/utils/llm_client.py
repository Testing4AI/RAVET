"""
LLM client wrapper for watermark generation and detection.
"""

import ast
import json
import re
import time


def parse_json_content(content: str):
    """Parse common OpenAI-compatible JSON response wrappers safely."""

    stripped = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()

    candidates = [stripped]
    first_object = stripped.find("{")
    last_object = stripped.rfind("}")
    if 0 <= first_object < last_object:
        candidates.append(stripped[first_object : last_object + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    if first_object >= 0:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(stripped[first_object:])
            return parsed
        except json.JSONDecodeError:
            pass

    for candidate in candidates:
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    raise json.JSONDecodeError("Could not parse JSON response", stripped, 0)


class LLMClient:
    def __init__(self, client, model, async_client=None, llm_arg=None):
        """
        Args:
            client: OpenAI-compatible client
            model: Model name to use
            async_client: Async OpenAI-compatible client (optional)
            llm_arg: Default chat completion arguments from the model preset
        """
        self.client = client
        self.async_client = async_client
        self.model = model
        self.llm_arg = llm_arg or {}
    
    def ask_llm(self, prompt: str, is_json: bool = False, **kwargs):
        content = ""
        last_error = None
        for attempt in range(5):
            try:
                messages = [{"role": "user", "content": prompt}]
                response_format = {"type": "json_object"} if is_json else {"type": "text"}
                request_kwargs = {**self.llm_arg, **kwargs}
                request_kwargs.setdefault("response_format", response_format)
                
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **request_kwargs
                )
                
                content = completion.choices[0].message.content
                if content is None:
                    raise ValueError("Received None content from LLM")
                return parse_json_content(content) if is_json else content
            except Exception as e:  # noqa: BLE001 - provider SDK errors vary
                last_error = e
                wait_time = 5 * (2 ** attempt)
                print(f"LLM call failed (attempt {attempt+1}/5): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        raise RuntimeError(
            f"Too many LLM call failures. Last error: {last_error}. "
            f"\nResponse content: {content}"
        ) from last_error

    async def ask_llm_async(self, prompt: str, is_json: bool = False, **kwargs):
        """Async version of ask_llm"""
        if not self.async_client:
            raise ValueError("Async client not initialized")
            
        content = ""
        import asyncio
        last_error = None
        for attempt in range(5):
            try:
                messages = [{"role": "user", "content": prompt}]
                response_format = {"type": "json_object"} if is_json else {"type": "text"}
                request_kwargs = {**self.llm_arg, **kwargs}
                request_kwargs.setdefault("response_format", response_format)
                
                completion = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **request_kwargs
                )
                
                content = completion.choices[0].message.content
                if content is None:
                    raise ValueError("Received None content from LLM")
                return parse_json_content(content) if is_json else content
            except Exception as e:  # noqa: BLE001 - provider SDK errors vary
                last_error = e
                wait_time = 5 * (2 ** attempt)  # Exponential backoff: 5, 10, 20, 40, 80
                print(f"Async LLM call failed (attempt {attempt+1}/5): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        raise RuntimeError(
            f"Too many Async LLM call failures. Last error: {last_error}. "
            f"Prompt: {prompt}\nResponse content: {content}"
        ) from last_error
