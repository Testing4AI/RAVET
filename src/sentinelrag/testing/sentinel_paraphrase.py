"""Blind carrier paraphrasing and cumulative decision summaries."""

import json
import re

from sentinelrag.testing.corpus_migrations import text_sha256
from sentinelrag.testing.ragwm import evidence_curve


def paraphrase_prompt(documents):
    rows = [{"id": text_sha256(text), "text": text} for text in documents]
    return (
        "Paraphrase each document independently using different wording, syntax and sentence organization. "
        "Preserve its full factual meaning, named entities, numerical values, relations and qualifications. "
        "Do not summarize, omit facts, invent facts, add explanations, or convert the text into questions and answers. "
        "Return only a JSON object mapping every source id to a nonempty paraphrased document string.\n"
        "Source documents:\n" + json.dumps(rows, ensure_ascii=True)
    )


def validate_paraphrases(value, documents):
    expected = {text_sha256(text): text for text in documents}
    if not isinstance(value, dict) or value.keys() != expected.keys():
        raise ValueError("Paraphrase identities differ from frozen sources")
    normalized = lambda text: re.sub(r"\s+", " ", text).strip().casefold()
    for key, source in expected.items():
        text = value[key]
        if not isinstance(text, str) or not text.strip() or normalized(text) == normalized(source):
            raise ValueError("Empty or unchanged paraphrase")
    return {key: text.strip() for key, text in value.items()}


def cumulative_metrics(labels, p0, alpha, horizons=(6, 15, 30)):
    curve = evidence_curve(labels, p0, alpha)["curve"]
    result = {}
    for horizon in horizons:
        if horizon > len(curve):
            raise ValueError("Observation window exceeds distinct evidence bank")
        result[f"vsr{horizon}"] = int(curve[horizon - 1]["anytime_verified"])
        result[f"mq{horizon}"] = sum(
            1 - (int(curve[i - 1]["anytime_verified"]) if i else 0)
            for i in range(horizon)
        )
    return result, curve
