"""Outcome-blind private graph and probe acceptance for ownership pilots."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping

from sentinelrag.testing.adaptive import stable_seed

SYLLABLES = (
    "ba", "be", "bi", "bo", "da", "de", "di", "do", "fa", "fe", "fi", "fo",
    "ga", "ge", "gi", "go", "ka", "ke", "ki", "ko", "la", "le", "li", "lo",
    "ma", "me", "mi", "mo", "na", "ne", "ni", "no",
)


def identity(rng: random.Random) -> str:
    return " ".join("".join(rng.choice(SYLLABLES) for _ in range(4)).title() for _ in range(2))


def private_graph(source: Mapping, *, block: str, index: int, seed: int, split: str) -> dict:
    rng = random.Random(stable_seed("private-graph-v1", seed, block, index))
    entities = source.get("entities", source.get("core_entities", []))
    relationships = source.get("relationships", [])
    if "triplets" in source:
        names = list(dict.fromkeys(str(t[k]) for t in source["triplets"] for k in ("entity_1", "entity_2")))
        entities = [{"id": name, "type": "Concept"} for name in names]
        relationships = [{"source": t["entity_1"], "target": t["entity_2"],
                          "relationship": t["relationship"]} for t in source["triplets"]]
    aliases, renamed, used = {}, [], set()
    for entity in entities:
        old_id = str(entity.get("id", entity.get("name", "")))
        if not old_id or old_id in aliases:
            raise ValueError("Missing or duplicate source entity identity")
        name = identity(rng)
        while name in used:
            name = identity(rng)
        used.add(name)
        aliases[old_id] = name
        renamed.append({"id": old_id, "name": name, "type": str(entity.get("type", "Entity"))})
    facts, seen = [], set()
    for edge in relationships:
        subject = str(edge.get("source", edge.get("subject", "")))
        obj = str(edge.get("target", edge.get("object", "")))
        predicate = str(edge.get("relationship", edge.get("predicate", edge.get("type", edge.get("relation", "")))))
        key = subject, predicate, obj
        if subject not in aliases or obj not in aliases or subject == obj or not predicate or key in seen:
            continue
        seen.add(key)
        facts.append({"subject": aliases[subject], "relation": predicate.replace("_", " "),
                      "object": aliases[obj]})
    if len(facts) < 5:
        raise ValueError(f"Source graph {block}/{index} has fewer than five valid relations")
    # Keep graph order fixed; do not select facts using retrieval or target answers.
    facts = facts[:5]
    return {"unit_id": f"private-{split}-c{index:03d}", "block": block, "split": split,
            "source_index": index, "entities": renamed, "facts": facts,
            "alias_entropy_bits_per_identity": 40}


def check_identity_bank(units, ordinary_texts):
    corpus = "\n".join(ordinary_texts).casefold()
    names = [e["name"] for u in units for e in u["entities"]]
    if len(names) != len(set(names)):
        raise ValueError("Synthetic identities collide across units or splits")
    collisions = [name for name in names if name.casefold() in corpus]
    if collisions:
        raise ValueError("Synthetic identity occurs in ordinary/source corpus")
    return {"units": len(units), "identities": len(names), "corpus_collisions": 0,
            "scope": "Exact local corpus collision check, not a global nonexistence certificate"}


def validate_document(text, unit):
    if not isinstance(text, str) or len(text.split()) < 40:
        raise ValueError("Carrier must be a nonempty standalone paragraph")
    for fact in unit["facts"]:
        if fact["subject"] not in text or fact["object"] not in text:
            raise ValueError("Carrier omits a private relation endpoint")
    return text.strip()


def validate_questions(value, unit):
    rows = value.get("questions") if isinstance(value, dict) else None
    if not isinstance(rows, list) or len(rows) != len(unit["facts"]):
        raise ValueError("Need exactly one question per registered private relation")
    result = []
    for index, (row, fact) in enumerate(zip(rows, unit["facts"], strict=True)):
        if not isinstance(row, dict):
            raise ValueError("Question row is not an object")
        question, answer = row.get("question"), row.get("correct_answer")
        if not isinstance(question, str) or answer != fact["object"]:
            raise ValueError("Question changed the private answer contract")
        if fact["subject"] not in question:
            raise ValueError("Question does not identify its private subject")
        # Both answer words are random; even partial disclosure must be rejected.
        words = re.findall(r"\w+", question.casefold())
        if any(word.casefold() in words for word in answer.split()):
            raise ValueError("Question leaks its private expected answer")
        result.append({"probe_id": f"{unit['unit_id']}-p{index:02d}",
                       "unit_id": unit["unit_id"], "question": question.strip(),
                       "expected_answer": answer, "question_index": index,
                       "private_subject": fact["subject"], "private_relation": fact["relation"]})
    if len({row["question"] for row in result}) != len(result):
        raise ValueError("Duplicate question")
    return result


def question_prompt(document, unit):
    from sentinelrag.prompts.templates import PromptTemplates

    return PromptTemplates.generate_simple_verification_questions(document, len(unit["facts"])) + (
        "\nAcceptance contract: questions MUST follow the supplied relations in exactly this order. "
        "For each relation, include the subject's full two-word private name and ask for the object's "
        "full name. Set correct_answer to that exact object name. Do not reveal either word of the "
        "object name in the question. Do not ask about general background knowledge.\n" +
        json.dumps(unit["facts"], ensure_ascii=True))


def relation_audit_prompt(baseline, optimized, unit, questions):
    return (
        "Check two supplied paragraphs against the supplied private graph and questions. "
        "These are invented benchmark entities: assess internal textual support, not real-world truth. "
        "For each of the five facts in order, return whether each paragraph unambiguously supports "
        "the subject-relation-object association AND whether the corresponding question asks for "
        "that object of that subject. Literal endpoint presence alone is insufficient. "
        "Return JSON only: {\"baseline\":[true,true,true,true,true],"
        "\"optimized\":[true,true,true,true,true],\"questions\":[true,true,true,true,true]}.\n" +
        json.dumps({"facts": unit["facts"], "questions": questions,
                    "baseline": baseline, "optimized": optimized}, ensure_ascii=True))


def validate_relation_audit(value, count=5):
    if not isinstance(value, dict):
        raise ValueError("Invalid support audit")
    for key in ("baseline", "optimized", "questions"):
        rows = value.get(key)
        if not isinstance(rows, list) or len(rows) != count or not all(v is True for v in rows):
            raise ValueError(f"Unconfirmed semantic contract: {key}")
    return value
