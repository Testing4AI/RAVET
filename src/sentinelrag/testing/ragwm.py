"""Auditable adapters for the released RAG-WM generation/injection mechanism."""

from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from pathlib import Path

from sentinelrag.utils.stats import bernoulli_mixture_e_value, binomial_test_greater


def upstream_prompts(path: Path) -> dict[str, str]:
    wanted = {
        "WATERMARK_GENERATE", "WATERMARK_GENERATE_FEEDBACK_BOTH",
        "WATERMARK_ASK_RAG", "WATERMARK_CHECK", "MULTIPLE_PROMPT",
    }
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    if not isinstance(node.value.value, str):
                        raise ValueError("Non-string upstream prompt")
                    values[target.id] = node.value.value
    if values.keys() != wanted:
        raise ValueError("Required upstream prompts missing")
    return values


def keyed_units(real_kos, secret: str, count: int, *, entity_limit=100, relation_limit=20):
    entity_counts, type_counts, relations = Counter(), Counter(), Counter()
    entity_types = {}
    observed = set()
    schema_counts = Counter()
    for ko in real_kos:
        nodes = ko.get("entities", ko.get("core_entities", []))
        edges = ko.get("relationships", ko.get("relations", []))
        if not isinstance(nodes, list) or not isinstance(edges, list):
            schema_counts["unsupported_graph_container"] += 1
            continue
        aliases = {}
        for entity in nodes:
            if not isinstance(entity, dict):
                schema_counts["unsupported_entity"] += 1
                continue
            name = entity.get("name", entity.get("id"))
            kind = entity.get("type", entity.get("category", "Entity"))
            if not isinstance(name, str) or not name.strip() or not isinstance(kind, str):
                schema_counts["unsupported_entity"] += 1
                continue
            name = name.strip()
            if isinstance(entity.get("id"), str):
                aliases[entity["id"]] = name
            aliases[name] = name
            entity_counts[name] += 1
            type_counts[kind] += 1
            entity_types.setdefault(name, kind)
        for edge in edges:
            if not isinstance(edge, dict):
                schema_counts["unsupported_edge"] += 1
                continue
            a = edge.get("subject", edge.get("source", edge.get("entity1")))
            b = edge.get("object", edge.get("target", edge.get("entity2")))
            relation = edge.get("predicate", edge.get("relationship", edge.get("relation", edge.get("type"))))
            if not all(isinstance(v, str) and v.strip() for v in (a, b, relation)):
                schema_counts["unsupported_edge"] += 1
                continue
            a, b = aliases.get(a, a), aliases.get(b, b)
            relations[relation] += 1
            observed.add((a, b, relation))
            schema_counts["accepted_edge"] += 1
    types = {name for name, _ in type_counts.most_common(20)}
    entities = [e for e, _ in sorted(entity_counts.items(), key=lambda p: (-p[1], p[0]))
                if entity_types[e] in types][:entity_limit]
    relation_names = [r for r, _ in sorted(relations.items(), key=lambda p: (-p[1], p[0]))][:relation_limit]
    if len(entities) < 2 or not relation_names:
        raise ValueError("Insufficient real entities or relations")
    # The released implementation hashes entity pairs plus the owner secret.
    pairs = sorted(((a, b) for i, a in enumerate(entities) for b in entities[i + 1:]),
                   key=lambda p: hashlib.sha256((p[0] + p[1] + secret).encode()).hexdigest())
    if len(pairs) < count:
        raise ValueError("Insufficient distinct entity pairs")
    units = []
    for index, (a, b) in enumerate(pairs[:count]):
        value = hashlib.sha256((a + b + secret).encode()).hexdigest()
        relation = relation_names[int(value, 16) % len(relation_names)]
        units.append({"unit_id": f"ragwm-{index:03d}", "e1": a, "e2": b, "relation": relation,
                      "observed_in_reference_graph": (a, b, relation) in observed})
    return {"units": units, "entities": entities, "relations": relation_names,
            "reference_entity_count": len(entity_counts), "reference_relation_count": len(relations),
            "schema_audit": dict(schema_counts),
            "pair_selection": "SHA256 keyed pair ranking; distinct pairs; no target-based filtering"}


def yes_no(text: str) -> bool:
    match = re.fullmatch(r"\s*(yes|no)[.!]?\s*", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError("Malformed semantic verdict, not a negative outcome")
    return match.group(1).lower() == "yes"


def feedback_text(value):
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict):
        raise TypeError("Invalid feedback schema")
    text = value.get("watermark_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty feedback text")
    return text


def eligible_variants(variants):
    both = [v for v in variants if v["direct_ok"] and v["shadow_ok"]]
    if both:
        return both
    direct = [v for v in variants if v["direct_ok"]]
    shadow = [v for v in variants if v["shadow_ok"]]
    return (direct or shadow or variants)[:1]


def evidence_curve(labels, p0: float, alpha: float):
    successes, first = 0, None
    rows = []
    for count, outcome in enumerate(labels, 1):
        successes += int(outcome)
        p = binomial_test_greater(successes, count, p0)
        e = bernoulli_mixture_e_value(successes, count, p0)
        if first is None and e >= 1 / alpha:
            first = count
        rows.append({"budget": count, "successes": successes, "edr": successes / count,
                     "p_value": p, "fixed_verified": p <= alpha,
                     "e_value": e, "anytime_verified": first is not None})
    return {"curve": rows, "first_anytime": first,
            "restricted_queries": first if first is not None else len(labels)}
