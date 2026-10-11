"""Canary artifact adapters shared by retrieval and black-box experiments."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sentinelrag.testing.prioritization import ProbeCandidate
from sentinelrag.utils.io import load_json


@dataclass(frozen=True)
class CanaryArtifact:
    """One independent secret unit, potentially spanning multiple documents."""

    canary_id: str
    scheme: str
    documents: tuple[str, ...]
    probes: tuple[ProbeCandidate, ...]
    metadata: dict[str, Any]


def _probe(
    *,
    canary_id: str,
    scheme: str,
    index: int,
    question: str,
    answer: str,
    documents: Sequence[str],
    required_document_indices: Sequence[int],
    required_evidence: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> ProbeCandidate:
    probe_metadata = {
        "scheme": scheme,
        "source_canary_id": canary_id,
        "documents": list(documents),
        "required_document_indices": list(required_document_indices),
        "required_evidence": list(required_evidence),
    }
    if metadata:
        probe_metadata.update(metadata)
    return ProbeCandidate(
        probe_id=f"{scheme}-{canary_id}-p{index:02d}",
        canary_id=f"{scheme}-{canary_id}",
        question=question,
        expected_answer=answer,
        canary_text="\n".join(documents),
        metadata=probe_metadata,
    )


def load_factual_artifacts(path: str | Path) -> list[CanaryArtifact]:
    """Load SentinelRAG output as single-document factual canaries."""

    payload = load_json(path)
    texts = payload["watermark_texts"]
    groups = payload["watermark_questions"]
    artifacts = []
    for group_index, group in enumerate(groups):
        source_index = int(group.get("watermark_index", group_index))
        canary_id = f"c{source_index:03d}"
        documents = (texts[source_index].strip(),)
        probes = tuple(
            _probe(
                canary_id=canary_id,
                scheme="factual",
                index=probe_index,
                question=question.strip(),
                answer=answer.strip(),
                documents=documents,
                required_document_indices=(0,),
                required_evidence=(answer.strip(),),
            )
            for probe_index, (question, answer) in enumerate(
                zip(group.get("questions", []), group.get("correct_answers", []))
            )
        )
        if probes:
            artifacts.append(
                CanaryArtifact(
                    canary_id=f"factual-{canary_id}",
                    scheme="factual",
                    documents=documents,
                    probes=probes,
                    metadata={"source_index": source_index},
                )
            )
    return artifacts


def derive_lexical_artifacts(
    factual_artifacts: Sequence[CanaryArtifact],
) -> list[CanaryArtifact]:
    """Derive rare-code canaries while preserving the source document style."""

    templates = (
        "What archival verification code is assigned to {record}?",
        "Which verification code identifies {record}?",
        "Give the archive code recorded for {record}.",
        "Under what verification code is {record} catalogued?",
        "What is the exact audit code associated with {record}?",
    )
    artifacts = []
    for source in factual_artifacts:
        digest = hashlib.sha256("\n".join(source.documents).encode("utf-8")).hexdigest()
        record = f"Record-{digest[:8].upper()}"
        code = f"VX-{digest[8:14].upper()}-{digest[14:18].upper()}"
        documents = (
            (
                f"{source.documents[0]} This entry is catalogued as {record}. "
                f"The archival verification code assigned to {record} is {code}."
            ),
        )
        canary_id = source.metadata.get("source_index", source.canary_id)
        source_id = f"c{int(canary_id):03d}" if isinstance(canary_id, int) else str(canary_id)
        probes = tuple(
            _probe(
                canary_id=source_id,
                scheme="lexical",
                index=index,
                question=template.format(record=record),
                answer=code,
                documents=documents,
                required_document_indices=(0,),
                required_evidence=(code,),
            )
            for index, template in enumerate(templates)
        )
        artifacts.append(
            CanaryArtifact(
                canary_id=f"lexical-{source_id}",
                scheme="lexical",
                documents=documents,
                probes=probes,
                metadata={"record": record, "code": code},
            )
        )
    return artifacts


def derive_compositional_artifacts(
    factual_artifacts: Sequence[CanaryArtifact],
) -> list[CanaryArtifact]:
    """Derive two-document canaries whose answer requires both support links."""

    templates = (
        "What verification value is linked to {subject} in {project}?",
        "Under {project}, which value ultimately corresponds to {subject}?",
        "Resolve the evidence token for {subject}; what value does it carry?",
        "What is the final audit value for {subject} in the {project} registry?",
        "Following the registry link from {subject}, which verification value is recorded?",
    )
    artifacts = []
    for source in factual_artifacts:
        digest = hashlib.sha256(
            ("\n".join(source.documents) + "compositional").encode("utf-8")
        ).hexdigest()
        project = f"Project-{digest[:7].upper()}"
        subject = f"Subject-{digest[7:14].upper()}"
        bridge = f"Token-{digest[14:22].upper()}"
        value = f"CV-{digest[22:30].upper()}"
        documents = (
            f"Within the {project} registry, {subject} is linked to evidence identifier {bridge}.",
            f"The registry entry for evidence identifier {bridge} records verification value {value}.",
        )
        canary_id = source.metadata.get("source_index", source.canary_id)
        source_id = f"c{int(canary_id):03d}" if isinstance(canary_id, int) else str(canary_id)
        probes = tuple(
            _probe(
                canary_id=source_id,
                scheme="compositional",
                index=index,
                question=template.format(
                    subject=subject, project=project, bridge=bridge
                ),
                answer=value,
                documents=documents,
                required_document_indices=(0, 1),
                required_evidence=(bridge, value),
            )
            for index, template in enumerate(templates)
        )
        artifacts.append(
            CanaryArtifact(
                canary_id=f"compositional-{source_id}",
                scheme="compositional",
                documents=documents,
                probes=probes,
                metadata={
                    "project": project,
                    "subject": subject,
                    "bridge": bridge,
                    "value": value,
                },
            )
        )
    return artifacts


def _source_id(source: CanaryArtifact) -> str:
    canary_id = source.metadata.get("source_index", source.canary_id)
    return f"c{int(canary_id):03d}" if isinstance(canary_id, int) else str(canary_id)


def _content_keywords(text: str, limit: int = 7) -> str:
    stopwords = {
        "what", "which", "where", "when", "whose", "does", "did", "was",
        "were", "with", "from", "that", "this", "into", "about", "according",
        "used", "using", "have", "has", "been", "being", "their", "they",
    }
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    selected = []
    for token in tokens:
        normalized = token.casefold()
        if len(normalized) < 4 or normalized in stopwords or normalized in selected:
            continue
        selected.append(normalized)
    return ", ".join(selected[:limit]) or "the archived topic"


def derive_lexical_v2_artifacts(
    factual_artifacts: Sequence[CanaryArtifact],
) -> list[CanaryArtifact]:
    """Derive contextual code canaries without a unique identifier in the query."""

    artifacts = []
    for source in factual_artifacts:
        digest = hashlib.sha256(
            ("\n".join(source.documents) + "lexical-v2").encode("utf-8")
        ).hexdigest()
        code = f"LX-{digest[:6].upper()}-{digest[6:10].upper()}"
        primary = (list(source.probes) * 5)[:5][0]
        topic_keywords = _content_keywords(primary.question, limit=9)
        document = (
            f"{source.documents[0]} The archived topic covering {topic_keywords}, "
            f"whose recorded fact is {primary.expected_answer}, has internal "
            f"verification code {code}."
        )
        source_id = _source_id(source)
        source_probes = (list(source.probes) * 5)[:5]
        first = source_probes[0]
        last = source_probes[-1]
        anchors = [
            f'the passage that answers the question "{first.question}"',
            f'the record containing the fact "{first.expected_answer}"',
            f'the archived topic addressed by "{source_probes[2].question}"',
            f"the entry about {_content_keywords(first.question)}",
            (
                f'the record whose answer to "{last.question}" is '
                f'"{last.expected_answer}"'
            ),
        ]
        questions = [
            f"What internal verification code is assigned to {anchor}?"
            for anchor in anchors
        ]
        probes = tuple(
            _probe(
                canary_id=source_id,
                scheme="lexical_v2",
                index=index,
                question=question,
                answer=code,
                documents=(document,),
                required_document_indices=(0,),
                required_evidence=(code,),
                metadata={"probe_level": index, "anchor": anchors[index]},
            )
            for index, question in enumerate(questions)
        )
        artifacts.append(
            CanaryArtifact(
                canary_id=f"lexical_v2-{source_id}",
                scheme="lexical_v2",
                documents=(document,),
                probes=probes,
                metadata={
                    "source_canary_id": source.canary_id,
                    "code": code,
                    "benchmark_version": "v2",
                    "benchmark_revision": 2,
                    "topic_keywords": topic_keywords,
                },
            )
        )
    return artifacts


def derive_compositional_v2_artifacts(
    factual_artifacts: Sequence[CanaryArtifact],
) -> list[CanaryArtifact]:
    """Derive two-hop canaries with topic-sharing support and distractor docs."""

    artifacts = []
    for source in factual_artifacts:
        digest = hashlib.sha256(
            ("\n".join(source.documents) + "compositional-v2").encode("utf-8")
        ).hexdigest()
        bridge = f"Route-{digest[:8].upper()}"
        value = f"CV-{digest[8:16].upper()}"
        distractor_bridge_1 = f"Route-{digest[16:24].upper()}"
        distractor_value_1 = f"CV-{digest[24:32].upper()}"
        distractor_bridge_2 = f"Route-{digest[32:40].upper()}"
        distractor_value_2 = f"CV-{digest[40:48].upper()}"
        source_probes = (list(source.probes) * 5)[:5]
        primary = source_probes[0]
        distractor_one = source_probes[1]
        distractor_two = source_probes[2]
        topic_keywords = _content_keywords(primary.question, limit=9)
        documents = (
            (
                f"{source.documents[0]} For the archived topic covering "
                f"{topic_keywords}, whose recorded fact is {primary.expected_answer}, "
                f"the evidence route is {bridge}."
            ),
            (
                f"For the archive topic covering {topic_keywords} and the fact "
                f"{primary.expected_answer}, evidence route {bridge} maps to final "
                f"audit value {value}."
            ),
            (
                f'A superseded note concerns "{distractor_one.question}". '
                f"It maps evidence route {distractor_bridge_1} to obsolete value "
                f"{distractor_value_1}."
            ),
            (
                f'An unrelated note concerns "{distractor_two.question}". '
                f"It maps evidence route {distractor_bridge_2} to obsolete value "
                f"{distractor_value_2}."
            ),
        )
        source_id = _source_id(source)
        probes = tuple(
            _probe(
                canary_id=source_id,
                scheme="compositional_v2",
                index=index,
                question=(
                    f'For the archived topic addressed by "{source_probe.question}", '
                    "what final audit value is reached through its evidence route?"
                ),
                answer=value,
                documents=documents,
                required_document_indices=(0, 1),
                required_evidence=(bridge, value),
                metadata={
                    "probe_level": index,
                    "source_probe_id": source_probe.probe_id,
                    "distractor_document_indices": [2, 3],
                },
            )
            for index, source_probe in enumerate(source_probes)
        )
        artifacts.append(
            CanaryArtifact(
                canary_id=f"compositional_v2-{source_id}",
                scheme="compositional_v2",
                documents=documents,
                probes=probes,
                metadata={
                    "source_canary_id": source.canary_id,
                    "bridge": bridge,
                    "value": value,
                    "distractor_values": [distractor_value_1, distractor_value_2],
                    "benchmark_version": "v2",
                    "benchmark_revision": 2,
                    "topic_keywords": topic_keywords,
                },
            )
        )
    return artifacts


def build_artifact_pools(path: str | Path) -> dict[str, list[CanaryArtifact]]:
    factual = load_factual_artifacts(path)
    return {
        "factual": factual,
        "lexical": derive_lexical_artifacts(factual),
        "compositional": derive_compositional_artifacts(factual),
        "lexical_v2": derive_lexical_v2_artifacts(factual),
        "compositional_v2": derive_compositional_v2_artifacts(factual),
    }


def save_artifact_pools(
    pools: dict[str, Sequence[CanaryArtifact]], path: str | Path
) -> None:
    payload = {
        "schemes": {
            scheme: [asdict(artifact) for artifact in artifacts]
            for scheme, artifacts in pools.items()
        }
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
