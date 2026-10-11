# Evidence Guide

## What Is Reproducible Here?

| Evidence | Included | Offline verification |
| --- | --- | --- |
| Primary effectiveness and injection-density study | All 144 paired protected configurations and 48 clean configurations | Replay boolean outcomes; recover all 1--15 query curves and capped query costs |
| Compression, with and without transfer | Both archived draws, including clean controls | Replay up to 30 distinct evidence identities |
| Sentinel paraphrase | Eight protected and eight clean configurations per method | Replay the frozen labels and original decision bound |
| Corpus expansion | 48 expanded and eight reference configurations, plus clean controls | Compare within each domain/factor and against its own reference |
| Retrieval depth and re-chunking | Archived completion trajectories and clean controls | Replay each deployment separately |
| Relation-explicit versus general construction | Ten numeric corpus/retriever comparisons in `construction.json` | Inspect support-hit improvements; source text is not redistributed |
| New model answers and semantic judgments | Not included | Require model access and licensed corpus inputs |
| Later, separately reported shadow-screening counts | Not included | No locally verifiable raw records were available for this export |

## Provenance and Decision Bounds

`trajectories.json` contains **observed oracle labels**, not regenerated answers or simulated successes. Each trajectory preserves execution order and evidence identity, with corpus, retriever, target, sampling replicate, ordering seed, and decision parameters. Question text, answer text, and retrieved context are omitted. Source-file hashes provide an export trail without redistributing those texts.

`reproduce.py` recomputes a 200-alternative Bernoulli mixture e-process, checks every archived first-crossing curve, and then checks condition-level aggregation. For the primary study, export also checks every query point against the original `condition_means.json`. This verifies statistical replay and aggregation, **not the correctness of the original semantic oracle**.

| Study | Original archived `p0` | Alpha |
| --- | ---: | ---: |
| Primary; retrieval/index changes | 0.15 | 0.01 |
| Compression and compression + transfer | 0.3004170799667139 | 0.01 |
| Paraphrase | 0.2883755391484502 | 0.01 |
| Knowledge expansion | 0.3906278341482568 | 0.01 |

These are the bounds attached to the exported trajectories. A later draft describes joint deployment calibration at 0.30; its external calibration records are not part of this snapshot. This release does not silently relabel the earlier results as a 0.30 experiment. The frozen-bank studies also precede the separate shadow-screening study. The archive therefore makes no claim that historical banks were screened using that later procedure.

The original protocol model identifiers are retained with infrastructure prefixes anonymized. Names in presentation prose should not override execution provenance. The public export does not retroactively substitute a different generation or judging model in historical records.

## Weighting and Interpretation

Baseline random-order seeds are averaged within a condition, and conditions then receive equal weight. An unsuccessful run incurs the full query window in MQ. Primary clean controls use the largest bank once per corpus/replicate/pipeline; they are not counted once for each positive density. Expansion has a separate unexpanded reference and must not be pooled with the primary intact comparison. Multiple corpus draws share source pools and are not independent new knowledge bases.

The two-method results compare original SentinelRAG carriers/default probes/random ordering with RAVET under the **same comparison-specific oracle and evidence rule**. They do not reproduce all of SentinelRAG's native statistical choices. No RAG-WM or recent private-fact pilot is mixed into this release's ranking.

## Contribution Controls

- **Construction:** `results/construction.json` preserves original, general-rewrite, and relation-explicit support-hit scores. Questions and fact contracts were held fixed.
- **Selection and scheduling:** the original implementation and focused tests are included. Static-versus-feedback summary evidence is in `results/component_controls.json`; its baseline is the frozen RAVET initial ranking, not SentinelRAG. The separate 128-condition factorial is not re-exported as a two-method primary comparison.
- **Utility and screening:** do not infer normal-answer utility or fact exclusivity from the demo's verification curves. They are different evaluation tasks; this compact release does not ship restricted answer text for independent oracle auditing.

## Outcome Boundaries

RAVET does not win in every configuration. The demo retains all paired rows, including ties and regressions. Combined compression and transfer is difficult because facts can lose support and entire carriers can disappear. A separate very-sparse transfer stress in the paper reverses the advantage; it is not part of the primary 0.5--2% density matrix. No result here establishes resistance to targeted sentinel removal, an invisible query footprint, or universally valid attribution for arbitrary target services.
