# Reproduction

## Three Levels

1. **Offline replay:** `python3 reproduce.py --verify`. Standard library only. No API requests or dataset downloads. Recomputes decisions and aggregates on included oracle labels.
2. **Implementation checks:** install `.[test]`, run `pytest`, then `python examples/feedback.py`. Exercises construction contracts, selection, scheduler behavior, retrieval helpers, and replay against the production evidence function.
3. **New experiments:** install `.[live]`, obtain licensed datasets and model access, create model presets, generate carriers, build indexes and profiles, then run the selected experiment drivers. This is not a one-command recreation of private historical caches.

No Docker is needed for levels 1 or 2. Dense retrieval may benefit from a GPU; CPU-only offline replay does not load weights. Python 3.10+ is supported by the package specification. `VALIDATION.md` records the environment actually tested for this release rather than claiming all versions were tested.

## Implementation Map

| Component | Source |
| --- | --- |
| Relation-explicit prompts, length/answer checks | `src/sentinelrag/testing/sentinel_optimization.py` |
| Candidate and shadow profile data types | `src/sentinelrag/testing/prioritization.py` |
| Four shadow representations | `src/sentinelrag/testing/matrix.py` |
| One-probe selection and Bayesian group ordering | `src/sentinelrag/testing/adaptive.py` |
| Prefix-only next-test interface | `src/sentinelrag/testing/orthogonal_policy.py` |
| Common sequential evidence function | `src/sentinelrag/utils/stats.py` |
| Frozen-label, text-free evidence replay | `reproduce.py` |
| Original baseline generator, injector, detector | `src/sentinelrag/core/` |

Some internal symbols retain historical names: `relation_explicit` denotes RAVET's carrier construction, `failure_repair` denotes its offline probe selector, and `bayes` denotes its feedback scheduler. Those names do not imply online question rewriting. The same `sentinelrag` namespace also contains the upstream baseline and retained development utilities.

## Using Your Own Models

Historical experiment scripts read presets under `models/`; that directory is deliberately excluded from version control and from this release. Start with `models.example/generator.json`, set the URL/model to a provider you can access, and supply the key only through the `RAVET_API_KEY` environment variable. The placeholder endpoint cannot make a real request.

```bash
python -m pip install -e '.[live]'
mkdir -p models
cp models.example/generator.json models/generator.json
# Configure models/generator.json locally. Do not commit credentials.
export RAVET_API_KEY='YOUR_OWN_KEY'
python scripts/generate_sentinel_candidates.py --help
python scripts/generate_sentinel_candidates.py \
  --injection-result path/to/your/injection_result.json \
  --output-dir output/your-carriers \
  --model generator --strategies relation_explicit \
  --max-slots 2 --batch-size 1 --bounded-requests
```

The input follows SentinelRAG's injection schema (`watermark_texts`, `watermark_questions`, and fact contracts). The example bounds generation to two slots; it does not promise fact exclusivity or recreate paper fixtures. Use the original baseline generation modules to generate the starting facts, and audit them for your intended use.

Experiment entry points include `run_density_curve.py`, `run_summary_density.py`, `run_sentinel_paraphrase.py`, `run_corpus_expansion.py`, `run_retrieval_deployment_changes.py`, and `run_retrieval_deployment_completion.py`. Their transitive script imports are included. Historical protocols still refer to frozen preparation outputs and caches that are **not redistributed**. These scripts document and implement the experiment workflow but need regenerated preparation inputs and local configuration before a fresh run. Do not expect missing licensed datasets, indices, fixtures, or model presets to appear automatically.

Acquire datasets through their original [BEIR distributions](https://github.com/beir-cellar/beir). Do not upload entire corpus caches as part of an anonymous artifact. Re-running with a different endpoint or model revision is a new experiment and should retain new execution provenance.

## Offline Outputs

`reproduced/comparison.csv` reports each study/state/method separately at 6 and 15 queries, and 30 where the archive supports it. `reproduced/conditions.json` preserves condition-level curves for arbitrary budgets within each observation window. The demo consumes the same exported conditions through `demo/data.js`, so it works under `file://` without a web server.

## Integrity and Anonymity

`MANIFEST.json` hashes all packaged files except itself. `results/source_hashes.json` identifies the source archives used to create the text-minimized traces. `ANONYMITY_AUDIT.json` records automated checks for common credential patterns, internal endpoints, local user paths, and identifying project strings. This is a curation and integrity check, not a guarantee against all possible re-identification.

Keep `.git`, `models`, `.env`, request logs, browser captures, manuscript edit history, review prompts, and raw corpus/answer caches out of the public repository. Upstream copyright attribution is intentionally preserved and is not the RAVET authorship statement.
