# Third-Party Notices

## SentinelRAG

The baseline and original repository scaffolding derive from [SentinelRAG](https://github.com/ansonk4/sentinelrag), associated with [SentinelRAG: Synthetic Sentinel Knowledge for RAG Database Copyright Protection](https://arxiv.org/abs/2606.05787).

The upstream MIT license and copyright notice are retained in `LICENSE`. RAVET adds testing, construction, scheduling, replay, and evaluation components; some original utilities are modified. Keeping the upstream `sentinelrag` namespace is a compatibility decision, not a claim of upstream authorship or endorsement.

## Contriever

`src/sentinelrag/vendor/contriever_src/` contains upstream Contriever code with Facebook copyright headers and local import integration. Its original [license](https://github.com/facebookresearch/contriever/blob/main/LICENSE) is **Creative Commons Attribution-NonCommercial 4.0 International**, not the root MIT license. A copy is in `third_party/CONTRIEVER_LICENSE`. Source: [facebookresearch/contriever](https://github.com/facebookresearch/contriever). The artifact's MIT notice does not override those terms.

## Datasets and Models

BEIR is a collection of datasets with separate original licenses. No corpus texts, corpus transformations, retrieved passages, generated target answers, or model weights are distributed in this compact release. Obtain them from their original sources and follow their terms. Numeric observations and example synthetic inputs must not be mistaken for the original corpus distribution.

## Browser Demo

The demo is plain HTML, CSS, JavaScript, and local images. It has no CDN scripts, analytics, remote fonts, external model requests, or authentication. The charts visualize exported measurements, not live generated outcomes.
