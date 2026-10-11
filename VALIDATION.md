# Validation

## Checked for This Export

- Offline replay recomputes 1,792 frozen trajectories and 896 method-configuration rows. Every archived crossing curve agrees.
- The primary study recovers 144 paired protected conditions, 47.2% versus 69.4% VSR(6), 74.8% versus 83.3% VSR(15), and 8.31 versus 6.66 MQ(15).
- 67 focused tests pass, including comparison of the standard-library replay with the production statistical function over every success count for 1--30 observations and all four archived background bounds.
- The synthetic feedback example exercises the actual selection/scheduler modules, with different next queries after positive and negative feedback.
- Headless Chrome checks desktop and mobile widths of 1440, 900, 390, and 320 pixels, nonblank charts, loaded overview artwork, filters, query-budget controls, clean controls, view switching, and CSV export. No page errors or HTTP(S) requests occur.
- Export checks scan text and PNG metadata for common credential patterns, user paths, internal endpoints, and identifying project strings. No matching findings remain. Third-party copyright attribution is retained.

## Environment and Scope

Offline replay was tested with Python 3.13. Core implementation tests use Python 3.13, NumPy 1.26.4, SciPy 1.17.1, scikit-learn 1.9.0, rank-bm25 0.2.2, and pytest 9.1.1 in the existing research environment, explicitly importing the exported source tree. Other Python versions are not claimed as tested merely because the package allows them.

A separate fresh-environment installation was attempted with Python 3.12 and 3.13; the host's package index could not resolve scikit-learn. Clean network installation is therefore not independently confirmed here. The dependency-free offline replay and browser demo are unaffected. No fresh paid-API experiment or independent rejudging of original answers was performed during packaging.

The archive deliberately excludes corpus texts, answer texts, request caches, local model presets, identifying history, and later externally reported screening records. See `docs/EVIDENCE.md` for the original decision bounds and study boundaries.
