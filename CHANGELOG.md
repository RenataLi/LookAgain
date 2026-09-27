# Changelog

## 0.1.0 — 2026-09-28

The first release includes controlled visual re-observation experiments and a
learned region-ranking pipeline for a frozen VLM.

- Nine-window region selection from pooled multimodal features, a fixed projection,
  query interactions and a compact ridge ranker.
- Five-fold out-of-fold analysis on 124 reused development sources, with image,
  position, prompted-selection and full-page resolution comparisons.
- Paired controls for native visual detail, answer extraction and conversation context
  across eight completed experiments.
- Published numerical score/cost observations, ranker coefficients, configurations
  and quality/compute measurements.
- One-command standard-library replay of 4,806 numerical comparisons across 22 strategies.
- CPU tests, documentation, architecture graphics and a synthetic animated walkthrough.

The numerical release preserves paired observations and fixed fold assignments.
`release_manifest.json` records the distributed file identities. Public tests use
numerical cohort projections and synthetic execution fixtures.
