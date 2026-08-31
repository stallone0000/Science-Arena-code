# Camera-ready reproducibility release

This release accompanies the final ScienceArena camera-ready paper. It provides
the complete runnable evaluation pipeline, compact paper-facing results, frozen
2026 inputs, and archived visible-response/rubric-score evidence.

Highlights:

- original official image bytes are delivered directly to multimodal solvers;
- text-only routes receive answer-blind question staticizations and no images;
- rubric judging, repeated solver calls, cross-family judge stability, and
  bootstrap aggregation are reproducible from documented commands;
- the final PDF and complete LaTeX source are versioned in the repository;
- large assets are checksum-addressed in `release_assets.json`.

All model archives intentionally omit hidden chain-of-thought. Every release
asset was scanned for credential-shaped tokens and private-key material before
upload. API credentials must be supplied only through environment variables.
