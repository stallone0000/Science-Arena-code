# Archived protocol notes

The paper spans two code generations.

`legacy/` preserves the text-interleaved 2025 runners, objective-key IBO runner,
candidate-benchmark ingestion, older judge calibration, and dataset-export
utilities. Those scripts are retained to explain and rerun the first nine
columns of the 13-contest table. The corresponding large data package is the
`archived-2025-reproduction` release asset.

The archived open-ended protocol sent staticized text to all routes and retried
a non-concrete response up to five times, retaining the first usable answer.
The IBO path used deterministic objective-key scoring. Some early scripts still
accept a deprecated `--api_key` argument for historical compatibility; do not
use it. Set `QIHOO_API_KEY` in the environment.

The current `science_arena/` runner supersedes the legacy path for new data. It
adds exact-route modality probes, direct-PNG guarantees for multimodal solvers,
answer-blind staticization for text-only solvers, strict rubric criteria,
evidence quotations, fail-closed validation, corpus seals, and no-CoT storage.

Some original provider wire directories for the archived nine columns were not
retained. Their runnable protocol and final aggregate artifacts are available,
but those cells are correctly described as aggregate replay rather than
byte-identical raw replay.
