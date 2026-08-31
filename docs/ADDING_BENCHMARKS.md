# Adding a new science competition

Use the sealed 2026 path for every new benchmark.

## 1. Freeze official sources

Record the official problem, answer, detailed solution, marking scheme, public
release date, source URL, file SHA-256, language, theory/experimental scope, and
reported full score. Render relevant pages to PNG without redrawing or
model-generated replacement.

## 2. Build canonical turns

Split a problem only at official scoring/subpart boundaries. Preserve
`sequence_id` and `turn_index` so error-forward context is well-defined. Every
criterion must have a stable ID and the criterion maxima must sum to the item
maximum. The full competition aggregation must close to its reported scale.

Do not infer a rubric if an official one exists. Curator adjudications must be
explicit metadata, not silent edits.

## 3. Audit images and gold isolation

For every image, store the source PDF hash, source page, PNG hash, purpose, and
whether it contains question or solution material. Confirm that solver-side
question assets contain no marking layer or official answer.

## 4. Create the text-only representation

Run question-only staticization, review every visual dependency, merge explicit
corrections, and freeze the result. Text-only solvers receive this description;
multimodal solvers always receive the frozen original question PNG bytes
directly instead.

## 5. Add route and judge manifests

Probe exact routes rather than model families. Record API mode, token field,
thinking options, timeout, retry cap, input mode, capability proof path/hash,
release date, and display name. Fix the production judge before opening results.

## 6. Dry run and close the ledger

Run `validate`, a one-item visibility probe, a solver/judge smoke test, then the
full `solve`, `judge`, and `aggregate` stages. Require every planned item/route
cell to close; never treat a partial aggregate as a leaderboard score.

## 7. Report uncertainty separately

- Independent solver+judge repeats measure end-to-end API variability.
- Rejudging a fixed answer measures judge variability.
- Sequence-cluster bootstrap measures benchmark sampling uncertainty.
- Modality ablation measures representation sensitivity.

Do not combine these into one undifferentiated error bar.

## Checklist

- [ ] Official URLs, dates, PDFs, hashes, and scale recorded
- [ ] Item IDs/order and score closure validated
- [ ] Direct question/solution PNG provenance validated
- [ ] Solver/gold isolation passed
- [ ] Text-only staticization reviewed and frozen
- [ ] Exact route capability proofs passed
- [ ] Multimodal payload decodes to original PNG bytes
- [ ] Text-only payload contains zero image blocks/hidden URLs
- [ ] Judge criteria, bounds, evidence, and arithmetic validated
- [ ] All planned terminal cells present
- [ ] Raw public artifact contains no API key or hidden reasoning
