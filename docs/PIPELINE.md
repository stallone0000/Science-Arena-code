# End-to-end pipeline

ScienceArena treats evaluation as an auditable data pipeline, not a collection
of prompt snippets.

```text
official PDFs + rubrics
        |
        v
canonical item JSONL + frozen page PNGs
        |
        +--> native-vision route: exact question PNG bytes
        |
        +--> text-only route: answer-blind reviewed staticization, no images
        |
        v
capability gate -> interleaved solver -> rubric judge -> aggregation
                                           |
                                           v
                              item evidence + score ledger
```

## 1. Canonical benchmark construction

Each scored turn is one JSON object. A minimal open-ended item contains:

```json
{
  "schema_version": "science_arena_item_v1",
  "item_id": "contest2027_p1_a",
  "benchmark_id": "contest2027",
  "sequence_id": "contest2027_p1",
  "turn_index": 1,
  "max_score": 2.0,
  "question_text_native": "...",
  "question_text_staticized": "...",
  "question_image_paths": ["question_images/page-01.png"],
  "official_answer": "...",
  "official_solution": "...",
  "official_rubric": "...",
  "official_rubric_criteria": [
    {"criterion_id": "P1-A-C1", "max_score": 2.0, "text": "..."}
  ],
  "official_solution_image_paths": ["solution_images/page-03.png"]
}
```

Builders and fail-closed validators for IPhO, IChO, NBPhO, and USAPhO 2026 are
in `science_arena/`. Validators check item order, unique IDs, source hashes,
score closure, image magic bytes, official page provenance, rubric criterion
maxima, and solver/gold isolation.

## 2. Question-only staticization

Staticization exists only for routes that cannot receive images. The generator
is given question text plus official question-page images. It is not given the
official answer, solution, rubric, maximum-score rationale, or solution images.

The process is:

1. `staticize_question_only.py` creates review candidates and lossless audit
   sidecars.
2. A human reviews mathematical layout, graphs, molecule structures,
   stereochemistry, labels, units, and cross-turn dependencies.
3. `merge_staticization_reviews.py` applies explicit review decisions.
4. `freeze_staticizations.py` checks leakage boundaries and writes immutable
   item JSONL plus provenance hashes.

The final 2026 freeze covers 153 turns. Its four item files and provenance are
under `metadata/staticizations/`; original page PNGs are in the `inputs-2026`
release asset.

## 3. Exact-route capability gate

Model family names are not used as a proxy for modality. Every exact API route
must pass a frozen capability probe. A vision route must recover an unseen
nonce/shape from a synthetic pixel image and pass direct official science-image
checks. A text route must pass the text nonce probe. Proofs are stored in
`metadata/capabilities/` and bound by SHA-256 in `configs/models_table6.json`.

A route classified as vision receives byte-preserving `data:image/...;base64`
encoding of frozen PNG files. The decoded payload must hash to the official
file. A text-only route receives no image block, hidden image URL, or generated
caption field beyond the reviewed staticized question text.

## 4. Interleaved solving

Items are grouped by `sequence_id` and ordered by `turn_index`. At turn `t`, the
solver sees the current subproblem and the preceding question/answer turns from
the same problem. It never sees gold material.

The sealed runner stores a terminal item record only after a concrete visible
answer is validated. It records one initial request and at most six retries in
a contiguous ledger. If all seven calls fail to yield a usable answer, the
explicit `不会做这道题` sentinel remains in context and the sequence continues.
The older archived runner used its own five-retry policy; see `LEGACY.md`.

Provider hidden reasoning is not persisted. The storage layer removes common
reasoning/thinking subtrees while retaining visible output, token counters,
finish reason, response identity, and request/image proofs.

## 5. Rubric judging

The primary judge is fixed by manifest rather than selected after seeing model
scores. For each item, the judge receives:

- the candidate's visible response;
- the official answer and detailed solution;
- exact point-bearing rubric criterion IDs, text, and maxima;
- the item's total maximum;
- original official question and solution/marking PNGs when present.

The judge returns one JSON object with bounded criterion sub-scores, a total,
candidate evidence quotations, deductions, and confidence. The validator
requires criterion identity and cardinality, per-criterion bounds, arithmetic
closure, exact candidate evidence, and the correct item maximum. Invalid output
is rejected and retried; scores are not silently clamped.

An exact solver-no-answer sentinel is deterministically assigned zero and does
not consume a judge call. IBO is objective-keyed and bypasses the LLM judge.

Because official chemistry and physics gold material can be graphical, a
production judge should be vision-capable. A text-only judge is a distinct
ablation condition and requires separately audited question- and gold-side
staticization; it is not silently substituted into the main judge route.

## 6. Aggregation

Aggregation is allowed only after every selected route/item has a valid terminal
solver record and either a valid judgment or deterministic-zero record.

- IPhO/NBPhO/USAPhO: sum item scores on the reported exam scale.
- IChO: normalize each problem's raw rubric points to its official theory
  weight, then sum the nine problem weights to 60.
- IBO: sum deterministic keyed item points to 453.
- Cross-contest macro: normalize each contest to percent and take an unweighted
  arithmetic mean. Missing contests are never imputed.

`analysis/build_table6_analysis.py` independently reconstructs the four 2026
columns and a 10,000-resample sequence-cluster bootstrap. The bootstrap moves
whole problems, not individual dependent subparts.

## 7. Resumption and provenance

Solver and judge stages are separately resumable. Terminal records, attempt
sidecars, manifests, source hashes, payload fingerprints, route manifests, and
image proofs are checked before reuse. A run cannot combine records from a
different item file, route manifest, prompt contract, or previous history.

Use a distinct run root for every independent replicate. Reusing one root is a
resume operation, not a new sample.
