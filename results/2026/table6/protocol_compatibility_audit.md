# Cross-generation protocol compatibility audit

Overall status: **PASS_all_four_complete_and_independently_sealed**.

The generations are compatible on the requested scientific comparison invariants, but they are not byte-identical protocols. Pipeline and validation contracts evolved; the audit therefore proves the stable modality, payload, roster, score, judge-route, and frozen-rubric properties directly from stored requests and terminals.

| Competition | Generation | Aggregate | Vision payload | Text payload | Judge/rubric | Independent seal |
|---|---|---|---|---|---|---|
| IPhO 2026 | formal_qonly_v1_with_sealed_vision_reuse | complete; b00415adaeff… | PASS (8 routes) | PASS (6 routes) | PASS; `google/gemini-3.5-flash`; rubric set d0e7a4cf8c11… | PASS |
| IChO 2026 | formal_qonly_v2_with_sealed_vision_reuse | complete; 4aa5ffbb4706… | PASS (8 routes) | PASS (6 routes) | PASS; `google/gemini-3.5-flash`; rubric set 1a4275c30f7d… | PASS |
| NBPhO 2026 | materialized_full_fresh_judge_v8 | complete; f0563636088c… | PASS (8 routes) | PASS (6 routes) | PASS; `google/gemini-3.5-flash`; rubric set 3cdd16b5371e… | PASS |
| USAPhO 2026 | sealed_full_fresh_judge_v5 | complete; 354f87646071… | PASS (8 routes) | PASS (6 routes) | PASS; `google/gemini-3.5-flash`; rubric set 3cce349fd5bb… | PASS |

Evidence scope:

- Every stored solver attempt for every vision route was checked for direct PNG data-image blocks whose byte count and SHA-256 bind to the frozen official question-page files and source-PDF metadata.
- Every stored solver attempt for every text route was checked for zero image blocks; each terminal prompt hash was recomputed from the frozen question-only staticized field and its freeze record.
- Every judge terminal and wire request was checked for the fixed `google/gemini-3.5-flash` route. Every non-deterministic request contains the exact normalized frozen item rubric and exact criterion-ID/max-score contract.
- Deterministic model-no-answer rows use zero judge attempts and score zero; they do not introduce a different rubric judgment.

Generation differences retained in the JSON audit:

- IPhO/IChO are run-schema v1 roots with independently audited vision-tree reuse and fresh text-staticized routes.
- USAPhO is the sealed full-fresh judge-v5 generation; the earlier two-missing-row aggregate is excluded.
- NBPhO is the materialized full-fresh judge-v8 generation with stronger judge evidence/fingerprint validation.
- Pipeline, run-schema, fingerprint, retry, and evidence-validation contracts therefore differ by generation; compatibility is asserted only for the explicitly audited scientific input/scoring invariants, not byte identity.
