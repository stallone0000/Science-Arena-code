# Archived expert-feedback base: export and validation

Fixed audit date: 2026-08-29  
Status: **PASS**  
Execution: fully local; 0 API calls.

## Result

The deterministic export contains **525** real expert-scored rows. Exactly **410** rows preserve a nonempty archived expert comment; **115** rows explicitly store `human_comment: null` because the archived source has no comment.

| Exam | Rows | Nonempty comments | Missing comments |
|---|---:|---:|---:|
| icho2025 | 325 | 325 | 0 |
| ipho2025 | 200 | 85 | 115 |

All 525 records have `annotation_status: not_provided_in_archived_source`. The fields for credit status, normalized error/ability tags, evidence, uncertainty, discrepancy, and conflict are `null`; an empty tag list is deliberately not used because that could be misread as a completed human review finding no tags.

## Provenance and validation

- JSONL SHA-256: `1fc337441cd882beff8b925ec34044906d32170dd054ebbf680ae59d9be20f56`
- Ground-truth files: 2; both match the frozen inventory hashes.
- Referenced original grading files: 11; all exist and match frozen hashes.
- Every row carries the canonical ground-truth path/hash/JSON pointer plus separate score-source and, when available, comment-source path/hash/locator fields.
- The validator enforces exact keys, types, score ranges, comment/null consistency, provenance hashes, unique identifiers, the complete 11-file source set, and the 525/410/115 census.

## Reproduction

```bash
python3 audits/cr_rebuttal_local_supplements_20260829/export_validate_expert_feedback_base_20260829.py --check audits/cr_rebuttal_local_supplements_20260829/expert_feedback_base_20260829.jsonl
```

The exporter emits deterministic artifacts to stdout; it does not write files or call a network API.

## Limitations

- The archive covers five model answers on IPhO 2025 and five on IChO 2025 only; it is not broader human calibration.
- The 115 unavailable comments are absent in the archived ground truth and are not reconstructed.
- No normalized error/ability tags, credit-status labels, evidence spans, uncertainty labels, discrepancy labels, or conflict adjudications are present in the archived source; all such fields remain null pending human QA.
- Original binary DOCX/PDF provenance is hash-addressed, but this release validates row values against the frozen canonical ground-truth JSON rather than re-OCRing or re-parsing binary source layout.
- The frozen IChO expert ground truth has raw row scores and problem-level maxima but no reliable per-row maximum; therefore IChO `max_score` is null instead of being borrowed from a judge run.
