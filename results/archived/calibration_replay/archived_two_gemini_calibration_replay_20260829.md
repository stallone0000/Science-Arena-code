# Archived two-Gemini-judge calibration replay

Fixed audit date: 2026-08-29  
Status: **PASS**  
Execution: fully local; 0 API calls; 0 key reads.

## Row accounting

The replay finds exactly **1,050 archived judgment rows** over **525 distinct expert rows**. All 1,050 rows have a unique expected key, a finite score, a finite archived row maximum, and no stored error. Raw expert/judge metrics therefore use 1,050/1,050 pairs. Normalized expert/judge metrics use 1,048/1,050 pairs because one underlying IChO source-scale mismatch appears once for each judge and is not repaired.

| Judge | Exam | Stored / expected | Raw pairs | Normalized pairs | Missing | Errors |
|---|---|---:|---:|---:|---:|---:|
| gemini_3_1_pro_preview | ipho2025 | 200 / 200 | 200 | 200 | 0 | 0 |
| gemini_3_1_pro_preview | icho2025 | 325 / 325 | 325 | 324 | 0 | 0 |
| gemini_3_5_flash | ipho2025 | 200 / 200 | 200 | 200 | 0 | 0 |
| gemini_3_5_flash | icho2025 | 325 / 325 | 325 | 324 | 0 | 0 |

## Agreement with experts

Item-level raw-point metrics retain every archived score. Fraction-of-row-maximum metrics exclude only the explicitly listed invalid source-scale pair(s). Candidate-total metrics use five saved answer models per judge/exam.

| Judge | Exam | Item n | Raw MAE | Raw RMSE | Exact | Normalized n | Normalized MAE | Pearson (norm.) | Lin CCC (norm.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini_3_1_pro_preview | ipho2025 | 200 | 0.065500 | 0.139821 | 139 (0.695) | 200 | 0.091546 | 0.832826 | 0.831373 |
| gemini_3_1_pro_preview | icho2025 | 325 | 0.332554 | 0.917620 | 246 (0.757) | 324 | 0.075211 | 0.909267 | 0.908972 |
| gemini_3_5_flash | ipho2025 | 200 | 0.068550 | 0.136936 | 133 (0.665) | 200 | 0.097231 | 0.828759 | 0.827884 |
| gemini_3_5_flash | icho2025 | 325 | 0.565938 | 1.075477 | 148 (0.455) | 324 | 0.138375 | 0.867625 | 0.855432 |

| Judge | Exam | Total n | Total MAE | Mean signed error | Maximum absolute error |
|---|---|---:|---:|---:|---:|
| gemini_3_1_pro_preview | ipho2025 | 5 | 0.740000 | -0.220000 | 1.000000 |
| gemini_3_1_pro_preview | icho2025 | 5 | 0.564809 | 0.349169 | 0.812417 |
| gemini_3_5_flash | ipho2025 | 5 | 0.602000 | -0.230000 | 0.930000 |
| gemini_3_5_flash | icho2025 | 5 | 0.297015 | -0.297015 | 0.638667 |

## Interjudge agreement

Here Gemini 3.1 Pro Preview is the reference and Gemini 3.5 Flash is the comparator; this orientation affects only the sign of signed differences.

| Exam | Shared item n | Raw MAE | Exact | Normalized MAE | Pearson (norm.) | Shared total n | Total MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ipho2025 | 200 | 0.050750 | 157 (0.785) | 0.067344 | 0.862130 | 5 | 0.450000 |
| icho2025 | 325 | 0.449385 | 182 (0.560) | 0.108428 | 0.925918 | 5 | 0.766862 |
| combined | 525 | 0.297524 | 339 (0.646) | 0.092777 | 0.907134 | 10 | 0.608431 |

## Explicit source-scale exclusion

The frozen expert row `gpt-5 / ICHO-2-2.3c` has raw expert score 10.0, while each archived judge row declares `max_score=6.0`. The two raw score comparisons remain in the 1,050-row metrics; both are excluded from normalized expert/judge metrics. The replay does not guess which field should change.

## Reproduction

```bash
python3 audits/cr_rebuttal_local_supplements_20260829/replay_archived_two_gemini_calibration_20260829.py --check audits/cr_rebuttal_local_supplements_20260829/archived_two_gemini_calibration_rows_20260829.csv
```

Row CSV SHA-256: `36aa01ae8dcd85b1a8680b2f7034f833907a67b7e377a2e22d8cf27efd257d98`. The JSON report contains complete metric definitions, denominators, per-bin calibration summaries, candidate totals, source hashes, missingness, and exclusion records.

## Limitations

- Both archived judges are Gemini-family models; this is not a cross-family judge robustness study.
- The calibration covers only five saved candidate answers on IPhO 2025 and five on IChO 2025; candidate-total denominators are five per judge/exam.
- The replay evaluates archived final merged/reparsed result arrays. It does not resend requests and cannot measure current endpoint behavior or run-to-run stochasticity.
- IChO row ICHO-2-2.3c for gpt-5 has expert_score=10.0 but archived judge-row max_score=6.0. Both judge copies are retained in raw-point metrics and explicitly excluded from normalized expert/judge metrics; no score or maximum is repaired.
- Candidate-total agreement can look strong even when individual rubric rows disagree; item-level and total-level denominators and metrics are therefore reported separately.
- Correlation and regression on five candidate totals per judge/exam are descriptive and statistically fragile; no confidence interval or significance claim is made.
