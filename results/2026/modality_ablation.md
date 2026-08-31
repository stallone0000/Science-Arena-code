# Modality Ablation Analysis

Status: **protocol-valid paired complete-case analysis**. Scores are macro means of normalized turn scores. Open-answer cells use the mean of three judges; IBO cells are recomputed deterministically.

## Completion

- Design: 12 models x 12 turns x 3 arms x 4 repetitions.
- Raw responses: 1727 successful and 1 exhausted provider failure out of 1728 planned cells.
- Paired analysis: 575/576 complete model-item-repetition triplets (1725 arm cells).
- Available judgments: 4031/4032; analyzed judgments: 4029.
- Analyzed objective judgments: 573; open judgments: 3456.

## Overall Arms

| Arm | Cells | Mean % | SD | Paired-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| native | 575 | 72.75 | 30.02 | [70.23, 75.23] |
| caption | 575 | 71.73 | 31.45 | [69.11, 74.25] |
| blind | 575 | 49.85 | 34.78 | [47.03, 52.76] |

## Paired Arm Deltas

Positive values favor the left arm.

| Contrast | Paired units | Delta (pp) | SD of paired difference | 95% CI |
|---|---:|---:|---:|---:|
| native_minus_caption | 575 | 1.02 | 27.29 | [-1.20, 3.27] |
| native_minus_blind | 575 | 22.90 | 35.09 | [20.02, 25.77] |
| caption_minus_blind | 575 | 21.88 | 34.98 | [19.00, 24.68] |

## By Model

Arm cells show mean percent and paired-bootstrap 95% CI.

| Model | Native | Caption | Blind | Native - caption (pp) | Native - blind (pp) |
|---|---:|---:|---:|---:|---:|
| alibaba/qwen3.7-plus | 74.58 [65.90, 82.67] | 62.49 [53.48, 70.99] | 42.57 [33.43, 51.64] | 12.09 [3.71, 21.57] | 32.01 [21.67, 42.67] |
| anthropic/claude-opus-4.7 | 73.33 [64.41, 81.83] | 76.09 [67.04, 84.41] | 54.57 [44.40, 64.53] | -2.76 [-9.06, 4.43] | 18.76 [8.56, 29.68] |
| bytedance/doubao-seed-2-0-pro | 68.29 [59.84, 76.39] | 71.10 [61.23, 80.38] | 50.21 [40.90, 59.37] | -2.82 [-12.36, 7.48] | 18.08 [9.30, 27.29] |
| cloudsway/gemini-3-flash-preview | 82.31 [74.62, 89.11] | 81.69 [73.81, 88.67] | 59.97 [49.79, 70.15] | 0.62 [-3.81, 5.63] | 22.34 [12.98, 32.14] |
| google/gemini-3.5-flash | 75.67 [66.51, 84.34] | 71.55 [62.37, 80.21] | 54.20 [44.11, 64.32] | 4.12 [-2.12, 10.66] | 21.47 [9.72, 32.56] |
| google/google/gemini-3.1-pro-preview | 80.39 [72.06, 87.63] | 83.86 [77.25, 89.81] | 60.59 [50.71, 69.72] | -3.47 [-10.55, 2.34] | 19.80 [12.63, 27.83] |
| minimax/MiniMax-M3 | 70.37 [62.18, 78.16] | 70.62 [62.72, 78.09] | 44.71 [35.20, 54.46] | -0.25 [-7.29, 6.29] | 25.66 [13.95, 37.52] |
| moonshotai/kimi-k2.6-internal | 65.68 [56.93, 74.01] | 62.13 [51.66, 72.39] | 39.49 [29.84, 49.20] | 3.55 [-5.32, 12.95] | 26.19 [17.24, 35.86] |
| openai/gpt-5.5 | 86.30 [77.95, 93.29] | 81.78 [72.37, 90.18] | 57.43 [47.13, 67.50] | 4.52 [-0.66, 10.74] | 28.87 [18.94, 38.83] |
| qiniu/openai/gpt-5.4 | 72.07 [64.38, 79.21] | 64.96 [57.07, 72.49] | 51.88 [42.30, 61.41] | 7.11 [0.51, 14.15] | 20.18 [12.03, 28.57] |
| qwen/qwen3.6-plus | 65.39 [56.40, 73.93] | 73.22 [63.80, 82.10] | 45.17 [36.50, 53.79] | -7.83 [-16.27, 0.52] | 20.22 [11.12, 30.07] |
| stepfun/step-3.7-flash | 58.61 [49.82, 67.30] | 61.26 [51.90, 70.60] | 37.32 [29.06, 45.92] | -2.65 [-11.27, 6.21] | 21.29 [11.33, 30.69] |

## By Domain

| Domain | Native | Caption | Blind | Native - caption (pp) | Native - blind (pp) |
|---|---:|---:|---:|---:|---:|
| physics | 78.82 [75.35, 82.09] | 81.57 [78.14, 84.75] | 55.47 [51.14, 59.75] | -2.75 [-6.62, 1.15] | 23.35 [18.32, 28.05] |
| chemistry | 53.72 [48.79, 58.67] | 48.70 [43.95, 53.56] | 17.73 [14.87, 20.82] | 5.01 [0.57, 9.56] | 35.99 [30.32, 41.73] |
| biology | 85.79 [83.02, 88.37] | 84.99 [81.82, 87.98] | 76.50 [73.30, 79.64] | 0.80 [-2.13, 3.74] | 9.29 [6.02, 12.57] |

## Payload And Read-Check Audit

- Native solver: 861/861 audited image blocks and 575/575 valid read checks.
- Caption solver: 575/575 zero-image, 575/575 hidden-URL-free.
- Blind solver: 575/575 zero-image, 575/575 hidden-URL-free.
- Open judges: 7344/7344 image blocks, 3456/3456 read checks, and 3456/3456 complete observation audits.

## Judge Disagreement

Across 1152 open-answer panels, mean judge range was 20.40 pp (p95 70.00 pp; max 100.00 pp).

| Judge pair | Mean signed difference (pp) | Mean absolute difference (pp) | RMSE (pp) | Correlation |
|---|---:|---:|---:|---:|
| google/gemini-3.5-flash vs openai/gpt-5.5 | -1.53 | 13.43 | 25.43 | 0.792 |
| google/gemini-3.5-flash vs anthropic/claude-opus-4.7 | -0.40 | 15.22 | 25.05 | 0.783 |
| openai/gpt-5.5 vs anthropic/claude-opus-4.7 | 1.13 | 12.15 | 20.49 | 0.852 |

Bootstrap: 10000 deterministic paired resamples, seed 20260710.
Input fingerprint: `063c2d3fa47e71cf3a969cd595d647e6361d524a8ec1d683aa5922f9951c1c94`.

