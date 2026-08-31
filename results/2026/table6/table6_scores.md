# Fresh 14-model four-competition score table

Status: **complete_four_competition_table**.

Scores are shown in competition score units with explicit denominators: IPhO /30, IChO theory /60, NBPhO /72, and USAPhO /150. The macro is the unweighted mean of the four competition-normalized percentages and is left missing unless all four scores are complete.

| # | Model | Modality | IPhO /30 | IChO /60 | NBPhO /72 | USAPhO /150 | Macro | Missing |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | Gemini 3 Flash | vision | 28.00 | 50.35 | 46.80 | 150.00 | 85.56% | none |
| 2 | Doubao Seed 2.0 Pro | vision | 20.75 | 40.50 | 35.60 | 147.00 | 71.03% | none |
| 3 | Gemini 3.1 Pro | vision | 29.75 | 54.18 | 48.30 | 146.00 | 88.47% | none |
| 4 | GPT-5.4 | vision | 21.95 | 31.87 | 37.60 | 145.00 | 68.79% | none |
| 5 | GLM-5.1 | text_staticized | 29.75 | 48.54 | 48.60 | 150.00 | 86.89% | none |
| 6 | Qwen3.6-Plus | vision | 27.25 | 45.12 | 46.80 | 149.25 | 82.63% | none |
| 7 | MiniMax M2.7 | text_staticized | 27.00 | 27.51 | 35.70 | 139.50 | 69.61% | none |
| 8 | Claude Opus 4.7 | vision | 26.05 | 45.99 | 48.70 | 132.50 | 79.86% | none |
| 9 | MiMo v2.5 Pro | text_staticized | 22.80 | 29.83 | 26.30 | 139.50 | 63.81% | none |
| 10 | GPT-5.5 | vision | 29.75 | 55.12 | 59.20 | 150.00 | 93.31% | none |
| 11 | DeepSeek V4 Pro | text_staticized | 29.75 | 48.98 | 51.40 | 148.25 | 87.76% | none |
| 12 | DeepSeek V4 Flash | text_staticized | 27.75 | 49.46 | 51.50 | 150.00 | 86.62% | none |
| 13 | Gemini 3.5 Flash | vision | 29.75 | 54.91 | 49.30 | 148.50 | 89.54% | none |
| 14 | Qwen3.7-Max | text_staticized | 29.75 | 46.68 | 46.80 | 148.00 | 85.16% | none |

IChO's /60 score is the official problem-weighted theory scale; the underlying frozen item-rubric maxima sum to 437 before the nine problem weights are applied.
