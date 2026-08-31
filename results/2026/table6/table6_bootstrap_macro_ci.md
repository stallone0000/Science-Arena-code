# Sequence-cluster bootstrap for the normalized macro mean

Status: **complete_four_competition_target**; publication eligible: **true**.

The calculation uses 10,000 paired resamples, seed `20260829`, and whole question/problem sequences as clusters within each competition. A sampled cluster carries all of its item scores together. The interval is a two-sided 95% percentile interval (type-7 interpolation).

This estimates **item/question-sequence sampling uncertainty only**. It does not estimate model-generation variance or judge variance; every model-item answer and judgment is one frozen realization.

| # | Model | Modality | Competitions | Macro | Bootstrap mean | 95% CI |
|---:|---|---|---:|---:|---:|---:|
| 1 | Gemini 3 Flash | vision | 4 | 85.56% | 85.65% | [79.24%, 91.66%] |
| 2 | Doubao Seed 2.0 Pro | vision | 4 | 71.03% | 71.17% | [63.67%, 78.85%] |
| 3 | Gemini 3.1 Pro | vision | 4 | 88.47% | 88.54% | [83.80%, 93.18%] |
| 4 | GPT-5.4 | vision | 4 | 68.79% | 68.91% | [61.86%, 75.78%] |
| 5 | GLM-5.1 | text_staticized | 4 | 86.89% | 86.96% | [81.13%, 92.05%] |
| 6 | Qwen3.6-Plus | vision | 4 | 82.63% | 82.73% | [75.55%, 89.74%] |
| 7 | MiniMax M2.7 | text_staticized | 4 | 69.61% | 69.74% | [62.01%, 77.48%] |
| 8 | Claude Opus 4.7 | vision | 4 | 79.86% | 79.97% | [73.19%, 86.37%] |
| 9 | MiMo v2.5 Pro | text_staticized | 4 | 63.81% | 63.85% | [54.46%, 72.66%] |
| 10 | GPT-5.5 | vision | 4 | 93.31% | 93.37% | [90.95%, 95.88%] |
| 11 | DeepSeek V4 Pro | text_staticized | 4 | 87.76% | 87.87% | [82.86%, 92.43%] |
| 12 | DeepSeek V4 Flash | text_staticized | 4 | 86.62% | 86.71% | [80.97%, 91.99%] |
| 13 | Gemini 3.5 Flash | vision | 4 | 89.54% | 89.60% | [86.12%, 93.19%] |
| 14 | Qwen3.7-Max | text_staticized | 4 | 85.16% | 85.24% | [79.82%, 90.54%] |
