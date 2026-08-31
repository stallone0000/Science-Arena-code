# Six-judge frozen-answer composite replay

Status: complete (1,638 effective terminal judgments; source and recovery ledgers passed fail-closed validation).

This covers all 91 items for one frozen Gemini-3.5 Flash solver answer corpus. It is not a 14-model ranking experiment and cannot establish cross-candidate robustness. Agreement measures reliability, not validity.

Four vision judges receive the original official question and solution/marking PNGs directly. Qwen and DeepSeek receive zero image blocks and are reported only as modality sensitivity.

## Repeat reliability

- google/gemini-3.5-flash: normalized repeat MAE 0.0158; mean case SD 0.0103; exact 3-way 94.5%.
- google/google/gemini-3.1-pro-preview: normalized repeat MAE 0.0087; mean case SD 0.0061; exact 3-way 96.7%.
- anthropic/claude-opus-4.7: normalized repeat MAE 0.0232; mean case SD 0.0163; exact 3-way 90.1%.
- openai/gpt-5.5: normalized repeat MAE 0.0073; mean case SD 0.0052; exact 3-way 96.7%.
- qwen/qwen3.7-max: normalized repeat MAE 0.0169; mean case SD 0.0111; exact 3-way 92.3%.
- deepseek/deepseek-v4-pro: normalized repeat MAE 0.0167; mean case SD 0.0116; exact 3-way 94.5%.

## Vision-judge agreement (91 item repeat means)

- google/gemini-3.5-flash vs google/google/gemini-3.1-pro-preview: MAE 0.0186; Pearson 0.8471; Spearman 0.8309; Kendall tau-b 0.8104.
- google/gemini-3.5-flash vs anthropic/claude-opus-4.7: MAE 0.0202; Pearson 0.8879; Spearman 0.7673; Kendall tau-b 0.7408.
- google/gemini-3.5-flash vs openai/gpt-5.5: MAE 0.0246; Pearson 0.7809; Spearman 0.7727; Kendall tau-b 0.7461.
- google/google/gemini-3.1-pro-preview vs anthropic/claude-opus-4.7: MAE 0.0168; Pearson 0.9160; Spearman 0.8406; Kendall tau-b 0.8173.
- google/google/gemini-3.1-pro-preview vs openai/gpt-5.5: MAE 0.0093; Pearson 0.9600; Spearman 0.9575; Kendall tau-b 0.9395.
- anthropic/claude-opus-4.7 vs openai/gpt-5.5: MAE 0.0144; Pearson 0.9335; Spearman 0.8898; Kendall tau-b 0.8603.

## Contest totals (mean ± sample SD across three repeats)

- google/gemini-3.5-flash — ipho2026: 29.7500 ± 0.0000 /30 (repeats 29.7500, 29.7500, 29.7500).
- google/gemini-3.5-flash — icho2026: 56.0108 ± 0.5158 /60 (repeats 56.1789, 55.4319, 56.4216).
- google/google/gemini-3.1-pro-preview — ipho2026: 28.7500 ± 1.0000 /30 (repeats 27.7500, 29.7500, 28.7500).
- google/google/gemini-3.1-pro-preview — icho2026: 56.4560 ± 0.0974 /60 (repeats 56.5122, 56.3435, 56.5122).
- anthropic/claude-opus-4.7 — ipho2026: 29.0833 ± 1.1547 /30 (repeats 27.7500, 29.7500, 29.7500).
- anthropic/claude-opus-4.7 — icho2026: 56.7377 ± 0.5080 /60 (repeats 56.7129, 57.2576, 56.2425).
- openai/gpt-5.5 — ipho2026: 28.2500 ± 0.8660 /30 (repeats 27.7500, 27.7500, 29.2500).
- openai/gpt-5.5 — icho2026: 57.0988 ± 0.1925 /60 (repeats 57.2099, 56.8765, 57.2099).
- qwen/qwen3.7-max — ipho2026: 27.7500 ± 1.0000 /30 (repeats 28.7500, 27.7500, 26.7500).
- qwen/qwen3.7-max — icho2026: 54.4418 ± 0.8147 /60 (repeats 54.5701, 53.5706, 55.1848).
- deepseek/deepseek-v4-pro — ipho2026: 28.0833 ± 0.5774 /30 (repeats 27.7500, 27.7500, 28.7500).
- deepseek/deepseek-v4-pro — icho2026: 54.7610 ± 1.5679 /60 (repeats 56.2099, 53.0964, 54.9768).

## Protocol recovery

One Gemini-3.1 cell required a same-configuration post-exhaustion recovery. DeepSeek is represented only by a complete uniform thinking-enabled 65k-token rerun; all old 16k DeepSeek scores are excluded and its spend is reported separately as failure-diagnostic cost.
