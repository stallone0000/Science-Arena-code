# Full Science Arena Benchmark Table, 2026-05-24

`*` means caveated; `invalid` is not a model-capability score; `pending` means no usable artifact yet.

## Benchmark Set

| Benchmark | Max | Status |
|---|---:|---|
| IPhO 2025 | 30 | paper_original; interleaved; HF rows fixed to /30 |
| IChO 2025 | 60 | paper_original; interleaved; image-only answer refs preserved |
| CChO 2025 | 100 | paper_original; CChO Q7 missing question-side structure image |
| CPhO 2025 | 320 | paper_original; interleaved |
| IBO 2023 | 453 | paper_original; objective key scored, not LLM-judge |
| EuPhO 2025 Theory | 30 | new official PDF; image-enabled static benchmark |
| APhO 2025 Theory | 30 | new official PDF; draft/human-review page-range + item-level data |
| INChO 2026 | 106 | new official PDF; official rubric page-range candidate |
| USAPhO 2025 | 150 | new official PDF; derived-rubric auxiliary benchmark |
| NBPhO 2025 | 72 | new official PDF; item-level draft, experimental/staticization caveats |

## Model Matrix

| Model | IPhO 2025 /30 | IChO 2025 /60 | CChO 2025 /100 | CPhO 2025 /320 | IBO 2023 /453 | EuPhO 2025 Theory /30 | APhO 2025 Theory /30 | INChO 2026 /106 | USAPhO 2025 /150 | NBPhO 2025 /72 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `google/gemini-3.5-flash` | 25.7 | 48.4 | 57.1 | 275 | 430 | 17.2 | 28.1 | 91.6 | 150 | 35.5 |
| `google/google/gemini-3.1-pro-preview` | 27.3 | 46.5 | 57.4 | 302 | 431 | 14.5 | 29.8 | 93.5 | 150 | 6 |
| `cloudsway/gemini-3-flash-preview` | 24.3 | 44.4 | 52.5 | 270 | 428 | 19.3 | 20.7 | 83.3 | 150 | 22.3 |
| `qiniu/openai/gpt-5.4` | 19.4 | 30.8 | 41.2 | 212 | 358 | 13 | 24.8 | 92.4 | 130 | 32.9 |
| `openai/gpt-5.5` | 25.6 | 34 | 46.6 | 290 | 415 | invalid | invalid | 74.9* | 150 | invalid |
| `anthropic/claude-opus-4.7` | 22.7 | 41.7 | 49.6 | 273 | 407 | 8.6 | 29 | 51.1 | 145 | 46.9 |
| `bytedance/doubao-seed-2-0-pro` | 18 | 38.3 | 45.2 | 228 | 415 | 7 | 16.1 | 71.7 | 63 | 11.7 |
| `qwen/qwen3.7-max` | 23.9* | 28.9* | 45.4* | 287* | 349* | 5.9* | 28* | 77.6* | 142* | 29.5* |
| `qwen/qwen3.6-plus` | 22.4 | 41.8 | 46.9 | 271 | 413 | 7 | 24.8 | 87.4 | 54 | 5.1 |
| `deepseek/deepseek-v4-pro` | 21.6* | 33.5* | 46.6* | 251* | 327* | invalid | invalid | invalid | invalid | invalid |
| `deepseek/deepseek-v4-flash` | 23.2 | 29.6 | 39.5 | 224 | 324 | 9.2 | 25.8 | 84 | 138 | 12.1* |
| `minimax/MiniMax-M2.7` | 17.7* | 19.9* | 36.4* | 211* | 337* | 1.2* | invalid | invalid | invalid | invalid |
| `z-ai/glm-5.1` | 22.3 | 31.2 | 43.8 | 262 | 349 | invalid | 26.9 | invalid | 144 | 30.6 |
| `xiaomi/mimo-v2.5-pro` | 22.4* | 29.1* | 35.2* | 219* | 317* | 1* | invalid | 74.1* | invalid | invalid |

## Invalid Or Caveated Candidate Rows

| Benchmark | Model | Status | Note | Source |
|---|---|---|---|---|
| eupho2025 | `openai/gpt-5.5` | invalid | content_filter; not a capability score | artifacts/runs/candidate_benchmarks/new_benchmark_results_20260524_collected.json |
| apho2025 | `openai/gpt-5.5` | invalid | page-render-only retry returned HTTP 500 | artifacts/runs/candidate_benchmarks/new_benchmark_results_20260524_collected.json |
| incho2026 | `openai/gpt-5.5` | caveated | problem-level recovery; P2 content-filter/truncated |  |
| nbpho2025 | `openai/gpt-5.5` | invalid | content_filter | artifacts/runs/candidate_benchmarks/nbpho2025_full_allmodels_20260524_v1_mm_openai__gpt-5.5_mm/summary.json |
| ipho2025 | `qwen/qwen3.7-max` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| icho2025 | `qwen/qwen3.7-max` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| ccho2025 | `qwen/qwen3.7-max` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| cpho2025 | `qwen/qwen3.7-max` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| ibo2023 | `qwen/qwen3.7-max` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| eupho2025 | `qwen/qwen3.7-max` | caveated | text-only fallback; text-only | artifacts/runs/candidate_benchmarks/new_benchmark_results_20260524_collected.json |
| apho2025 | `qwen/qwen3.7-max` | caveated | text-only fallback; text-only | artifacts/runs/candidate_benchmarks/new_benchmark_results_20260524_collected.json |
| incho2026 | `qwen/qwen3.7-max` | caveated | text-only | artifacts/runs/candidate_benchmarks/incho2026_full_allmodels_20260524_v2_textonly_summary.json |
| usapho2025 | `qwen/qwen3.7-max` | caveated | text-only | artifacts/runs/candidate_benchmarks/usapho2025_full_allmodels_20260524_v2_textonly_summary.json |
| nbpho2025 | `qwen/qwen3.7-max` | caveated | text-only | artifacts/runs/candidate_benchmarks/nbpho2025_full_allmodels_20260524_v2_textonly_summary.json |
| ipho2025 | `deepseek/deepseek-v4-pro` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| icho2025 | `deepseek/deepseek-v4-pro` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| ccho2025 | `deepseek/deepseek-v4-pro` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| cpho2025 | `deepseek/deepseek-v4-pro` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| ibo2023 | `deepseek/deepseek-v4-pro` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| eupho2025 | `deepseek/deepseek-v4-pro` | invalid | text-only | artifacts/runs/candidate_benchmarks/new_benchmark_results_20260524_collected.json |
| apho2025 | `deepseek/deepseek-v4-pro` | invalid | text-only | artifacts/runs/candidate_benchmarks/new_benchmark_results_20260524_collected.json |
| incho2026 | `deepseek/deepseek-v4-pro` | invalid | text-only; manual_timeout_after_diagnostic: terminated after >36 min with no output; route already diagnosed as long-tail timeout | artifacts/runs/candidate_benchmarks/incho2026_full_allmodels_20260524_v2_textonly_summary.json |
| usapho2025 | `deepseek/deepseek-v4-pro` | invalid | text-only; manual_timeout_after_diagnostic: terminated after >36 min with no output; route already diagnosed as long-tail timeout | artifacts/runs/candidate_benchmarks/usapho2025_full_allmodels_20260524_v2_textonly_summary.json |
| nbpho2025 | `deepseek/deepseek-v4-pro` | invalid | text-only; manual_timeout_after_diagnostic: terminated after >36 min with no output; route already diagnosed as long-tail timeout o | artifacts/runs/candidate_benchmarks/nbpho2025_full_allmodels_20260524_v2_textonly_summary.json |
| nbpho2025 | `deepseek/deepseek-v4-flash` | caveated | generation length stop | artifacts/runs/candidate_benchmarks/nbpho2025_full_allmodels_20260524_v1_mm_deepseek__deepseek-v4-flash_mm/summary.json |
| ipho2025 | `minimax/MiniMax-M2.7` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| icho2025 | `minimax/MiniMax-M2.7` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| ccho2025 | `minimax/MiniMax-M2.7` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| cpho2025 | `minimax/MiniMax-M2.7` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| ibo2023 | `minimax/MiniMax-M2.7` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| eupho2025 | `minimax/MiniMax-M2.7` | caveated | text-only fallback; text-only | artifacts/runs/candidate_benchmarks/new_benchmark_results_20260524_collected.json |
| apho2025 | `minimax/MiniMax-M2.7` | invalid | text-only | artifacts/runs/candidate_benchmarks/new_benchmark_results_20260524_collected.json |
| incho2026 | `minimax/MiniMax-M2.7` | invalid | text-only; manual_timeout_after_diagnostic: terminated after >36 min with no output; route already diagnosed as long-tail timeout | artifacts/runs/candidate_benchmarks/incho2026_full_allmodels_20260524_v2_textonly_summary.json |
| usapho2025 | `minimax/MiniMax-M2.7` | invalid | text-only; manual_timeout_after_diagnostic: terminated after >36 min with no output; route already diagnosed as long-tail timeout | artifacts/runs/candidate_benchmarks/usapho2025_full_allmodels_20260524_v2_textonly_summary.json |
| nbpho2025 | `minimax/MiniMax-M2.7` | invalid | text-only; manual_timeout_after_diagnostic: terminated after >36 min with no output; route already diagnosed as long-tail timeout o | artifacts/runs/candidate_benchmarks/nbpho2025_full_allmodels_20260524_v2_textonly_summary.json |
| eupho2025 | `z-ai/glm-5.1` | invalid |  | artifacts/runs/candidate_benchmarks/new_benchmark_results_20260524_collected.json |
| incho2026 | `z-ai/glm-5.1` | invalid | HTTP 400: {"error":{"code":"10137","message":"{\"request_id\":\"3bfc9e14-575a-11f1-94b2-1a845303fadf\",\"original_error\ | artifacts/runs/candidate_benchmarks/incho2026_full_allmodels_20260524_v1_mm_summary.json |
| ipho2025 | `xiaomi/mimo-v2.5-pro` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| icho2025 | `xiaomi/mimo-v2.5-pro` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| ccho2025 | `xiaomi/mimo-v2.5-pro` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| cpho2025 | `xiaomi/mimo-v2.5-pro` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| ibo2023 | `xiaomi/mimo-v2.5-pro` | caveated | text-only/image-caveated | docs/latest_model_checkpoint_20260524_zh.md |
| eupho2025 | `xiaomi/mimo-v2.5-pro` | caveated | text-only fallback; text-only | artifacts/runs/candidate_benchmarks/new_benchmark_results_20260524_collected.json |
| apho2025 | `xiaomi/mimo-v2.5-pro` | invalid | text-only route HTTP 400; text-only | artifacts/runs/candidate_benchmarks/new_benchmark_results_20260524_collected.json |
| incho2026 | `xiaomi/mimo-v2.5-pro` | caveated | text-only | artifacts/runs/candidate_benchmarks/incho2026_full_allmodels_20260524_v2_textonly_summary.json |
| usapho2025 | `xiaomi/mimo-v2.5-pro` | invalid | text-only; HTTP 400: {"error":{"code":"10137","message":"{\"request_id\":\"69ccec4f-5759-11f1-af2b-82348ca91c0d\",\"original_error\ | artifacts/runs/candidate_benchmarks/usapho2025_full_allmodels_20260524_v2_textonly_summary.json |
| nbpho2025 | `xiaomi/mimo-v2.5-pro` | invalid | text-only; HTTP 400: {"error":{"code":"10137","message":"{\"request_id\":\"5b51b0fc-5765-11f1-9114-f67143c3f1c0\",\"original_error\ | artifacts/runs/candidate_benchmarks/nbpho2025_full_allmodels_20260524_v2_textonly_summary.json |
