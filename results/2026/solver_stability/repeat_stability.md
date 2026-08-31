# IPhO/IChO 2026 three-fresh-repeat stability

Three all-new independent calls per route; the published Table-6 run is an external comparison only.

| Model | Input | IPhO A/B/C (/30) | IPhO mean±SD | IChO A/B/C (/60) | IChO mean±SD | Macro A/B/C (%) | Macro mean±SD (pp) | Range (pp) | Exact-answer same | End-to-end item score changed |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemini 3.5 Flash | vision | 29.75/27.75/28.75 | 28.75±1.00 | 54.76/52.99/53.45 | 53.73±0.92 | 95.22/90.41/92.46 | 92.70±2.41 | 4.81 | 0.0% | 22.0% |

SD is the sample standard deviation across the three fresh run-level macro scores. No explicit seed or temperature was sent. These repeats estimate end-to-end provider/backend stochasticity under an otherwise identical frozen protocol; they do not isolate solver variance from judge variance. Judge-only variance is reported by the separate fixed-answer cross-judge experiment.

IPhO repC required 1 deterministic evidence-quote overlay(s) (1/23 selected cells): all 7 saved HTTP-200 responses replayed strict-valid after quote-lexeme-only repair, with scores and all rubric decisions unchanged and no new API call. A conservative sensitivity that instead assigns each recovered cell zero is included in the JSON output.

IChO scores are independently recomputed problem by problem as raw rubric points divided by the official raw maximum and multiplied by the official problem weight; the nine weights sum to 60. The machine-readable protocol audit must be PASS before this table is emitted.
