# Paper table and figure evidence map

The main source is `paper/source/latex/science-arena-CR-final.tex`.

| Paper item | Reproduction source | Evidence class |
|---|---|---|
| Benchmark source table | `metadata/benchmark_composition_2025.json`, 2026 manifests, final LaTeX | Row/display replay |
| Human grader table | Final LaTeX | Display source |
| Judge calibration figure | `results/archived/calibration_replay/`, `analysis/plot_judge_calibration.py` | Row replay |
| One-shot vs interleaved table | `legacy/`, archived release, final aggregate table | Aggregate replay |
| Archived five-model human table | `results/archived/calibration_replay/`, IBO saved scores in archived release | Row replay |
| 13-contest main table, first nine columns | `results/archived/full_benchmark_table/`, archived release, final LaTeX snapshot | Aggregate replay |
| 13-contest main table, four 2026 columns | `results/2026/table6/` plus two raw Table-6 assets | Raw replay |
| Solver response stability | `results/2026/solver_stability_all_models.json`, compact repeat asset | Raw replay |
| Controlled modality ablation | `results/2026/modality_ablation.md` | Aggregate replay; original raw tree unavailable |
| Subfield diagnostics figure/table | `analysis/build_section3_analysis.py`, `analysis/plot_subfield_diagnostics.py` | Row/aggregate replay |
| Four-contest macro CIs | `results/2026/table6/table6_bootstrap_macro_ci.*` | Raw/row replay |
| Modality domain split | `results/2026/modality_ablation.md` | Aggregate replay |
| Six-judge fixed-answer study | `results/2026/crossjudge/`, cross-judge raw asset | Raw replay |
| Archived two-judge item replay | `results/archived/calibration_replay/` | Row replay |
| Structured expert feedback | `results/2026/expert_feedback/` | Row replay |
| Pipeline and protocol figures | TikZ/source in final LaTeX | Display source |
| Domain ability radars | Paper source assets and archived expert taxonomy | Aggregate/display replay |

`scripts/reproduce_tables.py` rebuilds every paper-facing numeric table for
which a structured release artifact exists and validates the rounded 2026 cells
against the submitted LaTeX. The retained final LaTeX is the authoritative
display snapshot when an older raw run tree is unavailable.
