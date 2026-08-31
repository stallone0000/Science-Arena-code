# Reproducibility guide

## Level A: offline paper replay

This level has no model-provider dependency.

```bash
pip install -e '.[analysis,dev]'
python scripts/reproduce_tables.py --output reproduced
python scripts/verify_release.py
```

The command writes:

- `reproduced/main_table_2026.csv`: the four newly sealed contest columns;
- `reproduced/four_contest_macro_ci.csv`: normalized macro and cluster CI;
- `reproduced/solver_stability.csv`: three-route, three-repeat API stability;
- `reproduced/cross_family_judges.csv`: fixed-answer judge means/SDs;
- `reproduced/judge_calibration.csv`: archived expert-vs-judge replay;
- `reproduced/modality_ablation.csv`: the retained three-arm aggregate;
- `reproduced/report.md` and `validation.json`.

The verifier checks file hashes, score/max closure, all 14 four-contest rows,
the published rounded cells in the final LaTeX, credential patterns, hidden
reasoning fields, and Git blob sizes.

## Level B: raw-artifact replay

Download the raw no-CoT assets:

```bash
python scripts/download_artifacts.py --name runs-table6-ipho-icho
python scripts/download_artifacts.py --name runs-table6-nbpho-usapho
python scripts/download_artifacts.py --name runs-solver-repeats
python scripts/download_artifacts.py --name runs-cross-judges
```

Each asset has a SHA-256 in `release_assets.json`. Keep archives compressed if
you only need preservation; extraction creates many small item records.

The raw ledgers support independent score reconstruction for the four 2026
columns and the cross-family study. The compact repeat archive contains visible
responses and rubric scores for Gemini 3.5 Flash, GPT-5.5, and Claude Opus 4.7;
it intentionally excludes provider chain-of-thought and duplicated payloads.

## Level C: live API rerun

Download/extract inputs, set an environment credential, and run each stage:

```bash
python scripts/download_artifacts.py --name inputs-2026 --extract
export QIHOO_API_KEY='YOUR_KEY'

python scripts/run_2026.py validate --contest icho2026
python scripts/run_2026.py probe --contest icho2026
python scripts/run_2026.py solve --contest icho2026 --workers 8
python scripts/run_2026.py judge --contest icho2026 --workers 8
python scripts/run_2026.py aggregate --contest icho2026
```

Run roots default to `runs/<contest>`. Use `--run-root` for a new independent
replicate. Hosted model behavior is not immutable, so regenerated answers need
not equal the August 2026 archive even when the request packet is identical.

## Paper source

The complete final source is at `paper/source/`; the main file is
`paper/source/latex/science-arena-CR-final.tex`. Compile from
`paper/source/` so paths beginning with `latex/` resolve correctly:

```bash
cd paper/source
latexmk -pdf latex/science-arena-CR-final.tex
```

The submitted PDF snapshot is `paper/ScienceArena-CR-0831.pdf`.

## Evidence classes

- **Raw replay:** visible response/judgment records and aggregation inputs are
  preserved.
- **Row replay:** item/expert rows and scores are preserved, without all
  provider transport envelopes.
- **Aggregate replay:** the paper-facing aggregate and protocol code are
  preserved, but the original raw provider tree is unavailable.
- **Display source:** a figure/table is authored directly in LaTeX rather than
  generated from experimental rows.

`PAPER_MAP.md` assigns one class to every experimental table and figure.
