# GitHub Release assets

Tag: `cr-final-2026-08-31`. Machine-readable filenames, sizes, hashes, and
extraction targets are in `release_assets.json`.

| Logical name | Size | Contents |
|---|---:|---|
| `inputs-2026` | 99.2 MiB | Official source PDFs, canonical JSONL, original page PNGs, frozen question-only descriptions, manifests, proofs, snapshots |
| `runs-table6-ipho-icho` | 527.1 MiB | IPhO/IChO 2026 solver and judge ledgers |
| `runs-table6-nbpho-usapho` | 2.0 MiB | NBPhO/USAPhO 2026 solver and judge ledgers |
| `runs-solver-repeats` | 1.9 MiB | Three-model, three-repeat visible responses and scores |
| `runs-cross-judges` | 1.7 MiB | Six-judge, three-repeat fixed-answer study |
| `archived-2025-reproduction` | 111.6 MiB | Earlier benchmark datasets, code, saved outputs, and calibration artifacts |
| `legacy-data` | 251.7 MiB | Consolidated legacy benchmark images, source materials, and scoring details |

Use `python scripts/download_artifacts.py --list` to inspect assets. The
downloader verifies SHA-256 before extraction and rejects archive path traversal.

The four raw-run assets were scanned member-by-member before upload. No
credential-shaped token, private key, or nonempty hidden-reasoning field was
found. `inputs-2026` and the archived package received the same credential
scan. The consolidated legacy data asset was also scanned across all text and
structured-data members and contains neither credentials nor nonempty hidden
reasoning. Do not add a new asset without rerunning the release verifier and
an archive-content scan.

GitHub Releases are used because Git rejects blobs above 100 MB and because
keeping raw ledgers out of ordinary history makes clones small. Release assets
are immutable by convention: replace an asset only with a new tag and a new
hash manifest.
