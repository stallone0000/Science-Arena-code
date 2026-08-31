# Security and credentials

ScienceArena never requires a credential file inside the repository.

- Set the 360 credential as `QIHOO_API_KEY` (or the compatible
  `API_360_KEY`/`SCIENCE_ARENA_API_KEY` names accepted by the sealed runner).
- Do not pass credentials as CLI arguments; they can appear in shell history
  and process listings.
- Do not commit `.env`, `API_KEY.sh`, request headers, provider dashboards, or
  screenshots containing tokens.
- Raw release artifacts must pass `python scripts/verify_release.py` before
  upload. Hidden reasoning/chain-of-thought is intentionally discarded.

If a credential is exposed, revoke it at the provider immediately, remove it
from every Git object and release asset, rotate it, and rerun the verifier.
