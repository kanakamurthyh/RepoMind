# RepoMind Data Directory

Runtime analysis artifacts (all ignored by Git; `.gitkeep` files preserve the layout):

- `repos/`: Git clones, one directory per analyzed repository.
- `indexes/`: chunk sidecars + BM25 data (`*.chunks.json`), dense vectors (`*.npy`) or `chroma/`.
- `graphs/`: full code knowledge graphs (`*.json.gz`).
- `metadata/`: JSON fallback for analysis metadata when the database is unavailable.
- `db/`: SQLite database when `DATABASE_URL` is unset.
# Analysis runs and evaluation artifacts

`runs/<repo_id>/<run_id>/` stores an immutable analysis, manifest and redacted source snapshots.
`runs/<repo_id>/active.json` is the atomically published active-run pointer. Vectors and full graphs use the run's unique artifact id in `indexes/` and `graphs/`. Failed staging is cleaned up; previous successful runs remain readable until the repository is deleted. Do not edit run artifacts in place or prune runs while readers use them.

`evals/` contains external checkouts, model weights, draft labels and experiment results. `review_2026-10-22/` contains the audit, verification and research-review kit. These are generated local artifacts, not application source. Do not commit cloned repositories, indexes or downloaded models.
