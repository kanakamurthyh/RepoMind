# RepoMind Data Directory

Runtime analysis artifacts are stored here:

- `repos/`: shallow Git clones, one directory per analyzed repository.
- `indexes/`: local vector indexes.
- `metadata/`: JSON summaries, graph facts, chunks, and weakness reports.

The generated contents are ignored by Git. The `.gitkeep` files preserve the expected directory structure.
