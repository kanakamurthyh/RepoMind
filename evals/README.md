# Evaluation workflow

The ShopAPI fixture is a development regression guard, not an external quality claim:

```bash
python -m evals.run_eval
```

`hit@k` means at least one gold file was found. `recall@k` measures the fraction
of required files found; `complete@k` requires every file in a valid evidence
set. Unsupported questions are excluded from those denominators. Span recall
requires full line-range coverage, not a one-line overlap.

## External corpus

`corpus_seeds.py` defines eight repositories and 20 draft questions per
repository. There are four Python and four JS/TS repositories, with four
development and four test repositories. `prepare_corpus` records checkout
commit SHAs, license hashes, source hashes and candidate evidence spans.

The seeds are **draft annotations authored during implementation**. Some spans
are broad; inferred answers and unsupported cases need independent review.
They are not human-validated gold and cannot establish answer accuracy.

Fetch each public repository named in `REPOSITORIES` into
`data/evals/repos/<owner>--<repo>` using a shallow clone. Do not install or run
the repositories. Then:

```bash
python -m evals.prepare_corpus
python -m evals.run_corpus --allow-draft --split dev
```

The runner verifies each checkout and source hash. It refuses draft labels
unless `--allow-draft` is supplied and records that status in the results.
The test split stays unused until independent labels and evaluation settings
are frozen. Do not tune on the test set or present development numbers as
held-out results.

Optional models require `pip install '.[semantic]'` and download public weights:

```bash
HF_HOME=data/evals/models TOKENIZERS_PARALLELISM=false \
  python -m evals.run_corpus --allow-draft --semantic --neural-reranker \
  --output data/evals/results-dev-neural.json
```

This adds a semantic retriever and a neural reranker as separate changes. It
does not change application defaults. Neural-provider initialization must
succeed explicitly; the experiment never counts a lexical fallback as a
semantic run. Reports record the resolved embedding revision and provider
configuration, per-repository metrics, latency and paired cluster-bootstrap
differences. Model downloads and runtime are excluded from warm query latency.

## Before using results as a quality gate

1. Independently narrow candidate spans and finalize answers, including evidence
   that refutes false premises. Recheck unsupported cases against coverage.
2. Double-review at least 25% of questions and every disputed/unsupported case;
   record reviewer identifiers, disagreements and resolutions.
3. Set `review_status` to `approved` only after that process. Freeze the corpus
   content hash, model revisions, prompts, budgets and scoring rubric.
4. Run the test split once under the frozen settings. Keep per-repository results
   and uncertainty visible; four test repositories are still a small sample.
5. Independently judge claim correctness and citation support. Source-reference
   validity and retrieval recall cannot substitute for these judgments.

Artifacts and checkouts stay under `data/evals/` and are ignored by Git. An
annotation guide and developer-study kit are provided alongside the local
execution report under `data/review_2026-10-22/`.
