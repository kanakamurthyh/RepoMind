# RepoMind: Repository Intelligence with Hybrid Code RAG

[![CI](https://github.com/kanakamurthyh/RepoMind/actions/workflows/tests.yml/badge.svg)](https://github.com/kanakamurthyh/RepoMind/actions/workflows/tests.yml)

RepoMind analyzes a public GitHub repository and answers questions about it with **line-level citations that are checked against the evidence it actually retrieved**. It combines a code knowledge graph, multi-channel hybrid retrieval, and deterministic static analysis. Analysis and Q&A run without an LLM or API key; an LLM (Gemini or any OpenAI-compatible endpoint, including local Ollama) is optional.

> Full project status, architecture, and design notes: **[CURRENT_STATE.md](CURRENT_STATE.md)**

## What it does

| Capability | How |
| --- | --- |
| **Grounded Q&A** | 7-channel retrieval (dense · BM25 · exact symbol · mentioned files · intent priority · extracted facts · graph expansion) → weighted RRF → reranker → answer with `[S#]` citations. Answers report evidence strength and abstain when the repo lacks strong evidence |
| **Repo Detective agent** | Tool-using investigation (search, grep, read_file, find_symbol, callers, callees, call_path). Works with an LLM or with a deterministic plan |
| **Code knowledge graph** | Files, symbols (qualified `Class.method`), resolved imports, resolved calls, routes → handlers, env vars, inheritance; PageRank centrality, import cycles |
| **Docs-vs-code drift** | Checks README commands, `npm run`/`make` targets, env vars, documented endpoints and links against the actual code |
| **Dependency risk** | OSV.dev scan of pinned versions with **import-level** reachability and fixed versions (network; can be disabled) |
| **Git hotspots** | Churn × complexity hotspots, contributors, bus factor (within the shallow-clone history window) |
| **Static findings** | AST-based Python checks, regex JS/TS XSS/exec checks, entropy-aware secret detection, committed `.env` files |
| **Health score** | Traceable hygiene signals with an assessed denominator and possible score range; unavailable dependency checks remain unknown. An indicator, not a correctness guarantee |
| **MCP server** | Analysis, search, Q&A, the Detective agent, symbol lookup, file reads and insights as MCP tools (stdio) for Claude Code, Cursor, or any MCP client |

## What is (and isn't) verified

- **Default "dense" channel is lexical.** Out of the box it uses an offline feature-hashing embedder, not a neural model. Semantic embeddings (`all-MiniLM-L6-v2`) and the cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) are general-purpose, not code-tuned, and opt-in: install `requirements-advanced.txt` **and** set `EMBEDDING_PROVIDER` / `RERANKER_PROVIDER`.
- **Citation checks are structural.** Source ids and complete line ranges are checked against text actually presented in the prompt or tool observation. Symbol locations alone do not count as inspected source. Whether a cited span actually *supports* the sentence remains **unassessed**.
- **Partial coverage is reported.** Repository caps, per-file truncation, unreadable files and Python parse failures are explicit. File-count fractions do not measure semantic understanding.
- **Repo text is untrusted.** It is escaped inside the prompt's source blocks, secrets are redacted before reaching an LLM, the UI or logs, and the agent's `grep` is literal by default with bounded regex.

## Core vs. optional extras

| Feature | Needs |
| --- | --- |
| Analysis, hybrid retrieval (hashing embedder + BM25), graph, drift, static findings, API, UI | `requirements.txt` |
| Tree-sitter parsing for JS/TS/Java/Go/Rust/C/C++/C#/Ruby/PHP/Kotlin/Swift/Scala (otherwise regex fallback; Python always uses its AST) | `requirements-advanced.txt` (included in the Docker image) |
| Semantic embeddings, cross-encoder reranking | `requirements-advanced.txt` + `EMBEDDING_PROVIDER=sentence-transformers`, `RERANKER_PROVIDER=cross-encoder` |
| Chroma dense backend | `requirements-advanced.txt` + `VECTOR_STORE_PROVIDER=chroma` |
| MCP server | `requirements-advanced.txt` |
| LLM-written answers, LLM-driven Detective | `GEMINI_API_KEY` or `OPENAI_*` settings |

## Retrieval benchmark

`python -m evals.run_eval` (file-level retrieval, default hashing embedder):

| system | hit@1 | hit@5 | MRR |
|---|---|---|---|
| dense only (hashing embedder) | 0.85 | 1.00 | 0.89 |
| BM25 only | 0.80 | 1.00 | 0.89 |
| **RepoMind hybrid** | **1.00** | **1.00** | **1.00** |

**Read these as a regression guard, not a quality claim:** 20 questions written by the author, on one synthetic fixture repo that the retriever's weights were tuned against, and the "dense" baseline is the lexical hashing embedder. With the optional general-purpose semantic embedder and cross-encoder, one local run scored hybrid recall@1 0.90 / MRR 0.95; CI's advanced job reports this each run. A held-out, multi-repo eval is the top roadmap item. Details in [CURRENT_STATE.md](CURRENT_STATE.md#8-evaluation).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt            # core
python -m pip install -r requirements-advanced.txt   # optional: tree-sitter, Chroma, semantic models (PyTorch), MCP
cp .env.example .env                                 # optionally add GEMINI_API_KEY / OPENAI_* / GITHUB_TOKEN

bash scripts/run_backend.sh     # FastAPI on http://127.0.0.1:8000 (docs at /docs)
bash scripts/run_frontend.sh    # Streamlit UI on http://localhost:8501
```

Analyze from the command line:

```bash
curl -X POST http://127.0.0.1:8000/analyze_repo_async -H 'Content-Type: application/json' \
  -d '{"repo_url": "https://github.com/tiangolo/fastapi"}'
# → {"job_id": "...", "events_url": "/jobs/<id>/events", ...}   (stream progress with SSE)
```

Use it from Claude Code over MCP (requires `requirements-advanced.txt`):

```bash
claude mcp add repomind -- "$PWD/.venv/bin/python" -m app.mcp_server
```

Repo size is pre-checked through the GitHub API (an optional `GITHUB_TOKEN` raises the rate limit) and the on-disk clone size is capped (`MAX_REPO_SIZE_MB`). For air-gapped runs set `OSV_SCAN_ENABLED=false` and `GITHUB_API_CHECKS_ENABLED=false`.

## Docker

```bash
echo "POSTGRES_PASSWORD=change-me" >> .env
docker compose up --build        # postgres + api (:8000) + ui (:8501), bound to localhost
```

The image installs runtime dependencies plus tree-sitter; test tooling and the semantic/Chroma/MCP extras are not included.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/analyze_repo_async` | Start analysis job (returns `job_id`) |
| GET | `/jobs/{job_id}` · `/jobs/{job_id}/events` | Job status · Server-Sent Events progress |
| POST | `/analyze_repo` | Synchronous analysis |
| GET | `/repos` · `/analysis/{id}/status` | Analyzed repositories · analysis status |
| GET | `/repo/{id}/summary` · `/weaknesses` · `/insights` | Analysis artifacts |
| GET | `/repo/{id}/graph?view=summary\|files` | Code graph / file dependency graph |
| GET | `/repo/{id}/symbols?name=` · `/file?path=&start=&end=` | Symbol lookup with callers/callees · code viewer |
| POST | `/repo/{id}/chat` · `/search` · `/investigate` | Grounded answer · hybrid search · Detective agent |
| DELETE | `/repo/{id}` | Remove all artifacts |
| GET | `/health` · `/ready` | Liveness · configuration/readiness |

If `REPOMIND_API_KEY` is set, every call except `/health` requires the `X-API-Key` header. Rate limiting and job execution are in-process. One API worker is enforced per data directory; `ANALYSIS_QUEUE_CAPACITY` bounds queued plus running jobs (default 8). Run on Linux or macOS; distributed workers are not configured.

## Snapshot consistency and CLI

Analyses build immutable vector, graph, metadata and redacted source artifacts under unique run ids. A completed run becomes active with one atomic pointer replacement. Failed rebuilds retain the previous analysis; questions pin their run, and citations can open that historical source after a refresh. Old runs are retained until repository deletion. Monitor `data/` disk usage when refreshing frequently; automatic run pruning is not implemented.

Install the CLI from this checkout:

```bash
python -m pip install -e .
repomind analyze https://github.com/owner/repo
repomind analyze ./path/to/local/repo --force
repomind ask <repo_id> 'How does this function work?'
repomind check ./path/to/local/repo --format sarif --fail-on high
```

Local analysis reads source files; it does not execute repository code or delete the original directory. A copied run's source viewer remains available without the local checkout. The Health tab includes filterable static findings with evidence and suggested fixes.

## External evaluation

The [evaluation workflow](evals/README.md) prepares 160 draft questions across eight SHA-pinned Python and JS/TS repositories, with repository-level development/test splits. It compares BM25, hashing, hybrid retrieval, graph and reranker ablations, and optional semantic/neural models. Results distinguish Hit@k, fractional recall, complete evidence retrieval and span coverage. Draft labels require an explicit flag and are not a release quality gate; independent answer/citation review and developer studies remain necessary. Application defaults are unchanged by experiments.

## Development

```bash
python -m pip install -r requirements-dev.txt      # runtime + pytest, pytest-cov, ruff
ruff check . && python -m pytest -q && python -m evals.run_eval
python -m pytest --cov=app                          # coverage report
alembic upgrade head    # PostgreSQL/SQLite schema migrations (DATABASE_URL, default SQLite under data/db)
```

100+ offline tests use temp repos and a fixture repo, never the network. CI runs lint, tests with coverage and the benchmark; a second job installs `requirements-advanced.txt` and exercises the optional backends plus the semantic embedder and cross-encoder; a third builds the Docker image.
