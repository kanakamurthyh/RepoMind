# RepoMind: Hybrid RAG Chatbot for GitHub Repository Intelligence

RepoMind analyzes a public GitHub repository, builds a local hybrid knowledge base, generates a structured repo summary, detects weaknesses, and answers questions with source citations.

Unlike a basic "chunk files and ask an LLM" demo, RepoMind combines:

- Semantic vector retrieval over README files, docs, configs, tests, and code chunks.
- Code-aware graph retrieval over files, folders, imports, functions, classes, routes, entry points, and dependencies.
- Static weakness analysis for documentation, testing, deployment, maintainability, and security risks.
- Deterministic repo-fact extraction for summaries, architecture, run instructions, important files, and quality scoring.
- A FastAPI backend and Streamlit demo UI.

## Architecture

```mermaid
flowchart TD
    User["User"] --> UI["Streamlit UI"]
    UI --> API["FastAPI API"]
    API --> Analyzer["RepoAnalyzer"]
    Analyzer --> Git["Git clone/cache"]
    Analyzer --> Scanner["FileScanner"]
    Scanner --> Chunker["Role-aware Chunker"]
    Chunker --> Vector["Vector Store: Chroma optional, local JSON fallback"]
    Analyzer --> Postgres["SQLAlchemy Metadata DB: PostgreSQL/SQLite"]
    Scanner --> Parser["Python AST + optional Tree-sitter parsers"]
    Parser --> Graph["NetworkX Code Graph + persisted metadata"]
    Scanner --> Weaknesses["Weakness Detector"]
    Vector --> Retriever["Hybrid Retriever"]
    Graph --> Retriever
    Retriever --> Chat["Grounded Chat Service"]
```

## Features

- Accepts public `https://github.com/{owner}/{repo}` URLs.
- Avoids re-cloning analyzed repositories unless force rebuild is requested.
- Uses a configurable ingestion policy for file count, repo byte budget, per-file byte limits, symlink handling, and binary metadata capture.
- Ignores `.git`, `node_modules`, virtualenvs, build outputs, caches, and other noisy paths while preserving safe metadata for unsupported binaries.
- Detects languages, docs, configs, test files, important files, framework hints, and entry points.
- Generates polished repo-level reports from structured facts instead of copying arbitrary retrieved chunks.
- Prioritizes official docs for setup/run questions: `INSTALL.md`, top-level `README.md`, setup docs, package files, then only helper scripts explicitly mentioned by docs.
- Uses different chunking strategies for docs, configs, Python code, and fallback text.
- Uses Python AST chunking and JS/TS symbol-aware chunking for code files.
- Persists metadata through SQLAlchemy 2.x, with PostgreSQL/Alembic support and SQLite/JSON fallback for local demos.
- Stores vector indexes in Chroma when `VECTOR_STORE_PROVIDER=chroma`; otherwise uses a persistent local JSON fallback for no-network testing.
- Builds a NetworkX graph with file, folder, dependency, local file dependency, function, class, route, call, inheritance, and entrypoint relationships where detectable.
- Uses graph node/edge scoring to boost chunks from files connected to relevant routes, imports, functions, classes, and entry points.
- Classifies chat questions into overview, installation, architecture, file location, function/class, flow, dependency, weakness, testing, and security intents.
- Answers with file references and line ranges from retrieved chunks.
- Provides rendered Mermaid architecture diagrams, graph metrics, weakness reports, and a structured initial summary.
- Computes a 0-100 quality score across documentation, test signals, structure, and run/deployment readiness.
- Reports security as automated static review findings with severity, confidence, affected location, evidence, and remediation instead of a misleading numeric security score.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

RepoMind works in local extractive mode without an external LLM key. To use Gemini for grounded answer synthesis, put your key in `.env`; `LLM_PROVIDER=auto` will use Gemini automatically when the key is present.

```bash
GEMINI_API_KEY=your_real_key_here
LLM_PROVIDER=auto
GEMINI_MODEL=gemini-flash-latest
```

Restart the backend after changing `.env`. If you copied an older config that says `LLM_PROVIDER=local`, Gemini will still be used when `GEMINI_API_KEY` is present. Use `LLM_PROVIDER=off` only when you explicitly want to disable external LLM calls.

For semantic embeddings, Chroma, and Tree-sitter grammar packages, install the optional advanced stack:

```bash
python -m pip install -r requirements-advanced.txt
```

Chroma is the selected vector DB because it is simple to run locally, persists to disk without a separate service, supports metadata filters, and can be swapped behind RepoMind's vector-store interface. PostgreSQL is used for production-style metadata persistence; SQLite is the safe default when `DATABASE_URL` is unset.

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Optional key for future OpenAI synthesis/embedding providers. |
| `GEMINI_API_KEY` | Optional key for Gemini answer synthesis. |
| `EMBEDDING_PROVIDER` | Defaults to `sentence-transformers`; falls back to deterministic local hashing when optional semantic deps are unavailable. |
| `EMBEDDING_MODEL` | Semantic embedding model name, default `sentence-transformers/all-MiniLM-L6-v2`. |
| `EMBEDDING_BATCH_SIZE` | Batch size for embedding providers. |
| `VECTOR_STORE_PROVIDER` | `local` or `chroma`; defaults to `local` for reliable demos. |
| `LLM_PROVIDER` | Use `auto` for Gemini when `GEMINI_API_KEY` exists, or `off` to force no external LLM. |
| `GEMINI_MODEL` | Gemini model name, default `gemini-flash-latest`; RepoMind retries current Flash fallbacks if a configured model returns 404. |
| `LOCAL_MODEL_MODE` | Keep `true` for no-key local demos; Gemini still works automatically when a key is set. |
| `REPOMIND_DATA_DIR` | Artifact directory, default `data`. |
| `DATABASE_URL` | SQLAlchemy database URL. Use PostgreSQL for production-style runs; unset uses SQLite under `data/db`. |
| `USE_METADATA_DB` | Enables SQLAlchemy metadata persistence, default `true`. |
| `MAX_FILE_SIZE_BYTES` | Per-file scan cap, default `300000`. |
| `MAX_CHUNKS_PER_REPO` | Safety cap for vector indexing, default `5000`. |
| `MAX_FILES_PER_REPO` | Maximum scanned files before stopping, default `15000`. |
| `MAX_TOTAL_REPO_BYTES` | Maximum total bytes scanned per repo, default `250000000`. |
| `BINARY_METADATA_ENABLED` | Capture binary/unsupported metadata without embedding content. |
| `FOLLOW_SYMLINKS` | Defaults to `false` to avoid symlink traversal. |

## Database Setup

For PostgreSQL-backed metadata:

```bash
docker compose up -d postgres
source .venv/bin/activate
export DATABASE_URL=postgresql+psycopg://repomind:repomind@localhost:5432/repomind
alembic upgrade head
```

For local development without Docker, leave `DATABASE_URL` blank and RepoMind will create a SQLite database under `data/db/repomind.db`.

If you tried the PostgreSQL command but Docker is not installed, reset the shell variable before using the local SQLite path:

```bash
unset DATABASE_URL
alembic upgrade head
```

## Run Backend

```bash
bash scripts/run_backend.sh
```

Open API docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The backend script runs without reload by default because macOS can deny file watching across virtualenv or data folders. To opt into reload during development:

```bash
RELOAD=1 bash scripts/run_backend.sh
```

## Run Frontend

In a second terminal:

```bash
bash scripts/run_frontend.sh
```

The Streamlit app lets you analyze a repository, inspect the generated summary, review weaknesses, copy the Mermaid diagram text, chat with the repo, and view citations.

## Docker

```bash
docker build -t repomind .
docker run --rm -p 8000:8000 repomind
```

The Docker image runs the FastAPI backend. Run the Streamlit UI locally or create a separate UI image if you want a two-container demo.

## API Endpoints

- `POST /analyze_repo`
- `POST /analyze_repo_async`
- `GET /analysis/{repo_id}/status`
- `GET /repo/{repo_id}/summary`
- `POST /repo/{repo_id}/chat`
- `GET /repo/{repo_id}/weaknesses`
- `GET /repo/{repo_id}/graph`
- `DELETE /repo/{repo_id}`

Example:

```bash
curl -X POST http://127.0.0.1:8000/analyze_repo \
  -H "Content-Type: application/json" \
  -d '{"repo_url":"https://github.com/tiangolo/fastapi","force":false}'
```

## Example Questions

- What does this repository do?
- How do I install this on Linux?
- How do I run this repository locally?
- Which files should I read first?
- How do the main modules depend on each other?
- Where is the API entry point?
- What routes are defined?
- What tests exist and what is missing?
- What are the highest-risk weaknesses?
- How would you improve deployment readiness?

## How Hybrid Retrieval Works

1. The analyzer extracts deterministic repo facts: README summary, tree, languages, entrypoints, important files, package files, scripts, tests, docs, and deployment signals.
2. The query router classifies questions into repo overview, run instructions, architecture, file/function explanation, dependency, weakness, improvement, testing, security, or general code intents.
3. Vector search retrieves semantically relevant chunks with file path and line metadata.
4. Graph retrieval scores nodes and edges such as imports, definitions, routes, calls, entry points, and package dependencies.
5. Files discovered from facts and graph relationships are used to boost source chunks even when semantic similarity alone would miss them.
6. The chat service formats a clean answer with direct explanation, important files, evidence, and uncertainty notes while keeping retrieval internals out of the user-facing UI.

## Limitations

- Chat and the Run Instructions assistant answer use Gemini when `GEMINI_API_KEY` is configured. They fall back locally when the API key is missing or a Gemini request fails.
- Tree-sitter extraction is used when optional grammar packages are installed; regex remains an emergency fallback for languages without an available grammar.
- Function call edges are best-effort and do not resolve dynamic dispatch or imports across modules.
- Very large repositories may need higher chunk limits, a durable task queue, and a production-tuned database deployment.
- Binary/media files are intentionally represented as metadata only.
- Mermaid rendering in Streamlit uses a CDN component; the raw Mermaid source is always shown as a fallback.

## Future Improvements

- Add OpenAI provider support and richer provider selection for generative synthesis.
- Add a durable worker queue such as Celery/RQ for multi-user background analysis.
- Expand Tree-sitter query coverage for deeper call resolution and exports.
- Stream analysis progress from the backend instead of polling in-memory status.
- Add richer duplicate-code and modularity metrics.
- Add authentication and multi-user project storage.

## Resume Bullets

- Built a hybrid RAG repository intelligence app using FastAPI, Streamlit, NetworkX, AST parsing, and local vector retrieval.
- Implemented repository ingestion, role-aware chunking, code graph extraction, weakness detection, Mermaid architecture diagrams, and citation-grounded chat.
- Designed a provider-friendly architecture that runs offline by default and can be extended with external LLM or embedding APIs.
