# RepoMind Coding Guidelines

## Project Intent
RepoMind is a portfolio-grade hybrid RAG application for repository intelligence. Changes should keep the system grounded in repository evidence, with clear citations and modular analysis components.

## Engineering Rules
- Prefer small, typed modules over large orchestration files.
- Keep repo analysis deterministic where possible; LLM calls should be optional and gracefully degraded.
- Preserve source metadata: every chunk should retain repo id, file path, role, language, and line range.
- Avoid silently swallowing errors. Log recoverable failures with enough context to debug.
- Do not hardcode API keys or provider-specific secrets.
- Keep generated artifacts under `data/` and avoid committing cloned repos or indexes.

## Testing Expectations
- Add or update tests when scanner, chunker, parser, graph, retrieval, or weakness behavior changes.
- Tests should use temporary repos and small fixture files rather than external network calls.
- If an optional provider is unavailable, test the local fallback path.

## UI/API Expectations
- API responses should be stable and JSON-friendly.
- Streamlit should stay demo-friendly: show status, summary, weaknesses, diagram, and citations clearly.
- Answers must distinguish repo-grounded evidence from uncertainty.
