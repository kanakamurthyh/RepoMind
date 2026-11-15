FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 REPOMIND_DATA_DIR=/app/data
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Runtime deps only (test/lint tooling lives in requirements-dev.txt), plus tree-sitter for
# multi-language parsing. Semantic embeddings, Chroma and MCP stay opt-in (requirements-advanced.txt).
COPY requirements.txt .
RUN pip install -r requirements.txt "tree-sitter>=0.22" "tree-sitter-language-pack>=0.7"

COPY app ./app
COPY evals ./evals
COPY migrations ./migrations
COPY alembic.ini ./
RUN useradd --create-home --uid 10001 repomind && mkdir -p /app/data && chown -R repomind /app/data
USER repomind

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
