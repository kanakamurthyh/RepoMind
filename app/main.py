from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.utils.logging import configure_logging

configure_logging()

app = FastAPI(
    title="RepoMind",
    description="Hybrid RAG chatbot for GitHub repository intelligence.",
    version="0.1.0",
)
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
