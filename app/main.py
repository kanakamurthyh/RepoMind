from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api.responses import RedactedJSONResponse
from app.api.routes import router
from app.api.security import require_api_key
from app.api.services import ServiceContainer, container, get_container
from app.db.repository import MetadataStoreUnavailableError
from app.rag.llm import resolve_provider
from app.rag.vector_store import IndexStaleError
from app.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
HTTP_ERROR_CODES = {
    400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
    409: "conflict", 413: "payload_too_large", 422: "validation_error", 429: "rate_limited", 503: "service_unavailable",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    container.startup()
    yield
    container.shutdown()


app = FastAPI(
    title="RepoMind",
    description="Repository intelligence: hybrid code RAG, code knowledge graph, verified citations, docs-drift and dependency analysis.",
    version=__version__,
    lifespan=lifespan,
)
app.include_router(router)


@app.exception_handler(IndexStaleError)
async def stale_index_handler(_: Request, exc: IndexStaleError) -> JSONResponse:
    return error_response(409, "index_stale", f"{exc}. Re-run analysis with force=true.")


def error_response(status_code: int, code: str, message: str, headers: dict[str, str] | None = None, **extra: Any) -> JSONResponse:
    """Single error envelope for the whole API: {"detail": {"code": ..., "message": ..., ...}}."""
    return RedactedJSONResponse(status_code=status_code, content={"detail": {"code": code, "message": message, **extra}}, headers=headers)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Normalise every HTTPException (ours, FastAPI's, dependencies') to the {code, message} shape."""
    detail = exc.detail
    headers = getattr(exc, "headers", None)
    if isinstance(detail, dict) and "code" in detail and "message" in detail:
        return RedactedJSONResponse(status_code=exc.status_code, content={"detail": detail}, headers=headers)
    code = HTTP_ERROR_CODES.get(exc.status_code, "error" if exc.status_code < 500 else "internal_error")
    message = detail if isinstance(detail, str) and detail else HTTPStatus(exc.status_code).phrase
    return error_response(exc.status_code, code, str(message), headers=headers)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = jsonable_encoder(exc.errors())
    first = errors[0] if errors else {}
    where = ".".join(str(part) for part in first.get("loc", []) if part != "body")
    message = f"Invalid request: {where + ': ' if where else ''}{first.get('msg', 'validation failed')}"
    return error_response(422, "validation_error", message, errors=errors)


@app.exception_handler(MetadataStoreUnavailableError)
async def storage_unavailable_handler(_: Request, exc: MetadataStoreUnavailableError) -> JSONResponse:
    logger.error("Request failed: %s", exc)
    return error_response(503, "storage_unavailable", "The metadata database is unavailable; see server logs.")


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return error_response(400, "invalid_input", str(exc))


@app.exception_handler(Exception)
async def unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
    reference = uuid.uuid4().hex[:10]
    logger.exception("Unhandled error (ref %s)", reference, exc_info=exc)
    return error_response(500, "internal_error", "Internal error.", reference=reference)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["ops"], dependencies=[Depends(require_api_key)])
def ready(services: ServiceContainer = Depends(get_container)) -> Any:
    try:
        analyzer = services.analyzer
    except MetadataStoreUnavailableError as exc:
        logger.error("Not ready: %s", exc)
        return error_response(
            503, "storage_unavailable", "The metadata database is unavailable; see server logs.",
            storage={"backend": None, "degraded": True, "fallback_reason": str(exc)},
        )
    provider, reason = resolve_provider(services.settings)
    storage = analyzer.metadata.storage_info() if hasattr(analyzer.metadata, "storage_info") else {"backend": analyzer.metadata.backend}
    return {
        "status": "degraded" if storage.get("degraded") else "ready",
        "version": __version__,
        "metadata_backend": analyzer.metadata.backend,
        "storage": storage,
        "embedding_model": analyzer.vector_store.embedder.model_id,
        "vector_backend": analyzer.vector_store.backend_name,
        "tree_sitter": analyzer.parser.tree_sitter_available,
        "llm_provider": provider,
        "llm_note": reason,
        "reranker": services.chat.retriever.reranker.name,
        "osv_scan": services.settings.osv_scan_enabled,
        "auth_required": bool(services.settings.api_key),
    }
