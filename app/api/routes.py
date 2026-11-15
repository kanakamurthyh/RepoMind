from __future__ import annotations

import asyncio
import json
import logging
from contextlib import nullcontext
from functools import wraps
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.responses import RedactedJSONResponse
from app.api.security import rate_limit, require_api_key
from app.api.services import ServiceContainer, error_code_for, get_container
from app.ingestion.coverage import ensure_coverage
from app.utils.paths import REPO_ID_PATTERN

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_api_key)], default_response_class=RedactedJSONResponse)
RepoId = Annotated[str, Path(pattern=REPO_ID_PATTERN, max_length=240, description="Repository id returned by analysis")]
Services = Annotated[ServiceContainer, Depends(get_container)]


def snapshot_response(function):
    @wraps(function)
    def wrapped(repo_id, services, *args, **kwargs):
        analyzer = services.analyzer
        with analyzer.read_snapshot(repo_id) if hasattr(analyzer, "read_snapshot") else nullcontext():
            return function(repo_id, services, *args, **kwargs)
    return wrapped


class AnalyzeRequest(BaseModel):
    repo_url: str = Field(..., max_length=300, description="Public GitHub repository URL")
    force: bool = Field(False, description="Fetch the latest commit and rebuild every artifact")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)
    debug: bool = False


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(8, ge=1, le=20)


class InvestigateRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    max_steps: int = Field(6, ge=1, le=10)


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    """Every API error uses the same JSON detail shape: {"code": ..., "message": ...}."""
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _analysis_http_exception(exc: Exception) -> HTTPException:
    code, message = error_code_for(exc)
    status = {
        "invalid_input": 400, "repo_not_found": 404, "repo_too_large": 413, "clone_failed": 503,
        "clone_timeout": 504, "storage_unavailable": 503,
        "queue_full": 503,
    }.get(code, 500)
    return api_error(status, code, message)


def _not_analyzed(repo_id: str, exc: Exception | None = None) -> HTTPException:
    if exc is not None:
        logger.info("Artifacts for %s not found: %s", repo_id, exc)
    return api_error(404, "analysis_not_found", f"No analysis found for {repo_id}; run analysis first.")


def _analysis_payload(result: Any) -> dict[str, Any]:
    return {
        "repo_id": result.repo_id,
        "repo_url": result.repo_url,
        "commit_sha": result.commit_sha,
        "summary": result.summary,
        "weaknesses": [w.to_dict() for w in result.weaknesses],
        "graph": result.graph,
        "statuses": result.statuses,
    }


# --- analysis ------------------------------------------------------------------------------
@router.post("/analyze_repo", dependencies=[Depends(rate_limit)], tags=["analysis"])
def analyze_repo(request: AnalyzeRequest, services: Services) -> dict[str, Any]:
    """Synchronous analysis (blocks until done). Prefer /analyze_repo_async for UIs."""
    try:
        result = services.analyzer.analyze(request.repo_url, force=request.force)
    except Exception as exc:
        raise _analysis_http_exception(exc) from exc
    if hasattr(services.chat, "clear_cache"):
        services.chat.clear_cache()
    return _analysis_payload(result)


@router.post("/analyze_repo_async", dependencies=[Depends(rate_limit)], status_code=202, tags=["analysis"])
def analyze_repo_async(request: AnalyzeRequest, services: Services) -> dict[str, Any]:
    try:
        job = services.jobs.submit(request.repo_url, request.force)
    except Exception as exc:
        raise _analysis_http_exception(exc) from exc
    return {
        **job,
        "status_url": f"/jobs/{job['job_id']}",
        "events_url": f"/jobs/{job['job_id']}/events",
    }


@router.get("/jobs/{job_id}", tags=["analysis"])
def job_status(job_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")], services: Services) -> dict[str, Any]:
    job = services.analyzer.metadata.get_job(job_id)
    if job is None:
        raise api_error(404, "job_not_found", "Job not found.")
    return job


@router.get("/jobs/{job_id}/events", tags=["analysis"])
async def job_events(job_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")], services: Services) -> StreamingResponse:
    """Server-Sent Events stream of job progress until the job completes or fails."""
    if services.analyzer.metadata.get_job(job_id) is None:
        raise api_error(404, "job_not_found", "Job not found.")

    async def stream():
        last = None
        for _ in range(3600):
            job = await asyncio.to_thread(services.analyzer.metadata.get_job, job_id)
            if job is None:
                break
            snapshot = (job["state"], job["stage"], job["progress"])
            if snapshot != last:
                last = snapshot
                yield f"event: progress\ndata: {json.dumps(job)}\n\n"
            if job["state"] in {"completed", "failed"}:
                yield f"event: {job['state']}\ndata: {json.dumps(job)}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@router.get("/analysis/{repo_id}/status", tags=["analysis"])
def analysis_status(repo_id: RepoId, services: Services) -> dict[str, Any]:
    job = services.analyzer.metadata.latest_job(repo_id) if hasattr(services.analyzer.metadata, "latest_job") else None
    if job and job["state"] in {"queued", "running", "failed"}:
        return job
    try:
        result = services.analyzer.load(repo_id)
    except FileNotFoundError as exc:
        if job:
            return job
        raise _not_analyzed(repo_id, exc) from exc
    return {"repo_id": repo_id, "state": "completed", "repo_url": result.repo_url, "commit_sha": result.commit_sha, "statuses": result.statuses}


@router.get("/repos", tags=["analysis"])
def list_repos(services: Services) -> list[dict[str, Any]]:
    return services.analyzer.list_repos()


# --- analysis artifacts --------------------------------------------------------------------------
def _load(services: ServiceContainer, repo_id: str) -> Any:
    try:
        return services.analyzer.load(repo_id)
    except FileNotFoundError as exc:
        raise _not_analyzed(repo_id, exc) from exc


@router.get("/repo/{repo_id}/summary", tags=["repo"])
def repo_summary(repo_id: RepoId, services: Services) -> dict[str, Any]:
    """Analysis summary; always includes a ``coverage`` object (derived for older cached analyses)."""
    return ensure_coverage(_load(services, repo_id).summary)


@router.get("/repo/{repo_id}/weaknesses", tags=["repo"])
def repo_weaknesses(
    repo_id: RepoId,
    services: Services,
    severity: str | None = Query(None, pattern="^(critical|high|medium|low)$"),
    category: str | None = Query(None, max_length=40),
) -> list[dict[str, Any]]:
    weaknesses = _load(services, repo_id).weaknesses
    return [w.to_dict() for w in weaknesses if (severity is None or w.severity == severity) and (category is None or w.category == category)]


@router.get("/repo/{repo_id}/graph", tags=["repo"])
@snapshot_response
def repo_graph(repo_id: RepoId, services: Services, view: Literal["summary", "files"] = "summary") -> dict[str, Any]:
    graph = _load(services, repo_id).graph
    if view == "summary":
        return graph
    index = services.analyzer.graph_index(repo_id)
    if index is None:
        raise api_error(409, "graph_missing", "Full graph missing; re-run analysis.")
    files = services.analyzer.graph_builder.file_graph(index.graph)
    return {
        "nodes": [{"id": node, "degree": files.degree(node)} for node in files.nodes if files.degree(node) > 0][:400],
        "edges": [{"source": s, "target": t, "weight": a.get("weight", 1)} for s, t, a in files.edges(data=True)][:1500],
    }


@router.get("/repo/{repo_id}/insights", tags=["repo"])
def repo_insights(repo_id: RepoId, services: Services) -> dict[str, Any]:
    summary = ensure_coverage(_load(services, repo_id).summary)
    return {
        "insights": summary.get("insights", {}),
        "quality_score": summary.get("quality_score", {}),
        "ingestion": summary.get("ingestion", {}),
        "coverage": summary["coverage"],
    }


@router.get("/repo/{repo_id}/file", tags=["repo"])
def repo_file(
    repo_id: RepoId,
    services: Services,
    path: str = Query(..., min_length=1, max_length=500),
    start: int = Query(1, ge=1),
    end: int | None = Query(None, ge=1),
    run_id: str | None = Query(None, pattern=r"^[0-9a-f]{32}$"),
) -> dict[str, Any]:
    try:
        if run_id:
            return services.analyzer.read_file(repo_id, path, start, end, run_id=run_id)
        return services.analyzer.read_file(repo_id, path, start, end)
    except FileNotFoundError as exc:
        raise api_error(404, "file_not_found", f"File not found: {path}") from exc
    except ValueError as exc:
        raise api_error(400, "invalid_input", str(exc)) from exc


@router.get("/repo/{repo_id}/symbols", tags=["repo"])
def repo_symbols(repo_id: RepoId, services: Services, name: str = Query(..., min_length=1, max_length=200)) -> list[dict[str, Any]]:
    index = services.analyzer.graph_index(repo_id)
    if index is None:
        raise api_error(404, "graph_missing", "No graph for this repository; run analysis first.")
    rows = []
    for node in index.find_symbols(name):
        rows.append({**node, "callers": index.callers(node["id"], limit=10), "callees": index.callees(node["id"], limit=10)})
    return rows


# --- question answering ------------------------------------------------------------------------------
@router.post("/repo/{repo_id}/chat", dependencies=[Depends(rate_limit)], tags=["qa"])
def repo_chat(repo_id: RepoId, request: ChatRequest, services: Services) -> dict[str, Any]:
    try:
        return services.chat.answer(repo_id, request.question, [turn.model_dump() for turn in request.history], debug=request.debug)
    except FileNotFoundError as exc:
        raise _not_analyzed(repo_id, exc) from exc


@router.post("/repo/{repo_id}/search", dependencies=[Depends(rate_limit)], tags=["qa"])
def repo_search(repo_id: RepoId, request: SearchRequest, services: Services) -> dict[str, Any]:
    try:
        context = services.chat.retrieve(repo_id, request.query, top_k=request.k)
    except FileNotFoundError as exc:
        raise _not_analyzed(repo_id, exc) from exc
    return {
        "intents": context["intents"],
        "results": [{**ref, "channels": trace["channels"], "score": trace["score"]} for ref, trace in zip(context["source_references"], context["retrieval_trace"], strict=True)],
        "graph_paths": context["graph_paths"],
        "snapshot": context.get("snapshot"),
        "evidence": context.get("evidence"),
        "coverage": context.get("coverage"),
    }


@router.post("/repo/{repo_id}/investigate", dependencies=[Depends(rate_limit)], tags=["qa"])
def repo_investigate(repo_id: RepoId, request: InvestigateRequest, services: Services) -> dict[str, Any]:
    """Repo Detective: multi-step, tool-using investigation with verified evidence."""
    try:
        return services.detective.investigate(repo_id, request.question, request.max_steps)
    except FileNotFoundError as exc:
        raise _not_analyzed(repo_id, exc) from exc


@router.delete("/repo/{repo_id}", tags=["analysis"])
def delete_repo(repo_id: RepoId, services: Services) -> dict[str, str]:
    services.analyzer.delete(repo_id)
    if hasattr(services.chat, "clear_cache"):
        services.chat.clear_cache()
    return {"status": "deleted", "repo_id": repo_id}
