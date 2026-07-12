from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.core.chat import ChatService
from app.core.repo_analyzer import RepoAnalyzer
from app.utils.paths import repo_id_from_url

router = APIRouter()
analyzer = RepoAnalyzer()
chat_service = ChatService(analyzer)
analysis_jobs: dict[str, dict[str, Any]] = {}


class AnalyzeRequest(BaseModel):
    repo_url: str = Field(..., description="Public GitHub repository URL")
    force: bool = False


class ChatRequest(BaseModel):
    question: str


def _set_job(repo_id: str, **updates: Any) -> None:
    current = analysis_jobs.setdefault(
        repo_id,
        {
            "repo_id": repo_id,
            "state": "queued",
            "statuses": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    current.update(updates)
    current["updated_at"] = datetime.now(timezone.utc).isoformat()


def _run_analysis_job(repo_url: str, force: bool) -> None:
    repo_id = repo_id_from_url(repo_url)
    _set_job(repo_id, state="running", statuses=["queued", "running"])
    try:
        result = analyzer.analyze(repo_url, force=force)
        _set_job(repo_id, state="completed", statuses=result.statuses, repo_url=result.repo_url)
    except Exception as exc:
        _set_job(repo_id, state="failed", error=str(exc))


@router.post("/analyze_repo")
def analyze_repo(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        result = analyzer.analyze(request.repo_url, force=request.force)
        return {
            "repo_id": result.repo_id,
            "repo_url": result.repo_url,
            "summary": result.summary,
            "weaknesses": [w.to_dict() for w in result.weaknesses],
            "graph": result.graph,
            "statuses": result.statuses,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/analyze_repo_async")
def analyze_repo_async(request: AnalyzeRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    try:
        repo_id = repo_id_from_url(request.repo_url)
        _set_job(repo_id, state="queued", repo_url=request.repo_url, statuses=["queued"])
        background_tasks.add_task(_run_analysis_job, request.repo_url, request.force)
        return {"repo_id": repo_id, "state": "queued", "status_url": f"/analysis/{repo_id}/status"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/analysis/{repo_id}/status")
def analysis_status(repo_id: str) -> dict[str, Any]:
    if repo_id in analysis_jobs:
        return analysis_jobs[repo_id]
    try:
        result = analyzer.load(repo_id)
        return {"repo_id": repo_id, "state": "completed", "repo_url": result.repo_url, "statuses": result.statuses}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/repo/{repo_id}/summary")
def repo_summary(repo_id: str) -> dict[str, Any]:
    try:
        return analyzer.load(repo_id).summary
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/repo/{repo_id}/chat")
def repo_chat(repo_id: str, request: ChatRequest) -> dict[str, Any]:
    try:
        return chat_service.answer(repo_id, request.question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/repo/{repo_id}/weaknesses")
def repo_weaknesses(repo_id: str) -> list[dict[str, Any]]:
    try:
        return [weakness.to_dict() for weakness in analyzer.load(repo_id).weaknesses]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/repo/{repo_id}/graph")
def repo_graph(repo_id: str) -> dict[str, Any]:
    try:
        return analyzer.load(repo_id).graph
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/repo/{repo_id}")
def delete_repo(repo_id: str) -> dict[str, str]:
    analyzer.delete(repo_id)
    return {"status": "deleted", "repo_id": repo_id}
