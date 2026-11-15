from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _secret_env(name: str) -> str | None:
    value = os.getenv(name)
    if not value:
        return None
    stripped = value.strip()
    return stripped or None


def _list_env(name: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in os.getenv(name, "").split(",") if part.strip())


def _default_llm_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER")
    if explicit:
        return explicit
    return "auto"


def _env_field(name: str, default: str):
    return field(default_factory=lambda: _env(name, default))


def _bool_field(name: str, default: bool):
    return field(default_factory=lambda: _bool_env(name, default))


def _int_field(name: str, default: int):
    return field(default_factory=lambda: _int_env(name, default))


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _float_field(name: str, default: float):
    return field(default_factory=lambda: _float_env(name, default))


@dataclass(frozen=True)
class Settings:
    """Runtime settings.

    Values are read from the environment when a Settings object is created (not at
    import time), so tests and alternate deployments can build isolated instances.
    """

    data_dir: Path = field(default_factory=lambda: Path(_env("REPOMIND_DATA_DIR", "data")))

    # Retrieval
    embedding_provider: str = _env_field("EMBEDDING_PROVIDER", "local")
    embedding_model: str = _env_field("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_batch_size: int = _int_field("EMBEDDING_BATCH_SIZE", 32)
    vector_store_provider: str = _env_field("VECTOR_STORE_PROVIDER", "local")
    reranker_provider: str = _env_field("RERANKER_PROVIDER", "local")
    reranker_model: str = _env_field("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # LLM providers (all optional; RepoMind degrades to deterministic answers)
    llm_provider: str = field(default_factory=_default_llm_provider)
    gemini_model: str = _env_field("GEMINI_MODEL", "gemini-flash-latest")
    gemini_api_key: str | None = field(default_factory=lambda: _secret_env("GEMINI_API_KEY"))
    openai_api_key: str | None = field(default_factory=lambda: _secret_env("OPENAI_API_KEY"))
    openai_model: str = _env_field("OPENAI_MODEL", "")
    openai_base_url: str = _env_field("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_timeout_seconds: int = _int_field("LLM_TIMEOUT_SECONDS", 45)
    local_model_mode: bool = _bool_field("LOCAL_MODEL_MODE", True)

    # Persistence
    database_url: str = _env_field("DATABASE_URL", "")
    use_metadata_db: bool = _bool_field("USE_METADATA_DB", True)

    # Ingestion policy
    max_file_size_bytes: int = _int_field("MAX_FILE_SIZE_BYTES", 300_000)
    max_chunks_per_repo: int = _int_field("MAX_CHUNKS_PER_REPO", 5000)
    max_files_per_repo: int = _int_field("MAX_FILES_PER_REPO", 15000)
    max_total_repo_bytes: int = _int_field("MAX_TOTAL_REPO_BYTES", 250_000_000)
    max_repo_size_mb: int = _int_field("MAX_REPO_SIZE_MB", 1024)
    binary_metadata_enabled: bool = _bool_field("BINARY_METADATA_ENABLED", True)
    follow_symlinks: bool = _bool_field("FOLLOW_SYMLINKS", False)
    parser_provider: str = _env_field("PARSER_PROVIDER", "tree_sitter")
    git_history_depth: int = _int_field("GIT_HISTORY_DEPTH", 50)
    clone_timeout_seconds: int = _int_field("CLONE_TIMEOUT_SECONDS", 300)

    # Evidence strength: share of question terms one top chunk must cover, and the cross-encoder
    # relevance that alone counts as strong evidence. Below these, answers abstain instead of guessing.
    evidence_min_coverage: float = _float_field("EVIDENCE_MIN_COVERAGE", 0.6)
    evidence_rerank_threshold: float = _float_field("EVIDENCE_RERANK_THRESHOLD", 0.5)

    # Index one copy of byte-identical files (vendored copies, generated duplicates).
    collapse_duplicate_files: bool = _bool_field("COLLAPSE_DUPLICATE_FILES", True)
    # Metadata DB failure policy: "auto" falls back to JSON only for the implicit local SQLite file
    # and fails fast when DATABASE_URL is set explicitly; "true"/"false" force either behavior.
    metadata_db_fallback: str = _env_field("METADATA_DB_FALLBACK", "auto")

    # Optional network-backed analysis
    osv_scan_enabled: bool = _bool_field("OSV_SCAN_ENABLED", True)
    github_api_checks_enabled: bool = _bool_field("GITHUB_API_CHECKS_ENABLED", True)
    # Optional; raises the GitHub API rate limit for the clone size pre-check. Never logged.
    github_token: str | None = field(default_factory=lambda: _secret_env("GITHUB_TOKEN"), repr=False)

    # API
    analysis_async_enabled: bool = _bool_field("ANALYSIS_ASYNC_ENABLED", True)
    analysis_workers: int = _int_field("ANALYSIS_WORKERS", 2)
    analysis_queue_capacity: int = _int_field("ANALYSIS_QUEUE_CAPACITY", 8)
    api_key: str | None = field(default_factory=lambda: _secret_env("REPOMIND_API_KEY"))
    rate_limit_per_minute: int = _int_field("RATE_LIMIT_PER_MINUTE", 60)
    # Proxy IPs/CIDRs whose X-Forwarded-For header is trusted for client identification. Empty = ignore the header.
    trusted_proxies: tuple[str, ...] = field(default_factory=lambda: _list_env("TRUSTED_PROXIES"))

    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    @property
    def graphs_dir(self) -> Path:
        return self.data_dir / "graphs"

    @property
    def db_dir(self) -> Path:
        return self.data_dir / "db"

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.db_dir / 'repomind.db'}"

    def ensure_dirs(self) -> None:
        for path in (self.repos_dir, self.indexes_dir, self.metadata_dir, self.graphs_dir, self.db_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
