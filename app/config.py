from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_llm_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER")
    if explicit:
        return explicit
    return "gemini" if os.getenv("GEMINI_API_KEY") else "local"


def _secret_env(name: str) -> str | None:
    value = os.getenv(name)
    if not value:
        return None
    stripped = value.strip()
    return stripped or None


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("REPOMIND_DATA_DIR", "data"))
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    vector_store_provider: str = os.getenv("VECTOR_STORE_PROVIDER", "local")
    llm_provider: str = _default_llm_provider()
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    openai_api_key: str | None = _secret_env("OPENAI_API_KEY")
    gemini_api_key: str | None = _secret_env("GEMINI_API_KEY")
    local_model_mode: bool = _bool_env("LOCAL_MODEL_MODE", True)
    database_url: str = os.getenv("DATABASE_URL", "")
    use_metadata_db: bool = _bool_env("USE_METADATA_DB", True)
    max_file_size_bytes: int = int(os.getenv("MAX_FILE_SIZE_BYTES", "300000"))
    max_chunks_per_repo: int = int(os.getenv("MAX_CHUNKS_PER_REPO", "5000"))
    max_files_per_repo: int = int(os.getenv("MAX_FILES_PER_REPO", "15000"))
    max_total_repo_bytes: int = int(os.getenv("MAX_TOTAL_REPO_BYTES", "250000000"))
    binary_metadata_enabled: bool = _bool_env("BINARY_METADATA_ENABLED", True)
    follow_symlinks: bool = _bool_env("FOLLOW_SYMLINKS", False)
    parser_provider: str = os.getenv("PARSER_PROVIDER", "tree_sitter")
    analysis_async_enabled: bool = _bool_env("ANALYSIS_ASYNC_ENABLED", True)

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
    def db_dir(self) -> Path:
        return self.data_dir / "db"

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.db_dir / 'repomind.db'}"

    def ensure_dirs(self) -> None:
        for path in (self.repos_dir, self.indexes_dir, self.metadata_dir, self.db_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
