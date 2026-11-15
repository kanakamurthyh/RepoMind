"""Resolve an optional neural model's immutable cached revision without network I/O."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def model_revision(model_name: str, config: Any) -> str:
    revision = getattr(config, "_commit_hash", None)
    if revision:
        return str(revision)
    # Recent Transformers versions do not always keep _commit_hash on config.
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(model_name, "config.json")
        if isinstance(cached, str):
            for parent in Path(cached).parents:
                if parent.parent.name == "snapshots" and len(parent.name) == 40:
                    return parent.name
    except (ImportError, OSError, ValueError) as exc:
        logger.warning("Could not resolve cached revision for %s (%s)", model_name, type(exc).__name__)
    logger.warning("Model %s has no resolved immutable revision", model_name)
    return "unversioned"
