"""One redaction boundary for JSON API output, including legacy cached analyses."""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse

from app.utils.redaction import redact_data


class RedactedJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return super().render(redact_data(content))
