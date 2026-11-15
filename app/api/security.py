from __future__ import annotations

import hashlib
import hmac
import ipaddress
import logging
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request

from app.api.services import ServiceContainer, get_container

logger = logging.getLogger(__name__)


def require_api_key(request: Request, services: ServiceContainer = Depends(get_container)) -> None:
    """If REPOMIND_API_KEY is set, every API call must send it as X-API-Key."""
    expected = services.settings.api_key
    if not expected:
        return
    provided = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


class RateLimiter:
    """Sliding-window limiter per client for expensive endpoints (in-process)."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_prune = time.monotonic()

    def check(self, key: str, limit: int, window: float = 60.0) -> bool:
        if limit <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            if now - self._last_prune > window:
                self._prune(now, window)
            hits = self._hits[key]
            while hits and now - hits[0] > window:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True

    def _prune(self, now: float, window: float) -> None:
        """Drop clients whose newest hit is outside the window so `_hits` stays bounded."""
        stale = [key for key, hits in self._hits.items() if not hits or now - hits[-1] > window]
        for key in stale:
            del self._hits[key]
        self._last_prune = now


limiter = RateLimiter()


def _api_key_is_valid(request: Request, expected: str | None) -> bool:
    provided = request.headers.get("x-api-key", "")
    return bool(expected and provided) and hmac.compare_digest(provided.encode(), expected.encode())


def _in_networks(address: str, networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network]) -> bool:
    try:
        ip = ipaddress.ip_address(address.strip())
    except ValueError:
        return False
    return any(ip in network for network in networks)


def client_ip(request: Request, trusted_proxies: tuple[str, ...] = ()) -> str:
    """The peer address, or the X-Forwarded-For client when the peer is a configured trusted proxy."""
    peer = request.client.host if request.client else "unknown"
    if not trusted_proxies:
        return peer
    networks = []
    for entry in trusted_proxies:
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            logger.warning("Ignoring invalid TRUSTED_PROXIES entry %r", entry)
    if not _in_networks(peer, networks):
        return peer
    chain = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",") if part.strip()]
    # Walk right to left: the first hop not operated by a trusted proxy is the real client.
    for address in reversed(chain):
        if not _in_networks(address, networks):
            return address
    return chain[0] if chain else peer


def rate_limit(request: Request, services: ServiceContainer = Depends(get_container)) -> None:
    settings = services.settings
    if _api_key_is_valid(request, settings.api_key):
        # Only a key the server accepts earns its own bucket; never key on raw client-chosen headers.
        client = "key:" + hashlib.sha256(request.headers["x-api-key"].encode()).hexdigest()[:16]
    else:
        client = "ip:" + client_ip(request, settings.trusted_proxies)
    if not limiter.check(f"{client}:{request.url.path.split('/')[-1]}", settings.rate_limit_per_minute):
        raise HTTPException(status_code=429, detail="Rate limit exceeded; try again in a minute.")
