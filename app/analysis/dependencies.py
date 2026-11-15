"""Dependency inventory + OSV.dev vulnerability scan with import-level reachability."""

from __future__ import annotations

import json
import logging
import re
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any

import requests

from app.models import FileRecord

logger = logging.getLogger(__name__)
OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{id}"
IMPORT_NAME_OVERRIDES = {
    "pyyaml": "yaml", "beautifulsoup4": "bs4", "pillow": "PIL", "scikit-learn": "sklearn", "python-dotenv": "dotenv",
    "opencv-python": "cv2", "opencv-python-headless": "cv2", "psycopg2-binary": "psycopg2", "psycopg": "psycopg", "pyjwt": "jwt",
    "protobuf": "google", "python-dateutil": "dateutil", "pymongo": "pymongo", "gitpython": "git", "attrs": "attr",
    "python-multipart": "multipart", "google-generativeai": "google", "faiss-cpu": "faiss", "tree-sitter": "tree_sitter",
}


@dataclass(slots=True)
class PinnedPackage:
    name: str
    version: str
    ecosystem: str
    source: str
    line: int | None = None


@dataclass(slots=True)
class Vulnerability:
    package: str
    version: str
    ecosystem: str
    id: str
    source: str
    aliases: list[str] = field(default_factory=list)
    summary: str = ""
    severity: str = "unknown"
    fixed_versions: list[str] = field(default_factory=list)
    reachable: bool = False
    imported_in: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DependencyInventory:
    packages: list[PinnedPackage]
    unpinned: list[dict[str, Any]]
    warnings: list[str]


def import_name(package: str, ecosystem: str) -> str:
    lowered = package.lower()
    if ecosystem == "PyPI":
        return IMPORT_NAME_OVERRIDES.get(lowered, lowered.replace("-", "_"))
    return lowered


class DependencyScanner:
    def __init__(self, enabled: bool = True, timeout: float = 8.0, http_post: Callable[..., Any] | None = None, http_get: Callable[..., Any] | None = None, max_detail_lookups: int = 25) -> None:
        self.enabled = enabled
        self.timeout = timeout
        self.http_post = http_post or requests.post
        self.http_get = http_get or requests.get
        self.max_detail_lookups = max_detail_lookups

    def inventory(self, records: list[FileRecord]) -> tuple[list[PinnedPackage], list[dict[str, Any]]]:
        inventory = self._inventory(records)
        return inventory.packages, inventory.unpinned

    def _inventory(self, records: list[FileRecord]) -> DependencyInventory:
        pinned: list[PinnedPackage] = []
        unpinned: list[dict[str, Any]] = []
        warnings: list[str] = []
        for record in records:
            if record.kind != "text":
                continue
            name = PurePosixPath(record.path).name.lower()
            if name.startswith("requirements") and name.endswith(".txt"):
                for line_number, line in enumerate(record.text.splitlines(), start=1):
                    clean = line.split("#", 1)[0].strip()
                    if not clean or clean.startswith(("-", "git+", "http")):
                        continue
                    match = re.match(r"^([A-Za-z0-9_.\-\[\]]+?)\s*==\s*([\w.+!-]+)", clean)
                    if match:
                        pinned.append(PinnedPackage(re.sub(r"\[.*\]", "", match.group(1)), match.group(2), "PyPI", record.path, line_number))
                    else:
                        package = re.split(r"[<>=~!;\[\s@]", clean, maxsplit=1)[0]
                        if package:
                            unpinned.append({"package": package, "source": record.path, "line": line_number, "spec": clean[:80]})
            elif name == "package-lock.json":
                try:
                    data = json.loads(record.text)
                except json.JSONDecodeError:
                    warnings.append(f"Could not parse {record.path}")
                    logger.warning("Could not parse dependency lockfile %s in %s", record.path, record.repo_id)
                    continue
                if not isinstance(data, dict) or any(not isinstance(data.get(key, {}), dict) for key in ("packages", "dependencies")):
                    warnings.append(f"Invalid lockfile structure in {record.path}")
                    logger.warning("Invalid dependency lockfile structure in %s for %s", record.path, record.repo_id)
                    continue
                for key, value in (data.get("packages") or {}).items():
                    if key.startswith("node_modules/") and isinstance(value, dict) and value.get("version") and not value.get("dev"):
                        pinned.append(PinnedPackage(key.split("node_modules/")[-1], value["version"], "npm", record.path))
                if not data.get("packages"):
                    for key, value in (data.get("dependencies") or {}).items():
                        if isinstance(value, dict) and value.get("version"):
                            pinned.append(PinnedPackage(key, value["version"], "npm", record.path))
            elif name in {"poetry.lock", "cargo.lock", "uv.lock"}:
                ecosystem = "crates.io" if name == "cargo.lock" else "PyPI"
                try:
                    data = tomllib.loads(record.text)
                except tomllib.TOMLDecodeError:
                    warnings.append(f"Could not parse {record.path}")
                    logger.warning("Could not parse dependency lockfile %s in %s", record.path, record.repo_id)
                    continue
                if not isinstance(data.get("package", []), list):
                    warnings.append(f"Invalid lockfile structure in {record.path}")
                    logger.warning("Invalid dependency lockfile structure in %s for %s", record.path, record.repo_id)
                    continue
                for package in data.get("package", []):
                    if isinstance(package, dict) and isinstance(package.get("name"), str) and isinstance(package.get("version"), str):
                        pinned.append(PinnedPackage(package["name"], package["version"], ecosystem, record.path))
            elif name == "go.sum":
                seen = set()
                for match in re.finditer(r"^(\S+)\s+v([\w.+-]+?)(?:/go\.mod)?\s", record.text, re.MULTILINE):
                    key = (match.group(1), match.group(2))
                    if key not in seen:
                        seen.add(key)
                        pinned.append(PinnedPackage(match.group(1), match.group(2), "Go", record.path))
        unique: dict[tuple[str, str, str], PinnedPackage] = {}
        for package in pinned:
            unique.setdefault((package.ecosystem, package.name.lower(), package.version), package)
        return DependencyInventory(list(unique.values()), unpinned, warnings)

    def scan(self, records: list[FileRecord], imported_packages: dict[str, list[str]] | None = None) -> dict[str, Any]:
        inventory = self._inventory(records)
        packages, unpinned = inventory.packages[:1500], inventory.unpinned
        imported_packages = {key.lower(): value for key, value in (imported_packages or {}).items()}
        result: dict[str, Any] = {
            "packages_scanned": 0,
            "inventory_packages": len(inventory.packages),
            "inventory_truncated": len(inventory.packages) > len(packages),
            "inventory_warnings": inventory.warnings,
            "inventory_basis": "declared pinned versions; not a resolved runtime dependency graph",
            "ecosystems": sorted({p.ecosystem for p in packages}),
            "unpinned": unpinned[:200],
            "unpinned_count": len(unpinned),
            "vulnerabilities": [],
            "scan_status": "skipped",
            "note": "",
        }
        if not packages:
            result["note"] = "No pinned dependency versions (lockfile or ==) found; nothing to check against OSV."
            if inventory.warnings:
                result.update(scan_status="unavailable", note="Dependency inventory could not be completely parsed.")
            return result
        if not self.enabled:
            result["note"] = "OSV scanning disabled (OSV_SCAN_ENABLED=false)."
            return result
        try:
            response = self.http_post(
                OSV_BATCH_URL,
                json={"queries": [{"package": {"name": p.name, "ecosystem": p.ecosystem}, "version": p.version} for p in packages]},
                timeout=self.timeout,
            )
            response.raise_for_status()
            batch = response.json().get("results", [])
        except (requests.RequestException, ValueError) as exc:
            logger.warning("OSV scan failed: %s", exc)
            result.update({"scan_status": "offline", "note": "OSV.dev could not be reached; vulnerability data unavailable."})
            return result

        vulns: list[Vulnerability] = []
        for package, entry in zip(packages, batch, strict=False):
            for vuln in (entry or {}).get("vulns", []) or []:
                imports = imported_packages.get(import_name(package.name, package.ecosystem), []) or imported_packages.get(package.name.lower(), [])
                vulns.append(Vulnerability(package.name, package.version, package.ecosystem, vuln.get("id", "?"), package.source, reachable=bool(imports), imported_in=imports[:10]))
        details_cache: dict[str, dict[str, Any]] = {}
        for vuln in vulns[: self.max_detail_lookups]:
            if vuln.id not in details_cache:
                try:
                    detail_response = self.http_get(OSV_VULN_URL.format(id=vuln.id), timeout=self.timeout)
                    details_cache[vuln.id] = detail_response.json() if detail_response.ok else {}
                except (requests.RequestException, ValueError):
                    details_cache[vuln.id] = {}
            self._apply_details(vuln, details_cache[vuln.id])
        vulns.sort(key=lambda v: ({"critical": 0, "high": 1, "medium": 2, "moderate": 2, "low": 3}.get(v.severity, 4), not v.reachable, v.package))
        partial = result["inventory_truncated"] or bool(inventory.warnings) or len(batch) != len(packages)
        result.update({"scan_status": "partial" if partial else "ok", "packages_scanned": min(len(batch), len(packages)),
                       "vulnerabilities": [v.to_dict() for v in vulns[:200]], "vulnerable_packages": len({(v.package, v.version) for v in vulns}),
                       "note": "Inventory or advisory responses are incomplete; unchecked versions remain unknown." if partial else ""})
        return result

    def _apply_details(self, vuln: Vulnerability, detail: dict[str, Any]) -> None:
        if not detail:
            return
        vuln.summary = (detail.get("summary") or detail.get("details") or "")[:300]
        vuln.aliases = list(detail.get("aliases") or [])[:6]
        severity = (detail.get("database_specific") or {}).get("severity")
        if isinstance(severity, str):
            vuln.severity = severity.lower()
        fixed: list[str] = []
        for affected in detail.get("affected") or []:
            if (affected.get("package") or {}).get("name", "").lower() != vuln.package.lower():
                continue
            for version_range in affected.get("ranges") or []:
                fixed.extend(event["fixed"] for event in version_range.get("events", []) if "fixed" in event)
        vuln.fixed_versions = sorted(set(fixed))[:5]
