"""Client for a self-hosted MinerU 3 FastAPI service."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx

from .cloud import _extract_archive
from .config import MinerUConfig, MinerUError

_REQUEST_TIMEOUT_SECONDS = 900.0


def _headers(config: MinerUConfig) -> dict[str, str]:
    token = (config.api_token or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def verify_server(config: MinerUConfig) -> str:
    """Verify a MinerU 3 server without submitting a document."""
    base_url = (config.api_base_url or "").strip().rstrip("/")
    if not base_url:
        raise MinerUError("No self-hosted MinerU API base URL configured.")
    try:
        response = httpx.get(f"{base_url}/health", headers=_headers(config), timeout=30.0)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise MinerUError(f"Self-hosted MinerU API request failed: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "healthy":
        raise MinerUError("Self-hosted MinerU API did not report a healthy status.")
    return str(payload.get("version") or "unknown")


def parse_server(
    pdf_path: Path,
    output_base: Path,
    config: MinerUConfig,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """Upload a PDF to ``/file_parse`` and extract its canonical ZIP output."""
    base_url = (config.api_base_url or "").strip().rstrip("/")
    if not base_url:
        raise MinerUError("No self-hosted MinerU API base URL configured.")
    if not pdf_path.is_file():
        raise MinerUError(f"PDF file not found: {pdf_path}")

    if on_progress:
        on_progress(f"Self-hosted MinerU: uploading {pdf_path.name}")
    backend = "pipeline" if config.model_version == "pipeline" else "vlm-engine"
    data = {
        "backend": backend,
        "parse_method": "ocr" if config.is_ocr else "auto",
        "formula_enable": str(config.enable_formula).lower(),
        "table_enable": str(config.enable_table).lower(),
        "return_md": "true",
        "return_content_list": "true",
        "return_images": "true",
        "response_format_zip": "true",
    }
    if config.api_language:
        data["lang_list"] = config.api_language

    try:
        with pdf_path.open("rb") as source:
            response = httpx.post(
                f"{base_url}/file_parse",
                headers=_headers(config),
                data=data,
                files={"files": (pdf_path.name, source, "application/pdf")},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise MinerUError(f"Self-hosted MinerU parsing request failed: {exc}") from exc

    archive = response.content
    if not archive.startswith(b"PK"):
        detail = "unexpected response"
        try:
            detail = str(response.json())
        except ValueError:
            pass
        raise MinerUError(f"Self-hosted MinerU did not return a ZIP result: {detail[:500]}")

    workdir = output_base / pdf_path.stem
    if on_progress:
        on_progress("Self-hosted MinerU: extracting parsed artifacts")
    _extract_archive(archive, workdir)
    return workdir


__all__ = ["parse_server", "verify_server"]
