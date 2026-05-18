"""Build LLM context from the knowledge_base directory.

Reads .pdf, .md, and .txt files (recursive). No filename filtering -
take whatever the user dropped in.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from mli_synthetics.errors import KnowledgeBaseError
from mli_synthetics.logging_config import get_logger

logger = get_logger()

_CACHE_FILENAME = "knowledge_context.txt"
_CHARS_PER_TOKEN = 4  # rough estimate


def build_knowledge_context(
    knowledge_dir: Path | None = None,
    max_tokens: int = 30000,
    use_cache: bool = True,
) -> str:
    from mli_synthetics.settings import get_settings

    settings = get_settings()
    if knowledge_dir is None:
        knowledge_dir = settings.knowledge_base_dir

    if not knowledge_dir.exists():
        logger.warning("Knowledge base dir does not exist: {}", knowledge_dir)
        return _fallback_context()

    files = _collect_files(knowledge_dir)
    if not files:
        logger.warning("No knowledge files found; using fallback")
        return _fallback_context()

    # Cache key from file mtimes + sizes
    cache_path = settings.outputs_dir / _CACHE_FILENAME
    cache_key_path = settings.outputs_dir / (_CACHE_FILENAME + ".key")
    current_key = _signature(files, max_tokens)
    if use_cache and cache_path.exists() and cache_key_path.exists():
        try:
            if cache_key_path.read_text(encoding="utf-8").strip() == current_key:
                logger.info("Using cached knowledge context ({} files)", len(files))
                return cache_path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Cache read failed: {}", exc)

    sections: list[tuple[str, str]] = []
    for path in files:
        try:
            content = _extract_text(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read {}: {}", path.name, exc)
            continue
        if content.strip():
            sections.append((path.name, content))

    if not sections:
        logger.warning("All knowledge files failed to read; using fallback")
        return _fallback_context()

    context = _assemble_context(sections, max_tokens)
    try:
        cache_path.write_text(context, encoding="utf-8")
        cache_key_path.write_text(current_key, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Cache write failed: {}", exc)
    logger.info(
        "Built knowledge context: {} sources, ~{} tokens",
        len(sections),
        len(context) // _CHARS_PER_TOKEN,
    )
    return context


# ---------------------------------------------------------------------------
def _collect_files(root: Path) -> list[Path]:
    extensions = {".pdf", ".md", ".txt"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions and p.is_file())


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    return ""


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise KnowledgeBaseError("pypdf is required for PDF parsing") from exc

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            text = ""
        if len(text.strip()) >= 50:
            parts.append(text)
    return "\n\n".join(parts)


def _assemble_context(sections: list[tuple[str, str]], max_tokens: int) -> str:
    max_chars = max_tokens * _CHARS_PER_TOKEN
    header_template = "====== SOURCE: {name} ======\n"
    # Reserve room for headers
    headers_chars = sum(len(header_template.format(name=n)) for n, _ in sections)
    available_for_content = max(1000, max_chars - headers_chars)
    per_source_budget = available_for_content // len(sections)

    pieces: list[str] = []
    for name, content in sections:
        body = content
        if len(body) > per_source_budget:
            body = body[:per_source_budget] + "\n[...truncated...]"
        pieces.append(header_template.format(name=name) + body)
    full = "\n\n".join(pieces)
    if len(full) > max_chars:
        full = full[:max_chars] + "\n[...overall truncation...]"
    return full


def _signature(files: list[Path], max_tokens: int) -> str:
    h = hashlib.sha256()
    h.update(str(max_tokens).encode())
    for p in files:
        try:
            stat = p.stat()
            h.update(p.name.encode())
            h.update(str(stat.st_mtime_ns).encode())
            h.update(str(stat.st_size).encode())
        except OSError:
            continue
    return h.hexdigest()


def _fallback_context() -> str:
    return (
        "====== SOURCE: fallback ======\n"
        "Use general professional knowledge of stage lighting design. "
        "Common fixture categories include moving beams, spots, washes, "
        "LED pars, blinders, strobes, pixel bars, followspots, lasers, "
        "hazers, CO2 jets, and pyro. Match fixture behavior to song "
        "structure (verse/chorus/build/drop) and genre conventions."
    )
