"""Splits corpus markdown files into retrievable chunks.

Pure functions, no network/DB — kept separate from ``ingest.py`` so chunking
logic has its own fast unit tests (``tests/test_chunking.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHUNK_CHARS = 800


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    heading: str


def _split_long_section(text: str, max_chars: int) -> list[str]:
    """Split ``text`` on blank lines, greedily packing paragraphs under ``max_chars``."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = para
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [text]


def chunk_markdown(text: str, *, source: str, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split one markdown document on ``##`` headings, then further on paragraphs if needed.

    The document's ``#`` title (if any) is dropped from chunk text but not
    used as a heading — each chunk is tagged with the nearest ``##`` heading
    above it (or "" if the document has content before its first heading).
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    heading = ""
    body: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if body:
                sections.append((heading, body))
            heading = line.removeprefix("## ").strip()
            body = []
        elif line.startswith("# "):
            continue  # document title, not a chunk heading
        else:
            body.append(line)
    if body:
        sections.append((heading, body))

    chunks: list[Chunk] = []
    for section_heading, section_lines in sections:
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            continue
        for piece in _split_long_section(section_text, max_chars):
            chunks.append(Chunk(text=piece.strip(), source=source, heading=section_heading))
    return chunks


def chunk_corpus(corpus_dir: Path) -> list[Chunk]:
    """Chunk every ``*.md`` file directly under ``corpus_dir``."""
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), source=path.name))
    return chunks
