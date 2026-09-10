"""Tests for chunking.py — pure functions, no network or DB."""

from __future__ import annotations

from pathlib import Path

from chunking import chunk_corpus, chunk_markdown

SAMPLE = """# Title

## First Section

Some short text.

## Second Section

More text here.
"""


def test_chunk_markdown_splits_on_headings() -> None:
    chunks = chunk_markdown(SAMPLE, source="sample.md")
    assert [c.heading for c in chunks] == ["First Section", "Second Section"]
    assert all(c.source == "sample.md" for c in chunks)
    assert "Some short text." in chunks[0].text
    assert "More text here." in chunks[1].text


def test_chunk_markdown_drops_document_title() -> None:
    chunks = chunk_markdown(SAMPLE, source="sample.md")
    assert all("# Title" not in c.text for c in chunks)


def test_chunk_markdown_splits_long_section() -> None:
    paragraphs = "\n\n".join(f"Paragraph {i} " + "x" * 100 for i in range(20))
    text = f"# Title\n\n## Big Section\n\n{paragraphs}\n"
    chunks = chunk_markdown(text, source="big.md", max_chars=300)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 + 20 for c in chunks)  # small slack for join overlap
    assert all(c.heading == "Big Section" for c in chunks)


def test_chunk_markdown_empty_document() -> None:
    assert chunk_markdown("# Title\n", source="empty.md") == []


def test_chunk_corpus_reads_all_markdown_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\n## Sec\n\nHello from A.\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n\n## Sec\n\nHello from B.\n", encoding="utf-8")
    (tmp_path / "not_markdown.txt").write_text("ignore me", encoding="utf-8")

    chunks = chunk_corpus(tmp_path)
    sources = {c.source for c in chunks}
    assert sources == {"a.md", "b.md"}
    assert len(chunks) == 2


def test_real_corpus_chunks_cleanly() -> None:
    """Sanity check against the actual shipped corpus, not just synthetic samples."""
    corpus_dir = Path(__file__).resolve().parent.parent / "corpus"
    chunks = chunk_corpus(corpus_dir)
    assert len(chunks) >= 10
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.heading for chunk in chunks)
