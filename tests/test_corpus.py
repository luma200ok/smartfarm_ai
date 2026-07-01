"""src/llm/rag/corpus.py — frontmatter 파싱 + 문단 청킹."""
from llm.rag import corpus

_DOC = """---
title: 테스트 문서
disease: leaf_mold
source: https://example.org/guide
source_name: 테스트출처
---
첫 번째 문단이다.

두 번째 문단이다.
"""


def test_load_chunks_parses_and_splits(tmp_path):
    (tmp_path / "x.md").write_text(_DOC, encoding="utf-8")
    chunks = corpus.load_chunks(tmp_path)
    assert len(chunks) == 2
    assert chunks[0]["title"] == "테스트 문서"
    assert chunks[0]["disease"] == "leaf_mold"
    assert chunks[0]["source"] == "https://example.org/guide"
    assert {c["text"] for c in chunks} == {"첫 번째 문단이다.", "두 번째 문단이다."}


def test_load_chunks_missing_dir_returns_empty(tmp_path):
    assert corpus.load_chunks(tmp_path / "nope") == []


def test_load_chunks_no_frontmatter(tmp_path):
    (tmp_path / "y.md").write_text("그냥 본문 한 줄.", encoding="utf-8")
    chunks = corpus.load_chunks(tmp_path)
    assert len(chunks) == 1
    assert chunks[0]["title"] == "y"          # 파일명 폴백
    assert chunks[0]["text"] == "그냥 본문 한 줄."
