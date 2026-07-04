"""src/llm/rag/nongsaro_loader.py — NCPMS XML 파싱·정리·렌더 검증.

라이브 API는 호출하지 않는다 — 저장된 샘플 XML(fixture, tests/fixtures/ncpms_leaf_mold.xml)로
_clean/_parse_sections/_render_md 를 검증하고, build() 는 monkeypatch 로 네트워크 호출을 대체한다.
"""
from pathlib import Path

import pytest

from llm.rag import nongsaro_loader as loader

FIXTURES = Path(__file__).parent / "fixtures"
LEAF_MOLD_XML = (FIXTURES / "ncpms_leaf_mold.xml").read_text(encoding="utf-8")

_SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<service>
<developmentCondition>1차 전염원&lt;br/&gt;20～25℃ 온도조건.</developmentCondition>
<symptoms>잎에 발생한다.&lt;br/&gt;흰색 반점이 생긴다.</symptoms>
<preventionMethod>- 병든 잎 제거.&lt;br/&gt;- 통풍 확보.</preventionMethod>
</service>
"""


def test_clean_converts_br_and_unescapes_entities():
    raw = "1차&amp;나중&lt;br/&gt;2차&lt;br&gt;3차"
    cleaned = loader._clean(raw)
    assert cleaned == "1차&나중\n2차\n3차"


def test_clean_strips_blank_lines_and_whitespace():
    raw = "  첫줄  &lt;br/&gt;&lt;br/&gt;  둘째줄  "
    cleaned = loader._clean(raw)
    assert cleaned == "첫줄\n둘째줄"


def test_clean_handles_none_and_empty():
    assert loader._clean(None) == ""
    assert loader._clean("") == ""


def test_parse_sections_from_sample_xml():
    sections = loader._parse_sections(_SAMPLE_XML)
    assert sections["발생환경"] == "1차 전염원\n20～25℃ 온도조건."
    assert sections["증상"] == "잎에 발생한다.\n흰색 반점이 생긴다."
    assert sections["방제"] == "- 병든 잎 제거.\n- 통풍 확보."


def test_parse_sections_from_real_leaf_mold_fixture():
    sections = loader._parse_sections(LEAF_MOLD_XML)
    assert sections["발생환경"]
    assert sections["증상"]
    assert sections["방제"]
    assert "<br" not in sections["발생환경"]
    assert "&lt;" not in sections["발생환경"]


def test_render_md_matches_frontmatter_schema():
    spec = loader.DiseaseSpec(slug="leaf_mold", title="테스트 제목", sick_key="D00001533")
    sections = {"발생환경": "환경설명", "증상": "증상설명", "방제": "방제설명"}
    md = loader._render_md(spec, sections)
    assert md.startswith("---\ntitle: 테스트 제목\ndisease: leaf_mold\n")
    assert f"source: {loader.SOURCE_URL}" in md
    assert f"source_name: {loader.SOURCE_NAME}" in md
    assert md.count("---") == 2
    assert "환경설명\n\n증상설명\n\n방제설명" in md


def test_build_writes_md_using_cached_xml(tmp_path, monkeypatch):
    """build(use_cache=True)는 캐시된 XML만 읽고 네트워크를 호출하지 않는다."""
    corpus_dir = tmp_path / "nongsaro"
    cache_dir = corpus_dir / ".api_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "leaf_mold.xml").write_text(_SAMPLE_XML, encoding="utf-8")

    monkeypatch.setattr(loader, "CORPUS_DIR", corpus_dir)
    monkeypatch.setattr(loader, "CACHE_DIR", cache_dir)

    def _boom(*args, **kwargs):
        raise AssertionError("네트워크 호출 금지 — use_cache=True 는 캐시만 읽어야 함")

    monkeypatch.setattr(loader, "_http_get", _boom)

    written = loader.build(only="leaf_mold", use_cache=True)
    assert written == 1

    out = corpus_dir / "leaf_mold.md"
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "disease: leaf_mold" in text
    assert "잎에 발생한다." in text


def test_build_unknown_slug_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "CORPUS_DIR", tmp_path)
    monkeypatch.setattr(loader, "CACHE_DIR", tmp_path / ".api_cache")
    with pytest.raises(RuntimeError):
        loader.build(only="not_a_real_slug")


def test_build_raises_on_missing_section(tmp_path, monkeypatch):
    """방제/증상/발생환경 중 하나라도 비면 잘못된 근거로 코퍼스를 채우지 않도록 중단."""
    corpus_dir = tmp_path / "nongsaro"
    cache_dir = corpus_dir / ".api_cache"
    cache_dir.mkdir(parents=True)
    empty_xml = "<?xml version='1.0'?><service><symptoms>증상만 있음</symptoms></service>"
    (cache_dir / "late_blight.xml").write_text(empty_xml, encoding="utf-8")

    monkeypatch.setattr(loader, "CORPUS_DIR", corpus_dir)
    monkeypatch.setattr(loader, "CACHE_DIR", cache_dir)

    with pytest.raises(RuntimeError):
        loader.build(only="late_blight", use_cache=True)
