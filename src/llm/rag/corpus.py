"""
농사로/NCPMS 코퍼스 로더 — data/nongsaro/*.md 를 읽어 검색 단위(chunk)로 쪼갠다.

문서 형식: frontmatter(--- title/disease/source/source_name ---) + 본문.
본문은 빈 줄로 구분된 문단이 각각 하나의 chunk 가 된다.
의존성 0 — frontmatter 는 수기 파서로 처리(pyyaml 등 추가 안 함).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = ROOT / "data" / "nongsaro"


def _parse_frontmatter(text: str):
    """'--- k: v ... ---\\n본문' → (meta dict, body). frontmatter 없으면 ({}, 전체)."""
    if text.lstrip().startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            meta = {}
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
            return meta, parts[2].strip()
    return {}, text.strip()


def load_chunks(corpus_dir: Path = CORPUS_DIR) -> list[dict]:
    """코퍼스 전체 → chunk 리스트. 각 chunk = {text, title, source, source_name, disease}."""
    chunks: list[dict] = []
    if not corpus_dir.is_dir():
        return chunks
    for md in sorted(corpus_dir.glob("*.md")):
        meta, body = _parse_frontmatter(md.read_text(encoding="utf-8"))
        for para in re.split(r"\n\s*\n", body):
            para = para.strip()
            if not para or re.fullmatch(r"[-=*_]{3,}", para):   # 빈 문단·수평선(---)은 검색 대상 아님
                continue
            chunks.append({
                "text": para,
                "title": meta.get("title", md.stem),
                "source": meta.get("source", ""),
                "source_name": meta.get("source_name", ""),
                "disease": meta.get("disease", ""),
            })
    return chunks
