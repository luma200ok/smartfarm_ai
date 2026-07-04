"""
NCPMS(국가농작물병해충관리시스템) OpenAPI 로더 — data/nongsaro/{disease}.md 재생성.

이슈 #42: 수기 작성된 병해 코퍼스(leaf_mold·late_blight)를 NCPMS 공공 API 실데이터로 교체한다.
하위 파이프라인(corpus.load_chunks → store/pg_store → retrieve)은 무변경 — frontmatter
스키마(title/disease/source/source_name)만 기존과 동일하게 맞추면 그대로 동작한다.

확정 API 스펙(실호출 검증 완료):
  - 엔드포인트: http://ncpms.rda.go.kr/npmsAPI/service
  - 공통 파라미터: apiKey(env NCPMS_API_KEY), serviceType=AA001
  - 상세조회: serviceCode=SVC05 + sickKey={고정값} → 태그 developmentCondition(발생환경)·
    symptoms(증상)·preventionMethod(방제)
  - 응답은 XML, 본문에 <br/> 다수 포함 → 개행 정리 필요.

대상(확정 sickKey, 검색 모호성 때문에 하드코딩 — 검색 API는 참고용으로만 남김):
  - leaf_mold(잎곰팡이병/Leaf mold): D00001533
  - late_blight(잎마름역병/Late blight): D00001550
  tylcv·tomato_general 은 NCPMS 상세가 비어있거나 API 부적합 → 이 로더 대상에서 제외(수기 유지).

라이선스: NCPMS 자료는 공공데이터포털(KOGL 제2유형 — 출처표시, 비상업적 이용). 본 레포는
포트폴리오(비상업) 용도이며 source_name 에 출처를 명시한다.

의존성 0 — stdlib urllib + xml.etree 만 사용(requests 등 추가 안 함).

실행: OMP_NUM_THREADS=1 python -m llm.rag.nongsaro_loader [--only leaf_mold] [--dry-run]
"""
from __future__ import annotations

import argparse
import html
import logging
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = ROOT / "data" / "nongsaro"
CACHE_DIR = CORPUS_DIR / ".api_cache"   # 원시 XML 캐시(재현·API 부하 절감, gitignore 대상)

NCPMS_BASE = os.getenv("NCPMS_BASE", "http://ncpms.rda.go.kr/npmsAPI/service")
NCPMS_SVC_SEARCH = "SVC01"   # 검색(참고용 — 결과 모호해 build()에서는 쓰지 않음)
NCPMS_SVC_DETAIL = "SVC05"   # 확정: 병해충 상세조회

SOURCE_NAME = "국가농작물병해충관리시스템(NCPMS·농촌진흥청)"
# 실제 열리는 공공데이터포털 오픈API 소개 페이지(200 확인) — NCPMS API 콘솔 자체는
# 공개 브라우저 접근용 상세페이지 URL이 별도로 없어(로그인·세션 기반) 이 페이지를 출처로 둔다.
SOURCE_URL = "https://www.data.go.kr/data/15002034/openapi.do"


@dataclass(frozen=True)
class DiseaseSpec:
    slug: str
    title: str
    sick_key: str
    source_kind: str = "ncpms"


SPECS: list[DiseaseSpec] = [
    DiseaseSpec(
        slug="leaf_mold",
        title="토마토 잎곰팡이병(Leaf mold) 발생환경·증상·방제",
        sick_key="D00001533",
    ),
    DiseaseSpec(
        slug="late_blight",
        title="토마토 잎마름역병(Late blight) 발생환경·증상·방제",
        sick_key="D00001550",
    ),
]

_SPEC_BY_SLUG = {s.slug: s for s in SPECS}


def _api_key() -> str:
    key = os.getenv("NCPMS_API_KEY")
    if not key:
        raise RuntimeError("NCPMS_API_KEY 미설정 — .env 에 서비스키를 넣어라(하드코딩 금지)")
    return key


def _http_get(params: dict) -> str:
    url = f"{NCPMS_BASE}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:  # noqa: S310 (신뢰된 공공 API)
            return r.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise RuntimeError(f"NCPMS API 호출 실패: {e}") from e


def _clean(text: str | None) -> str:
    """<br/>·<br> → 개행, HTML 엔티티 해제, 앞뒤 공백/중복 개행 정리."""
    if not text:
        return ""
    t = html.unescape(text)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.IGNORECASE)
    lines = [line.strip() for line in t.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _fetch_detail_xml(spec: DiseaseSpec, use_cache: bool = True) -> str:
    """SVC05 상세조회 XML — 캐시 있으면 재사용(재현·API 부하 절감), 없으면 호출 후 저장."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{spec.slug}.xml"
    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    raw = _http_get({
        "apiKey": _api_key(),
        "serviceType": "AA001",
        "serviceCode": NCPMS_SVC_DETAIL,
        "sickKey": spec.sick_key,
    })
    cache_path.write_text(raw, encoding="utf-8")
    return raw


def _parse_sections(xml_text: str) -> dict[str, str]:
    root = ET.fromstring(xml_text)

    def t(tag: str) -> str:
        return root.findtext(f".//{tag}") or ""

    return {
        "발생환경": _clean(t("developmentCondition")),
        "증상": _clean(t("symptoms")),
        "방제": _clean(t("preventionMethod")),
    }


def _render_md(spec: DiseaseSpec, sections: dict[str, str]) -> str:
    fm = "\n".join([
        "---",
        f"title: {spec.title}",
        f"disease: {spec.slug}",
        f"source: {SOURCE_URL}",
        f"source_name: {SOURCE_NAME}",
        "---",
    ])
    body = "\n\n".join(
        sections[k] for k in ("발생환경", "증상", "방제") if sections.get(k)
    )
    return f"{fm}\n\n{body}\n"


def build(only: str | None = None, dry_run: bool = False, use_cache: bool = False) -> int:
    """NCPMS API 호출 → data/nongsaro/{slug}.md 재생성. use_cache=True면 캐시 XML 재사용."""
    specs = [s for s in SPECS if only is None or s.slug == only]
    if only is not None and not specs:
        raise RuntimeError(f"알 수 없는 slug: {only} (가능: {', '.join(_SPEC_BY_SLUG)})")
    written = 0
    for spec in specs:
        xml_text = _fetch_detail_xml(spec, use_cache=use_cache)
        sections = _parse_sections(xml_text)
        missing = [k for k, v in sections.items() if not v]
        if missing:
            raise RuntimeError(f"{spec.slug}: NCPMS 응답에 빈 필드 {missing} — 재생성 중단")
        md = _render_md(spec, sections)
        out = CORPUS_DIR / f"{spec.slug}.md"
        if dry_run:
            print(f"--- {out.name} (dry-run) ---\n{md}")
        else:
            out.write_text(md, encoding="utf-8")
            _log.info("재생성: %s", out.name)
        written += 1
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="NCPMS OpenAPI → data/nongsaro/*.md 재생성")
    ap.add_argument("--only", help="특정 disease 슬러그만(예: leaf_mold)")
    ap.add_argument("--dry-run", action="store_true", help="파일 안 쓰고 렌더 결과만 출력")
    ap.add_argument("--use-cache", action="store_true", help="캐시된 원시 XML 재사용(API 재호출 안 함)")
    a = ap.parse_args()
    try:
        n = build(only=a.only, dry_run=a.dry_run, use_cache=a.use_cache)
    except (RuntimeError, ET.ParseError) as e:
        print(f"❌ {e}")
        sys.exit(1)
    print(f"✅ {n}개 문서 처리")
