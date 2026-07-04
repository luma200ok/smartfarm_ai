# handoff: NCPMS OpenAPI 로더 — RAG 코퍼스 실데이터화 (이슈 #42)

- **이슈**: #42 / feat: 농사로·NCPMS OpenAPI 로더 — RAG 코퍼스 실데이터화(병해 2건)
- **사이클**: Normal (외부 API 연동, 서비스 계층 아님)
- **워크트리**: `../smartfarm_ai-ncpms` (생성 완료), 브랜치 `ml/42-ncpms-loader`
- **.env**: 워크트리에 복사됨(`NCPMS_API_KEY` 포함, gitignore). 실호출·재생성에 사용.

## 목표
`src/llm/rag/nongsaro_loader.py` 신규 작성 → NCPMS API로 `data/nongsaro/{disease}.md`를
재생성. 하위 파이프라인(`corpus.load_chunks` → `store`/`pg_store` → `retrieve`)은 **무변경**.
frontmatter 스키마(title/disease/source/source_name)만 맞추면 임베딩·검색·sync 그대로 동작.

## 확정 API 스펙 (실호출 검증 완료 — 그대로 사용)
- 엔드포인트: `http://ncpms.rda.go.kr/npmsAPI/service`
- 공통 파라미터: `apiKey`(env), `serviceType=AA001`
- **검색(sickKey 획득)**: `serviceCode=SVC01` + `cropName=토마토` + `sickNameKor={검색어}`
  → 응답 `.//item` 안에 `sickKey`, `sickNameKor`, `sickNameEng`, `cropCode`.
  ⚠️ 검색어 모호성 있음 → **sickKey 하드코딩 권장**(아래 확정값). 검색은 참고용.
- **상세**: `serviceCode=SVC05` + `sickKey={key}`
  → 태그: `developmentCondition`(발생환경)·`symptoms`(증상)·`preventionMethod`(방제).
  (부가: `infectionRoute` 감염경로, `chemicalPrvnbeMth` 화학방제 — 필요 시 활용)
- 응답은 **XML**. 본문에 `<br/>` 포함 → 개행/공백 정리 필요. stdlib `urllib`+`xml.etree`만(의존성 0).

## 대상 병해 (확정 sickKey)
| slug | sickNameKor | sickNameEng | sickKey | 상태 |
|---|---|---|---|---|
| leaf_mold | 잎곰팡이병 | Leaf mold | **D00001533** | 3필드 충실 → 교체 |
| late_blight | 잎마름역병 | Late blight | **D00001550** | 3필드 충실 → 교체 |

- **tylcv 제외**: NCPMS 바이러스 레코드(D00004252)는 발생환경·증상 비어있음(방제 1문장뿐) → 현행 수기본 유지. 로더 SPECS에 넣지 말 것.
- **tomato_general 제외**: 농사로 API 부적합 → 현행 수기 유지.

## 구현 요구
1. `nongsaro_loader.py`:
   - `DiseaseSpec(slug, title, sick_key, source_kind="ncpms")` — leaf_mold·late_blight 2건만.
   - `_fetch_detail(sick_key)` → SVC05 호출·캐시(`data/nongsaro/.api_cache/{slug}.xml`).
   - `_clean(text)` → `<br/>`·`<br>`를 개행으로, 앞뒤 공백/중복 개행 정리(HTML 엔티티도).
   - `_render_md(spec, sections)` → 기존 frontmatter 스키마와 **완전 일치**(아래 참고). 발생환경/증상/방제를 빈 줄로 구분된 문단으로.
   - `source_name`: `국가농작물병해충관리시스템(NCPMS·농촌진흥청)`.
   - `source`: 공개 접근 가능한 NCPMS 상세 URL을 **실제로 열리는 것으로 확인**해 기입(없으면 base + `?sickKey=` 형태로 두되 유효성 확인). 임의 URL 날조 금지.
   - CLI `--only {slug}`, `--dry-run`, 키 미설정 시 명확한 exit.
   - 라이선스 주석: NCPMS = KOGL 제2유형(출처표시·비상업). 포트폴리오(비상업) 전제.
2. `.env.example`에 `NCPMS_API_KEY=` 항목 추가(더미/빈값, 실키 금지).
3. 로더로 leaf_mold·late_blight `.md` 재생성(실호출 1회) → 내용 육안 확인.
4. **테스트**: 라이브 API 때리지 말 것 — 저장한 샘플 XML(fixture)로 `_clean`·`_render_md`·파서 검증. `OMP_NUM_THREADS=1 pytest` PASS.
5. RAG 스모크: 재생성 후 `retrieve("잎곰팡이 방제", disease="leaf_mold")` 정상 반환 확인.

## frontmatter 스키마 (기존 tylcv.md 참고 — 반드시 동일 형식)
```
---
title: 토마토 잎곰팡이병(Leaf mold) 발생환경·증상·방제
disease: leaf_mold
source: {실제 열리는 NCPMS URL}
source_name: 국가농작물병해충관리시스템(NCPMS·농촌진흥청)
---
{발생환경}

{증상}

{방제}
```

## 완료기준
- leaf_mold·late_blight `.md` 재생성·스키마 일치 · 로더 테스트 PASS · retrieve 스모크 OK.
- 단계별 커밋(로더 / 테스트 / 재생성물+.env.example / 캐시 gitignore).
- `.api_cache/`는 `.gitignore` 추가(원시 XML 커밋 여부는 A가 판단 — 기본 제외).
- push/PR/머지 금지. 완료 후 A(메인)에 보고 후 대기.
