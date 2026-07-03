# handoff — 이슈 #18: 처방 fast-path(1-call) 전환 + writer 모델 분리

- **이슈**: #18 (feat: 처방 fast-path — 서버 342s→목표 60s)
- **사이클**: Normal
- **워크트리**: `../smartfarm_ai-fastpath`(생성 완료) · 브랜치 `llm/18-prescribe-fastpath`

## 배경
이슈 #15(PR #16) 후에도 OCI(3코어 CPU·7b) 동기 처방 342.6s. 원인: tool 선택 라운드 2회(~3.8k tok 프롬프트) + 7b CPU 추론. 서버 벤치: `exaone3.5:2.4b` 생성 13.0/프롬프트 45.1 tok/s(7b의 ~2.8배), 처방 JSON 품질 우수.

## 구현 범위 (이슈 #18 체크리스트 그대로)
1. **fast-path**: `src/llm/prescribe.py` — 진단→RAG 검색→예보 조회를 **코드가 직접 실행**(tool 라운드 제거), LLM은 최종 구조화 JSON 처방 작성 **1-call**만. 로컬·서버 공통 기본 경로. 기존 환각 방어 3종(신뢰도 톤·게이트 안내·클래스 한정) 지시와 근거출처 코드 주입, JSON 스키마 검증·재시도·폴백은 그대로 유지.
2. **writer 모델 분리**: env `OLLAMA_WRITER_MODEL`(미설정 시 `OLLAMA_MODEL` 폴백) — fast-path 1-call에 사용. keep_alive 30m 유지.
3. **tool calling(agentic) 경로 보존**: 기존 tool 라운드 경로는 CLI·날씨 Q&A(`pipeline.py` weather_qa)에 유지 — 삭제 금지, 데모 가치 보존. 경로 선택은 코드 기본=fast-path, 옵션으로 agentic.
4. **UX**: `app/views/prescribe.py` — st.status 단계 표시를 fast-path 단계(진단→근거 검색→예보→처방 작성)로 갱신 + 처방문 생성 **스트리밍 표시**(Ollama stream, on_progress 콜백 확장 또는 chunk yield).
5. 서버 배포(.env 설정·재시작·실측)는 **메타(A)가 머지 후 수행** — 서브 범위 아님.

## 금지·경계 (병행 이슈 #17과 충돌 방지)
- `app/views/monitor.py`, `src/control/**`(신규 예정), `src/sim/virtual_sensor.py`, `src/llm/monitor.py` **수정 금지**.
- `pipeline.py`는 weather_qa/agentic 경로 유지에 필요한 최소 수정만.
- push·PR·머지·STATUS.md 수정 금지.

## 완료 기준
- `OMP_NUM_THREADS=1 pytest -m "not integration"` 전체 PASS — fast-path 단위 테스트 신규(1-call 프롬프트에 근거·예보 포함, 스키마 폴백, writer env 폴백) + 기존 처방 테스트 무회귀
- 로컬 실측 1회(qwen2.5:14b, 정상 진단 경로) 시간 기록 — 기대 8~10s
- 단계별 커밋(`이슈 #18` 표기), 완료 보고 후 대기
