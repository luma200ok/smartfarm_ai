# handoff — 이슈 #21: 설정 밴드 서버 파일 영속화

- **이슈**: #21 · **사이클**: Normal
- **워크트리**: `../smartfarm_ai-setpoints` · 브랜치 `app/21-setpoints-persist`

## 배경
관제 페이지의 설정 밴드(`Setpoints`)가 `st.session_state`(K_SETPOINTS) 세션 단위라 새로고침·재접속 시 기본값으로 초기화. 온실 설비 설정처럼 영속돼야 함(공개 데모라 방문자 공유 전역 설정 — 의도된 동작).

## 구현 범위
1. **`src/control/setpoints.py`에 저장/로드 추가** (또는 같은 모듈 내 함수):
   - `save(sp: Setpoints, path=None)` — `data/control_setpoints.json`에 원자적 쓰기(tmp 파일 → `os.replace`), 디렉터리 없으면 생성
   - `load(path=None) -> Setpoints` — 파일 없음·JSON 손상·키 누락 시 **기본값 폴백**(예외 전파 없음), 값 검증: `low < high`, 온도 0~40·습도 0~100 클램프, deadband는 파일 값 무시하고 기본 유지(슬라이더 노출 안 됨)
   - 기본 경로는 상수 `SETPOINTS_PATH = ROOT/"data/control_setpoints.json"` — `data/`는 gitignore 확인(파일 자체가 커밋되지 않게, 필요 시 `.gitignore`에 항목 추가)
2. **`app/views/monitor.py` 연결**:
   - `_get_setpoints()`: 세션에 없을 때 `Setpoints()` 대신 `load()`로 초기화
   - `render_setpoints()`: 슬라이더 값이 기존 값과 **달라진 경우에만** `save()` 호출(매 리런 디스크 쓰기 금지)
3. **테스트** (`tests/test_control.py`에 추가 또는 `test_setpoints_persist.py`): tmp_path 사용 — 저장→로드 왕복 일치 / 파일 없음·손상 JSON·키 누락 폴백 / 범위 밖 값 클램프·low>high 폴백 / 저장이 원자적 경로로 쓰는지(간단히 결과 파일 존재·내용 검증)

## 금지·경계
- 장치 수동/자동 모드 영속화는 범위 외(세션 유지).
- `src/llm/**`·`app/views/prescribe.py` 수정 금지. push·PR·머지·STATUS.md 수정 금지.

## 완료 기준
- `OMP_NUM_THREADS=1 /Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai/.venv/bin/pytest -m "not integration"` 전체 PASS
- headless streamlit /monitor 200
- 단계별 커밋(`이슈 #21` 표기), 완료 보고 후 대기
