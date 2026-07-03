# handoff — 이슈 #29: KMA stale-while-error 폴백 + 재시도 보강

- **이슈**: #29 · **사이클**: Normal
- **워크트리**: `../smartfarm_ai-weather` · 브랜치 `llm/29-kma-stale-fallback`
- **대상**: `src/llm/weather.py`(주) + 사용처 UI 캡션(`app/views/monitor.py` 최소 수정)

## 현상·원인 (서버 실재현 완료)
OCI에서 KMA API가 간헐적으로 timeout(10s) 초과 → 재시도 1회까지 연속 실패하면 unavailable + `_FAIL_TTL=60s` 캐시 → 관제 [오늘 운영] 탭에 "외기 조회 실패". 직전 성공 데이터가 캐시에 있는데도 버려짐.

## 구현 범위
1. **stale-while-error** (`src/llm/weather.py`):
   - 성공 캐시(`_CACHE`)를 TTL 만료 후에도 **삭제하지 않고 보관**. `_request` 실패 시 해당 endpoint의 만료된 성공 캐시가 있으면 그것을 반환하되 결과 dict에 `"stale": True` 추가(정상 신선 데이터는 stale 키 없음 또는 False).
   - stale 데이터의 최대 허용 나이 상한(예: 6시간 — 상수 `_STALE_MAX_AGE`)을 두고, 그보다 오래면 기존대로 unavailable.
   - `clear_cache()`는 지금처럼 전부 비움(강제 재조회 의미 유지).
2. **재시도 보강**: `_request` attempt 2→3회(백오프 1.5s, 3s). 4xx 즉시 포기(기존 유지, 429 예외도 기존 유지). `_TIMEOUT=10` 유지.
3. **UI 캡션**: `app/views/monitor.py` 오늘 운영 탭·시뮬 탭에서 `get_current()`/`get_forecast_3d()` 결과에 `stale`이 True면 "⏳ 기상청 갱신 지연 — 직전 조회 데이터 표시 중" 캡션 1줄(unavailable 분기와 별개, 데이터는 정상 표시). live.today_outdoor는 수정 불필요(stale이어도 동일 dict 구조).
4. **테스트** (`tests/test_weather.py`에 추가): ①신선 캐시 정상 ②TTL 만료+요청 실패 → stale 반환+플래그 ③stale 상한 초과 → unavailable ④성공 캐시 전무+실패 → 기존 unavailable ⑤재시도 3회 횟수 검증(requests mock) ⑥clear_cache 후 stale도 사라짐

## 금지·경계
- `src/llm/prescribe.py`·`pipeline.py` 수정 금지(weather 소비부는 시그니처 무변경이라 영향 없어야 함) · `src/control/**` 로직 수정 금지
- push·PR·머지·STATUS.md 수정 금지

## 완료 기준
- `OMP_NUM_THREADS=1 /Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai/.venv/bin/pytest -m "not integration"` 전체 PASS
- headless /monitor 200
- 단계별 커밋(`이슈 #29`), 완료 보고 후 대기
