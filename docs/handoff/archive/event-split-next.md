# handoff — 이슈 #31: 오늘 제어 이벤트 실행/예정 구분 표시

- **이슈**: #31 · **사이클**: Normal
- **워크트리**: `../smartfarm_ai-events` · 브랜치 `app/31-event-split` (최신 main=PR #30 포함으로 리베이스 완료)
- **대상**: `app/views/monitor.py`의 오늘 운영 탭 표시부만 — **시뮬 로직(`src/control/**`) 무변경**

## 현상 (사용자 지적)
00시대 접속에도 "오늘 제어 이벤트" 표에 0~23시 전체 ON/OFF가 실행된 로그처럼 표시. 미래 시간대는 예보 기반 시뮬 결과인데 구분이 없어 모순. 긴급/이상 피드도 동일.

## 구현 범위 (표시 로직만)
1. `render_live_tab()`에서 timeline 이벤트를 현재 시각(now_hour, 기존 계산 재사용) 기준 분리:
   - **실행된 제어**(hour ≤ now_hour): 기존 표 그대로 — 섹션 제목 "🧾 오늘 제어 이벤트(실행)"
   - **예정된 제어**(hour > now_hour): 별도 표(또는 같은 표에 구분 컬럼) — 제목 "🔮 예정된 제어(예보 기반)" + 캡션 "기상청 예보 기반 예상 — 매시 갱신되며 실제 실행과 다를 수 있어요"
   - 실행 이벤트가 0건이면 "아직 실행된 제어가 없어요"(빈 표 대신 안내), 예정 0건이면 섹션 생략
2. 긴급/이상 피드도 동일 분리: 현재 이하=기존 alert_box(경고), 미래=라벨 앞에 "🔮 사전 경보(예보)" 접두 + level은 "주의" 톤으로 낮춰 표시(발송 로직 아님, 표시만)
3. 디스코드·run_notify·simulate_control 등 로직 일절 무변경
4. 테스트: AppTest 또는 표시 분리용 순수 헬퍼(예: `_split_events(timeline, now_hour) -> (done, planned)`)를 만들어 단위 테스트 — 경계(hour==now_hour는 실행 쪽), 빈 리스트 처리

## 완료 기준
- `OMP_NUM_THREADS=1 /Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai/.venv/bin/pytest -m "not integration"` 전체 PASS
- headless /monitor 200
- 단계별 커밋(`이슈 #31`), 완료 보고 후 대기 (push·PR 금지)
