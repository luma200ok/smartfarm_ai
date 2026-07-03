# handoff — 이슈 #23: 관제 "오늘 운영 모드" + 상주 디스코드 알림

- **이슈**: #23 · **사이클**: Normal
- **워크트리**: `../smartfarm_ai-live` · 브랜치 `app/23-live-control`

## 목표 동작 (사용자 확정 스펙)
오늘(실제 날짜) 기준으로 돌아가는 농장: KMA 외기(실황+시간별 예보) → 기대값 모델로 **오늘 시간대별 내부 온·습도 예측(제어 전 기준선)** → 설정 밴드(파일 영속, 기존 `setpoints.load()`) 초과 시간대에 장치 자동 ON → **장치 효과가 반영된 "제어 후 내부온도"가 밴드 안으로 조정**되는 걸 보여준다. 장치 전환·이상·긴급은 디스코드 발송. 페이지를 안 열어도 서버 타이머(1시간)로 알림.

## 구현 범위

### 1. `src/control/live.py` (신규 — 규칙 기반, LLM 금지)
- `today_outdoor() -> list[dict] | None`: `weather.get_current()`(현재 실황) + `weather.get_forecast_3d()["hourly"]`(오늘 날짜분) → 오늘 시간대별 외기 `{hour, temp, humidity}`. 현재 시각 항목은 실황으로 대체. KMA unavailable이면 None.
- `indoor_baseline(outdoor, date) -> list[dict]`: 시간별 외기 온도를 `expect.predict()`(모델 입력: `온도외부_평균`·`일사량_평균`·doy) 에 넣어 시간별 내부 기준선 추정. 일사량은 시간별 데이터가 없으므로 **주간(07~18시)=학습 데이터 계절 평균 근사, 야간=0**으로 넣고 함수 docstring에 근사임을 명시. 내부 습도 기준선은 외기 습도 + 보정(온실 보습, 상수)으로 단순 근사 — 상수로 분리.
- `simulate_control(baseline, setpoints, states) -> timeline`: 시간 순으로 기존 `controller.decide()` 재사용 → ON 장치의 `effects` 델타를 다음 시간 내부온도에 반영(서모스탯: 밴드 복귀 시 히스테리시스로 OFF). 반환: `[{hour, out_temp, base_temp, ctrl_temp, base_hum, ctrl_hum, devices_on:[...], events:[ControlLog...]}]`. 장치 효과 상수는 `effects.py` 재사용(일 단위 델타 → 시간 단위로 나눠 적용, 환산 상수 분리).
- `emergency_hours(timeline, setpoints) -> list`: 풀가동에도 밴드 밖인 시간대 → 긴급.
- **상태 파일 dedup**: `data/control_live_state.json`(gitignore 추가) — 마지막 발송 장치 상태·긴급 키·날짜 저장. 세션이 아니라 파일 기반(타이머 프로세스와 앱이 공유). `setpoints.py`의 원자적 쓰기 패턴 재사용.
- `run_notify() -> int`: 타이머 진입점 — 현재 시각 판정 → 직전 상태 파일과 비교해 **전환분만** 디스코드 발송(`notify.send_discord`, embed 자체 구현): ①장치 ON/OFF 전환 ②이상(KMA 장애 연속 2회↑, 기준선이 물리적 이상치) ③긴급. 상태 파일 갱신.
- CLI: `python -m control.live --notify` (src 경로 기준, 기존 `src/llm/monitor.py`의 argparse 패턴 참고).

### 2. `app/views/monitor.py` 재편
- `st.tabs(["🟢 오늘 운영", "🧪 시뮬레이션"])`:
  - **[오늘 운영]**(기본): 현재 상태 카드(외기 실황·예측 내부·제어 후 내부·작동 중 장치) → 오늘 0~24시 차트(외기·제어 전 기준선·**제어 후 라인**·밴드 영역, 장치 작동 구간 표시 — altair 또는 st.line_chart+영역, 기존 render_trend 패턴 참고) → 오늘 제어 이벤트 로그 테이블 → 긴급/이상 alert_box. KMA unavailable이면 `unavailable()` 표시하고 시뮬 탭 안내.
  - **[시뮬레이션]**: 기존 리플레이 UI(연도·시나리오·다음날·슬라이더·장치카드·로그·긴급 피드) **그대로 이동**(로직 변경 금지 — 회귀 방지).
- 설정 밴드 슬라이더(파일 영속)는 탭 밖 공통 영역 — 양 탭이 같은 setpoints 사용.
- 디스코드 발송 토글도 공통 유지.

### 3. `deploy/` 유닛 파일 (설치는 메타가 머지 후 수행 — 서브는 파일 작성만)
- `deploy/smartfarm-control.service`(Type=oneshot, `ExecStart=/opt/smartfarm_ai/.venv/bin/python -m control.live --notify`, WorkingDirectory=/opt/smartfarm_ai, Environment PYTHONPATH=/opt/smartfarm_ai/src) + `deploy/smartfarm-control.timer`(OnCalendar=hourly, Persistent=true). 기존 유닛명과 충돌 금지(`smartfarm.service`=Java, `smartfarm-ai.service`=Streamlit — 반드시 별개 이름 유지).

### 4. 테스트 (`tests/test_control_live.py` 신규)
- KMA·expect는 mock: 외기 시퀀스 → 기준선 → simulate_control에서 ①밴드 초과 시 장치 ON·ctrl_temp가 base 대비 밴드 방향 조정 ②밴드 내 구간 장치 OFF ③풀가동에도 못 잡으면 emergency ④히스테리시스 채터링 없음
- run_notify: 상태 파일 dedup(같은 상태 재실행 시 발송 0건, 전환 시만 발송 — notify mock) · 날짜 바뀌면 상태 리셋
- KMA unavailable 시 None graceful
- 기존 테스트 무회귀

## 금지·경계
- `src/llm/**` 수정 금지(notify import만) · 기존 시뮬레이션 로직(controller·effects·virtual_sensor) 동작 변경 금지(재사용만)
- push·PR·머지·STATUS.md 수정 금지

## 완료 기준
- `OMP_NUM_THREADS=1 /Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai/.venv/bin/pytest -m "not integration"` 전체 PASS
- headless streamlit /monitor 200 (양 탭)
- `OMP_NUM_THREADS=1 python -m control.live --notify --dry-run`(발송 대신 stdout) 로컬 1회 실행 확인 — dry-run 플래그도 구현
- 단계별 커밋(`이슈 #23` 표기), 완료 보고 후 대기
