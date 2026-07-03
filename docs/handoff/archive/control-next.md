# handoff — 이슈 #17: 환경 모니터링 → 관제형 대시보드 개편

- **이슈**: #17 (feat: 관제형 대시보드 — 설정 밴드 기반 장치 자동제어(시뮬)·제어 로그·긴급 디스코드 알림)
- **사이클**: Normal
- **워크트리**: `../smartfarm_ai-control` · 브랜치 `app/17-control-dashboard`

## 구현 범위

### 1. 신규 `src/control/` (규칙 기반 — LLM 호출 금지)
- **`setpoints.py`** — `Setpoints` dataclass: 온도 밴드(기본 20.0~25.0℃), 습도 밴드(기본 60.0~85.0%), 히스테리시스 데드밴드(기본 0.5℃ / 2.0%p).
- **`actuators.py`** — 장치 4종 enum/상수: `vent`(환기)·`humidifier`(가습기)·`cooling_fan`(쿨링팬)·`heater`(히터). 상태 컨테이너(ON/OFF, 자동/수동 모드) + `ControlLog` 항목(date, device, action ON/OFF, reason, mode auto/manual) 리스트.
- **`controller.py`** — `decide(reading, setpoints, states) -> list[actions]`:
  - 온도 > 상한 → cooling_fan+vent ON, heater OFF / 온도 < 하한 → heater ON, cooling_fan·vent OFF
  - 습도 > 상한 → vent ON / 습도 < 하한 → humidifier ON
  - **히스테리시스**: ON은 밴드 밖에서만, OFF는 밴드 안쪽 데드밴드 복귀 시(채터링 방지)
  - 수동 모드 장치는 자동 결정에서 제외
  - 충돌 규칙: heater와 cooling_fan 동시 ON 금지(온도 우선순위: 이탈 폭 큰 쪽)
- **효과 피드백** — 장치 ON 상태로 틱 전진 시 `VirtualSensor.inject()`(src/sim/virtual_sensor.py L59, read-time overlay)로 다음 1일치 반영: heater +1.5℃, cooling_fan -1.5℃, vent 온도 -0.5℃·습도 -5.0%p, humidifier +5.0%p. 수치는 `effects.py` 또는 controller 상수로 분리(조정 가능하게). 대상 피처: `온도내부_평균`(+`온도내부_최저` 동일 delta), `습도내부_평균`.
- **긴급 판정** — `emergency(recent_readings, setpoints, states) -> alert|None`: 관련 장치 풀가동인데 **3틱 연속** 밴드 밖 → level "경고", reason "제어 한계 초과 — 설비 점검 필요". `monitor._akey` 스타일 dedup 키(`control_limit:{temp|hum}`).

### 2. 디스코드 알림 (기존 재사용)
- `src/llm/notify.send_discord(embed)` 재사용(L31). embed 빌더는 control 쪽에 작성(monitor._embed 참고, import하지 말고 자체 구현 — private 함수).
- 발송 3종: ①제어 ON/OFF 이벤트(옵션, 앱 토글 기본 OFF) ②기존 경보(monitor.evaluate 경로 그대로) ③긴급(위 emergency, dedup 후 발송).

### 3. `app/views/monitor.py` 전면 교체 (관제 중심)
- 유지: `render_sensor_controls()`(리플레이 컨트롤) 재사용 가능하면 유지, `ui.py` 헬퍼(section·metric_row·alert_box·unavailable·page_header) 사용.
- 신규 구성(위→아래):
  1. 현재 실측 vs **오늘 예측** metric_row — 예측은 `src/llm/expect.expected(reading, date)`(외기 기반 기대값) + KMA `weather.get_current()` 외기 요약(unavailable graceful)
  2. **설정 밴드 슬라이더**(온도·습도) — st.session_state 유지
  3. **장치 카드 4개**: 상태 ON/OFF·자동/수동 토글·수동 ON/OFF 버튼·작동 사유
  4. 온·습도 추이 차트(최근 window) + 밴드 영역 오버레이 + 장치 작동 구간 표시(가능한 수준에서, 과투자 금지)
  5. **제어 로그 테이블**(st.dataframe, 최근순) + 경보/긴급 피드(alert_box)
  6. 디스코드 발송 설정(제어 이벤트 발송 토글, 긴급은 항상)
- **제거**: `render_weather_qa()`·`render_coach()` 호출과 함수(LLM 사용부). 단 **`src/llm/pipeline.py`는 절대 수정 금지**(이슈 #18이 병행 수정 중 — 충돌 방지). `src/llm/prescribe.py`·`app/views/prescribe.py`도 수정 금지.
- 페이지 title/아이콘이 관제 성격에 맞게 필요하면 `app/nav.py`의 해당 st.Page 라벨만 수정 가능(예: "환경 모니터링" → "환경 관제"). url_path 변경 금지.

### 4. 테스트 (`tests/test_control.py` 신규)
- controller: 경계값(상한 초과/하한 미만/밴드 내), 히스테리시스(밴드 복귀 직후 OFF 안 됨·데드밴드 안쪽 복귀 시 OFF), 수동 오버라이드 제외, heater/cooling 동시 ON 금지
- 효과 피드백: ON 후 tick → reading에 delta 반영(inject 기반), OFF면 미반영
- emergency: 3틱 연속+풀가동일 때만 발동, dedup
- 기존 `tests/test_monitor.py`·`test_virtual_sensor.py` 무회귀

## 참조 인터페이스 (탐색 완료 — 그대로 신뢰 가능)
- `VirtualSensor`: `reading()`(ENV_FEATURES dict)·`tick()`·`inject(feature, start, days, delta)`(원본 불변 overlay)·`clear_injections()` — src/sim/virtual_sensor.py
- `monitor.evaluate(reading, active, expect) -> (new_alerts, active)` 순수함수 dedup — src/llm/monitor.py L107
- `expect.expected(reading, date) -> {"평균","최저","resid_sigma"}|None` — src/llm/expect.py L96
- `weather.get_current()/get_forecast_3d()` — 실패 시 `{"unavailable":True,...}` — src/llm/weather.py
- `app/state.get_vsensor(year)` 캐시 헬퍼

## 완료 기준
- `OMP_NUM_THREADS=1 pytest -m "not integration"` 전체 PASS(신규 테스트 포함)
- headless `streamlit run app/streamlit_app.py`로 /monitor 페이지 200 확인
- 단계별 커밋(커밋 메시지 한국어 컨벤션, `이슈 #17` 표기), **push·PR 금지**, 완료 보고 후 대기
