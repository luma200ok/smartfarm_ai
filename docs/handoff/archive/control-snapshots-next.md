# handoff: 관제 시간별 스냅샷 누적 (이슈 #40)

- **이슈**: #40 — feat: 관제 시간별 스냅샷 누적 — 오늘 차트 과거 구간 보존(과거=기록·미래=예보 합성)
- **사이클**: Normal
- **워크트리**: `../smartfarm_ai-control-snapshots` · 브랜치 `control/40-hourly-snapshots`

## 문제 배경
KMA 단기예보는 최신 발표분 이후 시간대만 제공 → [오늘 운영] 차트(`app/views/monitor.py::render_live_tab`)가 저녁이 될수록 과거 시간대를 잃음. 시간별 이력 저장이 없음(상태 파일 `data/control_live_state.json`에는 `last_ctrl` 1개뿐).

## 구현 범위

### 1. `src/control/state_io.py` (신규)
- `save_json_atomic(path: Path, data: dict) -> None` — tmp 파일 → `os.replace`, 실패 시 예외 전파 없음(기존 `live._save_state`/`setpoints.save` 패턴 그대로).
- `load_json(path: Path) -> dict` — 없거나 손상 시 `{}`.
- `src/control/live.py::_load_state/_save_state`와 `src/control/setpoints.py::save/load`가 이 유틸을 쓰도록 교체 — **공개 함수 시그니처는 불변**.

### 2. `src/control/live.py` — 상태 스키마 v2 + 스냅샷 API
상태 파일에 `version: 2`, `snapshots` 추가(기존 키 유지, v1 파일은 `.get("snapshots", {})`로 자연 흡수):
```json
"snapshots": {"14": {"out_temp": 26.0, "out_hum": 78.0, "base_temp": 26.7, "base_hum": 88.0,
  "ctrl_temp": 22.4, "ctrl_hum": 80.1, "devices_on": ["dehumidifier"],
  "events": [{"device": "dehumidifier", "action": "ON", "reason": "…"}],
  "source": "sim", "recorded_at": "2026-07-05T14:00:12"}}
```
- `record_snapshot(state: dict, item: dict, source: str = "sim") -> bool` — timeline 행(item)을 `snapshots[str(hour)]`에 기록. **first-write-wins**(이미 있으면 덮지 않고 False). events는 ControlLog → dict 직렬화 헬퍼(`_serialize_events`).
- `load_today_snapshots(today: date) -> dict` — 상태 파일 date가 today와 다르면 빈 dict. 공개 API(뷰가 내부 포맷에 결합되지 않게 — `load_last_ctrl` 패턴).
- `archive_snapshots(prev_state: dict) -> None` — `data/control_history.json`에 `{prev_date: snapshots}` 병합, **최근 30일만 보존**. state_io 유틸 사용.

### 3. `simulate_control(..., seed_ctrl: "tuple[float|None, float|None] | None" = None)`
- 미래 합성 전용 명시 시드 경로 — 주어지면 첫 항목 ctrl 시작값으로 사용(기존 `initial_ctrl` 어제-전용 검증 로직·주석 **무수정**, seed_ctrl이 우선). 기본 None → 기존 동작 완전 무회귀.
- **run_notify는 seed_ctrl을 사용하지 않는다** — 같은 날 재시딩 드리프트 방지 불변식(L204-214 주석) 유지.

### 4. `assemble_today_timeline(outdoor, setpoints, states, today, now_hour) -> list[dict]`
- 내부에서 `load_today_snapshots(today)` 로드.
- 과거(hour < now_hour): 스냅샷 기록을 timeline 행 포맷({hour, out_temp, base_temp, ctrl_temp, out_hum, base_hum, ctrl_hum, devices_on, events})으로 복원해 사용. events는 dict 그대로 두되 뷰가 소비 가능해야 함(아래 5 참고). 기록 없는 과거 시간: outdoor(예보)에 남아있으면 시뮬값 폴백, 아니면 결측 생략.
- 현재·미래(hour >= now_hour): 해당 구간 outdoor로 `indoor_baseline`+`simulate_control(seed_ctrl=마지막 스냅샷의 (ctrl_temp, ctrl_hum))`. 스냅샷이 하나도 없으면 기존 경로(load_last_ctrl 시드 + fallback_clamp=True)와 동일 거동.
- `outdoor=None`(KMA 실패)이면 과거 스냅샷 행만 반환(빈 리스트 가능).

### 5. `run_notify()` 수정 (판정 로직 무변경!)
- 정상 분기: 판정 후 `cur_item`을 `record_snapshot`으로 기록하고 new_state에 snapshots 포함(같은 날이면 prev 것을 이어받아 누적, 날짜 바뀌면 `archive_snapshots(prev_state)` 호출 후 새로 시작).
- KMA 실패 분기(L420-431): **snapshots도 보존**하도록 new_state에 포함(현재 코드는 유실).
- 장치 전환·긴급 dedup·last_ctrl 로직은 그대로.

### 6. `app/views/monitor.py::render_live_tab`
- timeline 구성을 `assemble_today_timeline(outdoor, setpoints, sim_states, today, now_hour)`로 교체(`_cached_today_outdoor()` 60s 캐시는 유지, outdoor 주입).
- events 소비부(L557-560): 스냅샷 복원 행은 events가 dict 리스트, 시뮬 행은 ControlLog — 두 형태 모두 처리(정규화 헬퍼 하나로).
- 앱 기록: 현재 시각 스냅샷이 없고 **전 장치가 auto일 때만** timeline의 현재 행을 `record_snapshot`으로 기록 후 저장(수동 조작 세션은 기록 생략 — 영구 기록 오염 방지).
- KMA 실패(outdoor None): 오늘 스냅샷이 있으면 과거 기록만으로 차트·표 부분 렌더 + 캡션 "KMA 예보 조회 실패 — 과거 기록만 표시 중". 스냅샷도 없으면 기존 unavailable 안내 유지.

### 7. gitignore
- `data/control_history.json`이 커밋되지 않는지 확인(기존 `data/` 정책 확인, 필요 시 .gitignore 추가).

## 테스트 (기존 픽스처 재사용: tests/test_control_live.py의 `_isolated_state`·`_isolated_setpoints`·`_forecast`·`_patch_notify`·`_patch_expect_model`)
- record_snapshot: first-write-wins(재기록 시 False·값 불변), events 직렬화.
- load_today_snapshots: 날짜 불일치 → 빈 dict, v1 파일(snapshots 없음) 하위호환.
- simulate_control seed_ctrl: 시드 적용·initial_ctrl보다 우선·None이면 기존 동작(기존 테스트 무회귀).
- assemble_today_timeline: 과거=기록·미래=시뮬 합성, 경계 연속성(마지막 스냅샷 ctrl → 미래 첫 행 시드), 기록 없는 과거 폴백/결측 생략, outdoor=None 부분 반환.
- run_notify: 스냅샷 기록·누적, KMA 실패 분기 snapshots 보존, 날짜 롤오버 시 archive + 30일 프룬. 기존 dedup·전환·긴급 테스트 전부 무회귀.
- history 아카이브: 이관 형식·30일 보존.
- (가능하면) test_app_monitor.py: KMA 실패+스냅샷 존재 시 부분 렌더.

## 완료 기준
- `pytest -m "not integration"` 전체 PASS
- 단계별 커밋(스키마/유틸 → 합성 → 뷰 → 테스트 등), push 금지
- 완료 후 A(메인)에 보고 후 대기 — self-review·PR·머지·STATUS.md 수정 금지
