# handoff — 이슈 #27: 장치 개편(환기→제습기) + 효과 시간당 상수화

- **이슈**: #27 · **사이클**: Normal
- **워크트리**: `../smartfarm_ai-devices` · 브랜치 `app/27-device-rework`

## 사용자 확정 스펙
1. **환기(vent) 제거 → 제습기(dehumidifier) 신설** — 장치 4종 대칭: 히터(+온도) / 쿨링팬(-온도) / 가습기(+습도) / 제습기(-습도)
2. **효과를 드라마틱하게** — "제어 후" 라인이 차트에서 눈에 띄게 밴드로 눌리도록

## 구현 범위

### 1. `src/control/actuators.py`
- DEVICES = ("dehumidifier", "humidifier", "cooling_fan", "heater") — vent 제거, 제습기 추가. DEVICE_LABEL_KR 갱신(제습기).

### 2. `src/control/effects.py`
- **시간당 상수로 재정의**: `EFFECTS_HOURLY = {heater: {온도 +2.0}, cooling_fan: {온도 -2.0}, humidifier: {습도 +8.0}, dehumidifier: {습도 -8.0}}` (조정 가능한 모듈 상수)
- 시뮬레이션 탭(1일 틱 리플레이)용은 `EFFECTS_DAILY`로 분리 — 기존 값 유지하되 vent→dehumidifier 치환(제습 -5.0/일 → dehumidifier로, 온도 효과 없음), apply_effects는 DAILY 사용(리플레이 무변경 원칙)

### 3. `src/control/controller.py`
- 습도 규칙: hum > high → **dehumidifier** ON / hum < low → humidifier ON. 온도 규칙에서 vent 제거(고온 → cooling_fan만). dehumidifier↔humidifier 동시 ON 금지(heater↔cooling_fan과 동일 패턴). P2-2 때 넣은 vent cause 구분 로직은 vent 제거로 단순화 가능(습도 장치가 전용이 되므로 cause 불필요해지면 제거)
- emergency 문구: "환기 풀가동에도 고습 지속" → "제습기 풀가동에도 고습 지속" 등

### 4. `src/control/live.py`
- `HOURLY_EFFECT_DIVISOR` 제거, `EFFECTS_HOURLY` 직접 적용
- **물리 클램프**: ctrl_hum 0~100%, ctrl_temp는 과도 발산 방지(예: base±10℃ 범위 클램프 — 상수로)
- 서모스탯 수렴: 밴드 복귀 시 히스테리시스 OFF(기존 로직 유지) — 효과가 커졌으니 오버슈트가 자연스럽게 데드밴드에서 멈추는지 테스트로 확인
- 상태 파일(`data/control_live_state.json`) 하위호환: 기존 devices에 vent 키 잔존 시 무시하고 신규 장치 키로 재구성(예외 없이)

### 5. `app/views/monitor.py`
- 장치 카드·라벨·아이콘 문구에서 환기→제습기 반영. K_LIVE_DEVICE_STATES/K_DEVICE_STATES 초기화는 default_states() 기반이라 자동 반영되지만, 세션에 옛 vent 상태가 남는 경우(서버 리런) 방어: states 키셋이 DEVICES와 다르면 리셋.

### 6. 테스트
- 기존 vent 관련 테스트 전면 갱신(controller·live·control 테스트)
- 신규: ①고습 프로파일에서 제습기 ON → ctrl_hum이 시간 경과에 따라 밴드로 **수렴**(드라마틱 효과 검증: 몇 시간 내 hum_high 아래로) ②클램프(0~100%) ③가습기/제습기 동시 ON 금지 ④상태 파일 vent 잔존 하위호환 ⑤긴급이 "정말 용량 초과"(제습 -8%p/h로도 못 잡는 초고습)에서만 발동

## 금지·경계
- `src/llm/**` 수정 금지 · 시뮬레이션 탭 리플레이 로직은 장치 치환 외 동작 변경 금지
- push·PR·머지·STATUS.md 수정 금지

## 완료 기준
- `OMP_NUM_THREADS=1 /Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai/.venv/bin/pytest -m "not integration"` 전체 PASS
- headless /monitor 200 (양 탭)
- 단계별 커밋(`이슈 #27`), 완료 보고 후 대기
