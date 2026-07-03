# handoff — 이슈 #33: 습도 밴드 중앙 목표 P-제어 + 차트 interactive

- **이슈**: #33 · **사이클**: Normal
- **워크트리**: `../smartfarm_ai-humidity` · 브랜치 `app/33-humidity-pcontrol`
- **주의**: PR #32(이벤트 분리, app/views/monitor.py)가 오픈 중 — monitor.py는 차트 함수(`_live_trend_chart`)만 건드려 충돌 최소화. 이벤트/긴급 섹션 코드는 절대 수정 금지.

## 설계 확정 (사용자 합의)
- **온도 = 서모스탯(bang-bang+히스테리시스) 현행 유지** — 톱니 진동이 의도된 거동
- **습도 = 밴드 중앙 목표 P-제어**: 목표 `hum_mid = (hum_low + hum_high) / 2`. 장치(제습기/가습기) ON 중 시간당 델타 = `clamp(K_P × (hum_mid - ctrl_hum), -8.0, +8.0)` (K_P는 상수, 예: 0.6 — 오차 13%p 이상이면 캡 도달하는 수준으로 튜닝·주석)
- ON/OFF 조건: ON은 기존(밴드 이탈 시), **OFF는 중앙 근접 시**(|ctrl_hum - hum_mid| ≤ hum_deadband). controller.decide()의 습도 판정을 이 기준으로 변경(온도 판정 로직은 무변경).
- 기존 습도 관통 방지 클램프는 P-제어 특성상 중앙을 넘어가지 않으므로 단순화 가능(제거 또는 중앙 기준으로 조정 — 구현하며 판단, docstring 명시)
- 시뮬레이션 탭(리플레이, EFFECTS_DAILY 경로)은 무변경.

## 구현 범위
1. `src/control/controller.py` — 습도 밴드 판정을 "이탈 시 ON → 중앙 도달 시 OFF"로 변경(전용 로직, 온도 코드 공유부 주의)
2. `src/control/live.py` — 습도 델타를 고정 ±8 → P-제어(위 식)로. `EFFECTS_HOURLY`의 습도 값은 "최대 출력"으로 의미 재정의(주석)
3. `app/views/monitor.py` — `_live_trend_chart()` 반환 차트에 `.interactive()` 추가(온·습도 공통)
4. run_notify/emergency: emergency의 "풀가동에도 지속" 판정은 최대 출력 기준이라 로직 유지 — P-제어 도입으로 오판(출력이 작아서 못 잡는데 긴급) 없는지 확인: 긴급 판정 시점엔 밴드 밖(오차 큼)이라 델타가 캡(±8)에 있으므로 의미 유지됨을 테스트로 확인
5. 테스트(`tests/test_control_live.py`·`test_control.py` 갱신/추가):
   - 고습 시작 → 제습기 ON → ctrl_hum이 **중앙(72.5) 부근으로 수렴**하고 진동 없음(전환 횟수 검증)
   - 중앙 근접 시 델타 감소(비례성), 캡 ±8 준수
   - 저습 대칭(가습기)
   - 온도 거동 무회귀(기존 온도 테스트 그대로 PASS)

## 완료 기준
- `OMP_NUM_THREADS=1 /Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai/.venv/bin/pytest -m "not integration"` 전체 PASS
- headless /monitor 200
- 단계별 커밋(`이슈 #33`), 완료 보고 후 대기 (push·PR 금지)
