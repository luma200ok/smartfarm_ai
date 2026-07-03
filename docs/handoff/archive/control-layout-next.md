# handoff — 이슈 #25: 관제 페이지 레이아웃 재배치

- **이슈**: #25 · **사이클**: Normal
- **워크트리**: `../smartfarm_ai-layout` · 브랜치 `app/25-control-layout`
- **대상**: `app/views/monitor.py`(주) — `src/control/live.py`는 수동 장치 반영에 필요한 최소 수정만

## 사용자 확정 레이아웃

### [🟢 오늘 운영] 탭 (위→아래)
1. 현재 상태 카드(기존: 외기 실황·예측 내부 제어 전/후·작동 중 장치)
2. **⚙️ 설정 밴드** (탭 밖 공통 영역에서 이동 — 파일 영속 로직 그대로)
3. **🔌 장치 카드 4개** (시뮬레이션 탭에서 이동) — 자동/수동 토글·수동 ON/OFF가 **live 판정에 반영**: 세션에 live 전용 장치 상태(K_LIVE_DEVICE_STATES 신설, 시뮬용 K_DEVICE_STATES와 분리)를 두고 `simulate_control(..., states=live_states)`에 전달(수동 장치는 decide 제외 — 기존 controller 로직 그대로). 카드의 ON/OFF 뱃지는 현재 시각 타임라인 항목의 devices_on 기준으로 표시
4. 📈 오늘 0~24시 추이 — **온도(좌)·습도(우) st.columns(2) 반반**, 각각 외기/제어 전/제어 후 라인 + 해당 밴드 영역 오버레이(기존 온도 차트 구성 재사용해 습도 버전 추가)
5. 🧾 오늘 제어 이벤트
6. 🚨 긴급/이상
7. 📣 디스코드 알림 설정(탭 밖에서 이동)

### [🧪 시뮬레이션] 탭
- 장치 카드 섹션 **제거**(자동 제어만 데모 — _run_control_step은 세션 K_DEVICE_STATES 기존 그대로 사용, 수동 UI만 없어짐)
- "🎯 오늘 예측 vs 실측" 헤딩 → "🎯 **기대값 vs 실측**"으로 리네이밍(내용 무변경)
- 나머지(재생 작기·시나리오·날짜 이동·추이·제어 로그·긴급 피드) 무변경

### 탭 밖 공통 영역
- 전부 탭 안으로 이동해 비움(페이지 하단 캡션은 유지 가능)

## 주의
- 설정 밴드 파일 영속(`setpoints.save_changed`)·디스코드 토글 로직 무변경 — 위치만 이동
- `src/llm/**` 수정 금지 · run_notify(타이머 경로)는 수동 상태를 모름(세션 밖) — 현행 유지, docstring에 "수동 오버라이드는 앱 세션 한정" 한 줄 명시
- 기존 테스트(test_app_monitor AppTest smoke 포함) 무회귀 — 레이아웃 변경으로 AppTest가 깨지면 테스트를 새 구조에 맞게 갱신

## 완료 기준
- `OMP_NUM_THREADS=1 /Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai/.venv/bin/pytest -m "not integration"` 전체 PASS
- headless /monitor 200
- 단계별 커밋(`이슈 #25`), 완료 보고 후 대기 (push·PR 금지)
