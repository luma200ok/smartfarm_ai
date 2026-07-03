# handoff — 이슈 #35: 관제 차트 정합(미래 점선·축 라벨 잘림)

- **이슈**: #35 · **사이클**: Normal(표시 전용)
- **워크트리**: `../smartfarm_ai-chart` · 브랜치 `app/35-chart-polish`
- **대상**: `app/views/monitor.py`의 `_live_trend_chart()`(및 필요 시 데이터 준비부)만 — 시뮬 로직(`src/control/**`) 무변경

## 사용자 지적
1. x축 제목 "시간(시)"이 차트 하단에서 잘림
2. "제어 후" 라인이 미래 구간까지 실선 — 실행처럼 보여 이벤트 표(실행/🔮예정 분리)와 불일치
3. 습도 차트가 라인 교차로 어수선

## 구현 범위 (표시만)
1. **미래 구간 점선화**: 제어 후 라인(온·습도)을 now_hour 기준 두 세그먼트로 분리 —
   - `제어 후(실행)`: hour ≤ now, 실선(기존 색)
   - `제어 후(계획)`: hour ≥ now(연결되게 now 포인트 공유), **점선(strokeDash)** + 같은 색 계열
   - 범례에 두 항목이 구분되어 나오게(altair color/strokeDash 인코딩 조합 — detail/조건부 인코딩 활용)
   - `제어 전(기준선)`·`외기`도 미래 구간 점선 통일 여부는 가독성 보고 판단(과하면 제어 후만)
2. **축 라벨 잘림**: x축 제목 잘림 해결 — `st.altair_chart(use_container_width=True)` + 차트 `height` 명시(예: 300) 및 `padding`/`titlePadding` 조정. 렌더 확인 필수
3. **습도 가독성**: 교차 구간에서 라인 두께·투명도 미세 조정(기준선을 얇게/반투명, 제어 후를 굵게) — 과투자 금지, 점선화로 충분하면 생략 가능
4. 기존 "지금" 세로선·음영·캡션 유지

## 완료 기준
- `OMP_NUM_THREADS=1 /Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai/.venv/bin/pytest -m "not integration"` 전체 PASS(기존 테스트 무회귀, AppTest smoke 포함)
- headless /monitor 200
- 단계별 커밋(`이슈 #35`), 완료 보고 후 대기 (push·PR 금지)
