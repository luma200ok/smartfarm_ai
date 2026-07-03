# handoff — 이슈 #37+#38: 습도 기대값 회귀 · 차트 토글/네이밍 · 디스코드 사전경보 분리

- **이슈**: #37(습도 모델·차트) + #38(디스코드 알림) — 한 브랜치에서 순차 구현, PR은 둘 다 Closes
- **사이클**: Normal · **워크트리**: `../smartfarm_ai-humexpect` · 브랜치 `ml/37-humidity-expect`
- 데이터: 워크트리에 `data/processed`(심볼릭 링크→메인, env_daily.csv 사용 가능) 준비됨. 학습 산출 pkl은 **워크트리 `models/`에 생성**(gitignore, 커밋 금지 — 배포는 메타가 push_models.sh로).

## A. 습도 기대값 회귀 (#37)
1. `src/ml/train_expect.py` 정독 후 **실내 습도(습도내부_평균) 타깃 추가** — 기존 온도 파이프라인(피처: 온도외부_평균·일사량_평균·doy sin/cos, XGB, GroupKFold)과 동일 체계. 필요 시 외기 습도 피처 추가 검토(있으면 성능 비교 후 채택 여부 결정·기록).
2. 재학습 → pkl 스키마에 습도 키 추가(기존 키 유지·확장). **온도 MAE가 기존(평균 1.11℃) 대비 급락하지 않는지 비교 기록**, 습도 GKF-MAE 수치 보고. 실행은 `OMP_NUM_THREADS=1`.
3. `src/llm/expect.py` predict()에 습도 반환 추가(모델에 습도 키 없으면 미포함 — 구모델 하위호환). `src/control/live.py` indoor_baseline(): 모델 습도 있으면 사용, 없으면 기존 `외기+INDOOR_HUMIDITY_OFFSET` 폴백(docstring 명시).
4. `src/llm/monitor.py` 등 기존 온도 잔차 소비처 무회귀 확인.

## B. 차트 토글·네이밍 (#37, 사용자 확정)
1. **비제어 기준선 라인: 기본 숨김 + 체크박스 토글**("제어 없을 때 비교 보기" 류 쉬운 문구) — 켜면 얇은 회색 반투명으로 표시.
2. **범례 네이밍(온/습도 각각)**: `외부 온도`/`실내 온도(제어)`/(토글 시)`실내 온도(제어 없음)` · `외부 습도`/`실내 습도(제어)`/(토글 시)`실내 습도(제어 없음)`. 미래 구간은 라벨 없이 기존 점선+음영 유지, 캡션 "점선 구간 = 예보 기반 예측"으로 문구 정리.
3. 관련 캡션·툴팁·`_split_control_segments` 테스트 라벨 기대값 갱신.

## C. 디스코드 사전경보 분리 (#38, A안 확정)
`run_notify()`의 긴급 발송을 시점별 분기:
- **현재 시각(hour == now.hour) 긴급만 "🚨 긴급"** — 기존 빨강 embed
- **미래 시간대(hour > now.hour)는 "🔮 사전 경보(예보 기반 예상)"** — 다른 색(예: 주황), **여러 시간대를 1건 embed로 요약**(fields에 시간대별 사유 나열), dedup은 기존 `{hour}:{kind}` 키 유지(요약에 포함된 키 전부 상태 저장)
- **과거 시간대(hour < now.hour)는 발송 안 함**(뒷북 제거 — 상태 키에는 남겨 재발송 방지)
- 테스트: 시점 분기(현재/미래/과거), 요약 묶음 1건 발송, dedup(재실행 0건), 자정 리셋 후 재판정

## 금지·경계
- `app/views/prescribe.py`·`src/llm/prescribe.py`·`pipeline.py` 수정 금지 · push·PR·머지·STATUS.md·모델 파일 커밋 금지

## 완료 기준
- `OMP_NUM_THREADS=1 /Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai/.venv/bin/pytest -m "not integration"` 전체 PASS
- 재학습 성공(온도 MAE 비교 + 습도 MAE 보고) · headless /monitor 200
- 단계별 커밋(A→B→C 순, `이슈 #37`/`이슈 #38` 표기), 완료 보고 후 대기
