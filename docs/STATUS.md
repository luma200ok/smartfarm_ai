# 📊 SmartFarm AI — 진행 현황 (STATUS)

> 마지막 갱신: **2026-07-04(스냅샷 누적 PR #41 반영)** · 레포 [github.com/luma200ok/smartfarm_ai](https://github.com/luma200ok/smartfarm_ai) (branch `main`)
> 새 세션은 이 문서 + [README](../README.md) + [roadmap](roadmap.md)로 현황 파악.

## 🟢 전체 상태: Phase 1·2·3 완료 (ML → DL → LSTM → LLM + 알림)

| Phase | 내용 | 상태 | 핵심 성과 |
|---|---|---|---|
| 1 ML | 환경센서 → 작물 분류 | ✅ | test F1 0.68 · GKF 0.49(누수 교훈) |
| 2 DL | 잎 진단(CNN·YOLO) + LSTM | ✅ | 진단 4분류 acc 0.96 · YOLO mAP@50 0.78 · LSTM |
| 3 LLM | 처방·RAG·통합·알림 | ✅ | 아래 표 |

## 🧩 Phase 3 구성 (파일 맵)
| 청크 | 산출물 | 요약 |
|---|---|---|
| 3-1 처방 | `src/llm/prescribe.py`·`tools.py` | Ollama `qwen2.5:14b` function calling(get_diagnosis·get_detection·get_forecast) + 환각방어 3종 + 구조화 JSON. 지연 개선: tool 라운드 캡·keep_alive 30m·프롬프트 다이어트·진행 콜백(이슈 #15, PR #16) → **fast-path 1-call**(`prescribe_fast`, 진단·RAG·예보 코드 직행 + writer 모델 분리, 이슈 #18, PR #19 — 서버 342.6→16.2s) |
| 3-2 RAG | `src/llm/rag/`·`data/nongsaro/*.md` | 농사로/NCPMS 코퍼스 → bge-m3 임베딩·numpy 코사인 → 근거 출처 코드 주입 |
| 3-3 통합 | `src/dl/infer.py`(forecast)·`src/dl/train_lstm.py`·`src/llm/pipeline.py`·`src/sim/virtual_sensor.py` | LSTM 환경예측(토마토 전용, MAE 1.11℃) + 시간축 처방 + 일일코치·조기경보 + 가상센서 재생 |
| 3-4 알림 | `src/llm/notify.py` | 경보·처방 디스코드 Webhook 발송(수동 버튼, 앱) |
| ➕ 자동감시 | `src/llm/monitor.py` | 규칙 임계값(습도·온도) 위험 시 디스코드 **자동** 알림(중복방지) |
| ➕ 날씨 | `src/llm/weather.py`·`tools.py`(get_weather)·`pipeline.py`(weather_qa) | 기상청 단기예보 API — 외기 실황·3일 예보 + 날씨 Q&A(앱 외부날씨 섹션). PR #9·#11, 이슈 #6 1단계 |
| ➕ 기대값 | `src/ml/train_expect.py`·`src/llm/expect.py`·`monitor.py`(cause·equip_anom·feedforward)·`sim/virtual_sensor.py`(inject) | 외기→실내 기대값 회귀(XGB MAE 1.11/1.44℃) — 원인 구분 경보·조기 감지·사전 경보·시나리오 데모. PR #12·#13, 이슈 #6 완결(+#2 흡수) |
| ➕ 관제 | `src/control/`(setpoints·actuators·controller·effects)·`app/views/monitor.py` | 규칙 기반(LLM 무관) 설정 밴드+히스테리시스 → 장치 4종(제습기·가습기·쿨링팬·히터, 이슈 #27에서 환기→제습기) 자동 ON/OFF·수동 토글·효과 피드백(inject tag 분리)·제어 로그·긴급 알림(자기정리 dedup). 앱 «환경 관제»로 개편(LLM 코치·날씨 Q&A 제거, pipeline·CLI는 유지). PR #20, 이슈 #17. 설정 밴드는 `data/control_setpoints.json` 파일 영속(전역 공유, 폴백·병합 저장 — PR #22, 이슈 #21) |
| ➕ 오늘 운영 | `src/control/live.py`·`deploy/smartfarm-control.{service,timer}` | **오늘 날짜 운영 모드**: KMA 실황·시간별 예보 → 기대값 모델로 오늘 내부 기준선 → 밴드 초과 시 장치 ON → 제어 후 값 조정(온도=서모스탯 관성 제어·습도=밴드 중앙 P-제어, 이슈 #27·#33). 앱 [오늘 운영/시뮬레이션] 탭(온·습도 반반 차트+지금 마커·예보 음영, 이슈 #25). 상주 알림: 서버 systemd 타이머 매시 실행 → 장치 전환·이상·긴급 디스코드(`data/control_live_state.json` dedup). PR #24, 이슈 #23 |

앱: `streamlit run app/streamlit_app.py` → **서비스형 2그룹 네비**([서비스] 농장 대시보드·잎 병해 진단·AI 처방·환경 관제·작물 환경 추천 / [프로젝트 기록] 개요·성과, ML/DL 실험 기록). `app/views/` 8페이지 + 공통 `ui.py`·`state.py`·`nav.py`·`.streamlit/config.toml` 테마 (이슈 #10, PR #14).

## 🖥 인프라 · 로컬 전제
| 항목 | 값 |
|---|---|
| LLM | Ollama — 로컬: `qwen2.5:14b`(처방·agentic) · 서버: `qwen2.5:7b`(agentic) + `exaone3.5:2.4b`(처방 fast-path writer, `OLLAMA_WRITER_MODEL`) · 공통: `bge-m3`(RAG 임베딩). `ollama pull` 필요, 데몬 구동 |
| 알림 | 디스코드 Webhook — `.env`의 `DISCORD_WEBHOOK_URL`(gitignore, 현재 쉘 env에 설정됨) |
| 데이터(로컬) | `data/processed/env_daily.csv`(LSTM·센서, gitignore) · `data/tomato/*`(진단) · `data/nongsaro/*.md`(RAG, 커밋됨) |
| 모델(로컬) | `models/*.pt`(gitignore) — `tomato_resnet18/mobilenet_v2/part/yolov8n`, `env_lstm.pt`(+meta json 커밋). `phase1_crop_env_clf.pkl` |
| 테스트 | `pytest`(통합=실 Ollama·실 PG). `pytest -m "not integration"`으로 모킹만 |
| 배포 | OCI(공용서버) — `docs/_local/deploy/oci-deployment.md` 참조 |
| 날씨 API | 기상청 단기예보(공공데이터포털) — `.env`의 `KMA_SERVICE_KEY`(로컬·서버 설정 완료), `FARM_LAT/LON` 미설정 시 서울 기본값. 미설정·장애 시 unavailable graceful |
| DB(선택) | PostgreSQL16+pgvector — `RAG_BACKEND=pgvector`·`DATABASE_URL` 설정 시만 사용(기본은 `memory`, npz+무이력 그대로). RAG 검색 저장 + 처방/경보 이력. 미설정·장애 시 자동 폴백 |

## 📌 다음 작업 (백로그 — roadmap "향후 확장" 참조)
- [x] **관제 시간별 스냅샷 누적**(이슈 #40 클로즈): KMA 예보가 과거 시간대를 안 줘 오늘 차트가 "지금~24시"로 퇴화 → 매시 스냅샷(first-write-wins, 상태 파일 v2 `snapshots`) 누적 + 타임라인 과거=기록·미래=예보 시뮬 합성(`assemble_today_timeline`·`seed_ctrl`) + KMA 실패 시 과거 기록 부분 렌더 + 롤오버 시 `data/control_history.json` 30일 아카이브(추후 실 센서 일별 이력 호환, source 필드) + 원자적 쓰기 `state_io.py` 공용화 — 배포 완료 (PR #41). 후속: 상태 파일 동시 쓰기 레이스(P2)·부분 렌더 기준 시각 라벨(P2)·실 센서 어댑터
- [x] **습도 기대값 회귀·차트 토글·사전경보 분리**(이슈 #37·#38 클로즈): 실내 습도를 외기+10%p 근사→학습 모델(GKF MAE 7.39%p, 온도 무회귀)로 교체, 비제어 라인 기본 숨김+비교 토글·범례 새 네이밍, 디스코드 현재=🚨긴급/미래=🔮사전 경보 요약/과거 미발송 — 모델 push_models.sh 배포·실검증 (PR #39). 후속 스타일: 차트 실행=빨강 실선·계획=회색 점선(207e513)
- [x] **관제 차트 정합·자정 연속성**(이슈 #35 클로즈): 제어 후 미래 구간 점선(계획) 분리·축 라벨 잘림 수정 + 타이머가 매시 last_ctrl 기록 → 0시가 어제 제어 값에서 이어짐(멱등성 보장, 폴백=밴드 클램프) (PR #36)
- [x] **날씨 인지 1단계**(이슈 #6): 기상청 API + get_weather + 날씨 Q&A + 앱 외기 데모 (PR #9·#11)
- [x] **날씨 인지 2·3단계 통합**(이슈 #6 클로즈): 외기→실내 기대값 회귀 + 원인 구분·equip_anom·feedforward + 시나리오 데모 + diseases 병해군(#2 흡수) (PR #12·#13)
- [x] **Streamlit 앱 전면 정리**(이슈 #10 클로즈): app/views/ 분해·서비스형 네비·신규 대시보드·문구 통일 + weather 4xx 재시도 후속(PR #11 P2) (PR #14)
- [x] **처방 지연 개선**(이슈 #15 클로즈): tool 라운드 num_predict 캡·keep_alive 30m·프롬프트 -17~20%·st.status 진행 표시 — 로컬 웜 28→17~19s(-35%), OCI CPU 추론 부담 완화 (PR #16)
- [x] **처방 fast-path 전환**(이슈 #18 클로즈): 진단·RAG·예보 코드 직행 + LLM 최종 JSON 1-call + writer 모델 분리(`OLLAMA_WRITER_MODEL`, 서버=exaone3.5:2.4b) — **서버 342.6→웜 16.2s(-95%)**, 로컬 10~15s. agentic prescribe()는 CLI·Q&A 유지 (PR #19)
- [x] **관제형 대시보드 개편**(이슈 #17 클로즈): 규칙 기반 밴드 자동제어(장치 4종 시뮬+효과 피드백)·제어 로그·긴급 디스코드 알림, 모니터링→«환경 관제» 전환 — 서버 배포·실검증 완료 (PR #20)
- [x] **습도 P-제어 전환**(이슈 #33 클로즈): 온도=서모스탯 유지·습도=밴드 중앙 목표 비례 제어(캡 ±8%p/h, hum_mode 분리로 리플레이 무영향) + 차트 줌 — 서버 실검증(중앙 수렴) (PR #34)
- [x] **관제 표시·복원력 개선**(이슈 #29·#31 클로즈): KMA stale-while-error 폴백·재시도 3회·UI 60s 캐시(PR #30) + 오늘 제어 이벤트 실행/🔮예정 분리 표시(PR #32)
- [x] **관제 장치 개편**(이슈 #27 클로즈): 환기→제습기(장치 4종 대칭), 시간당 효과 상수(±2℃/h·±8%p/h)+관성(누적) 제어·관통 방지 클램프 — 제어 후 라인 가시화, 채터링·교대 진동 해소(구버전 대조 실증) (PR #28)
- [x] **관제 레이아웃 재배치**(이슈 #25 클로즈): 설정 밴드·장치 카드를 오늘 운영 탭으로(수동 토글 live 반영, deepcopy로 리런 드리프트 차단), 온·습도 반반 차트+지금 마커·예보 음영, 시뮬 탭 '기대값 vs 실측' 리네이밍, altair 의존성 명시 (PR #26)
- [x] **관제 오늘 운영 모드**(이슈 #23 클로즈): KMA 기반 오늘 내부환경 예측→장치 제어 온도 조정 + 매시 상주 디스코드 알림(smartfarm-control.timer 서버 설치·실발송 검증) — 리플레이는 시뮬레이션 탭으로 (PR #24)
- [x] **설정 밴드 파일 영속화**(이슈 #21 클로즈): 새로고침·재접속 초기화 → `data/control_setpoints.json` 원자적 저장+폴백+변경 필드 병합, 서버 실검증(새 세션 유지) 완료 (PR #22)
- [x] **진단 병해 클래스 확장 1차**(전이학습): 잎마름역병(late_blight) 추가 → **4분류**(PV 898/100장 혼합, resnet18 acc 0.96·late_blight f1 0.95) + RAG 코퍼스 `late_blight.md`. (PR #3)
### 🔥 활성
- [ ] **농사로 OpenAPI 로더**(진행 중): 수기 코퍼스 4문서 대체 + normal 코퍼스 실자료화 — RAG 검색 품질 상한 해소, 이후 병해 확장 시 코퍼스 자동화 기반
- [ ] **대화형 Q&A 디스코드 봇**(검토 중): pipeline Q&A 경로를 봇으로 노출(Webhook과 별개) — CPU 서버 응답 지연(agentic 경로 수십 초) 감안해 설계 필요

### 💤 보류 (재개 조건 명시)
- [ ] 실센서/스프링 서버 sensor API 어댑터 + 분 단위 스냅샷(source:"sensor" 자리 확보됨, PR #41) — 실 센서/연동 계획 확정 시
- [ ] 진단 병해 클래스 확장 2차(흰가루·잿빛) — 데이터 소스 발굴이 선행(PV에 잿빛 없음)
- [ ] 관제 선제 가동(예측 제어) — 예보상 밴드 이탈 1~2h 전 선가동(잔여 논의 항목)
- [ ] 상태 파일 동시 쓰기 레이스·KMA 실패 부분 렌더 기준 시각 라벨(PR #41 리뷰 P2 이월) — 운영 실측 후 격상 검토
- [ ] 예보 경로 stale 캡션 노출(PR #30 P3 이월) / 처방 사후 클래스 검증·CI pytest 편입·PART_CLASSES 드리프트(3-1 리뷰 이월) — 위생 항목, 실익 낮음
- [ ] **문서 보강**: README·figures stale 이미지 교체 + 환경 관제 서술 보강(서모스탯/P-제어·자정 연속성·스냅샷 누적) + 노션 반영
- ~~monitor cron 상주·쿨다운~~ → systemd 타이머 상주 알림(PR #24)으로 대체되어 백로그 제외

## ⚠️ 알려진 이슈 / 주의
- **macOS 로컬**: torch+xgboost(env_expect_reg.pkl) 동시 로드 시 libomp 세그폴트 — `OMP_NUM_THREADS=1 python …`으로 실행. OCI 서버는 재현 안 됨(스모크 통과).
- monitor.py·앱 전송 버튼은 **실제 디스코드로 발송**됨(웹훅 설정 상태) — 데모/테스트 시 유의.
- 관제 [오늘 운영]은 **KMA 실시간 예보 기반**(실센서는 미연동 — 내부값은 기대값 모델 추정). 처방의 LSTM forecast·[시뮬레이션] 탭은 과거 데이터 재생(replay), 가상센서는 대표 토마토 농가 1개 시계열 사용.
- 스프링 `smartfarm_server`는 **별개 프로젝트**(백엔드/IoT) — smartfarm_ai와 결합 X, 웹훅만 공유.
