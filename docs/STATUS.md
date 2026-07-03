# 📊 SmartFarm AI — 진행 현황 (STATUS)

> 마지막 갱신: **2026-07-03** · 레포 [github.com/luma200ok/smartfarm_ai](https://github.com/luma200ok/smartfarm_ai) (branch `main`)
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
| 3-1 처방 | `src/llm/prescribe.py`·`tools.py` | Ollama `qwen2.5:14b` function calling(get_diagnosis·get_detection·get_forecast) + 환각방어 3종 + 구조화 JSON. 지연 개선: tool 라운드 캡·keep_alive 30m·프롬프트 다이어트·진행 콜백(이슈 #15, PR #16) |
| 3-2 RAG | `src/llm/rag/`·`data/nongsaro/*.md` | 농사로/NCPMS 코퍼스 → bge-m3 임베딩·numpy 코사인 → 근거 출처 코드 주입 |
| 3-3 통합 | `src/dl/infer.py`(forecast)·`src/dl/train_lstm.py`·`src/llm/pipeline.py`·`src/sim/virtual_sensor.py` | LSTM 환경예측(토마토 전용, MAE 1.11℃) + 시간축 처방 + 일일코치·조기경보 + 가상센서 재생 |
| 3-4 알림 | `src/llm/notify.py` | 경보·처방 디스코드 Webhook 발송(수동 버튼, 앱) |
| ➕ 자동감시 | `src/llm/monitor.py` | 규칙 임계값(습도·온도) 위험 시 디스코드 **자동** 알림(중복방지) |
| ➕ 날씨 | `src/llm/weather.py`·`tools.py`(get_weather)·`pipeline.py`(weather_qa) | 기상청 단기예보 API — 외기 실황·3일 예보 + 날씨 Q&A(앱 외부날씨 섹션). PR #9·#11, 이슈 #6 1단계 |
| ➕ 기대값 | `src/ml/train_expect.py`·`src/llm/expect.py`·`monitor.py`(cause·equip_anom·feedforward)·`sim/virtual_sensor.py`(inject) | 외기→실내 기대값 회귀(XGB MAE 1.11/1.44℃) — 원인 구분 경보·조기 감지·사전 경보·시나리오 데모. PR #12·#13, 이슈 #6 완결(+#2 흡수) |
| ➕ 관제 | `src/control/`(setpoints·actuators·controller·effects)·`app/views/monitor.py` | 규칙 기반(LLM 무관) 설정 밴드(온도 20~25℃·습도 60~85%)+히스테리시스 → 장치 4종(환기·가습기·쿨링팬·히터) 자동 ON/OFF·수동 토글·효과 피드백(inject tag 분리)·제어 로그·긴급 알림(자기정리 dedup). 앱 «환경 관제»로 개편(LLM 코치·날씨 Q&A 제거, pipeline·CLI는 유지). PR #20, 이슈 #17. 설정 밴드는 `data/control_setpoints.json` 파일 영속(전역 공유, 폴백·병합 저장 — PR #22, 이슈 #21) |
| ➕ 오늘 운영 | `src/control/live.py`·`deploy/smartfarm-control.{service,timer}` | **오늘 날짜 운영 모드**: KMA 실황·시간별 예보 → 기대값 모델로 오늘 내부 기준선 → 밴드 초과 시 장치 ON → 제어 후 온도 조정(서모스탯 시뮬). 앱 [오늘 운영/시뮬레이션] 탭. 상주 알림: 서버 systemd 타이머 매시 실행 → 장치 전환·이상·긴급 디스코드(`data/control_live_state.json` dedup). PR #24, 이슈 #23 |

앱: `streamlit run app/streamlit_app.py` → **서비스형 2그룹 네비**([서비스] 농장 대시보드·잎 병해 진단·AI 처방·환경 관제·작물 환경 추천 / [프로젝트 기록] 개요·성과, ML/DL 실험 기록). `app/views/` 8페이지 + 공통 `ui.py`·`state.py`·`nav.py`·`.streamlit/config.toml` 테마 (이슈 #10, PR #14).

## 🖥 인프라 · 로컬 전제
| 항목 | 값 |
|---|---|
| LLM | Ollama 로컬 `qwen2.5:14b`(처방) · `bge-m3`(RAG 임베딩) — `ollama pull` 필요, 데몬 구동 |
| 알림 | 디스코드 Webhook — `.env`의 `DISCORD_WEBHOOK_URL`(gitignore, 현재 쉘 env에 설정됨) |
| 데이터(로컬) | `data/processed/env_daily.csv`(LSTM·센서, gitignore) · `data/tomato/*`(진단) · `data/nongsaro/*.md`(RAG, 커밋됨) |
| 모델(로컬) | `models/*.pt`(gitignore) — `tomato_resnet18/mobilenet_v2/part/yolov8n`, `env_lstm.pt`(+meta json 커밋). `phase1_crop_env_clf.pkl` |
| 테스트 | `pytest`(통합=실 Ollama·실 PG). `pytest -m "not integration"`으로 모킹만 |
| 배포 | OCI(공용서버) — `docs/_local/deploy/oci-deployment.md` 참조 |
| 날씨 API | 기상청 단기예보(공공데이터포털) — `.env`의 `KMA_SERVICE_KEY`(로컬·서버 설정 완료), `FARM_LAT/LON` 미설정 시 서울 기본값. 미설정·장애 시 unavailable graceful |
| DB(선택) | PostgreSQL16+pgvector — `RAG_BACKEND=pgvector`·`DATABASE_URL` 설정 시만 사용(기본은 `memory`, npz+무이력 그대로). RAG 검색 저장 + 처방/경보 이력. 미설정·장애 시 자동 폴백 |

## 📌 다음 작업 (백로그 — roadmap "향후 확장" 참조)
- [x] **날씨 인지 1단계**(이슈 #6): 기상청 API + get_weather + 날씨 Q&A + 앱 외기 데모 (PR #9·#11)
- [x] **날씨 인지 2·3단계 통합**(이슈 #6 클로즈): 외기→실내 기대값 회귀 + 원인 구분·equip_anom·feedforward + 시나리오 데모 + diseases 병해군(#2 흡수) (PR #12·#13)
- [x] **Streamlit 앱 전면 정리**(이슈 #10 클로즈): app/views/ 분해·서비스형 네비·신규 대시보드·문구 통일 + weather 4xx 재시도 후속(PR #11 P2) (PR #14)
- [x] **처방 지연 개선**(이슈 #15 클로즈): tool 라운드 num_predict 캡·keep_alive 30m·프롬프트 -17~20%·st.status 진행 표시 — 로컬 웜 28→17~19s(-35%), OCI CPU 추론 부담 완화 (PR #16)
- [x] **처방 fast-path 전환**(이슈 #18 클로즈): 진단·RAG·예보 코드 직행 + LLM 최종 JSON 1-call + writer 모델 분리(`OLLAMA_WRITER_MODEL`, 서버=exaone3.5:2.4b) — **서버 342.6→웜 16.2s(-95%)**, 로컬 10~15s. agentic prescribe()는 CLI·Q&A 유지 (PR #19)
- [x] **관제형 대시보드 개편**(이슈 #17 클로즈): 규칙 기반 밴드 자동제어(장치 4종 시뮬+효과 피드백)·제어 로그·긴급 디스코드 알림, 모니터링→«환경 관제» 전환 — 서버 배포·실검증 완료 (PR #20)
- [x] **관제 오늘 운영 모드**(이슈 #23 클로즈): KMA 기반 오늘 내부환경 예측→장치 제어 온도 조정 + 매시 상주 디스코드 알림(smartfarm-control.timer 서버 설치·실발송 검증) — 리플레이는 시뮬레이션 탭으로 (PR #24)
- [x] **설정 밴드 파일 영속화**(이슈 #21 클로즈): 새로고침·재접속 초기화 → `data/control_setpoints.json` 원자적 저장+폴백+변경 필드 병합, 서버 실검증(새 세션 유지) 완료 (PR #22)
- [x] **진단 병해 클래스 확장 1차**(전이학습): 잎마름역병(late_blight) 추가 → **4분류**(PV 898/100장 혼합, resnet18 acc 0.96·late_blight f1 0.95) + RAG 코퍼스 `late_blight.md`. (PR #3)
- [ ] 진단 병해 클래스 확장 2차: 흰가루·잿빛(데이터 수집 필요).
- [ ] 실센서/스프링 서버 sensor API를 `monitor.py`·가상센서 소스로 어댑터 연결
- [ ] 농사로 OpenAPI 로더(수기 코퍼스 대체) · normal 코퍼스 실자료화
- [ ] 대화형 Q&A 디스코드 봇(Webhook과 별개) · monitor cron 상주·쿨다운 정책
- [ ] (3-1 리뷰 이월) 처방 사후 클래스 검증 · CI에 pytest 편입(모델 아티팩트 필요) · PART_CLASSES↔json 드리프트

## ⚠️ 알려진 이슈 / 주의
- **macOS 로컬**: torch+xgboost(env_expect_reg.pkl) 동시 로드 시 libomp 세그폴트 — `OMP_NUM_THREADS=1 python …`으로 실행. OCI 서버는 재현 안 됨(스모크 통과).
- monitor.py·앱 전송 버튼은 **실제 디스코드로 발송**됨(웹훅 설정 상태) — 데모/테스트 시 유의.
- forecast는 **과거 데이터 재생(replay)** 데모 — 실시간 아님(실센서 미연동). 가상센서가 대표 토마토 농가 1개 시계열 사용.
- 스프링 `smartfarm_server`는 **별개 프로젝트**(백엔드/IoT) — smartfarm_ai와 결합 X, 웹훅만 공유.
