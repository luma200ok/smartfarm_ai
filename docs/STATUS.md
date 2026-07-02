# 📊 SmartFarm AI — 진행 현황 (STATUS)

> 마지막 갱신: **2026-07-02** · 레포 [github.com/luma200ok/smartfarm_ai](https://github.com/luma200ok/smartfarm_ai) (branch `main`)
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
| 3-1 처방 | `src/llm/prescribe.py`·`tools.py` | Ollama `qwen2.5:14b` function calling(get_diagnosis·get_detection·get_forecast) + 환각방어 3종 + 구조화 JSON |
| 3-2 RAG | `src/llm/rag/`·`data/nongsaro/*.md` | 농사로/NCPMS 코퍼스 → bge-m3 임베딩·numpy 코사인 → 근거 출처 코드 주입 |
| 3-3 통합 | `src/dl/infer.py`(forecast)·`src/dl/train_lstm.py`·`src/llm/pipeline.py`·`src/sim/virtual_sensor.py` | LSTM 환경예측(토마토 전용, MAE 1.11℃) + 시간축 처방 + 일일코치·조기경보 + 가상센서 재생 |
| 3-4 알림 | `src/llm/notify.py` | 경보·처방 디스코드 Webhook 발송(수동 버튼, 앱) |
| ➕ 자동감시 | `src/llm/monitor.py` | 규칙 임계값(습도·온도) 위험 시 디스코드 **자동** 알림(중복방지) |

앱: `streamlit run app/streamlit_app.py` → Phase 1/2/3 멀티페이지. 진단·처방·근거·가상센서·코치·경보·전송 버튼.

## 🖥 인프라 · 로컬 전제
| 항목 | 값 |
|---|---|
| LLM | Ollama 로컬 `qwen2.5:14b`(처방) · `bge-m3`(RAG 임베딩) — `ollama pull` 필요, 데몬 구동 |
| 알림 | 디스코드 Webhook — `.env`의 `DISCORD_WEBHOOK_URL`(gitignore, 현재 쉘 env에 설정됨) |
| 데이터(로컬) | `data/processed/env_daily.csv`(LSTM·센서, gitignore) · `data/tomato/*`(진단) · `data/nongsaro/*.md`(RAG, 커밋됨) |
| 모델(로컬) | `models/*.pt`(gitignore) — `tomato_resnet18/mobilenet_v2/part/yolov8n`, `env_lstm.pt`(+meta json 커밋). `phase1_crop_env_clf.pkl` |
| 테스트 | `pytest`(통합=실 Ollama·실 PG). `pytest -m "not integration"`으로 모킹만 |
| 배포 | OCI(공용서버) — `docs/_local/deploy/oci-deployment.md` 참조 |
| DB(선택) | PostgreSQL16+pgvector — `RAG_BACKEND=pgvector`·`DATABASE_URL` 설정 시만 사용(기본은 `memory`, npz+무이력 그대로). RAG 검색 저장 + 처방/경보 이력. 미설정·장애 시 자동 폴백 |

## 📌 다음 작업 (백로그 — roadmap "향후 확장" 참조)
- [x] **진단 병해 클래스 확장 1차**(전이학습): 잎마름역병(late_blight) 추가 → **4분류**(PV 898/100장 혼합, resnet18 acc 0.96·late_blight f1 0.95) + RAG 코퍼스 `late_blight.md`. (PR #3)
- [ ] 진단 병해 클래스 확장 2차: 흰가루·잿빛(데이터 수집 필요).
- [ ] 실센서/스프링 서버 sensor API를 `monitor.py`·가상센서 소스로 어댑터 연결
- [ ] 농사로 OpenAPI 로더(수기 코퍼스 대체) · normal 코퍼스 실자료화
- [ ] 대화형 Q&A 디스코드 봇(Webhook과 별개) · monitor cron 상주·쿨다운 정책
- [ ] (3-1 리뷰 이월) 처방 사후 클래스 검증 · CI에 pytest 편입(모델 아티팩트 필요) · PART_CLASSES↔json 드리프트

## ⚠️ 알려진 이슈 / 주의
- monitor.py·앱 전송 버튼은 **실제 디스코드로 발송**됨(웹훅 설정 상태) — 데모/테스트 시 유의.
- forecast는 **과거 데이터 재생(replay)** 데모 — 실시간 아님(실센서 미연동). 가상센서가 대표 토마토 농가 1개 시계열 사용.
- 스프링 `smartfarm_server`는 **별개 프로젝트**(백엔드/IoT) — smartfarm_ai와 결합 X, 웹훅만 공유.
