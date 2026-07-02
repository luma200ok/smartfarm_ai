# 🌱 스마트팜 재배 도우미 로드맵 (ML → DL → LLM)

> **목표:** 스마트팜 환경 데이터 + 잎 사진을 학습해서,
> "💧 토양 건조하니 관수, 🔬 잎에 잎곰팡이병 의심, 🌡️ 곧 고온이니 환기" 처방해주는 AI.
>
> **전략:** 하나의 도메인(스마트팜)으로 ML→DL→LLM을 쌓되, 단계마다 새 모달리티(정형→이미지→언어)를 도입.
> **작물:** 토마토 단일로 시작 → **전이학습으로 다작물 확장** (→ [decisions.md ADR-002](decisions.md)).
> ← [PRD](prd.md) · [데이터 출처](data_sources.md) · [설계 결정](decisions.md)

---

## 🏗️ 전체 아키텍처

```
[Phase 1: ML]  스마트팜 환경 CSV → 작물 분류 / 적합도          ✅ 완료
                    ↓ 새 모달리티(이미지·시퀀스)
[Phase 2: DL]  ① CNN  → 잎 사진 병해충 진단  ⭐ 차별화 핵심
               ② LSTM → 환경 시계열 예측
                    ↓ 예측·진단 결과를 가져다 씀
[Phase 3: LLM] CNN진단 + 환경예측 + 재배가이드(RAG)
                → LLM 자연어 처방 + 📱 알림
   "🔬 잎곰팡이병 의심(87%). 감염 잎 제거, 습도↓.
    💧 토양수분 30% 낮음 — 관수. 🌡️ 2시간 뒤 32도, 환기 준비."
```

---

## 🧭 청크 지도

> Phase = 큰 흐름, 청크 = 1회분. ✅완료 · ▶다음 · ⬜예정.

### Phase 1 — ML (환경 → 작물 8종 분류) · ⭐ ✅ 완료
- ✅ 1-1 레포 분리 + 농진청 데이터 확보
- ✅ 1-2 EDA (작물 분포·농가-작물 관계·센서 결측)
- ✅ 1-3 전처리 `preprocess.py` (288만 시간별 → 116,365 일별 집계, **2022~24 다년**)
- ✅ 1-4 모델 비교 `train.py` (로지스틱·RF·XGBoost) → XGBoost 베스트(test F1 0.68 · GKF 0.49)
- ✅ 1-5 평가 3겹 — **데이터 누수 실증**(랜덤 0.67 vs GroupKFold 0.49)
- ✅ 1-6 포트폴리오 `phase1_ml.md` + 그림 3종
- ✅ 1-7 **다년 결합(2022~24)** — `compare_years.py` 데이터 양 효과(공통 8작물 F1 +0.073) + 수박 신규 커버
- ⬜ (선택) Streamlit 데모 — "환경만으로 작물 식별 한계" 시연
- 🏁 **Phase 1 끝 = 분류 파이프라인 + 누수 교훈 보고서**

### Phase 2 — DL (비전 + 시계열) · ⭐⭐⭐⭐ ← 차별화 핵심 ✅ 완료
> **ML 5스텝처럼 DL도 4스텝 + 검출(고급)**: 기초 → 핵심 → 평가 → 데모 → 검출. (`2-N`=불변 ID, 묶음=스텝)

**STEP 1 · 기초 — 신경망 원리 (forward → 학습 → 데이터 파이프)** — `01_basics.py`
- ✅ 2-0 환경·PyTorch 첫걸음 (설치·MPS 확인)
- ✅ 2-1 신경망 기초(뉴런·활성화) — "활성화 없으면 직선" 실증
- ✅ 2-2 학습 메커니즘(손실·역전파·Adam) — 5단계 루프
- ✅ 2-3 Dataset/DataLoader · batch 학습 골격 — make_moons 곡선 결정경계 + mini vs full 비교

**STEP 2 · 핵심 — 모델 구축 ⭐ (비전 + 시계열)** — `02_core.py`
- ✅ 2-4 CNN 기초 (Conv·Pooling) — FashionMNIST acc 0.87
- ✅ 2-5 전이학습 — **백본 비교(MLflow 추적)** mobilenet_v2 0.987·resnet18 0.971, 토마토 잎 **3분류**(정상·잎곰팡이·황화잎말이) · 정상 원천 **잎(area3)만 정제**(train 3,255·val 691) ⭐
- ✅ 2-6 **Grad-CAM** (설명가능 AI — 어느 병반을 보고 판단했나, 서빙=resnet18 layer4) ⭐
- ✅ 2-8 LSTM — 환경 **다변량 8변수·485개 시계열**(2022~24 다년) 통합 (MAE 1.18℃ < baseline 1.25℃ · 단년 1.22→다년 1.18 데이터 양 효과)

**STEP 3 · 평가 — 강건화·검증** — `03_eval.py`
- ✅ 2-7 과적합·불균형·학습안정 (클래스가중치 → 질병 recall 0.86/0.93→0.96/0.98)
- ✅ 2-7b **서빙 강건화** — plant_score(0.04) + **부위 게이트**(과실/꽃/잎/줄기 acc 0.932) 2단 OOD 방어
- ✅ 2-9 평가 심화 (3×3 혼동행렬·acc 0.97·ROC-AUC 0.997·FN 6건 분석)

**STEP 4 · 데모 — 배포·마무리** — `04_demo.py` · `app/phase2_dl.py`
- ✅ 2-10 모델 저장(.pt) + Streamlit (사진 업로드 → 진단 + Grad-CAM)
- ✅ 2-12 회고 (코드 회고 완료 · 정식 `phase2_dl.md` 수행내역서 작성 완료)

**STEP 5 · 검출(고급) — YOLO 위치 검출** — `prepare_tomato_yolo.py` · `05_detect.py`
- ✅ 2-11 YOLO 병해 잎 위치 검출 (YOLOv8n 전이학습, **3클래스** normal/leaf_mold/tylcv, **mAP@50 0.78**) ⭐
- 🏁 **Phase 2 끝 = 사진 올리면 진단+히트맵+위치박스, 작물 1개씩 확장 가능한 파이프라인**

> 청크 상세 = `_local/concepts/DL_devlog.md` · 이론 = `DL.md`.

### Phase 3 — LLM (통합 처방 + 알림) · ⭐⭐⭐ ✅ 완료
- ✅ 3-1 Ollama(qwen2.5:14b) 연동 (숫자·라벨 → 자연어 처방, function calling + 환각방어 3종)
- ✅ 3-2 재배지식 RAG (농사로/NCPMS 가이드 검색 → 근거 인용, bge-m3 임베딩)
- ✅ 3-3 통합 파이프라인 (LSTM 환경예측 실연동 + 시간축 처방 + 일일코치·조기경보)
- ✅ 3-4 알림(디스코드 Webhook) — 경보·처방 발송(수동 버튼) + Streamlit 대시보드
- 🏁 **Phase 3 끝 = 사진+센서 → 자연어 처방 + 알림**

---

## 📅 Phase별 데이터·기술 요약

| | Phase 1 (ML) ✅ | Phase 2 (DL) ✅ | Phase 3 (LLM) ▶ |
|---|---|---|---|
| **목표** | 환경 → 작물 분류 | 잎 진단 + 환경 예측 | 말로 처방+알림 |
| **모달리티** | 정형 센서 | + 이미지 | + 언어 |
| **데이터** | 농진청 스마트팜 현장 농가 | AI Hub 071·PlantVillage·534 | 진단+예측+농사로 RAG |
| **핵심 기술** | scikit-learn·XGBoost | PyTorch·전이학습·Grad-CAM | Ollama(qwen2.5:14b)·RAG |
| **결과물** | 분류 모델(test F1 0.68·GKF 0.49) | 진단 모델(.pt) | 📱 처방 알림 |
| **배포** | (선택) Streamlit | Streamlit 진단 데모 | 대시보드+알림봇 |

> 데이터 출처 상세 → [data_sources.md](data_sources.md) · 데이터 맵 → [decisions.md 📦](decisions.md).
> 배포: 단계별 개별 Streamlit 앱(`app/phaseN_*.py`), 백엔드(`data/`·`models/`·`src/common/`) 공유.

---

## ✅ 결정됨 (→ [decisions.md](decisions.md))
- **작물:** 토마토 단일 시작 → 딸기·오이·참외 확장 (ML∩DL 교집합) · ADR-002
- **Phase 1 데이터:** 농진청 스마트팜 현장 농가 데이터 (옛 노지 Kaggle은 [ml-learn](https://github.com/luma200ok/smartfarm_ml_learn) v1로 분리) · ADR-001
- **Phase 2 병진단:** PlantVillage(즉시) → AI Hub 071(국내) · ADR-005
- **DL 프레임워크:** PyTorch (MPS) · **알림:** 디스코드 Webhook

## ✅ 다음 액션 (Phase 3 LLM 시작)
- Phase 2 DL 완료: 3분류(서빙 resnet18 acc 0.97·백본 best mobilenet 0.987, MLflow)+설명(Grad-CAM)+검출(YOLO mAP@50 0.78)+강건화(부위 게이트 0.932)+다변량 시계열(LSTM MAE 1.18℃<baseline 1.25, 485개 다년) · 수행내역서 `phase2_dl.md` · 데모 smartfarm-ai.rkqkdrnportfolio.shop
1. 청크 3-1(LLM 기초·프롬프트)부터 → `src/llm/`
2. CNN 진단 + LSTM 예측 + RAG 재배가이드 → 자연어 처방 통합 파이프라인

---

## 🔮 향후 확장 백로그 (기능 구현 후, 전이학습식 점진 확장)

### ⭐ 진단 병해 클래스 확장 (Phase 2 재학습 — 최우선 확장)
현재 진단 모델은 **3종(잎곰팡이병·정상·황화잎말이)**만 구분. 농사로 곰팡이병 큐레이션엔 4종이 있으나 나머지는 학습 데이터가 없어 진단 불가.
**확장 = 이미지 데이터 확보 → CNN 재학습 → 재평가 → 부위/OOD 게이트·클래스한정 프롬프트·RAG 코퍼스 동반 갱신** (진단 클래스와 RAG는 반드시 세트로 확장).
- **잎마름역병(late blight):** ✅ PlantVillage(CC0)에 데이터 있음 → **가장 먼저 추가 가능**
- **흰가루병(powdery mildew)·잿빛곰팡이병(gray mold):** ❌ PlantVillage에 없음 → 별도 데이터셋 수집 선행 필요
- 다작물 확장(딸기·오이·참외)도 같은 전이학습 패턴.

### 그 외 후속 (리뷰·설계에서 미룬 항목)
- **RAG 코퍼스:** `normal`(일반재배) 시드 → 농사로 재배기술 실자료로 교체 · (선택) 비진단 병해 3종 `disease: reference`로 보관해 자유 Q&A용
- **농사로 OpenAPI 로더:** 수기 코퍼스 → API 자동 수집으로 교체(`corpus.py` 소스 어댑터)
- **3-1 리뷰 이월:** 처방 사후 클래스 검증(범위밖 자동 차단) · CI에 pytest 편입(모델 아티팩트 필요) · `PART_CLASSES` ↔ `tomato_part_classes.json` 드리프트 방지
