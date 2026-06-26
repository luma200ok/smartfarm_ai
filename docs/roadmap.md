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
- ✅ 1-3 전처리 `preprocess.py` (833만 시간별 → 33,278 일별 집계)
- ✅ 1-4 모델 비교 `train.py` (로지스틱·RF·XGBoost) → XGBoost 베스트(F1 0.78)
- ✅ 1-5 평가 3겹 — **데이터 누수 실증**(랜덤 0.77 vs GroupKFold 0.41)
- ✅ 1-6 포트폴리오 `phase1_ml.md` + 그림 3종
- ⬜ (선택) Streamlit 데모 — "환경만으로 작물 식별 한계" 시연
- 🏁 **Phase 1 끝 = 분류 파이프라인 + 누수 교훈 보고서**

### Phase 2 — DL (비전 + 시계열) · ⭐⭐⭐⭐ ← 차별화 핵심 ▶ 다음
> **ML 5스텝처럼 DL도 4스텝**: 기초 → 핵심 → 평가 → 데모. (`2-N`=불변 ID, 묶음=스텝)

**STEP 1 · 기초 — 신경망 원리 (forward → 학습 → 데이터 파이프)** — `01_basics.py`
- ✅ 2-0 환경·PyTorch 첫걸음 (설치·MPS 확인)
- ✅ 2-1 신경망 기초(뉴런·활성화) — "활성화 없으면 직선" 실증
- ✅ 2-2 학습 메커니즘(손실·역전파·Adam) — 5단계 루프
- ✅ 2-3 Dataset/DataLoader · batch 학습 골격 — make_moons 곡선 결정경계 + mini vs full 비교

**STEP 2 · 핵심 — 모델 구축 ⭐ (비전 + 시계열)** — `02_core.py`
- ✅ 2-4 CNN 기초 (Conv·Pooling) — FashionMNIST acc 0.87
- ✅ 2-5 전이학습 — **여러 백본 비교**(resnet18 0.94 vs mobilenet_v2 0.95), 토마토 잎 진단 ⭐
- ✅ 2-6 **Grad-CAM** (설명가능 AI — 어느 병반을 보고 판단했나) ⭐
- ✅ 2-8 LSTM — 환경 시계열 (내부온도 예측 MAE 1.2℃)

**STEP 3 · 평가 — 강건화·검증** — `03_eval.py`
- ✅ 2-7 과적합·불균형·학습안정 (클래스가중치 → 질병 recall 0.10→0.82)
- ✅ 2-9 평가 심화 (혼동행렬·ROC/AUC 0.97·오분류 FN 분석)

**STEP 4 · 데모 — 배포·마무리** — `04_demo.py` · `app/phase2_dl.py`
- ✅ 2-10 모델 저장(.pt) + Streamlit (사진 업로드 → 진단 + Grad-CAM)
- ⬜ 2-11 (고급/선택) YOLO 병반 위치 검출
- ✅ 2-12 회고 (코드 회고 완료 · 정식 `phase2_dl.md` 작성은 예정)
- 🏁 **Phase 2 끝 = 사진 올리면 진단+히트맵, 작물 1개씩 확장 가능한 파이프라인**

> 청크 상세 = `_local/concepts/DL_devlog.md` · 이론 = `DL.md`.

### Phase 3 — LLM (통합 처방 + 알림) · ⭐⭐⭐ ⚪ 예정
- ⬜ 3-1 Claude API 연동 (숫자·라벨 → 자연어 처방)
- ⬜ 3-2 재배지식 RAG (농사로 가이드 검색)
- ⬜ 3-3 통합 파이프라인 (ML/LSTM 예측 + CNN 진단 + RAG → 처방)
- ⬜ 3-4 알림봇(텔레그램) + Streamlit 대시보드
- 🏁 **Phase 3 끝 = 사진+센서 → 자연어 처방 + 알림**

---

## 📅 Phase별 데이터·기술 요약

| | Phase 1 (ML) ✅ | Phase 2 (DL) ▶ | Phase 3 (LLM) ⚪ |
|---|---|---|---|
| **목표** | 환경 → 작물 분류 | 잎 진단 + 환경 예측 | 말로 처방+알림 |
| **모달리티** | 정형 센서 | + 이미지 | + 언어 |
| **데이터** | 농진청 스마트팜 현장 농가 | AI Hub 153·PlantVillage·534 | 진단+예측+농사로 RAG |
| **핵심 기술** | scikit-learn·XGBoost | PyTorch·전이학습·Grad-CAM | Claude API·RAG |
| **결과물** | 분류 모델(F1 0.78) | 진단 모델(.pt) | 📱 처방 알림 |
| **배포** | (선택) Streamlit | Streamlit 진단 데모 | 대시보드+알림봇 |

> 데이터 출처 상세 → [data_sources.md](data_sources.md) · 데이터 맵 → [decisions.md 📦](decisions.md).
> 배포: 단계별 개별 Streamlit 앱(`app/phaseN_*.py`), 백엔드(`data/`·`models/`·`src/common/`) 공유.

---

## ✅ 결정됨 (→ [decisions.md](decisions.md))
- **작물:** 토마토 단일 시작 → 딸기·오이·참외 확장 (ML∩DL 교집합) · ADR-002
- **Phase 1 데이터:** 농진청 스마트팜 현장 농가 데이터 (옛 노지 Kaggle은 [ml-learn](https://github.com/luma200ok/smartfarm_ml_learn) v1로 분리) · ADR-001
- **Phase 2 병진단:** PlantVillage(즉시) → AI Hub 153(국내) · ADR-005
- **DL 프레임워크:** PyTorch (MPS) · **알림:** 텔레그램

## ✅ 다음 액션 (Phase 2 DL 시작)
1. 153 토마토 데이터(14GB) 다운 완료 → `data/images/` (외장하드)
2. 청크 2-1(신경망 기초)부터 → `src/dl/`
3. 전이학습으로 토마토 진단 파이프라인 완성 → 작물 확장
