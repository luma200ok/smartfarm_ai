# 🌱 SmartFarm AI — 작물 재배 도우미 (ML → DL → LLM)

> **센서는 환경 숫자를 보여주지만, 이 AI는 작물에 뭘 해줘야 할지를 알려준다.**
> 스마트팜 환경·잎 사진을 받아 **작물 분류 → 잎 병해충 진단 → 자연어 처방**까지 가는 멀티모달 AI.
> 작물 **토마토 단일로 시작 → 전이학습으로 다작물 확장** (딸기·오이·참외…).

[![Streamlit](https://img.shields.io/badge/Streamlit-라이브_데모-FF4B4B?logo=streamlit&logoColor=white)](https://smartfarm-ai.streamlit.app/)

---

## 📌 진행 단계

| Phase | 내용 | 기술 | 상태 |
|---|---|---|---|
| **1. ML** | 스마트팜 환경 → 작물 8종 분류 | RandomForest·XGBoost | ✅ 완료 |
| **2. DL** | 잎 사진 → 병해충 진단(CNN) + 환경 시계열(LSTM) | PyTorch·전이학습·Grad-CAM | ⏳ 진행 |
| **3. LLM** | 진단+환경 → 자연어 처방·알림 | Claude API·RAG | ⚪ 예정 |

문서: [PRD](docs/prd.md) · [로드맵](docs/roadmap.md) · [설계 결정(ADR)](docs/decisions.md) · [Phase1 ML](docs/phase1_ml.md) · [Phase2 DL](docs/phase2_dl.md)

---

## ✅ Phase 1 (ML) — 환경 기반 작물 분류

농촌진흥청 스마트팜 현장 농가 데이터(2022)로 **환경 센서 → 작물 8종 분류**.

- 833만 시간별 데이터 → **33,278 일별 집계**, 8작물(완숙토마토·방울토마토·딸기·오이·참외·파프리카·가지·국화)
- 모델 비교: **XGBoost 베스트** (test F1 0.78)
- 🔑 **핵심 교훈 — 데이터 누수:** 랜덤 분리 F1 **0.77** vs 농가 단위(GroupKFold) F1 **0.41**.
  → 같은 농가가 train·test에 섞이면 성능이 과대평가됨. **정직한 일반화 성능은 0.41**. (자세히 → [phase1_ml.md](docs/phase1_ml.md))

![혼동행렬](docs/figures/phase1_ml/confusion_matrix.png)

- 🚀 **라이브 데모:** https://smartfarm-ai.streamlit.app/ — 환경값 입력 → 작물 8종 예측 (4탭: 예측·작물 가이드·모델 평가·EDA)

---

## 🗂️ 구조

```
smartfarm-ai/
├── src/ml/        preprocess.py · train.py   (Phase 1)
├── src/dl/        (Phase 2 — 잎 진단 CNN, 예정)
├── app/           phase1_ml.py — Streamlit 데모 (배포 중)
├── data/          데이터 (git 제외 — 포털에서 재다운)
├── models/        학습 모델
└── docs/          PRD · 로드맵 · ADR · Phase 문서 · 그림
```

## 📊 데이터 출처

- **ML:** [농촌진흥청 스마트팜 현장 농가 데이터](https://www.data.go.kr/data/15108734/fileData.do) (공공데이터포털)
- **DL:** [AI Hub 시설작물 질병진단 이미지](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=153) · PlantVillage
- 데이터는 용량이 커서 git에 포함하지 않음 (위 출처에서 재다운로드)

## 🔧 실행

```bash
uv venv && uv pip install -r requirements.txt
python src/ml/preprocess.py   # 환경 데이터 → 일별 집계
python src/ml/train.py        # 모델 학습·평가·저장
```

---

## 🌿 관련 레포

- **[smartfarm_ml_learn](https://github.com/luma200ok/smartfarm_ml_learn)** — ML 입문 단계(노지 작물 추천, Kaggle Crop Recommendation). 이 프로젝트의 **출발점(v1)**으로, 범용 ML 학습 후 본 레포에서 스마트팜에 특화. (→ [ADR-001](docs/decisions.md))

---

© 2026 luma200ok(정재봉). 학습·포트폴리오 목적 프로젝트.
