# 📦 데이터 출처 기록 (Data Sources)

> 단계별 데이터셋·출처를 한곳에 기록. 원본은 git에 안 올림(`.gitignore`) → 여기 링크로 재다운.
> 데이터 전략·작물 매핑 상세 → [decisions.md 📦 데이터 소스 맵](decisions.md) · ← [README 허브](../README.md)

---

## Phase 1 — ML (스마트팜 환경 → 작물 분류) ✅

| 항목 | 내용 |
|---|---|
| **데이터셋** | 농촌진흥청 스마트팜 현장 농가 데이터 |
| **출처(URL)** | https://www.data.go.kr/data/15108734/fileData.do (공공데이터포털) |
| **내용** | 시설채소 농가의 **환경·생육·생산·재배정보** (8작물: 완숙/방울토마토·딸기·오이·참외·파프리카·가지·국화, 2020~2024) |
| **사용 범위** | 2022 환경(833만 시간별) → **일별 집계 33,278행** → 작물 8종 분류 |
| **저장 위치** | `data/nongjincheong/` (원본, git 제외) · 집계본 `data/processed/env_daily.csv` |
| **받는 법** | 공공데이터포털 로그인 → CSV(zip) 다운 → 압축 해제 |
| **라이선스** | 공공데이터(이용허락범위 확인) · 학습/포트폴리오용 |
| **확장 후보** | 다년치(2020~2024) 결합 · AI Hub 지능형 스마트팜 통합데이터(토마토 534) 환경 시계열 |

> 📌 옛 노지 데이터(Kaggle Crop Recommendation)는 **입문 단계(v1)** 로 분리 → [smartfarm-ml-learn](https://github.com/luma200ok/smartfarm-ml-learn). (→ [ADR-001](decisions.md))

---

## Phase 2 — DL (잎 병해충 진단 + 환경 시계열) ⏳

| 항목 | 내용 |
|---|---|
| **비전(병진단) ①** | **PlantVillage 토마토** (즉시·검증·라벨 확실) — https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset (토마토 정상+병 9종) |
| **비전(병진단) ②** | **AI Hub 시설작물 질병진단**(국내) — https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=153 (12작물·토마토 잎곰팡이병/황화잎말이+정상, **신청·승인**) |
| ⚠️ **주의** | AI Hub **통합데이터(534)** 이미지는 **생육용(병명 없음)** → 병 진단엔 **못 씀**. 받을 땐 토마토만(원천 14GB) |
| **시계열(환경)** | AI Hub 통합데이터(534) 환경 시계열 / 농진청 환경 데이터 |
| **대상 작물** | **토마토 단일 시작 → 딸기·오이·참외 확장** (ML∩DL 데이터 교집합, → [ADR-002](decisions.md)) |

> 전략: PlantVillage로 파이프라인 완성 → AI Hub 153(국내)로 교체·고도화 (전이학습). 데이터 맵 → [decisions.md](decisions.md).

---

## Phase 3 — LLM (RAG 코퍼스) ⚪

| 항목 | 내용 |
|---|---|
| **재배 가이드** | 농사로(농촌진흥청) — https://www.nongsaro.go.kr (작물별 재배·방제 매뉴얼) |
| **용도** | LLM 처방 생성 시 RAG 지식베이스 |

---

## 🔑 다운로드 메모

- **농진청(Phase1)**: 공공데이터포털 로그인 후 수동 다운 (API 키 불필요)
- **PlantVillage(Phase2)**: Kaggle 로그인 → 다운, 또는 GitHub 공개 미러(`spMohanty/PlantVillage-Dataset`) 토마토만
- **AI Hub 153(Phase2)**: 한국 국적 회원 신청 → 승인 후 다운 (Training/Validation + 원천, **토마토만** 선택)
```
