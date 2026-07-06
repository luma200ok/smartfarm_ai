# 🔧 트러블슈팅 내역 (smartfarm_ai)

> 프로젝트 진행 중 실제로 막혔던 문제 → 원인 → 해결을 정리한 기록.
> **ML · DL · OCI 배포 · LLM** 4개 묶음으로 정리.
> 근거: `fix` 커밋 / devlog / 배포 가이드(`docs/_local/deploy/oci-deployment.md`).
> 정리: 2026-06-30(ML·DL·배포) · 2026-07-06(Phase 3 LLM 추가).

**바로가기:** [A. ML](#ml) · [B. DL](#dl) · [C. 배포·OCI](#deploy) · [D. LLM](#llm)

| # | 단계 | 한 줄 요약 | 근거 커밋 |
|---|---|---|---|
| A-1 | ML | 데이터 누수로 평가 점수 부풀림 → GroupKFold로 차단 | b84dbb1 |
| A-2 | ML | 베스트 모델 선정 기준이 누수 지표였음 → 정직화 | b84dbb1 |
| A-3 | ML 배포 | Streamlit `FileNotFoundError`(모델·csv 미반영) | 16bd186 |
| B-1 | DL | 토마토 분할 시 질병 val 0장 버그 | 1cc24c6 |
| B-2 | DL | Grad-CAM grad가 conv층까지 안 흐름 | 1cc24c6 |
| B-3 | DL | OOD 가드 도입(잎 아닌 사진 무조건 진단) | d2b8781, bde10fb |
| B-4 | DL | OOD 게이트(YOLO)가 병든 잎 오차단 → ImageNet로 교체 | 9e8505a |
| B-5 | DL | 정상에 부위 혼재 → 과육 오분류 → 부위 게이트(정제 재학습) | 4041656 |
| C-1 | 배포 | systemd 서비스명 충돌(Java 유닛 덮어씀) | 59cb6ec |
| C-2 | 배포 | Java EnvironmentFile 유실 | — |
| C-3 | 배포 | Python 3.9로 sklearn 설치 실패 | — |
| C-4 | 배포 | phik 빌드 실패(C++ 컴파일러 없음) | 59cb6ec |
| C-5 | 배포 | SELinux 203/EXEC — /home 실행 차단 | — |
| C-6 | 배포 | 리버스프록시 nginx 점유(Caddy 불가) | — |
| C-7 | 배포 | certbot command not found(secure_path) | — |
| C-8 | 배포 | matplotlib 직접 의존성 미명시 | 2a1a572 |
| C-9 | 배포 | 신규 모델(.pt) CI/CD 미반영 → 진단탭 크래시 | 3cdc30f |
| D-1 | LLM | 서버 처방 342초 → fast-path 1-call로 16.2초(-95%) | f8ad3d8 |
| D-2 | LLM | exaone function calling 약함 → fast-path로 tool 제거 | 63969e3 |
| D-3 | LLM | 관제 1차 구현이 리플레이 기반 → 오늘 운영 재설계 | 2662909 |
| D-4 | LLM | 장치 효과 안 보임(델타÷24) → 시간당 상수·관성 제어 | 6d62019 |
| D-5 | LLM | KMA 간헐 타임아웃 → 캐시·폴백·재시도 3회 | 7294a5b |
| D-6 | LLM | 오늘 차트 과거 구간 증발 → 스냅샷 누적·합성 | 2afb844 |
| D-7 | LLM | 습도 중앙 위 정체 → 밴드 중앙 히스테리시스 통일 | 1848550 |
| D-8 | LLM | 라이트 테마 글자·탭 색 실종 → 잉크색·color:inherit | 61006c6 |

---

<a id="ml"></a>

## A. Phase 1 — ML (환경 → 작물 분류)

### A-1. 데이터 누수로 평가 점수 부풀림 (3겹 평가 교훈)
- **증상**: 단순 8:2 분할·Stratified K-Fold에서는 F1이 높게(XGB SKF F1 0.673) 나오는데 일반화 성능과 괴리. 단년(2022) 데이터에선 0.77 → 0.41로 36%p나 떨어짐.
- **원인**: 단순 분할은 "같은 농가·작기의 다른 날짜" 데이터가 train/test 양쪽에 섞임 → 모델이 "어느 농가냐"로 작물을 외워버리는 데이터 누수. 식별자(농가·지역)를 피처에 넣으면 더 심해짐.
- **해결**:
  - 식별자를 입력 피처에서 제외.
  - **GroupKFold**(groups=연도+농가+작기)로 "같은 농가·작기는 한 폴드에만" 배치 → 누수 차단(현실적 F1 0.492).
  - 다년(2022~24) 결합으로 단년 대비 격차를 18%p까지 완화.
  - 클래스 불균형(최대 25배)으로 Accuracy를 못 믿어 **macro F1**을 주지표로 채택.
- **커밋**: `b84dbb1` · 근거: `ML_devlog.md`

### A-2. 베스트 모델 선정 기준이 누수 지표였음 (평가 정직화)
- **증상**: 베스트 모델을 누수된 test F1 기준으로 선정. 교차검증은 RF만 돌림. 배포용 pkl이 80%만 학습된 모델이었음.
- **원인**: test F1은 A-1의 누수가 반영된 낙관적 지표. 3모델을 동일 기준으로 비교하지 않았고, 배포 모델이 전체 데이터를 못 봄.
- **해결**:
  - 베스트 선정 기준을 GroupKFold F1로 변경.
  - 3모델 전부 SKF/GKF 교차검증 후 GroupKFold로 베스트 선정(XGB GKF 0.492 ≈ RF 0.496 통계적 동률 → 일관성상 XGB 고정).
  - 배포 pkl을 **전체 데이터로 refit** 후 저장.
  - 하드코딩 절대경로 → `__file__` 기준 상대경로.
  - 문서 헤드라인을 정직한 GroupKFold F1(0.41) 우선 표기.
- **커밋**: `b84dbb1`

### A-3. Streamlit 배포 시 `FileNotFoundError` (모델 미반영)
- **증상**: 배포된 앱이 모델 로드에 실패하며 `FileNotFoundError`.
- **원인**: 모델 pkl이 `.gitignore`의 `*.pkl` 패턴에 걸려 GitHub 미반영 → 배포 환경에 모델 없음. 추가로 앱이 런타임에 `env_daily.csv`에 의존했는데 배포 환경엔 그 csv가 없었음.
- **해결**:
  - `.gitignore`에 본편 모델명(`phase1_crop_env_clf.pkl`) 예외 추가 후 커밋.
  - train.py가 슬라이더 범위·작물별 통계를 pkl payload에 동봉.
  - 앱이 csv 런타임 의존을 제거하고 payload만으로 자립 동작.
- **커밋**: `16bd186`

---

<a id="dl"></a>

## B. Phase 2 — DL (잎 진단 CNN / LSTM)

### B-1. 토마토 데이터 분할 — 질병 val 0장 버그
- **증상**: 토마토 데이터 준비 시 질병 클래스의 validation 이미지가 0장이 됨.
- **원인**: 원천 데이터가 Validation 구조만 가진 형태였고, 데이터가 부족할 때 분할이 제대로 안 됨.
- **해결**: `prepare_tomato`에서 원천이 Validation만 있는 구조에 대응 + 데이터 부족 시 80/20 분할 로직 추가.
- **커밋**: `1cc24c6` (`src/dl/prepare_tomato.py`)

### B-2. Grad-CAM에서 grad가 conv층(layer4)까지 안 흐름
- **증상**: Grad-CAM 계산 시 중간 conv 층에 기울기가 안 쌓여 grad=None → CAM 계산 불가.
- **원인**: 전이학습에서 backbone을 freeze(`requires_grad=False`)해 forward는 되지만 역전파 시 conv층에 grad가 안 쌓임. 또한 중간 텐서의 grad는 기본적으로 버려짐.
- **해결**: 입력 텐서에 `requires_grad_(True)`로 grad 경로를 강제로 열고, hook 안에서 `retain_grad()`로 중간 활성화의 grad 보존.
- **커밋**: `1cc24c6` (`src/dl/02_core.py` 2-6) · 근거: `DL_devlog.md` 청크 2-6

### B-3. OOD(분포 밖 입력) 가드 도입 — 잎 아닌 사진 진단 차단
- **증상**: 진단 앱에 잎이 아닌 사진을 넣어도 무조건 진단 결과를 내놓음. 신뢰도가 낮아도 진단을 강행.
- **도입(2단계)**:
  - `d2b8781`: 신뢰도 < 70% 경고 + 토마토 잎 전용 안내(OOD 가드 1차).
  - `bde10fb`: 토마토잎 YOLO로 잎 박스 0개면 OOD로 판정하는 게이트 추가(YOLO+신뢰도 2단 방어). YOLO 모델 없으면 게이트 자동 스킵.
- ※ 이 두 건은 `fix`가 아니라 가드를 **도입한 feat**. 실제 결함은 B-4에서 드러남.

### B-4. OOD 게이트(YOLO)가 병든 잎을 오차단 → ImageNet 식물판별로 교체
- **증상**: B-3의 YOLO 게이트가 클로즈업 진단 입력과 분포가 맞지 않아 **실제 병든 잎을 OOD로 잘못 차단**(leaf_mold conf 0.19, tylcv 0.048로 잎 박스 미검출). 자체 3-클래스 분류기의 logit/energy도 잎↔OOD가 ~17% 겹쳐 판별 불가. MSP 0.70 경고도 정상 잎까지 오탐.
- **원인**: YOLO 객체탐지의 학습 분포(전체 잎 장면)와 진단용 클로즈업 입력의 분포 불일치. 자체 분류기의 logit/energy는 잎/OOD를 가를 분별력 부족.
- **해결**: 게이트를 **ImageNet 사전학습 resnet18의 식물·잎·채소 클래스 확률 합(plant_score)** 으로 교체(임계값 0.04). 실측 — 진짜 잎 9/9 통과, 합성 OOD(노이즈·단색·그라데) 4/4 차단. 추가 학습·설치 0(캐시 가중치 재사용). 오탐하던 MSP 0.70 경고 제거.
- **커밋**: `9e8505a` (`app/phase2_dl.py`)

### B-5. 정상 데이터에 부위 혼재 → 과육을 병으로 오진 → 부위 게이트
- **증상**: 토마토 과육(열매) 사진을 넣으면 YOLO는 미검출인데 Grad-CAM 분류기가 '병 의심' 결과를 냄. 잎이 아닌데 진단을 강행.
- **원인**: AI Hub 071 정상(0.정상) 원천에 잎뿐 아니라 과실·꽃·줄기가 섞여 있는데, 초기 `prepare_tomato`가 부위 라벨(area)을 안 읽고 전부 `normal`로 투입 → 닫힌 3분류기가 과육을 한 클래스로 강제 분류. ImageNet plant_score 게이트(B-4)는 '식물 여부'만 보므로 과육(식물)은 통과.
- **해결**: ① 진단 `normal`을 area=3(잎)만으로 정제 + 질병 Train 확보로 재학습(0.938→**0.97~0.99**, 서빙 resnet18 0.97·백본 best mobilenet_v2 0.987, MLflow 추적). ② 부위 분류기(과실/꽃/잎/줄기, **0.932**)를 잎 진단 앞단 게이트로 추가 — 잎이 아니면 진단 차단(식물→부위 2단 방어).
- **커밋**: `4041656` (`src/dl/prepare_tomato.py`·`02_core.py`·`app/phase2_dl.py`)

---

<a id="deploy"></a>

## C. 배포 (OCI / Streamlit)

> 환경: Oracle Linux 9 (aarch64, OCI Ampere A1), 멀티앱 공용 호스트.
> **상세 원본(단계별 명령 포함): `docs/_local/deploy/oci-deployment.md` §4 T1~T8.** 운영 FAQ는 `deploy/deploy_oci.md`.

### C-1. systemd 서비스명 충돌 — 기존 Java 백엔드 유닛을 덮어씀 ⚠️ (가장 위험)
- **증상**: `cp deploy/smartfarm.service /etc/systemd/system/` 했더니 기존 Java 백엔드 `smartfarm.service`(8084, 실시간 센서 수신)를 Streamlit 정의로 덮어씀. `journalctl -u smartfarm`에 Java 로그가 떠서 발견, `curl 8501` 거부.
- **원인**: systemd 유닛 이름 = `.service` 파일명. 레포 파일이 `smartfarm.service`라 기존 Java와 이름 충돌.
- **해결**: 실행 중 프로세스에서 원본 명령 추출(`/proc/<PID>/cmdline`·`cwd`·`environ`) → Java 유닛 재작성 → Streamlit을 `smartfarm-ai.service`로 분리. 레포 서비스 파일명도 영구 변경.
- **커밋**: `59cb6ec`

### C-2. Java 환경변수 누락 — EnvironmentFile 유실
- **증상**: C-1로 덮어쓴 Java 유닛에 환경변수(`DB_PASSWORD`, `REDIS_*`, `DISCORD_WEBHOOK_URL` 등) 누락.
- **원인**: 원본 유닛의 `EnvironmentFile=/etc/app-secrets/smartfarm.env` 라인이 덮어쓰며 사라짐.
- **해결**: `grep -rl DB_PASSWORD`로 env 파일 위치를 찾아 `EnvironmentFile=` 한 줄 복원. `daemon-reload`만 해서 운영 Java 무중단.

### C-3. Python 3.9로 sklearn 설치 실패
- **증상**: `uv venv`가 시스템 기본 Python 3.9 사용 → `scikit-learn==1.7.2` 설치/로딩 불가.
- **원인**: sklearn 1.7.x는 Python 3.10+ 요구, OL9 기본은 3.9.
- **해결**: dnf 시스템 파이썬 사용. OL9 dnf에 3.10이 없어(3.9/3.11/3.12) 3.11 채택. pickle 호환은 sklearn 버전(==1.7.2 고정)이 좌우하므로 3.10→3.11 영향 없음.

### C-4. pip 빌드 실패 — phik / ydata-profiling (C++ 컴파일러 없음)
- **증상**: `uv pip install -r requirements.txt` 중 `phik==0.12.5` 빌드 실패(`CMAKE_CXX_COMPILER not set`).
- **원인**: OL9 미니멀에 컴파일러 미설치. phik는 `ydata-profiling`(개발용 EDA 전용, 앱 미사용) 의존성.
- **해결**: 런타임 전용 `requirements-deploy.txt` 신설. dev/노트북/EDA(ydata-profiling·phik·jupyter·ipykernel·nbformat·seaborn) 제외 → aarch64 휠로 빌드 없이 설치.
- **커밋**: `59cb6ec`

### C-5. SELinux 203/EXEC — /home에서 systemd 실행 차단 (가장 오래 걸림)
- **증상**: `smartfarm-ai.service`가 `status=203/EXEC`로 무한 재시작. 바이너리는 존재·실행권한 정상.
- **원인**: SELinux Enforcing이 systemd(init_t)의 `/home`(user_home_t) 파일 exec/read를 차단(AVC: `denied { execute }`). `sudo -u opc` 수동 실행은 됨.
- **시도/실패**: venv를 시스템 파이썬 심볼릭으로 만듦 → 심볼릭 자체가 `/home`에 있어 여전히 차단.
- **해결**: 앱+venv를 `/home` 밖 `/opt/smartfarm_ai`로 이동 + `.venv`를 `bin_t`로 라벨(`semanage fcontext` + `restorecon`). `setenforce 0`은 타 앱 보안 영향으로 금지.

### C-6. 리버스프록시 — Caddy 아님, nginx 점유
- **증상**: 가이드대로 Caddy 설치하려 했으나 이미 nginx가 80/443 점유(다른 앱 서빙).
- **해결**: Caddy 취소. 기존 nginx에 server 블록 1개 추가 + certbot. Streamlit websocket 필수 → `proxy_set_header Upgrade/Connection`(기존 `$connection_upgrade`) + `proxy_read_timeout 86400`.

### C-7. certbot — command not found
- **증상**: `sudo certbot ...` → command not found(기존 인증서는 certbot 관리 중).
- **원인**: certbot이 `/usr/local/bin/certbot`인데 sudo의 secure_path에 `/usr/local/bin`이 없음.
- **해결**: 절대경로 `sudo /usr/local/bin/certbot --nginx -d ...`.

### C-8. matplotlib — 전이 의존성에 기대던 직접 import
- **증상**: `app/phase2_dl.py`가 `import matplotlib.cm`을 직접 쓰는데 `requirements-deploy.txt`엔 matplotlib이 없었음(당시 `ultralytics`가 끌고 와서 우연히 동작).
- **원인**: 직접 의존성을 명시하지 않고 전이 의존성에 기댐 → 전이 의존성이 깨지면 ImportError 위험.
- **해결**: `requirements-deploy.txt`에 `matplotlib` 명시 추가.
- **커밋**: `2a1a572`

### C-9. 신규 모델(.pt)이 CI/CD로 안 넘어가 진단 탭 크래시
- **증상**: 배포 서버에서 `FileNotFoundError: /opt/smartfarm_ai/models/tomato_part.pt` → phase2_dl 진단 탭 전체가 죽음.
- **원인**: CI/CD(`deploy.yml`→서버 `deploy.sh`)는 `git reset --hard origin/main`으로 **git 추적 파일만** 반영. 모델 `.pt`는 `.gitignore`(`*.pt`) 제외라 push해도 서버로 안 감 → 오늘 새로 만든 부위 게이트 모델(`tomato_part.pt`)이 미전송. 게다가 `predict_part`가 파일 부재를 가드하지 않아 하드 크래시.
- **해결**: ① 코드 — `predict_part`가 `PART_CKPT` 없으면 `leaf`로 폴백(앞단 plant_score 게이트가 비잎 1차 차단) → 앱 비중단. ② 모델 — 수동 배포 스크립트 `deploy/push_models.sh`(rsync `.pt` 3종 + 서비스 재시작) 신설, 배포 문서 rsync에 `tomato_part.pt` 추가.
- **교훈**: **코드=CD 자동 / 모델=수동 rsync** 로 경로가 갈린다. 모델 바꾸면 `bash deploy/push_models.sh` 필수.
- **커밋**: `3cdc30f`(가드)·`ca8d727`(스크립트)

---

<a id="llm"></a>

## D. Phase 3 — LLM (처방 · 관제 · 알림)

> 환경: Ollama 로컬(qwen2.5:14b) + OCI 서버(GPU 없는 3코어 ARM CPU·16GB). 처방=LLM, 환경 관제=규칙 기반으로 분업.

### D-1. 서버 처방 342초 → fast-path 1-call로 16.2초(-95%)
- **증상**: OCI 서버에서 처방 1건에 **342.6초** — UX 불성립. 로컬 웜도 28초로 김.
- **원인**: agentic tool calling 경로가 tool 라운드를 2회 돌며 매번 생성, 콜드 시 모델 재적재까지 겹침. GPU 없는 CPU에선 이 낭비가 치명적.
- **해결**: ① 로컬 웜 — 생성 `num_predict` 캡 + `keep_alive` 30m(콜드 재적재 제거) + 프롬프트 다이어트(-17~20%)로 **28→17~19초(-35%, 이슈 #15)**. ② 서버 — 진단·RAG·예보를 **코드가 직접 실행**하고 LLM은 최종 JSON 처방 1-call만(fast-path). **342.6→16.2초(-95%, 이슈 #18)**. 근거 주입·환각 방어는 그대로 유지.
- **커밋**: `6cbb93a`(#15, PR #16) · `f8ad3d8`·`63969e3`(#18, PR #19)

### D-2. exaone3.5:2.4b의 function calling이 약함
- **증상**: 서버용 소형 모델 후보 exaone3.5:2.4b가 한국어 처방 품질·속도는 좋은데 tool(function) 선택 신뢰도가 낮음.
- **원인**: 소형 모델의 구조적 한계 — tool calling 판단이 불안정.
- **해결**: 모델 교체가 아니라 **아키텍처로 무효화**. D-1의 fast-path(1-call)가 tool calling 자체를 없애므로, exaone은 "작성 전용(writer, `OLLAMA_WRITER_MODEL`)"으로만 채택. tool calling 경로(로컬·CLI)는 qwen2.5로 분리 유지.
- **커밋**: `63969e3`(#18, PR #19)

### D-3. 환경 관제 1차 구현이 의도와 어긋남
- **증상**: 관제 첫 구현이 2024 과거 데이터 **리플레이** 기반이라 "오늘 날짜 실운영" 의도와 불일치.
- **원인**: 스펙 해석 차이 — 과거 재생 시뮬 vs 오늘 운영.
- **해결**: 스펙 재확인 후 **"오늘 운영 모드"로 재설계** — KMA 예보 → 오늘 기준선 → 장치 4종 제어 → 매시 알림.
- **커밋**: `2662909`·`2339228`(#23, PR #24)

### D-4. 장치 제어 효과가 차트에 안 나타남
- **증상**: 장치를 켜도 값 변화가 미미해 제어 반응이 무의미하게 보임.
- **원인**: 1일 델타를 24로 나눠 **시간당 -0.2%p** 수준으로만 반영 → 관측 불가.
- **해결**: 시간당 상수(±2℃/±8%p) + 관성(누적) 제어 + 밴드 관통 방지 클램프. **채터링 9회→3회 이하**로 실증.
- **커밋**: `6d62019`·`8b0a5bf`(#27, PR #28)

### D-5. KMA 예보 API 간헐 타임아웃(OCI)
- **증상**: 10초 초과 응답이 2연속 발생하며 관제가 예보 없이 멈춤.
- **원인**: 외부 KMA API 지연 + 재시도·캐시 부재.
- **해결**: 60초 캐시(`st.cache_data`) + 직전 성공 데이터 폴백(**stale-while-error**) + 재시도 3회 백오프.
- **커밋**: `7294a5b`·`190a21a`(#29)

### D-6. 오늘 차트의 과거 구간이 사라짐
- **증상**: 오늘 0시~현재 실측 구간이 차트에서 비어 보임.
- **원인**: KMA 예보는 **발표분 이후(미래)만 제공** + 시간별 기록을 안 쌓고 있었음.
- **해결**: 매시 systemd 타이머로 **스냅샷 누적** → 과거=기록·미래=예보 합성으로 0~24시 연속 차트.
- **커밋**: `e0a2b4c`·`2afb844`(#40, PR #41)

### D-7. 습도가 밴드 중앙 위에 정체
- **증상**: 여름 고습 지속 시 습도가 밴드 상단(74~85%)에 갇혀 중앙으로 안 내려옴(온도는 겨울 하한 정체로 대칭).
- **원인**: P-제어 수식이 아니라 **ON 진입 기준이 "밴드 밖에서만 ON"(밴드 경계)** 이라 한쪽 외란에서 상단 고정.
- **해결**: ON/OFF를 **밴드 중앙(mid) 기준 히스테리시스**로 온·습도 통일 → 여름=위에서·겨울=아래에서 중앙 수렴(**여름 71%·겨울 69%**).
- **커밋**: `331bf08`(#33)·`1848550`(#51)

### D-8. 라이트 테마에서 글자·탭 색 실종
- **증상**: 다크→라이트 토글 시 일부 텍스트·탭 라벨 색이 안 보임.
- **원인**: 다크 base textColor 잔존 + 오버라이드가 탭 라벨에 직접 매치돼 캐스케이드 깨짐.
- **해결**: 네이티브 텍스트 잉크색 오버라이드 + 탭 `color:inherit`로 캐스케이드 복원.
- **커밋**: `61006c6`(#48)

---

## 참고 / 범위 밖
- **데이터 가용성 제약(ADR-004)**: "환경→다작물 추천형 스마트팜 데이터가 사실상 없음(시설은 보통 단일작물)"을 발견해 ML 과제를 "추천"에서 "단일작물 적합도/생육 예측"으로 재정의 → 코드 버그가 아니라 **설계 결정**(`docs/decisions.md`).
- **macOS Accelerate BLAS 경고**(`RuntimeWarning: divide by zero in matmul`): numpy(Apple Accelerate)+Apple 칩의 알려진 헛 경고, 결과 정상이라 무시. 실제 수정 아님.
