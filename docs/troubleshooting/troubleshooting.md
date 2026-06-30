# 🔧 트러블슈팅 내역 (smartfarm_ai)

> 프로젝트 진행 중 실제로 막혔던 문제 → 원인 → 해결을 정리한 기록.
> **ML · DL · OCI 배포** 3개 묶음으로 정리. (Phase 3 LLM은 아직 진행 전이라 기록 없음.)
> 근거: `fix` 커밋 / devlog / 배포 가이드(`docs/_local/deploy/oci-deployment.md`).
> 정리: 2026-06-30.

| # | 단계 | 한 줄 요약 | 근거 커밋 |
|---|---|---|---|
| A-1 | ML | 데이터 누수로 평가 점수 부풀림 → GroupKFold로 차단 | b84dbb1 |
| A-2 | ML | 베스트 모델 선정 기준이 누수 지표였음 → 정직화 | b84dbb1 |
| A-3 | ML 배포 | Streamlit `FileNotFoundError`(모델·csv 미반영) | 16bd186 |
| B-1 | DL | 토마토 분할 시 질병 val 0장 버그 | 1cc24c6 |
| B-2 | DL | Grad-CAM grad가 conv층까지 안 흐름 | 1cc24c6 |
| B-3 | DL | OOD 가드 도입(잎 아닌 사진 무조건 진단) | d2b8781, bde10fb |
| B-4 | DL | OOD 게이트(YOLO)가 병든 잎 오차단 → ImageNet로 교체 | 9e8505a |
| C-1 | 배포 | systemd 서비스명 충돌(Java 유닛 덮어씀) | 59cb6ec |
| C-2 | 배포 | Java EnvironmentFile 유실 | — |
| C-3 | 배포 | Python 3.9로 sklearn 설치 실패 | — |
| C-4 | 배포 | phik 빌드 실패(C++ 컴파일러 없음) | 59cb6ec |
| C-5 | 배포 | SELinux 203/EXEC — /home 실행 차단 | — |
| C-6 | 배포 | 리버스프록시 nginx 점유(Caddy 불가) | — |
| C-7 | 배포 | certbot command not found(secure_path) | — |
| C-8 | 배포 | matplotlib 직접 의존성 미명시 | 2a1a572 |

---

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

---

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

---

## 참고 / 범위 밖
- **Phase 3(LLM)**: 아직 진행 전이라 트러블슈팅 기록 없음.
- **데이터 가용성 제약(ADR-004)**: "환경→다작물 추천형 스마트팜 데이터가 사실상 없음(시설은 보통 단일작물)"을 발견해 ML 과제를 "추천"에서 "단일작물 적합도/생육 예측"으로 재정의 → 코드 버그가 아니라 **설계 결정**(`docs/decisions.md`).
- **macOS Accelerate BLAS 경고**(`RuntimeWarning: divide by zero in matmul`): numpy(Apple Accelerate)+Apple 칩의 알려진 헛 경고, 결과 정상이라 무시. 실제 수정 아님.
