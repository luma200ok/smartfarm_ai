# smartfarm_ai — 홈서버 이전 도커화(이슈 #68, smartfarm_service#27 파일럿 패턴 복제).
# api(FastAPI, 8000) · streamlit(8501) 공용 단일 이미지 — 실행 커맨드는 compose가 command로 구분한다.
# 기존 OCI 배포(deploy/deploy.sh, systemd 유닛)는 이 이미지와 무관하게 그대로 병행 유지된다.

FROM python:3.11-slim

WORKDIR /app

# ultralytics가 끌어오는 opencv-python(비-headless, requirements-deploy.txt에 직접 핀은 없음)이
# libGL·X11 계열 공유 라이브러리를 요구 — 실빌드로 `import cv2`가 libxcb.so.1 누락으로 실패하는 것을
# 확인 후 최소 세트만 추가(libgomp1은 xgboost/torch OpenMP 런타임용).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision CUDA 휠 방지 — x86_64(홈서버)에서 pip이 기본으로 고르는 CUDA 번들 대신
# CPU 전용 인덱스를 추가로 참조시킨다. arm64(예: 이 이미지의 로컬 빌드 검증)는 PyTorch CPU 인덱스에
# arm64 휠이 없어 PyPI 기본 인덱스로 폴백되는데, PyPI의 linux/aarch64 torch 휠도 CUDA 미포함이라 안전.
# ENV로 영구 고정하지 않고 이 RUN 한 줄에만 스코프(런타임 pip 호출에 영향 안 주기 위함).
COPY requirements-deploy.txt ./
RUN PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    pip install --no-cache-dir -r requirements-deploy.txt \
    # xgboost>=3.x가 GPU 유무와 무관하게 platform_system=='Linux'면 nvidia-nccl-cu12(약 340MB
    # CUDA 라이브러리)를 무조건 설치 요구(install_requires, extras 아님 — 실빌드로 확인).
    # 이 이미지는 CPU 전용 단일 노드 추론만 하므로 NCCL(다중 GPU 집단통신)이 불필요 — 설치 직후 제거.
    # xgboost/torch/torchvision import는 제거 후에도 정상 동작 확인됨(nccl은 optional runtime dep).
    && pip uninstall -y nvidia-nccl-cu12

# UID/GID 1000 고정 — 호스트 DATA_DIR(models/·data/) 바인드 마운트 소유권과 맞추기 위해
# (smartfarm_service backend/Dockerfile과 동일 이유). 이 베이스(Debian slim, python:3.11-slim)는
# Ubuntu 계열과 달리 기본 UID 1000 유저가 없어(확인됨: getent passwd 1000 → 없음) userdel 불필요.
# COPY --chown으로 소유권을 그때그때 지정하므로(아래) 여기서는 유저·그룹 생성만 한다(리뷰 P3-3 —
# 레이어마다 COPY --chown이 이미 개별 소유권을 정하므로 마지막에 /app 전체를 다시 훑는 chown -R은 불필요).
RUN groupadd --gid 1000 smartfarm \
    && useradd --uid 1000 --gid 1000 --no-create-home --shell /usr/sbin/nologin smartfarm

# models/·data/ 는 복사하지 않는다 — compose에서 호스트 DATA_DIR을 /app/models(:ro)·/app/data로
# 바인드 마운트한다(.dockerignore로도 제외, 마운트가 이 마운트포인트의 소유권을 덮어쓰므로 별도
# chown 불필요). .env 는 절대 복사하지 않는다 — app/api_client.py 등 여러 모듈이
# `load_dotenv(ROOT/".env", override=True)`를 호출하므로, 이미지에 .env가 들어가면 compose가
# 주입한 환경변수를 덮어써버린다(override=True). 파일이 없으면 dotenv는 조용히 no-op.
COPY --chown=smartfarm:smartfarm api ./api
COPY --chown=smartfarm:smartfarm app ./app
COPY --chown=smartfarm:smartfarm src ./src
COPY --chown=smartfarm:smartfarm db ./db
COPY --chown=smartfarm:smartfarm .streamlit ./.streamlit
RUN mkdir -p /app/models /app/data

USER smartfarm

EXPOSE 8000 8501

# ENTRYPOINT 없이 CMD만 — compose가 서비스별 command(uvicorn/streamlit)로 오버라이드한다.
# 기본값은 api(uvicorn), workers=1 고정 이유는 api/main.py 상단 주석(lru_cache 모델 캐시) 참고.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
