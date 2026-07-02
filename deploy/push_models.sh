#!/usr/bin/env bash
# 모델 .pt 를 OCI 서버로 전송 + 서비스 재시작.
# CD(deploy.sh)는 git 추적 파일만 반영하고 .pt 는 .gitignore 제외라 안 옮긴다 → 모델은 이 스크립트로 수동 배포.
# 사용: bash deploy/push_models.sh [ssh_host]   (기본 host=oci-arm1)
set -euo pipefail

HOST="${1:-oci-arm1}"
DEST="/opt/smartfarm_ai/models/"

cd "$(dirname "$0")/.."   # 레포 루트로 이동(어디서 실행해도 models/ 경로 일치)

echo "▶ 모델 전송 → ${HOST}:${DEST}"
rsync -avz \
  models/tomato_resnet18.pt \
  models/tomato_yolov8n.pt \
  models/tomato_part.pt \
  models/env_lstm.pt \
  models/env_expect_reg.pkl \
  "${HOST}:${DEST}"

# 가상센서·LSTM 예측용 환경 시계열(git 제외) — 없으면 Phase3 코치·경보 섹션이 비활성
echo "▶ 환경 데이터 전송 → ${HOST}:/opt/smartfarm_ai/data/processed/"
ssh "${HOST}" 'mkdir -p /opt/smartfarm_ai/data/processed'
rsync -avz data/processed/env_daily.csv "${HOST}:/opt/smartfarm_ai/data/processed/"

echo "▶ 서비스 재시작"
ssh "${HOST}" 'sudo systemctl restart smartfarm-ai'

echo "✅ 모델 전송 + 재시작 완료"
