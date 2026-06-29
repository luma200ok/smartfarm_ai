#!/usr/bin/env bash
# 서버(OCI) 배포 스크립트 — GitHub Actions가 SSH로 이걸 실행.
# 멱등: main 최신으로 맞추고 의존성/라벨/재시작/헬스체크까지.
set -euo pipefail

APP_DIR=/opt/smartfarm_ai
UV="$HOME/.local/bin/uv"            # 비대화형 SSH는 PATH가 좁음 → 절대경로
export PATH="$HOME/.local/bin:$PATH"

echo "▶ [1/5] git pull"
cd "$APP_DIR"
git fetch --quiet origin main
git reset --hard origin/main        # 서버 로컬 변경 없음 전제 → 깔끔히 main에 정렬

echo "▶ [2/5] deps 동기화 (requirements-deploy.txt)"
"$UV" pip install -q -r requirements-deploy.txt

echo "▶ [3/5] SELinux 라벨 (새 파일 대비)"
sudo restorecon -RF "$APP_DIR" || true

echo "▶ [4/5] 서비스 재시작"
sudo systemctl restart smartfarm-ai

echo "▶ [5/5] 헬스체크"
for i in $(seq 1 10); do
  if curl -fsS http://127.0.0.1:8501/_stcore/health >/dev/null; then
    echo "✅ 배포 성공 (health ok)"
    exit 0
  fi
  sleep 2
done
echo "❌ 헬스체크 실패 — journalctl -u smartfarm-ai -e 확인 필요"
exit 1
