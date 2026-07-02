#!/usr/bin/env bash
# 서버(OCI) 배포 스크립트 — GitHub Actions가 SSH로 이걸 실행.
# 멱등: main 최신으로 맞추고 의존성/라벨/재시작/헬스체크까지.
set -euo pipefail

APP_DIR=/opt/smartfarm_ai
UV="$HOME/.local/bin/uv"            # 비대화형 SSH는 PATH가 좁음 → 절대경로
export PATH="$HOME/.local/bin:$PATH"

echo "▶ [1/6] git pull"
cd "$APP_DIR"
git fetch --quiet origin main
git reset --hard origin/main        # 서버 로컬 변경 없음 전제 → 깔끔히 main에 정렬

echo "▶ [2/6] deps 동기화 (requirements-deploy.txt)"
"$UV" pip install -q -r requirements-deploy.txt

echo "▶ [3/6] PostgreSQL 스키마·RAG sync (DATABASE_URL 설정 시만, 실패해도 배포 계속)"
if [ -f .env ] && grep -q '^DATABASE_URL=.\+' .env; then
  set -a; source .env; set +a
  psql "$DATABASE_URL" -f db/schema.sql || echo "⚠️ 스키마 적용 실패 — 계속 진행"
  PYTHONPATH="$APP_DIR/src" "$UV" run python -m llm.rag.sync || echo "⚠️ RAG sync 실패 — 계속 진행"
else
  echo "   DATABASE_URL 미설정 — 스킵(memory 백엔드로 계속)"
fi

echo "▶ [4/6] SELinux 라벨 (새 파일 대비)"
sudo restorecon -RF "$APP_DIR" || true

echo "▶ [5/6] 서비스 재시작"
sudo systemctl restart smartfarm-ai

echo "▶ [6/6] 헬스체크"
for i in $(seq 1 10); do
  if curl -fsS http://127.0.0.1:8501/_stcore/health >/dev/null; then
    echo "✅ 배포 성공 (health ok)"
    exit 0
  fi
  sleep 2
done
echo "❌ 헬스체크 실패 — journalctl -u smartfarm-ai -e 확인 필요"
exit 1
