#!/usr/bin/env bash
# 서버(OCI) 배포 스크립트 — GitHub Actions가 SSH로 이걸 실행.
# 멱등: main 최신으로 맞추고 의존성/라벨/재시작/헬스체크까지.
set -euo pipefail

# ⚠️ 전체를 main()으로 감싼다 — [1/N]의 git reset이 이 스크립트 자신을 교체하는데, bash는 스크립트를
# 증분 읽기 하므로 함수로 안 감싸면 "실행 중 run은 구버전+신버전이 섞여" 실행된다(이슈 #59 후속:
# PR #63/#64 CD가 직전 버전 스크립트로 돌아 red 2회). 함수 정의는 전체 파싱 후 마지막 줄에서 호출.
main() {

  APP_DIR=/opt/smartfarm_ai
  UV="$HOME/.local/bin/uv"            # 비대화형 SSH는 PATH가 좁음 → 절대경로
  export PATH="$HOME/.local/bin:$PATH"

  echo "▶ [1/7] git pull"
  cd "$APP_DIR"
  git fetch --quiet origin main
  git reset --hard origin/main        # 서버 로컬 변경 없음 전제 → 깔끔히 main에 정렬

  echo "▶ [2/7] deps 동기화 (requirements-deploy.txt)"
  "$UV" pip install -q -r requirements-deploy.txt

  echo "▶ [3/7] PostgreSQL 스키마·RAG sync (DATABASE_URL 설정 시만, 실패해도 배포 계속)"
  # source 대신 값만 추출 — 비밀번호에 $/백틱이 있어도 셸 확장·명령 실행 없이 안전
  DATABASE_URL="$(grep -m1 '^DATABASE_URL=' "$APP_DIR/.env" 2>/dev/null | cut -d= -f2- || true)"
  if [ -n "$DATABASE_URL" ]; then
    psql "$DATABASE_URL" -f db/schema.sql || echo "⚠️ 스키마 적용 실패 — 계속 진행"
    # sync는 db.py가 자체적으로 .env를 load_dotenv 하므로 env 전달 불필요
    PYTHONPATH="$APP_DIR/src" "$UV" run python -m llm.rag.sync || echo "⚠️ RAG sync 실패 — 계속 진행"
  else
    echo "   DATABASE_URL 미설정 — 스킵(memory 백엔드로 계속)"
  fi

  echo "▶ [4/7] SELinux 라벨 (새 파일 대비)"
  sudo restorecon -RF "$APP_DIR" || true

  echo "▶ [5/7] 서비스 재시작 — smartfarm-ai(Streamlit)"
  sudo systemctl restart smartfarm-ai

  echo "▶ [6/7] 서비스 재시작 — smartfarm-api(FastAPI 서빙, 이슈 #59)"
  # 서버에 smartfarm-api.service 가 아직 설치 안 됐을 수 있음(PR 4 머지 직후 최초 배포 전).
  # 여기서 곧장 exit 1 하면 안 됨 — 이미 [5/7]에서 재시작된 smartfarm-ai(Streamlit)의 헬스체크가
  # [7/7]에서 통째로 스킵돼 "새 코드로 떠 있는데 무점검 방치"가 된다(code-reviewer P1).
  # 그래서 여기서는 플래그만 남기고 계속 진행 → [7/7]에서 smartfarm-ai는 항상 검증하고,
  # API_INSTALLED=0 이면 최종 결과는 그대로 실패 처리(설치 강제 의도는 보존).
  API_INSTALLED=1
  # ※ `systemctl list-unit-files | grep -q`는 set -o pipefail 하에서 grep 조기 종료 → systemctl SIGPIPE(141)로
  #    파이프라인이 비0이 되는 레이스가 있어(서버 실측 30/30 재현) 설치돼 있어도 미설치로 오판한다.
  #    파이프 없는 systemctl cat(무권한 동작 확인)으로 존재 판정.
  if systemctl cat smartfarm-api.service >/dev/null 2>&1; then
    sudo systemctl restart smartfarm-api
  else
    API_INSTALLED=0
    echo "❌ smartfarm-api.service 미설치 — deploy/deploy_oci.md '9. 서빙 API(이슈 #59)' 섹션대로 서버에 먼저 설치·enable 하세요(smartfarm-ai 헬스체크는 계속 진행)"
  fi

  echo "▶ [7/7] 헬스체크"
  # 재시도 포함(기존 /_stcore/health 패턴 미러)
  check_health() {
    local url="$1" label="$2"
    for i in $(seq 1 10); do
      if curl -fsS "$url" >/dev/null; then
        echo "✅ $label health ok"
        return 0
      fi
      sleep 2
    done
    echo "❌ $label 헬스체크 실패 — journalctl -u $label -e 확인 필요"
    return 1
  }

  HEALTH_FAILED=0
  # smartfarm-ai는 API 설치 여부와 무관하게 항상 검증(이미 restart된 걸 무점검 방치하지 않음)
  check_health "http://127.0.0.1:8501/_stcore/health" smartfarm-ai || HEALTH_FAILED=1
  if [ "$API_INSTALLED" = "1" ]; then
    check_health "http://127.0.0.1:8000/api/health" smartfarm-api || HEALTH_FAILED=1
  else
    # 미설치 상태를 실패로 집계 — exit 0으로 새지 않게 해서 설치 강제 의도를 유지(CD 실패 가시화)
    HEALTH_FAILED=1
  fi

  # 종료 코드: smartfarm-ai·smartfarm-api(설치+헬스) 모두 정상이면 0 / 그 외(둘 중 헬스 실패, 또는
  # smartfarm-api 미설치 자체)면 1 — 두 경로 모두 아래 한 줄로 귀결된다.
  if [ "$HEALTH_FAILED" = "1" ]; then
    exit 1
  fi
  echo "✅ 배포 성공 (health ok)"
  exit 0
}

main "$@"
