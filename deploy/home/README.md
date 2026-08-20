# 홈서버 이전 도커화(이슈 #68) — 1회 세팅 체크리스트

smartfarm_service#27 파일럿 패턴을 그대로 복제했다. 이 문서는 홈서버(x86, self-hosted runner)에서
**메타(A)가 직접** 실행하는 1회 세팅 절차다. 기존 OCI(arm1) 배포(`deploy/deploy.sh`,
`deploy/*.service`, `.github/workflows/deploy.yml`)는 **이 작업과 무관하게 그대로 병행 유지**된다 —
어떤 기존 파일도 수정하지 않았다.

참조: smartfarm_service 레포 `deploy/home/README.md`(#27·#28·#29 리뷰 완료 패턴), 이 레포
`deploy/deploy_oci.md`(환경변수 의미), `docs/api-contract.md`가 있다면 함께 참고.

## 0. 전제

- 홈서버는 x86_64, Docker + docker compose plugin 설치돼 있음.
- Ollama가 호스트(또는 별도 관리되는 프로세스)에서 이미 구동 중 — 이 스택은 Ollama 컨테이너를
  새로 만들지 않고 `host.docker.internal`(compose의 `extra_hosts: host-gateway`)로만 접근한다.
- smartfarm_service backend가 이미 `shared-net`을 쓰고 있다면(#27 파일럿) 그 네트워크를 그대로
  재사용한다 — 이 스택의 `api` 서비스가 컨테이너명 `smartfarm-ai`로 그 네트워크에 조인해야
  backend의 `AI_SERVER_URL=http://smartfarm-ai:8000` 호출이 성립한다(이름 계약, 변경 금지).
- PostgreSQL(pgvector)을 쓸 경우 `db-postgres` 컨테이너도 `shared-net`에 있어야 한다(선택 기능,
  `DATABASE_URL` 미설정이면 RAG는 npz 폴백으로 동작).

## 1. 공유 네트워크 확인/생성

```
docker network create shared-net   # 이미 있으면 생략(smartfarm_service#27에서 먼저 만들었을 수 있음)
```

- [ ] backend(smartfarm_service)가 이미 `shared-net`에 조인돼 있는지 확인(`docker network inspect shared-net`)
- [ ] `DATABASE_URL`을 쓸 경우 `db-postgres`도 `shared-net` 조인 확인
- [ ] **pgvector 모드(`RAG_BACKEND=pgvector`) 사용 시** 최초 1회만 스키마 적용(멱등, 반복 적용 가능):

      ```
      docker exec -i db-postgres psql -U smartfarm -d smartfarm_ai < db/schema.sql
      ```

      참고: 현재 홈서버 `db-postgres`는 OCI 백업 복원본이라 이 스키마가 이미 적용돼 있을 가능성이 높다 —
      멱등(`CREATE ... IF NOT EXISTS`)이므로 재실행해도 안전하며, 스킵해도 무방하다(존재 여부만 확인).

## 2. `.env` 작성

```
mkdir -p /home/jb/srv/smartfarm-ai
cp deploy/home/.env.example /home/jb/srv/smartfarm-ai/.env
# 편집기로 실제 값 채우기(KMA_SERVICE_KEY, DATABASE_URL, DISCORD_WEBHOOK_URL 등)
chmod 700 /home/jb/srv/smartfarm-ai
chmod 600 /home/jb/srv/smartfarm-ai/.env
```

- [ ] **소유·접근 모델**: `.env`는 관리자 `jb` 소유 600 + 디렉터리 700 — `runner` 계정은 접근 불가하고,
      배포 시엔 sudo 헬퍼(§4)가 root 권한으로 읽어 `--env-file`로 compose에 공급한다.
- [ ] `OLLAMA_TIMEOUT`은 빈 문자열로 두지 말 것 — `src/llm/prescribe.py`가
      `float(os.getenv("OLLAMA_TIMEOUT", "180"))`로 파싱해 빈 문자열이면 기동 자체가 죽는다
      (미설정=변수 자체를 없애는 게 아니라 compose가 `${OLLAMA_TIMEOUT}`을 빈 문자열로 주입하기
      때문 — 반드시 숫자 값을 채운다. `.env.example` 기본값 180 유지 권장).
- [ ] 절대 `.env`를 git에 커밋하지 말 것(루트 `.gitignore`가 `.env`/`.env.*`를 이미 차단, `.env.example`만 예외)

## 3. 모델·데이터 디렉터리 (rsync)

```
mkdir -p /home/jb/srv/smartfarm-ai/models /home/jb/srv/smartfarm-ai/data
sudo chown -R 1000:1000 /home/jb/srv/smartfarm-ai/models /home/jb/srv/smartfarm-ai/data
```

- [ ] `chown` UID/GID 1000은 `Dockerfile`의 비루트 유저(smartfarm)와 고정 매칭된다 — 컨테이너 내부에서
      읽기(모델)·쓰기(데이터) 권한이 필요하므로 반드시 먼저 실행. 검증: `docker run --rm <api 이미지> id -u` → `1000`
- [ ] 모델·데이터 전송은 기존 `deploy/push_models.sh`를 참고해 대상만 홈서버로 바꾼 절차를 쓴다
      (OCI arm1과 별도 서버이므로 각자 독립적으로 채워야 함):

      ```
      rsync -avz \
        models/tomato_resnet18.pt models/tomato_yolov8n.pt models/tomato_part.pt \
        models/env_lstm.pt models/env_expect_reg.pkl \
        jb@<홈서버>:/home/jb/srv/smartfarm-ai/models/
      rsync -avz data/processed/env_daily.csv jb@<홈서버>:/home/jb/srv/smartfarm-ai/data/processed/
      ```
- [ ] 디스크 여유 공간 확인(torch/ultralytics 이미지 자체가 ~1GB 내외, 모델 파일은 별도)

## 4. GitHub self-hosted runner 설치 (라벨 `home`)

smartfarm_service#27과 **동일 홈서버**에서 같은 러너를 공유하는 것을 전제로 한다(러너 자체를
새로 설치할 필요는 보통 없음 — 이미 `home` 라벨 러너가 있다면 그대로 재사용). 이 레포용으로
추가로 필요한 것은 §의 root clone·헬퍼·sudoers뿐이다.

- [ ] (러너가 아직 없다면) 레포 Settings → Actions → Runners → New self-hosted runner, 라벨에 `home` 추가.
      전용 `runner` 계정으로 설치(`docker` 그룹에 넣지 않음) — smartfarm_service#27과 동일 원칙.
- [ ] **배포 입력용 레포 최초 1회 root clone** — 헬퍼는 러너 워크스페이스가 아닌 root 소유 사본에서만
      compose를 실행한다:

      ```
      sudo mkdir -p /opt/smartfarm
      sudo git clone https://github.com/<org>/smartfarm_ai.git /opt/smartfarm/repo-ai
      ```
- [ ] **헬퍼 설치**(레포 버전관리본 `deploy/home/smartfarm-ai-deploy.sh` — 갱신 시에도 동일 명령 재실행):

      ```
      sudo install -o root -g root -m 755 deploy/home/smartfarm-ai-deploy.sh /usr/local/bin/smartfarm-ai-deploy
      ```
- [ ] 배포 실행 권한은 sudo 헬퍼 **1개만, 인자까지 고정**해 허용 — sudoers(`visudo -f /etc/sudoers.d/smartfarm-ai-deploy`):

      ```
      runner ALL=(root) NOPASSWD: /usr/local/bin/smartfarm-ai-deploy up, /usr/local/bin/smartfarm-ai-deploy ps, /usr/local/bin/smartfarm-ai-deploy prune, /usr/local/bin/smartfarm-ai-deploy diagnose
      ```
- [ ] 이 워크플로우는 **`pull_request` 트리거를 절대 사용하지 않는다**(fork PR RCE 방지) —
      `.github/workflows/deploy-home.yml`은 `workflow_dispatch`만 사용하고 main ref 가드를 건다.

## 5. 외부 노출

이 파일럿은 로컬(127.0.0.1) 바인딩만 한다 — 외부 노출(터널 등)은 별도 결정 사항이며 이 PR
범위 밖이다. smartfarm_service backend가 `AI_SERVER_URL=http://smartfarm-ai:8000`로 컨테이너
내부에서 직접 호출하므로, backend만 정상 동작한다면 이 API를 별도로 인터넷에 노출할 필요는
당장 없다.

## 6. 스모크 절차

배포는 항상 헬퍼 경유로 실행한다(§4의 배포 경로와 동일 — origin/main 기준 빌드·기동):

```
sudo /usr/local/bin/smartfarm-ai-deploy up
```

- [ ] `sudo /usr/local/bin/smartfarm-ai-deploy ps` — api healthy, streamlit running 확인
- [ ] `curl -s http://127.0.0.1:8000/api/health` → `{"status":"ok",...}` (로컬에서만 접근 가능)
- [ ] `curl -I http://127.0.0.1:8501/` → 200 (streamlit)
- [ ] smartfarm_service backend 쪽에서 진단/처방 API 호출이 `smartfarm-ai:8000`으로 정상 도달하는지 확인
- [ ] Ollama 연동 확인: `/api/health` 응답의 `ollama.online`이 `true`인지(호스트 Ollama가 떠 있어야 함)

## 7. 롤백 (수동, 관리자)

`prune`은 `--filter until=72h`라 **직전 배포의 dangling 이미지가 3일간 보존**된다.

```
sudo docker images --filter dangling=true          # 직전 api/streamlit 이미지 ID 확인(CREATED 시각으로 식별)
sudo docker tag <직전 api 이미지ID> smartfarm-ai-home-api:latest
sudo docker tag <직전 streamlit 이미지ID> smartfarm-ai-home-streamlit:latest
sudo docker compose -f /opt/smartfarm/repo-ai/deploy/home/compose.yml --env-file /home/jb/srv/smartfarm-ai/.env up -d --no-build
```

이미지 이름은 compose 프로젝트명(`name: smartfarm-ai-home`) 기반 `smartfarm-ai-home-{서비스}`로 고정된다.
롤백 후에도 §6 스모크를 반복해 확인한다.

## 8. 알려진 제약 / 후속

- `requirements-deploy.txt`에 `ollama`·`python-dotenv`가 누락돼 있던 것을 이 작업(도커 빌드 검증) 중
  발견해 추가했다 — `api/routers/prescriptions.py` 임포트 체인이 이 두 패키지 없이는 기동 자체가
  불가능했다(기존 OCI 운영 venv는 최초 `requirements.txt` 전체 설치 이력이 남아있어 이 누락이
  가려져 있었을 가능성이 있다 — 신선한 이미지 빌드에서만 드러남).
- 외부 노출(Cloudflare Tunnel 등)은 범위 밖 — 필요해지면 smartfarm_service#27 패턴(cloudflared
  컨테이너 추가) 참고해 별도 PR로.
- `docker compose logs`는 stdout 기반이라 로그 보존 정책은 후속 검토 필요(예: `logging: driver:
  json-file, options: max-size`).
