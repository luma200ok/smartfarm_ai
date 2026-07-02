# OCI A1 배포 가이드 — SmartFarm AI (멀티페이지)

VM(OCI Ampere A1, Ubuntu)이 **이미 생성·실행 중**이고 **도메인 보유** 전제.
구성: `streamlit(127.0.0.1:8501)` ← `Caddy(80/443, 자동 HTTPS)` ← 인터넷.

> 경로는 사용자 `ubuntu`, 레포 위치 `/home/ubuntu/smartfarm_ai` 기준. 다르면 치환.

---

## 1. 방화벽 2겹 열기 (80·443만, 8501은 내부 전용)

### (a) OCI 콘솔 — Security List / NSG 인그레스
VCN → 서브넷의 Security List(또는 인스턴스 NSG)에 **Ingress Rule** 추가:
- Source `0.0.0.0/0` · TCP · Dest port **80**
- Source `0.0.0.0/0` · TCP · Dest port **443**

> 8501은 열지 않는다(streamlit은 루프백 바인딩, 외부는 Caddy만 통과).

### (b) 인스턴스 iptables (Oracle Ubuntu 이미지는 기본 차단)
```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80  -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save     # 재부팅 후에도 유지
```

---

## 2. 앱 셋업
```bash
# uv 설치(없으면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 레포 클론(최초) 또는 갱신
git clone <레포URL> /home/ubuntu/smartfarm_ai      # 최초
# cd /home/ubuntu/smartfarm_ai && git pull         # 이후 업데이트

cd /home/ubuntu/smartfarm_ai
uv venv
uv pip install -r requirements.txt
```
> aarch64에선 `torch`가 CPU 휠로 자동 설치된다(별도 index 불필요).

---

## 3. 모델 전송 (git 밖의 `.pt` 해결)
DL 모델 `*.pt`는 `.gitignore` 제외라 클론에 없음 → **로컬 맥에서** 한 번 전송:
```bash
# 로컬(맥)에서 실행
rsync -avz models/tomato_resnet18.pt models/tomato_yolov8n.pt \
      ubuntu@<VM공인IP>:/home/ubuntu/smartfarm_ai/models/
```
> `phase1_crop_env_clf.pkl`은 git에 포함되어 clone에 이미 있음.

---

## 4. systemd 등록 (상시 구동)
```bash
sudo cp /home/ubuntu/smartfarm_ai/deploy/smartfarm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now smartfarm
systemctl status smartfarm                 # active (running) 확인
curl -I http://127.0.0.1:8501              # 200 OK 확인
journalctl -u smartfarm -f                 # 로그 확인(문제 시)
```

---

## 5. DNS
도메인 관리 콘솔에서 **A 레코드** → VM 공인 IP 로 지정. (전파 후 `dig <도메인>` 로 확인)

---

## 6. Caddy 설치 + HTTPS
```bash
# Caddy 설치(공식 apt 저장소)
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy

# Caddyfile 배치(<도메인> 치환 후)
sudo cp /home/ubuntu/smartfarm_ai/deploy/Caddyfile /etc/caddy/Caddyfile
sudo sed -i 's/<도메인>/your.domain.com/' /etc/caddy/Caddyfile   # 실제 도메인으로
sudo systemctl reload caddy
```
> Caddy가 80→443 리다이렉트 + Let's Encrypt 인증서 자동 발급/갱신.

---

## 7. 검증
```bash
curl -I https://<도메인>          # 200 OK + HTTPS
```
브라우저로 `https://<도메인>` 접속 → 사이드바 **홈 / Phase 1 ML / Phase 2 DL / Phase 3 LLM** 4페이지 전환,
ML 슬라이더 예측 · DL 사진 업로드 진단/검출 · LLM 플레이스홀더 동작 확인.

---

## 업데이트 절차 (재배포)
```bash
cd /home/ubuntu/smartfarm_ai && git pull
uv pip install -r requirements.txt        # 의존성 변경 시
sudo systemctl restart smartfarm
```
모델이 바뀌면 3번 rsync 재실행.

## 8. PostgreSQL 16 + pgvector (선택 — RAG_BACKEND=pgvector 쓸 때만)

Docker 미사용 전제라 **dnf(Oracle Linux)로 직접 설치**. 실 운영 경로는 `/opt/smartfarm_ai`
(GitHub Actions가 `bash /opt/smartfarm_ai/deploy/deploy.sh` 호출) — 아래는 그 기준.
> 서버가 Ubuntu면 `dnf` 대신 `apt`(postgresql-16, postgresql-16-pgvector 패키지명 확인) 로 치환.

```bash
# PGDG repo 등록 + 설치
sudo dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-9-$(uname -m)/pgdg-redhat-repo-latest.noarch.rpm
sudo dnf install -y postgresql16-server pgvector_16

# 최초 1회 initdb + 기동
sudo /usr/pgsql-16/bin/postgresql-16-setup initdb
sudo systemctl enable --now postgresql-16
systemctl status postgresql-16                 # active (running) 확인
```

앱 유저·DB·확장 생성 (superuser로 `CREATE EXTENSION`까지 선행 — schema.sql의 동일 구문은 멱등 no-op):
```bash
sudo -u postgres psql <<'SQL'
CREATE USER smartfarm WITH PASSWORD '<강한 비밀번호>';
CREATE DATABASE smartfarm OWNER smartfarm;
\c smartfarm
CREATE EXTENSION IF NOT EXISTS vector;
SQL
```

서버 `.env`(기존 시크릿 관리 방식과 동일 — git에 올리지 않음)에 추가:
```
RAG_BACKEND=pgvector
DATABASE_URL=postgresql://smartfarm:<비밀번호>@localhost:5432/smartfarm
```
> ⚠️ env 변경은 **프로세스 재시작 후에만 반영**(`sudo systemctl restart smartfarm-ai`).

접속은 localhost 전용(방화벽/Caddy 무변경 — 5432는 외부에 열지 않음).
이후 `deploy.sh` 재배포 시 `.env`에 `DATABASE_URL`이 있으면 스키마 적용(`db/schema.sql`)과
RAG 코퍼스 적재(`python -m llm.rag.sync`)가 자동 실행된다(실패해도 배포는 계속됨).

**장애 폴백 확인**: `sudo systemctl stop postgresql-16` 후 처방 실행 → npz 폴백으로 정상 응답(3초 타임아웃 내) 확인.

---

## 트러블슈팅
> 아래는 운영 중 빠른 대응 FAQ. **배포 당시 실제 겪은 문제 전체 기록은 [트러블슈팅 내역](../docs/troubleshooting/troubleshooting.md) §C 참고.**

- **502/연결 안 됨**: `systemctl status smartfarm` 확인 → streamlit 죽었으면 `journalctl -u smartfarm -e`.
- **HTTPS 발급 실패**: DNS A레코드 전파 전이거나 80/443 미개방 → 1번·5번 재확인.
- **DL 모델 에러 화면**: 3번 rsync 누락 → `ls models/*.pt` 확인.
- **메모리**: A1 24GB면 ML+DL 동시 충분. `htop`으로 모니터.
