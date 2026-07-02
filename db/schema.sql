-- SmartFarm AI — PostgreSQL + pgvector 스키마 (멱등, 반복 적용 가능)
-- 적용: psql "$DATABASE_URL" -f db/schema.sql
-- 용도: ① RAG 벡터 저장·검색(rag_chunks) ② 처방·경보 이력(prescriptions/alerts)
-- 과설계 배제: 청크가 수십 개 규모라 HNSW 등 벡터 인덱스는 불필요(전수 스캔으로 충분).
--            임베딩 차원(1024)은 bge-m3 고정 — 모델을 바꾸면 이 스키마도 함께 바뀌어야 함.

CREATE EXTENSION IF NOT EXISTS vector;

-- ── RAG 벡터 저장소 ──────────────────────────────────────────────
-- 컬럼 = 기존 corpus.load_chunks() 가 만드는 chunk dict 그대로(text/title/source/source_name/disease).
CREATE TABLE IF NOT EXISTS rag_chunks (
    id          BIGSERIAL PRIMARY KEY,
    text        TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    source      TEXT NOT NULL DEFAULT '',
    source_name TEXT NOT NULL DEFAULT '',
    disease     TEXT NOT NULL DEFAULT '',
    embedding   VECTOR(1024) NOT NULL,     -- bge-m3 임베딩 차원
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_disease ON rag_chunks (disease);

-- sync.py 의 멱등성 근거(코퍼스 해시가 같으면 재적재 스킵)
CREATE TABLE IF NOT EXISTS rag_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ── 처방 이력 (prescribe.py 결과) ────────────────────────────────
-- diag/prescription 은 이미 pydantic 으로 검증된 한국어 필드 스키마 → jsonb 통저장(별도 정규화 안 함).
CREATE TABLE IF NOT EXISTS prescriptions (
    id           BIGSERIAL PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_msg     TEXT NOT NULL,
    image_path   TEXT,
    diag         JSONB,
    prescription JSONB NOT NULL
);

-- ── 경보 이력 (조기경보 early_warning · 자동감시 monitor 공용) ─────
CREATE TABLE IF NOT EXISTS alerts (
    id         BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    kind       TEXT NOT NULL CHECK (kind IN ('early_warning', 'monitor')),
    level      TEXT NOT NULL,
    disease    TEXT NOT NULL DEFAULT '',
    reason     TEXT NOT NULL DEFAULT '',
    payload    JSONB
);

CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts (created_at DESC);
