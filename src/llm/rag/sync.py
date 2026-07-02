"""
코퍼스(data/nongsaro/*.md) → PostgreSQL(rag_chunks) 전체 재적재 CLI.

청크 수십 개 규모라 증분 diff는 복잡도만 추가 — 코퍼스 해시(store._corpus_key)가
rag_meta와 같으면 스킵, 다르면 전량 삭제 후 재삽입(멱등).

실행:  python -m llm.rag.sync [--force]
전제:  .env 의 DATABASE_URL (미설정이면 명시적으로 exit 1)
"""
import argparse
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm import db  # noqa: E402
from llm.rag.corpus import load_chunks  # noqa: E402
from llm.rag.store import EMBED_MODEL, _corpus_key, _embed  # noqa: E402

_log = logging.getLogger(__name__)


def _meta_get(conn, key: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT value FROM rag_meta WHERE key = %s", (key,))
        row = cur.fetchone()
        return row[0] if row else None


def sync(force: bool = False) -> int:
    """코퍼스를 PG에 재적재. 변경 없으면 스킵(0), 적재하면 삽입 건수 반환."""
    conn = db.get_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL 미설정 — sync 불가")

    with conn:                       # 종료 시 close — connect-per-call이라 누수 방지 필수
        chunks = load_chunks()
        if not chunks:
            _log.warning("코퍼스가 비어 있음(data/nongsaro/*.md) — 적재할 chunk 없음")
            return 0

        key = _corpus_key(chunks)
        if not force and _meta_get(conn, "corpus_key") == key:
            _log.info("코퍼스 변경 없음(key=%s) — sync 스킵", key[:12])
            return 0

        emb = _embed([c["text"] for c in chunks])

        # G1 픽스: get_conn()이 autocommit=True라 명시 트랜잭션 없으면 INSERT 도중 실패 시
        # DELETE만 반영되고 테이블이 빈 채로 남을 수 있음 — conn.transaction()으로 원자성 보장.
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("DELETE FROM rag_chunks")
                cur.executemany(
                    """
                    INSERT INTO rag_chunks (text, title, source, source_name, disease, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    [(c["text"], c["title"], c["source"], c["source_name"], c["disease"], emb[i])
                     for i, c in enumerate(chunks)],
                )
                cur.execute(
                    """
                    INSERT INTO rag_meta (key, value) VALUES ('corpus_key', %s), ('embed_model', %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                    """,
                    (key, EMBED_MODEL),
                )
                cur.execute(
                    "INSERT INTO rag_meta (key, value) VALUES ('synced_at', now()::text) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
                )
        _log.info("sync 완료 — %d개 chunk 적재(key=%s)", len(chunks), key[:12])
        return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="코퍼스 → PostgreSQL(rag_chunks) 재적재")
    ap.add_argument("--force", action="store_true", help="코퍼스 변경 없어도 강제 재적재")
    a = ap.parse_args()
    try:
        n = sync(force=a.force)
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)
    print(f"✅ {n}개 chunk 적재" if n else "✅ 변경 없음 — 스킵")
