"""
pgvector 기반 RAG 검색 백엔드 — store.py(npz+numpy)의 pgvector 버전.

임베딩은 store._embed() 그대로 재사용(임베딩 소스는 백엔드 무관, Ollama bge-m3 동일).
쿼리는 코사인 거리(pgvector `<=>`)로 top-k → 1 - 거리 로 기존 코사인 유사도 score와 의미 일치.
"""
from .. import db
from . import store


def search(query: str, disease: str | None = None, k: int = 3) -> list[dict]:
    """disease 로 WHERE 필터 후 코사인 거리 top-k. 매칭 chunk 없으면 [] (엉뚱한 병해 근거 방지)."""
    conn = db.get_conn()
    if conn is None:
        return []
    q = store._embed([query])[0]
    with conn:                       # 종료 시 close — connect-per-call이라 누수 방지 필수
        with conn.cursor() as cur:
            if disease is not None:
                cur.execute(
                    """
                    SELECT text, title, source, source_name, disease,
                           1 - (embedding <=> %s) AS score
                    FROM rag_chunks
                    WHERE disease = %s
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (q, disease, q, k),
                )
            else:
                cur.execute(
                    """
                    SELECT text, title, source, source_name, disease,
                           1 - (embedding <=> %s) AS score
                    FROM rag_chunks
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (q, q, k),
                )
            rows = cur.fetchall()
    cols = ("text", "title", "source", "source_name", "disease", "score")
    return [dict(zip(cols, row)) for row in rows]
