"""
처방(Prescription)·경보(Warning/monitor alert) 이력 저장 — best-effort.

notify.py 와 동일 계약: DATABASE_URL 미설정 → no-op(False), DB 장애 → 예외 절대 전파 금지
(이력 저장 실패가 처방/경보 흐름을 막으면 안 됨). Streamlit뿐 아니라 CLI 경로
(monitor.py, prescribe.py __main__)도 이력이 남도록 훅은 호출자가 아닌 함수 내부에 둔다.
"""
import logging

from psycopg.types.json import Jsonb

from . import db

_log = logging.getLogger(__name__)


def save_prescription(user_msg: str, image_path: str | None, diag: dict | None, prescription,
                       caller_ref: str | None = None) -> bool:
    """처방 1건 저장. 성공 True, DB 미설정/실패 시 False(예외 전파 없음).

    caller_ref(smartfarm_ai#66) — 서비스(smartfarm_service)가 넘긴 optional 호출자 식별자
    (최대 64자, API 계층에서 검증). 이력 보존 정책의 "태깅" 안 — 값 검증·의미 부여는
    호출자 책임이고 이 함수는 그대로 저장만 한다.
    """
    try:
        conn = db.get_conn()
        if conn is None:
            return False
        with conn:                   # 종료 시 close — connect-per-call이라 누수 방지 필수
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO prescriptions (user_msg, image_path, diag, prescription, caller_ref) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (user_msg, image_path, Jsonb(diag) if diag is not None else None,
                     Jsonb(prescription.model_dump()), caller_ref),
                )
        return True
    except Exception as e:
        _log.warning("처방 이력 저장 실패(무시): %s", e)
        return False


def save_alert(kind: str, level: str, disease: str, reason: str, payload: dict) -> bool:
    """경보 1건 저장(kind='early_warning'|'monitor'). 성공 True, DB 미설정/실패 시 False."""
    try:
        conn = db.get_conn()
        if conn is None:
            return False
        with conn:                   # 종료 시 close
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO alerts (kind, level, disease, reason, payload) VALUES (%s, %s, %s, %s, %s)",
                    (kind, level, disease, reason, Jsonb(payload)),
                )
        return True
    except Exception as e:
        _log.warning("경보 이력 저장 실패(무시): %s", e)
        return False
