"""제어 상태 파일 공통 원자적 읽기/쓰기 유틸(이슈 #40) — live.py의 _load_state/_save_state,
setpoints.py의 load/save가 각각 구현하던 동일 패턴(tmp 파일 → os.replace, 읽기 실패 시
빈 dict 폴백)을 한 곳으로 모은다. 두 모듈의 공개 함수 시그니처는 그대로 유지된다."""
import json
import os
from pathlib import Path


def save_json_atomic(path: Path, data: dict) -> None:
    """tmp 파일에 쓴 뒤 os.replace로 교체 — 쓰기 실패(권한·디스크 등) 시 예외를 전파하지
    않고 조용히 반환한다(기존 live._save_state/setpoints.save 패턴 그대로)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except OSError:
        return


def load_json(path: Path) -> dict:
    """파일 없거나 손상됐으면 빈 dict(예외 전파 없음)."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
