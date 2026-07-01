"""
Phase 3-1 — LLM function-calling tool 정의.

LLM(qwen2.5:14b)이 호출하는 tool 2종을 실제 DL 추론(src/dl/infer.py)에 연결한다.
분업 원칙: 진단·검출은 여기(ML/DL), 설명·처방은 LLM(prescribe.py).

- get_diagnosis: OOD 게이트 → 부위 게이트 → 잎 진단(환각 방어 ②의 차단 경로 내장)
- get_detection: YOLO 병변 위치
TOOL_SCHEMAS 는 Ollama `tools=` 인자로 넘길 OpenAI 호환 스키마.
"""
import sys
from pathlib import Path

from PIL import Image

# src 를 경로에 추가 → `python src/llm/prescribe.py` 단독 실행·pytest 양쪽에서 import 되게.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dl import infer  # noqa: E402


def _round(d, n=3):
    return {k: round(v, n) for k, v in d.items()}


def get_diagnosis(image_path: str) -> dict:
    """잎 사진 → 진단. 게이트를 통과 못 하면 진단 대신 차단 사유를 반환(환각 방어 ②)."""
    pil = Image.open(image_path)

    # ① OOD 게이트 — 식물/잎이 아니면 차단
    score = infer.ood_plant_score(pil)
    if score < infer.PLANT_THRESHOLD:
        return {"ood_blocked": True,
                "reason": "식물·잎으로 보이지 않는 사진(OOD)",
                "plant_score": round(score, 3)}

    # ② 부위 게이트 — 잎이 아닌 부위(과실/꽃/줄기)면 차단
    part, part_prob = infer.part_of(pil)
    if part != "leaf":
        return {"ood_blocked": True,
                "reason": f"잎이 아닌 부위로 판정({infer.PART_KR.get(part, part)})",
                "part": part, "part_prob": round(part_prob, 3)}

    # ③ 통과 → 잎 진단
    d = infer.diagnose(pil)
    return {
        "ood_blocked": False,
        "label": d["label"],
        "label_kr": infer.LABEL_KR[d["label"]],
        "prob": round(d["prob"], 3),
        "probs": _round(d["probs"]),
        "part": part,
    }


def get_detection(image_path: str) -> dict:
    """잎 사진 → 병변 위치 검출(YOLO). 비식물(OOD)이면 검출 대신 차단(진단과 동일 안전망)."""
    pil = Image.open(image_path)
    score = infer.ood_plant_score(pil)
    if score < infer.PLANT_THRESHOLD:
        return {"ood_blocked": True,
                "reason": "식물·잎으로 보이지 않는 사진(OOD)",
                "plant_score": round(score, 3)}
    boxes = infer.detect(pil)
    return {"ood_blocked": False, "boxes": boxes, "lesion_count": len(boxes)}


def get_forecast() -> dict:
    """LSTM 환경 예측 — 다음날 내부온도·추세 + 최근 7일 습도위험. 실패 시 unavailable(예외 전파 안 함)."""
    try:
        window = infer.latest_window()
        if window is None:
            return {"unavailable": True, "reason": "환경 데이터(env_daily.csv) 없음"}
        fc = infer.forecast(window)
        if fc is None:
            return {"unavailable": True, "reason": "LSTM 모델 없음 (train_lstm.py 실행 필요)"}
        return fc
    except Exception as e:                              # CSV 손상·컬럼 변경 등도 죽지 않게
        return {"unavailable": True, "reason": f"{type(e).__name__}: {e}"}


# ── Ollama tools= 스키마 (OpenAI 호환) ────────────────────────────────────
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_diagnosis",
            "description": "토마토 잎 사진을 진단한다. 잎이 아니면(과실·꽃·줄기·비식물) 진단 대신 차단 사유를 돌려준다. "
                           "진단 클래스는 잎곰팡이병(leaf_mold)·정상(normal)·황화잎말이바이러스(tylcv) 3종뿐.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "진단할 잎 사진 파일 경로"},
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_detection",
            "description": "토마토 잎 사진에서 병변 위치를 검출한다(YOLO). 병변 박스와 상/중/하 위치를 반환.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "검출할 잎 사진 파일 경로"},
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "환경 예측 — 다음날 내부온도(℃)·추세는 LSTM 예측값, 습도위험(높음/보통/낮음)은 최근 7일 습도 평균 기준이다. "
                           "습도가 높으면 곰팡이병 위험이 커지므로 선제 처방(환기 등)에 활용한다. 파라미터 없음.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_REGISTRY = {
    "get_diagnosis": get_diagnosis,
    "get_detection": get_detection,
    "get_forecast": get_forecast,
}
