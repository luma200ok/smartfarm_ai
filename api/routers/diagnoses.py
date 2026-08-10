"""POST /api/diagnoses — 잎 사진 → OOD 게이트 → 부위 게이트 → CNN 진단(+Grad-CAM) → YOLO 검출.

파이프라인 순서는 `app/views/diagnosis.py`(thin wrapper) 현행 흐름을 그대로 따른다. 게이트가
차단해도 200 + `ood_blocked: true`(현행 뷰·`src/llm/tools.get_diagnosis` 동작과 동일 의미).

이미지 검증(크기 캡·픽셀 폭탄 방어·디코드)은 `api/uploads.py` 공용 헬퍼를 쓴다 — `dl.infer`의
진단 함수들은 PIL.Image를 직접 받으므로(경로 기반 아님) tempfile 없이 검증된 PIL을 바로 쓴다
(prescriptions 라우터는 경로 기반 `get_diagnosis`를 쓰므로 tempfile이 필요).

무거운 추론(Grad-CAM·YOLO)은 `api/concurrency.py`의 동시성 캡(최대 2) 안에서만 실행한다.

`include_visuals=False`(code-reviewer P2-1)면 Grad-CAM(backward pass)·YOLO 계산을 건너뛰고
`infer.diagnose()`(순전파만)로 라벨·확률만 낸다 — 텍스트 전용 호출(처방 뷰의 진단 재사용)이
쓰지도 않는 시각화를 매번 계산하는 걸 막는다. 검출 전용은 `/api/detections`(PR 3-1)를 쓴다.
"""
import numpy as np
from dl import infer
from fastapi import APIRouter, File, Query, UploadFile

from ..concurrency import inference_slot
from ..images import ndarray_to_png_base64
from ..schemas import DetectionBox, DiagnosisResponse, YoloResult
from ..uploads import load_pil

router = APIRouter(tags=["diagnoses"])


def _overlay_png_base64(img: np.ndarray, cam: np.ndarray) -> str:
    """원본(224,224,3 float[0,1]) 위에 jet 히트맵 반투명 합성 → PNG base64.

    `app/views/diagnosis.py:overlay()`와 동일 로직(사람이 보는 Grad-CAM 근거 이미지).
    """
    import matplotlib.cm as cm

    heat = cm.jet(cam)[..., :3]
    blended = (0.55 * img + 0.45 * heat).clip(0, 1)
    return ndarray_to_png_base64((blended * 255).astype("uint8"))


@router.post("/diagnoses", response_model=DiagnosisResponse)
def create_diagnosis(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="YOLO 검출 신뢰도 임계값"),
    include_visuals: bool = Query(
        True, description="False면 Grad-CAM·YOLO 계산을 생략(cam_png_base64·yolo=None) — "
                           "라벨·확률만 필요한 텍스트 전용 호출에서 추론 비용을 아낀다."),
) -> DiagnosisResponse:
    with inference_slot():  # 서버 혼잡 시 429(P2-2) — 디코드도 CPU·메모리를 쓰므로 슬롯 안에서
        pil = load_pil(file)  # 크기 캡(413)·픽셀 폭탄(400)·디코드 실패(400) 모두 여기서 처리

        # ① OOD 게이트 — 식물/잎이 아니면 차단
        score = infer.ood_plant_score(pil)
        if score < infer.PLANT_THRESHOLD:
            return DiagnosisResponse(ood_blocked=True, reason="식물·잎으로 보이지 않는 사진(OOD)",
                                      plant_score=score)

        # ② 부위 게이트 — 잎이 아닌 부위(과실/꽃/줄기)면 차단
        part, part_prob = infer.part_of(pil)
        if part != "leaf":
            return DiagnosisResponse(
                ood_blocked=True,
                reason=f"잎이 아닌 부위로 판정({infer.PART_KR.get(part, part)})",
                part=part,
                plant_score=score,
                part_prob=part_prob,
            )

        # ③ 통과 → 잎 진단(+ include_visuals면 Grad-CAM·YOLO 검출까지)
        if include_visuals:
            label, prob, probs_arr, cam, img = infer.predict_with_cam(pil)
            probs = {c: float(p) for c, p in zip(infer.CLASSES, probs_arr)}
            cam_png_base64 = _overlay_png_base64(img, cam)
            annotated, dets = infer.detect_annotated(pil, conf=conf)
            yolo = YoloResult(
                annotated_png_base64=ndarray_to_png_base64(annotated.astype("uint8")),
                boxes=[DetectionBox(label=lab, conf=float(c)) for lab, c in dets],
            )
        else:
            diag = infer.diagnose(pil)  # 순전파만(backward·CAM·YOLO 없음) — infer.predict_with_cam보다 가볍다
            label, prob, probs = diag["label"], diag["prob"], diag["probs"]
            cam_png_base64 = None
            yolo = None

        return DiagnosisResponse(
            ood_blocked=False,
            label=label,
            label_kr=infer.LABEL_KR[label],
            prob=float(prob),
            probs=probs,
            part=part,
            # plant_score·part_prob는 게이트 차단 안내 문구에만 쓰인다(`tools.get_diagnosis()`도
            # 성공 시엔 두 값을 싣지 않음) — 성공 응답은 위 두 필드를 None으로 둔다.
            cam_png_base64=cam_png_base64,
            yolo=yolo,
        )
