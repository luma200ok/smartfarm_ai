"""POST /api/detections — 잎 사진 → YOLO 병변 위치 검출(게이트 없음).

`app/views/diagnosis.py` 탭2(YOLO 위치 검출)는 원래 OOD·부위 게이트를 거치지 않고 바로
`infer.detect_annotated()`를 호출한다(진단 탭1과 달리 "장면 속 잎을 찾는" 용도라 게이트가
의미 없음). PR 3에서 API 모드 탭2가 `/api/diagnoses`(진단+검출 결합, 게이트 있음)를 재사용하게
했더니 in-process와 결과가 달라지고(게이트로 막히는 사진이 API 모드에선 검출조차 안 됨)
Grad-CAM까지 매번 낭비 계산됐다(reviewer 픽스) — 그래서 검출 전용 엔드포인트를 분리한다.

이미지 검증(크기 캡·픽셀 폭탄 방어·디코드)은 `api/uploads.py` 공용 헬퍼, 동시성 캡은
`api/concurrency.py`를 `diagnoses`·`prescriptions`와 동일하게 재사용한다(무거운 추론 요청이므로
같은 슬롯을 공유해야 캡의 취지가 유지된다).
"""
from dl import infer
from fastapi import APIRouter, File, Query, UploadFile

from ..concurrency import inference_slot
from ..images import ndarray_to_png_base64
from ..schemas import DetectionBox, YoloResult
from ..uploads import load_pil

router = APIRouter(tags=["detections"])


@router.post("/detections", response_model=YoloResult)
def create_detection(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="YOLO 검출 신뢰도 임계값"),
) -> YoloResult:
    with inference_slot():  # 서버 혼잡 시 429 — 디코드도 CPU·메모리를 쓰므로 슬롯 안에서
        pil = load_pil(file)  # 크기 캡(413)·픽셀 폭탄(400)·디코드 실패(400) 모두 여기서 처리
        annotated, dets = infer.detect_annotated(pil, conf=conf)
        return YoloResult(
            annotated_png_base64=ndarray_to_png_base64(annotated.astype("uint8")),
            boxes=[DetectionBox(label=lab, conf=float(c)) for lab, c in dets],
        )
