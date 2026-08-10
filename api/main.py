"""FastAPI 서빙 진입점(이슈 #59 PR 2) — `api/routers/*`의 health·diagnoses·detections·prescriptions 라우터.

기동:  uvicorn api.main:app --workers 1   (프로젝트 루트에서)

**workers=1 필수** — `src/dl/infer.py`의 모델 캐시(`@lru_cache`)는 프로세스 단위라 워커를 늘리면
모델(resnet18·YOLO·부위 게이트 등, 수십~수백MB)이 워커 수만큼 메모리에 중복 로드된다. 동시
요청 처리량은 torch/ollama 동기 호출을 threadpool(FastAPI 기본)로 넘겨 확보하고, Grad-CAM은
`infer._cam_lock`으로 내부 직렬화되므로 API 계층에서 추가 락은 불필요하다(단, 처리량 상한은
그 직렬화 구간에 걸린다).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI  # noqa: E402

from .errors import register_error_handlers  # noqa: E402
from .routers import detections, diagnoses, health, prescriptions  # noqa: E402

app = FastAPI(
    title="SmartFarm AI API",
    version="1.0.0",
    description="토마토 잎 병해 진단(CNN·YOLO·Grad-CAM) + LLM 자연어 처방 서빙 API",
)
register_error_handlers(app)

app.include_router(health.router, prefix="/api")
app.include_router(diagnoses.router, prefix="/api")
app.include_router(detections.router, prefix="/api")
app.include_router(prescriptions.router, prefix="/api")
