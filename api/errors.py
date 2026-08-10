"""API 커스텀 예외 + FastAPI 예외 핸들러 일괄 등록.

전역 컨벤션(Flask/FastAPI ApiError 단일 패턴)을 FastAPI로 이식 — 도메인 예외는
`raise ApiError(status_code, message)` 하나로 통일하고, 응답은 `{"detail": message}`로 고정한다.
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """status_code + message 를 갖는 단일 도메인 예외(라우터에서 raise)."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


def register_error_handlers(app: FastAPI) -> None:
    """`ApiError` → `{"detail": message}` 응답으로 일괄 변환."""

    @app.exception_handler(ApiError)
    async def _handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})
