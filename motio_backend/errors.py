"""Motio 공통 에러 포맷: {"error": {"code", "message"}} (명세 2.5)."""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class MotioError(Exception):
    """명세 코드 기반 에러. 핸들러가 통일 포맷으로 변환."""

    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


# 권장 코드 (어댑터 설계 §Error)
INVALID_UPLOAD = "INVALID_UPLOAD"
FILE_TOO_LARGE = "FILE_TOO_LARGE"
UNSUPPORTED_MEDIA = "UNSUPPORTED_MEDIA_TYPE"
VIDEO_NOT_FOUND = "VIDEO_NOT_FOUND"
CLIP_NOT_FOUND = "CLIP_NOT_FOUND"
DETECTION_NOT_READY = "DETECTION_NOT_READY"
TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
JOB_NOT_FOUND = "JOB_NOT_FOUND"
INFERENCE_FAILED = "INFERENCE_FAILED"
EXPORT_FAILED = "EXPORT_FAILED"
ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
GPU_BUSY = "GPU_BUSY"
BAD_REQUEST = "BAD_REQUEST"
INTERNAL = "INTERNAL_ERROR"


def _body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


async def motio_error_handler(_: Request, exc: MotioError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=_body(exc.code, exc.message))


async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content=_body(INTERNAL, str(exc) or "Internal server error"))


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=_body(code, message))
