"""Pydantic 요청/응답 모델 (명세 4.x)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


# ── 요청 ────────────────────────────────────────────────
class ClipCreateRequest(BaseModel):
    startTime: str  # "MM:SS"
    endTime: str    # "MM:SS"


class SelectTargetRequest(BaseModel):
    targetId: str


class JobCreateRequest(BaseModel):
    videoId: str
    clipId: str
    targetId: str


# ── 응답 ────────────────────────────────────────────────
class VideoCreatedResponse(BaseModel):
    videoId: str


class ClipCreatedResponse(BaseModel):
    clipId: str
    duration: float
    previewUrl: Optional[str] = None


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class TrackPoint(BaseModel):
    frame: int
    bbox: BBox


class DetectedObject(BaseModel):
    id: str
    label: str
    type: str
    confidence: float
    track: list[TrackPoint]


class DetectionResponse(BaseModel):
    fps: float
    objects: list[DetectedObject]


class SelectTargetResponse(BaseModel):
    targetId: str


class JobCreatedResponse(BaseModel):
    jobId: str
    status: str  # queued


class JobStatusResponse(BaseModel):
    jobId: str
    status: str  # queued | running | complete | failed
    error: Optional[dict] = None


class ResultFiles(BaseModel):
    fbx: Optional[str] = None
    bvh: Optional[str] = None


class ResultMetrics(BaseModel):
    frames: int
    poseAccuracy: float


class ResultResponse(BaseModel):
    jobId: str
    files: ResultFiles
    metrics: ResultMetrics
