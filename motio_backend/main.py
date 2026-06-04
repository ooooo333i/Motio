"""Motio 백엔드 어댑터 FastAPI 앱 — 프론트 명세(motio-demo-api-spec.md)를 그대로 구현.

실행: uvicorn motio_backend.main:app --host 0.0.0.0 --port 8000
플로우: upload -> clip(trim) -> detect -> select-target -> extraction-job -> poll -> result/download
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, detection, errors, jobs, media, storage
from .errors import MotioError
from .schemas import (
    ClipCreateRequest, JobCreateRequest, SelectTargetRequest,
)

config.ensure_dirs()

app = FastAPI(title="Motio Backend Adapter")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",     # 프론트가 credentials:include 라 * 대신 정규식 사용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(MotioError, errors.motio_error_handler)
app.add_exception_handler(Exception, errors.unhandled_error_handler)


# ── 요청 타이밍 트레이스(병목 분석용) ───────────────────
# upload/clip/detect/select/job-create 각 단계의 벽시계 소요를 access_trace.log 에 기록.
# /media·정적 자산·헬스는 잡음이라 제외. (trace.py 가 이 로그를 읽어 타임라인 구성.)
_TRACE_LOG = config.SERVICE_ROOT / "access_trace.log"


@app.middleware("http")
async def _trace_timing(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    dt_ms = (time.time() - t0) * 1000.0
    path = request.url.path
    if path.startswith("/api/") and not path.startswith("/api/extraction-jobs/"):
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))
            line = f"{ts}\t{t0:.3f}\t{request.method}\t{path}\t{response.status_code}\t{dt_ms:.1f}ms\n"
            with _TRACE_LOG.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:
            pass
    return response

# /media -> 서비스 루트 (preview/결과 파일 공개 제공)
app.mount(config.MEDIA_MOUNT, StaticFiles(directory=str(config.SERVICE_ROOT)), name="media")


@app.on_event("startup")
def _startup() -> None:
    config.ensure_dirs()
    jobs.recover_on_startup()


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "service_root": str(config.SERVICE_ROOT)}


# ── 4.1 업로드 ──────────────────────────────────────────
@app.post("/api/videos", status_code=201)
async def create_video(response: Response, file: UploadFile = File(...)) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXT:
        raise MotioError(415, errors.UNSUPPORTED_MEDIA, f"지원하지 않는 형식: {ext or '?'} (MP4/MOV)")

    video_id = storage.new_id("vid")
    vdir = storage.video_dir(video_id)
    vdir.mkdir(parents=True, exist_ok=True)
    dst = vdir / f"original{ext}"

    size = 0
    with dst.open("wb") as out:
        while True:
            chunk = await file.read(config.UPLOAD_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > config.MAX_UPLOAD_BYTES:
                out.close()
                dst.unlink(missing_ok=True)
                raise MotioError(413, errors.FILE_TOO_LARGE, "업로드 가능한 최대 크기를 초과했습니다.")
            out.write(chunk)
    if size == 0:
        dst.unlink(missing_ok=True)
        raise MotioError(400, errors.INVALID_UPLOAD, "빈 파일입니다.")

    storage.save_video_meta(video_id, {
        "videoId": video_id, "original_name": file.filename,
        "path": str(dst), "size": size, "created_at": time.time(),
    })
    response.headers["Location"] = f"/api/videos/{video_id}"
    return {"videoId": video_id}


# ── 4.3 클립(구간) 생성 ─────────────────────────────────
@app.post("/api/videos/{video_id}/clip", status_code=201)
async def create_clip(video_id: str, body: ClipCreateRequest, response: Response) -> dict:
    vmeta = storage.load_video_meta(video_id)
    if vmeta is None:
        raise MotioError(404, errors.VIDEO_NOT_FOUND, "비디오를 찾을 수 없습니다.")
    src = vmeta["path"]

    clip_id = storage.new_id("clip")
    cdir = storage.clip_dir(clip_id)
    cdir.mkdir(parents=True, exist_ok=True)
    clip_mp4 = cdir / "clip.mp4"
    preview_mp4 = cdir / "preview.mp4"

    try:
        duration = await media.trim_clip(src, body.startTime, body.endTime, clip_mp4)
    except Exception as exc:
        raise MotioError(400, errors.BAD_REQUEST, f"트림 실패: {exc}")

    preview_url = None
    try:
        await media.make_preview(clip_mp4, preview_mp4)
        preview_url = config.media_url(preview_mp4)
    except Exception:
        preview_url = None  # 데모: preview 실패해도 clip 자체는 유효

    storage.save_clip_meta(clip_id, {
        "clipId": clip_id, "videoId": video_id,
        "clip_path": str(clip_mp4),
        "preview_path": str(preview_mp4) if preview_url else None,
        "duration": duration, "range": [body.startTime, body.endTime],
        "created_at": time.time(),
    })
    response.headers["Location"] = f"/api/videos/{video_id}/clip/{clip_id}"
    return {"clipId": clip_id, "duration": round(duration, 3), "previewUrl": preview_url}


# ── 4.4 검출 (POST=실행 동기, GET=결과) ─────────────────
@app.post("/api/videos/{video_id}/clip/{clip_id}/detect", status_code=200)
async def post_detect(video_id: str, clip_id: str) -> dict:
    cmeta = storage.load_clip_meta(clip_id)
    if cmeta is None:
        raise MotioError(404, errors.CLIP_NOT_FOUND, "클립을 찾을 수 없습니다.")
    # 동기 검출(프론트는 POST 완료 후 GET 으로 결과 조회)
    try:
        await detection.run_detection(clip_id, cmeta["clip_path"])
    except Exception as exc:
        raise MotioError(500, errors.INFERENCE_FAILED, f"검출 실패: {exc}")
    return {"status": "complete"}


@app.get("/api/videos/{video_id}/clip/{clip_id}/detect")
def get_detect(video_id: str, clip_id: str) -> dict:
    data = storage.load_detection(clip_id)
    if data is None:
        raise MotioError(404, errors.DETECTION_NOT_READY, "검출 결과가 아직 없습니다. 먼저 검출을 요청하세요.")
    return data


# ── 4.5 대상 선택 ───────────────────────────────────────
@app.post("/api/videos/{video_id}/clip/{clip_id}/select-target")
def select_target(video_id: str, clip_id: str, body: SelectTargetRequest) -> dict:
    det = storage.load_detection(clip_id)
    if det is None:
        raise MotioError(404, errors.DETECTION_NOT_READY, "검출 결과가 없습니다.")
    valid = {str(o.get("id")) for o in det.get("objects", [])}
    if str(body.targetId) not in valid:
        raise MotioError(404, errors.TARGET_NOT_FOUND, f"대상 {body.targetId} 가 검출 결과에 없습니다.")
    try:
        person_track_id = int(body.targetId)
    except ValueError:
        raise MotioError(400, errors.BAD_REQUEST, "targetId 는 정수 track id 여야 합니다.")
    storage.save_selection(clip_id, {
        "clipId": clip_id, "targetId": body.targetId,
        "person_track_id": person_track_id, "created_at": time.time(),
    })
    return {"targetId": body.targetId}


# ── 4.6 추출 job 생성 ───────────────────────────────────
@app.post("/api/extraction-jobs", status_code=202)
async def create_extraction_job(body: JobCreateRequest) -> dict:
    if storage.load_video_meta(body.videoId) is None:
        raise MotioError(404, errors.VIDEO_NOT_FOUND, "비디오를 찾을 수 없습니다.")
    cmeta = storage.load_clip_meta(body.clipId)
    if cmeta is None:
        raise MotioError(404, errors.CLIP_NOT_FOUND, "클립을 찾을 수 없습니다.")
    det = storage.load_detection(body.clipId)
    if det is None:
        raise MotioError(404, errors.DETECTION_NOT_READY, "검출 결과가 없습니다.")
    if str(body.targetId) not in {str(o.get("id")) for o in det.get("objects", [])}:
        raise MotioError(404, errors.TARGET_NOT_FOUND, f"대상 {body.targetId} 가 검출 결과에 없습니다.")
    try:
        person_track_id = int(body.targetId)
    except ValueError:
        raise MotioError(400, errors.BAD_REQUEST, "targetId 는 정수 track id 여야 합니다.")

    job = jobs.create_job(body.videoId, body.clipId, body.targetId, person_track_id, cmeta["clip_path"])
    return {"jobId": job["jobId"], "status": "queued"}


# ── 4.7 job 상태 (폴링) ─────────────────────────────────
@app.get("/api/extraction-jobs/{job_id}")
def get_job(job_id: str, response: Response) -> dict:
    job = storage.load_job(job_id)
    if job is None:
        raise MotioError(404, errors.JOB_NOT_FOUND, "작업을 찾을 수 없습니다.")
    if job["status"] in ("queued", "running"):
        response.headers["Retry-After"] = str(config.RETRY_AFTER)
    return jobs.status_payload(job)


# ── 4.8 결과 / 다운로드 ─────────────────────────────────
@app.get("/api/extraction-jobs/{job_id}/result")
def get_result(job_id: str) -> dict:
    job = storage.load_job(job_id)
    if job is None:
        raise MotioError(404, errors.JOB_NOT_FOUND, "작업을 찾을 수 없습니다.")
    if job["status"] != "complete":
        raise MotioError(409, errors.ARTIFACT_NOT_FOUND, f"작업이 아직 완료되지 않았습니다(status={job['status']}).")
    return jobs.build_result(job)


@app.get("/api/extraction-jobs/{job_id}/download")
def download(job_id: str, format: str = "FBX") -> FileResponse:
    job = storage.load_job(job_id)
    if job is None:
        raise MotioError(404, errors.JOB_NOT_FOUND, "작업을 찾을 수 없습니다.")
    files = job.get("files_internal") or {}
    key = format.strip().lower()
    if key not in ("fbx", "bvh"):
        raise MotioError(400, errors.BAD_REQUEST, "format 은 FBX 또는 BVH 여야 합니다.")
    path = files.get(key)
    if not path or not Path(path).exists():
        raise MotioError(404, errors.ARTIFACT_NOT_FOUND, f"{format} 산출물이 없습니다.")
    return FileResponse(path, filename=f"output.{key}", media_type="application/octet-stream")


# ── 프론트 정적 서빙(단일 오리진) ────────────────────────
# motio-frontend/dist 가 빌드돼 있으면 같은 백엔드에서 SPA 로 서빙 → CORS/오리진 불필요, 단일 URL.
# (없으면 API 만 제공. 빌드: VITE_USE_MOCK_API=false VITE_API_BASE_URL="" npm run build)
FRONTEND_DIST = (config.REPO_ROOT.parent / "motio-frontend" / "dist")
if FRONTEND_DIST.is_dir():
    _assets = FRONTEND_DIST / "assets"
    if _assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        # API/미디어/문서 경로는 위 라우트가 이미 처리. 그 외는 SPA index.html.
        if full_path.startswith(("api/", "media/", "healthz", "docs", "redoc", "openapi")):
            raise MotioError(404, errors.ARTIFACT_NOT_FOUND, "not found")
        candidate = FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
