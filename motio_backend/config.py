"""경로/한도/CORS/미디어 등 백엔드 설정. (docs/MOTIO_INTEGRATION.md)"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# capstone_deploy/ (추론서버 repo 루트)
REPO_ROOT = Path(__file__).resolve().parent.parent

# 서비스 데이터 루트 (모든 업로드/클립/검출/잡 산출물) — /media 로 정적 제공
SERVICE_ROOT = Path(os.environ.get("MOTIO_SERVICE_ROOT", REPO_ROOT / "outputs" / "motio_service"))
VIDEOS_DIR = SERVICE_ROOT / "videos"
CLIPS_DIR = SERVICE_ROOT / "clips"
DETECT_DIR = SERVICE_ROOT / "detections"
SELECT_DIR = SERVICE_ROOT / "selections"
JOBS_DIR = SERVICE_ROOT / "jobs"

MEDIA_MOUNT = "/media"  # StaticFiles 마운트 경로 -> SERVICE_ROOT

# 체인 단계 실행 인터프리터 (백엔드가 cap-deploy 에서 돌면 sys.executable = cap-deploy python)
CAP_PYTHON = os.environ.get("CAP_PYTHON", sys.executable)
ENTRY_POINT_OPT = REPO_ROOT / "entry_point_opt.py"
DETECT_SCRIPT = Path(__file__).resolve().parent / "detect_people.py"

# 업로드 정책
ALLOWED_EXT = {".mp4", ".mov"}
ALLOWED_MIME = {"video/mp4", "video/quicktime", "application/octet-stream"}
MAX_UPLOAD_BYTES = int(os.environ.get("MOTIO_MAX_UPLOAD_BYTES", str(2 * 1024 ** 3)))  # 2GB (화면 문구와 일치)
UPLOAD_CHUNK = 1024 * 1024

# 미리보기(재생용 경량) 인코딩 높이
PREVIEW_HEIGHT = int(os.environ.get("MOTIO_PREVIEW_HEIGHT", "720"))

# CORS (프론트 오리진). 데모는 * 허용.
CORS_ORIGINS = [o.strip() for o in os.environ.get("MOTIO_CORS_ORIGINS", "*").split(",") if o.strip()]

# 폴링 안내 간격(초)
RETRY_AFTER = int(os.environ.get("MOTIO_RETRY_AFTER", "2"))


def ensure_dirs() -> None:
    for d in (SERVICE_ROOT, VIDEOS_DIR, CLIPS_DIR, DETECT_DIR, SELECT_DIR, JOBS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def media_url(path: str | Path) -> str:
    """SERVICE_ROOT 하위 절대경로 -> 공개 /media URL. (내부 절대경로 노출 금지)"""
    rel = Path(path).resolve().relative_to(SERVICE_ROOT.resolve())
    return f"{MEDIA_MOUNT}/{rel.as_posix()}"
