"""JSON 파일 기반 상태 저장 (videos/clips/detections/selections/jobs).

데모 규모(단일 서버·로컬 디스크)라 SQLite 대신 폴더+JSON 으로 단순화.
ID 는 추측 불가능한 랜덤(명세 2.3) — ID 자체가 접근 키.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any

from . import config


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # 원자적 교체


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── video ──────────────────────────────────────────────
def video_dir(video_id: str) -> Path:
    return config.VIDEOS_DIR / video_id


def save_video_meta(video_id: str, meta: dict) -> None:
    _write_json(video_dir(video_id) / "meta.json", meta)


def load_video_meta(video_id: str) -> dict | None:
    return _read_json(video_dir(video_id) / "meta.json")


# ── clip ───────────────────────────────────────────────
def clip_dir(clip_id: str) -> Path:
    return config.CLIPS_DIR / clip_id


def save_clip_meta(clip_id: str, meta: dict) -> None:
    _write_json(clip_dir(clip_id) / "meta.json", meta)


def load_clip_meta(clip_id: str) -> dict | None:
    return _read_json(clip_dir(clip_id) / "meta.json")


# ── detection / selection ──────────────────────────────
def detection_path(clip_id: str) -> Path:
    return config.DETECT_DIR / f"{clip_id}.json"


def save_detection(clip_id: str, data: dict) -> None:
    _write_json(detection_path(clip_id), data)


def load_detection(clip_id: str) -> dict | None:
    return _read_json(detection_path(clip_id))


def selection_path(clip_id: str) -> Path:
    return config.SELECT_DIR / f"{clip_id}.json"


def save_selection(clip_id: str, data: dict) -> None:
    _write_json(selection_path(clip_id), data)


def load_selection(clip_id: str) -> dict | None:
    return _read_json(selection_path(clip_id))


# ── job ────────────────────────────────────────────────
def job_dir(job_id: str) -> Path:
    return config.JOBS_DIR / job_id


def job_output_dir(job_id: str) -> Path:
    return job_dir(job_id) / "output"


def save_job(job_id: str, data: dict) -> None:
    _write_json(job_dir(job_id) / "status.json", data)


def load_job(job_id: str) -> dict | None:
    return _read_json(job_dir(job_id) / "status.json")


def iter_jobs() -> list[dict]:
    if not config.JOBS_DIR.exists():
        return []
    out = []
    for p in config.JOBS_DIR.glob("*/status.json"):
        d = _read_json(p)
        if d:
            out.append(d)
    return out
