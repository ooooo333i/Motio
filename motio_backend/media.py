"""ffmpeg/ffprobe 미디어 처리: 메타 조회, 트림(무손실), 미리보기 인코딩.

- 추론용 클립: ffmpeg stream copy(-c copy) → 재인코딩 없이 4K 원본 화질 보존(명세 4.3).
- 재생용 preview: 720p h264 로 경량 인코딩(select 화면 재생용).
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from . import config


def _mmss_to_seconds(value: str) -> float:
    """'MM:SS' 또는 'HH:MM:SS' 또는 숫자초 → 초."""
    value = str(value).strip()
    if ":" not in value:
        return float(value)
    parts = [float(p) for p in value.split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def probe(video_path: str | Path) -> dict:
    """ffprobe 로 duration/fps/width/height 조회."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
        "-of", "json", str(video_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    # avg_frame_rate "30000/1001" → float
    fr = stream.get("avg_frame_rate", "0/0")
    try:
        num, den = fr.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0
    return {
        "duration": float(fmt.get("duration", 0.0) or 0.0),
        "fps": round(fps, 3) if fps else 0.0,
        "width": int(stream.get("width", 0) or 0),
        "height": int(stream.get("height", 0) or 0),
    }


async def _run_ffmpeg(cmd: list[str]) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        tail = (out or b"").decode("utf-8", "replace")[-2000:]
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}):\n{tail}")


async def trim_clip(src: str | Path, start: str, end: str, out_clip: Path) -> float:
    """무손실 트림(-c copy). 반환=클립 길이(초)."""
    s = _mmss_to_seconds(start)
    e = _mmss_to_seconds(end)
    dur = max(e - s, 0.01)
    out_clip.parent.mkdir(parents=True, exist_ok=True)
    # -ss 를 입력 앞(빠름, 키프레임 단위) + -c copy (무손실). 명세상 1~2초 유연성 허용.
    cmd = [
        "ffmpeg", "-y", "-ss", f"{s:.3f}", "-i", str(src),
        "-t", f"{dur:.3f}", "-c", "copy", "-avoid_negative_ts", "make_zero",
        str(out_clip),
    ]
    await _run_ffmpeg(cmd)
    return dur


async def make_preview(src_clip: str | Path, out_preview: Path, height: int | None = None) -> None:
    """재생용 경량 preview(h264, 지정 높이로 축소)."""
    h = height or config.PREVIEW_HEIGHT
    out_preview.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src_clip),
        "-vf", f"scale=-2:{h}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-movflags", "+faststart", "-an",
        str(out_preview),
    ]
    await _run_ffmpeg(cmd)
