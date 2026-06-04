"""검출 어댑터: 사람 검출 실행 → 검출 JSON 저장/반환.

기본은 **in-process**(YOLO 모델 1회 상주, 매 호출 재로딩 제거 → detect ~3.6s↓).
import/CUDA 등 어떤 이유로든 실패하면 기존 **서브프로세스**로 자동 폴백(품질·동작 동일).
강제 서브프로세스: 환경변수 CAP_DETECT_SUBPROCESS=1.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from . import config, gpu, storage


def _detect_inproc(clip_path: str, out_json: str, log_path: str) -> None:
    """서버 프로세스 내에서 캐시된 YOLO 모델로 검출(스레드에서 호출)."""
    if str(config.REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(config.REPO_ROOT))
    from . import detect_people  # 같은 패키지(모듈 캐시 → 모델 1회 로딩)
    t0 = time.time()
    detect_people.detect(clip_path, out_json)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%H:%M:%S}] [DETECT] in-process (warm model) {time.time()-t0:.1f}s\n")


async def run_detection(clip_id: str, clip_path: str | Path) -> dict:
    """클립에 대해 사람 검출 실행(단일 GPU 락). 결과 dict 반환."""
    out_json = storage.detection_path(clip_id)
    log_path = storage.clip_dir(clip_id) / "detect.log"

    # 1) in-process (warm model) 시도
    if os.environ.get("CAP_DETECT_SUBPROCESS") != "1":
        try:
            async with gpu.GPU_LOCK:
                await asyncio.to_thread(_detect_inproc, str(clip_path), str(out_json), str(log_path))
            data = storage.load_detection(clip_id)
            if data is not None:
                return data
        except Exception as exc:  # noqa: BLE001 — 어떤 실패든 서브프로세스로 폴백
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now():%H:%M:%S}] [DETECT] in-process 실패 → 서브프로세스 폴백: {exc}\n")
            except Exception:
                pass

    # 2) 서브프로세스 폴백(기존 경로)
    cmd = [
        config.CAP_PYTHON, str(config.DETECT_SCRIPT),
        "--clip", str(clip_path),
        "--out", str(out_json),
    ]
    async with gpu.GPU_LOCK:
        rc, output = await gpu.run_logged(cmd, log_path, config.REPO_ROOT)
    if rc != 0 or not out_json.exists():
        raise RuntimeError(f"detection failed (exit {rc}). tail:\n{output[-1500:]}")
    data = storage.load_detection(clip_id)
    if data is None:
        raise RuntimeError("detection produced no JSON")
    return data
