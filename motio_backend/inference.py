"""추론 어댑터: entry_point_opt.py(cap-deploy) subprocess 호출 → manifest 파싱."""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import config, gpu, storage


def _parse_manifest(output: str, out_root: Path) -> dict | None:
    for line in output.splitlines():
        if line.startswith("[OPT_MANIFEST]"):
            try:
                return json.loads(line.split("[OPT_MANIFEST]", 1)[1].strip())
            except Exception:
                pass
    # fallback: 산출 폴더의 opt_manifest.json
    cands = sorted(out_root.glob("*/opt_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        try:
            return json.loads(cands[0].read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


async def run_extraction(job_id: str, clip_path: str | Path, person_track_id: int | None) -> dict:
    """추론 1건 실행(단일 GPU 락). entry_point_opt manifest(dict) 반환."""
    out_root = storage.job_output_dir(job_id)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = storage.job_dir(job_id) / "pipeline.log"

    # entry_point_opt 기본값: auto_person=True, no_interactive=True (추가 플래그 불필요)
    # 기본 mode=quality(=adaptive v4): 독립 MediaPipe 2D 최고·twist/충돌 최저 검증됨(poster/V4_EVIDENCE.md).
    # 되돌리기: 환경변수 CAP_OPT_MODE=fast (기존 fit2d+finger_kalman 경로).
    opt_mode = os.environ.get("CAP_OPT_MODE", "quality")
    cmd = [
        config.CAP_PYTHON, str(config.ENTRY_POINT_OPT),
        "--video", str(clip_path),
        "--output-root", str(out_root),
        "--mode", opt_mode,
    ]
    if person_track_id is not None:
        cmd += ["--person-track-id", str(person_track_id)]

    # YOLO 재추적 스킵: 검출 단계 raw 트랙(_tracks.npz)을 추출이 재사용 → YOLO 1패스(~20-30s) 절약.
    # 기본 ON. 두 YOLO 트래킹이 완전 동일하진 않아 손 포즈 한 축 최악 ~30° 차이 가능하나(mean<0.7°),
    # side·front 영상 오버레이 검증에서 육안 차이 없음 확인(2026-06-03). 되돌리기: CAP_REUSE_DETECTION=0.
    env = None
    if os.environ.get("CAP_REUSE_DETECTION", "1") != "0" and person_track_id is not None:
        clip_id = Path(clip_path).parent.name
        tracks_npz = config.DETECT_DIR / f"{clip_id}_tracks.npz"
        if tracks_npz.exists():
            env = {"CAP_REUSE_BBX_NPZ": str(tracks_npz)}

    async with gpu.GPU_LOCK:
        rc, output = await gpu.run_logged(cmd, log_path, config.REPO_ROOT, env=env)
    if rc != 0:
        raise RuntimeError(f"inference failed (exit {rc}). tail:\n{output[-1500:]}")

    manifest = _parse_manifest(output, out_root)
    if manifest is None or not manifest.get("outputs", {}).get("final"):
        raise RuntimeError("inference finished but manifest/final output missing")
    return manifest
