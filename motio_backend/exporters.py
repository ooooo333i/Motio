"""FBX/BVH export 훅.

정제 .pt → run_export.py(슬림 후처리 .npz → Blender FBX) → out_dir/output.fbx.
Blender(MOTIO_BLENDER) 미설정/미설치면 npz 까지만 만들고 fbx=None(명세상 null 허용).
BVH 는 확장 지점(현재 미생성) — 추후 npz/pt → BVH 변환을 여기 끼운다.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import config, gpu


async def export_artifacts_async(final_pt: str | Path, out_dir: str | Path,
                                 log_path: str | Path | None = None,
                                 fps: float = 30.0) -> dict:
    """run_export.py 를 subprocess 로 실행(GPU 불필요). 반환={'fbx': path|None, 'bvh': path|None}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "run_export.py"
    cmd = [
        config.CAP_PYTHON, str(script),
        "--final-pt", str(final_pt),
        "--out-dir", str(out_dir),
        "--fps", str(fps),
    ]
    blender = os.environ.get("MOTIO_BLENDER", "")
    if blender:
        cmd += ["--blender", blender]
    # export 는 GPU 불필요 → 락 없이 실행
    await gpu.run_logged(cmd, log_path, config.REPO_ROOT)
    return _collect(out_dir)


def _collect(out_dir: Path) -> dict:
    fbx = out_dir / "output.fbx"
    bvh = out_dir / "output.bvh"
    return {
        "fbx": str(fbx) if fbx.exists() else None,
        "bvh": str(bvh) if bvh.exists() else None,
    }


def export_motion_artifacts(final_pt: str | Path, out_dir: str | Path) -> dict:
    """동기 호환 훅(이미 생성된 산출물 인식). jobs 는 export_artifacts_async 를 사용."""
    return _collect(Path(out_dir))
