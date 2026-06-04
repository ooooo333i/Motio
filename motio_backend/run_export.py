#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_export.py — 정제 .pt → (슬림 후처리 .npz) → (Blender) FBX. (cap-deploy subprocess)

Stage A(후처리 .npz): postprocess_pipeline/animate_postprocess.py 는 pyvista 를 import 하므로
  (cap-deploy 에 pyvista 미설치 + C1 으로 설치 불가) 여기서는 **동일한 rotation_postprocess
  (numpy/scipy 전용)** 로 npz 를 슬림 재현한다. 산출 npz 키는 export_to_unreal_fbx.py 계약과 동일:
  global_orient(T,3), body_pose(T,63), left/right_hand_pose(T,45), transl(T,3), betas, gender.

Stage B(FBX): Blender 가 설정(--blender or MOTIO_BLENDER)되어 있고 존재하면
  `blender -b smplx_template.blend -P export_to_unreal_fbx.py -- <npz> <out_dir>` 로 FBX 생성 후
  결과(*_smplx_unreal.fbx, _anim 제외)를 <out_dir>/output.fbx 로 복사. 없으면 npz 까지만.

usage: python run_export.py --final-pt <pt> --out-dir <dir> [--fps 30] [--blender <bin>] [--model-path <smplx.npz>]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PP_DIR = Path(__file__).resolve().parent.parent / "postprocess_pipeline"
sys.path.insert(0, str(PP_DIR))


# ── transl One-Euro (animate_postprocess.filter_translations 동등, pyvista import 회피) ──
class _OneEuro:
    def __init__(self, t0, x0, min_cutoff=0.8, beta=0.03, d_cutoff=1.0):
        import numpy as np
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.xp = x0.copy(); self.dxp = np.zeros_like(x0); self.tp = t0; self._np = np

    def _a(self, te, c):
        r = 2 * 3.141592653589793 * c * te
        return r / (r + 1)

    def __call__(self, t, x):
        te = t - self.tp
        if te <= 0:
            return x
        dx = (x - self.xp) / te
        ad = self._a(te, self.dc)
        dxh = ad * dx + (1 - ad) * self.dxp
        c = self.mc + self.beta * self._np.abs(dxh)
        a = self._a(te, c)
        xh = a * x + (1 - a) * self.xp
        self.xp, self.dxp, self.tp = xh.copy(), dxh.copy(), t
        return xh


def _smooth_transl(transl, fps):
    import numpy as np
    T = transl.shape[0]; dt = 1.0 / fps
    oef = _OneEuro(0.0, transl[0].copy())
    out = np.zeros_like(transl)
    for i in range(T):
        out[i] = oef(i * dt, transl[i])
    return out


def stage_a(final_pt: str, out_dir: Path, fps: float) -> Path:
    """슬림 후처리 → npz. 반환=npz 경로."""
    import numpy as np
    import torch
    from rotation_postprocess import smooth_rotations, stabilize_body_pose

    d = torch.load(final_pt, map_location="cpu")
    p = d["smpl_params_global"]

    def npy(k):
        v = p[k]
        return (v.numpy() if hasattr(v, "numpy") else np.asarray(v)).astype(np.float32)

    go = npy("global_orient")
    T = go.shape[0]
    bp = npy("body_pose").reshape(T, 21, 3)
    lh = npy("left_hand_pose").reshape(T, -1)
    rh = npy("right_hand_pose").reshape(T, -1)
    transl = npy("transl")
    betas = npy("betas")

    go_s = smooth_rotations(go.reshape(T, 1, 3), fps=fps).reshape(T, 3)
    bp_s = stabilize_body_pose(bp, fps=fps).reshape(T, 63)
    lh_s = smooth_rotations(lh.reshape(T, 15, 3), fps=fps).reshape(T, 45)
    rh_s = smooth_rotations(rh.reshape(T, 15, 3), fps=fps).reshape(T, 45)
    transl_s = _smooth_transl(transl, fps)
    # 접지는 stage_b 의 Blender 후처리(_ground_fbx)에서 실제 메시 Z 로 수행한다
    # (SMPL-X 공간 접지는 애드온/AMASS 변환이 더하는 수직 오프셋 때문에 최종 FBX 와 불일치).

    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "postprocessed.npz"
    np.savez(
        str(npz),
        global_orient=go_s, body_pose=bp_s,
        left_hand_pose=lh_s, right_hand_pose=rh_s,
        transl=transl_s, betas=betas, gender="neutral",
    )
    print(f"[EXPORT] stage A npz -> {npz}  (T={T}, fps={fps})")
    return npz


def stage_b(npz: Path, out_dir: Path, blender: str, fps: float = 30.0) -> Path | None:
    """Blender 로 npz → FBX. 반환=output.fbx 경로(or None)."""
    blend_tpl = PP_DIR / "smplx_template.blend"
    export_py = PP_DIR / "export_to_unreal_fbx.py"
    if not Path(blender).exists():
        print(f"[EXPORT] blender 없음({blender}) → Stage B 건너뜀 (npz 까지만).")
        return None
    if not blend_tpl.exists() or not export_py.exists():
        print(f"[EXPORT] template/export 스크립트 누락 → Stage B 건너뜀.")
        return None

    # Blender 는 cwd=PP_DIR 에서 실행되므로 npz/out_dir 는 절대경로로 전달(상대경로 오해석 방지)
    # ⚠️ --fps 를 반드시 전달해야 한다. 미전달 시 export 스크립트가 기본 30fps 로 굽고,
    #    SMPL-X 애드온이 mocap(30)→target(30)/ 또는 60→30 으로 다운샘플해 프레임 손실 + 슬로모션.
    #    실제 영상 fps 를 넘기면 scene fps=원본, 애드온 keep-fps 로 step_size=1(프레임 보존).
    cmd = [blender, "-b", str(blend_tpl), "-P", str(export_py), "--",
           str(npz.resolve()), str(out_dir.resolve()), "--fps", str(fps)]
    print(f"[EXPORT] stage B: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(PP_DIR), capture_output=True, text=True)
    sys.stdout.write(r.stdout[-3000:] if r.stdout else "")
    if r.returncode != 0:
        sys.stderr.write((r.stderr or "")[-3000:])
        print(f"[EXPORT] Stage B 실패(exit {r.returncode}).")
        return None

    # 산출 FBX(_anim 제외)를 output.fbx 로 복사
    cands = [p for p in out_dir.glob("*_smplx_unreal.fbx") if "_anim" not in p.name]
    if not cands:
        cands = list(out_dir.glob("*.fbx"))
    if not cands:
        print("[EXPORT] FBX 산출을 찾지 못함.")
        return None
    out_fbx = out_dir / "output.fbx"
    shutil.copy2(str(cands[0]), str(out_fbx))
    print(f"[EXPORT] FBX -> {out_fbx}")

    # 접지 후처리: 완성된 FBX 를 import 해 실제 메시 월드 Z 최저값으로 발을 지면(Z=0)에 앉힌다.
    # (export 의 bake_space_transform/축변환은 export 시점에 적용되므로, 라이브 씬이 아니라
    #  *완성 FBX*(Z-up)를 다시 읽어 접지해야 좌표계가 맞다 — in-proc 접지는 축이 어긋나 실패.)
    # CAP_GROUND=0 이면 접지 생략(공중부양 원본 유지).
    if os.environ.get("CAP_GROUND", "1") != "1":
        return out_fbx
    try:
        ground_py = Path(__file__).resolve().parent / "ground_fbx.py"
        grounded = out_dir / "output_grounded.fbx"
        gr = subprocess.run(
            [blender, "-b", "-P", str(ground_py), "--",
             str(out_fbx.resolve()), str(grounded.resolve()), str(fps)],
            cwd=str(PP_DIR), capture_output=True, text=True, timeout=600)
        sys.stderr.write((gr.stderr or "")[-1500:])
        if gr.returncode == 0 and grounded.exists() and grounded.stat().st_size > 0:
            shutil.move(str(grounded), str(out_fbx))
            print(f"[EXPORT] 접지 완료 -> {out_fbx}")
        else:
            print(f"[EXPORT] 접지 생략(rc={gr.returncode}) — 원본 FBX 유지.")
    except Exception as exc:
        print(f"[EXPORT] 접지 후처리 실패(무시): {type(exc).__name__}: {exc}")
    return out_fbx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="정제 .pt → 후처리 npz → (Blender) FBX")
    ap.add_argument("--final-pt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--blender", default=os.environ.get("MOTIO_BLENDER", ""))
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    try:
        npz = stage_a(args.final_pt, out_dir, args.fps)
    except Exception as exc:
        print(f"[EXPORT_ERROR] stage A 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.blender:
        try:
            stage_b(npz, out_dir, args.blender, fps=args.fps)
        except Exception as exc:
            print(f"[EXPORT_WARN] stage B 실패(무시): {type(exc).__name__}: {exc}", file=sys.stderr)
    else:
        print("[EXPORT] MOTIO_BLENDER 미설정 → npz 까지만(FBX 미생성).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
