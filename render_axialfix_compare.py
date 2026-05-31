"""
render_axialfix_compare.py

axial flip 보정 전/후 비교 영상 생성:
  Left panel:  stage4 원본 (smplx_merged_stage4.pt)
  Right panel: axial fix 결과 (smplx_merged_stage4_axfix.pt)

두 가지 출력:
  1. 전체 영상 (1495 frames) — render_axialfix_full_compare.mp4
  2. 스윙 구간 클립 (f320-420) — render_axialfix_swing_clip.mp4

각 패널에 프레임 번호 + axial 값 오버레이.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))
os.chdir(str(PROJ_ROOT))

IN_DIR   = PROJ_ROOT / "outputs/cap_pipeline/input"
VIDEO_IN = IN_DIR / "0_input_video.mp4"

ORIG_PT  = IN_DIR / "smplx_merged_stage4.pt"
FIX_PT   = IN_DIR / "smplx_merged_stage4_axfix.pt"

OUT_FULL  = IN_DIR / "render_axialfix_full_compare.mp4"
OUT_CLIP  = IN_DIR / "render_axialfix_swing_clip.mp4"

CLIP_START = 320
CLIP_END   = 430
MESH_ORIG  = (0.9, 0.6, 0.5)   # 붉은빛 — flip 있는 원본
MESH_FIX   = (0.5, 0.85, 0.6)  # 초록빛 — 보정본

LABEL_H   = 50
AX_THRESH = 80.0


# ─────────────────────────────────────────────────────────
# rotation helpers (standalone)
# ─────────────────────────────────────────────────────────

def aa_to_rotmat_np(aa: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(aa))
    if theta < 1e-9: return np.eye(3)
    ax = aa/theta; x,y,z = ax
    c,s,t = np.cos(theta),np.sin(theta),1-np.cos(theta)
    return np.array([[t*x*x+c,t*x*y-s*z,t*x*z+s*y],
                     [t*x*y+s*z,t*y*y+c,t*y*z-s*x],
                     [t*x*z-s*y,t*y*z+s*x,t*z*z+c]])

def euler_axial(aa: np.ndarray) -> float:
    R = aa_to_rotmat_np(aa)
    sy = float(np.clip(-R[2,0],-1,1)); cy = np.cos(np.arcsin(sy))
    return float(np.degrees(np.arctan2(R[2,1],R[2,2]))) if abs(cy)>1e-6 else \
           float(np.degrees(np.arctan2(-R[1,2],R[1,1])))

def filter_smplx(params: dict) -> dict:
    allowed = {"betas","body_pose","global_orient","transl",
               "left_hand_pose","right_hand_pose","jaw_pose","leye_pose","reye_pose"}
    return {k: v for k, v in params.items() if k in allowed}


# ─────────────────────────────────────────────────────────
# load & precompute
# ─────────────────────────────────────────────────────────
print("[render] Loading SMPL-X model ...")
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.net_utils import to_cuda
from hmr4d.utils.vis.renderer import Renderer
from hmr4d.utils.video_io_utils import get_video_fps, get_video_lwh, get_video_reader, get_writer

smplx_m = make_smplx("supermotion_fullhands").cuda()
faces   = smplx_m.faces

for label, pt_path in [("ORIG", ORIG_PT), ("FIX", FIX_PT)]:
    print(f"[render] SMPL-X forward ({label}) ...")
    pred = torch.load(pt_path, map_location="cpu")
    with torch.no_grad():
        out = smplx_m(**to_cuda(filter_smplx(pred["smpl_params_incam"])))
    if label == "ORIG":
        verts_orig = out.vertices.cpu()
        bp_orig    = pred["smpl_params_incam"]["body_pose"].numpy()
        K          = pred["K_fullimg"][0]
        F_total    = verts_orig.shape[0]
    else:
        verts_fix  = out.vertices.cpu()
        bp_fix     = pred["smpl_params_incam"]["body_pose"].numpy()

print(f"[render] Vertices: {verts_orig.shape}  K: {K.shape}")

# axial 사전 계산
ax_orig_l = np.array([euler_axial(bp_orig[f,19]) for f in range(F_total)])
ax_orig_r = np.array([euler_axial(bp_orig[f,20]) for f in range(F_total)])
ax_fix_l  = np.array([euler_axial(bp_fix[f,19])  for f in range(F_total)])
ax_fix_r  = np.array([euler_axial(bp_fix[f,20])  for f in range(F_total)])

fps      = get_video_fps(str(VIDEO_IN))
_, W, H  = get_video_lwh(str(VIDEO_IN))
print(f"[render] Video: {W}×{H} @ {fps:.2f}fps, {F_total}f")

renderer = Renderer(W, H, device="cuda", faces=faces, K=K, bin_size=0)


# ─────────────────────────────────────────────────────────
# helper: draw overlay label
# ─────────────────────────────────────────────────────────
def draw_label(frame: np.ndarray, fidx: int,
               ax_l: float, ax_r: float,
               title: str, title_col: tuple) -> np.ndarray:
    h, w = frame.shape[:2]
    bar = np.zeros((LABEL_H, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)

    flip_l = abs(ax_l) > AX_THRESH
    flip_r = abs(ax_r) > AX_THRESH
    l_col  = (0, 80, 255) if flip_l else (150, 255, 150)   # red if flip, green if ok
    r_col  = (0, 80, 255) if flip_r else (150, 255, 150)
    l_tag  = " FLIP" if flip_l else ""
    r_tag  = " FLIP" if flip_r else ""

    cv2.putText(bar, f"f{fidx:04d} | {title}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, title_col, 2, cv2.LINE_AA)
    cv2.putText(bar, f"L axial={ax_l:+.0f}°{l_tag}",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, l_col, 1, cv2.LINE_AA)
    cv2.putText(bar, f"R axial={ax_r:+.0f}°{r_tag}",
                (w//2, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, r_col, 1, cv2.LINE_AA)
    return np.concatenate([bar, frame], axis=0)


# ─────────────────────────────────────────────────────────
# render loop helper
# ─────────────────────────────────────────────────────────
def render_comparison(frame_range: range, out_path: Path) -> None:
    reader = get_video_reader(str(VIDEO_IN))
    cell_w, cell_h = W, H + LABEL_H
    writer = get_writer(str(out_path), fps=fps, crf=20)

    for fidx, img_raw in tqdm(enumerate(reader),
                               total=F_total, desc=f"→ {out_path.name}"):
        if fidx not in frame_range:
            if fidx > max(frame_range):
                break
            continue
        if fidx >= F_total:
            break

        v_o = verts_orig[fidx].cuda()
        v_f = verts_fix[fidx].cuda()

        cell_orig = renderer.render_mesh(v_o, img_raw, list(MESH_ORIG))
        cell_fix  = renderer.render_mesh(v_f, img_raw, list(MESH_FIX))

        cell_orig = draw_label(cell_orig, fidx,
                                ax_orig_l[fidx], ax_orig_r[fidx],
                                "BEFORE (stage4)", (180, 180, 255))
        cell_fix  = draw_label(cell_fix, fidx,
                                ax_fix_l[fidx], ax_fix_r[fidx],
                                "AFTER (axfix)", (180, 255, 180))

        # side-by-side
        grid = np.concatenate([cell_orig, cell_fix], axis=1)
        writer.write_frame(grid)

    writer.close()
    print(f"[render] Saved → {out_path}")


# ─────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────

# 1. 스윙 클립 (빠름)
print(f"\n[render] Rendering swing clip f{CLIP_START}-{CLIP_END} ...")
render_comparison(range(CLIP_START, CLIP_END + 1), OUT_CLIP)

# 2. 전체 영상 (느림)
print(f"\n[render] Rendering full compare ({F_total} frames) ...")
render_comparison(range(F_total), OUT_FULL)

print("\n[render] Done.")
print(f"  Swing clip : {OUT_CLIP}")
print(f"  Full video : {OUT_FULL}")
