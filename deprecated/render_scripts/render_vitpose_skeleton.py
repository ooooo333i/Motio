#!/usr/bin/env python3
"""Render ViTPose arm + hand skeleton on the original video.

Body arm keypoints: COCO17 (from preprocess/vitpose.pt)
Hand keypoints:     WholeBody133 indices 91-111 (left), 112-132 (right)

Output: outputs/cap_pipeline/input/vitpose_skeleton.mp4
"""
import os, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch

PROJ_ROOT  = Path(__file__).resolve().parent
OUT_DIR    = PROJ_ROOT / 'outputs/cap_pipeline/input'
VIDEO_IN   = OUT_DIR / '0_input_video.mp4'
VIDEO_OUT  = OUT_DIR / 'vitpose_skeleton.mp4'
WB_CACHE   = OUT_DIR / 'preprocess/wholebody_kp2d.npy'  # cached wholebody output
HAMER_ROOT = Path(os.environ.get('HAMER_ROOT', PROJ_ROOT / 'external/hamer'))

# ── COCO17 arm connections (shoulder→elbow→wrist) ─────────────────────────────
# COCO17 indices: 5=L_shoulder,6=R_shoulder,7=L_elbow,8=R_elbow,9=L_wrist,10=R_wrist
COCO_ARM_BONES = [
    (5, 7, (255, 80, 80)),    # left  shoulder→elbow  (blue)
    (7, 9, (255, 80, 80)),    # left  elbow→wrist     (blue)
    (6, 8, (80, 80, 255)),    # right shoulder→elbow  (red)
    (8, 10, (80, 80, 255)),   # right elbow→wrist     (red)
]
COCO_ARM_JOINTS = {
    5: (255, 80, 80), 7: (255, 80, 80), 9: (255, 80, 80),   # left  arm: blue
    6: (80, 80, 255), 8: (80, 80, 255), 10: (80, 80, 255),  # right arm: red
}

# ── MANO hand skeleton (21 keypoints per hand) ────────────────────────────────
# 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
HAND_CHAINS = [
    [0, 1, 2, 3, 4],    # thumb
    [0, 5, 6, 7, 8],    # index
    [0, 9, 10, 11, 12], # middle
    [0, 13, 14, 15, 16],# ring
    [0, 17, 18, 19, 20],# pinky
]
# Left hand = blue shades, Right hand = red shades
HAND_COLOR = {
    'left':  (200, 100, 50),
    'right': (50,  100, 200),
}

MIN_CONF = 0.3


def load_or_extract_wholebody(frame_dir: Path, bbx_path: Path) -> np.ndarray:
    """Load cached wholebody kp2d or extract it fresh. Returns (N,133,3)."""
    if WB_CACHE.exists():
        print(f"Loading cached wholebody kp2d: {WB_CACHE}")
        return np.load(str(WB_CACHE))

    print("Extracting wholebody kp2d (this takes a few minutes)...")
    sys.path.insert(0, str(PROJ_ROOT))
    import entry_point_hamer as eh
    kp2d_coco = torch.load(str(OUT_DIR / 'preprocess/vitpose.pt'), map_location='cpu')
    kp2d_coco = kp2d_coco.numpy() if hasattr(kp2d_coco, 'numpy') else np.asarray(kp2d_coco)
    kp2d_133, elapsed = eh._extract_wholebody_kp2d(
        frame_dir=frame_dir,
        kp2d_coco17=kp2d_coco,
        bbx_path=bbx_path,
        hamer_root=HAMER_ROOT,
        min_conf=MIN_CONF,
    )
    np.save(str(WB_CACHE), kp2d_133)
    print(f"Saved wholebody kp2d to {WB_CACHE}  ({elapsed:.1f}s)")
    return kp2d_133


def draw_joints_and_bones(img, pts, bones, joint_colors, min_conf=MIN_CONF, radius=6, thickness=3):
    """pts: list of (x,y,conf). bones: list of (i,j,color)."""
    h, w = img.shape[:2]
    # Draw bones first (underneath joints)
    for (i, j, color) in bones:
        if i >= len(pts) or j >= len(pts):
            continue
        xi, yi, ci = pts[i]
        xj, yj, cj = pts[j]
        if ci < min_conf or cj < min_conf:
            continue
        if not (0 <= xi < w and 0 <= yi < h and 0 <= xj < w and 0 <= yj < h):
            continue
        cv2.line(img, (int(xi), int(yi)), (int(xj), int(yj)), color, thickness, cv2.LINE_AA)
    # Draw joints
    for idx, color in joint_colors.items():
        if idx >= len(pts):
            continue
        x, y, c = pts[idx]
        if c < min_conf:
            continue
        if not (0 <= x < w and 0 <= y < h):
            continue
        cv2.circle(img, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hand(img, hand_kp21, side, min_conf=MIN_CONF, radius=5, thickness=2):
    """hand_kp21: (21,3) array (x,y,conf). side: 'left' or 'right'."""
    h, w = img.shape[:2]
    color = HAND_COLOR[side]
    for chain in HAND_CHAINS:
        for k in range(len(chain) - 1):
            i, j = chain[k], chain[k + 1]
            xi, yi, ci = hand_kp21[i]
            xj, yj, cj = hand_kp21[j]
            if ci < min_conf or cj < min_conf:
                continue
            if not (0 <= xi < w and 0 <= yi < h and 0 <= xj < w and 0 <= yj < h):
                continue
            cv2.line(img, (int(xi), int(yi)), (int(xj), int(yj)), color, thickness, cv2.LINE_AA)
    for i in range(21):
        x, y, c = hand_kp21[i]
        if c < min_conf:
            continue
        if not (0 <= x < w and 0 <= y < h):
            continue
        r = radius + 1 if i == 0 else radius  # wrist slightly larger
        cv2.circle(img, (int(x), int(y)), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), r + 1, (0, 0, 0), 1, cv2.LINE_AA)


# ── Load data ─────────────────────────────────────────────────────────────────
frame_dir = OUT_DIR / 'hamer_frames'
bbx_path  = OUT_DIR / 'preprocess/bbx.pt'

# COCO17 keypoints (body)
kp_coco = torch.load(str(OUT_DIR / 'preprocess/vitpose.pt'), map_location='cpu')
kp_coco = kp_coco.numpy() if hasattr(kp_coco, 'numpy') else np.asarray(kp_coco)
print(f"COCO17 shape: {kp_coco.shape}")

# WholeBody133 (arm + hand)
kp_wb = load_or_extract_wholebody(frame_dir, bbx_path)
print(f"WholeBody133 shape: {kp_wb.shape}")

# ── Render ─────────────────────────────────────────────────────────────────────
from hmr4d.utils.video_io_utils import get_video_fps
fps         = get_video_fps(str(VIDEO_IN))
frame_paths = sorted(frame_dir.glob('*.jpg'))
N           = min(len(frame_paths), len(kp_coco), len(kp_wb))

sample = cv2.imread(str(frame_paths[0]))
H, W   = sample.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (W, H))

print(f"Rendering {N} frames → {VIDEO_OUT.name}")
tic = time.perf_counter()
for i, fp in enumerate(frame_paths[:N]):
    img = cv2.imread(str(fp))

    # ── Arm skeleton from COCO17 ──────────────────────────────────────────
    coco_pts = kp_coco[i].tolist() if hasattr(kp_coco[i], 'tolist') else list(kp_coco[i])
    draw_joints_and_bones(img, coco_pts, COCO_ARM_BONES, COCO_ARM_JOINTS)

    # ── Hand skeleton from WholeBody133 ───────────────────────────────────
    wb = kp_wb[i]  # (133,3)
    draw_hand(img, wb[91:112],  'left')   # indices 91-111
    draw_hand(img, wb[112:133], 'right')  # indices 112-132

    writer.write(img)
    if (i + 1) % 200 == 0:
        elapsed = time.perf_counter() - tic
        print(f"  {i+1}/{N}  {(i+1)/elapsed:.1f}fps  ETA {(N-i-1)/((i+1)/elapsed):.0f}s")

writer.release()
elapsed = time.perf_counter() - tic
print(f"Done in {elapsed:.1f}s  ({N/elapsed:.1f} fps)")
print(f"Saved: {VIDEO_OUT}")
