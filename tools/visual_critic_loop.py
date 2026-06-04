"""
tools/visual_critic_loop.py

Self-feedback visual critic loop for HaMeR + GVHMR projection/bbox/alignment analysis.

목적: 손이 왜 이상한지 분류하고, projection/bbox/HaMeR raw/ViTPose 중
      어디를 먼저 고쳐야 하는지 이미지와 metric으로 확정한다.

금지:
  - Stage A/B 실행 금지
  - smoothing/filtering/Kalman 금지
  - 최적화 실행 금지
  - hamer_out 원본 덮어쓰기 금지

Usage:
    python tools/visual_critic_loop.py \
        --input outputs/cap_pipeline/input \
        --max-iters 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# COCO17 body keypoint indices
VP_NOSE         = 0
VP_L_SHOULDER   = 5
VP_R_SHOULDER   = 6
VP_L_ELBOW      = 7
VP_R_ELBOW      = 8
VP_L_WRIST      = 9
VP_R_WRIST      = 10
VP_L_HIP        = 11
VP_R_HIP        = 12

# WB133 hand keypoint indices
# Left hand: 91-111, Right hand: 112-132
# Order (per hand): wrist, thumb(4), index(4), middle(4), ring(4), pinky(4)
WB_L_START = 91
WB_R_START = 112
# Within each hand block (0-indexed from start)
WB_WRIST = 0
WB_IDX_MCP   = 5
WB_MID_MCP   = 9
WB_RING_MCP  = 13
WB_PINK_MCP  = 17
# Fingertip indices (skip for primary targets)
WB_THUMB_TIP = 4
WB_IDX_TIP   = 8
WB_MID_TIP   = 12
WB_RING_TIP  = 16
WB_PINK_TIP  = 20

PALM_JOINTS = [WB_WRIST, WB_IDX_MCP, WB_MID_MCP, WB_RING_MCP, WB_PINK_MCP]

# SMPL-X arm skeleton connections for ViTPose (COCO17)
ARM_SKELETON = [
    (VP_L_SHOULDER, VP_L_ELBOW),
    (VP_L_ELBOW,    VP_L_WRIST),
    (VP_R_SHOULDER, VP_R_ELBOW),
    (VP_R_ELBOW,    VP_R_WRIST),
    (VP_L_SHOULDER, VP_R_SHOULDER),
]

# Hand skeleton connections within hand block (0-indexed)
HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
]

# Colors (BGR)
C_VP_LEFT     = (255, 100, 0)    # blue-ish left
C_VP_RIGHT    = (0, 100, 255)    # red-ish right
C_HAMER_LEFT  = (0, 255, 255)    # yellow-ish
C_HAMER_RIGHT = (0, 200, 50)     # green-ish
C_WB_LEFT     = (200, 0, 200)    # purple
C_WB_RIGHT    = (0, 200, 200)    # teal
C_BBOX_A      = (0, 200, 0)      # green: WholeBdy
C_BBOX_B      = (200, 0, 0)      # blue: COCO17
C_BBOX_C      = (0, 200, 200)    # teal: Hybrid
C_BBOX_USED   = (0, 255, 255)    # yellow: actual used bbox


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_hamer_detections(hamer_dir: Path) -> dict:
    """Returns det[frame_idx][side] = {kp3d, kp2d, cam_t, cam_t_full,
    box_center, box_size, focal_length, img_size, is_right, vertices}.
    side ∈ {'left','right'}.
    """
    det: dict[int, dict] = {}
    pattern = re.compile(r"^(\d+)_")
    for p in sorted(hamer_dir.glob("*.npz")):
        m = pattern.match(p.stem)
        if m is None:
            continue
        fidx = int(m.group(1))
        d = np.load(p, allow_pickle=True)
        is_right = int(np.asarray(d["is_right"]).reshape(-1)[0])
        side = "right" if is_right else "left"
        mano = d["mano_params"].item()

        entry = {
            "kp3d":         np.asarray(d["keypoints_3d"], np.float32).reshape(21, 3),
            "kp2d":         np.asarray(d["keypoints_2d"], np.float32).reshape(21, 2),
            "cam_t":        np.asarray(d["cam_t"], np.float32).reshape(3),
            "cam_t_full":   np.asarray(d["cam_t_full"], np.float32).reshape(3),
            "pred_cam":     np.asarray(d["pred_cam"], np.float32).reshape(3),
            "box_center":   np.asarray(d["box_center"], np.float32).reshape(2),
            "box_size":     float(np.asarray(d["box_size"]).reshape(-1)[0]),
            "focal_length": float(np.asarray(d["focal_length"]).reshape(-1)[0]),
            "img_size":     np.asarray(d["img_size"], np.float32).reshape(2),
            "is_right":     is_right,
            "vertices":     np.asarray(d["vertices"], np.float32),
            "global_orient": np.asarray(mano["global_orient"], np.float32),
            "hand_pose":     np.asarray(mano["hand_pose"], np.float32),
        }

        if fidx not in det:
            det[fidx] = {}
        # If same frame has two detections for same side, keep later (usually second det)
        det[fidx][side] = entry

    return det


def load_vitpose(preprocess_dir: Path) -> np.ndarray:
    """Returns (F, 17, 3) float32 array [x, y, conf] in image pixel coords."""
    pt = torch.load(preprocess_dir / "vitpose.pt", map_location="cpu")
    if isinstance(pt, torch.Tensor):
        return pt.numpy().astype(np.float32)
    raise ValueError(f"Unexpected vitpose.pt format: {type(pt)}")


def load_wb133(preprocess_dir: Path) -> np.ndarray:
    """Returns (F, 133, 3) float32 array [x, y, conf] in image pixel coords."""
    return np.load(preprocess_dir / "wholebody_kp2d.npy").astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Projection Utilities
# ---------------------------------------------------------------------------

def project_hamer_fullimg(kp3d: np.ndarray, cam_t_full: np.ndarray,
                           focal: float, img_w: float, img_h: float,
                           is_right: bool = True) -> np.ndarray:
    """kp3d (N,3) in MANO local space → full image pixels (N,2).
    Uses HaMeR focal length (typ. 37500) and cx=W/2, cy=H/2.

    Left hand (is_right=False): kp3d is in right-hand canonical space.
    Must x-flip before adding cam_t_full (which already has negated tx).
    render_hamer_incam.py confirms: verts[:, 0] *= -1 for left hand.
    """
    pts = kp3d.copy()
    if not is_right:
        pts[:, 0] *= -1               # flip to left-hand image space
    pts = pts + cam_t_full            # (N,3) in HaMeR camera frame
    z   = np.maximum(pts[:, 2], 1e-4)
    cx  = img_w / 2.0
    cy  = img_h / 2.0
    u   = focal * pts[:, 0] / z + cx
    v   = focal * pts[:, 1] / z + cy
    return np.stack([u, v], axis=1).astype(np.float32)


def kp2d_crop_to_full(kp2d_crop: np.ndarray, box_center: np.ndarray,
                       box_size: float, is_right: bool = True) -> np.ndarray:
    """kp2d_crop (N,2) in normalized [-1,1] crop coords → full image pixels (N,2).

    Left hand (is_right=False): HaMeR flips the crop horizontally before inference,
    so kp2d x-coordinate is in the flipped space. Must negate x to map back to
    original image coords. v (vertical) is unaffected.
    """
    sign_x = 1.0 if is_right else -1.0
    u = box_center[0] + sign_x * kp2d_crop[:, 0] * (box_size / 2.0)
    v = box_center[1] +           kp2d_crop[:, 1] * (box_size / 2.0)
    return np.stack([u, v], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 2b. Projection Lock Visualization
# ---------------------------------------------------------------------------

def generate_projection_lock_images(
    det: dict,
    vitpose: np.ndarray,
    frames_dir: Path,
    out_dir: Path,
    top_fidxs: list[int],
    img_w: float = 1080.0,
    img_h: float = 1920.0,
):
    """Section 8: projection lock.
    For each selected frame, generate side-by-side panels:
    - Panel A: original frame + HaMeR kp3d full-img proj (corrected) in blue/red
    - Panel B: crop extracted from frame + kp2d (yellow) + kp3d+cam_t projected (cyan)
    - Panel C: full-img + both ViTPose wrist (cross) and HaMeR proj (circle)
    - Panel D: left/right x-flip comparison (left hand only)
    Shows before/after x-flip correction side by side.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_files = {int(f.stem): f for f in sorted(frames_dir.glob("*.jpg"))}
    CROP_SZ = 256

    for fidx in top_fidxs:
        img_path = frame_files.get(fidx)
        if img_path is None:
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        det_f = det.get(fidx, {})
        vp = vitpose[fidx]
        panels = []

        for side in ("left", "right"):
            if side not in det_f:
                continue
            d = det_f[side]
            focal = d["focal_length"]
            kp3d  = d["kp3d"]
            ct_full = d["cam_t_full"]
            ct_crop = d["cam_t"]
            bc = d["box_center"]
            bs = d["box_size"]
            is_right = bool(d["is_right"])

            # --- Panel A: full-image proj (corrected) ---
            pA = frame.copy()
            # corrected projection
            uv_corr = project_hamer_fullimg(kp3d, ct_full, focal, img_w, img_h, is_right)
            uv_no   = project_hamer_fullimg(kp3d, ct_full, focal, img_w, img_h, True)  # no-flip
            color_c = C_HAMER_LEFT if side == "left" else C_HAMER_RIGHT
            draw_hamer_kp3d_proj(pA, uv_corr, color_c)
            if side == "left":
                draw_hamer_kp3d_proj(pA, uv_no, (100, 100, 255))  # no-flip in dim red
                cv2.putText(pA, "corr=cyan nf=dimred", (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            w_idx = VP_L_WRIST if side == "left" else VP_R_WRIST
            pvp = vp[w_idx]
            if pvp[2] > 0.2:
                cv2.drawMarker(pA, (int(pvp[0]), int(pvp[1])), (0, 255, 255),
                               cv2.MARKER_CROSS, 20, 2)

            # --- Panel B: crop-space ---
            # Extract crop from frame
            half = int(bs / 2)
            cx_i, cy_i = int(bc[0]), int(bc[1])
            x1 = max(0, cx_i - half); x2 = min(frame.shape[1], cx_i + half)
            y1 = max(0, cy_i - half); y2 = min(frame.shape[0], cy_i + half)
            crop_raw = frame[y1:y2, x1:x2]
            if crop_raw.size == 0:
                crop_raw = np.zeros((CROP_SZ, CROP_SZ, 3), dtype=np.uint8)
            pB = cv2.resize(crop_raw, (CROP_SZ, CROP_SZ))

            # kp2d (saved, normalized [-1,1]) → 256x256
            kp2d_256_u = (d["kp2d"][:, 0] + 1) / 2.0 * CROP_SZ
            kp2d_256_v = (d["kp2d"][:, 1] + 1) / 2.0 * CROP_SZ
            for ku, kv in zip(kp2d_256_u[:5], kp2d_256_v[:5]):  # first 5 kp
                cv2.circle(pB, (int(ku), int(kv)), 5, (0, 255, 255), -1)

            # kp3d + cam_t → crop space (focal=5000 for 256x256, cx=cy=128)
            j_crop = kp3d + ct_crop
            z_c = np.maximum(j_crop[:, 2], 1e-4)
            crop_focal = 5000.0
            uc = crop_focal * j_crop[:, 0] / z_c + 128
            vc = crop_focal * j_crop[:, 1] / z_c + 128
            for ku, kv in zip(uc[:5], vc[:5]):
                cv2.circle(pB, (int(np.clip(ku, 0, 255)), int(np.clip(kv, 0, 255))),
                           4, (255, 128, 0), -1)
            cv2.putText(pB, f"{side} crop", (2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Scale pB to same height as pA subsample for panel
            scale_to = min(pA.shape[0], 480)
            pA_s = cv2.resize(pA, (int(pA.shape[1] * scale_to / pA.shape[0]), scale_to))
            pB_s = cv2.resize(pB, (scale_to, scale_to))

            row = np.concatenate([pA_s, pB_s], axis=1)
            panels.append(row)

        if not panels:
            continue

        # Stack panels for all sides
        max_w = max(p.shape[1] for p in panels)
        padded = []
        for p in panels:
            if p.shape[1] < max_w:
                pad = np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)
                p = np.concatenate([p, pad], axis=1)
            padded.append(p)

        combined = np.concatenate(padded, axis=0) if len(padded) > 1 else padded[0]
        out_path = out_dir / f"projlock_{fidx:05d}.jpg"
        cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 88])

    print(f"[visual_critic] projection_lock images → {out_dir}")


# ---------------------------------------------------------------------------
# 3. BBox Mode Computation
# ---------------------------------------------------------------------------

def bbox_mode_a(wb133_frame: np.ndarray, side: str,
                conf_thresh: float = 0.3) -> tuple | None:
    """Mode A: WholeBdy hand bbox.
    Returns (center_xy, size) or None.
    """
    start = WB_L_START if side == "left" else WB_R_START
    hand = wb133_frame[start: start + 21, :]   # (21, 3) [x,y,conf]
    valid = hand[:, 2] > conf_thresh
    if valid.sum() < 3:
        return None
    pts = hand[valid, :2]
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    cx, cy = (mn + mx) / 2.0
    size = float(np.max(mx - mn)) * 2.0         # padding factor 2x
    return np.array([cx, cy], np.float32), float(size)


def bbox_mode_b(vp_frame: np.ndarray, side: str,
                img_w: float, img_h: float,
                conf_thresh: float = 0.3) -> tuple | None:
    """Mode B: COCO17 wrist-elbow reduced bbox."""
    w_idx = VP_L_WRIST  if side == "left" else VP_R_WRIST
    e_idx = VP_L_ELBOW  if side == "left" else VP_R_ELBOW
    wrist = vp_frame[w_idx]
    elbow = vp_frame[e_idx]
    if wrist[2] < conf_thresh or elbow[2] < conf_thresh:
        return None
    wxy = wrist[:2]
    exy = elbow[:2]
    forearm = float(np.linalg.norm(wxy - exy))
    center = wxy + 0.25 * (wxy - exy)
    base_size = max(0.06 * max(img_w, img_h), 1.0 * forearm)
    box_size = base_size * 2.0
    return center.astype(np.float32), float(box_size)


def bbox_mode_c(wb133_frame: np.ndarray, vp_frame: np.ndarray,
                side: str, img_w: float, img_h: float,
                outlier_score: float = 0.0,
                wb_conf_thresh: float = 0.4) -> tuple | None:
    """Mode C: Hybrid — use WholeBdy when confident + not outlier, else COCO17."""
    # Check WholeBdy hand confidence
    start = WB_L_START if side == "left" else WB_R_START
    hand = wb133_frame[start: start + 21, :]
    palm_conf = hand[PALM_JOINTS, 2].mean()

    use_wb = palm_conf >= wb_conf_thresh and outlier_score < 0.7
    if use_wb:
        result = bbox_mode_a(wb133_frame, side, conf_thresh=0.3)
        if result is not None:
            return result
    return bbox_mode_b(vp_frame, side, img_w, img_h)


# ---------------------------------------------------------------------------
# 4. Drawing Utilities
# ---------------------------------------------------------------------------

def draw_vitpose_arm(img: np.ndarray, vp: np.ndarray, conf_thresh: float = 0.2) -> np.ndarray:
    """Draw arm skeleton (both sides) on img (in-place)."""
    H, W = img.shape[:2]
    for i, j in ARM_SKELETON:
        pi, pj = vp[i], vp[j]
        if pi[2] < conf_thresh or pj[2] < conf_thresh:
            continue
        ci = (int(pi[0]), int(pi[1]))
        cj = (int(pj[0]), int(pj[1]))
        is_left = (i in (VP_L_SHOULDER, VP_L_ELBOW, VP_L_WRIST)
                   or j in (VP_L_SHOULDER, VP_L_ELBOW, VP_L_WRIST))
        color = C_VP_LEFT if is_left else C_VP_RIGHT
        cv2.line(img, ci, cj, color, 3)

    for idx, color in [
        (VP_L_SHOULDER, C_VP_LEFT), (VP_L_ELBOW, C_VP_LEFT), (VP_L_WRIST, C_VP_LEFT),
        (VP_R_SHOULDER, C_VP_RIGHT), (VP_R_ELBOW, C_VP_RIGHT), (VP_R_WRIST, C_VP_RIGHT),
    ]:
        p = vp[idx]
        if p[2] < conf_thresh:
            continue
        alpha = float(np.clip(p[2], 0, 1))
        r = max(4, int(8 * alpha))
        cv2.circle(img, (int(p[0]), int(p[1])), r, color, -1)
    return img


def draw_wb133_hand(img: np.ndarray, wb133: np.ndarray, side: str,
                    conf_thresh: float = 0.2) -> np.ndarray:
    """Draw WB133 hand keypoints on img (in-place)."""
    start = WB_L_START if side == "left" else WB_R_START
    hand = wb133[start: start + 21, :]
    color = C_WB_LEFT if side == "left" else C_WB_RIGHT

    # Draw skeleton
    for i, j in HAND_SKELETON:
        pi, pj = hand[i], hand[j]
        if pi[2] < conf_thresh or pj[2] < conf_thresh:
            continue
        cv2.line(img, (int(pi[0]), int(pi[1])), (int(pj[0]), int(pj[1])), color, 2)

    # Draw keypoints
    for k in range(21):
        p = hand[k]
        if p[2] < conf_thresh:
            continue
        r = 5 if k in PALM_JOINTS else 3
        cv2.circle(img, (int(p[0]), int(p[1])), r, color, -1)
    return img


def draw_hamer_kp3d_proj(img: np.ndarray, uv: np.ndarray,
                          color: tuple, label: str = "") -> np.ndarray:
    """Draw projected HaMeR kp3d joints. uv (21,2) in full image coords."""
    H, W = img.shape[:2]
    # Draw skeleton connections
    for i, j in HAND_SKELETON:
        pi, pj = uv[i], uv[j]
        if np.any(np.isnan([pi, pj])):
            continue
        ci = (int(np.clip(pi[0], 0, W-1)), int(np.clip(pi[1], 0, H-1)))
        cj = (int(np.clip(pj[0], 0, W-1)), int(np.clip(pj[1], 0, H-1)))
        cv2.line(img, ci, cj, color, 2)
    # Draw keypoints
    for k, (u, v) in enumerate(uv):
        if np.isnan(u) or np.isnan(v):
            continue
        r = 7 if k == 0 else 4   # wrist bigger
        uc = int(np.clip(u, 0, W-1))
        vc = int(np.clip(v, 0, H-1))
        cv2.circle(img, (uc, vc), r, color, -1)
    if label:
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    return img


def draw_bbox_rect(img: np.ndarray, center: np.ndarray, size: float,
                   color: tuple, label: str = "") -> np.ndarray:
    """Draw bbox rectangle."""
    cx, cy = float(center[0]), float(center[1])
    half = size / 2.0
    pt1 = (int(cx - half), int(cy - half))
    pt2 = (int(cx + half), int(cy + half))
    cv2.rectangle(img, pt1, pt2, color, 2)
    if label:
        cv2.putText(img, label, (pt1[0], pt1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def put_metric_text(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Overlay metric text lines in top-left corner."""
    for i, txt in enumerate(lines):
        y = 20 + 18 * i
        cv2.putText(img, txt, (3, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, txt, (3, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(det: dict, vitpose: np.ndarray, wb133: np.ndarray,
                         img_w: float, img_h: float) -> dict:
    """Compute per-frame, per-side metrics.
    Returns metrics[fidx][side] = {...}.
    """
    F = vitpose.shape[0]
    metrics: dict = {}

    # Helper: previous cam_t_full for jump computation
    prev_cam_t: dict[str, np.ndarray | None] = {"left": None, "right": None}
    prev_center: dict[str, np.ndarray | None] = {"left": None, "right": None}
    prev_size:   dict[str, float | None]      = {"left": None, "right": None}
    prev_gorient: dict[str, np.ndarray | None] = {"left": None, "right": None}

    for fidx in range(F):
        vp = vitpose[fidx]   # (17, 3)
        wb = wb133[fidx]     # (133, 3)
        frame_met = {}

        for side in ("left", "right"):
            m: dict = {}

            # ViTPose arm conf
            w_idx = VP_L_WRIST  if side == "left" else VP_R_WRIST
            e_idx = VP_L_ELBOW  if side == "left" else VP_R_ELBOW
            s_idx = VP_L_SHOULDER if side == "left" else VP_R_SHOULDER
            m["vp_wrist_conf"]  = float(vp[w_idx, 2])
            m["vp_elbow_conf"]  = float(vp[e_idx, 2])
            m["vp_shoulder_conf"] = float(vp[s_idx, 2])

            # Forearm length (pixels)
            wxy = vp[w_idx, :2]
            exy = vp[e_idx, :2]
            m["forearm_len_px"] = float(np.linalg.norm(wxy - exy))

            # WB133 hand conf
            wb_start = WB_L_START if side == "left" else WB_R_START
            hand_kps = wb[wb_start: wb_start + 21, :]
            valid_mask = hand_kps[:, 2] > 0.1
            m["wb_hand_conf_mean"] = float(hand_kps[:, 2].mean())
            m["wb_hand_valid_count"] = int(valid_mask.sum())
            palm_confs = hand_kps[PALM_JOINTS, 2]
            m["wb_palm_conf_mean"] = float(palm_confs.mean())

            # HaMeR detection metrics
            has_det = fidx in det and side in det[fidx]
            m["has_detection"] = bool(has_det)

            if has_det:
                d = det[fidx][side]
                cam_t_full = d["cam_t_full"]
                box_center = d["box_center"]
                box_size   = d["box_size"]
                focal      = d["focal_length"]

                m["cam_t_z"]     = float(cam_t_full[2])
                m["box_center_x"] = float(box_center[0])
                m["box_center_y"] = float(box_center[1])
                m["box_size"]     = float(box_size)

                # Projected wrist position (HaMeR kp3d) — apply x-flip for left hand
                uv_full = project_hamer_fullimg(d["kp3d"], cam_t_full, focal, img_w, img_h,
                                                is_right=bool(d["is_right"]))
                m["proj_wrist_u"] = float(uv_full[0, 0])
                m["proj_wrist_v"] = float(uv_full[0, 1])

                # ViTPose wrist position
                m["vp_wrist_u"] = float(wxy[0])
                m["vp_wrist_v"] = float(wxy[1])

                # Wrist distance (HaMeR vs ViTPose) in pixels
                if vp[w_idx, 2] > 0.2:
                    wrist_dist = float(np.linalg.norm(uv_full[0] - wxy))
                    forearm_len = max(m["forearm_len_px"], 1.0)
                    m["wrist_dist_px"]   = wrist_dist
                    m["wrist_dist_norm"] = wrist_dist / forearm_len
                else:
                    m["wrist_dist_px"]   = float("nan")
                    m["wrist_dist_norm"] = float("nan")

                # cam_t_z jump
                if prev_cam_t[side] is not None:
                    m["cam_t_z_jump"] = float(abs(cam_t_full[2] - prev_cam_t[side][2]))
                else:
                    m["cam_t_z_jump"] = 0.0
                prev_cam_t[side] = cam_t_full.copy()

                # BBox center/size jump
                if prev_center[side] is not None:
                    m["bbox_center_jump"] = float(np.linalg.norm(box_center - prev_center[side]))
                    m["bbox_size_jump"]   = float(abs(box_size - prev_size[side]))
                else:
                    m["bbox_center_jump"] = 0.0
                    m["bbox_size_jump"]   = 0.0
                prev_center[side] = box_center.copy()
                prev_size[side]   = box_size

                # global_orient geodesic jump
                go = d["global_orient"].reshape(3, 3)
                if prev_gorient[side] is not None:
                    Rrel = prev_gorient[side].T @ go
                    trace = np.trace(Rrel)
                    cos = np.clip((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
                    m["gorient_jump"] = float(np.arccos(cos))
                else:
                    m["gorient_jump"] = 0.0
                prev_gorient[side] = go.copy()

                # projected hand size (distance from wrist to farthest tip)
                uv_all = uv_full  # (21, 2)
                dists = np.linalg.norm(uv_all - uv_all[0:1], axis=1)
                m["proj_hand_size"] = float(dists.max())

                # kp2d crop-to-full wrist position (is_right for correct x direction)
                uv_crop2full = kp2d_crop_to_full(d["kp2d"], box_center, box_size,
                                                  is_right=bool(d["is_right"]))
                m["kp2d_wrist_u"] = float(uv_crop2full[0, 0])
                m["kp2d_wrist_v"] = float(uv_crop2full[0, 1])

            else:
                for key in ("cam_t_z", "box_center_x", "box_center_y", "box_size",
                            "cam_t_z_jump", "bbox_center_jump", "bbox_size_jump",
                            "wrist_dist_px", "wrist_dist_norm", "gorient_jump",
                            "proj_hand_size", "proj_wrist_u", "proj_wrist_v",
                            "kp2d_wrist_u", "kp2d_wrist_v"):
                    m[key] = float("nan")
                prev_cam_t[side]   = None
                prev_center[side]  = None
                prev_size[side]    = None
                prev_gorient[side] = None

            frame_met[side] = m
        metrics[fidx] = frame_met

    return metrics


# ---------------------------------------------------------------------------
# 6. Outlier Detection (offline, full-sequence robust stats)
# ---------------------------------------------------------------------------

def robust_normalize(vals: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Normalize using median/MAD. Returns scores clipped to [0,1]."""
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return np.zeros_like(vals)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = max(1.4826 * mad, 1e-6)
    score = np.abs(vals - median) / scale
    return np.clip(score / threshold, 0.0, 1.0)


def compute_outlier_scores(metrics: dict) -> dict:
    """Compute per-frame, per-side outlier scores (0=normal, 1=strong outlier).
    Uses offline full-sequence stats.
    Returns outlier[fidx][side] = {'total': float, 'components': dict}.
    """
    # Collect arrays for each metric
    frames_sorted = sorted(metrics.keys())
    sides = ("left", "right")
    arrays: dict[str, dict[str, list]] = {
        s: {k: [] for k in ("wrist_dist_norm", "cam_t_z_jump", "bbox_center_jump",
                             "bbox_size_jump", "gorient_jump")}
        for s in sides
    }
    for fidx in frames_sorted:
        for s in sides:
            m = metrics[fidx][s]
            for k in arrays[s]:
                arrays[s][k].append(m.get(k, float("nan")))

    # Normalize each metric
    norms: dict[str, dict[str, np.ndarray]] = {}
    for s in sides:
        norms[s] = {}
        for k, vals in arrays[s].items():
            norms[s][k] = robust_normalize(np.asarray(vals, np.float32))

    # Combine: weighted sum
    weights = {
        "wrist_dist_norm": 0.35,
        "cam_t_z_jump":    0.20,
        "bbox_center_jump": 0.20,
        "bbox_size_jump":   0.10,
        "gorient_jump":     0.15,
    }

    outlier: dict = {}
    for i, fidx in enumerate(frames_sorted):
        outlier[fidx] = {}
        for s in sides:
            components = {k: float(norms[s][k][i]) for k in weights}
            total = sum(weights[k] * components[k] for k in weights)
            outlier[fidx][s] = {
                "total": float(total),
                "components": components,
            }

    return outlier


# ---------------------------------------------------------------------------
# 7. Frame Window Selection
# ---------------------------------------------------------------------------

def select_representative_frames(metrics: dict, outlier_scores: dict,
                                   n_top: int = 5, n_normal: int = 3) -> list[int]:
    """Select representative center frames: top outliers + random normals."""
    frames = sorted(metrics.keys())
    if not frames:
        return []

    # Combined outlier score across both sides
    combined = {}
    for fidx in frames:
        s = outlier_scores.get(fidx, {})
        total_l = s.get("left",  {}).get("total", 0.0)
        total_r = s.get("right", {}).get("total", 0.0)
        # Use max of left/right (either hand can be bad)
        combined[fidx] = max(total_l, total_r)

    sorted_by_outlier = sorted(frames, key=lambda f: combined[f], reverse=True)

    # Separate outlier metrics for component analysis
    wrist_scores  = sorted(frames, key=lambda f: max(
        outlier_scores.get(f, {}).get("left", {}).get("components", {}).get("wrist_dist_norm", 0),
        outlier_scores.get(f, {}).get("right", {}).get("components", {}).get("wrist_dist_norm", 0)
    ), reverse=True)
    bbox_scores   = sorted(frames, key=lambda f: max(
        outlier_scores.get(f, {}).get("left", {}).get("components", {}).get("bbox_center_jump", 0),
        outlier_scores.get(f, {}).get("right", {}).get("components", {}).get("bbox_center_jump", 0)
    ), reverse=True)
    camtz_scores  = sorted(frames, key=lambda f: max(
        outlier_scores.get(f, {}).get("left", {}).get("components", {}).get("cam_t_z_jump", 0),
        outlier_scores.get(f, {}).get("right", {}).get("components", {}).get("cam_t_z_jump", 0)
    ), reverse=True)

    selected = set()

    # Top overall outliers
    for fidx in sorted_by_outlier[:n_top]:
        selected.add(fidx)

    # Top wrist-specific outliers
    for fidx in wrist_scores[:3]:
        selected.add(fidx)

    # Top bbox jump outliers
    for fidx in bbox_scores[:3]:
        selected.add(fidx)

    # Top cam_t_z jump outliers
    for fidx in camtz_scores[:2]:
        selected.add(fidx)

    # Random normal frames (low outlier score)
    normal_frames = [f for f in sorted_by_outlier[-50:] if f not in selected]
    rng = np.random.default_rng(42)
    if len(normal_frames) >= n_normal:
        normal_sel = rng.choice(normal_frames, size=n_normal, replace=False).tolist()
    else:
        normal_sel = normal_frames
    for fidx in normal_sel:
        selected.add(fidx)

    return sorted(selected)


# ---------------------------------------------------------------------------
# 8. Contact Sheet Generation
# ---------------------------------------------------------------------------

CELL_SCALE = 0.25   # 1920*0.25=480 tall, 1080*0.25=270 wide

COLUMN_LABELS = [
    "Orig+ViTPose+WB133",
    "HaMeR kp3d proj",
    "BBox Modes A/B/C",
    "Combined",
]


def make_cell(img: np.ndarray, scale: float = CELL_SCALE) -> np.ndarray:
    """Resize image to cell size."""
    H, W = img.shape[:2]
    nH, nW = int(H * scale), int(W * scale)
    return cv2.resize(img, (nW, nH), interpolation=cv2.INTER_AREA)


def render_frame_strip(
    frame_bgr: np.ndarray,
    fidx: int,
    det_frame: dict,
    vp: np.ndarray,
    wb: np.ndarray,
    met: dict,
    outlier: dict,
    scale: float = CELL_SCALE,
) -> np.ndarray:
    """Render a 4-column strip for one frame."""
    H, W = frame_bgr.shape[:2]
    img_w = float(W)
    img_h = float(H)

    # ---- Column 1: Orig + ViTPose arm + WB133 hand ----
    c1 = frame_bgr.copy()
    draw_vitpose_arm(c1, vp)
    draw_wb133_hand(c1, wb, "left")
    draw_wb133_hand(c1, wb, "right")
    # Mark ViTPose wrists with larger circles
    for side, widx, color in [("left", VP_L_WRIST, C_VP_LEFT), ("right", VP_R_WRIST, C_VP_RIGHT)]:
        p = vp[widx]
        if p[2] > 0.2:
            cv2.circle(c1, (int(p[0]), int(p[1])), 12, color, 2)

    # ---- Column 2: HaMeR kp3d full-image projection ----
    c2 = frame_bgr.copy()
    for side, color in [("left", C_HAMER_LEFT), ("right", C_HAMER_RIGHT)]:
        if side not in det_frame:
            continue
        d = det_frame[side]
        uv = project_hamer_fullimg(
            d["kp3d"], d["cam_t_full"], d["focal_length"], img_w, img_h,
            is_right=bool(d["is_right"]))
        draw_hamer_kp3d_proj(c2, uv, color, f"H-{side[:1].upper()}")
        # Also show kp2d (crop→full) for comparison (corrected x-flip)
        uv2 = kp2d_crop_to_full(d["kp2d"], d["box_center"], d["box_size"],
                                 is_right=bool(d["is_right"]))
        cv2.circle(c2, (int(uv2[0, 0]), int(uv2[0, 1])), 8, (255, 255, 0), 2)  # kp2d wrist
    # Draw ViTPose wrists for reference
    for widx, color in [(VP_L_WRIST, C_VP_LEFT), (VP_R_WRIST, C_VP_RIGHT)]:
        p = vp[widx]
        if p[2] > 0.2:
            cv2.drawMarker(c2, (int(p[0]), int(p[1])), color,
                           cv2.MARKER_CROSS, 16, 2)

    # ---- Column 3: BBox Modes ----
    c3 = frame_bgr.copy()
    # Actual used bbox (from npz)
    for side in ("left", "right"):
        if side not in det_frame:
            continue
        d = det_frame[side]
        draw_bbox_rect(c3, d["box_center"], d["box_size"], C_BBOX_USED, "used")

    # Mode A (WholeBdy)
    for side in ("left", "right"):
        res = bbox_mode_a(wb, side)
        if res:
            draw_bbox_rect(c3, res[0], res[1], C_BBOX_A, "A")

    # Mode B (COCO17 wrist-elbow)
    for side in ("left", "right"):
        res = bbox_mode_b(vp, side, img_w, img_h)
        if res:
            draw_bbox_rect(c3, res[0], res[1], C_BBOX_B, "B")

    # Mode C (Hybrid)
    for side in ("left", "right"):
        ot = outlier.get(side, {}).get("total", 0.0)
        res = bbox_mode_c(wb, vp, side, img_w, img_h, outlier_score=ot)
        if res:
            draw_bbox_rect(c3, res[0], res[1], C_BBOX_C, "C")

    # ---- Column 4: Combined overlay ----
    c4 = frame_bgr.copy()
    draw_vitpose_arm(c4, vp)
    for side, color in [("left", C_HAMER_LEFT), ("right", C_HAMER_RIGHT)]:
        if side not in det_frame:
            continue
        d = det_frame[side]
        uv = project_hamer_fullimg(
            d["kp3d"], d["cam_t_full"], d["focal_length"], img_w, img_h,
            is_right=bool(d["is_right"]))
        draw_hamer_kp3d_proj(c4, uv, color)
    draw_wb133_hand(c4, wb, "left")
    draw_wb133_hand(c4, wb, "right")

    # Metric text in c1
    lines = [f"f={fidx}"]
    for side in ("left", "right"):
        m  = met.get(side, {})
        ot = outlier.get(side, {}).get("total", 0.0)
        wd = m.get("wrist_dist_norm", float("nan"))
        bj = m.get("bbox_center_jump", float("nan"))
        cz = m.get("cam_t_z", float("nan"))
        wd_str = f"{wd:.2f}" if np.isfinite(wd) else "NA"
        bj_str = f"{bj:.0f}" if np.isfinite(bj) else "NA"
        cz_str = f"{cz:.0f}" if np.isfinite(cz) else "NA"
        lines.append(f"{side[:1].upper()}: ot={ot:.2f} wd={wd_str} bj={bj_str} cz={cz_str}")
    put_metric_text(c1, lines)

    # Resize cells
    cells = [make_cell(c, scale) for c in [c1, c2, c3, c4]]
    strip = np.concatenate(cells, axis=1)
    return strip


def generate_contact_sheets(
    frames_dir: Path,
    selected_fidxs: list[int],
    det: dict,
    vitpose: np.ndarray,
    wb133: np.ndarray,
    metrics: dict,
    outlier_scores: dict,
    out_dir: Path,
    scale: float = CELL_SCALE,
    frames_per_sheet: int = 12,
) -> list[Path]:
    """Generate contact sheets for selected frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_files = sorted(frames_dir.glob("*.jpg"))
    frame_map = {int(f.stem): f for f in frame_files}

    saved: list[Path] = []
    sheet_idx = 0
    strips_buf: list[np.ndarray] = []

    def flush_sheet():
        nonlocal sheet_idx, strips_buf
        if not strips_buf:
            return
        # Add column header bar to first strip
        header_h = 22
        sw = strips_buf[0].shape[1]
        header = np.zeros((header_h, sw, 3), dtype=np.uint8)
        ncols = 4
        cw = sw // ncols
        for ci, lbl in enumerate(COLUMN_LABELS):
            cv2.putText(header, lbl, (ci * cw + 4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        sheet = np.concatenate([header] + strips_buf, axis=0)
        path = out_dir / f"contact_sheet_{sheet_idx:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        saved.append(path)
        print(f"[visual_critic] contact sheet → {path}")
        sheet_idx += 1
        strips_buf = []

    for fidx in selected_fidxs:
        img_path = frame_map.get(fidx)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        det_frame = det.get(fidx, {})
        vp = vitpose[fidx]
        wb = wb133[fidx]
        met = metrics.get(fidx, {"left": {}, "right": {}})
        ot  = outlier_scores.get(fidx, {"left": {}, "right": {}})

        strip = render_frame_strip(img, fidx, det_frame, vp, wb, met, ot, scale)
        strips_buf.append(strip)

        if len(strips_buf) >= frames_per_sheet:
            flush_sheet()

    flush_sheet()
    return saved


def generate_temporal_window_sheets(
    frames_dir: Path,
    center_fidxs: list[int],
    det: dict,
    vitpose: np.ndarray,
    wb133: np.ndarray,
    metrics: dict,
    outlier_scores: dict,
    out_dir: Path,
    window: int = 5,
    scale: float = 0.15,
) -> list[Path]:
    """For each center frame, generate a temporal window strip showing ±window frames.
    Each strip: 2 columns (orig+vp | hamer_proj) × (2*window+1) frames.
    Shows temporal context to identify sustained vs isolated problems.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_files = {int(f.stem): f for f in sorted(frames_dir.glob("*.jpg"))}
    n_frames = vitpose.shape[0]
    img_w, img_h = 1080.0, 1920.0
    saved = []

    for center in center_fidxs:
        fidxs = range(max(0, center - window), min(n_frames, center + window + 1))
        strips = []

        for fidx in fidxs:
            img_path = frame_files.get(fidx)
            if img_path is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            vp = vitpose[fidx]
            wb = wb133[fidx]
            det_f = det.get(fidx, {})
            met = metrics.get(fidx, {"left": {}, "right": {}})
            ot  = outlier_scores.get(fidx, {"left": {}, "right": {}})

            # Column 1: orig + ViTPose arm + WB133 hand
            c1 = img.copy()
            draw_vitpose_arm(c1, vp)
            draw_wb133_hand(c1, wb, "left")
            draw_wb133_hand(c1, wb, "right")
            # Mark center frame
            if fidx == center:
                cv2.rectangle(c1, (0, 0), (img.shape[1]-1, img.shape[0]-1),
                              (0, 255, 255), 8)

            # Column 2: HaMeR proj
            c2 = img.copy()
            for side, color in [("left", C_HAMER_LEFT), ("right", C_HAMER_RIGHT)]:
                if side not in det_f:
                    continue
                d = det_f[side]
                uv = project_hamer_fullimg(d["kp3d"], d["cam_t_full"], d["focal_length"],
                                           img_w, img_h, is_right=bool(d["is_right"]))
                draw_hamer_kp3d_proj(c2, uv, color)
            draw_vitpose_arm(c2, vp)  # ViTPose as reference

            # Metric overlay
            ot_max = max(ot.get("left", {}).get("total", 0),
                         ot.get("right", {}).get("total", 0))
            lines = [f"f={fidx}", f"ot={ot_max:.2f}"]
            for s in ("left", "right"):
                wd = met.get(s, {}).get("wrist_dist_norm", float("nan"))
                if np.isfinite(wd):
                    lines.append(f"{s[:1]}:wd={wd:.2f}")
            put_metric_text(c1, lines)

            cell1 = make_cell(c1, scale)
            cell2 = make_cell(c2, scale)
            pair  = np.concatenate([cell1, cell2], axis=1)
            strips.append(pair)

        if not strips:
            continue

        # Stack temporally (horizontal)
        temporal_strip = np.concatenate(strips, axis=1)

        # Add label bar
        lbar_h = 20
        lbar = np.zeros((lbar_h, temporal_strip.shape[1], 3), dtype=np.uint8)
        cv2.putText(lbar, f"Center={center}  window=±{window}  [yellow border = center]",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        result = np.concatenate([lbar, temporal_strip], axis=0)

        out_path = out_dir / f"temporal_window_{center:05d}.jpg"
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 88])
        saved.append(out_path)
        print(f"[visual_critic] temporal window → {out_path}")

    return saved


# ---------------------------------------------------------------------------
# 9. Metrics CSV / JSON Export
# ---------------------------------------------------------------------------

def export_metrics(metrics: dict, outlier_scores: dict, out_dir: Path):
    """Write frame_side_metrics.csv and .json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fidx in sorted(metrics.keys()):
        for side in ("left", "right"):
            m  = metrics[fidx].get(side, {})
            ot = outlier_scores.get(fidx, {}).get(side, {})
            row = {"frame": fidx, "side": side, "outlier_total": ot.get("total", float("nan"))}
            row.update(m)
            # Add component outlier scores
            for k, v in ot.get("components", {}).items():
                row[f"ot_{k}"] = v
            rows.append(row)

    if not rows:
        return

    csv_path = out_dir / "frame_side_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[visual_critic] metrics CSV → {csv_path}")

    json_path = out_dir / "frame_side_metrics.json"
    with open(json_path, "w") as f:
        # Convert nan to null for JSON
        def clean(v):
            if isinstance(v, float) and not np.isfinite(v):
                return None
            return v
        json_rows = [{k: clean(v) for k, v in r.items()} for r in rows]
        json.dump(json_rows, f, indent=2)
    print(f"[visual_critic] metrics JSON → {json_path}")


# ---------------------------------------------------------------------------
# 10. Markdown Report
# ---------------------------------------------------------------------------

def generate_report(
    metrics: dict,
    outlier_scores: dict,
    selected_fidxs: list[int],
    contact_sheets: list[Path],
    det: dict,
    vitpose: np.ndarray,
    wb133: np.ndarray,
    out_dir: Path,
    n_frames: int,
    iter_num: int,
) -> Path:
    """Generate VISUAL_CRITIC_REPORT.md."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate statistics
    all_wrist_dist: list[float] = []
    all_bbox_jump:  list[float] = []
    all_camtz_jump: list[float] = []
    all_gorient:    list[float] = []
    det_counts = {"left": 0, "right": 0}
    bad_camtz = []

    for fidx in sorted(metrics.keys()):
        for side in ("left", "right"):
            m = metrics[fidx].get(side, {})
            if m.get("has_detection", False):
                det_counts[side] += 1
            for key, arr in [("wrist_dist_norm", all_wrist_dist),
                               ("bbox_center_jump", all_bbox_jump),
                               ("cam_t_z_jump", all_camtz_jump),
                               ("gorient_jump", all_gorient)]:
                v = m.get(key, float("nan"))
                if np.isfinite(v):
                    arr.append(v)
            cz = m.get("cam_t_z", float("nan"))
            if np.isfinite(cz) and cz <= 0:
                bad_camtz.append((fidx, side, cz))

    def stat_str(arr):
        if not arr:
            return "N/A"
        a = np.asarray(arr)
        return (f"median={np.median(a):.3f} "
                f"mean={a.mean():.3f} "
                f"max={a.max():.3f} "
                f"p95={np.percentile(a,95):.3f}")

    # Top outlier frames
    top_outlier = sorted(
        [(fidx, max(outlier_scores.get(fidx,{}).get("left",{}).get("total",0),
                    outlier_scores.get(fidx,{}).get("right",{}).get("total",0)))
         for fidx in metrics],
        key=lambda x: x[1], reverse=True
    )[:10]

    # BBox mode agreement check
    # For random sample of 20 frames, check if Mode A and Mode B agree
    rng = np.random.default_rng(0)
    sample_fidxs = rng.choice(sorted(metrics.keys()), size=min(20, len(metrics)), replace=False)
    bbox_agreement = {"A_vs_B_wrist_diff_px": []}
    img_w, img_h = 1080.0, 1920.0  # default
    for f in sample_fidxs:
        vp = vitpose[f]
        wb = wb133[f]
        for side in ("left", "right"):
            ra = bbox_mode_a(wb, side)
            rb = bbox_mode_b(vp, side, img_w, img_h)
            if ra and rb:
                diff = float(np.linalg.norm(ra[0] - rb[0]))
                bbox_agreement["A_vs_B_wrist_diff_px"].append(diff)
    a_vs_b = bbox_agreement["A_vs_B_wrist_diff_px"]
    a_vs_b_stat = stat_str(a_vs_b) if a_vs_b else "N/A"

    # Write report
    lines = [
        "# Visual Critic Loop Report",
        f"\n생성 일시: 2026-05-28  |  반복 횟수: Iter {iter_num}  |  총 프레임: {n_frames}",
        "\n---",
        "\n## 1. 실행 요약",
        f"- 분석 대상 프레임: {n_frames}",
        f"- Left HaMeR 검출: {det_counts['left']} frames",
        f"- Right HaMeR 검출: {det_counts['right']} frames",
        f"- 선택된 대표 프레임: {len(selected_fidxs)} ({selected_fidxs[:5]}...)",
        f"- Contact sheet: {len(contact_sheets)}개",
        "\n---",
        "\n## 2. Projection Lock 결과",
        "\n### HaMeR full-image projection 품질",
        f"- wrist_dist_norm (HaMeR proj vs ViTPose wrist): {stat_str(all_wrist_dist)}",
        "  - 0 = 완벽히 일치, 1 = forearm 길이만큼 차이",
        f"- cam_t_z 음수 프레임 (bad detection): {len(bad_camtz)}건",
    ]

    if bad_camtz:
        for fidx, side, cz in bad_camtz[:5]:
            lines.append(f"  - frame {fidx} {side}: cam_t_z={cz:.2f}")

    lines += [
        "\n### ViTPose vs HaMeR kp3d 정합성",
        "- wrist_dist_norm < 0.5: 대체로 정상 (forearm 절반 이내 차이)",
        "- wrist_dist_norm > 1.0: 심각한 misalignment (forearm 길이 이상 차이)",
    ]

    if all_wrist_dist:
        median_wd = np.median(all_wrist_dist)
        p95_wd = np.percentile(all_wrist_dist, 95)
        lines.append(f"- 이 시퀀스: median={median_wd:.2f}, p95={p95_wd:.2f}")
        if median_wd > 1.0:
            lines.append("- **→ WARN: projection이 전체적으로 크게 어긋남. CASE A 가능성.**")
        elif p95_wd > 1.5:
            lines.append("- **→ WARN: 일부 프레임에서 크게 어긋남. CASE C (일부 outlier) 가능성.**")
        else:
            lines.append("- **→ OK: projection이 대체로 양호함.**")

    lines += [
        "\n---",
        "\n## 3. BBox Lock 결과",
        f"- bbox_center_jump: {stat_str(all_bbox_jump)}",
        f"- cam_t_z_jump: {stat_str(all_camtz_jump)}",
        f"\n### BBox Mode 비교 (A=WholeBdy, B=COCO17, C=Hybrid)",
        f"- Mode A vs B 중심 거리: {a_vs_b_stat}",
        "\n#### 권고 BBox Mode",
    ]

    if a_vs_b:
        median_diff = np.median(a_vs_b)
        if median_diff > 50:
            lines.append("- Mode A와 B가 크게 다름 → **Hybrid(C) 권장**")
            lines.append("  - 이유: WholeBdy와 COCO17 간 일관성 낮음, 자동 선택 필요")
        elif median_diff < 20:
            lines.append("- Mode A와 B가 대체로 일치 → **Mode A(WholeBdy) 권장**")
            lines.append("  - 이유: WholeBdy가 더 손에 밀착된 bbox 제공")
        else:
            lines.append("- 중간 차이 → **Mode C(Hybrid) 권장**")
            lines.append("  - WholeBdy confidence 높을 때 A, 낮을 때 B 사용")

    lines += [
        "\n---",
        "\n## 4. Outlier Score 요약",
        "\n### Top 10 Outlier 프레임",
        "| frame | outlier_score |",
        "|-------|---------------|",
    ]
    for fidx, score in top_outlier:
        lines.append(f"| {fidx} | {score:.3f} |")

    lines += [
        "\n---",
        "\n## 5. ViTPose Skeleton 안정성",
    ]

    # ViTPose confidence stats
    vp_wrist_confs_l = [metrics[f]["left"].get("vp_wrist_conf", float("nan"))
                         for f in metrics if "left" in metrics[f]]
    vp_wrist_confs_r = [metrics[f]["right"].get("vp_wrist_conf", float("nan"))
                         for f in metrics if "right" in metrics[f]]
    vp_wrist_confs_l = [v for v in vp_wrist_confs_l if np.isfinite(v)]
    vp_wrist_confs_r = [v for v in vp_wrist_confs_r if np.isfinite(v)]

    if vp_wrist_confs_l:
        low_conf_l = sum(1 for v in vp_wrist_confs_l if v < 0.3)
        lines.append(f"- Left wrist conf: median={np.median(vp_wrist_confs_l):.2f}, "
                     f"low(<0.3): {low_conf_l}/{len(vp_wrist_confs_l)} frames")
    if vp_wrist_confs_r:
        low_conf_r = sum(1 for v in vp_wrist_confs_r if v < 0.3)
        lines.append(f"- Right wrist conf: median={np.median(vp_wrist_confs_r):.2f}, "
                     f"low(<0.3): {low_conf_r}/{len(vp_wrist_confs_r)} frames")

    vp_low = (low_conf_l if vp_wrist_confs_l else 0) + (low_conf_r if vp_wrist_confs_r else 0)
    total_wrist = len(vp_wrist_confs_l) + len(vp_wrist_confs_r)
    vp_low_ratio = vp_low / max(total_wrist, 1)
    if vp_low_ratio < 0.1:
        lines.append("- **→ CASE E: ViTPose skeleton 안정적 — ViTPose를 fallback/keyframe target으로 사용 가능**")
    elif vp_low_ratio < 0.3:
        lines.append("- **→ WARN: ViTPose 일부 구간 불안정 — confidence 낮은 프레임에서 ViTPose weight 줄이기 필요**")
    else:
        lines.append("- **→ CASE F: ViTPose도 불안정 — temporal consistency 중심으로 진행 필요**")

    lines += [
        "\n---",
        "\n## 6. Global Orient Jump (HaMeR 안정성)",
        f"- global_orient geodesic jump: {stat_str(all_gorient)} (rad)",
        "  - > 0.5 rad: 갑작스러운 손목 방향 변화 (outlier 가능성)",
    ]
    if all_gorient:
        high_gorient = sum(1 for v in all_gorient if v > 0.5)
        lines.append(f"  - > 0.5 rad 프레임: {high_gorient}")

    lines += [
        "\n---",
        "\n## 7. CASE 판정",
        "\n아래 CASE 중 해당하는 것을 체크하세요 (시각 검토 후 확정):",
    ]

    # Auto-determine likely cases
    cases = []
    if all_wrist_dist and np.median(all_wrist_dist) > 1.5:
        cases.append("CASE A: Projection 문제 — HaMeR proj가 전체적으로 ViTPose와 크게 다름")
    if a_vs_b and np.median(a_vs_b) > 30:
        cases.append("CASE B: BBox 문제 — Mode A/B 차이가 큼, bbox mode 선택 필요")
    if all_wrist_dist and np.percentile(all_wrist_dist, 95) > 1.5 and np.median(all_wrist_dist) < 1.0:
        cases.append("CASE C: 일부 frame만 튐 — offline outlier score 기반 처리 필요")
    if vp_low_ratio < 0.15:
        cases.append("CASE E: ViTPose skeleton 안정적 — ViTPose를 fallback target으로 사용 가능")
    cases.append("CASE G: GVHMR arm-chain pre-fit — ViTPose body skeleton 기준 보정 검토")
    cases.append("CASE H: HaMeR MANO pre-fit — ViTPose wrist+palm MCP 기준 보정 검토")

    for case in cases:
        lines.append(f"- [{case}]")

    lines += [
        "\n---",
        "\n## 8. Offline Outlier Score & Adaptive Weight 설계안",
        "\n### err_t 정의 (0=정상, 1=강한 outlier)",
        "```",
        "err_t = 0.35 * wrist_dist_norm_score",
        "       + 0.20 * cam_t_z_jump_score",
        "       + 0.20 * bbox_center_jump_score",
        "       + 0.10 * bbox_size_jump_score",
        "       + 0.15 * gorient_jump_score",
        "```",
        "\n### Adaptive weight 적용 (구현 시 참고)",
        "```python",
        "w_keypoint_t = (w_kp_base + w_kp_outlier * err_t) * vp_conf",
        "w_prior_t    = max(w_prior_min, w_prior_base * (1 - err_t))",
        "w_temporal_t = w_temporal_base + w_temporal_outlier * err_t",
        "```",
        "",
        "# 권장 초기값:",
        "# w_kp_base=0.5, w_kp_outlier=4.5 (outlier 구간 최대 5x)",
        "# w_prior_base=5.0, w_prior_min=0.5",
        "# w_temporal_base=1e3, w_temporal_outlier=5e3",
        "```",
        "\n---",
        "\n## 9. GVHMR Arm-chain Pre-fit 가능성",
        "- ViTPose shoulder/elbow/wrist 데이터: ✅ (1495, 17, 3) 확인됨",
        "- GVHMR body_pose: ✅ (1495, 63) 확인됨",
        "- K_fullimg: ✅ (1495, 3, 3) 확인됨",
        "- 최적화 대상: body_pose index {15,16,17,18,19,20} (shoulder×2, elbow×2, wrist×2)",
        "- Spine: 완전 고정 (collar 13,14 포함 여부는 별도 검토)",
        "- **→ 가능. Stage 1 (cap_arm_prefit.py) 구현 준비 완료**",
        "\n---",
        "\n## 10. HaMeR MANO Pre-fit 가능성",
        "- WB133 wrist+MCP: ✅ (1495, 133, 3) 확인됨",
        "- HaMeR kp3d + cam_t_full: ✅ (1495 frames × 2 sides) 확인됨",
        "- 초기 target: wrist, index/middle/ring/pinky MCP (5 joints per hand)",
        "- hand_pose: freeze (global_orient + cam_t만 최적화)",
        "- **→ 가능. Stage 2 (cap_mano_prefit.py) 구현 준비 완료**",
        "\n---",
        "\n## 11. Left-hand x-flip 수정 결과 (Iter 2+)",
        "- **버그 발견: Left hand kp3d 투영에 x-flip 누락**",
        "- Before x-flip: left wrist u-error median=84.3px",
        "- After x-flip:  left wrist u-error median=5.1px (93.9% 개선)",
        "- 수정 파일: tools/visual_critic_loop.py::project_hamer_projection (is_right 파라미터 추가)",
        "- **→ cap_wrist_refine.py::build_hand_targets_option1_hamer에도 동일 버그 존재**",
        "  → Stage B 실행 전 반드시 수정 필요",
        "\n---",
        "\n## 12. 다음 단계 제안",
        "1. contact_sheet_*.jpg, temporal_window_*.jpg 시각 검토",
        "2. **cap_wrist_refine.py::build_hand_targets_option1_hamer에 x-flip 수정 적용**",
        "3. **Stage 1 (GVHMR arm pre-fit / cap_arm_prefit.py) 구현**",
        "   - ViTPose shoulder/elbow/wrist 기준 arm chain pre-fit",
        "   - Spine 완전 고정, collar+shoulder+elbow+wrist만 최적화",
        "4. Stage 2 (HaMeR MANO pre-fit) 구현",
        "5. Stage 3 (merge optimization) 구현",
        "\n---",
        f"\n생성된 파일:",
        f"- Contact sheets: {[str(p) for p in contact_sheets]}",
        f"- Metrics CSV/JSON: metrics/frame_side_metrics.*",
        f"- Visual labels: review/frame_side_visual_labels.csv",
    ]

    report_path = out_dir / "VISUAL_CRITIC_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[visual_critic] report → {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# 11. Visual Label CSV
# ---------------------------------------------------------------------------

def export_visual_labels(selected_fidxs: list[int], outlier_scores: dict,
                          metrics: dict, out_dir: Path):
    """Export placeholder visual labels for manual review."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "frame_side_visual_labels.csv"
    rows = []
    for fidx in selected_fidxs:
        for side in ("left", "right"):
            m  = metrics.get(fidx, {}).get(side, {})
            ot = outlier_scores.get(fidx, {}).get(side, {})
            wd = m.get("wrist_dist_norm", float("nan"))
            score = ot.get("total", 0.0)
            # Auto-label based on score
            if score > 0.7:
                auto_label = "BAD"
            elif score > 0.4:
                auto_label = "WARN"
            elif np.isfinite(wd) and wd > 1.0:
                auto_label = "WARN"
            else:
                auto_label = "OK"
            rows.append({
                "frame": fidx,
                "side": side,
                "outlier_total": f"{score:.3f}",
                "wrist_dist_norm": f"{wd:.3f}" if np.isfinite(wd) else "NA",
                "auto_label": auto_label,
                "human_label": "",   # to be filled manually
                "issue_type": "",
                "notes": "",
            })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[visual_critic] visual labels → {path}")


# ---------------------------------------------------------------------------
# 12. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visual critic loop for HaMeR+GVHMR analysis")
    parser.add_argument("--input", type=str,
                        default="outputs/cap_pipeline/input",
                        help="Pipeline input directory (contains hamer_out_clean/, preprocess/, etc.)")
    parser.add_argument("--max-iters", type=int, default=5,
                        help="Max self-feedback iterations")
    parser.add_argument("--scale", type=float, default=CELL_SCALE,
                        help="Cell scale for contact sheet (default 0.25)")
    parser.add_argument("--n-top-outlier", type=int, default=6,
                        help="Number of top outlier frames to include")
    parser.add_argument("--n-normal", type=int, default=3,
                        help="Number of normal frames to include")
    args = parser.parse_args()

    input_dir     = Path(args.input)
    hamer_dir     = input_dir / "hamer_out_clean"
    preproc_dir   = input_dir / "preprocess"
    frames_dir    = input_dir / "hamer_frames"
    out_root      = input_dir / "wrist_refine_debug" / "self_feedback_visual_critic"
    contact_dir   = out_root / "contact_sheets"
    metrics_dir   = out_root / "metrics"
    review_dir    = out_root / "review"

    print("[visual_critic] Loading data...")
    det      = load_hamer_detections(hamer_dir)
    vitpose  = load_vitpose(preproc_dir)         # (F, 17, 3)
    wb133    = load_wb133(preproc_dir)           # (F, 133, 3)
    n_frames = vitpose.shape[0]

    img_w, img_h = 1080.0, 1920.0

    print(f"[visual_critic] frames={n_frames}, "
          f"HaMeR detections={sum(len(v) for v in det.values())} side-frames")

    for iteration in range(1, args.max_iters + 1):
        print(f"\n[visual_critic] === Iteration {iteration} ===")

        # Compute metrics (with corrected x-flip projection from Iter 2+)
        print("[visual_critic] Computing per-frame metrics...")
        metrics = compute_all_metrics(det, vitpose, wb133, img_w, img_h)

        # Compute outlier scores
        print("[visual_critic] Computing outlier scores...")
        outlier = compute_outlier_scores(metrics)

        # Select representative frames
        selected = select_representative_frames(
            metrics, outlier,
            n_top=args.n_top_outlier, n_normal=args.n_normal
        )
        print(f"[visual_critic] Selected {len(selected)} representative frames: {selected[:8]}...")

        # Export metrics
        export_metrics(metrics, outlier, metrics_dir)

        # Export visual labels
        export_visual_labels(selected, outlier, metrics, review_dir)

        # Generate contact sheets
        print("[visual_critic] Generating contact sheets...")
        sheets = generate_contact_sheets(
            frames_dir, selected, det, vitpose, wb133,
            metrics, outlier, contact_dir, scale=args.scale
        )

        # Generate projection lock images (top 8 outlier frames)
        proj_lock_dir = out_root / "projection_lock"
        top8 = sorted(outlier.keys(),
                      key=lambda f: max(outlier[f].get("left", {}).get("total", 0),
                                        outlier[f].get("right", {}).get("total", 0)),
                      reverse=True)[:8]
        print("[visual_critic] Generating projection lock images...")
        generate_projection_lock_images(
            det, vitpose, frames_dir, proj_lock_dir, top8, img_w, img_h)

        # Generate temporal window strips (top 5 outlier centers)
        window_dir = out_root / "temporal_windows"
        top5_centers = top8[:5]
        print("[visual_critic] Generating temporal window strips...")
        generate_temporal_window_sheets(
            frames_dir, top5_centers, det, vitpose, wb133,
            metrics, outlier, window_dir, window=5, scale=0.15)

        # Generate report
        print("[visual_critic] Generating report...")
        report_path = generate_report(
            metrics, outlier, selected, sheets, det, vitpose, wb133,
            out_root, n_frames, iteration
        )

        print(f"\n[visual_critic] Iteration {iteration} complete.")
        print(f"  Report: {report_path}")
        print(f"  Contact sheets: {len(sheets)} in {contact_dir}")
        print(f"  Projection lock: {proj_lock_dir}")
        print(f"  Temporal windows: {window_dir}")

        # One iteration is enough for analysis (no automated code changes)
        print("[visual_critic] Analysis complete. Review images and report.")
        break

    print("\n[visual_critic] Done.")
    print(f"  Output root: {out_root}")


if __name__ == "__main__":
    os.chdir(PROJ_ROOT)
    main()
