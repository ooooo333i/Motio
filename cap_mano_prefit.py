"""
cap_mano_prefit.py

Stage 2: HaMeR MANO Pre-fit
  저장된 HaMeR kp3d에 6-DoF rigid correction (delta_R + delta_t)을 적용해
  WB133 wrist + palm MCP keypoint와 2D 정합을 맞춥니다.

hand_pose / betas 는 freeze. global_orient(→delta_R) + cam_t_full(→delta_t) 만 최적화.

Loss:
  L = w_palm * L_2d_palm  +  w_prior * L_prior  +  w_tc * L_temporal

  - L_2d_palm: 5개 palm joint (wrist + 4 MCP) projection vs WB133 target
  - L_prior:   ||delta_rvec||^2 + ||delta_t||^2
  - L_temporal: adjacent-frame smoothness of delta_rvec, delta_t

WB133 → MANO palm joint index 대응 (동일 index):
  wrist=0, index_MCP=5, middle_MCP=9, ring_MCP=13, pinky_MCP=17

출력: outputs/cap_pipeline/input/hamer_mano_prefit.pt
  prefit[frame][side] = {
      'joints_cam': (21, 3)  # pre-fit camera-space joints (after x-flip)
      'uv_palm':    (5, 2)   # projected 5 palm joints (final 2D position)
      'delta_rvec': (3,)     # applied rotation
      'delta_t':    (3,)     # applied translation
  }

사용법:
  python cap_mano_prefit.py
  python cap_mano_prefit.py --iters 200 --w-prior 5.0 --w-tc 1e2
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))
os.chdir(str(PROJ_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MANO kp3d palm joint indices (same ordering as WB133 hand block)
MANO_PALM_IDX = [0, 5, 9, 13, 17]   # wrist, idx_MCP, mid_MCP, ring_MCP, pinky_MCP

# WB133 hand block start indices
WB_L_START = 91
WB_R_START = 112
# Within hand block: PALM_JOINTS offset
WB_PALM_LOCAL = [0, 5, 9, 13, 17]   # same offsets as MANO

# Confidence threshold for WB133
WB_CONF_THRESH = 0.5

# ---------------------------------------------------------------------------
# Rotation helper
# ---------------------------------------------------------------------------

def aa_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle → (..., 3, 3). Rodrigues, differentiable."""
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis  = aa / theta
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = torch.cos(theta)[..., 0]
    s = torch.sin(theta)[..., 0]
    C = 1 - c
    return torch.stack([
        c + x*x*C,    x*y*C - z*s, x*z*C + y*s,
        y*x*C + z*s, c + y*y*C,   y*z*C - x*s,
        z*x*C - y*s, z*y*C + x*s, c + z*z*C,
    ], dim=-1).reshape(aa.shape[:-1] + (3, 3))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hamer_side(hamer_dir: Path, side: str) -> dict[int, dict]:
    """Load all HaMeR npz for one side (left or right).
    Returns {frame_idx: {kp3d, cam_t_full, focal, img_size, is_right}}.
    """
    want_right = (side == "right")
    pattern    = re.compile(r"^(\d+)_")
    result     = {}
    for p in sorted(hamer_dir.glob("*.npz")):
        m = pattern.match(p.stem)
        if m is None:
            continue
        fidx     = int(m.group(1))
        d        = np.load(p, allow_pickle=True)
        is_right = int(np.asarray(d["is_right"]).reshape(-1)[0])
        if bool(is_right) != want_right:
            continue
        cam_t_full = np.asarray(d["cam_t_full"], np.float32).reshape(3)
        if cam_t_full[2] <= 0:
            continue
        result[fidx] = {
            "kp3d":      np.asarray(d["keypoints_3d"], np.float32).reshape(21, 3),
            "cam_t_full": cam_t_full,
            "focal":     float(np.asarray(d["focal_length"]).reshape(-1)[0]),
            "img_size":  np.asarray(d["img_size"], np.float32).reshape(2),
            "is_right":  is_right,
        }
    return result


def load_wb133_palm(wb133: np.ndarray, frame_ids: list[int],
                    side: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns wb_uv (N,5,2) and wb_conf (N,5) for selected frames."""
    wb_start = WB_R_START if side == "right" else WB_L_START
    uvs, confs = [], []
    for f in frame_ids:
        hand = wb133[f, wb_start: wb_start + 21, :]   # (21, 3)
        palm = hand[WB_PALM_LOCAL]                     # (5, 3) [x,y,conf]
        uvs.append(palm[:, :2])
        confs.append(palm[:, 2])
    return np.array(uvs, np.float32), np.array(confs, np.float32)


# ---------------------------------------------------------------------------
# Core optimization
# ---------------------------------------------------------------------------

def optimize_side(
    frame_ids:    list[int],
    kp3d_all:     torch.Tensor,   # (N, 21, 3) - saved MANO joints in cam-oriented space
    cam_t_all:    torch.Tensor,   # (N, 3)
    focal:        float,
    img_w:        float,
    img_h:        float,
    is_right:     bool,
    wb_uv:        torch.Tensor,   # (N, 5, 2)  WB133 palm targets
    wb_conf:      torch.Tensor,   # (N, 5)
    err_t:        torch.Tensor,   # (N,) outlier score
    *,
    iters:         int   = 200,
    lr:            float = 1e-2,
    w_palm_base:   float = 2.0,
    w_palm_outlier:float = 3.0,  # outlier frames get stronger WB pull
    w_prior:       float = 5.0,
    w_tc:          float = 1e2,
    min_iter:      int   = 40,
    patience:      int   = 30,
    target_px:     float = 10.0,
    device:        str   = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimize delta_rvec (N,3) and delta_t (N,3) for one side.
    Returns (delta_rvec, delta_t) tensors.
    """
    N = kp3d_all.shape[0]

    # x-flip for left hand (kp3d is in right-hand canonical space)
    if not is_right:
        kp3d_xflip = kp3d_all.clone()
        kp3d_xflip[:, :, 0] *= -1
    else:
        kp3d_xflip = kp3d_all

    # Camera-space joints BEFORE correction: (N, 21, 3)
    joints_cam_orig = kp3d_xflip + cam_t_all.unsqueeze(1)

    # Wrist position in camera space (pivot for rotation): (N, 1, 3)
    wrist_cam = joints_cam_orig[:, 0:1, :].detach()

    # Adaptive weights: outlier frames get stronger WB pull, weaker prior
    w_palm_t  = w_palm_base + w_palm_outlier * err_t                   # (N,)
    w_prior_t = torch.clamp(w_prior * (1.0 - 0.8 * err_t), min=0.5)  # (N,)

    # Optimization variables (initialized at zero = no change)
    delta_rvec = torch.zeros(N, 3, device=device, requires_grad=True)
    delta_t    = torch.zeros(N, 3, device=device, requires_grad=True)
    opt = torch.optim.Adam([delta_rvec, delta_t], lr=lr)

    # Pre-select palm joints
    palm_orig = joints_cam_orig[:, MANO_PALM_IDX, :]   # (N, 5, 3)
    cx = img_w / 2.0
    cy = img_h / 2.0

    def _palm_rms(uv_det: torch.Tensor) -> float:
        diff2 = ((uv_det - wb_uv) ** 2).sum(-1)  # (N,5)
        w = wb_conf
        numer = float((diff2 * w).sum().item())
        denom = float(w.sum().item())
        return float(np.sqrt(numer / max(denom, 1e-6)))

    best_rms   = float("inf")
    best_state = None
    best_it    = -1
    bad_count  = 0
    stop_reason = "max_iter"
    palm_rms = float("nan")

    for it in range(iters):
        opt.zero_grad()

        # Rotation around wrist pivot
        R_delta = aa_to_rotmat(delta_rvec)              # (N, 3, 3)

        # Apply rotation: centered_rot = R_delta @ (joints - wrist)
        centered   = joints_cam_orig - wrist_cam        # (N, 21, 3)
        centered_r = torch.bmm(centered, R_delta.transpose(1, 2))  # (N, 21, 3)

        # New camera-space joints
        joints_new = centered_r + wrist_cam + delta_t.unsqueeze(1)   # (N, 21, 3)

        # Project palm joints
        palm_new = joints_new[:, MANO_PALM_IDX, :]     # (N, 5, 3)
        z = palm_new[:, :, 2].clamp(min=1e-4)
        u = focal * palm_new[:, :, 0] / z + cx
        v = focal * palm_new[:, :, 1] / z + cy
        proj_palm = torch.stack([u, v], dim=-1)         # (N, 5, 2)

        # L_2d (conf-weighted, adaptive per-frame)
        diff  = (proj_palm - wb_uv) ** 2                # (N, 5, 2)
        l2_pf = (diff.sum(-1) * wb_conf).sum(-1)        # (N,)  per-frame
        L_2d  = (l2_pf * w_palm_t).mean()

        # L_prior (adaptive: tighter for normal frames)
        L_prior = ((delta_rvec ** 2).sum(-1) * w_prior_t).mean() + \
                  ((delta_t    ** 2).sum(-1) * w_prior_t).mean()

        # L_temporal (smooth corrections across adjacent frames)
        L_tc = torch.zeros(1, device=device)
        if N > 1:
            L_tc = (((delta_rvec[1:] - delta_rvec[:-1]) ** 2).mean() +
                    ((delta_t   [1:] - delta_t   [:-1]) ** 2).mean())

        loss = L_2d + L_prior + w_tc * L_tc

        # Track palm RMS
        with torch.no_grad():
            palm_rms = _palm_rms(proj_palm.detach())

        improved = palm_rms < best_rms - 0.3
        if improved:
            best_rms   = palm_rms
            best_it    = it
            bad_count  = 0
            best_state = (delta_rvec.detach().clone(), delta_t.detach().clone())
        else:
            bad_count += 1

        if it % 50 == 0 or it == iters - 1:
            pass  # suppress per-iter prints; main log below

        if it >= min_iter:
            if palm_rms <= target_px:
                stop_reason = "target_px"
                break
            if bad_count >= patience:
                stop_reason = "plateau"
                break

        loss.backward()
        opt.step()

    if best_state is not None:
        with torch.no_grad():
            delta_rvec.copy_(best_state[0])
            delta_t.copy_(best_state[1])

    return delta_rvec.detach(), delta_t.detach(), palm_rms, best_rms, stop_reason, it + 1


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------

def generate_prefit_debug_sheet(
    frame_dir:   Path,
    prefit:      dict,
    wb133:       np.ndarray,
    selected:    list[int],
    out_path:    Path,
    scale:       float = 0.22,
):
    """Before/After 2D hand position comparison contact sheet."""
    frame_files = {int(p.stem): p for p in sorted(frame_dir.glob("*.jpg"))}
    strips = []

    for fidx in selected:
        img_path = frame_files.get(fidx)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        col_b = img.copy()  # BEFORE
        col_a = img.copy()  # AFTER

        for side, c_hamer, c_wb in [
            ("left",  (255, 200, 0), (200, 0, 200)),
            ("right", (0, 200, 255), (0, 200, 0)),
        ]:
            # WB133 palm (ground truth target)
            wb_start = WB_L_START if side == "left" else WB_R_START
            for off in WB_PALM_LOCAL:
                p = wb133[fidx, wb_start + off]
                if p[2] > WB_CONF_THRESH:
                    cv2.circle(col_b, (int(p[0]), int(p[1])), 6, c_wb, -1)
                    cv2.circle(col_a, (int(p[0]), int(p[1])), 6, c_wb, -1)

            # BEFORE: original HaMeR projection (from visual_critic_loop kp3d direct)
            if fidx in prefit and side in prefit[fidx]:
                d = prefit[fidx][side]
                # Before: joints_cam_orig projection
                j_orig = d.get("joints_cam_orig")  # (21, 3)
                if j_orig is not None:
                    focal = d["focal"]
                    w, h = d["img_size"]
                    cx, cy = w/2, h/2
                    for pj in MANO_PALM_IDX:
                        z = max(j_orig[pj, 2], 1e-4)
                        u = int(focal * j_orig[pj, 0] / z + cx)
                        v = int(focal * j_orig[pj, 1] / z + cy)
                        cv2.circle(col_b, (u, v), 7, c_hamer, -1)
                        cv2.circle(col_b, (u, v), 9, (0,0,0), 1)

                # After: pre-fit projected palm
                uv_palm = d.get("uv_palm")  # (5, 2)
                if uv_palm is not None:
                    for u, v in uv_palm:
                        cv2.circle(col_a, (int(u), int(v)), 7, c_hamer, -1)
                        cv2.circle(col_a, (int(u), int(v)), 9, (0,0,0), 1)

        # Labels
        for col, lbl in [(col_b, f"f={fidx} BEFORE"), (col_a, f"f={fidx} AFTER")]:
            cv2.putText(col, lbl, (5, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2); cv2.putText(col, lbl, (5, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        H, W = img.shape[:2]
        cell = lambda c: cv2.resize(c, (int(W*scale), int(H*scale)),
                                    interpolation=cv2.INTER_AREA)
        strips.append(np.concatenate([cell(col_b), cell(col_a)], axis=1))

    if not strips:
        return
    sw = strips[0].shape[1]
    header = np.zeros((22, sw, 3), dtype=np.uint8)
    for i, lbl in enumerate(["BEFORE (HaMeR raw)", "AFTER (MANO pre-fit)"]):
        cv2.putText(header, lbl, (i*sw//2+4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    sheet = np.concatenate([header] + strips, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"[mano_prefit] debug sheet → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _apply_z_anchor(
    kp3d_np:        np.ndarray,   # (N, 21, 3) right-hand canonical (no x-flip yet)
    cam_t_np:       np.ndarray,   # (N, 3)
    frame_ids:      list[int],
    gvhmr_wrist_cam:np.ndarray,   # (F, 2, 3) meters, [:, 0]=left, [:, 1]=right
    focal_hamer:    float,
    focal_gvhmr:    float,
    is_right:       bool,
    thresh_m:       float,
    side_name:      str,
    alpha:          float = 1.0,  # blend: 0=HaMeR only, 1=full GVHMR anchor
) -> np.ndarray:
    """Return corrected cam_t_np (copy) with blended Z+XY anchor applied per-frame.

    For each frame where HaMeR wrist Z > GVHMR wrist Z + thresh_m:
      Z_new = Z_hamer * (1 - alpha) + Z_gvhmr_in_hamer_units * alpha
      XY scaled proportionally so the projected wrist pixel stays unchanged.

    alpha=1.0 : full override (original behaviour)
    alpha=0.5 : blend halfway between HaMeR and GVHMR depth
    """
    N = kp3d_np.shape[0]
    wrist_kp3d = kp3d_np[:, 0, :].copy()   # (N, 3)
    if not is_right:
        wrist_kp3d[:, 0] *= -1

    # Wrist in camera space (HaMeR units)
    wrist_cam_x = wrist_kp3d[:, 0] + cam_t_np[:, 0]
    wrist_cam_y = wrist_kp3d[:, 1] + cam_t_np[:, 1]
    wrist_cam_z = wrist_kp3d[:, 2] + cam_t_np[:, 2]

    # GVHMR anchor in HaMeR units
    wi = 1 if is_right else 0
    F_gvhmr = gvhmr_wrist_cam.shape[0]
    z_gvhmr_m = np.array(
        [gvhmr_wrist_cam[f, wi, 2] if f < F_gvhmr else float("nan")
         for f in frame_ids], np.float32
    )
    z_anchor_hamer = z_gvhmr_m * (focal_hamer / focal_gvhmr)  # (N,)

    # Apply only when HaMeR Z exceeds anchor by thresh
    z_diff_m = (wrist_cam_z - z_anchor_hamer) * (focal_gvhmr / focal_hamer)
    apply_mask = (z_diff_m > thresh_m) & np.isfinite(z_anchor_hamer)

    if not apply_mask.any():
        return cam_t_np

    cam_t_out = cam_t_np.copy()
    idx = np.where(apply_mask)[0]

    # Blended Z target
    z_new = wrist_cam_z[idx] * (1.0 - alpha) + z_anchor_hamer[idx] * alpha
    scale = z_new / wrist_cam_z[idx]               # z_new / z_old

    # XY adjusted so projected wrist pixel is unchanged
    cam_t_out[idx, 0] = wrist_cam_x[idx] * scale - wrist_kp3d[idx, 0]
    cam_t_out[idx, 1] = wrist_cam_y[idx] * scale - wrist_kp3d[idx, 1]
    cam_t_out[idx, 2] = z_new - wrist_kp3d[idx, 2]

    n = apply_mask.sum()
    z_diff_applied = z_diff_m[apply_mask]
    print(f"[mano_prefit] {side_name}: Z anchor applied to {n}/{N} frames "
          f"(alpha={alpha:.1f} thresh={thresh_m:.1f}m "
          f"z_diff mean={z_diff_applied.mean():.2f}m max={z_diff_applied.max():.2f}m)")
    return cam_t_out


def _compute_gvhmr_wrist_cam(
    stage1a_path: Path,
    device: str,
) -> tuple[np.ndarray, float]:
    """SMPLX forward pass on stage1a params → wrist camera-space positions.

    Returns:
        wrist_cam: (F, 2, 3)  float32, camera-space XYZ in GVHMR meters.
                   axis-1: 0=left wrist (joint 20), 1=right wrist (joint 21).
        focal_gvhmr: scalar focal length of the GVHMR camera.
    """
    import sys
    sys.path.insert(0, str(PROJ_ROOT))
    from hmr4d.utils.smplx_utils import make_smplx

    data   = torch.load(stage1a_path, map_location="cpu")
    params = data["smpl_params_incam"]
    K      = data["K_fullimg"]              # (F, 3, 3)
    focal_gvhmr = float(K[0, 0, 0].item())

    smplx_m = make_smplx("supermotion_fullhands").to(device)
    F_total = params["body_pose"].shape[0]
    BATCH   = 256
    wrist_l, wrist_r = [], []

    with torch.no_grad():
        for i in range(0, F_total, BATCH):
            sl = slice(i, i + BATCH)
            out = smplx_m.forward(
                betas          = params["betas"][sl].to(device),
                global_orient  = params["global_orient"][sl].to(device),
                body_pose      = params["body_pose"][sl].to(device),
                left_hand_pose = params["left_hand_pose"][sl].to(device),
                right_hand_pose= params["right_hand_pose"][sl].to(device),
                transl         = params["transl"][sl].to(device),
            )
            j = out.joints.cpu()            # (B, J, 3)
            wrist_l.append(j[:, 20, :])     # left  wrist
            wrist_r.append(j[:, 21, :])     # right wrist

    wl = torch.cat(wrist_l, dim=0).numpy()  # (F, 3)
    wr = torch.cat(wrist_r, dim=0).numpy()  # (F, 3)
    wrist_cam = np.stack([wl, wr], axis=1).astype(np.float32)  # (F, 2, 3)
    return wrist_cam, focal_gvhmr


def run_mano_prefit(
    hamer_dir:    Path,
    wb133_path:   Path,
    output_path:  Path,
    frame_dir:    Path,
    debug_dir:    Path,
    metrics_csv:  Path,
    *,
    iters:         int   = 200,
    lr:            float = 1e-2,
    w_palm_base:   float = 2.0,
    w_palm_outlier:float = 3.0,  # outlier frames get stronger WB pull
    w_prior:       float = 5.0,
    w_tc:          float = 1e2,
    target_px:     float = 10.0,
    debug_frames:  int   = 12,
    device:        str   = "cuda",
    # Z anchor
    gvhmr_stage1a_path: Path | None = None,
    z_anchor_thresh_m:  float       = 0.5,
    z_anchor_alpha:     float       = 1.0,
) -> Path:

    print("[mano_prefit] Loading WB133...")
    wb133 = np.load(wb133_path).astype(np.float32)   # (F, 133, 3)
    F_total = wb133.shape[0]

    # ---- GVHMR wrist Z anchor (optional) ----
    gvhmr_wrist_cam: np.ndarray | None = None
    focal_gvhmr: float = 0.0
    if gvhmr_stage1a_path is not None and Path(gvhmr_stage1a_path).exists():
        print(f"[mano_prefit] Computing GVHMR wrist positions from {gvhmr_stage1a_path.name} ...")
        gvhmr_wrist_cam, focal_gvhmr = _compute_gvhmr_wrist_cam(
            Path(gvhmr_stage1a_path), device
        )
        print(f"[mano_prefit] GVHMR wrist cam: F={gvhmr_wrist_cam.shape[0]}"
              f"  focal_gvhmr={focal_gvhmr:.1f}"
              f"  Z_left mean={gvhmr_wrist_cam[:,0,2].mean():.3f}m"
              f"  Z_right mean={gvhmr_wrist_cam[:,1,2].mean():.3f}m")
    else:
        if gvhmr_stage1a_path is not None:
            print(f"[mano_prefit] WARNING: {gvhmr_stage1a_path} not found — Z anchor skipped")
        else:
            print("[mano_prefit] Z anchor disabled (no --gvhmr-stage1a)")

    # Load outlier scores
    try:
        import pandas as pd
        df_m = pd.read_csv(metrics_csv)
        err_per_frame = np.zeros(F_total, np.float32)
        for row in df_m.itertuples():
            fidx = int(row.frame)
            if fidx < F_total:
                err_per_frame[fidx] = max(err_per_frame[fidx],
                                          float(row.outlier_total))
        print(f"[mano_prefit] outlier scores loaded: "
              f"mean={err_per_frame.mean():.3f} max={err_per_frame.max():.3f}")
    except Exception as e:
        print(f"[mano_prefit] outlier scores unavailable ({e})")
        err_per_frame = np.zeros(F_total, np.float32)

    prefit: dict = {}   # final output

    for side in ("left", "right"):
        print(f"\n[mano_prefit] === {side.upper()} hand ===")

        # Load HaMeR detections for this side
        det = load_hamer_side(hamer_dir, side)
        if not det:
            print(f"[mano_prefit] no detections for {side}, skip")
            continue

        frame_ids = sorted(det.keys())
        N = len(frame_ids)
        print(f"[mano_prefit] {N} detected frames")

        # Build tensors
        kp3d_np  = np.array([det[f]["kp3d"]      for f in frame_ids], np.float32)
        cam_t_np = np.array([det[f]["cam_t_full"] for f in frame_ids], np.float32)
        focal    = float(det[frame_ids[0]]["focal"])
        img_size = det[frame_ids[0]]["img_size"]   # [W, H]
        img_w, img_h = float(img_size[0]), float(img_size[1])
        is_right = (side == "right")

        # ---- Z + XY anchor: pre-correct cam_t before optimization ----
        # Brings HaMeR hand depth in line with GVHMR wrist Z.
        # Applied only when z_diff > z_anchor_thresh_m (avoids worsening frames
        # where HaMeR Z is already at or below GVHMR).
        # XY is scaled simultaneously to keep projected wrist position unchanged,
        # so optimizer delta_t starts with minimal XY displacement.
        if gvhmr_wrist_cam is not None and focal_gvhmr > 0:
            cam_t_np = _apply_z_anchor(
                kp3d_np, cam_t_np, frame_ids,
                gvhmr_wrist_cam, focal, focal_gvhmr,
                is_right, z_anchor_thresh_m,
                side, alpha=z_anchor_alpha,
            )

        kp3d_t  = torch.from_numpy(kp3d_np).to(device)
        cam_t_t = torch.from_numpy(cam_t_np).to(device)

        # WB133 palm targets (no conf gating — always use WB133)
        wb_uv_np, wb_conf_np = load_wb133_palm(wb133, frame_ids, side)
        wb_uv_t   = torch.from_numpy(wb_uv_np).to(device)
        wb_conf_t = torch.from_numpy(wb_conf_np).to(device)

        # Per-frame outlier scores
        err_t_np = np.array([err_per_frame[f] for f in frame_ids], np.float32)
        err_t    = torch.from_numpy(err_t_np).to(device)

        # Initial palm RMS (before optimization)
        if not is_right:
            kp3d_xflip_np = kp3d_np.copy(); kp3d_xflip_np[:, :, 0] *= -1
        else:
            kp3d_xflip_np = kp3d_np
        j_orig_np = kp3d_xflip_np + cam_t_np[:, np.newaxis, :]   # (N,21,3)
        palm_orig_np = j_orig_np[:, MANO_PALM_IDX, :]              # (N,5,3)
        z0 = np.maximum(palm_orig_np[:, :, 2], 1e-4)
        u0 = focal * palm_orig_np[:, :, 0] / z0 + img_w / 2
        v0 = focal * palm_orig_np[:, :, 1] / z0 + img_h / 2
        diff0 = np.stack([u0, v0], -1) - wb_uv_np   # (N,5,2)
        mask0 = wb_conf_np > 0
        if mask0.sum() > 0:
            rms_init = float(np.sqrt(((diff0**2).sum(-1)[mask0]).mean()))
        else:
            rms_init = float("nan")
        print(f"[mano_prefit] initial palm RMS = {rms_init:.1f}px")

        # ---- Optimize ----
        delta_rvec, delta_t, last_rms, best_rms, stop_reason, iters_run = optimize_side(
            frame_ids, kp3d_t, cam_t_t, focal, img_w, img_h, is_right,
            wb_uv_t, wb_conf_t, err_t,
            iters=iters, lr=lr,
            w_palm_base=w_palm_base, w_palm_outlier=w_palm_outlier,
            w_prior=w_prior, w_tc=w_tc,
            target_px=target_px, device=device,
        )
        print(f"[mano_prefit] {side}: {rms_init:.1f}px → best={best_rms:.1f}px  "
              f"stop={stop_reason} iters={iters_run}")

        # ---- Compute final pre-fit joints_cam for each frame ----
        delta_rvec_np = delta_rvec.cpu().numpy()  # (N, 3)
        delta_t_np    = delta_t.cpu().numpy()     # (N, 3)

        kp3d_xflip = kp3d_xflip_np.copy()
        joints_orig = kp3d_xflip + cam_t_np[:, np.newaxis, :]   # (N,21,3)
        wrist_cam   = joints_orig[:, 0:1, :].copy()

        for ni, fidx in enumerate(frame_ids):
            # Apply delta_R and delta_t
            rvec = delta_rvec_np[ni]
            import cv2 as _cv2
            R_delta, _ = _cv2.Rodrigues(rvec.astype(np.float64))
            R_delta = R_delta.astype(np.float32)

            centered  = joints_orig[ni] - wrist_cam[ni, 0]      # (21,3)
            centered_r = (R_delta @ centered.T).T                 # (21,3)
            joints_new = centered_r + wrist_cam[ni, 0] + delta_t_np[ni]  # (21,3)

            # Projected palm joints
            palm_new = joints_new[MANO_PALM_IDX]   # (5,3)
            z2 = np.maximum(palm_new[:, 2], 1e-4)
            u2 = focal * palm_new[:, 0] / z2 + img_w / 2
            v2 = focal * palm_new[:, 1] / z2 + img_h / 2
            uv_palm = np.stack([u2, v2], axis=-1).astype(np.float32)  # (5,2)

            if fidx not in prefit:
                prefit[fidx] = {}
            prefit[fidx][side] = {
                "joints_cam":      joints_new.astype(np.float32),    # (21,3) after prefit
                "joints_cam_orig": joints_orig[ni].astype(np.float32), # (21,3) before
                "uv_palm":         uv_palm,                           # (5,2)
                "delta_rvec":      rvec.astype(np.float32),
                "delta_t":         delta_t_np[ni].astype(np.float32),
                "focal":           focal,
                "img_size":        list(img_size),
            }

    # ---- Save ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prefit, output_path)
    print(f"\n[mano_prefit] saved → {output_path}  ({len(prefit)} frames)")

    # ---- Debug contact sheet ----
    all_fidxs  = sorted(prefit.keys())
    step       = max(1, len(all_fidxs) // debug_frames)
    selected   = all_fidxs[::step][:debug_frames]
    debug_dir.mkdir(parents=True, exist_ok=True)
    generate_prefit_debug_sheet(
        frame_dir, prefit, wb133, selected,
        debug_dir / "mano_prefit_contact_sheet.jpg",
    )

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2: HaMeR MANO pre-fit")
    parser.add_argument("--input-dir", default="outputs/cap_pipeline/input")
    parser.add_argument("--iters",          type=int,   default=200)
    parser.add_argument("--lr",             type=float, default=1e-2)
    parser.add_argument("--w-palm-base",    type=float, default=2.0)
    parser.add_argument("--w-palm-outlier", type=float, default=3.0)
    parser.add_argument("--w-prior",        type=float, default=5.0)
    parser.add_argument("--w-tc",           type=float, default=1e2)
    parser.add_argument("--target-px",      type=float, default=10.0)
    parser.add_argument("--debug-frames",   type=int,   default=12)
    parser.add_argument("--device",                     default="cuda")
    parser.add_argument("--gvhmr-stage1a",  default=None,
                        help="Path to smplx_stage1a.pt for GVHMR wrist Z anchor. "
                             "Defaults to <input-dir>/smplx_stage1a.pt. "
                             "Pass 'none' to disable.")
    parser.add_argument("--z-anchor-thresh", type=float, default=0.5,
                        help="Only anchor frames where z_diff > this value (meters).")
    parser.add_argument("--z-anchor-alpha",  type=float, default=1.0,
                        help="Blend: 0=HaMeR only, 1=full GVHMR anchor (default 1.0).")
    args = parser.parse_args()

    d = Path(args.input_dir)

    # Resolve GVHMR stage1a path
    if args.gvhmr_stage1a is None:
        candidate = d / "smplx_stage1a.pt"
        stage1a_path = candidate if candidate.exists() else None
    elif args.gvhmr_stage1a.lower() == "none":
        stage1a_path = None
    else:
        stage1a_path = Path(args.gvhmr_stage1a)

    run_mano_prefit(
        hamer_dir    = d / "hamer_out_clean",
        wb133_path   = d / "preprocess" / "wholebody_kp2d.npy",
        output_path  = d / "hamer_mano_prefit.pt",
        frame_dir    = d / "hamer_frames",
        debug_dir    = d / "wrist_refine_debug" / "mano_prefit",
        metrics_csv  = d / "wrist_refine_debug" / "self_feedback_visual_critic"
                         / "metrics" / "frame_side_metrics.csv",
        iters          = args.iters,
        lr             = args.lr,
        w_palm_base    = args.w_palm_base,
        w_palm_outlier = args.w_palm_outlier,
        w_prior        = args.w_prior,
        w_tc           = args.w_tc,
        target_px      = args.target_px,
        debug_frames   = args.debug_frames,
        device         = args.device,
        gvhmr_stage1a_path = stage1a_path,
        z_anchor_thresh_m  = args.z_anchor_thresh,
        z_anchor_alpha     = args.z_anchor_alpha,
    )


if __name__ == "__main__":
    main()
