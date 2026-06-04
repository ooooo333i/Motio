"""
cap_arm_prefit.py

Stage 1: GVHMR Arm-chain Pre-fit
  GVHMR body_pose의 팔 chain (collar + shoulder + elbow + wrist)을
  ViTPose 2D keypoint 기준으로 약하게 보정합니다.

Loss:
  L = w_vp(t) * L_2d  +  w_prior(t) * L_prior  +  w_tc * L_temporal

  - L_2d:    SMPL-X projected shoulder/elbow/wrist  vs  ViTPose 2D keypoints
  - L_prior: 원래 GVHMR body_pose에서 멀어지지 않도록 L2 regularization
  - L_temporal: adjacent frame 간 geodesic distance smoothness
  - w_vp(t) / w_prior(t): outlier score에 따라 adaptive per-frame weight

최적화 대상 (body_pose index = full_joint_id - 1):
  collar_L(12), collar_R(13)
  shoulder_L(15), shoulder_R(16)
  elbow_L(17),   elbow_R(18)
  wrist_L(19),   wrist_R(20)

고정: spine, pelvis, legs, head, global_orient, transl, betas, hand_pose

사용법:
  python cap_arm_prefit.py
  python cap_arm_prefit.py --iters 300 --lr 2e-2 --debug-frames 12
"""

from __future__ import annotations

import argparse
import copy
import os
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

# body_pose index = full SMPL-X joint id - 1
ARM_OPT_IDS = sorted({
    12, 13,         # collar_L, collar_R       (joint 13, 14)
    15, 16,         # shoulder_L, shoulder_R   (joint 16, 17)
    17, 18,         # elbow_L, elbow_R         (joint 18, 19)
    19, 20,         # wrist_L, wrist_R         (joint 20, 21)
})

# SMPL-X *output* joint index (out.joints) for 2D loss
# Mapping: (ViTPose COCO17 index, SMPL-X joints output index)
VP_ARM_MAP = [
    (5, 16),   # left_shoulder
    (6, 17),   # right_shoulder
    (7, 18),   # left_elbow
    (8, 19),   # right_elbow
    (9, 20),   # left_wrist
    (10, 21),  # right_wrist
]
VP_IDXS    = [v for v, _ in VP_ARM_MAP]
SMPLX_JIDS = [s for _, s in VP_ARM_MAP]

# ---------------------------------------------------------------------------
# Rotation helpers (reused from cap_wrist_refine.py)
# ---------------------------------------------------------------------------

def aa_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle → (..., 3, 3)."""
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / theta
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = torch.cos(theta)[..., 0]
    s = torch.sin(theta)[..., 0]
    C = 1 - c
    return torch.stack([
        c + x*x*C,      x*y*C - z*s,  x*z*C + y*s,
        y*x*C + z*s,  c + y*y*C,      y*z*C - x*s,
        z*x*C - y*s,  z*y*C + x*s,  c + z*z*C,
    ], dim=-1).reshape(aa.shape[:-1] + (3, 3))


def geodesic_distance_batch(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Mean geodesic distance over all (...) dims of (F, J, 3, 3) rotmats."""
    Rrel = R1.transpose(-1, -2) @ R2
    trace = Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]
    cos = ((trace - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos).mean()


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_incam(joints3d: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """(F, J, 3) in-cam + (F,3,3) K → (F, J, 2) pixels."""
    x, y, z = joints3d[..., 0], joints3d[..., 1], joints3d[..., 2].clamp(min=1e-4)
    fx = K[:, 0, 0].unsqueeze(-1)
    fy = K[:, 1, 1].unsqueeze(-1)
    cx = K[:, 0, 2].unsqueeze(-1)
    cy = K[:, 1, 2].unsqueeze(-1)
    return torch.stack([fx * (x / z) + cx, fy * (y / z) + cy], dim=-1)


# ---------------------------------------------------------------------------
# Outlier score loading
# ---------------------------------------------------------------------------

def load_outlier_scores_per_side(
    metrics_csv: Path, n_frames: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (err_left, err_right) each (F,) ∈ [0,1], per-side outlier score."""
    zeros = torch.zeros(n_frames), torch.zeros(n_frames)
    try:
        import pandas as pd
        df = pd.read_csv(metrics_csv)
        err_L = torch.zeros(n_frames)
        err_R = torch.zeros(n_frames)
        for fidx in range(n_frames):
            rows = df[df.frame == fidx]
            if rows.empty:
                continue
            row_l = rows[rows.side == "left"]
            row_r = rows[rows.side == "right"]
            if not row_l.empty:
                err_L[fidx] = float(row_l["outlier_total"].max())
            if not row_r.empty:
                err_R[fidx] = float(row_r["outlier_total"].max())
        bad_L = int((err_L > 0.5).sum())
        bad_R = int((err_R > 0.5).sum())
        print(f"[arm_prefit] outlier scores per-side: "
              f"L mean={err_L.mean():.3f} bad>{0.5}={bad_L}  "
              f"R mean={err_R.mean():.3f} bad>{0.5}={bad_R}")
        return err_L, err_R
    except Exception as e:
        print(f"[arm_prefit] outlier scores not available ({e}), using uniform weights")
        return zeros


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------

def draw_arm_overlay(img: np.ndarray, uv: np.ndarray,
                     vp: np.ndarray, label: str = "") -> np.ndarray:
    """Draw SMPL-X arm joints (circles) + ViTPose arm (crosses) on image."""
    out = img.copy()
    H, W = out.shape[:2]

    # SMPL-X projected joints
    smplx_colors = {
        16: (255, 100, 0), 17: (0, 100, 255),    # shoulder (blue/red)
        18: (200, 150, 0), 19: (0, 150, 200),    # elbow
        20: (255, 50, 50), 21: (50, 50, 255),    # wrist
    }
    for jid, color in smplx_colors.items():
        u, v = float(uv[jid, 0]), float(uv[jid, 1])
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(out, (int(u), int(v)), 10, color, -1)
            cv2.circle(out, (int(u), int(v)), 12, (0, 0, 0), 1)

    # ViTPose arm keypoints
    vp_map = [(5, 16), (6, 17), (7, 18), (8, 19), (9, 20), (10, 21)]
    for vp_idx, smplx_jid in vp_map:
        p = vp[vp_idx]
        if p[2] < 0.2:
            continue
        color = smplx_colors.get(smplx_jid, (200, 200, 200))
        cv2.drawMarker(out, (int(p[0]), int(p[1])), color,
                       cv2.MARKER_CROSS, 18, 2)

    # Draw lines: shoulder-elbow-wrist
    for a, b in [(16, 18), (18, 20), (17, 19), (19, 21)]:
        ua, va = float(uv[a, 0]), float(uv[a, 1])
        ub, vb = float(uv[b, 0]), float(uv[b, 1])
        if all(0 <= x < W for x in [ua, ub]) and all(0 <= y < H for y in [va, vb]):
            cv2.line(out, (int(ua), int(va)), (int(ub), int(vb)),
                     (180, 180, 180), 2)

    if label:
        cv2.putText(out, label, (5, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, label, (5, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def generate_debug_contact_sheet(
    smplx_model,
    params_before: dict,
    params_after: dict,
    K_fullimg: torch.Tensor,
    vitpose: torch.Tensor,
    frame_dir: Path,
    out_path: Path,
    n_frames: int = 12,
    device: str = "cuda",
):
    """Before/After contact sheet for arm pre-fit verification."""
    F = params_before["body_pose"].shape[0]
    K = K_fullimg.to(device)
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(F, -1, -1)

    frame_files = sorted(frame_dir.glob("*.jpg"))
    step = max(1, F // n_frames)
    selected = list(range(0, F, step))[:n_frames]

    def get_uv(params, fidx):
        with torch.no_grad():
            betas = params["betas"][fidx:fidx+1].to(device)
            go    = params["global_orient"][fidx:fidx+1].to(device)
            bp    = params["body_pose"][fidx:fidx+1].to(device)
            lh    = params["left_hand_pose"][fidx:fidx+1].to(device)
            rh    = params["right_hand_pose"][fidx:fidx+1].to(device)
            tr    = params["transl"][fidx:fidx+1].to(device)
            fwd_kw = dict(betas=betas, global_orient=go, body_pose=bp,
                          left_hand_pose=lh, right_hand_pose=rh, transl=tr)
            for key in ("jaw_pose", "leye_pose", "reye_pose"):
                if key in params:
                    fwd_kw[key] = params[key][fidx:fidx+1].to(device)
            out = smplx_model.forward(**fwd_kw)
            j3d = out.joints[0]           # (J, 3)
            Kf  = K[fidx]                 # (3, 3)
            z   = j3d[:, 2].clamp(min=1e-4)
            u   = Kf[0, 0] * j3d[:, 0] / z + Kf[0, 2]
            v   = Kf[1, 1] * j3d[:, 1] / z + Kf[1, 2]
            return torch.stack([u, v], dim=1).cpu().numpy()

    SCALE = 0.22
    strips = []
    for fidx in selected:
        if fidx >= len(frame_files):
            break
        img = cv2.imread(str(frame_files[fidx]))
        if img is None:
            continue
        vp = vitpose[fidx].numpy()

        uv_bef = get_uv(params_before, fidx)
        uv_aft = get_uv(params_after,  fidx)

        # Compute per-joint pixel error (wrist and elbow)
        errs = []
        for vp_i, smplx_j in [(9, 20), (10, 21), (7, 18), (8, 19)]:
            if vp[vp_i, 2] > 0.3:
                e = np.linalg.norm(uv_aft[smplx_j] - vp[vp_i, :2])
                errs.append(e)
        err_str = f"arm_err={np.mean(errs):.0f}px" if errs else ""

        c_bef = draw_arm_overlay(img, uv_bef, vp, label=f"f={fidx} BEFORE")
        c_aft = draw_arm_overlay(img, uv_aft, vp, label=f"f={fidx} AFTER  {err_str}")

        def cell(im):
            H, W = im.shape[:2]
            return cv2.resize(im, (int(W*SCALE), int(H*SCALE)),
                              interpolation=cv2.INTER_AREA)

        strip = np.concatenate([cell(c_bef), cell(c_aft)], axis=1)
        strips.append(strip)

    if not strips:
        print("[arm_prefit] no frames available for debug sheet")
        return

    # Header
    sw = strips[0].shape[1]
    header = np.zeros((22, sw, 3), dtype=np.uint8)
    cw = sw // 2
    for i, lbl in enumerate(["BEFORE (GVHMR)", "AFTER (arm pre-fit)"]):
        cv2.putText(header, lbl, (i*cw+4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    sheet = np.concatenate([header] + strips, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"[arm_prefit] debug contact sheet → {out_path}")


# ---------------------------------------------------------------------------
# Main optimization
# ---------------------------------------------------------------------------

def run_arm_prefit(
    gvhmr_path: Path,
    vitpose_path: Path,
    output_path: Path,
    frame_dir: Path,
    debug_dir: Path,
    metrics_csv: Path,
    *,
    iters: int = 300,
    lr: float = 2e-2,
    w_vp_base: float = 1.0,
    w_wrist_base: float = 0.6,         # wrist gets lower base weight than shoulder/elbow
    w_wrist_outlier_scale: float = 1.0, # per-side: w_wrist * max(1 - scale*err_side, 0.1)
    w_prior_base: float = 5.0,
    w_prior_min: float = 0.5,
    w_tc: float = 1e3,
    conf_thresh: float = 0.3,
    min_iter: int = 60,
    patience: int = 40,
    target_px: float = 20.0,
    debug_frames: int = 12,
    device: str = "cuda",
) -> Path:
    """Stage 1: arm-chain pre-fit on full F-frame sequence."""

    # ---- Load data ----
    print("[arm_prefit] Loading GVHMR results...")
    pred = torch.load(gvhmr_path, map_location="cpu")
    params_orig = copy.deepcopy(pred["smpl_params_incam"])
    K_fullimg   = pred["K_fullimg"].to(device)   # (F,3,3)
    if K_fullimg.dim() == 2:
        F_from_K = 1
    else:
        F_from_K = K_fullimg.shape[0]

    F = params_orig["body_pose"].shape[0]
    if K_fullimg.dim() == 2:
        K_fullimg = K_fullimg.unsqueeze(0).expand(F, -1, -1)
    print(f"[arm_prefit] F={F} frames, device={device}")

    print("[arm_prefit] Loading ViTPose...")
    vitpose = torch.load(vitpose_path, map_location="cpu")  # (F, 17, 3)
    if isinstance(vitpose, torch.Tensor):
        vitpose_np = vitpose.numpy()
    else:
        vitpose_np = np.array(vitpose, dtype=np.float32)

    # ---- Outlier scores (per-side for wrist gating) ----
    err_L, err_R = load_outlier_scores_per_side(metrics_csv, F)
    if float(err_L.abs().sum() + err_R.abs().sum()) == 0.0:
        # csv 없음(일반화) → ViTPose 손목 conf+속도로 신뢰도 게이팅 직접 계산.
        # 안 보이는 구간(저conf)·극단 점프(노이즈)에서 손목 2D 가중을 낮춰 GVHMR(평활)로 폴백. ~ms, 병목 아님.
        from scipy.ndimage import gaussian_filter1d as _g1
        def _conf_err(coco_idx):
            cf = vitpose_np[:, coco_idx, 2].astype(np.float32)
            xy = vitpose_np[:, coco_idx, :2].astype(np.float32)
            e_conf = np.clip((0.6 - cf) / 0.4, 0, 1)                    # conf<0.6 → 점점 unreliable, <0.2 → 1
            v = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
            vm = max(float(np.median(v)), 1.0)
            e_vel = np.clip((v - 6 * vm) / (6 * vm), 0, 1)             # 극단 점프(>6×중앙값=노이즈)만
            return _g1(np.maximum(e_conf, e_vel), 2.0)                  # 게이트 부드럽게
        err_L = torch.from_numpy(_conf_err(9))   # COCO17 L wrist
        err_R = torch.from_numpy(_conf_err(10))  # COCO17 R wrist
        print(f"[arm_prefit] conf기반 게이팅 폴백: L bad>0.5={int((err_L>0.5).sum())} R bad>0.5={int((err_R>0.5).sum())}")
    err_L = err_L.to(device)   # (F,)
    err_R = err_R.to(device)   # (F,)

    # Shoulder/elbow: uniform weight (body has essentially no outliers)
    # Wrist: per-side reliability — inverted so bad detection → lower weight
    wrist_rel_L = torch.clamp(1.0 - w_wrist_outlier_scale * err_L, min=0.1)  # (F,)
    wrist_rel_R = torch.clamp(1.0 - w_wrist_outlier_scale * err_R, min=0.1)  # (F,)

    # joint_scale: (F, 6) — shoulder/elbow=1.0, wrist L/R = per-side reliability
    # VP_ARM_MAP order: L_shoulder(0), R_shoulder(1), L_elbow(2), R_elbow(3), L_wrist(4), R_wrist(5)
    ones_F = torch.ones(F, device=device)
    joint_scale = torch.stack([
        ones_F, ones_F,         # shoulder L, R
        ones_F, ones_F,         # elbow L, R
        w_wrist_base * wrist_rel_L,   # wrist L
        w_wrist_base * wrist_rel_R,   # wrist R
    ], dim=1)   # (F, 6)

    # Prior: uniform (body doesn't need outlier-adaptive prior)
    w_prior_t = torch.full((F,), w_prior_base, device=device)

    # Log wrist weight stats
    print(f"[arm_prefit] wrist_rel_L: mean={wrist_rel_L.mean():.3f} "
          f"min={wrist_rel_L.min():.3f} ({(wrist_rel_L < 0.2).sum()} frames < 0.2)")
    print(f"[arm_prefit] wrist_rel_R: mean={wrist_rel_R.mean():.3f} "
          f"min={wrist_rel_R.min():.3f} ({(wrist_rel_R < 0.2).sum()} frames < 0.2)")

    # ---- ViTPose 2D targets ----
    # vp_uv: (F, 6, 2), vp_conf: (F, 6)
    vp_uv_np   = vitpose_np[:, VP_IDXS, :2].astype(np.float32)  # (F,6,2)
    vp_conf_np = vitpose_np[:, VP_IDXS, 2].astype(np.float32)   # (F,6)
    vp_conf_np = np.where(vp_conf_np >= conf_thresh, vp_conf_np, 0.0)
    vp_uv   = torch.from_numpy(vp_uv_np).to(device)      # (F,6,2)
    # Apply joint_scale: shoulder/elbow unchanged, wrist per-side reliability gated
    vp_conf = torch.from_numpy(vp_conf_np).to(device) * joint_scale  # (F,6)
    smplx_jids = torch.tensor(SMPLX_JIDS, device=device, dtype=torch.long)  # (6,)

    # ---- Load SMPL-X model ----
    print("[arm_prefit] Loading SMPL-X model...")
    from hmr4d.utils.smplx_utils import make_smplx
    smplx_model = make_smplx("supermotion_fullhands").to(device)

    # ---- Prepare frozen params ----
    body_pose_orig = params_orig["body_pose"].reshape(F, 21, 3).float().to(device)
    betas        = params_orig["betas"].float().to(device)
    global_orient = params_orig["global_orient"].float().to(device)
    transl       = params_orig["transl"].float().to(device)
    lhand        = params_orig["left_hand_pose"].reshape(F, 45).float().to(device)
    rhand        = params_orig["right_hand_pose"].reshape(F, 45).float().to(device)
    extra_keys = {}
    for key in ("jaw_pose", "leye_pose", "reye_pose"):
        if key in params_orig:
            extra_keys[key] = params_orig[key].float().to(device)

    # ---- Optimization variable: full (F,21,3) — non-arm grads zeroed each step ----
    arm_opt_ids = torch.tensor(ARM_OPT_IDS, device=device, dtype=torch.long)
    body_free = body_pose_orig.clone().requires_grad_(True)  # (F, 21, 3)
    opt = torch.optim.Adam([body_free], lr=lr)

    # body_mask[i]=1 iff i in ARM_OPT_IDS; broadcast to (21,1) → (F,21,3) later
    body_mask = torch.zeros(21, 1, device=device)
    for i in ARM_OPT_IDS:
        body_mask[i] = 1.0

    # For L_prior: only arm joints matter
    arm_orig = body_pose_orig[:, arm_opt_ids, :].detach()   # (F, 8, 3)
    # Alias arm_free to mean body_free (renaming to keep rest of code consistent)
    arm_free = body_free

    print(f"[arm_prefit] Optimizing {len(ARM_OPT_IDS)} joints × {F} frames = "
          f"{len(ARM_OPT_IDS)*F*3} parameters")
    print(f"[arm_prefit] ARM_OPT_IDS (body_pose): {ARM_OPT_IDS}")

    # ---- Optimization loop ----
    best_rms = float("inf")
    best_state = None
    best_it = -1
    bad_count = 0
    stop_reason = "max_iter"
    arm_rms = float("nan")

    def _arm_rms_px(uv_detach: torch.Tensor) -> float:
        """RMS pixel error on shoulder/elbow/wrist."""
        pred_uv = uv_detach[:, smplx_jids, :]         # (F,6,2)
        diff2   = ((pred_uv - vp_uv) ** 2).sum(-1)    # (F,6)
        w       = vp_conf                              # (F,6)
        numer   = float((diff2 * w).sum().item())
        denom   = float(w.sum().item())
        if denom <= 0:
            return float("nan")
        return float(np.sqrt(numer / denom))

    # Depth anchor: GVHMR arm-joint cam-z (매끄러운 깊이 기준). env CAP_ARM_DEPTH (기본 0=off).
    import os as _os
    W_DEPTH = float(_os.environ.get("CAP_ARM_DEPTH", "0"))
    with torch.no_grad():
        _gv = smplx_model.forward(betas=betas, global_orient=global_orient,
            body_pose=body_pose_orig.detach().reshape(F, -1),
            left_hand_pose=lhand, right_hand_pose=rhand, transl=transl, **extra_keys)
        z_gv = _gv.joints[:, smplx_jids, 2].detach()   # (F,6) GVHMR 팔관절 cam-z
    if W_DEPTH > 0:
        print(f"[arm_prefit] depth anchor ON: w_depth={W_DEPTH}")

    for it in range(iters):
        opt.zero_grad()

        # Assemble body_pose: frozen rows from orig, arm rows from body_free
        # body_mask shape (21,1) broadcasts to (F,21,3)
        bp = body_pose_orig.detach() * (1 - body_mask) + body_free * body_mask  # (F,21,3)

        # SMPL-X forward
        fwd_kw = dict(
            betas=betas,
            global_orient=global_orient,
            body_pose=bp.reshape(F, -1),
            left_hand_pose=lhand,
            right_hand_pose=rhand,
            transl=transl,
        )
        fwd_kw.update(extra_keys)
        out = smplx_model.forward(**fwd_kw)
        joints3d = out.joints   # (F, 127, 3)

        # Project to 2D
        uv = project_incam(joints3d, K_fullimg)   # (F, 127, 2)

        # L_2d: arm joint reprojection
        # vp_conf already includes joint_scale (per-side wrist reliability gating)
        pred_arm_uv = uv[:, smplx_jids, :]        # (F, 6, 2)
        diff = (pred_arm_uv - vp_uv)              # (F, 6, 2)
        l2_per_joint = (diff ** 2).sum(-1)        # (F, 6) squared px dist
        l2_per_frame = (l2_per_joint * vp_conf).sum(-1)   # (F,) weighted
        L_2d = l2_per_frame.mean()

        # L_prior: L2 from original GVHMR arm poses
        arm_current = body_free[:, arm_opt_ids, :]  # (F, 8, 3)
        prior_diff = (arm_current - arm_orig) ** 2  # (F, 8, 3)
        prior_per_frame = prior_diff.sum(-1).sum(-1)  # (F,)
        L_prior = (prior_per_frame * w_prior_t).mean()

        # L_temporal: geodesic smoothness (adjacent frames, arm joints only)
        R_arm = aa_to_rotmat(body_free[:, arm_opt_ids, :])  # (F, 8, 3, 3)
        L_tc  = geodesic_distance_batch(R_arm[:-1], R_arm[1:])

        # L_depth: 팔관절 cam-z를 GVHMR에 soft 앵커 (깊이 깜빡임 방지, 2D는 유지)
        L_depth = ((joints3d[:, smplx_jids, 2] - z_gv) ** 2).mean()
        loss = L_2d + L_prior + w_tc * L_tc + W_DEPTH * L_depth

        # Track arm RMS
        with torch.no_grad():
            arm_rms = _arm_rms_px(uv.detach())

        improved = arm_rms < best_rms - 0.5
        if improved:
            best_rms = arm_rms
            best_it  = it
            bad_count = 0
            best_state = body_free.detach().clone()
        else:
            bad_count += 1

        if it % 50 == 0 or it == iters - 1:
            print(f"[arm_prefit] it={it:4d}  loss={loss.item():.3f}  "
                  f"L_2d={L_2d.item():.3f}  L_prior={L_prior.item():.3f}  "
                  f"L_tc={float(L_tc):.4f}  arm_rms={arm_rms:.1f}px  "
                  f"best={best_rms:.1f}  bad={bad_count}")

        if it >= min_iter:
            if arm_rms <= target_px:
                stop_reason = "target_px"
                print(f"[arm_prefit] early stop @ it={it}: arm_rms={arm_rms:.1f} "
                      f"<= target_px={target_px}")
                break
            if bad_count >= patience:
                stop_reason = "plateau"
                print(f"[arm_prefit] early stop @ it={it}: plateau "
                      f"best={best_rms:.1f}px @ it={best_it}")
                break

        loss.backward()
        # Zero gradients for non-arm joints (belt-and-suspenders with body_mask)
        with torch.no_grad():
            if body_free.grad is not None:
                body_free.grad *= body_mask
        opt.step()

    # Restore best
    if best_state is not None:
        with torch.no_grad():
            body_free.copy_(best_state)
        print(f"[arm_prefit] restored best state @ it={best_it} "
              f"(arm_rms={best_rms:.1f}px)")

    # ---- Assemble final body_pose ----
    with torch.no_grad():
        bp_final = (body_pose_orig.detach() * (1 - body_mask)
                    + body_free.detach() * body_mask).cpu()    # (F, 21, 3)

    # ---- Build output params dict ----
    result = copy.deepcopy(pred)
    params_out = copy.deepcopy(params_orig)
    params_out["body_pose"] = bp_final.reshape(F, 21, 3)
    result["smpl_params_incam"] = params_out

    # Sync to global if present
    if "smpl_params_global" in result:
        bp_global = result["smpl_params_global"]["body_pose"].reshape(F, 21, 3).clone()
        for idx in ARM_OPT_IDS:
            bp_global[:, idx] = bp_final[:, idx]
        result["smpl_params_global"]["body_pose"] = bp_global.reshape(F, 21, 3)
        print("[arm_prefit] synced arm body_pose incam → smpl_params_global")

    # Store meta
    result["cap_arm_prefit_meta"] = {
        "iters_run": it + 1,
        "stop_reason": stop_reason,
        "best_arm_rms_px": float(best_rms),
        "best_it": best_it,
        "arm_opt_ids": ARM_OPT_IDS,
        "w_vp_base": w_vp_base,
        "w_wrist_base": w_wrist_base,
        "w_wrist_outlier_scale": w_wrist_outlier_scale,
        "w_prior_base": w_prior_base,
        "w_tc": w_tc,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    print(f"[arm_prefit] saved → {output_path}")
    print(f"[arm_prefit] done: stop={stop_reason} iters={it+1} "
          f"best_rms={best_rms:.1f}px")

    # ---- Debug contact sheet ----
    debug_dir.mkdir(parents=True, exist_ok=True)
    try:
        vitpose_t = torch.from_numpy(vitpose_np)
        generate_debug_contact_sheet(
            smplx_model,
            params_orig, params_out,
            pred["K_fullimg"],
            vitpose_t,
            frame_dir,
            debug_dir / "arm_prefit_contact_sheet.jpg",
            n_frames=debug_frames,
            device=device,
        )
    except Exception as e:
        print(f"[arm_prefit] debug sheet failed: {e}")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1: GVHMR arm-chain pre-fit")
    parser.add_argument("--input-dir", default="outputs/cap_pipeline/input",
                        help="Pipeline input directory")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--w-vp-base", type=float, default=1.0,
                        help="Base ViTPose reprojection weight for shoulder/elbow")
    parser.add_argument("--w-wrist-base", type=float, default=0.6,
                        help="Base weight for wrist joints (lower than shoulder/elbow)")
    parser.add_argument("--w-wrist-outlier-scale", type=float, default=1.0,
                        help="Per-side wrist gating: w_wrist*max(1-scale*err_side, 0.1)")
    parser.add_argument("--w-prior", type=float, default=5.0,
                        help="GVHMR prior weight")
    parser.add_argument("--w-tc", type=float, default=1e3,
                        help="Temporal consistency weight")
    parser.add_argument("--conf-thresh", type=float, default=0.3,
                        help="ViTPose keypoint confidence threshold")
    parser.add_argument("--target-px", type=float, default=20.0,
                        help="Early stop when arm RMS error < this (pixels)")
    parser.add_argument("--debug-frames", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    run_arm_prefit(
        gvhmr_path   = input_dir / "hmr4d_results.pt",
        vitpose_path = input_dir / "preprocess" / "vitpose.pt",
        output_path  = input_dir / "smplx_arm_prefit.pt",
        frame_dir    = input_dir / "hamer_frames",
        debug_dir    = input_dir / "wrist_refine_debug" / "arm_prefit",
        metrics_csv  = input_dir / "wrist_refine_debug"
                       / "self_feedback_visual_critic" / "metrics"
                       / "frame_side_metrics.csv",
        iters                    = args.iters,
        lr                       = args.lr,
        w_vp_base                = args.w_vp_base,
        w_wrist_base             = args.w_wrist_base,
        w_wrist_outlier_scale    = args.w_wrist_outlier_scale,
        w_prior_base             = args.w_prior,
        w_tc                     = args.w_tc,
        conf_thresh              = args.conf_thresh,
        target_px                = args.target_px,
        debug_frames             = args.debug_frames,
        device                   = args.device,
    )


if __name__ == "__main__":
    main()
