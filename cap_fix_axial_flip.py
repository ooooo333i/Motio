"""
cap_fix_axial_flip.py

Axial flip 보정:
  1. velocity(|Δaxial|>60°/frame) + |axial|>80° 결합으로 flip 구간 검출
  2. 각 구간 anchor_before / anchor_after (flip 경계 밖 첫 정상 프레임) 확정
  3. wrist LOCAL rotation (body_pose[f, idx]) → SO(3) SLERP → axis-angle 복원
  4. smplx_merged_stage4_axfix.pt 저장

입력:  outputs/cap_pipeline/input/smplx_merged_stage4.pt
출력:  outputs/cap_pipeline/input/smplx_merged_stage4_axfix.pt
       outputs/cap_pipeline/input/wrist_refine_debug/axial_analysis/flip_regions.txt
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

IN_PT  = PROJ_ROOT / "outputs/cap_pipeline/input/smplx_merged_stage4.pt"
OUT_PT = PROJ_ROOT / "outputs/cap_pipeline/input/smplx_merged_stage4_axfix.pt"
DIAG_DIR = PROJ_ROOT / "outputs/cap_pipeline/input/wrist_refine_debug/axial_analysis"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

VEL_THRESH = 60.0   # °/frame — velocity jump 감지
AX_THRESH  = 80.0   # °     — flip 내부 확인
ANCHOR_MARGIN = 3   # anchor 찾을 때 경계에서 최소 몇 프레임 떨어질지


# ─────────────────────────────────────────────────────────
# rotation helpers
# ─────────────────────────────────────────────────────────

def aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(aa))
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64)
    ax = aa / theta; x, y, z = ax
    c, s, t = np.cos(theta), np.sin(theta), 1.0 - np.cos(theta)
    return np.array([[t*x*x+c,   t*x*y-s*z, t*x*z+s*y],
                     [t*x*y+s*z, t*y*y+c,   t*y*z-s*x],
                     [t*x*z-s*y, t*y*z+s*x, t*z*z+c  ]], dtype=np.float64)


def rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    """SO(3) → axis-angle (Rodrigues log map)."""
    trace = float(np.clip(np.trace(R), -1.0 + 1e-7, 3.0 - 1e-7))
    theta = np.arccos((trace - 1.0) / 2.0)
    if abs(theta) < 1e-9:
        return np.zeros(3, dtype=np.float64)
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], dtype=np.float64)
    axis /= (2.0 * np.sin(theta))
    return axis * theta


def slerp_rotmat(Ra: np.ndarray, Rb: np.ndarray, t: float) -> np.ndarray:
    """Geodesic SLERP on SO(3): R(t) = Ra @ expm(t * logm(Ra^T @ Rb))."""
    Rrel = Ra.T @ Rb
    aa_rel = rotmat_to_aa(Rrel)   # log map
    aa_t   = aa_rel * t           # scale
    return Ra @ aa_to_rotmat(aa_t)


def euler_xyz_axial(aa: np.ndarray) -> float:
    """axis-angle → Euler X (axial) only."""
    R = aa_to_rotmat(aa)
    sy = float(np.clip(-R[2,0], -1.0, 1.0))
    cy = np.cos(np.arcsin(sy))
    if abs(cy) > 1e-6:
        return float(np.degrees(np.arctan2(R[2,1], R[2,2])))
    else:
        return float(np.degrees(np.arctan2(-R[1,2], R[1,1])))


# ─────────────────────────────────────────────────────────
# flip region 검출
# ─────────────────────────────────────────────────────────

def detect_flip_regions(axials: np.ndarray,
                        vel_thresh: float = VEL_THRESH,
                        ax_thresh:  float = AX_THRESH) -> list[dict]:
    """
    velocity 점프로 entry/exit 경계 찾고, |axial|>ax_thresh 로 내부 확인.
    반환: list of {entry_jump, exit_jump, region_start, region_end,
                  anchor_before, anchor_after}
    """
    F = len(axials)
    vel = np.diff(axials)

    # 1) velocity jump 위치 (점프 전 프레임 인덱스)
    jump_idxs = np.where(np.abs(vel) > vel_thresh)[0]

    # 2) flip 내부 프레임 (|axial| > ax_thresh)
    flip_mask = np.abs(axials) > ax_thresh

    # 3) flip mask 연속 구간
    regions = []
    in_flip  = False
    seg_start = None
    for f in range(F):
        if flip_mask[f] and not in_flip:
            in_flip   = True
            seg_start = f
        elif not flip_mask[f] and in_flip:
            in_flip = False
            regions.append((seg_start, f - 1))
    if in_flip:
        regions.append((seg_start, F - 1))

    results = []
    for rs, re in regions:
        # entry velocity jump: 마지막 jump 직전 rs 이전에 있는 것
        entry_candidates = [j for j in jump_idxs if j < rs and j >= rs - 15]
        entry_jump = entry_candidates[-1] if entry_candidates else (rs - 1)

        # exit velocity jump: rs 이후 첫 번째 큰 jump
        exit_candidates = [j for j in jump_idxs if j >= re and j <= re + 15]
        exit_jump = exit_candidates[0] if exit_candidates else re

        # anchor_before: entry_jump 이전 |axial|<ax_thresh 마지막 프레임
        ab = entry_jump
        while ab >= 0 and abs(axials[ab]) > ax_thresh:
            ab -= 1
        anchor_before = max(0, ab)

        # anchor_after: exit_jump+1 이후 |axial|<ax_thresh 첫 프레임
        aa_ = exit_jump + 1
        while aa_ < F and abs(axials[aa_]) > ax_thresh:
            aa_ += 1
        anchor_after = min(F - 1, aa_)

        results.append({
            "region_start":   rs,
            "region_end":     re,
            "entry_jump":     entry_jump,
            "exit_jump":      exit_jump,
            "anchor_before":  anchor_before,
            "anchor_after":   anchor_after,
            "n_frames":       re - rs + 1,
            "axial_min":      float(axials[rs:re+1].min()),
            "axial_max":      float(axials[rs:re+1].max()),
            "anchor_b_axial": float(axials[anchor_before]),
            "anchor_a_axial": float(axials[anchor_after]),
        })
    return results


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

print(f"[axfix] Loading {IN_PT} ...")
merged = torch.load(IN_PT, map_location="cpu")
params = merged["smpl_params_incam"]

body_pose_orig = params["body_pose"].numpy().copy()   # (F, 21, 3)
body_pose_fixed = body_pose_orig.copy()
F = body_pose_orig.shape[0]
print(f"[axfix] Total frames: {F}")

# wrist indices in body_pose (bp_idx = full_joint_id - 1)
WRIST_SPECS = [
    ("LEFT",  19),   # full joint 20
    ("RIGHT", 20),   # full joint 21
]

diag_lines = []
total_fixed = 0

for side, bp_idx in WRIST_SPECS:
    print(f"\n[axfix] ── {side} WRIST (body_pose[:,{bp_idx},:]) ──")

    axials = np.array([euler_xyz_axial(body_pose_orig[f, bp_idx]) for f in range(F)])
    regions = detect_flip_regions(axials)

    print(f"  Detected {len(regions)} flip region(s):")
    diag_lines.append(f"\n{side} WRIST — flip regions")
    diag_lines.append(f"{'rs':>6} {'re':>6} {'n':>4}  {'axial_range':>22}  "
                      f"{'ab':>6} {'aa':>6}  {'ab_ax':>8}  {'aa_ax':>8}  action")
    diag_lines.append("─"*85)

    for reg in regions:
        rs  = reg["region_start"]
        re  = reg["region_end"]
        ab  = reg["anchor_before"]
        aa_ = reg["anchor_after"]
        n   = reg["n_frames"]

        axial_str = f"[{reg['axial_min']:+.1f}..{reg['axial_max']:+.1f}]"
        ab_ax = reg["anchor_b_axial"]
        aa_ax = reg["anchor_a_axial"]

        print(f"  f{rs:04d}-f{re:04d} ({n} frames)  axial={axial_str}  "
              f"anchor: f{ab:04d}({ab_ax:+.1f}°) → f{aa_:04d}({aa_ax:+.1f}°)")

        # SLERP 적용: body_pose[rs..re] → interpolate between ab and aa_
        Ra = aa_to_rotmat(body_pose_orig[ab,  bp_idx])
        Rb = aa_to_rotmat(body_pose_orig[aa_, bp_idx])

        n_interp = aa_ - ab  # span from anchor_before to anchor_after
        if n_interp <= 0:
            print(f"    [WARN] anchor span <= 0, skipping")
            diag_lines.append(f"  {rs:6d} {re:6d} {n:4d}  {axial_str:>22}  "
                               f"{ab:6d} {aa_:6d}  {ab_ax:>+8.1f}  {aa_ax:>+8.1f}  SKIP(span=0)")
            continue

        frames_to_fix = list(range(rs, re + 1))
        for f in frames_to_fix:
            t = float(f - ab) / float(n_interp)
            t = float(np.clip(t, 0.0, 1.0))
            R_interp = slerp_rotmat(Ra, Rb, t)
            aa_new   = rotmat_to_aa(R_interp)
            body_pose_fixed[f, bp_idx] = aa_new.astype(np.float32)

        # 검증: 보정 후 axial 확인
        axials_after = np.array([euler_xyz_axial(body_pose_fixed[f, bp_idx])
                                  for f in range(rs, re + 1)])
        still_flip = np.sum(np.abs(axials_after) > AX_THRESH)
        total_fixed += len(frames_to_fix)
        status = f"OK(still_flip={still_flip})" if still_flip == 0 else f"WARN({still_flip} still flip)"

        print(f"    → fixed {len(frames_to_fix)} frames, axial after=[{axials_after.min():+.1f}..{axials_after.max():+.1f}]  {status}")
        diag_lines.append(f"  {rs:6d} {re:6d} {n:4d}  {axial_str:>22}  "
                           f"{ab:6d} {aa_:6d}  {ab_ax:>+8.1f}  {aa_ax:>+8.1f}  {status}")

# ─────────────────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────────────────
import copy
merged_fixed = copy.deepcopy(merged)
merged_fixed["smpl_params_incam"]["body_pose"] = torch.from_numpy(body_pose_fixed)
merged_fixed["cap_axial_fix_meta"] = {
    "vel_thresh": VEL_THRESH,
    "ax_thresh":  AX_THRESH,
    "total_fixed_frames": total_fixed,
}
torch.save(merged_fixed, OUT_PT)
print(f"\n[axfix] Saved → {OUT_PT}  (total fixed frames: {total_fixed})")

diag_path = DIAG_DIR / "flip_regions.txt"
with open(diag_path, "w") as fp:
    fp.write("\n".join(diag_lines))
print(f"[axfix] Diag → {diag_path}")
print("[axfix] Done.")
