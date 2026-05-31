"""
cap_wrist_refine.py

DeVI-style body+hand merge for the capstone_deploy (GVHMR + HaMeR) pipeline.

Two stages:
  (A) Wrist replacement:   R_w = G_e^{-1} G_w     (DeVI Appendix A.5)
  (B) 2D reprojection refine:  L = w_b L_b + w_h L_h + w_tc L_tc   (DeVI eq. 10, HOI term dropped)

Design goals for this skeleton:
  - Reuse GVHMR's own BodyModelSMPLX (differentiable) and K_fullimg. No new SMPL-X layer.
  - Operate entirely in the GVHMR *incam* coordinate system.
  - Keep the 2D hand target behind ONE function (`build_hand_targets_*`) so the
    "Option 1 (HaMeR reprojection)" -> "Option 2 (image keypoint detector)" switch
    is a single swap, not a loop rewrite.
  - Heavy on assertions + overlay debugging, because coordinate/index bugs here
    are silent (no crash, just wrong-looking hands).

This is a SKELETON: every spot that depends on the exact repo internals is marked
with `# TODO[verify]` and a cheap runtime check. Run top-to-bottom, fix asserts as
they fire. Nothing here should be trusted blindly until the overlay images look right.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


# =============================================================================
# 0. SMPL-X joint indexing  -- CONFIRM THESE FIRST, everything depends on them.
# =============================================================================
# GVHMR uses standard SMPL-X (use_pca=False, flat_hand_mean=False), so body_pose
# is (21, 3) axis-angle, NOT including the root (global_orient is separate).
#
# Standard SMPL-X *full* joint order (root = 0):
#   0 pelvis, 1 left_hip, 2 right_hip, 3 spine1, 4 left_knee, 5 right_knee,
#   6 spine2, 7 left_ankle, 8 right_ankle, 9 spine3, 10 left_foot, 11 right_foot,
#   12 neck, 13 left_collar, 14 right_collar, 15 head,
#   16 left_shoulder, 17 right_shoulder, 18 left_elbow, 19 right_elbow,
#   20 left_wrist, 21 right_wrist, ...
#
# body_pose in SMPL-X drops the root, so body_pose[i] corresponds to FULL joint (i+1).
# i.e. full_joint_id = body_pose_index + 1.
#
# Parent chain (full joint ids) pelvis -> ... -> wrist:
LEFT_WRIST_FULL  = 20
RIGHT_WRIST_FULL = 21
# Forward-kinematics parent chains, root-first, INCLUDING the wrist's parent (elbow)
# but the DeVI product G_e goes pelvis..elbow (parents of the wrist).
# pelvis(0) -> spine1(3) -> spine2(6) -> spine3(9) -> collar -> shoulder -> elbow
LEFT_CHAIN_TO_ELBOW  = [0, 3, 6, 9, 13, 16, 18]   # ...left_collar, left_shoulder, left_elbow
RIGHT_CHAIN_TO_ELBOW = [0, 3, 6, 9, 14, 17, 19]   # ...right_collar, right_shoulder, right_elbow
# TODO[verify]: confirm collar/shoulder/elbow ids against the SMPL-X kintree actually
# used by BodyModelSMPLX (print model.bm.parents). The skeleton derives the chain
# from parents below so this hardcoded list is only a fallback / sanity check.


def derive_parent_chain(parents: np.ndarray, joint_id: int) -> list[int]:
    """Walk the kinematic tree from `joint_id` up to root, return root-first chain
    EXCLUDING joint_id itself (i.e. its ancestors). parents[j] = parent of j, root parent = -1."""
    chain = []
    j = int(parents[joint_id])
    while j != -1:
        chain.append(j)
        j = int(parents[j])
    chain.reverse()  # root-first
    return chain


def verify_wrist_chains(smplx_model, smpl_params: dict | None = None) -> dict:
    """STEP 0 gate. Confirm the assumptions everything else depends on:
      - smplx_model.bm.parents is reachable
      - body_pose is (F, 21, 3)  (or (F, 63) which reshapes to that)
      - wrist parent chains derived from the kintree match LEFT/RIGHT_CHAIN_TO_ELBOW
        (printed warning if not — overwrite the hardcoded constants if the model is
        a SMPL-X variant with a different kintree)

    Returns a dict with the derived chains. Print this and eyeball before STEP 1.
    Raises AssertionError on hard mismatches so failures are loud, not silent.
    """
    assert hasattr(smplx_model, "bm"), \
        "smplx_model has no .bm — wrong SMPL-X variant. Expected BodyModelSMPLX " \
        "from make_smplx('supermotion_fullhands')."
    assert hasattr(smplx_model.bm, "parents"), \
        "smplx_model.bm has no .parents — kintree unavailable, cannot derive chain."
    parents = np.asarray(
        smplx_model.bm.parents.cpu() if hasattr(smplx_model.bm.parents, "cpu")
        else smplx_model.bm.parents
    )
    left_chain = derive_parent_chain(parents, LEFT_WRIST_FULL)
    right_chain = derive_parent_chain(parents, RIGHT_WRIST_FULL)
    print(f"[verify] parents (first 22) = {parents[:22].tolist()}")
    print(f"[verify] LEFT  wrist chain (pelvis..elbow): {left_chain}")
    print(f"[verify] RIGHT wrist chain (pelvis..elbow): {right_chain}")
    if left_chain != LEFT_CHAIN_TO_ELBOW:
        print(f"[verify] WARNING: left chain differs from hardcoded "
              f"{LEFT_CHAIN_TO_ELBOW}. Using derived chain — confirm it's correct "
              f"for this SMPL-X variant.")
    if right_chain != RIGHT_CHAIN_TO_ELBOW:
        print(f"[verify] WARNING: right chain differs from hardcoded "
              f"{RIGHT_CHAIN_TO_ELBOW}.")
    if smpl_params is not None:
        bp = smpl_params["body_pose"]
        if bp.dim() == 2:
            assert bp.shape[1] == 63, \
                f"body_pose 2D shape is {tuple(bp.shape)} — expected (F, 63) for 21 joints."
        elif bp.dim() == 3:
            assert bp.shape[1] == 21 and bp.shape[2] == 3, \
                f"body_pose 3D shape is {tuple(bp.shape)} — expected (F, 21, 3)."
        else:
            raise AssertionError(f"body_pose has unexpected ndim {bp.dim()}")
        print(f"[verify] body_pose shape: {tuple(bp.shape)} OK")
    return {"parents": parents, "left_chain": left_chain, "right_chain": right_chain}


# =============================================================================
# 1. rotation helpers (axis-angle <-> rotmat), batched, torch
# =============================================================================
def aa_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 3, 3). Rodrigues, differentiable."""
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / theta
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = torch.cos(theta)[..., 0]
    s = torch.sin(theta)[..., 0]
    C = 1 - c
    R = torch.stack([
        c + x * x * C,      x * y * C - z * s,  x * z * C + y * s,
        y * x * C + z * s,  c + y * y * C,      y * z * C - x * s,
        z * x * C - y * s,  z * y * C + x * s,  c + z * z * C,
    ], dim=-1).reshape(aa.shape[:-1] + (3, 3))
    return R


def rotmat_to_aa(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 3) axis-angle. Uses cv2 per-sample for numerical safety
    in the offline merge step (not in the optimization loop)."""
    import cv2
    shp = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3).detach().cpu().numpy().astype(np.float64)
    out = np.zeros((Rf.shape[0], 3), dtype=np.float32)
    for i in range(Rf.shape[0]):
        aa, _ = cv2.Rodrigues(Rf[i])
        out[i] = aa.reshape(3)
    return torch.from_numpy(out).reshape(shp + (3,))


def geodesic_distance(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Mean geodesic distance between two sets of rotation matrices (..., 3, 3)."""
    Rrel = torch.matmul(R1.transpose(-1, -2), R2)
    trace = Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]
    cos = ((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos).mean()


# =============================================================================
# 2. HaMeR detection loading  (extends the repo's existing loader)
# =============================================================================
def hamer_mat_to_rotmat(arr: np.ndarray) -> np.ndarray:
    """Normalize HaMeR global_orient to a (3,3) rotation matrix."""
    a = np.asarray(arr).squeeze()
    if a.shape == (3, 3):
        return a.astype(np.float32)
    if a.shape == (3,):  # axis-angle fallback
        import cv2
        R, _ = cv2.Rodrigues(a.astype(np.float64))
        return R.astype(np.float32)
    raise ValueError(f"Unexpected HaMeR global_orient shape: {a.shape}")


def mirror_rotmat_right_to_left(R: np.ndarray) -> np.ndarray:
    """HaMeR processes the left hand as a flipped right hand.
    The existing code mirrors hand_pose by negating y,z of each axis-angle joint.
    For a rotation MATRIX, the equivalent x-axis mirror is  M R M  with M=diag(-1,1,1).
    # TODO[verify]: confirm this matches the hand_pose mirror convention by checking
    # that a left-hand overlay lands on the left hand (not mirrored across the body).
    """
    M = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
    return (M @ R @ M).astype(np.float32)


def load_hamer_detections_full(hamer_out_dir: Path, num_frames: int) -> dict:
    """Like the repo's load_hamer_hand_detections, but ALSO returns:
      - global_orient rotmat (wrist global rotation, in HaMeR crop-cam frame)
      - hand_pose axis-angle (45,)  [already handled by repo, kept for completeness]
      - camera transform inputs (cam_t_full, box_center, box_size, img_size, focal)
        so a later step can lift HaMeR 3D joints/wrist into the GVHMR incam frame.
    Returns dict: detections[side][frame_idx] = {wrist_R, hand_pose, cam_t_full, ...}
    """
    import re
    det = {"left": {}, "right": {}}
    pattern = re.compile(r"^(\d+)_")
    for npz_path in sorted(hamer_out_dir.glob("*.npz")):
        m = pattern.match(npz_path.stem)
        if m is None:
            continue
        fidx = int(m.group(1))
        if fidx >= num_frames:
            continue
        try:
            data = np.load(npz_path, allow_pickle=True)
            is_right = int(np.asarray(data["is_right"]).reshape(-1)[0])
            side = "right" if is_right == 1 else "left"
            mano = data["mano_params"].item()

            wrist_R = hamer_mat_to_rotmat(mano["global_orient"])
            hand_pose = _hand_pose_to_aa(mano["hand_pose"])  # (45,)

            if side == "left":
                wrist_R = mirror_rotmat_right_to_left(wrist_R)
                hand_pose = _mirror_hand_pose_aa(hand_pose)

            det[side][fidx] = {
                "wrist_R": wrist_R,                         # (3,3) HaMeR camera frame (= GVHMR incam frame; R_cam_align=I)
                "hand_pose": hand_pose,                     # (45,)
                "cam_t_full": np.asarray(data["cam_t_full"], np.float32),
                "box_center": np.asarray(data["box_center"], np.float32),
                "box_size":   np.asarray(data["box_size"], np.float32),
                "img_size":   np.asarray(data["img_size"], np.float32),
                "focal":      float(np.asarray(data["focal_length"]).reshape(-1)[0]),
                "keypoints_3d": (
                    np.asarray(data["keypoints_3d"], np.float32).reshape(21, 3)
                    if "keypoints_3d" in data else None
                ),
                "keypoints_2d": (
                    np.asarray(data["keypoints_2d"], np.float32).reshape(21, 2)
                    if "keypoints_2d" in data else None
                ),
            }
        except Exception as exc:  # noqa
            print(f"[wrist_refine] skip {npz_path.name}: {exc}")
    return det


def _hand_pose_to_aa(hand_pose: Any) -> np.ndarray:
    """Mirror of repo's hamer_hand_pose_to_axis_angle, duplicated to keep this module
    standalone. Returns (45,)."""
    import cv2
    a = np.asarray(hand_pose).squeeze()
    if a.shape == (45,):
        return a.astype(np.float32)
    if a.shape == (15, 3):
        return a.reshape(45).astype(np.float32)
    if a.shape == (15, 3, 3):
        out = []
        for jm in a:
            aa, _ = cv2.Rodrigues(jm.astype(np.float64))
            out.append(aa.reshape(3))
        return np.asarray(out, np.float32).reshape(45)
    if a.size == 45:
        return a.reshape(45).astype(np.float32)
    raise ValueError(f"bad hand_pose shape {a.shape}")


def _mirror_hand_pose_aa(pose: np.ndarray) -> np.ndarray:
    m = np.asarray(pose, np.float32).reshape(15, 3).copy()
    m[:, 1:] *= -1.0
    return m.reshape(45)


# =============================================================================
# 3. STAGE A -- wrist replacement   R_w = G_e^{-1} G_w
# =============================================================================

WRIST_OUTLIER_DEG_RIGHT: float = 60.0   # right wrist threshold
WRIST_OUTLIER_DEG_LEFT:  float = 85.0   # left wrist threshold (higher: HaMeR left-hand accuracy lower)


def _slerp_fill_invalid(
    valid: dict[int, np.ndarray],   # {fidx: R_w (3,3)} for in-threshold frames
    invalid_fidxs: list[int],       # frames that exceeded the threshold
) -> dict[int, np.ndarray]:
    """SLERP-interpolate R_w for invalid frames from surrounding valid ones.

    For each invalid frame, find the nearest valid frame before and after it and
    lerp between them in rotation space.  Edge cases (no valid before or after)
    use the nearest valid endpoint (constant extrapolation, not rotation beyond).
    Returns {fidx: R_w (3,3)} for every invalid frame.
    """
    from scipy.spatial.transform import Rotation, Slerp  # type: ignore

    if not valid or not invalid_fidxs:
        return {}

    sorted_valid = sorted(valid.keys())
    times = np.array(sorted_valid, dtype=float)
    rots = Rotation.from_matrix(np.stack([valid[f] for f in sorted_valid]))

    # Build interpolator over valid keyframes
    if len(sorted_valid) == 1:
        single = rots.as_matrix()[0].astype(np.float32)
        return {f: single for f in invalid_fidxs}

    slerp_fn = Slerp(times, rots)

    result: dict[int, np.ndarray] = {}
    t_min, t_max = times[0], times[-1]
    for fidx in invalid_fidxs:
        t = float(np.clip(fidx, t_min, t_max))
        result[fidx] = slerp_fn(t).as_matrix().astype(np.float32)
    return result


def replace_wrists(
    smpl_params: dict,          # one of pred["smpl_params_incam"], in/out
    detections: dict,           # from load_hamer_detections_full
    parents: np.ndarray,        # model.bm.parents  (full SMPL-X kintree)
    num_frames: int,
    dtype=torch.float32,
    outlier_deg: float | None = None,           # overrides both sides if set
    outlier_deg_right: float = WRIST_OUTLIER_DEG_RIGHT,
    outlier_deg_left:  float = WRIST_OUTLIER_DEG_LEFT,
    use_slerp: bool = False,    # False: apply all HaMeR R_w directly (no mask/SLERP)
) -> dict:
    """Overwrite the wrist entries of body_pose with HaMeR-derived local wrist rotation.

    body_pose layout: (F, 21, 3) axis-angle, where body_pose[:, i] = full joint (i+1).
    For each frame & side:
        G_e = product over parent chain (pelvis..elbow) of R_parent
        G_w = HaMeR wrist global rotation (camera frame, same as G_e — R_cam_align = I)
        R_w = G_e^{-1} G_w

    use_slerp=False (default): all detected frames get R_w written directly; no
        angle-threshold masking, no SLERP interpolation for outliers.
    use_slerp=True: outlier frames (angle > threshold) are masked and SLERP-filled
        from nearest valid neighbours.
    """
    body_pose = smpl_params["body_pose"]
    if body_pose.dim() == 2:
        F = body_pose.shape[0]
        bp = body_pose.reshape(F, 21, 3).clone()
    else:
        F = body_pose.shape[0]
        bp = body_pose.clone()
    assert bp.shape[1] == 21, f"expected 21 body joints, got {bp.shape[1]}"

    global_orient = smpl_params["global_orient"].reshape(F, 3)

    # per-side threshold (only used when use_slerp=True)
    thr_right = outlier_deg if outlier_deg is not None else outlier_deg_right
    thr_left  = outlier_deg if outlier_deg is not None else outlier_deg_left

    sides = [
        ("left",  LEFT_WRIST_FULL,  derive_parent_chain(parents, LEFT_WRIST_FULL),  thr_left),
        ("right", RIGHT_WRIST_FULL, derive_parent_chain(parents, RIGHT_WRIST_FULL), thr_right),
    ]

    for side, wrist_full, chain, thr in sides:
        wrist_body_idx = wrist_full - 1
        print(f"[wrist_refine] {side}: wrist_full={wrist_full} "
              f"body_idx={wrist_body_idx} parent_chain={chain} "
              f"use_slerp={use_slerp}" + (f" threshold={thr}°" if use_slerp else ""))

        valid_rw:   dict[int, np.ndarray] = {}
        invalid_fidxs: list[int] = []

        for fidx, d in detections[side].items():
            if fidx >= F:
                continue
            G = aa_to_rotmat(global_orient[fidx])
            for jid in chain:
                if jid == 0:
                    continue
                G = G @ aa_to_rotmat(bp[fidx, jid - 1])
            G_e = G  # (3,3) global elbow rotation

            G_w = torch.from_numpy(d["wrist_R"]).to(dtype)
            R_w = G_e.transpose(-1, -2) @ G_w

            if use_slerp:
                # Compute local wrist angle to classify valid vs outlier
                trace = (R_w[0, 0] + R_w[1, 1] + R_w[2, 2]).clamp(-1 + 1e-6, 3 - 1e-6)
                angle_deg = float(torch.acos((trace - 1.0) / 2.0) * (180.0 / 3.141592653589793))
                if angle_deg <= thr:
                    valid_rw[fidx] = R_w.detach().cpu().numpy().astype(np.float32)
                else:
                    invalid_fidxs.append(fidx)
            else:
                # No filter: write all HaMeR detections directly
                valid_rw[fidx] = R_w.detach().cpu().numpy().astype(np.float32)

        if use_slerp:
            slerp_rw = _slerp_fill_invalid(valid_rw, invalid_fidxs)
            n_invalid = len(invalid_fidxs)
            print(f"[wrist_refine] {side}: {len(valid_rw)} valid, "
                  f"{n_invalid} outlier → {len(slerp_rw)} SLERP-filled "
                  f"(threshold={thr}°)")
            if invalid_fidxs:
                print(f"[wrist_refine] {side}: outlier frames: "
                      f"{sorted(invalid_fidxs)[:20]}"
                      f"{'...' if n_invalid > 20 else ''}")
            all_rw = {**valid_rw, **slerp_rw}
        else:
            print(f"[wrist_refine] {side}: {len(valid_rw)} frames written directly "
                  f"(no mask/SLERP)")
            all_rw = valid_rw

        for fidx, R_w_np in all_rw.items():
            R_w_t = torch.from_numpy(R_w_np).to(dtype)
            bp[fidx, wrist_body_idx] = rotmat_to_aa(R_w_t).to(dtype)

    smpl_params["body_pose"] = bp.reshape(F, 21, 3)
    return smpl_params


def sync_wrist_rows(src_params: dict, dst_params: dict) -> dict:
    """Stage A sync: copy ONLY the wrist body_pose rows from src to dst.

    body_pose stores LOCAL (parent-relative) joint rotations, so the same wrist row
    applies verbatim to both incam and global SMPL-X representations of the same
    motion — only `global_orient`/`transl` differ between them.

    Use this AFTER Stage A (wrist replacement) only. After Stage B, use
    `sync_refined_local_pose` instead (which covers all optimized rows + hands).
    """
    src_bp = src_params["body_pose"].reshape(-1, 21, 3)
    dst_bp = dst_params["body_pose"].reshape(src_bp.shape[0], 21, 3).clone()
    for full in (LEFT_WRIST_FULL, RIGHT_WRIST_FULL):
        body_idx = full - 1
        dst_bp[:, body_idx] = src_bp[:, body_idx]
    dst_params["body_pose"] = dst_bp.reshape(src_bp.shape[0], 21, 3)
    return dst_params


def sync_refined_local_pose(src_params: dict, dst_params: dict) -> dict:
    """Stage B sync: copy ALL body_pose rows that were optimized + both hand poses.

    Stage B (`refine_2d`) optimizes UPPER_BODY_OPT_IDS rows of body_pose AND both
    `left_hand_pose` / `right_hand_pose`. So after Stage B, syncing only the wrist
    rows would leave shoulders / elbows / spine / hands stale in `smpl_params_global`,
    which downstream readers of the global representation would see as inconsistent
    with the incam result. This copies the full optimized scope.

    All copied quantities are LOCAL rotations, so values transfer verbatim between
    incam and global — same justification as `sync_wrist_rows`.
    """
    src_bp = src_params["body_pose"].reshape(-1, 21, 3)
    F = src_bp.shape[0]
    dst_bp = dst_params["body_pose"].reshape(F, 21, 3).clone()
    for idx in UPPER_BODY_OPT_IDS:
        dst_bp[:, idx] = src_bp[:, idx]
    dst_params["body_pose"] = dst_bp.reshape(F, 21, 3)
    if "left_hand_pose" in src_params:
        dst_params["left_hand_pose"] = src_params["left_hand_pose"].clone()
    if "right_hand_pose" in src_params:
        dst_params["right_hand_pose"] = src_params["right_hand_pose"].clone()
    return dst_params


# =============================================================================
# 4. HAND 2D TARGETS  -- the ONE swap point for Option 1 vs Option 2
# =============================================================================
# Both functions must return: targets[side][frame_idx] = {
#     "uv":   (K, 2) float  target pixel coords,
#     "conf": (K,)  float    per-keypoint confidence/weight,
#     "smplx_hand_joint_ids": (K,) int  which SMPL-X joints these correspond to,
# }
# The optimization loop is agnostic to which one produced them.

def build_hand_targets_option1_hamer(
    detections: dict,
    K_fullimg: torch.Tensor,
) -> dict:
    """OPTION 1: reproject HaMeR's own 3D hand joints into the GVHMR incam image plane.

    Lift: joints_cam = keypoints_3d + cam_t_full  (MANO space → HaMeR camera frame)
    Project: uv = K_fullimg @ joints_cam / z

    SMPL-X joint ID mapping (standard smplx output with use_pca=False, ~55 joints):
      Body joint 20 = left wrist,  body joint 21 = right wrist
      Left hand finger kinematic joints:  25-39  (15 joints, no wrist)
      Right hand finger kinematic joints: 40-54  (15 joints, no wrist)
    HaMeR MANO 21 joints:  0=wrist, 1-15=finger kinematic, 16-20=fingertips (skipped)
    We use 16 joints per side (wrist + 15 kinematic fingers).

    # TODO[verify] STEP 3: run debug_target_overlay and confirm dots land on the hand.
    # If SMPL-X outputs fingertips at 55-59 (left) / 60-64 (right), extend to all 21.
    # If the index range is wrong (e.g. out.joints.shape[1] < 55), you will get an
    # IndexError in refine_2d — check out.joints.shape[1] at STEP 0 and adjust.
    """
    N = 16  # wrist (1) + 15 kinematic finger joints per side
    LEFT_SMPLX_IDS = np.array([20] + list(range(25, 40)), dtype=np.int64)   # 16 ids
    RIGHT_SMPLX_IDS = np.array([21] + list(range(40, 55)), dtype=np.int64)  # 16 ids
    # HaMeR joint order: 0=wrist, 1-15=kinematic fingers (16-20 fingertips skipped)
    HAMER_JOINTS = list(range(N))

    side_ids = {"left": LEFT_SMPLX_IDS, "right": RIGHT_SMPLX_IDS}

    K_np = (K_fullimg.detach().cpu().numpy()
            if hasattr(K_fullimg, "detach") else np.asarray(K_fullimg, dtype=np.float32))

    targets: dict = {"left": {}, "right": {}}
    for side, perframe in detections.items():
        smplx_ids = side_ids[side]
        missing_kp3d = 0
        for fidx, det in perframe.items():
            kp3d = det.get("keypoints_3d")  # (21, 3) MANO space, or None
            if kp3d is None:
                missing_kp3d += 1
                continue
            cam_t = det["cam_t_full"].reshape(3)
            # Skip detections where Z <= 0 (hand behind camera — bad detection)
            if cam_t[2] <= 0:
                print(f"[wrist_refine] {side} frame {fidx}: cam_t_full Z={cam_t[2]:.2f} <= 0, skip")
                continue
            # Lift MANO joints to HaMeR camera frame, then select the 16 we use.
            # Left hand: kp3d is in right-hand canonical space → x-flip before projection.
            # render_hamer_incam.py confirms: verts[:, 0] *= -1 for left hand.
            # cam_t_full.x is already side-corrected (entry_point_hamer.py line 476).
            kp3d_proj = kp3d[HAMER_JOINTS].copy()
            if side == "left":
                kp3d_proj[:, 0] *= -1
            joints_cam = kp3d_proj + cam_t  # (16, 3)
            # Project with HaMeR's own intrinsics (cam_t_full was computed with these).
            # K_fullimg uses GVHMR focal length (~2200) but cam_t_full assumes HaMeR
            # focal length (~37500) — mixing them shifts the target by ~17x in x/y.
            fl = float(det["focal"])           # stored as "focal" by load_hamer_detections_full
            img_sz = np.asarray(det["img_size"]).reshape(-1)  # (W, H)
            cx = float(img_sz[0]) / 2.0
            cy = float(img_sz[1]) / 2.0
            z = np.maximum(joints_cam[:, 2], 1e-4)
            u = fl * joints_cam[:, 0] / z + cx
            v = fl * joints_cam[:, 1] / z + cy
            uv = np.stack([u, v], axis=-1).astype(np.float32)  # (16, 2)
            targets[side][fidx] = {
                "uv": uv,
                "conf": np.ones(N, dtype=np.float32),
                "smplx_hand_joint_ids": smplx_ids,
            }
        if missing_kp3d:
            print(f"[wrist_refine] {side}: {missing_kp3d} frames had no keypoints_3d "
                  f"(npz may be old — re-run HaMeR to regenerate with keypoints_3d)")
    return targets


def build_hand_targets_option2_detector(
    detector_keypoints: dict,   # frame_idx -> {"left": (21,3), "right": (21,3)} in IMAGE px
    mp_to_smplx_hand: dict,     # mapping detector kp index -> smplx hand joint id
) -> dict:
    """OPTION 2 (image observation): real 2D hand keypoints detected in the rendered
    frames (MediaPipe Hands or ViTPose-wholebody). No reprojection, no circularity.

    # TODO[implement]: run detector in preprocessing, fill detector_keypoints,
    # build the index mapping once.
    """
    raise NotImplementedError("Option 2 target builder -- add detector then implement")


# =============================================================================
# 5. STAGE B -- 2D reprojection refinement
# =============================================================================
def project_incam(joints3d: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Pinhole projection. joints3d (..., J, 3) in incam frame, K (3,3) or (F,3,3)
    -> (..., J, 2) px.

    K=(3,3): single intrinsics for all frames.
    K=(F,3,3): per-frame intrinsics. Caller guarantees joints3d's leading dim is F.
    """
    x = joints3d[..., 0]
    y = joints3d[..., 1]
    z = joints3d[..., 2].clamp(min=1e-4)
    if K.dim() == 2:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
    elif K.dim() == 3:
        # (F,3,3): broadcast against (F, J) by adding a joint-axis
        fx = K[:, 0, 0].unsqueeze(-1)
        fy = K[:, 1, 1].unsqueeze(-1)
        cx = K[:, 0, 2].unsqueeze(-1)
        cy = K[:, 1, 2].unsqueeze(-1)
    else:
        raise ValueError(f"project_incam: unexpected K shape {tuple(K.shape)}")
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return torch.stack([u, v], dim=-1)


# which body_pose joints are allowed to move (DeVI A.6: hands, wrists, elbows,
# shoulders, spine). body_pose indices = full_id - 1.
UPPER_BODY_OPT_IDS = sorted({
    (3 - 1), (6 - 1), (9 - 1),          # spine1,2,3
    (16 - 1), (17 - 1),                 # shoulders
    (18 - 1), (19 - 1),                 # elbows
    # wrists excluded: Stage A HaMeR wrist_R is preserved through B1
})


def refine_2d(
    smplx_model,                 # BodyModelSMPLX (from make_smplx("supermotion_fullhands"))
    smpl_params: dict,           # incam params AFTER wrist replacement (tensors, F-leading)
    K_fullimg: torch.Tensor,     # (F,3,3) or (3,3)
    hand_targets: dict,          # from one of build_hand_targets_*
    body_targets: dict | None,   # optional; None means body frozen / no L_b
    *,
    w_b: float = 1.0,
    w_h: float = 1.0,
    w_tc: float = 1.0e4,
    lr: float = 2e-2,
    iters: int = 300,            # MAX iterations
    device: str = "cuda",
    debug_every: int = 50,
    # --- early stopping (stop on hand RMS pixel error, NOT total loss) ---
    min_iter: int = 60,
    patience: int = 30,
    target_px: float = 12.0,
    min_delta: float = 0.1,
    restore_best: bool = True,
    # --- B1/B2 phase control ---
    freeze_hand_pose: bool = False,  # B1: arm-chain only, hand_pose fixed
    freeze_body_pose: bool = False,  # B2: hand-only, body_pose fixed
    w_hand_prior: float = 0.0,       # B2: ||hand_pose - hand_pose_init||² regularization
    hand_pose_init: dict | None = None,  # B2 prior reference; None = use smpl_params values
) -> dict:
    """Optimize upper-body + hand pose params to minimize DeVI eq.10 (no HOI term).

    Early stopping is driven by `hand_rms_px` (RMS pixel error of the hand
    reprojection), NOT by total loss: with w_tc=1e4 the total loss is dominated by
    the temporal term, so a falling total loss does not imply better hand alignment.
    We track the best `hand_rms_px` state and (by default) restore it at the end.

    CAVEAT: under Option-1 targets, `hand_rms_px` is partly self-referential (the
    target is itself HaMeR reprojection), so treat it as a stopping convenience, not
    a quality claim. Under Option-2 (detector) targets it is a meaningful number.
    `target_px=12.0` is resolution/hand-size dependent — consider normalizing by
    hand-box size across clips.

    Returns (refined_params, refine_meta):
      - refined_params: same dict layout as input, PURE TENSORS only (safe to feed to
        downstream render/penetration that assumes every value is a tensor).
      - refine_meta: dict with best_hand_rms_px, final_hand_rms_px, stop_reason,
        iters_run, etc. Caller stores this OUTSIDE smpl_params_incam.
    """
    F = smpl_params["body_pose"].shape[0]
    K = K_fullimg.to(device)
    if K.dim() == 2:
        K = K.unsqueeze(0).repeat(F, 1, 1)

    # --- set up leaf tensors ---
    body_pose = smpl_params["body_pose"].reshape(F, 21, 3).clone().to(device)
    lhand = smpl_params["left_hand_pose"].reshape(F, 45).clone().to(device)
    rhand = smpl_params["right_hand_pose"].reshape(F, 45).clone().to(device)
    global_orient = smpl_params["global_orient"].reshape(F, 3).clone().to(device)
    transl = smpl_params["transl"].reshape(F, 3).clone().to(device)
    betas = smpl_params["betas"].to(device) if "betas" in smpl_params else None

    # B1 mode: arm-chain only (hand_pose frozen)
    # B2 mode: hand-only (body_pose frozen)
    body_free = body_pose.clone()
    opt_params = []
    if not freeze_body_pose:
        body_free.requires_grad_(True)
        opt_params.append(body_free)
    if not freeze_hand_pose:
        lhand.requires_grad_(True)
        rhand.requires_grad_(True)
        opt_params.append(lhand)
        opt_params.append(rhand)
    if not opt_params:
        raise ValueError("refine_2d: both freeze_body_pose and freeze_hand_pose are True — nothing to optimize")
    opt = torch.optim.Adam(opt_params, lr=lr)

    # mask: body_free only gets gradient on UPPER_BODY_OPT_IDS rows
    body_mask = torch.zeros(21, 1, device=device)
    for i in UPPER_BODY_OPT_IDS:
        body_mask[i] = 1.0

    # B2 hand prior reference
    if w_hand_prior > 0 and not freeze_hand_pose:
        if hand_pose_init is not None:
            lhand_prior = hand_pose_init["left"].reshape(F, 45).to(device).detach()
            rhand_prior = hand_pose_init["right"].reshape(F, 45).to(device).detach()
        else:
            lhand_prior = lhand.detach().clone()
            rhand_prior = rhand.detach().clone()

    # precompute target tensors per side  -> dict[side] = (frames, kp_ids, uv, conf)
    def pack_targets(tgt):
        packed = {}
        for side, perframe in tgt.items():
            fs, ids, uvs, cfs = [], None, [], []
            for fidx, t in perframe.items():
                cur_ids = list(np.asarray(t["smplx_hand_joint_ids"]).reshape(-1))
                if ids is None:
                    ids = cur_ids
                else:
                    # uv/conf are stacked across frames against a SINGLE id list, so
                    # the joint id set MUST be identical every frame or indices desync.
                    assert cur_ids == ids, (
                        f"pack_targets[{side}]: smplx_hand_joint_ids differ across "
                        f"frames (frame {fidx}: {cur_ids} vs {ids}). The target builder "
                        f"must emit the same joint id set every frame."
                    )
                fs.append(fidx); uvs.append(t["uv"]); cfs.append(t["conf"])
            if not fs:
                continue
            packed[side] = {
                "fidx": torch.tensor(fs, device=device, dtype=torch.long),
                "ids":  torch.tensor(ids, device=device, dtype=torch.long),
                "uv":   torch.tensor(np.stack(uvs), device=device, dtype=torch.float32),
                "conf": torch.tensor(np.stack(cfs), device=device, dtype=torch.float32),
            }
        return packed

    hand_t = pack_targets(hand_targets)
    body_t = pack_targets(body_targets) if body_targets else {}

    if not hand_t:
        raise ValueError(
            "refine_2d: no valid hand targets. The hand target builder returned nothing "
            "for either side. Without hand targets, hand_rms_px is undefined and early "
            "stopping/refinement are meaningless. Check the Option-1/Option-2 builder."
        )

    # --- early-stopping state (tracked on hand_rms_px) ---
    best_rms = float("inf")
    best_state = None
    best_it = -1
    bad_count = 0
    stop_reason = "max_iter"
    hand_rms_px = float("nan")  # defined even if loop body somehow doesn't run

    def _hand_rms_px(uv_detached) -> float:
        """RMS pixel error of hand reprojection vs targets (conf-weighted, detached)."""
        sq_sum = 0.0
        w_sum = 0.0
        for side, t in hand_t.items():
            pred_uv = uv_detached[t["fidx"]][:, t["ids"], :]      # (n,K,2)
            d2 = ((pred_uv - t["uv"]) ** 2).sum(-1)               # (n,K) squared px dist
            w = t["conf"]                                         # (n,K) or broadcastable
            sq_sum += float((d2 * w).sum().item())
            w_sum += float(w.sum().item())
        if w_sum <= 0:
            return float("nan")
        return float(np.sqrt(sq_sum / w_sum))

    for it in range(iters):
        opt.zero_grad()

        # re-assemble body_pose: frozen rows from original, free rows from leaf
        bp = body_pose * (1 - body_mask) + body_free * body_mask

        out = smplx_model.forward(
            betas=betas.expand(F, -1) if betas is not None and betas.dim() == 1 else betas,
            global_orient=global_orient,
            body_pose=bp.reshape(F, -1),
            left_hand_pose=lhand,
            right_hand_pose=rhand,
            transl=transl,
        )
        joints3d = out.joints  # (F, J, 3) incam   # TODO[verify] joints are in incam frame
        uv = project_incam(joints3d, K)            # (F, J, 2)

        loss = joints3d.new_zeros(())

        # L_h : hand projection
        L_h = joints3d.new_zeros(())
        for side, t in hand_t.items():
            pred_uv = uv[t["fidx"]][:, t["ids"], :]      # (n, K, 2)
            diff = (pred_uv - t["uv"]) * t["conf"].unsqueeze(-1)
            L_h = L_h + (diff ** 2).sum(-1).mean()
        loss = loss + w_h * L_h

        # L_b : body projection (optional anchor)
        L_b = joints3d.new_zeros(())
        for side, t in body_t.items():
            pred_uv = uv[t["fidx"]][:, t["ids"], :]
            diff = (pred_uv - t["uv"]) * t["conf"].unsqueeze(-1)
            L_b = L_b + (diff ** 2).sum(-1).mean()
        loss = loss + w_b * L_b

        # L_hand_prior : regularize hand_pose toward prior (B2 mode)
        if w_hand_prior > 0 and not freeze_hand_pose:
            L_hp = (((lhand - lhand_prior) ** 2).mean()
                    + ((rhand - rhand_prior) ** 2).mean())
            loss = loss + w_hand_prior * L_hp

        # L_tc : temporal consistency (geodesic on body + hands), adjacent frames
        if F > 1:
            R_body = aa_to_rotmat(bp)                 # (F,21,3,3)
            R_lh = aa_to_rotmat(lhand.reshape(F, 15, 3))
            R_rh = aa_to_rotmat(rhand.reshape(F, 15, 3))
            L_tc = (geodesic_distance(R_body[:-1], R_body[1:])
                    + geodesic_distance(R_lh[:-1], R_lh[1:])
                    + geodesic_distance(R_rh[:-1], R_rh[1:]))
            loss = loss + w_tc * L_tc
        else:
            L_tc = joints3d.new_zeros(())

        # --- early stopping on hand RMS pixel error (NOT total loss) ---
        # IMPORTANT: measure, capture best_state, and decide to stop BEFORE opt.step(),
        # so best_hand_rms_px and best_state refer to the SAME parameter state (the
        # one that produced `uv` this iteration). Measuring after step() would record
        # a metric and a state that are one update apart.
        with torch.no_grad():
            hand_rms_px = _hand_rms_px(uv.detach())

        improved = hand_rms_px < (best_rms - min_delta)
        if improved:
            best_rms = hand_rms_px
            best_it = it
            bad_count = 0
            if restore_best:
                st = {"body_free": body_free.detach().clone()}
                if not freeze_hand_pose:
                    st["lhand"] = lhand.detach().clone()
                    st["rhand"] = rhand.detach().clone()
                best_state = st
        else:
            bad_count += 1

        if it % debug_every == 0 or it == iters - 1:
            print(f"[refine] it={it:4d} loss={loss.item():.3f} "
                  f"L_h={L_h.item():.3f} L_b={L_b.item():.3f} L_tc={float(L_tc):.5f} "
                  f"hand_rms_px={hand_rms_px:.2f} best={best_rms:.2f} bad={bad_count}")

        # stop conditions (only after min_iter) -- decided on THIS state, before update
        if it >= min_iter:
            if hand_rms_px <= target_px:
                stop_reason = "target_px"
                print(f"[refine] early stop @ it={it}: hand_rms_px={hand_rms_px:.2f} "
                      f"<= target_px={target_px}")
                break
            if bad_count >= patience:
                stop_reason = "plateau"
                print(f"[refine] early stop @ it={it}: no >= {min_delta}px improvement "
                      f"for {patience} iters (best={best_rms:.2f} @ it={best_it})")
                break

        # not stopping -> take the optimization step
        loss.backward()
        # zero grads on frozen body rows (belt-and-suspenders with body_mask)
        if not freeze_body_pose:
            with torch.no_grad():
                if body_free.grad is not None:
                    body_free.grad *= body_mask
        opt.step()

    # restore best-hand_rms_px state (not the last iteration) if requested
    final_hand_rms_px = hand_rms_px  # last measured value (pre-restore)
    if restore_best and best_state is not None:
        with torch.no_grad():
            body_free.copy_(best_state["body_free"])
            if not freeze_hand_pose:
                lhand.copy_(best_state["lhand"])
                rhand.copy_(best_state["rhand"])
        print(f"[refine] restored best state @ it={best_it} (hand_rms_px={best_rms:.2f})")

    bp_final = (body_pose * (1 - body_mask) + body_free * body_mask).detach()
    refined = dict(smpl_params)
    refined["body_pose"] = bp_final.reshape(F, 21, 3).cpu()
    refined["left_hand_pose"] = lhand.detach().reshape(F, 45).cpu()
    refined["right_hand_pose"] = rhand.detach().reshape(F, 45).cpu()
    # NOTE: refine_meta is returned SEPARATELY, not stored inside `refined`, because
    # downstream render/penetration code may assume every value in smpl_params_incam
    # is a tensor. Keep this dict pure-tensor.
    refine_meta = {
        "best_hand_rms_px": best_rms,
        "final_hand_rms_px": final_hand_rms_px,
        "best_it": best_it,
        "stop_reason": stop_reason,
        "iters_run": it + 1,
        "restore_best": bool(restore_best and best_state is not None),
        "target_px": target_px,
        "w_b": w_b, "w_h": w_h, "w_tc": w_tc,
    }
    print(f"[refine] done: stop_reason={stop_reason} iters_run={it + 1} "
          f"best_hand_rms_px={best_rms:.2f} final_hand_rms_px={final_hand_rms_px:.2f}")
    return refined, refine_meta


# =============================================================================
# 6. DEBUG: overlay projected joints on frames  -- USE THIS CONSTANTLY
# =============================================================================
def debug_overlay(
    smplx_model, smpl_params, K_fullimg, frame_dir: Path, out_dir: Path,
    joint_ids: list[int], n_frames: int = 8, device: str = "cuda",
):
    """Render projected joint_ids onto a handful of frames so you can SEE whether
    the wrist/hand lands where it should. Coordinate/index bugs are invisible in
    loss numbers but obvious here. Run before AND after each stage."""
    import cv2
    # Color/radius scheme (joint-specific for clarity on high-res frames):
    #   left  wrist (20) → blue  (255,  0,  0), r=14
    #   right wrist (21) → red   (  0,  0,255), r=14
    #   left  elbow (18) → light-blue (200,100,0), r=10
    #   right elbow (19) → orange    (  0,100,200), r=10
    #   other joints     → white (255,255,255), r=8
    JOINT_STYLE: dict[int, tuple] = {
        20: ((255,   0,   0), 14),   # left  wrist  — blue
        21: ((  0,   0, 255), 14),   # right wrist  — red
        18: ((200, 100,   0), 10),   # left  elbow  — teal
        19: ((  0, 100, 200), 10),   # right elbow  — orange
    }
    LABEL: dict[int, str] = {20: "LW", 21: "RW", 18: "LE", 19: "RE"}
    DEFAULT_STYLE = ((255, 255, 255), 8)

    out_dir.mkdir(parents=True, exist_ok=True)
    F = smpl_params["body_pose"].shape[0]
    K = K_fullimg.to(device)
    if K.dim() == 2:
        K = K.unsqueeze(0).repeat(F, 1, 1)
    with torch.no_grad():
        betas = smpl_params.get("betas")
        if betas is not None:
            betas = betas.to(device)
        fwd_kwargs = dict(
            betas=betas,
            global_orient=smpl_params["global_orient"].reshape(F, 3).to(device),
            body_pose=smpl_params["body_pose"].reshape(F, -1).to(device),
            left_hand_pose=smpl_params["left_hand_pose"].reshape(F, 45).to(device),
            right_hand_pose=smpl_params["right_hand_pose"].reshape(F, 45).to(device),
            transl=smpl_params["transl"].reshape(F, 3).to(device),
        )
        for extra_key in ("jaw_pose", "leye_pose", "reye_pose"):
            if extra_key in smpl_params:
                fwd_kwargs[extra_key] = smpl_params[extra_key].reshape(F, 3).to(device)
        out = smplx_model.forward(**fwd_kwargs)
        uv = project_incam(out.joints, K).cpu().numpy()
    frames = sorted(frame_dir.glob("*.jpg"))
    step = max(1, F // n_frames)
    for fidx in range(0, F, step):
        if fidx >= len(frames):
            break
        img = cv2.imread(str(frames[fidx]))
        if img is None:
            continue
        img_h = img.shape[0]
        font_scale = img_h / 1000.0
        for jid in joint_ids:
            u, v = float(uv[fidx, jid, 0]), float(uv[fidx, jid, 1])
            color, radius = JOINT_STYLE.get(jid, DEFAULT_STYLE)
            cv2.circle(img, (int(u), int(v)), radius, color, -1)
            cv2.circle(img, (int(u), int(v)), radius + 2, (0, 0, 0), 1)  # black outline
            label = LABEL.get(jid, str(jid))
            cv2.putText(img, label, (int(u) + radius + 2, int(v) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, color, 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"overlay_{fidx:05d}.jpg"), img)
    print(f"[debug] wrote overlays to {out_dir}")


def debug_target_overlay(
    hand_targets: dict, frame_dir: Path, out_dir: Path, n_frames: int = 8,
):
    """Overlay the 2D hand TARGET uv (NOT the prediction) onto frames.

    STEP 3 verification requires checking that the *targets* sit on the actual hand.
    debug_overlay draws SMPL-X prediction joints; this draws the optimization target
    so you can catch a bad target builder (wrong lift / wrong frame / wrong mapping)
    BEFORE it silently pulls hands to the wrong place during refinement.
    Yellow dots = target keypoints (conf > 0).
    """
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frame_dir.glob("*.jpg"))
    all_fidx = sorted({fidx for side in hand_targets for fidx in hand_targets[side].keys()})
    if not all_fidx:
        raise ValueError("debug_target_overlay: no target frames to overlay.")
    step = max(1, len(all_fidx) // n_frames)
    for fidx in all_fidx[::step]:
        if fidx >= len(frames):
            continue
        img = cv2.imread(str(frames[fidx]))
        if img is None:
            continue
        for side, perframe in hand_targets.items():
            if fidx not in perframe:
                continue
            uv_arr = np.asarray(perframe[fidx]["uv"])
            conf = np.asarray(perframe[fidx]["conf"]).reshape(-1)
            # left=cyan, right=green so you can tell sides apart
            dot_color = (255, 255, 0) if side == "left" else (0, 255, 0)
            for (u, v), c in zip(uv_arr.reshape(-1, 2), conf):
                if c <= 0:
                    continue
                cv2.circle(img, (int(u), int(v)), 8, dot_color, -1)
                cv2.circle(img, (int(u), int(v)), 10, (0, 0, 0), 1)
        cv2.imwrite(str(out_dir / f"target_{fidx:05d}.jpg"), img)
    print(f"[debug] wrote TARGET overlays to {out_dir}")


# =============================================================================
# 7. TOP-LEVEL ENTRY  -- call from entry_point_hamer.py AFTER
#    merge_hamer_hands_into_gvhmr_postprocessed() produces smplx_merged_hamer_post.pt
# =============================================================================
def run_wrist_refine(
    merged_input_path: Path,         # post-processed merge result (e.g. smplx_merged_hamer_post.pt)
    hamer_out_dir: Path,
    output_path: Path,
    frame_dir: Path,
    smplx_model,                 # make_smplx("supermotion_fullhands").cuda()
    *,
    do_refine: bool = True,
    target_mode: str = "option1",   # "option1" | "option2"
    debug_dir: Path | None = None,
    device: str = "cuda",
) -> Path:
    """Apply Stage A (wrist replacement) and Stage B (2D refine) on top of an already
    merged + post-processed `.pt`. Reads the existing merge result; writes a new file
    so the baseline `.pt` is never overwritten (see HANDOFF STEP 6 comparison harness).
    """
    pred = torch.load(merged_input_path, map_location="cpu")
    merged = copy.deepcopy(pred)

    params = merged.get("smpl_params_incam")
    assert params is not None, "need smpl_params_incam for incam projection"
    K_fullimg = merged["K_fullimg"]          # (F,3,3) or (3,3) — project_incam handles both

    # STEP 0 gate: confirm kintree / body_pose layout before anything that depends on them.
    verify_info = verify_wrist_chains(smplx_model, params)
    parents = verify_info["parents"]

    F = params["body_pose"].shape[0]
    det = load_hamer_detections_full(hamer_out_dir, F)

    if debug_dir:
        debug_overlay(smplx_model, params, K_fullimg, frame_dir,
                      debug_dir / "before", [18, 19, LEFT_WRIST_FULL, RIGHT_WRIST_FULL], device=device)

    # --- Stage A: wrist replacement (incam) + sync wrists to global ---
    params = replace_wrists(params, det, parents, F)
    merged["smpl_params_incam"] = params
    if "smpl_params_global" in merged:
        merged["smpl_params_global"] = sync_wrist_rows(
            src_params=merged["smpl_params_incam"],
            dst_params=merged["smpl_params_global"],
        )
        print("[wrist_refine] Stage A: synced wrist rows incam -> smpl_params_global")

    if debug_dir:
        debug_overlay(smplx_model, params, K_fullimg, frame_dir,
                      debug_dir / "after_wrist", [18, 19, LEFT_WRIST_FULL, RIGHT_WRIST_FULL], device=device)

    # --- Stage B: 2D refine (incam) + sync ALL optimized rows + hands to global ---
    if do_refine:
        for k in ("body_pose", "global_orient", "transl", "left_hand_pose", "right_hand_pose"):
            params[k] = params[k].float()
        if target_mode == "option1":
            hand_targets = build_hand_targets_option1_hamer(det, K_fullimg)
        else:
            hand_targets = build_hand_targets_option2_detector({}, {})  # fill in
        # STEP 3 gate: verify TARGETS land on the hand BEFORE optimizing to them.
        if debug_dir:
            debug_target_overlay(hand_targets, frame_dir, debug_dir / "target_uv")
        params, refine_meta = refine_2d(smplx_model, params, K_fullimg, hand_targets,
                                        body_targets=None, device=device)
        merged["smpl_params_incam"] = params
        if "smpl_params_global" in merged:
            # Stage B changes UPPER_BODY_OPT_IDS rows + both hand poses, so a wrist-only
            # sync would leave shoulders/elbows/spine/hands stale in the global params.
            merged["smpl_params_global"] = sync_refined_local_pose(
                src_params=merged["smpl_params_incam"],
                dst_params=merged["smpl_params_global"],
            )
            print("[wrist_refine] Stage B: synced refined local pose + hands "
                  "incam -> smpl_params_global")
        if debug_dir:
            debug_overlay(smplx_model, params, K_fullimg, frame_dir,
                          debug_dir / "after_refine", [18, 19, LEFT_WRIST_FULL, RIGHT_WRIST_FULL], device=device)
    else:
        refine_meta = None

    # refine_meta is kept OUTSIDE smpl_params_incam so that dict stays pure-tensor.
    meta = merged.setdefault("cap_wrist_refine_meta", {})
    meta["target_mode"] = target_mode
    meta["do_refine"] = bool(do_refine)
    meta["input_path"] = str(merged_input_path)
    if refine_meta is not None:
        meta["refine"] = refine_meta

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    print(f"[wrist_refine] saved {output_path}")
    return output_path


# =============================================================================
# 8. FIRST-RUN CHECKLIST (do these in order; each catches a class of silent bugs)
# =============================================================================
"""
STEP 0  print(smplx_model.bm.parents) and verify the wrist parent chains.
        Confirm body_pose is (F,21,3) and full_joint = body_idx + 1.
        ---> fixes the indexing class of bugs.

STEP 1  Run debug_overlay BEFORE any change. The GVHMR wrist dots MUST land on the
        wrists in the image. If not, K_fullimg / incam frame assumption is wrong
        (out.joints not in incam, or K mismatch) -> fix project_incam / frame first.
        ---> fixes the camera/projection class of bugs.

STEP 2  Run replace_wrists, then overlay 'after_wrist'. Wrist *position* should be
        ~unchanged (we only changed rotation, not translation). If the whole hand
        teleports, the local/global frame of G_w is wrong (see NOTE in replace_wrists)
        or the mirror is wrong -> check mirror_rotmat_right_to_left on a left frame.
        ---> fixes the wrist-frame / mirror class of bugs.

STEP 3  Implement build_hand_targets_option1_hamer. Overlay the TARGET uv (not the
        prediction) on the frame to confirm targets sit on the hand. Use
        debug_target_overlay (NOT debug_overlay -- that draws predictions). A bad
        target means refine will pull the hand to the wrong place.
        ---> fixes the target class of bugs.

STEP 4  Run refine_2d with w_tc=0 first (pure data fit), confirm L_h drops and hands
        snap toward targets in 'after_refine'. THEN restore w_tc=1e4 for smoothness.
        If hands jitter, w_tc too low; if frozen/stiff, w_tc too high.
        Early stopping is on hand_rms_px (NOT total loss; w_tc=1e4 dominates total).
        Confirm the run stops on 'target_px' or 'plateau' and restores best state;
        check refine_meta (best_hand_rms_px, stop_reason, iters_run) on the output.

STEP 5  Only after overlays look right end-to-end, hand the merged .pt to the existing
        penetration post-process (unchanged) and render.

If Option-1 overlays look biased/wrong after honest debugging -> switch target_mode
to "option2", implement the detector target builder, rerun from STEP 3. The loop,
losses, SMPL-X, projection, wrist replacement all stay identical.
"""
