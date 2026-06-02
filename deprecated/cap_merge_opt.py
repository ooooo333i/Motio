"""
cap_merge_opt.py

Stage 3: GVHMR + HaMeR Merge Optimization
  Stage 1 pre-fit body + Stage 2 pre-fit HaMeR → 최종 SMPL-X merge

Pipeline:
  A) Wrist replacement: Stage 1 body에 HaMeR wrist rotation 삽입
  B) Merge refine:
       L = w_h * L_hand  +  w_b * L_body  +  w_tc * L_temporal
       - L_hand: Stage 2 palm UV targets (wrist + 4 MCP)
       - L_body: ViTPose arm UV targets (shoulder + elbow + wrist)
       - L_tc:   temporal geodesic smoothness

SMPL-X left palm joint IDs: wrist=20, idx_MCP=28, mid_MCP=31, ring_MCP=34, pinky_MCP=37
SMPL-X right palm joint IDs: wrist=21, idx_MCP=43, mid_MCP=46, ring_MCP=49, pinky_MCP=52

사용법:
  python cap_merge_opt.py
  python cap_merge_opt.py --w-h 1.0 --w-b 0.5 --w-tc 1e4 --iters 300
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))
os.chdir(str(PROJ_ROOT))

import cap_wrist_refine as cwr

# ---------------------------------------------------------------------------
# SMPL-X palm joint mapping (verified empirically)
# ---------------------------------------------------------------------------

# (MANO_palm_idx, SMPLX_joint_left, SMPLX_joint_right)
PALM_MAP = [
    (0, 20, 21),   # wrist
    (1, 28, 43),   # index_MCP
    (2, 31, 46),   # middle_MCP
    (3, 34, 49),   # ring_MCP
    (4, 37, 52),   # pinky_MCP
]

LEFT_PALM_SMPLX  = [row[1] for row in PALM_MAP]   # [20,28,31,34,37]
RIGHT_PALM_SMPLX = [row[2] for row in PALM_MAP]   # [21,43,46,49,52]

# WB133 hand-block offsets for 4 MCP joints (no wrist)
WB_L_START   = 91
WB_R_START   = 112
WB_MCP_LOCAL = [5, 9, 13, 17]   # idx_MCP, mid_MCP, ring_MCP, pinky_MCP

# ViTPose → SMPL-X for arm body targets
VP_BODY_MAP = [
    (5, 16),   # L_shoulder
    (6, 17),   # R_shoulder
    (7, 18),   # L_elbow
    (8, 19),   # R_elbow
    (9, 20),   # L_wrist
    (10, 21),  # R_wrist
]


# ---------------------------------------------------------------------------
# Target builders
# ---------------------------------------------------------------------------

def build_hand_targets_from_prefit(
    prefit: dict,
    use_wrist: bool = False,
) -> dict:
    """Build hand targets from Stage 2 prefit palm UV.

    By default (use_wrist=False), only the 4 MCP joints are used.
    The wrist joint is intentionally excluded to avoid conflict with the
    ViTPose COCO17 wrist target in body_targets — COCO17 vs WB133 wrist
    differ by median 13-15px, which would pull the SMPL-X wrist to a
    compromise position and degrade arm alignment.

    Returns targets compatible with cap_wrist_refine.refine_2d:
      targets[side][fidx] = {uv, conf, smplx_hand_joint_ids}
    """
    # Indices within LEFT/RIGHT_PALM_SMPLX to use
    # Full: [20,28,31,34,37] / [21,43,46,49,52] = wrist + 4 MCP
    # MCP-only: skip index 0 (wrist)
    start = 0 if use_wrist else 1   # 0=wrist, 1-4=MCP
    targets: dict = {"left": {}, "right": {}}
    for fidx, frame_data in prefit.items():
        for side, d in frame_data.items():
            uv_palm   = np.asarray(d["uv_palm"], np.float32)[start:]    # (4,2) or (5,2)
            smplx_ids = (LEFT_PALM_SMPLX if side == "left"
                         else RIGHT_PALM_SMPLX)[start:]
            targets[side][fidx] = {
                "uv":   uv_palm,
                "conf": np.ones(len(smplx_ids), dtype=np.float32),
                "smplx_hand_joint_ids": np.array(smplx_ids, np.int64),
            }
    return targets


def build_hand_targets_blended(
    prefit:    dict,
    wb133:     np.ndarray,   # (F, 133, 3)  [x, y, conf]
    *,
    blend_lo:  float = 0.4,  # below this: pure HaMeR UV
    blend_hi:  float = 0.6,  # above this: pure WB133
) -> dict:
    """Palm targets with per-MCP conf-gated WB133 / HaMeR blend.

    For each of the 4 MCP joints per frame per side:
      alpha = clamp((wb_conf - blend_lo) / (blend_hi - blend_lo), 0, 1)
      uv    = alpha * uv_wb133 + (1-alpha) * uv_hamer
      weight = 1.0  (blended UV carries full weight; blend is in UV space)

    This prevents hard switches at the conf=0.5 boundary that could create
    target discontinuities between adjacent frames.

    alpha=0 (pure HaMeR):  wb_conf ≤ blend_lo  → Stage-2 UV used as target
    alpha=1 (pure WB133):  wb_conf ≥ blend_hi  → WB133 position used directly
    """
    F_wb = wb133.shape[0]
    targets: dict = {"left": {}, "right": {}}
    counts = {s: {"wb": 0, "hm": 0, "blend": 0} for s in ("left", "right")}

    for fidx, frame_data in prefit.items():
        for side, d in frame_data.items():
            uv_palm   = np.asarray(d["uv_palm"], np.float32)         # (5,2): wrist+4MCP
            smplx_ids = LEFT_PALM_SMPLX if side == "left" else RIGHT_PALM_SMPLX
            wb_start  = WB_L_START if side == "left" else WB_R_START
            cnt       = counts[side]

            uv_out, weight_out, ids_out = [], [], []
            for i, wb_off in enumerate(WB_MCP_LOCAL):           # i=0..3
                uv_h     = uv_palm[i + 1]                       # hamer UV (skip wrist)
                smplx_id = smplx_ids[i + 1]

                if fidx < F_wb:
                    wb_pt   = wb133[fidx, wb_start + wb_off]    # [x, y, conf]
                    wb_conf = float(wb_pt[2])
                else:
                    wb_conf = 0.0

                alpha = float(np.clip(
                    (wb_conf - blend_lo) / max(blend_hi - blend_lo, 1e-6),
                    0.0, 1.0,
                ))

                if alpha <= 0.0:
                    uv = uv_h
                    cnt["hm"] += 1
                elif alpha >= 1.0:
                    uv = wb_pt[:2].astype(np.float32)
                    cnt["wb"] += 1
                else:
                    uv = (alpha * wb_pt[:2] + (1.0 - alpha) * uv_h).astype(np.float32)
                    cnt["blend"] += 1

                uv_out.append(uv)
                weight_out.append(1.0)
                ids_out.append(smplx_id)

            targets[side][fidx] = {
                "uv":   np.array(uv_out,   np.float32),
                "conf": np.array(weight_out, np.float32),
                "smplx_hand_joint_ids": np.array(ids_out, np.int64),
            }

    for side in ("left", "right"):
        c = counts[side]
        total = c["wb"] + c["hm"] + c["blend"]
        print(f"[merge_opt] {side} palm targets: "
              f"WB133={c['wb']} HaMeR={c['hm']} blend={c['blend']} / {total} MCP slots")
    return targets


def find_both_bad_frames(
    prefit: dict,
    wb133:  np.ndarray,
    *,
    wb_conf_thresh:   float = 0.5,
    dist_thresh_px:   float = 50.0,
    dist_warn_thresh: float = 20.0,
) -> dict:
    """Identify frames where BOTH WB133 and HaMeR UV are questionable.

    Returns a dict of {frame: {side: reason_str}} for logging only.
    Three categories:
      'wb_low_hm_missing'  : WB133 conf<thresh AND HaMeR prefit entry absent
      'wb_low_hm_far'      : WB133 conf<thresh AND WB-HaMeR dist>dist_thresh (when measurable)
      'wb_low_hm_moderate' : WB133 conf<thresh AND WB-HaMeR dist in (dist_warn,dist_thresh)
    These are candidates for temporal interpolation in a future step.
    """
    F_wb = wb133.shape[0]
    result: dict = {}

    for side, wb_start in [("left", WB_L_START), ("right", WB_R_START)]:
        for fidx in range(F_wb):
            # WB133 mean conf over 4 MCP
            confs = [float(wb133[fidx, wb_start + off, 2]) for off in WB_MCP_LOCAL]
            mean_conf = float(np.mean(confs))
            if mean_conf >= wb_conf_thresh:
                continue   # WB133 is fine → not a "both bad" frame

            # WB133 is unreliable — now check HaMeR
            if fidx not in prefit or side not in prefit[fidx]:
                reason = "wb_low_hm_missing"
            else:
                uv_hamer = np.asarray(prefit[fidx][side]["uv_palm"], np.float32)
                dists = []
                for i, off in enumerate(WB_MCP_LOCAL):
                    wb_pt = wb133[fidx, wb_start + off]
                    if float(wb_pt[2]) >= 0.3:          # only measure where WB133 has some signal
                        dists.append(float(np.linalg.norm(uv_hamer[i+1] - wb_pt[:2])))
                if not dists:
                    reason = "wb_low_hm_unknown"        # can't measure, WB133 fully absent
                elif np.mean(dists) > dist_thresh_px:
                    reason = "wb_low_hm_far"
                elif np.mean(dists) > dist_warn_thresh:
                    reason = "wb_low_hm_moderate"
                else:
                    continue   # HaMeR is plausible even if WB133 is bad

            if fidx not in result:
                result[fidx] = {}
            result[fidx][side] = (reason, round(mean_conf, 3))

    return result


def build_body_targets_vitpose(
    vitpose: np.ndarray,       # (F, 17, 3)
    n_frames: int,
    conf_thresh: float = 0.3,
) -> dict:
    """Build ViTPose arm 2D targets for body anchor in Stage B.
    Returns targets[side][fidx] = {uv, conf, smplx_hand_joint_ids}.
    Side = "left" covers L_shoulder/L_elbow/L_wrist,
    side = "right" covers R_shoulder/R_elbow/R_wrist.
    """
    VP_LEFT  = [(5, 16), (7, 18), (9, 20)]   # VP_idx, SMPLX_joint
    VP_RIGHT = [(6, 17), (8, 19), (10, 21)]

    targets: dict = {"left": {}, "right": {}}
    for fidx in range(n_frames):
        for side, mapping in [("left", VP_LEFT), ("right", VP_RIGHT)]:
            uvs, confs, ids = [], [], []
            for vp_i, smplx_j in mapping:
                p = vitpose[fidx, vp_i]
                c = float(p[2]) if float(p[2]) >= conf_thresh else 0.0
                uvs.append(p[:2].astype(np.float32))
                confs.append(c)
                ids.append(smplx_j)
            targets[side][fidx] = {
                "uv":   np.array(uvs, np.float32),
                "conf": np.array(confs, np.float32),
                "smplx_hand_joint_ids": np.array(ids, np.int64),
            }
    return targets


def merge_body_targets(body_t1: dict, body_t2: dict) -> dict:
    """Merge two body target dicts by concatenating joint arrays per frame."""
    result: dict = {}
    all_sides = set(body_t1) | set(body_t2)
    for side in all_sides:
        result[side] = {}
        frames1 = body_t1.get(side, {})
        frames2 = body_t2.get(side, {})
        all_frames = set(frames1) | set(frames2)
        for fidx in all_frames:
            t1 = frames1.get(fidx)
            t2 = frames2.get(fidx)
            if t1 is None:
                result[side][fidx] = t2
            elif t2 is None:
                result[side][fidx] = t1
            else:
                # Concatenate, ensuring same joint ids list (no duplicates)
                ids1 = list(t1["smplx_hand_joint_ids"])
                ids2 = list(t2["smplx_hand_joint_ids"])
                # Only keep t2 joints not already in t1
                keep = [i for i, jid in enumerate(ids2) if jid not in ids1]
                if not keep:
                    result[side][fidx] = t1
                    continue
                uv   = np.concatenate([t1["uv"],
                                       t2["uv"][keep]], axis=0)
                conf = np.concatenate([t1["conf"],
                                       t2["conf"][keep]], axis=0)
                ids  = ids1 + [ids2[k] for k in keep]
                result[side][fidx] = {
                    "uv": uv, "conf": conf,
                    "smplx_hand_joint_ids": np.array(ids, np.int64),
                }
    return result


# ---------------------------------------------------------------------------
# Main merge pipeline
# ---------------------------------------------------------------------------

def run_merge_opt(
    arm_prefit_path: Path,      # Stage 1 output
    hamer_dir:       Path,      # original hamer_out_clean/
    mano_prefit_path:Path,      # Stage 2 output
    vitpose_path:    Path,
    wb133_path:      Path,      # WB133 wholebody 2D keypoints
    output_path:     Path,
    frame_dir:       Path,
    debug_dir:       Path,
    smplx_model_ref,            # loaded SMPL-X model
    *,
    w_h:           float = 1.0,     # hand target weight
    w_b:           float = 2.0,     # body target weight (ViTPose arm)
    w_tc:          float = 1e4,
    w_hand_prior:  float = 5.0,     # B2 hand pose prior (keep near Stage-A shape)
    lr:            float = 2e-2,
    iters_b1:      int   = 200,     # B1: arm-only iterations
    iters_b2:      int   = 150,     # B2: hand-only iterations
    target_px:     float = 12.0,
    min_iter:      int   = 60,
    patience:      int   = 40,
    device:        str   = "cuda",
    debug_frames:  int   = 12,
    palm_blend_lo: float = 0.4,     # WB133 blend range lower bound
    palm_blend_hi: float = 0.6,     # WB133 blend range upper bound
) -> Path:

    print("[merge_opt] Loading Stage 1 body params...")
    pred_s1 = torch.load(arm_prefit_path, map_location="cpu")
    params  = copy.deepcopy(pred_s1["smpl_params_incam"])
    K_fullimg = pred_s1["K_fullimg"]
    F = params["body_pose"].shape[0]

    print("[merge_opt] Loading Stage 2 MANO prefit...")
    prefit_mano = torch.load(mano_prefit_path, map_location="cpu")

    print("[merge_opt] Loading WB133...")
    wb133 = np.load(wb133_path).astype(np.float32)   # (F, 133, 3)

    print("[merge_opt] Loading ViTPose...")
    vp_raw = torch.load(vitpose_path, map_location="cpu")
    vitpose = vp_raw.numpy() if isinstance(vp_raw, torch.Tensor) else np.array(vp_raw)

    print("[merge_opt] Loading HaMeR detections (for Stage A wrist)...")
    det = cwr.load_hamer_detections_full(hamer_dir, F)

    # ---- Stage A: Wrist replacement ----
    print("[merge_opt] Stage A: wrist replacement...")
    from hmr4d.utils.smplx_utils import make_smplx
    smplx_model = make_smplx("supermotion_fullhands").to(device)

    chain_info = cwr.verify_wrist_chains(smplx_model, params)
    parents = chain_info["parents"]

    for k in params:
        if isinstance(params[k], torch.Tensor):
            params[k] = params[k].float()

    params = cwr.replace_wrists(params, det, parents, F)

    # ---- Stage A+: HaMeR hand_pose 직접 주입 ----
    # HaMeR는 pose2rot=False (rotation matrix 직접) → hand_mean 미적용 상태로 저장됨.
    # SMPL-X는 pose2rot=True → hand_pose_aa + hand_mean → rot_mat.
    # 따라서 주입 전 hand_mean을 빼야 SMPL-X 적용 결과가 HaMeR 원본과 일치.
    l_mean = smplx_model.bm.left_hand_mean.detach().cpu()   # (45,)
    r_mean = smplx_model.bm.right_hand_mean.detach().cpu()  # (45,)

    lhp = params["left_hand_pose"].reshape(F, 45).clone()
    rhp = params["right_hand_pose"].reshape(F, 45).clone()
    n_injected = {"left": 0, "right": 0}
    for side, hp_t, mean in [("left", lhp, l_mean), ("right", rhp, r_mean)]:
        for fidx, d_f in det[side].items():
            if fidx < F:
                hp_t[fidx] = torch.from_numpy(d_f["hand_pose"]).float() - mean
                n_injected[side] += 1
    params["left_hand_pose"]  = lhp
    params["right_hand_pose"] = rhp
    print(f"[merge_opt] Stage A+: hand_pose injected (- hand_mean) "
          f"left={n_injected['left']} right={n_injected['right']} frames")
    print("[merge_opt] Stage A done.")

    # ---- Build targets ----
    print("[merge_opt] Building targets...")

    # Hand targets: conf-gated blend of WB133 (reliable) and HaMeR UV (fallback)
    hand_targets = build_hand_targets_blended(
        prefit_mano, wb133,
        blend_lo=palm_blend_lo,
        blend_hi=palm_blend_hi,
    )

    # ---- Diagnostic: both-bad frames (WB133 low AND HaMeR UV questionable) ----
    both_bad = find_both_bad_frames(prefit_mano, wb133)
    if both_bad:
        print(f"[merge_opt] Both-bad frames (WB conf<0.5 AND HaMeR uncertain): "
              f"{len(both_bad)} frames — temporal interp candidates")
        for fidx in sorted(both_bad)[:20]:
            print(f"  f{fidx}: {both_bad[fidx]}")
        if len(both_bad) > 20:
            print(f"  ... (+{len(both_bad)-20} more)")

    # Body targets: ViTPose arm (shoulder+elbow+wrist per side)
    body_targets = build_body_targets_vitpose(vitpose, F)

    print(f"[merge_opt] Hand targets: "
          f"left={len(hand_targets.get('left',{}))} "
          f"right={len(hand_targets.get('right',{}))} frames")
    print(f"[merge_opt] Body targets: "
          f"left={len(body_targets.get('left',{}))} frames (arm anchor)")

    # ---- Debug: target overlay (before merge) ----
    debug_dir.mkdir(parents=True, exist_ok=True)
    cwr.debug_overlay(
        smplx_model, params, K_fullimg, frame_dir,
        debug_dir / "before_merge",
        [18, 19, 20, 21] + LEFT_PALM_SMPLX[1:] + RIGHT_PALM_SMPLX[1:],
        n_frames=6, device=device,
    )

    # ---- Stage B1/B2: 비활성화 (새 파이프라인: HaMeR 직접 주입만 사용) ----
    # refine_2d 함수는 유지, 호출만 끔.
    params_refined = params
    meta = {"b1": "disabled", "b2": "disabled"}

    # ---- Debug: after merge ----
    cwr.debug_overlay(
        smplx_model, params_refined, K_fullimg, frame_dir,
        debug_dir / "after_merge",
        [18, 19, 20, 21] + LEFT_PALM_SMPLX[1:] + RIGHT_PALM_SMPLX[1:],
        n_frames=6, device=device,
    )

    # ---- Build output ----
    result = copy.deepcopy(pred_s1)
    result["smpl_params_incam"] = params_refined

    if "smpl_params_global" in result:
        # wrist body_pose 동기화
        result["smpl_params_global"] = cwr.sync_wrist_rows(
            src_params=params_refined,
            dst_params=result["smpl_params_global"],
        )
        # hand_pose 동기화
        result["smpl_params_global"]["left_hand_pose"]  = params_refined["left_hand_pose"].clone()
        result["smpl_params_global"]["right_hand_pose"] = params_refined["right_hand_pose"].clone()
        print("[merge_opt] synced wrist rows + hand_pose incam → global")

    result.setdefault("cap_merge_meta", {}).update({
        "merge_refine_meta": meta,
        "w_h": w_h, "w_b": w_b, "w_tc": w_tc,
        "stage1_source": str(arm_prefit_path),
        "stage2_source": str(mano_prefit_path),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    print(f"[merge_opt] saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 3: GVHMR+HaMeR merge optimization")
    parser.add_argument("--input-dir",   default="outputs/cap_pipeline/input")
    parser.add_argument("--w-h",           type=float, default=1.0,
                        help="Hand target (Stage 2 palm) weight")
    parser.add_argument("--w-b",           type=float, default=2.0,
                        help="Body target (ViTPose arm) weight for B1")
    parser.add_argument("--w-tc",          type=float, default=1e4,
                        help="Temporal consistency weight")
    parser.add_argument("--w-hand-prior",  type=float, default=5.0,
                        help="B2 hand pose prior weight (keep near Stage-A shape)")
    parser.add_argument("--lr",            type=float, default=2e-2)
    parser.add_argument("--iters-b1",      type=int,   default=200)
    parser.add_argument("--iters-b2",      type=int,   default=150)
    parser.add_argument("--target-px",     type=float, default=12.0)
    parser.add_argument("--debug-frames",  type=int,   default=12)
    parser.add_argument("--device",                    default="cuda")
    parser.add_argument("--palm-blend-lo", type=float, default=0.4,
                        help="WB133 blend lower bound (below→pure HaMeR)")
    parser.add_argument("--palm-blend-hi", type=float, default=0.6,
                        help="WB133 blend upper bound (above→pure WB133)")
    args = parser.parse_args()

    d = Path(args.input_dir)

    from hmr4d.utils.smplx_utils import make_smplx
    smplx_model_ref = make_smplx("supermotion_fullhands").to(args.device)

    run_merge_opt(
        arm_prefit_path  = d / "smplx_arm_prefit.pt",
        hamer_dir        = d / "hamer_out_clean",
        mano_prefit_path = d / "hamer_mano_prefit.pt",
        vitpose_path     = d / "preprocess" / "vitpose.pt",
        wb133_path       = d / "preprocess" / "wholebody_kp2d.npy",
        output_path      = d / "smplx_merged_prefit.pt",
        frame_dir        = d / "hamer_frames",
        debug_dir        = d / "wrist_refine_debug" / "merge_opt",
        smplx_model_ref  = smplx_model_ref,
        w_h              = args.w_h,
        w_b              = args.w_b,
        w_tc             = args.w_tc,
        w_hand_prior     = args.w_hand_prior,
        lr               = args.lr,
        iters_b1         = args.iters_b1,
        iters_b2         = args.iters_b2,
        target_px        = args.target_px,
        device           = args.device,
        debug_frames     = args.debug_frames,
        palm_blend_lo    = args.palm_blend_lo,
        palm_blend_hi    = args.palm_blend_hi,
    )


if __name__ == "__main__":
    main()
