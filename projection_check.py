"""
projection_check.py

Lightweight 2D projection alignment check — no optimization, no rendering.

Per selected frame, overlays on the actual video frame:
  - SMPL-X/GVHMR projected hand joints (red dots)   — from smplx_merged_hamer_post.pt
  - HaMeR keypoints_2d (yellow dots)                — directly from npz, crop-to-full transform
  - HaMeR bbox (blue rectangle)                     — from box_center / box_size

Saves per-side and combined images to:
  outputs/cap_pipeline/input/wrist_refine_debug/projection_check/
"""

import sys, os, re, copy
import numpy as np
import cv2
import torch
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))
os.chdir(PROJ_ROOT)

MERGED_PT    = PROJ_ROOT / "outputs/cap_pipeline/input/smplx_merged_hamer_post.pt"
HAMER_DIR    = PROJ_ROOT / "outputs/cap_pipeline/input/hamer_out_clean"
FRAME_DIR    = PROJ_ROOT / "outputs/cap_pipeline/input/hamer_frames"
OUT_DIR      = PROJ_ROOT / "outputs/cap_pipeline/input/wrist_refine_debug/projection_check"

CHECK_FRAMES = [0, 100, 200, 300, 500, 654, 700, 1000, 1400]

# SMPL-X joint IDs for hand overlay
BODY_JOINTS_DRAW = [20, 21]          # wrists (confirmed in incam)
LEFT_HAND_JOINTS = [20] + list(range(25, 40))   # wrist + 15 finger kinematic
RIGHT_HAND_JOINTS = [21] + list(range(40, 55))


def load_hamer_frame(hamer_dir: Path, frame_idx: int):
    """Return {side: {box_center, box_size, kp2d_full}} for detected hands in a frame."""
    pattern = re.compile(r"^(\d+)_")
    detections = {}
    for npz_path in sorted(hamer_dir.glob("*.npz")):
        m = pattern.match(npz_path.stem)
        if m is None or int(m.group(1)) != frame_idx:
            continue
        d = np.load(npz_path, allow_pickle=True)
        is_right = int(np.asarray(d["is_right"]).reshape(-1)[0])
        side = "right" if is_right == 1 else "left"

        bc = np.asarray(d["box_center"], np.float32).reshape(-1)   # (2,) [cx, cy] full-image px
        bs = float(np.asarray(d["box_size"]).reshape(-1)[0])       # scalar
        kp2d_crop = np.asarray(d["keypoints_2d"], np.float32).reshape(21, 2)

        # Convert from crop normalized [-1,1] to full-image pixel coords
        kp2d_full = np.stack([
            bc[0] + kp2d_crop[:, 0] * (bs / 2),
            bc[1] + kp2d_crop[:, 1] * (bs / 2),
        ], axis=1)  # (21, 2)

        detections[side] = {
            "box_center": bc,
            "box_size": bs,
            "kp2d_full": kp2d_full,
            "npz_stem": npz_path.stem,
        }
    return detections


def project_smplx_joints(smplx_model, params, K, frame_idx: int, device="cuda"):
    """Run SMPLX forward for ONE frame and project joints. Returns (127, 2) uv in pixels."""
    fi = frame_idx
    with torch.no_grad():
        out = smplx_model.forward(
            betas=params["betas"][fi:fi+1].to(device),
            global_orient=params["global_orient"][fi:fi+1].to(device),
            body_pose=params["body_pose"][fi:fi+1].to(device),
            left_hand_pose=params["left_hand_pose"][fi:fi+1].to(device),
            right_hand_pose=params["right_hand_pose"][fi:fi+1].to(device),
            transl=params["transl"][fi:fi+1].to(device),
            jaw_pose=params["jaw_pose"][fi:fi+1].to(device) if "jaw_pose" in params else None,
            leye_pose=params["leye_pose"][fi:fi+1].to(device) if "leye_pose" in params else None,
            reye_pose=params["reye_pose"][fi:fi+1].to(device) if "reye_pose" in params else None,
        )
        j3d = out.joints  # (1, J, 3) in incam

    K_f = K[fi] if K.dim() == 3 else K  # (3, 3)
    z = j3d[0, :, 2].clamp(min=1e-4).cpu()
    u = K_f[0, 0].item() * j3d[0, :, 0].cpu() / z + K_f[0, 2].item()
    v = K_f[1, 1].item() * j3d[0, :, 1].cpu() / z + K_f[1, 2].item()
    uv = torch.stack([u, v], dim=1).numpy()  # (J, 2)
    return uv


def draw_overlay(img, uv_smplx, hamer_det, side, frame_idx):
    """Draw overlay for one side onto img (in-place). Returns img."""
    out = img.copy()
    label_color = (0, 255, 0)  # text label

    # --- HaMeR bbox (blue) ---
    if hamer_det is not None:
        bc = hamer_det["box_center"]
        bs = hamer_det["box_size"]
        x1 = int(bc[0] - bs / 2); y1 = int(bc[1] - bs / 2)
        x2 = int(bc[0] + bs / 2); y2 = int(bc[1] + bs / 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 100, 0), 2)  # blue
        cv2.putText(out, f"{side} bbox", (x1 + 4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2, cv2.LINE_AA)

        # --- HaMeR keypoints_2d (yellow) ---
        kp = hamer_det["kp2d_full"]
        for i, (u, v) in enumerate(kp):
            u_i, v_i = int(round(u)), int(round(v))
            radius = 5 if i == 0 else 3  # bigger dot for wrist
            cv2.circle(out, (u_i, v_i), radius, (0, 230, 230), -1)  # yellow

    # --- SMPL-X projected joints (red) ---
    if uv_smplx is not None:
        hand_ids = LEFT_HAND_JOINTS if side == "left" else RIGHT_HAND_JOINTS
        for ji, jid in enumerate(hand_ids):
            u_i, v_i = int(round(uv_smplx[jid, 0])), int(round(uv_smplx[jid, 1]))
            radius = 5 if ji == 0 else 3
            cv2.circle(out, (u_i, v_i), radius, (0, 0, 220), -1)  # red

    # Frame / side label
    cv2.putText(out, f"f={frame_idx} {side}", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, f"f={frame_idx} {side}", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

    # Legend
    h = out.shape[0]
    cv2.putText(out, "RED=SMPL-X  YELLOW=HaMeR2d  BLUE=bbox",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "RED=SMPL-X  YELLOW=HaMeR2d  BLUE=bbox",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load merged .pt
    print(f"Loading {MERGED_PT.name} ...")
    pred = torch.load(MERGED_PT, map_location="cpu", weights_only=False)
    params = pred["smpl_params_incam"]
    K = pred["K_fullimg"]  # (F, 3, 3) or (3, 3)

    # Load SMPL-X model
    print("Loading SMPL-X model ...")
    from hmr4d.utils.smplx_utils import make_smplx
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smplx_model = make_smplx("supermotion_fullhands").to(device)

    frame_paths = sorted(FRAME_DIR.glob("*.jpg"))
    F = params["body_pose"].shape[0]
    print(f"Total frames in .pt: {F}")

    saved = []
    for fidx in CHECK_FRAMES:
        if fidx >= len(frame_paths):
            print(f"  frame {fidx}: no image file, skip")
            continue
        img_path = frame_paths[fidx]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  frame {fidx}: cannot read {img_path.name}, skip")
            continue

        print(f"  frame {fidx}: {img.shape[:2]} ...")

        # SMPL-X forward + project
        uv_smplx = project_smplx_joints(smplx_model, params, K, fidx, device=device)

        # HaMeR detections
        hamer_dets = load_hamer_frame(HAMER_DIR, fidx)

        # Per-side images
        side_imgs = {}
        for side in ("left", "right"):
            det = hamer_dets.get(side, None)
            side_img = draw_overlay(img, uv_smplx, det, side, fidx)
            fname = OUT_DIR / f"f{fidx:04d}_{side}.jpg"
            cv2.imwrite(str(fname), side_img)
            side_imgs[side] = side_img
            saved.append(fname)

        # Combined: both sides on same image
        combined = img.copy()
        for side in ("left", "right"):
            det = hamer_dets.get(side, None)
            # bbox
            if det is not None:
                bc = det["box_center"]
                bs = det["box_size"]
                x1 = int(bc[0] - bs/2); y1 = int(bc[1] - bs/2)
                x2 = int(bc[0] + bs/2); y2 = int(bc[1] + bs/2)
                cv2.rectangle(combined, (x1, y1), (x2, y2), (255, 100, 0), 2)
                # HaMeR kp2d (yellow)
                for i, (u, v) in enumerate(det["kp2d_full"]):
                    r = 5 if i == 0 else 3
                    cv2.circle(combined, (int(round(u)), int(round(v))), r, (0, 230, 230), -1)
            # SMPL-X joints (red)
            hand_ids = LEFT_HAND_JOINTS if side == "left" else RIGHT_HAND_JOINTS
            for ji, jid in enumerate(hand_ids):
                u_i, v_i = int(round(uv_smplx[jid, 0])), int(round(uv_smplx[jid, 1]))
                r = 6 if ji == 0 else 3
                cv2.circle(combined, (u_i, v_i), r, (0, 0, 220), -1)

        cv2.putText(combined, f"f={fidx}  L+R combined", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(combined, f"f={fidx}  L+R combined", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
        h = combined.shape[0]
        cv2.putText(combined, "RED=SMPL-X  YELLOW=HaMeR2d  BLUE=bbox",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, "RED=SMPL-X  YELLOW=HaMeR2d  BLUE=bbox",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

        fname_c = OUT_DIR / f"f{fidx:04d}_combined.jpg"
        cv2.imwrite(str(fname_c), combined)
        saved.append(fname_c)
        print(f"    saved f{fidx:04d}_left/right/combined.jpg")

    print(f"\nAll images saved to: {OUT_DIR}")
    print(f"Total files: {len(saved)}")
    for p in sorted(saved):
        print(f"  {p}")


if __name__ == "__main__":
    main()
