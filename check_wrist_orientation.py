"""
check_wrist_orientation.py

HaMeR vs SMPL-X 손목 orientation 비교 contact sheet.

각 프레임에서:
  - HaMeR wrist_R 의 x/y/z 축 → 실선 화살표 (적/녹/청)
  - SMPL-X FK로 계산한 wrist global rotation → 점선 화살표 (마젠타/노랑/시안)
  - 두 rotation 사이 geodesic angle (deg) 텍스트

출력: outputs/cap_pipeline/input/wrist_refine_debug/wrist_orientation/
       forehand_f360-370.jpg
       backhand_f1150-1160.jpg
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import cv2

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

from hmr4d.utils.smplx_utils import make_smplx
from cap_wrist_refine import (
    aa_to_rotmat, project_incam, load_hamer_detections_full,
)

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
INPUT_DIR   = Path("outputs/cap_pipeline/input")
OUT_DIR     = INPUT_DIR / "wrist_refine_debug" / "wrist_orientation"
DEVICE      = "cuda"
ARROW_M     = 0.07   # 3D 화살표 길이 (meter) — 뷰어 거리 ~5m 기준 약 30px

SEGMENTS = {
    "forehand_f360-370":    list(range(360, 371)),
    "backhand_f1150-1160":  list(range(1150, 1161)),
}

# BGR 색상
#   HaMeR  : x=빨강  y=초록  z=파랑   (RGB 표준)
#   SMPL-X : x=마젠타 y=노랑  z=시안   (CMY 보색)
HAMER_COLS  = [(0, 0, 220), (0, 200, 0), (200, 0, 0)]      # B G R
SMPLX_COLS  = [(220, 0, 220), (0, 220, 220), (220, 200, 0)]

# FK 체인: [pelvis, ..., wrist] (full SMPL-X joint IDs, root-first)
LEFT_CHAIN  = [0, 3, 6, 9, 13, 16, 18, 20]
RIGHT_CHAIN = [0, 3, 6, 9, 14, 17, 19, 21]


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────
def fk_global_R(fidx: int, chain: list[int],
                global_orient: torch.Tensor,
                body_pose: torch.Tensor) -> np.ndarray:
    """SMPL-X FK로 wrist global rotation (3,3) 계산."""
    G = aa_to_rotmat(global_orient[fidx])        # pelvis: global_orient
    for jid in chain[1:]:                        # chain[0]=pelvis, 이미 위에서 처리
        G = G @ aa_to_rotmat(body_pose[fidx, jid - 1])  # body_pose idx = jid-1
    return G.detach().cpu().numpy().astype(np.float32)


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    Rrel  = Ra.T @ Rb
    trace = float(np.clip(np.trace(Rrel), -1 + 1e-6, 3 - 1e-6))
    return float(np.degrees(np.arccos((trace - 1.0) / 2.0)))


def proj2d(pt3: np.ndarray, K: np.ndarray) -> tuple[int, int]:
    """3D (X,Y,Z) → 픽셀 (u,v)."""
    z = max(float(pt3[2]), 1e-4)
    u = K[0, 0] * float(pt3[0]) / z + K[0, 2]
    v = K[1, 1] * float(pt3[1]) / z + K[1, 2]
    return int(round(u)), int(round(v))


def draw_axes(img: np.ndarray, R: np.ndarray, center3d: np.ndarray,
              K: np.ndarray, colors: list, thickness: int = 3,
              dashed: bool = False) -> None:
    """R의 x/y/z 컬럼 축을 center3d에서 화살표로 그린다."""
    c2d = proj2d(center3d, K)
    for ax in range(3):
        tip3 = center3d + ARROW_M * R[:, ax]
        t2d  = proj2d(tip3, K)
        col  = colors[ax]
        if dashed:
            # 점선: 짧은 선분들로 분할
            n_seg = 6
            for i in range(n_seg):
                if i % 2 == 0:
                    s = np.array(c2d) + (np.array(t2d) - np.array(c2d)) * i / n_seg
                    e = np.array(c2d) + (np.array(t2d) - np.array(c2d)) * (i + 1) / n_seg
                    cv2.line(img, tuple(s.astype(int)), tuple(e.astype(int)), col, thickness)
        else:
            cv2.arrowedLine(img, c2d, t2d, col, thickness, tipLength=0.25)


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────
print("[orient] Loading merged params...")
merged = torch.load(INPUT_DIR / "smplx_merged_prefit.pt", map_location="cpu")
params = merged["smpl_params_incam"]
K_raw  = merged["K_fullimg"]          # (F,3,3) or (3,3)
F_total = params["body_pose"].shape[0]

body_pose     = params["body_pose"].reshape(F_total, 21, 3)
global_orient = params["global_orient"].reshape(F_total, 3)

smplx_model = make_smplx("supermotion_fullhands").to(DEVICE)
K_t = K_raw.to(DEVICE)
if K_t.dim() == 2:
    K_t = K_t.unsqueeze(0).repeat(F_total, 1, 1)

print("[orient] SMPL-X forward pass...")
with torch.no_grad():
    smplx_out = smplx_model.forward(
        betas           = params["betas"].to(DEVICE) if "betas" in params else None,
        global_orient   = global_orient.to(DEVICE),
        body_pose       = body_pose.reshape(F_total, -1).to(DEVICE),
        left_hand_pose  = params["left_hand_pose"].reshape(F_total, 45).to(DEVICE),
        right_hand_pose = params["right_hand_pose"].reshape(F_total, 45).to(DEVICE),
        transl          = params["transl"].reshape(F_total, 3).to(DEVICE),
    )
    joints3d = smplx_out.joints.cpu().numpy()  # (F, J, 3)

print("[orient] Loading HaMeR detections...")
det = load_hamer_detections_full(INPUT_DIR / "hamer_out_clean", F_total)

frames_dir = INPUT_DIR / "hamer_frames"
frame_paths = sorted(frames_dir.glob("*.jpg"))

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 세그먼트 루프
# ──────────────────────────────────────────────
CELL_SCALE = 0.35      # 원본의 35% → 378×672 per cell
COLS       = 5
LABEL_H    = 60        # 상단 레이블 바 높이

for seg_name, fidxs in SEGMENTS.items():
    print(f"\n[orient] Segment: {seg_name}")
    cells = []

    for fidx in fidxs:
        if fidx >= len(frame_paths):
            continue
        img_orig = cv2.imread(str(frame_paths[fidx]))
        if img_orig is None:
            continue
        img = img_orig.copy()
        h_orig, w_orig = img.shape[:2]

        # K for this frame (numpy)
        if K_raw.dim() == 3:
            K_np = K_raw[fidx].numpy().astype(np.float32)
        else:
            K_np = K_raw.numpy().astype(np.float32)

        angle_info = []

        for side, wrist_jid, chain in [
            ("left",  20, LEFT_CHAIN),
            ("right", 21, RIGHT_CHAIN),
        ]:
            # ── SMPL-X 손목 global rotation (FK) ──
            R_smplx = fk_global_R(fidx, chain, global_orient, body_pose)

            # ── SMPL-X 손목 3D 위치 (incam, meters) ──
            wrist3d_smplx = joints3d[fidx, wrist_jid].astype(np.float32)  # (3,)

            # ── HaMeR 손목 rotation & 위치 ──
            d_f = det[side].get(fidx)
            if d_f is None:
                angle_info.append(f"{side[:1].upper()}: no HaMeR")
                # SMPL-X 축만 그리기
                draw_axes(img, R_smplx, wrist3d_smplx, K_np,
                          SMPLX_COLS, thickness=3, dashed=True)
                continue

            R_hamer = d_f["wrist_R"].astype(np.float32)   # (3,3) camera frame

            # HaMeR 손목 3D 위치 (camera frame, same as incam)
            kp3d = d_f["keypoints_3d"]          # (21,3) MANO space
            cam_t = d_f["cam_t_full"].reshape(3)
            w3d = kp3d[0].copy()                 # MANO joint 0 = wrist
            if side == "left":
                w3d[0] *= -1
            wrist3d_hamer = (w3d + cam_t).astype(np.float32)

            # ── 각도 차이 ──
            angle_deg = geodesic_deg(R_smplx, R_hamer)
            angle_info.append(f"{side[:1].upper()}:{angle_deg:.1f}°")

            # ── HaMeR 화살표 (실선) ──
            draw_axes(img, R_hamer, wrist3d_hamer, K_np,
                      HAMER_COLS, thickness=4, dashed=False)

            # ── SMPL-X 화살표 (점선) ──
            draw_axes(img, R_smplx, wrist3d_smplx, K_np,
                      SMPLX_COLS, thickness=3, dashed=True)

        # 각도 텍스트 표시
        angle_str = "  ".join(angle_info)
        fscale = h_orig / 1000.0
        cv2.putText(img, f"f{fidx}  {angle_str}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    fscale * 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, f"f{fidx}  {angle_str}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    fscale * 0.9, (20, 20, 20), 1, cv2.LINE_AA)

        # 범례 (첫 프레임에만 기입하면 충분하지만 일단 매 프레임)
        ly = h_orig - 280
        for txt, col in [
            ("HaMeR  x", HAMER_COLS[0]), ("HaMeR  y", HAMER_COLS[1]),
            ("HaMeR  z", HAMER_COLS[2]), ("SMPLX  x", SMPLX_COLS[0]),
            ("SMPLX  y", SMPLX_COLS[1]), ("SMPLX  z", SMPLX_COLS[2]),
        ]:
            cv2.putText(img, txt, (18, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        fscale * 0.38, col, 2, cv2.LINE_AA)
            ly += int(fscale * 42)

        # 셀로 축소
        cw = int(w_orig * CELL_SCALE)
        ch = int(h_orig * CELL_SCALE)
        cell = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
        cells.append(cell)

    if not cells:
        print(f"[orient] No frames for {seg_name}")
        continue

    # ── contact sheet 조립 ──
    cw, ch = cells[0].shape[1], cells[0].shape[0]
    rows   = (len(cells) + COLS - 1) // COLS

    # 헤더 바 추가
    sheet_w = cw * COLS
    header  = np.zeros((LABEL_H, sheet_w, 3), dtype=np.uint8)
    header[:] = (40, 40, 40)
    legend_x = 10
    for txt, col in [
        ("HaMeR: x=red  y=green  z=blue   |   SMPL-X: x=magenta  y=yellow  z=cyan   |   angle = geodesic deg", (200, 200, 200)),
    ]:
        cv2.putText(header, txt, (legend_x, LABEL_H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    row_imgs = []
    for r in range(rows):
        row_cells = cells[r * COLS: (r + 1) * COLS]
        # pad if last row is short
        while len(row_cells) < COLS:
            row_cells.append(np.zeros((ch, cw, 3), dtype=np.uint8))
        row_imgs.append(np.concatenate(row_cells, axis=1))

    sheet = np.concatenate([header] + row_imgs, axis=0)
    out_path = OUT_DIR / f"{seg_name}.jpg"
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[orient] Saved → {out_path}  ({sheet.shape[1]}×{sheet.shape[0]})")

print("\n[orient] Done.")
