"""
좌측 손 HMP 결과 시각 확인 스크립트
- HMP는 rgb_pseudo_right(미러 프레임)에서 최적화 → 결과는 미러 공간
- 원본 프레임에 시각화 시 X축 반전 필요
- body_pose[19] = left wrist 구간(f355) 검증
"""
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation

# ── 설정 ──────────────────────────────────────────────────
NPZ_PATH = "/workspace/hmp_test/optim/in_the_wild_sample_config/in_the_wild/tennis_swing_left/recon_000_30fps.npz"
FRAME_DIR = "/workspace/capstone_deploy/outputs/cap_pipeline/input/hamer_frames"
OUT_DIR = "/workspace/capstone_deploy/outputs/vis_hmp_left"
START_FRAME = 320
NUM_INPUT_FRAMES = 111
HMP_FRAMES = 128
IMG_W = 1080   # 실제 portrait 이미지 width
IMG_H = 1920   # 실제 portrait 이미지 height
# 확인할 비디오 프레임 번호 목록
CHECK_VID_FRAMES = [320, 335, 340, 355, 365, 375, 390, 400, 420, 430]
# ──────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

npz = np.load(NPZ_PATH, allow_pickle=True)
print("handedness:", str(npz["handedness"]))
print("img_size:", int(npz["img_height"]), "x", int(npz["img_width"]))
print("cam_f:", npz["cam_f"], "cam_center:", npz["cam_center"])

cam_f = float(npz["cam_f"][0, 0])
cx = float(npz["cam_center"][0, 0])
cy = float(npz["cam_center"][0, 1])

vertices = npz["vertices"]     # (128, 778, 3) — mirrored camera frame
joints_3d = npz["joints_3d"]  # (128, 21, 3)  — mirrored camera frame
joints_2d_hmp = npz["joints_2d"]  # (128, 21, 2) — mirrored pixel coords

def vid_f_to_hmp_t(vid_f):
    rel = vid_f - START_FRAME
    return int(round(rel * HMP_FRAMES / NUM_INPUT_FRAMES))

def project_mirrored_to_original(pts3d, f, cx, cy, W=1080):
    """pts3d: (N,3) mirrored camera frame → 원본 이미지 pixel (u, v)
    HMP는 portrait W=1080 기준으로 최적화. npz의 img_width=1920은 오기.
    """
    u_mir = pts3d[:, 0] / pts3d[:, 2] * f + cx
    v     = pts3d[:, 1] / pts3d[:, 2] * f + cy
    u_orig = W - u_mir  # 미러 반전
    return np.stack([u_orig, v], axis=1)

def jts2d_mirrored_to_original(jts2d, W=1080):
    """HMP joints_2d (미러 픽셀) → 원본 이미지 픽셀
    HMP npz의 img_width=1920은 HMP 내부 저장 오류. 실제 portrait W=1080.
    """
    out = jts2d.copy()
    out[:, 0] = W - out[:, 0]
    return out

def compute_axial_deg(poses_aa):
    R = Rotation.from_rotvec(poses_aa).as_matrix()
    eul = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    return eul[2]

print(f"\n{'='*60}")
print("좌손 HMP 투영 시각화 (원본 프레임)")
print(f"{'='*60}")

for vid_f in CHECK_VID_FRAMES:
    hmp_t = vid_f_to_hmp_t(vid_f)
    hmp_t = max(0, min(HMP_FRAMES - 1, hmp_t))

    frame_path = os.path.join(FRAME_DIR, f"{vid_f:04d}.jpg")
    if not os.path.exists(frame_path):
        print(f"[SKIP] 프레임 없음: {frame_path}")
        continue

    img = cv2.imread(frame_path)
    if img is None:
        print(f"[SKIP] 이미지 로드 실패")
        continue

    verts = vertices[hmp_t]       # (778, 3) mirrored
    jts3d = joints_3d[hmp_t]     # (21, 3) mirrored
    jts2d_mir = joints_2d_hmp[hmp_t]  # (21, 2) mirrored

    # 원본 이미지 좌표로 변환
    pts2d_v = project_mirrored_to_original(verts, cam_f, cx, cy, IMG_W)
    pts2d_j = project_mirrored_to_original(jts3d, cam_f, cx, cy, IMG_W)
    pts2d_jraw = jts2d_mirrored_to_original(jts2d_mir, IMG_W)  # HMP joints_2d 미러 반전

    axial = compute_axial_deg(npz["poses"][hmp_t, :3])

    vis = img.copy()

    # vertices 점 (파란색, sparse)
    step = max(1, len(verts) // 150)
    for i in range(0, len(verts), step):
        u, v = int(pts2d_v[i, 0]), int(pts2d_v[i, 1])
        if 0 <= u < IMG_W and 0 <= v < IMG_H:
            cv2.circle(vis, (u, v), 2, (255, 80, 0), -1)

    # joints (초록색) — joints_3d 투영
    for i, (u, v) in enumerate(pts2d_j):
        u, v = int(u), int(v)
        if 0 <= u < IMG_W and 0 <= v < IMG_H:
            cv2.circle(vis, (u, v), 5, (0, 255, 0), -1)
            cv2.putText(vis, str(i), (u+3, v-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # HMP joints_2d (빨간색, 미러 반전)
    for i, (u, v) in enumerate(pts2d_jraw):
        u, v = int(u), int(v)
        if 0 <= u < IMG_W and 0 <= v < IMG_H:
            cv2.circle(vis, (u, v), 4, (0, 0, 255), -1)

    wrist_u, wrist_v = int(pts2d_j[0, 0]), int(pts2d_j[0, 1])
    cv2.circle(vis, (wrist_u, wrist_v), 10, (0, 255, 255), 2)
    cv2.putText(vis, "L-wrist", (wrist_u+5, wrist_v+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(vis, f"vid_f={vid_f} hmp_t={hmp_t} axial≈{axial:.1f}° [LEFT]",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(vis, "BLUE=vertices  GREEN=joints_3d→orig  RED=joints_2d→orig",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    out_path = os.path.join(OUT_DIR, f"left_f{vid_f:04d}_hmp{hmp_t:03d}.jpg")
    cv2.imwrite(out_path, vis)
    print(f"vid_f={vid_f} hmp_t={hmp_t:3d} | L-wrist=({wrist_u},{wrist_v}) | axial≈{axial:+.1f}° | -> {os.path.basename(out_path)}")

print(f"\n출력: {OUT_DIR}")
