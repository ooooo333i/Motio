"""
우측 손 HMP 결과 시각 확인 스크립트
- HMP vertices + joints_3d를 원본 프레임에 투영
- f355 등 flip 구간에서 실제로 손이 정상인지 확인
- axial 수치(-27.8°)와 화면이 일치하는지 검증
"""
import numpy as np
import cv2
import os
import sys

# ── 설정 ──────────────────────────────────────────────────
NPZ_PATH = "/workspace/hmp_test/optim/in_the_wild_sample_config/in_the_wild/tennis_swing/recon_000_30fps.npz"
FRAME_DIR = "/workspace/capstone_deploy/outputs/cap_pipeline/input/hamer_frames"
OUT_DIR = "/workspace/capstone_deploy/outputs/vis_hmp_right"
START_FRAME = 320  # frame_info.txt
NUM_INPUT_FRAMES = 111  # f320~f430 = 111 frames
HMP_FRAMES = 128

# 확인할 비디오 프레임 번호 목록
CHECK_VID_FRAMES = [320, 335, 340, 355, 365, 375, 390, 400, 420, 430]
# ──────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

npz = np.load(NPZ_PATH, allow_pickle=True)
print("keys:", list(npz.keys()))
print("handedness:", str(npz["handedness"]))
print("poses shape:", npz["poses"].shape)
print("vertices shape:", npz["vertices"].shape)
print("joints_3d shape:", npz["joints_3d"].shape)
print("joints_2d shape:", npz["joints_2d"].shape)
print("cam_f:", npz["cam_f"])
print("cam_center:", npz["cam_center"])
print("img_size:", int(npz["img_height"]), "x", int(npz["img_width"]))

cam_f = float(npz["cam_f"][0, 0])  # fx = fy
cx = float(npz["cam_center"][0, 0])
cy = float(npz["cam_center"][0, 1])
H = int(npz["img_height"])
W = int(npz["img_width"])

vertices = npz["vertices"]   # (128, 778, 3) — camera frame
joints_3d = npz["joints_3d"]  # (128, 21, 3)
joints_2d_hmp = npz["joints_2d"]  # (128, 21, 2) — HMP 자체 투영

def vid_f_to_hmp_t(vid_f):
    """원본 비디오 프레임 번호 → HMP 프레임 인덱스"""
    rel = vid_f - START_FRAME
    return int(round(rel * HMP_FRAMES / NUM_INPUT_FRAMES))

def project_pts(pts3d, f, cx, cy):
    """pts3d: (N,3) camera frame → 2D pixels"""
    x = pts3d[:, 0] / pts3d[:, 2] * f + cx
    y = pts3d[:, 1] / pts3d[:, 2] * f + cy
    return np.stack([x, y], axis=1)

def compute_axial_deg(poses_aa):
    """axis-angle (3,) → MANO wrist global_orient → axial rotation around Z estimate
    HMP document에서 axial = Z-axis rotation of the global_orient 사용
    여기서는 global_orient의 Z component magnitude × sign을 사용
    """
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(poses_aa).as_matrix()
    # wrist axial = Z-axis rotation in wrist local frame
    # 단순 추정: global_orient에서 hand-plane normal의 Y-component
    # 실제로는 FK 필요하지만 여기서는 HMP global orient의 axial 성분만 표시
    # R @ [0,0,1] = hand Z-direction in camera space
    # 대신 rotation matrix의 Euler ZYX 마지막 성분 사용
    eul = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    return eul[2]  # Z rotation (axial-ish, approximate)

print(f"\n{'='*60}")
print("프레임별 HMP 투영 시각화")
print(f"{'='*60}")

for vid_f in CHECK_VID_FRAMES:
    hmp_t = vid_f_to_hmp_t(vid_f)
    hmp_t = max(0, min(HMP_FRAMES - 1, hmp_t))

    # 원본 프레임 로드
    frame_path = os.path.join(FRAME_DIR, f"{vid_f:04d}.jpg")
    if not os.path.exists(frame_path):
        print(f"[SKIP] 프레임 없음: {frame_path}")
        continue

    img = cv2.imread(frame_path)
    if img is None:
        print(f"[SKIP] 이미지 로드 실패: {frame_path}")
        continue

    # HMP vertices 투영 (sparse: 1/5만 표시)
    verts = vertices[hmp_t]  # (778, 3)
    pts2d_v = project_pts(verts, cam_f, cx, cy)

    # HMP joints 투영
    jts3d = joints_3d[hmp_t]  # (21, 3)
    pts2d_j = project_pts(jts3d, cam_f, cx, cy)

    # HMP 자체 joints_2d (이미 계산된 것)
    jts2d_hmp = joints_2d_hmp[hmp_t]  # (21, 2)

    # axial 계산
    pose_aa = npz["poses"][hmp_t, :3]
    axial = compute_axial_deg(pose_aa)

    # 이미지 위에 그리기
    vis = img.copy()

    # vertices 점 (파란색, sparse)
    step = max(1, len(verts) // 150)
    for i in range(0, len(verts), step):
        u, v = int(pts2d_v[i, 0]), int(pts2d_v[i, 1])
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(vis, (u, v), 2, (255, 80, 0), -1)

    # joints 점 (초록색) — projected from joints_3d
    for i, (u, v) in enumerate(pts2d_j):
        u, v = int(u), int(v)
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(vis, (u, v), 5, (0, 255, 0), -1)
            cv2.putText(vis, str(i), (u+3, v-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # HMP 자체 joints_2d (빨간색)
    for i, (u, v) in enumerate(jts2d_hmp):
        u, v = int(u), int(v)
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(vis, (u, v), 4, (0, 0, 255), -1)

    # 정보 텍스트
    cv2.putText(vis, f"vid_f={vid_f}  hmp_t={hmp_t}  axial≈{axial:.1f}deg",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(vis, f"BLUE=vertices  GREEN=joints_3d-proj  RED=joints_2d-hmp",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # wrist (joint 0) 위치 표시
    wrist_u, wrist_v = int(pts2d_j[0, 0]), int(pts2d_j[0, 1])
    cv2.circle(vis, (wrist_u, wrist_v), 10, (0, 255, 255), 2)
    cv2.putText(vis, "wrist", (wrist_u+5, wrist_v+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    out_path = os.path.join(OUT_DIR, f"right_f{vid_f:04d}_hmp{hmp_t:03d}.jpg")
    cv2.imwrite(out_path, vis)
    print(f"vid_f={vid_f} hmp_t={hmp_t:3d} | wrist=({wrist_u},{wrist_v}) | axial≈{axial:+.1f}° | -> {out_path}")

print(f"\n출력 디렉터리: {OUT_DIR}")
print("확인할 핵심 프레임: f355 (hmp_t=40) — -27.8° 검증")
