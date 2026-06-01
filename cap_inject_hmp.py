"""
cap_inject_hmp.py
HMP(Hand Motion Prior) 출력의 wrist global_orient를 SMPL-X body_pose[19/20]에 주입.

주의사항:
- hand_mean 중복: SMPL-X flat_hand_mean=False → stored = actual - hand_mean
  HMP hand_pose 주입 시 반드시 hand_mean을 뺄 것.
- 좌손: HMP는 미러 공간에서 최적화 → R_global_orient는 mirrored camera space
  원본 카메라 공간으로 변환 필요: R_orig = M @ R_mirror @ M (M=diag(-1,1,1))
- body_pose[19] = left wrist (SMPL-X joint 20)
- body_pose[20] = right wrist (SMPL-X joint 21)

FK chain:
  left:  global_orient @ bp[2] @ bp[5] @ bp[8] @ bp[12] @ bp[15] @ bp[17] → G_elbow_left
  right: global_orient @ bp[2] @ bp[5] @ bp[8] @ bp[13] @ bp[16] @ bp[18] → G_elbow_right

Local wrist rotation:
  R_local = G_elbow^{-1} @ R_hmp_global
  body_pose[f, wrist_idx] = rotmat_to_aa(R_local)
"""
import numpy as np
import torch
import os
from scipy.spatial.transform import Rotation

# ── 모드 ─────────────────────────────────────────────────
# MODE=swing: f320-430 단일 128윈도우 (round 매핑)
# MODE=full : 전체 1495프레임 직접 인덱싱 (14윈도우 병합 결과)
MODE = os.environ.get("CAP_INJECT_MODE", "swing")

# ── 경로 ─────────────────────────────────────────────────
INPUT_PT = "/workspace/capstone_deploy/outputs/cap_pipeline/input/smplx_kalman_B.pt"
if MODE == "full":
    OUTPUT_PT = "/workspace/capstone_deploy/outputs/cap_pipeline/input/smplx_hmp_injected_full.pt"
    NPZ_RIGHT = "/workspace/hmp_test/optim/in_the_wild_sample_config/in_the_wild/tennis_full/recon_000_30fps.npz"
    NPZ_LEFT  = "/workspace/hmp_test/optim/in_the_wild_sample_config/in_the_wild/tennis_full_left/recon_000_30fps.npz"
    SWING_START = 0
    SWING_END   = 1494
    NUM_INPUT_FRAMES = 1495
    HMP_FRAMES = 1495   # 전체는 직접 인덱싱 (hmp_t = vid_f)
else:
    OUTPUT_PT = "/workspace/capstone_deploy/outputs/cap_pipeline/input/smplx_hmp_injected.pt"
    NPZ_RIGHT = "/workspace/hmp_test/optim/in_the_wild_sample_config/in_the_wild/tennis_swing/recon_000_30fps.npz"
    NPZ_LEFT  = "/workspace/hmp_test/optim/in_the_wild_sample_config/in_the_wild/tennis_swing_left/recon_000_30fps.npz"
    # 스윙 구간: 원본 비디오 f320~f430 (111 frames)
    SWING_START = 320
    SWING_END   = 430
    NUM_INPUT_FRAMES = SWING_END - SWING_START + 1  # 111
    HMP_FRAMES = 128

# SMPL-X kintree (위에서 확인한 값)
# body_pose[i] = SMPL-X joint (i+1). root=joint0 제외.
LEFT_WRIST_BP  = 19   # joint 20
RIGHT_WRIST_BP = 20   # joint 21
# FK chain to elbow (global_orient 포함, wrist 제외)
LEFT_CHAIN_BP  = [2, 5, 8, 12, 15, 17]   # to elbow (joint 18)
RIGHT_CHAIN_BP = [2, 5, 8, 13, 16, 18]   # to elbow (joint 19)

# M = diag(-1,1,1) for X-mirror
M = np.diag([-1., 1., 1.]).astype(np.float64)


def aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """axis-angle (3,) → rotation matrix (3,3)"""
    return Rotation.from_rotvec(aa).as_matrix()


def rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    """rotation matrix (3,3) → axis-angle (3,)"""
    return Rotation.from_matrix(R).as_rotvec()


def compute_fk_elbow(global_orient_aa: np.ndarray,
                     body_pose_aa: np.ndarray,
                     chain_bp: list) -> np.ndarray:
    """
    global_orient_aa: (3,) axis-angle
    body_pose_aa: (21, 3) or (63,) axis-angle
    chain_bp: list of body_pose indices from pelvis to elbow
    Returns: G_elbow (3,3) rotation matrix in camera space
    """
    body_pose_aa = body_pose_aa.reshape(21, 3)
    G = aa_to_rotmat(global_orient_aa)
    for bp_idx in chain_bp:
        G = G @ aa_to_rotmat(body_pose_aa[bp_idx])
    return G


def vid_f_to_hmp_t(vid_f: int) -> int:
    """원본 비디오 프레임 번호 → HMP 프레임 인덱스.
    full 모드: HMP 출력이 1495프레임 직접 인덱싱 → hmp_t = vid_f.
    swing 모드: 111프레임이 128윈도우로 padding → round 매핑.
    """
    if MODE == "full":
        return max(0, min(HMP_FRAMES - 1, vid_f))
    rel = vid_f - SWING_START
    t = int(round(rel * HMP_FRAMES / NUM_INPUT_FRAMES))
    return max(0, min(HMP_FRAMES - 1, t))


def inject_wrist_from_hmp(pt_data: dict,
                           npz_right: np.lib.npyio.NpzFile,
                           npz_left: np.lib.npyio.NpzFile,
                           inject_hand_pose: bool = True) -> dict:
    """
    wrist(body_pose[19/20])  ← HMP poses[:,:3]  (HaMeR보다 정확)
    hand_pose(그립)           ← HaMeR 원본       (HMP prior가 테니스 그립을 망가뜨림)

    핵심: R_local 계산은 INCAM 공간에서만 수행.
    body_pose는 local rotation이므로 global/incam에 동일값 적용.
    hand_mean 중복 주의: SMPL-X pose2rot=True → stored = actual - hand_mean
    """
    import copy, sys as _sys
    _sys.path.insert(0, '/workspace/capstone_deploy')
    from pathlib import Path as _Path
    from cap_wrist_refine import load_hamer_detections_full

    result = copy.deepcopy(pt_data)

    # HMP: wrist 용 (poses[:, :3] 만 사용)
    poses_right = npz_right["poses"]   # (128, 48)
    poses_left  = npz_left["poses"]    # (128, 48)

    # HaMeR 원본: hand_pose 용
    _hamer_out_dir = _Path('/workspace/capstone_deploy/outputs/cap_pipeline/input/hamer_out_clean')
    det_all = load_hamer_detections_full(_hamer_out_dir, SWING_END + 10)
    # det_all['right'][vid_f]['hand_pose'] : (45,) axis-angle, already mirrored for left

    import smplx as _smplx
    _smplx_model = _smplx.create(
        '/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz',
        model_type='smplx',
        gender='neutral',
        use_pca=False,
        flat_hand_mean=False,
        num_betas=10,
    )
    left_hand_mean  = _smplx_model.left_hand_mean.detach().numpy()   # (45,)
    right_hand_mean = _smplx_model.right_hand_mean.detach().numpy()  # (45,)

    # INCAM 공간에서 R_local 계산
    params_incam = result["smpl_params_incam"]
    bp_incam  = params_incam["body_pose"].numpy().copy()    # (1495, 21, 3)
    go_incam  = params_incam["global_orient"].numpy().copy()  # (1495, 3)
    lhp_incam = params_incam["left_hand_pose"].numpy().copy()   # (1495, 45)
    rhp_incam = params_incam["right_hand_pose"].numpy().copy()  # (1495, 45)

    params_global = result["smpl_params_global"]
    bp_global  = params_global["body_pose"].numpy().copy()
    lhp_global = params_global["left_hand_pose"].numpy().copy()
    rhp_global = params_global["right_hand_pose"].numpy().copy()

    for vid_f in range(SWING_START, SWING_END + 1):
        hmp_t = vid_f_to_hmp_t(vid_f)

        # ── 우측 손 (body_pose[20]) ─────────────────────────────
        R_hmp_right = aa_to_rotmat(poses_right[hmp_t, :3].astype(np.float64))
        G_elbow_right = compute_fk_elbow(go_incam[vid_f], bp_incam[vid_f], RIGHT_CHAIN_BP)
        R_local_right = G_elbow_right.T @ R_hmp_right
        aa_right = rotmat_to_aa(R_local_right).astype(np.float32)
        # 동일 local rotation → global/incam 모두 적용
        bp_incam[vid_f, RIGHT_WRIST_BP] = aa_right
        bp_global[vid_f, RIGHT_WRIST_BP] = aa_right

        # ── 좌측 손 (body_pose[19]) ─────────────────────────────
        # HMP 좌손: mirrored camera space → 원본 incam 공간으로 복원
        R_hmp_left_mirror = aa_to_rotmat(poses_left[hmp_t, :3].astype(np.float64))
        R_hmp_left = M @ R_hmp_left_mirror @ M   # 원본(incam) 카메라 공간
        G_elbow_left = compute_fk_elbow(go_incam[vid_f], bp_incam[vid_f], LEFT_CHAIN_BP)
        R_local_left = G_elbow_left.T @ R_hmp_left
        aa_left = rotmat_to_aa(R_local_left).astype(np.float32)
        bp_incam[vid_f, LEFT_WRIST_BP] = aa_left
        bp_global[vid_f, LEFT_WRIST_BP] = aa_left

        # ── hand_pose: HaMeR 원본 사용 (HMP prior가 테니스 그립을 망가뜨림) ─
        if inject_hand_pose:
            # 우측 hand_pose: HaMeR 원본 (vid_f 직접 인덱스, HMP 매핑 불필요)
            d_right = det_all['right'].get(vid_f)
            if d_right is not None:
                # det_all['right'][f]['hand_pose'] = (45,) 이미 original convention
                # SMPL-X stored = actual - hand_mean
                rhp_val = d_right['hand_pose'].astype(np.float32) - right_hand_mean
                rhp_incam[vid_f] = rhp_val
                rhp_global[vid_f] = rhp_val
            # 우측 검출 없으면 kalman_B 값 그대로 유지

            # 좌측 hand_pose: HaMeR 원본
            # det_all['left'][f]['hand_pose'] = mirror 이미 적용된 left-hand convention
            d_left = det_all['left'].get(vid_f)
            if d_left is not None:
                lhp_val = d_left['hand_pose'].astype(np.float32) - left_hand_mean
                lhp_incam[vid_f] = lhp_val
                lhp_global[vid_f] = lhp_val
            # 좌측 검출 없으면 kalman_B 값 그대로 유지

    params_incam["body_pose"] = torch.from_numpy(bp_incam)
    params_global["body_pose"] = torch.from_numpy(bp_global)
    if inject_hand_pose:
        params_incam["left_hand_pose"]   = torch.from_numpy(lhp_incam)
        params_incam["right_hand_pose"]  = torch.from_numpy(rhp_incam)
        params_global["left_hand_pose"]  = torch.from_numpy(lhp_global)
        params_global["right_hand_pose"] = torch.from_numpy(rhp_global)

    print(f"주입 완료: {SWING_END - SWING_START + 1} frames × 2 spaces (global+incam)")
    return result


def main():
    print("=" * 60)
    print("cap_inject_hmp.py")
    print("=" * 60)
    print(f"입력: {INPUT_PT}")
    print(f"우측 HMP: {NPZ_RIGHT}")
    print(f"좌측 HMP: {NPZ_LEFT}")

    # 입력 로드
    pt_data = torch.load(INPUT_PT, map_location="cpu")
    npz_right = np.load(NPZ_RIGHT, allow_pickle=True)
    npz_left  = np.load(NPZ_LEFT, allow_pickle=True)

    print(f"right handedness: {str(npz_right['handedness'])}")
    print(f"left  handedness: {str(npz_left['handedness'])}")
    print(f"스윙 구간: f{SWING_START}~f{SWING_END} ({NUM_INPUT_FRAMES} frames)")

    # 주입 전 우측 손목 각도 확인
    bp_before = pt_data["smpl_params_incam"]["body_pose"][355]  # f355
    aa_r_before = bp_before[RIGHT_WRIST_BP].numpy()
    aa_l_before = bp_before[LEFT_WRIST_BP].numpy()
    print(f"\n[Before] f355 right wrist aa: {aa_r_before}")
    print(f"[Before] f355 left  wrist aa: {aa_l_before}")

    # 주입 실행
    result = inject_wrist_from_hmp(pt_data, npz_right, npz_left, inject_hand_pose=True)

    # 주입 후 확인
    bp_after = result["smpl_params_incam"]["body_pose"][355]
    aa_r_after = bp_after[RIGHT_WRIST_BP].numpy()
    aa_l_after = bp_after[LEFT_WRIST_BP].numpy()
    print(f"\n[After] f355 right wrist aa: {aa_r_after}")
    print(f"[After] f355 left  wrist aa: {aa_l_after}")
    print(f"  right delta_deg: {np.degrees(np.linalg.norm(aa_r_after - aa_r_before)):.1f}°")
    print(f"  left  delta_deg: {np.degrees(np.linalg.norm(aa_l_after - aa_l_before)):.1f}°")

    # global/incam 동기 확인 (local rotation이므로 동일해야 함)
    bp_g = result["smpl_params_global"]["body_pose"][355, RIGHT_WRIST_BP].numpy()
    bp_i = result["smpl_params_incam"]["body_pose"][355, RIGHT_WRIST_BP].numpy()
    diff = np.linalg.norm(bp_g - bp_i)
    print(f"\n[Sync check] global vs incam right wrist diff: {diff:.6f} (should be ~0)")

    # 저장
    result["cap_inject_hmp_meta"] = {
        "swing_start": SWING_START,
        "swing_end": SWING_END,
        "hmp_frames": HMP_FRAMES,
        "inject_hand_pose": True,
        "source_right": NPZ_RIGHT,
        "source_left": NPZ_LEFT,
    }

    os.makedirs(os.path.dirname(OUTPUT_PT), exist_ok=True)
    torch.save(result, OUTPUT_PT)
    print(f"\n저장 완료: {OUTPUT_PT}")
    print(f"파일 크기: {os.path.getsize(OUTPUT_PT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
