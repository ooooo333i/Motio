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
# base(.pt) — 손목/팔/손가락의 출발점. CAP_INPUT_PT로 오버라이드.
#   기본 smplx_kalman_B.pt (arm-prefit + HaMeR 손목 + kalman)
#   버전 A(entry_point_hamer 스타일)는 hmr4d_results.pt (raw GVHMR 손목) 사용 권장.
INPUT_PT = os.environ.get("CAP_INPUT_PT",
    "/workspace/capstone_deploy/outputs/cap_pipeline/input/smplx_kalman_B.pt")
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
                           inject_hand_pose: bool = True,
                           inject_wrist: bool = True,
                           hand_postprocess: bool = True) -> dict:
    """
    wrist(body_pose[19/20])  ← HMP poses[:,:3]  (HaMeR보다 정확)   [inject_wrist=True일 때]
    hand_pose(그립)           ← HaMeR + 시계열 후처리 (entry_point_hamer와 동일)

    inject_wrist=False → wrist는 base(pt_data) 그대로 유지 (버전 A: 손목 최적화 X).
    hand_postprocess=True → postprocess_hand_sequence (temporal gate + outlier + fill + smooth)
       적용. False면 원본 HaMeR hand_pose 직접 사용 (구 동작).

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
    NF = pt_data["smpl_params_incam"]["body_pose"].shape[0]
    det_all = load_hamer_detections_full(_hamer_out_dir, NF)
    # det_all['right'][vid_f]['hand_pose'] : (45,) axis-angle, already mirrored for left

    # ── 손가락 시계열 후처리 (entry_point_hamer.postprocess_hand_sequence 복원) ──
    # 전체 프레임 시퀀스로 후처리 → 스무딩 temporal context 확보. 결과는 (NF,45).
    hp_processed = {"right": None, "left": None}
    if inject_hand_pose and hand_postprocess:
        from entry_point_hamer import postprocess_hand_sequence, HandPostprocessConfig
        _cfg = HandPostprocessConfig()
        for _side in ("right", "left"):
            _dd = {f: det_all[_side][f]['hand_pose'].reshape(45)
                   for f in range(NF) if det_all[_side].get(f) is not None}
            _fallback = np.zeros((NF, 45), dtype=np.float32)
            _proc, _ = postprocess_hand_sequence(_dd, _fallback, _cfg)
            hp_processed[_side] = _proc   # (NF,45) 후처리된 hand_pose (convention 유지)

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

    # body_pose가 (F,63)이면 (F,21,3)으로 변환 (raw GVHMR hmr4d_results.pt 대응)
    def _bp_to_213(t):
        a = t.numpy().copy()
        return a.reshape(a.shape[0], 21, 3) if a.ndim == 2 else a

    # INCAM 공간에서 R_local 계산
    params_incam = result["smpl_params_incam"]
    bp_incam  = _bp_to_213(params_incam["body_pose"])         # (F, 21, 3)
    go_incam  = params_incam["global_orient"].numpy().copy()  # (F, 3)
    lhp_incam = params_incam["left_hand_pose"].numpy().copy()   # (F, 45)
    rhp_incam = params_incam["right_hand_pose"].numpy().copy()  # (F, 45)

    params_global = result["smpl_params_global"]
    bp_global  = _bp_to_213(params_global["body_pose"])
    lhp_global = params_global["left_hand_pose"].numpy().copy()
    rhp_global = params_global["right_hand_pose"].numpy().copy()

    for vid_f in range(SWING_START, SWING_END + 1):
        hmp_t = vid_f_to_hmp_t(vid_f)

        # ── wrist 주입 (inject_wrist=True일 때만; False면 base 유지) ──
        if inject_wrist:
            # 우측 손 (body_pose[20])
            R_hmp_right = aa_to_rotmat(poses_right[hmp_t, :3].astype(np.float64))
            G_elbow_right = compute_fk_elbow(go_incam[vid_f], bp_incam[vid_f], RIGHT_CHAIN_BP)
            R_local_right = G_elbow_right.T @ R_hmp_right
            aa_right = rotmat_to_aa(R_local_right).astype(np.float32)
            bp_incam[vid_f, RIGHT_WRIST_BP] = aa_right
            bp_global[vid_f, RIGHT_WRIST_BP] = aa_right

            # 좌측 손 (body_pose[19]) — HMP 좌손: mirrored → 원본 incam 복원
            R_hmp_left_mirror = aa_to_rotmat(poses_left[hmp_t, :3].astype(np.float64))
            R_hmp_left = M @ R_hmp_left_mirror @ M
            G_elbow_left = compute_fk_elbow(go_incam[vid_f], bp_incam[vid_f], LEFT_CHAIN_BP)
            R_local_left = G_elbow_left.T @ R_hmp_left
            aa_left = rotmat_to_aa(R_local_left).astype(np.float32)
            bp_incam[vid_f, LEFT_WRIST_BP] = aa_left
            bp_global[vid_f, LEFT_WRIST_BP] = aa_left

        # ── hand_pose: HaMeR + 시계열 후처리 (hand_postprocess) 또는 원본 ──
        if inject_hand_pose:
            # 우측
            if hp_processed['right'] is not None:
                rhp_src = hp_processed['right'][vid_f]            # 후처리된 값
                has_r = det_all['right'].get(vid_f) is not None
            else:
                d_right = det_all['right'].get(vid_f)
                rhp_src = d_right['hand_pose'].astype(np.float32) if d_right is not None else None
                has_r = d_right is not None
            if has_r:
                rhp_val = rhp_src.astype(np.float32) - right_hand_mean
                rhp_incam[vid_f] = rhp_val
                rhp_global[vid_f] = rhp_val

            # 좌측 (det/hp_processed 모두 left-hand convention)
            if hp_processed['left'] is not None:
                lhp_src = hp_processed['left'][vid_f]
                has_l = det_all['left'].get(vid_f) is not None
            else:
                d_left = det_all['left'].get(vid_f)
                lhp_src = d_left['hand_pose'].astype(np.float32) if d_left is not None else None
                has_l = d_left is not None
            if has_l:
                lhp_val = lhp_src.astype(np.float32) - left_hand_mean
                lhp_incam[vid_f] = lhp_val
                lhp_global[vid_f] = lhp_val

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

    # 주입 전 우측 손목 각도 확인 (body_pose (F,63) 또는 (F,21,3) 대응)
    _bpb = pt_data["smpl_params_incam"]["body_pose"][355].numpy().reshape(21, 3)
    aa_r_before = _bpb[RIGHT_WRIST_BP]
    aa_l_before = _bpb[LEFT_WRIST_BP]
    print(f"\n[Before] f355 right wrist aa: {aa_r_before}")
    print(f"[Before] f355 left  wrist aa: {aa_l_before}")

    # 옵션 (환경변수):
    #   CAP_INJECT_WRIST=0 → wrist 미주입 (버전 A: 손목 최적화 X, base wrist 유지)
    #   CAP_HAND_POSTPROCESS=0 → 손가락 후처리 끔 (구 동작, 원본 HaMeR)
    #   CAP_OUTPUT_PT → 출력 경로 오버라이드
    inject_wrist = os.environ.get("CAP_INJECT_WRIST", "1") != "0"
    hand_pp = os.environ.get("CAP_HAND_POSTPROCESS", "1") != "0"
    out_pt = os.environ.get("CAP_OUTPUT_PT", OUTPUT_PT)
    print(f"옵션: inject_wrist={inject_wrist}  hand_postprocess={hand_pp}")

    # 주입 실행
    result = inject_wrist_from_hmp(pt_data, npz_right, npz_left,
                                   inject_hand_pose=True,
                                   inject_wrist=inject_wrist,
                                   hand_postprocess=hand_pp)

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
        "inject_wrist": inject_wrist,
        "hand_postprocess": hand_pp,
        "source_right": NPZ_RIGHT,
        "source_left": NPZ_LEFT,
    }

    os.makedirs(os.path.dirname(out_pt), exist_ok=True)
    torch.save(result, out_pt)
    print(f"\n저장 완료: {out_pt}")
    print(f"파일 크기: {os.path.getsize(out_pt) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
