"""
graft_dynhamr_to_gvhmr.py  (2026-06-03)

비교군 D 생성: Dyn-HaMR(정적 GVHMR카메라 focal=2202로 재실행한 결과)의 손가락+손목을
GVHMR SMPL-X 바디(A와 동일 바디)에 이식(graft). "손목 그대로"(우리 axial 보정 없음).

근거/관례:
  - Dyn-HaMR(body_model/utils.py run_mano)는 우손 MANO 모델로 양손 처리 + 왼손은 출력 x반전
    → root_orient/pose_body 는 HaMeR와 동일한 '우손-정준' 프레임. → 우리 파이프라인의
      왼손 미러(rotmat M@R@M, 손가락 aa y,z 반전)를 그대로 적용.
  - Dyn-HaMR HMP MANO flat_hand_mean=True → pose_body 는 flat 기준 → SMPL-X(flat_hand_mean=False)
    주입 시 hand_mean 차감 (HaMeR 경로와 동일).
  - 카메라가 정적·focal=2202 = GVHMR incam 프레임과 동일 → R_cam_align=I. 손목 전역회전 직접 이식.

프레임 대응(확정): 입력클립 1495f. cap_win = 입력클립 [700:1151](오프셋700, MSE매칭).
  Dyn-HaMR 트랙구간 (7,449) → world_results frame j(0..441) = cap_win (7+j) = 입력클립 (707+j).
  → 윈도우 = 입력클립 [707:1149] 442f. GVHMR/A 바디 슬라이스 동일.
"""
import sys, copy, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/workspace/capstone_deploy")
import cap_wrist_refine as cwr

_ap = argparse.ArgumentParser()
_ap.add_argument("--dyn", default="/workspace/Dyn-HaMR/outputs/logs/video-custom/2026-06-03/cap_win-all-shot-0-0--1/smooth_fit/cap_win_000300_world_results.npz")
_ap.add_argument("--base", default="/workspace/capstone_deploy/outputs/cap_pipeline/input/_archive/compare_pt/smplx_A_gvhmrwrist_ppfinger.pt")
_ap.add_argument("--out", default="/workspace/capstone_deploy/outputs/cap_pipeline/input/cmp_dynhamr_grafted.pt")
_ap.add_argument("--offset", type=int, default=707)
_ap.add_argument("--nwin", type=int, default=442)
_args = _ap.parse_args()
DYN = _args.dyn; A_PT = _args.base; OUT = _args.out
OFFSET = _args.offset      # base(GVHMR) 클립에서 Dyn-HaMR world_results frame 0 의 위치
NWIN   = _args.nwin        # Dyn-HaMR 프레임 수

def smplx_hand_mean_offsets():
    from hmr4d.utils.smplx_utils import make_smplx
    smplx = make_smplx("supermotion_fullhands")
    return {
        "left":  smplx.bm.left_hand_mean.detach().cpu().numpy().astype(np.float32).reshape(45),
        "right": smplx.bm.right_hand_mean.detach().cpu().numpy().astype(np.float32).reshape(45),
        "_model": smplx,
    }

def main():
    d = np.load(DYN)
    root_orient = d["root_orient"]   # (2,442,3) 손 전역회전(손목) axis-angle
    pose_body   = d["pose_body"]     # (2,442,15,3) MANO 손가락
    is_right    = d["is_right"]      # (2,442)  idx0=left(0), idx1=right(1)
    print(f"[dyn] root_orient {root_orient.shape} pose_body {pose_body.shape} is_right[:, 0]={is_right[:,0].tolist()}")
    assert root_orient.shape[1] == NWIN, f"기대 {NWIN}, 실제 {root_orient.shape[1]}"
    # idx0=left, idx1=right 확정
    li = 0 if is_right[0,0] == 0 else 1
    ri = 1 - li
    print(f"[dyn] left idx={li}, right idx={ri}")

    # --- detections dict (replace_wrists 포맷, window frame j 로 키) ---
    det = {"left": {}, "right": {}}
    hp_left  = np.zeros((NWIN,45), np.float32)
    hp_right = np.zeros((NWIN,45), np.float32)
    for j in range(NWIN):
        # LEFT (우손-정준 → 미러)
        Rl = cwr.aa_to_rotmat(torch.from_numpy(root_orient[li,j]).float()).numpy()
        Rl = cwr.mirror_rotmat_right_to_left(Rl)
        pl = cwr._mirror_hand_pose_aa(pose_body[li,j].reshape(45))
        det["left"][j]  = {"wrist_R": Rl.astype(np.float32)}
        hp_left[j] = pl
        # RIGHT (그대로)
        Rr = cwr.aa_to_rotmat(torch.from_numpy(root_orient[ri,j]).float()).numpy().astype(np.float32)
        pr = pose_body[ri,j].reshape(45).astype(np.float32)
        det["right"][j] = {"wrist_R": Rr}
        hp_right[j] = pr

    # --- GVHMR/A 바디 로드 + 윈도우 슬라이스 ---
    A = torch.load(A_PT, map_location="cpu")
    sp_full = A["smpl_params_incam"]
    F = sp_full["body_pose"].shape[0]
    print(f"[A] smpl_params_incam frames={F}, keys={list(sp_full.keys())}")
    sl = slice(OFFSET, OFFSET+NWIN)
    sp = {k: (v[sl].clone() if torch.is_tensor(v) and v.shape[0]==F else
              (v.clone() if torch.is_tensor(v) else copy.deepcopy(v))) for k,v in sp_full.items()}
    print(f"[win] sliced body_pose {tuple(sp['body_pose'].shape)} (입력클립 [{OFFSET}:{OFFSET+NWIN}])")

    # --- hand_mean ---
    hm = smplx_hand_mean_offsets()
    parents = np.asarray(hm["_model"].bm.parents.cpu())
    cwr.verify_wrist_chains(hm["_model"], sp)

    # --- 손목 이식 (R_w = G_e^T @ G_w, use_slerp=False = 그대로) ---
    sp = cwr.replace_wrists(sp, det, parents, num_frames=NWIN, use_slerp=False)

    # --- 손가락 주입 (Dyn-HaMR pose_body - hand_mean) ---
    sp["left_hand_pose"]  = torch.from_numpy(hp_left  - hm["left"].reshape(1,45)).float()
    sp["right_hand_pose"] = torch.from_numpy(hp_right - hm["right"].reshape(1,45)).float()

    # --- K 슬라이스 ---
    K = A["K_fullimg"]
    if torch.is_tensor(K) and K.shape[0]==F:
        K = K[sl].clone()
    out = {"smpl_params_incam": sp, "K_fullimg": K}
    torch.save(out, OUT)

    # --- 진단: 손목 로컬각/손가락 범위 ---
    bp = sp["body_pose"]
    def ang(idx):
        a = bp[:,idx,:]; return float(a.norm(dim=-1).mean()*180/np.pi)
    print(f"\n[OK] saved {OUT}")
    print(f"  좌손목 body_pose[19] mean|angle|={ang(19):.1f}°, 우손목[20]={ang(20):.1f}°")
    print(f"  left_hand_pose range [{sp['left_hand_pose'].min():.2f},{sp['left_hand_pose'].max():.2f}]"
          f"  right [{sp['right_hand_pose'].min():.2f},{sp['right_hand_pose'].max():.2f}]")
    print(f"  frames={bp.shape[0]} (=442), K={tuple(K.shape) if torch.is_tensor(K) else type(K)}")

if __name__ == "__main__":
    main()
