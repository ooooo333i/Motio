"""
cap_compare_hmp_modes.py
전체 HMP vs 선택적 HMP 비교용 3개 버전 생성.

공통: base=smplx_arm_prefit.pt(ViTPose 팔, 고정), 손가락=HaMeR 후처리(동일).
세 버전 차이는 wrist 처리 방식만:
  - baseline : HaMeR 손목(Stage A: G_e^-1 @ HaMeR wrist_R) 전 구간
  - full     : HMP 손목 전 구간
  - selective: 정상=HaMeR, 튐 구간=HMP, 경계 SLERP blend

튐 검출: wrist local velocity > 33°/f (physical_limits) ∪ bbox 오탐(bad_frames_full).
경계 blend: bad run 가장자리 ±BLEND 프레임 선형 ramp로 quaternion slerp.
"""
import os, sys, json
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, '/workspace/capstone_deploy')
from cap_wrist_refine import load_hamer_detections_full
from physical_limits import MAX_WRIST_DEG_PER_FRAME
from entry_point_hamer import postprocess_hand_sequence, HandPostprocessConfig

IN = Path('/workspace/capstone_deploy/outputs/cap_pipeline/input')
BASE = IN / 'smplx_arm_prefit.pt'
NPZ_R = Path('/workspace/hmp_test/optim/in_the_wild_sample_config/in_the_wild/tennis_full/recon_000_30fps.npz')
NPZ_L = Path('/workspace/hmp_test/optim/in_the_wild_sample_config/in_the_wild/tennis_full_left/recon_000_30fps.npz')
HAMER = IN / 'hamer_out_clean'
F = 1495
RIGHT_CHAIN = [2, 5, 8, 13, 16, 18]; LEFT_CHAIN = [2, 5, 8, 12, 15, 17]
R_WR, L_WR = 20, 19
M = np.diag([-1., 1., 1.]).astype(np.float64)
BLEND = 4          # 경계 ramp 프레임 수
CONTEXT = 3        # bad run 확장 (양쪽)

def aaR(a): return Rotation.from_rotvec(a).as_matrix()
def R2aa(R): return Rotation.from_matrix(R).as_rotvec()

def fk_elbow(go, bp, chain):
    G = aaR(go)
    for i in chain: G = G @ aaR(bp[i])
    return G

def smplx_means():
    import smplx
    m = smplx.create('/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz',
        model_type='smplx', gender='neutral', use_pca=False, flat_hand_mean=False, num_betas=10)
    return m.left_hand_mean.detach().numpy(), m.right_hand_mean.detach().numpy()

def pp_fingers(det, side):
    cfg = HandPostprocessConfig()
    dd = {f: det[side][f]['hand_pose'].reshape(45) for f in range(F) if det[side].get(f) is not None}
    proc, _ = postprocess_hand_sequence(dd, np.zeros((F,45),np.float32), cfg)
    return proc  # (F,45) side-convention

def detect_bad(side, det, Gel_fn, chain):
    """wrist local velocity>33°/f ∪ bbox 오탐. 반환 bool (F,)."""
    JIT = MAX_WRIST_DEG_PER_FRAME
    prev=None; vel=np.zeros(F)
    for f in range(F):
        d=det[side].get(f)
        if d is None: prev=None; continue
        Rl=Gel_fn(f,chain).T@d['wrist_R']
        if prev is not None:
            vel[f]=np.degrees(np.linalg.norm(Rotation.from_matrix(prev.T@Rl).as_rotvec()))
        prev=Rl
    bad = vel > JIT
    tag = 'R' if side=='right' else 'L'
    bbox=set(json.load(open(IN/f'bad_frames_full_{tag}.json')))
    for b in bbox:
        if 0<=b<F: bad[b]=True
    # context 확장
    out=bad.copy()
    idx=np.where(bad)[0]
    for b in idx:
        for d in range(-CONTEXT,CONTEXT+1):
            if 0<=b+d<F: out[b+d]=True
    return out

def blend_weight(bad):
    """bad=1 → HMP(w=1), 정상=HaMeR(w=0), 경계 BLEND 선형 ramp."""
    w = bad.astype(np.float64).copy()
    # 거리 변환 기반 ramp: 각 정상↔bad 경계에서 BLEND 프레임 선형
    from scipy.ndimage import distance_transform_edt
    # bad 외부에서 bad까지 거리
    dist_out = distance_transform_edt(~bad)
    ramp = np.clip(1.0 - dist_out/(BLEND+1), 0, 1)
    w = np.maximum(w, ramp)
    return w

def slerp_pair(Ra, Rb, t):
    """두 회전 Ra,Rb를 t로 slerp (t=0→Ra, 1→Rb). 단일."""
    key=Slerp([0,1], Rotation.from_matrix(np.stack([Ra,Rb])))
    return key([t]).as_matrix()[0]

def main():
    det = load_hamer_detections_full(HAMER, F)
    npz_r = np.load(NPZ_R, allow_pickle=True); npz_l = np.load(NPZ_L, allow_pickle=True)
    poses_r = npz_r['poses']; poses_l = npz_l['poses']
    lmean, rmean = smplx_means()

    base = torch.load(BASE, map_location='cpu')
    inc = base['smpl_params_incam']; glb = base['smpl_params_global']
    bp = inc['body_pose'].numpy().reshape(F,21,3); go = inc['global_orient'].numpy()
    def Gel(f,chain): return fk_elbow(go[f], bp[f], chain)

    # 손가락 (pp, 동일)
    rpp = pp_fingers(det,'right'); lpp = pp_fingers(det,'left')

    # 프레임별 wrist local: HaMeR / HMP (R, L)
    def wrist_locals(side, chain, wr_idx):
        ham=np.tile(bp[:,wr_idx],(1,1)).astype(np.float64)  # 기본 base wrist (검출 없을 때)
        hmp=ham.copy()
        poses = poses_r if side=='right' else poses_l
        for f in range(F):
            Ge=Gel(f,chain)
            d=det[side].get(f)
            if d is not None:
                ham[f]=R2aa(Ge.T@d['wrist_R'])
            # HMP
            Rg=aaR(poses[f,:3].astype(np.float64))
            if side=='left': Rg=M@Rg@M
            hmp[f]=R2aa(Ge.T@Rg)
        return ham, hmp

    ham_R, hmp_R = wrist_locals('right', RIGHT_CHAIN, R_WR)
    ham_L, hmp_L = wrist_locals('left',  LEFT_CHAIN,  L_WR)

    bad_R = detect_bad('right', det, Gel, RIGHT_CHAIN)
    bad_L = detect_bad('left',  det, Gel, LEFT_CHAIN)
    wR = blend_weight(bad_R); wL = blend_weight(bad_L)
    print(f'[detect] R bad(+context)={int(bad_R.sum())} ({100*bad_R.mean():.0f}%), L bad={int(bad_L.sum())} ({100*bad_L.mean():.0f}%)')

    def sel(ham, hmp, w):
        out=ham.copy()
        for f in range(F):
            if w[f]<=0: out[f]=ham[f]
            elif w[f]>=1: out[f]=hmp[f]
            else:
                Ra=aaR(ham[f]); Rb=aaR(hmp[f])
                out[f]=R2aa(slerp_pair(Ra,Rb,w[f]))
        return out
    selR=sel(ham_R,hmp_R,wR); selL=sel(ham_L,hmp_L,wL)

    def build(name, wrR, wrL):
        import copy
        r=copy.deepcopy(base)
        for sp in ('smpl_params_incam','smpl_params_global'):
            p=r[sp]
            b=p['body_pose'].numpy().reshape(F,21,3).copy()
            b[:,R_WR]=wrR; b[:,L_WR]=wrL
            p['body_pose']=torch.from_numpy(b)
            p['right_hand_pose']=torch.from_numpy((rpp-rmean).astype(np.float32))
            p['left_hand_pose'] =torch.from_numpy((lpp-lmean).astype(np.float32))
        out=IN/name; torch.save(r,out); print('saved',out)

    # ★ LIVE: cmp_baseline_hamer.pt = adaptive_v3 파이프라인의 base (arm-prefit + HaMeR손목 + pp손가락).
    #   wrist_pipeline/01_fit_faithful.py 가 이 파일을 읽음. 재현에 필수.
    build('cmp_baseline_hamer.pt', ham_R, ham_L)
    # DEAD (비교용 산출, adaptive_v3 미사용): HMP wrist 모드. 삭제하지 말 것(폐기기록).
    build('cmp_full_hmp.pt',       hmp_R, hmp_L)
    build('cmp_selective_hmp.pt',  selR, selL)

    # 수치 리포트
    def geo(a,b):
        return np.degrees(np.linalg.norm(Rotation.from_matrix(aaR(a).T@aaR(b)).as_rotvec()))
    normal=~bad_R
    dfull=np.array([geo(ham_R[f],hmp_R[f]) for f in range(F)])
    dsel =np.array([geo(ham_R[f],selR[f]) for f in range(F)])
    print(f'[수치 R] 정상구간 HaMeR와 차이: full={dfull[normal].mean():.1f}°, selective={dsel[normal].mean():.1f}° (selective가 작아야 보존)')
    print(f'[수치 R] 튐구간 HaMeR와 차이: full={dfull[bad_R].mean():.1f}°, selective={dsel[bad_R].mean():.1f}° (교정량)')
    # axial velocity
    def axvel(wr):
        v=[np.degrees(np.linalg.norm(Rotation.from_matrix(aaR(wr[f-1]).T@aaR(wr[f])).as_rotvec())) for f in range(1,F)]
        return np.array(v)
    for nm,wr in [('HaMeR',ham_R),('full',hmp_R),('selective',selR)]:
        v=axvel(wr); print(f'[velocity R] {nm}: mean={v.mean():.1f} p95={np.percentile(v,95):.1f} max={v.max():.1f} >33:{int((v>33).sum())}')

if __name__=='__main__':
    main()
