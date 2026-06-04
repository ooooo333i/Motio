"""
M2 (naive 베이스라인): GVHMR 몸 + 순수 HaMeR 손목 직접 병합 + 간단 Kalman 평활.
 ⚠️ 우리 핵심 아이디어(전완축 swing-twist 제한 / ROM clamp / conf 게이트) *전혀 없음* — 공정한 naive 비교용.
 - 손목 [19,20] = G_elbow⁻¹ @ HaMeR_wrist_global (full HaMeR, 미검출=GVHMR 손목 fallback)
 - Kalman 평활만 (게이트/슬러프/ROM 없음). 손가락 = HaMeR(_post 그대로). 팔 = GVHMR(_post 그대로).
출력: <clip>/baseline_m2_kalman.pt
usage: python build_m2_naive.py <clip_dir>
"""
import sys, numpy as np, copy
from pathlib import Path
sys.path.insert(0,'/workspace/capstone_deploy')
import torch
from scipy.spatial.transform import Rotation as Rt
from cap_wrist_refine import load_hamer_detections_full
from cap_kalman_wrist import kalman_smooth_rotation
clip=sys.argv[1].rstrip('/')
src=torch.load(f'{clip}/smplx_merged_hamer_post.pt',map_location='cpu')  # GVHMR 몸 + HaMeR 손가락 + GVHMR 손목
inc=src['smpl_params_incam']
bp=inc['body_pose'].reshape(-1,21,3).numpy().copy(); F=bp.shape[0]
# 팔 = arm_prefit(IK) — Ours와 동일 팔. HaMeR 손목이 IK 전완 기준이라 비틀림이 그대로 드러남(=smplx_kalman_B 재현).
arm=torch.load(f'{clip}/smplx_arm_prefit.pt',map_location='cpu')['smpl_params_incam']['body_pose'].reshape(F,21,3).numpy()
for j in [12,13,15,16,17,18]: bp[:,j]=arm[:,j]
go=inc['global_orient'].numpy()
if go.shape[0]==1: go=np.repeat(go,F,0)
det=load_hamer_detections_full(Path(f'{clip}/hamer_out_clean'),F)
RCH=[2,5,8,13,16,18]; LCH=[2,5,8,12,15,17]
def Gelbow(f,chain):
    Rm=Rt.from_rotvec(go[f]).as_matrix()
    for j in chain: Rm=Rm@Rt.from_rotvec(bp[f,j]).as_matrix()
    return Rm
out_bp=bp.copy()
nfill={}
for side,wr,chain in [('right',20,RCH),('left',19,LCH)]:
    new=bp[:,wr].copy()   # fallback = GVHMR 손목 (naive)
    n=0
    for f in range(F):
        if f not in det[side]: continue
        Rl=Gelbow(f,chain).T@det[side][f]['wrist_R']    # 순수 HaMeR 손목 local (제한 없음)
        new[f]=Rt.from_matrix(Rl).as_rotvec(); n+=1
    new=kalman_smooth_rotation(new,0.001,0.2,33.0,50.0)  # 옛 cap_kalman_wrist 동일 gate=33 (smplx_kalman_B 재현)
    out_bp[:,wr]=new; nfill[side]=n
out=copy.deepcopy(src)
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp in out:
        b=out[sp]['body_pose'].reshape(F,21,3).numpy().copy()
        for j in [12,13,15,16,17,18]: b[:,j]=bp[:,j]   # 팔=arm_prefit(IK, Ours와 동일)
        b[:,19]=out_bp[:,19]; b[:,20]=out_bp[:,20]      # 손목=HaMeR+Kalman(비틀림 남음)
        out[sp]['body_pose']=torch.from_numpy(b).float().reshape(out[sp]['body_pose'].shape)
torch.save(out,f'{clip}/baseline_m2_kalman.pt')
print(f'[M2 naive] HaMeR손목병합 R{nfill["right"]}f/L{nfill["left"]}f + Kalman, 전완축제한/게이트/ROM 없음')
print(f'[M2 naive] saved {clip}/baseline_m2_kalman.pt')
