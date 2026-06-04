"""
batch_arm_smooth: arm_prefit 팔 회전 후처리 평활 (일반화 누락분 복구).
 cap_arm_prefit엔 L_tc(최적화 내부)만 있고 후처리 평활이 없어, csv 게이팅 없는 클립은
 안 보이는 구간(저conf)에서 손목 위치가 슉슉 떨림. → conf 게이트 평활로 제거.
 - 항상 가벼운 σ_lo 평활 + 저conf(가려짐)에선 σ_hi로 더 세게 (slerp 블렌드).
 - 잘 보이는 구간(고conf)은 거의 안 건드림(실모션 보존).
 ⚠️ 팔(collar/shoulder/elbow)만. 손목/손가락 무관.
입력: <clip>/smplx_arm_prefit.pt, preprocess/vitpose.pt
출력: <clip>/smplx_arm_prefit.pt (in-place, 백업 _presmooth)
usage: python batch_arm_smooth.py <clip_dir> [sig_lo=1.0] [sig_hi=3.0]
"""
import sys, numpy as np, copy, shutil, os
import torch
from scipy.spatial.transform import Rotation as Rt
from scipy.ndimage import gaussian_filter1d
clip=sys.argv[1].rstrip('/'); SIG_LO=float(sys.argv[2]) if len(sys.argv)>2 else 1.0
SIG_HI=float(sys.argv[3]) if len(sys.argv)>3 else 3.0
pt=f'{clip}/smplx_arm_prefit.pt'; bak=f'{clip}/smplx_arm_prefit_presmooth.pt'
if not os.path.exists(bak): shutil.copy(pt,bak)
src=torch.load(bak,map_location='cpu')
vp=torch.load(f'{clip}/preprocess/vitpose.pt',map_location='cpu').numpy()  # (F,17,3) COCO17
F=src['smpl_params_incam']['body_pose'].reshape(-1,21,3).shape[0]
# 저conf 게이트 (손목 9/10 conf 낮음 = 가려짐) per side
def gate(coco):
    cf=vp[:,coco,2]; w=np.clip((0.6-cf)/0.4,0,1); return gaussian_filter1d(w,2.0)
wL=gate(9); wR=gate(10)
def smooth_q(aa,sig):
    q=Rt.from_rotvec(aa).as_quat()
    for f in range(1,F):
        if np.dot(q[f],q[f-1])<0: q[f]=-q[f]
    qs=np.stack([gaussian_filter1d(q[:,i],sig,mode='nearest') for i in range(4)],1); qs/=np.linalg.norm(qs,axis=1,keepdims=True)
    return qs
def slerp_q(qa,qb,w):  # per-frame slerp
    out=np.zeros_like(qa)
    for f in range(F):
        d=np.dot(qa[f],qb[f]); qbb=qb[f]*(1 if d>=0 else -1); d=abs(d)
        if d>0.9995: r=qa[f]+w[f]*(qbb-qa[f])
        else:
            th=np.arccos(d); s=np.sin(th); r=(np.sin((1-w[f])*th)*qa[f]+np.sin(w[f]*th)*qbb)/s
        out[f]=r/np.linalg.norm(r)
    return out
ROWS={'R':([13,16,18],wR),'L':([12,15,17],wL)}
out=copy.deepcopy(src)
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp not in src: continue
    bp=src[sp]['body_pose'].reshape(F,21,3).numpy().copy()
    for side,(rows,w) in ROWS.items():
        if sp=='smpl_params_incam': print(f'{side}팔: 저conf 평활 강화 {int((w>0.3).sum())}f')
        for j in rows:
            q_lo=smooth_q(bp[:,j],SIG_LO)          # 항상 가벼운 평활
            q_hi=smooth_q(bp[:,j],SIG_HI)          # 가려짐용 강한 평활
            q=slerp_q(q_lo,q_hi,w)                  # 저conf→강, 고conf→약
            bp[:,j]=Rt.from_quat(q).as_rotvec()
    out[sp]['body_pose']=torch.from_numpy(bp).float().reshape(out[sp]['body_pose'].shape)
torch.save(out,pt)
print(f'[arm_smooth] saved {pt} (σ_lo={SIG_LO} 기본 + σ_hi={SIG_HI} 가려짐)')
