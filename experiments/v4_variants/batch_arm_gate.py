"""
batch_arm_gate (v3, GVHMR 폴백): 가려짐(저conf)에선 팔을 GVHMR로 blend. 고conf는 arm_prefit(ViTPose 위치교정).
 사용자 관찰: GVHMR는 가려짐 중에도 손을 잘 추적(시계열 모델). arm_prefit은 가려짐 중 노이즈 ViTPose 따라가 나빠짐.
 → arm[f] = slerp(arm_prefit, GVHMR, w[f]),  w = 저conf 게이트 (occlusion=GVHMR, 보임=arm_prefit).
 + GVHMR 자체 점프(저크)는 그 위에 평활.
입력: <clip>/{smplx_arm_prefit_nogate.pt(=IK), smplx_merged_hamer_post.pt(=GVHMR)}, preprocess/wholebody_kp2d.npy
출력: <clip>/smplx_arm_prefit.pt
usage: python batch_arm_gate.py <clip_dir> [conf_hi=0.85] [conf_lo=0.55]
"""
import sys, numpy as np, copy, shutil, os
import torch
from scipy.spatial.transform import Rotation as Rt, Slerp
from scipy.ndimage import gaussian_filter1d
clip=sys.argv[1].rstrip('/'); CONF_HI=float(sys.argv[2]) if len(sys.argv)>2 else 0.85
CONF_LO=float(sys.argv[3]) if len(sys.argv)>3 else 0.55
pt=f'{clip}/smplx_arm_prefit.pt'; bak=f'{clip}/smplx_arm_prefit_nogate.pt'
if not os.path.exists(bak): shutil.copy(pt,bak); print(f'백업: {bak}')
ik=torch.load(bak,map_location='cpu')                                  # arm_prefit(IK, depth앵커+평활)
gv=torch.load(f'{clip}/smplx_merged_hamer_post.pt',map_location='cpu')  # GVHMR 원본 팔
F=ik['smpl_params_incam']['body_pose'].reshape(-1,21,3).shape[0]
WB=np.load(f'{clip}/preprocess/wholebody_kp2d.npy')
conf={'R':WB[:,112,2],'L':WB[:,91,2]}  # 손목 키포인트 conf(관절별)
def gate(cf):
    s=gaussian_filter1d(cf,3.0); return np.clip((CONF_HI-s)/(CONF_HI-CONF_LO),0,1)  # 저conf=1(GVHMR)
def blend(aIK,aGV,w):   # slerp per-frame, + 경계 평활
    out=np.zeros((F,3))
    for f in range(F):
        key=Rt.from_rotvec([aIK[f],aGV[f]]); out[f]=Slerp([0,1],key)([w[f]]).as_rotvec()[0]
    return out
def smooth(aa,sig=1.0):
    q=Rt.from_rotvec(aa).as_quat()
    for f in range(1,F):
        if np.dot(q[f],q[f-1])<0: q[f]=-q[f]
    qs=np.stack([gaussian_filter1d(q[:,i],sig,mode='nearest') for i in range(4)],1); qs/=np.linalg.norm(qs,axis=1,keepdims=True)
    return Rt.from_quat(qs).as_rotvec()
ROWS={'R':[13,16,18],'L':[12,15,17]}
out=copy.deepcopy(ik)
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp not in ik: continue
    bI=ik[sp]['body_pose'].reshape(F,21,3).numpy(); bG=gv[sp]['body_pose'].reshape(F,21,3).numpy()
    bp=bI.copy()
    for side,rows in ROWS.items():
        w=gate(conf[side])
        if sp=='smpl_params_incam': print(f'{side}팔: GVHMR 폴백 (저conf w>0.5: {int((w>0.5).sum())}f)')
        for j in rows: bp[:,j]=smooth(blend(bI[:,j],bG[:,j],w))
    out[sp]['body_pose']=torch.from_numpy(bp).float().reshape(out[sp]['body_pose'].shape)
torch.save(out,pt)
print(f'[arm_gate v3 GVHMR폴백] saved {pt}')
</content>
