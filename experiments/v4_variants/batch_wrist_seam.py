"""
batch_wrist_seam: 19.8초 같은 "손목 위치 스냅"만 국소 제거.
 진단: ViTPose 블러 구간에서 IK가 GVHMR보다 더 튀는 초과속도(IK-GVHMR)가 스냅 인공물.
 판별자 = 초과속도(IK-GVHMR 손목 투영속도). conf/속도로는 14초(가려짐스윙)와 19.8초(블러스윙)를 못 가르지만,
          초과속도는 19.8초(13.6px)≫14초(~5px)라 자연히 분리됨. 임계=클립별 percentile(일반화).
 수정: 목적지(GVHMR)로 당기지 않음(2D 보정 유지). 초과 프레임만 팔회전을 시간평활본으로 slerp → 속도스파이크(갑작스러움)만 제거.
       w=0 프레임(14초 등)은 완전 불변.
 ⚠️ 팔(collar/shoulder/elbow)만. 손목회전/손가락은 이후 단계에서.
입력: <clip>/{smplx_arm_prefit_nogate.pt(IK), smplx_merged_hamer_post.pt(GVHMR)}, fit2d.pt
출력: <clip>/smplx_arm_prefit.pt
usage: python batch_wrist_seam.py <clip_dir> [w_max=1.0] [sm_sigma=3.0]
"""
import sys, numpy as np, copy
import torch, smplx
from scipy.spatial.transform import Rotation as Rt, Slerp
from scipy.ndimage import gaussian_filter1d
clip=sys.argv[1].rstrip('/'); W_MAX=float(sys.argv[2]) if len(sys.argv)>2 else 1.0
SM=float(sys.argv[3]) if len(sys.argv)>3 else 3.0
SMPLX='/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz'
ik=torch.load(f'{clip}/smplx_arm_prefit_nogate.pt',map_location='cpu')
gv=torch.load(f'{clip}/smplx_merged_hamer_post.pt',map_location='cpu')
F=ik['smpl_params_incam']['body_pose'].reshape(-1,21,3).shape[0]
f2=torch.load(f'{clip}/fit2d.pt',map_location='cpu')['smpl_params_incam']
def fix(t):
    t=(t if torch.is_tensor(t) else torch.tensor(t)).float()
    if t.dim()==1:t=t.unsqueeze(0)
    if t.shape[0]==1:t=t.expand(F,-1).contiguous()
    return t
betas=fix(f2['betas']);go=fix(f2['global_orient']);trn=fix(f2['transl'])
K=torch.load(f'{clip}/fit2d.pt',map_location='cpu')['K_fullimg']; K=K.unsqueeze(0).expand(F,-1,-1) if K.dim()==2 else K; K=K.float().numpy()
m=smplx.create(SMPLX,model_type='smplx',gender='neutral',use_pca=False,flat_hand_mean=False,num_betas=betas.shape[1],batch_size=F)
def wrproj(bp,j):
    with torch.no_grad(): J=m(betas=betas,body_pose=fix(bp).reshape(F,63),global_orient=go,transl=trn).joints
    p=J[:,j].numpy(); z=np.clip(p[:,2],1e-4,None)
    return np.stack([K[:,0,0]*p[:,0]/z+K[:,0,2],K[:,1,1]*p[:,1]/z+K[:,1,2]],1)
def smooth_quat(aa,sig):
    q=Rt.from_rotvec(aa).as_quat()
    for f in range(1,F):
        if np.dot(q[f],q[f-1])<0: q[f]=-q[f]
    qs=np.stack([gaussian_filter1d(q[:,i],sig,mode='nearest') for i in range(4)],1); qs/=np.linalg.norm(qs,axis=1,keepdims=True)
    return Rt.from_quat(qs).as_rotvec()
ROWS={'R':(21,[13,16,18]),'L':(20,[12,15,17])}
out=copy.deepcopy(ik)
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp not in ik: continue
    bI=ik[sp]['body_pose'].reshape(F,21,3).numpy(); bp=bI.copy()
    for side,(jw,rows) in ROWS.items():
        ikp=wrproj(ik['smpl_params_incam']['body_pose'],jw); gvp=wrproj(gv['smpl_params_incam']['body_pose'],jw)
        iksp=np.r_[0,np.linalg.norm(np.diff(ikp,axis=0),axis=1)]; gvsp=np.r_[0,np.linalg.norm(np.diff(gvp,axis=0),axis=1)]
        ex=np.maximum(0,iksp-gvsp)                       # 초과속도(IK가 GVHMR보다 더 튐)=스냅
        LO=max(np.percentile(ex,99.3),8.0); HI=max(ex.max(),LO+4)  # 14초(~7px)와 19.8초(13.6px) 사이
        w=np.clip((ex-LO)/(HI-LO),0,W_MAX)               # 원시 ex로 게이트(스파이크 보존)
        w=gaussian_filter1d(w,2.0)                       # 보정 램프(경계 부드럽게)
        if sp=='smpl_params_incam': print(f'{side}손: 초과 p99={LO:.1f}px, 보정(w>0.1) {int((w>0.1).sum())}f, 최대w {w.max():.2f}')
        for j in rows:
            sm=smooth_quat(bI[:,j],SM)                   # 시간평활본(스파이크 제거)
            for f in range(F):
                key=Rt.from_rotvec([bI[f,j],sm[f]]); bp[f,j]=Slerp([0,1],key)([w[f]]).as_rotvec()[0]
    out[sp]['body_pose']=torch.from_numpy(bp).float().reshape(out[sp]['body_pose'].shape)
torch.save(out,f'{clip}/smplx_arm_prefit.pt')
print(f'[wrist_seam] saved {clip}/smplx_arm_prefit.pt (목적지 GVHMR로 안당김, 스파이크만 국소제거)')
