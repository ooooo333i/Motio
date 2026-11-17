"""
batch_finger_fit: 손가락 pose를 WB 21점 2D에 fit (영상과 맞춤) + HaMeR prior(그립 유지) + 시간평활(털기 제거).
 - 손가락(left/right_hand_pose 45) 만 최적화. 팔/손목/위치 고정.
 - smplx 손가락 joint(L25-39, R40-54, 3/finger) ↔ WB 손가락(MCP/PIP/DIP) 매핑.
 - loss: scale-norm gmof 2D + prior(HaMeR _post) + temporal.
입력: <clip>/{armik_axcol.pt, preprocess/wholebody_kp2d.npy}
출력: <clip>/armik_finger.pt
usage: python batch_finger_fit.py <clip_dir> [iters]
"""
import sys, numpy as np, copy, time
sys.path.insert(0,'/workspace/capstone_deploy')
import torch, smplx
clip=sys.argv[1].rstrip('/'); ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 250
dev='cuda'; out_pt=f'{clip}/armik_finger.pt'; SX='/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz'
import os
import os
W2D=float(os.environ.get("W2D",3)); W_PRIOR=float(os.environ.get("WP",15)); W_TC=float(os.environ.get("WTC",800)); SIGMA=80.0
t0=time.perf_counter()
src=torch.load(f'{clip}/armik_axcol.pt',map_location='cpu'); inc=src['smpl_params_incam']
F=inc['body_pose'].shape[0]
K=src['K_fullimg']; K=K.unsqueeze(0).expand(F,-1,-1) if K.dim()==2 else K; Kt=K.float().to(dev)
WB=torch.tensor(np.load(f'{clip}/preprocess/wholebody_kp2d.npy'),dtype=torch.float32,device=dev)
def fix(t,d=None):
    t=(t if torch.is_tensor(t) else torch.tensor(t)).float()
    if t.dim()==1:t=t.unsqueeze(0)
    if t.shape[0]==1 and F>1:t=t.expand(F,-1).contiguous()
    return t.to(dev)
betas=fix(inc['betas']);go=fix(inc['global_orient']);tr=fix(inc['transl'])
bp=inc['body_pose'].reshape(F,21,3).float().to(dev)
m=smplx.create(SX,model_type='smplx',gender='neutral',use_pca=False,flat_hand_mean=False,num_betas=betas.shape[1],batch_size=F).to(dev)
lhp0=fix(inc['left_hand_pose']).clone(); rhp0=fix(inc['right_hand_pose']).clone()
lhp=lhp0.clone().requires_grad_(True); rhp=rhp0.clone().requires_grad_(True)
# 매핑: smplx 손가락 joint → WB 손가락 keypoint (3/finger: MCP,PIP,DIP), 순서 thumb,index,middle,pinky,ring
def fmap(base_s, base_w):
    sj=[]; wj=[]
    fingers=[(0,1),(3,5),(6,9),(9,17),(12,13)]  # (smplx offset start, WB offset start) thumb/index/middle/pinky/ring
    for so,wo in fingers:
        for k in range(3): sj.append(base_s+so+k); wj.append(base_w+wo+k)
    return sj,wj
RSJ,RWJ=fmap(40,112); LSJ,LWJ=fmap(25,91)
def proj(J3):
    z=J3[...,2].clamp(min=1e-4); u=Kt[:,0,0:1]*J3[...,0]/z+Kt[:,0,2:3]; v=Kt[:,1,1:2]*J3[...,1]/z+Kt[:,1,2:3]; return torch.stack([u,v],-1)
def gmof(r2): return (SIGMA**2*r2)/(SIGMA**2+r2)
def loss2d(J,SID,WID):
    pj=proj(J[:,SID]); obs=WB[:,WID,:2]; conf=WB[:,WID,2:3].clamp(0,1)
    c=obs.mean(1,keepdim=True); mn=obs.min(1,keepdim=True)[0]; mx=obs.max(1,keepdim=True)[0]
    sc=torch.sqrt(((mx-mn)**2).sum(-1,keepdim=True)).clamp(min=1.0)
    res2=(((pj-c)/sc-(obs-c)/sc)**2).sum(-1,keepdim=True)
    return (conf**2*gmof(res2)).mean()
opt=torch.optim.Adam([lhp,rhp],lr=0.02)
for it in range(ITERS):
    out=m(betas=betas,body_pose=bp.reshape(F,63),global_orient=go,transl=tr,left_hand_pose=lhp,right_hand_pose=rhp); J=out.joints
    l2d=loss2d(J,RSJ,RWJ)+loss2d(J,LSJ,LWJ)
    lpri=((lhp-lhp0)**2).mean()+((rhp-rhp0)**2).mean()
    ltc=((lhp[1:]-lhp[:-1])**2).mean()+((rhp[1:]-rhp[:-1])**2).mean()
    loss=W2D*l2d*100+W_PRIOR*lpri+W_TC*ltc
    opt.zero_grad(); loss.backward(); opt.step()
    if it%50==0 or it==ITERS-1: print('  finger it%3d 2d=%.4f prior=%.4f tc=%.4f'%(it,float(l2d),float(lpri),float(ltc)))
from scipy.ndimage import gaussian_filter1d as _g1
_lh=lhp.detach().cpu().numpy(); _rh=rhp.detach().cpu().numpy()
for c in range(_lh.shape[1]): _lh[:,c]=_g1(_lh[:,c],1.5,mode='nearest'); _rh[:,c]=_g1(_rh[:,c],1.5,mode='nearest')
import torch as _t; lhp=_t.from_numpy(_lh).float().to(dev); rhp=_t.from_numpy(_rh).float().to(dev)
print('[finger_fit] 손가락 pose 후평활(sigma=1.5)')
out_dict=copy.deepcopy(src)
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp in out_dict:
        out_dict[sp]['left_hand_pose']=lhp.detach().cpu().reshape(out_dict[sp]['left_hand_pose'].shape)
        out_dict[sp]['right_hand_pose']=rhp.detach().cpu().reshape(out_dict[sp]['right_hand_pose'].shape)
torch.save(out_dict,out_pt); print(f'[finger_fit] saved {out_pt} ({time.perf_counter()-t0:.1f}s)')
