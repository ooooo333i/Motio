"""
batch_collonly_v4: '충돌 처리만'. 비충돌 프레임은 fit2d에 잠그고(per-frame anchor lock),
충돌 프레임만 손 2D를 유지한 채 깊이방향으로 분리.
- 충돌 프레임 = collision_fit2d.json (손-손/팔-팔/팔-몸) ∪ dilate(temporal 여유).
- per-frame anchor: 비충돌 = 매우 강함(=fit2d 고정), 충돌 = 약함(움직임 허용).
- 손2D 강함(평면이탈 방지 → 분리는 깊이로) + collision + temporal.
입력: <clip>/{fit2d.pt, collision_fit2d.json, preprocess/wholebody_kp2d.npy}
출력: <clip>/collonly_v4.pt
usage: python batch_collonly_v4.py <clip_dir> [iters]
"""
import sys, numpy as np, copy, time, json
sys.path.insert(0, '/workspace/capstone_deploy')
import torch, smplx
from scipy.ndimage import binary_dilation

clip = sys.argv[1].rstrip('/'); ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 250
BASE = sys.argv[3] if len(sys.argv) > 3 else 'fit2d.pt'   # 충돌처리 기반 pt
BASE_STEM = BASE.replace('.pt','')
dev = 'cuda'; out_pt = f'{clip}/{BASE_STEM}_col.pt' if BASE!='fit2d.pt' else f'{clip}/collonly_v4.pt'
SX = '/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz'
W_HAND2D=8.0; W_COL=200.0; W_TC=400.0
W_ANCH_LOCK=3000.0   # 비충돌 프레임: fit2d에 강하게 고정
W_ANCH_FREE=15.0     # 충돌 프레임: 약하게(움직임 허용)
RAMP_SIGMA=3.0       # 경계 anchor를 로그공간 gaussian ramp(떨림 방지: 보정 fade in/out)
MARGIN=0.020; SIGMA=100.0; NSUB=250

t0=time.perf_counter()
src=torch.load(f'{clip}/{BASE}',map_location='cpu'); inc=src['smpl_params_incam']
bp0=inc['body_pose'].reshape(-1,21,3).clone(); F=bp0.shape[0]
K=src['K_fullimg']; K=K.unsqueeze(0).expand(F,-1,-1) if K.dim()==2 else K; Kt=K.float().to(dev)
WB=torch.tensor(np.load(f'{clip}/preprocess/wholebody_kp2d.npy'),dtype=torch.float32,device=dev)
# 충돌 프레임 mask
col=json.load(open(f'{clip}/collision_{BASE_STEM}.json'))
bad=np.zeros(F,bool)
for k in ('L손-R손','L팔-R팔','R팔-몸통','L팔-몸통'):
    if k in col:
        for f in col[k]['bad_frames']: bad[f]=True
bad=binary_dilation(bad,iterations=3)   # temporal 여유
print(f'[collonly] 충돌(+dilate) 프레임 = {int(bad.sum())}/{F}')
# 경계 떨림 방지: binary 대신 로그공간 gaussian ramp (충돌=자유, 양옆 서서히 잠금복귀)
from scipy.ndimage import gaussian_filter1d
r=gaussian_filter1d(bad.astype(np.float64),RAMP_SIGMA); r=np.clip(r/max(r.max(),1e-6),0,1)
anch_w=np.exp(np.log(W_ANCH_LOCK)*(1-r)+np.log(W_ANCH_FREE)*r).astype(np.float32)
anch_wt=torch.tensor(anch_w,device=dev)

def fix(t):
    t=(t if torch.is_tensor(t) else torch.tensor(t)).float()
    if t.dim()==1:t=t.unsqueeze(0)
    if t.shape[0]==1 and F>1:t=t.expand(F,-1).contiguous()
    return t.to(dev)
betas=fix(inc['betas']);go=fix(inc['global_orient']);tr=fix(inc['transl']);lhp=fix(inc['left_hand_pose']);rhp=fix(inc['right_hand_pose'])
m=smplx.create(SX,model_type='smplx',gender='neutral',use_pca=False,flat_hand_mean=False,num_betas=betas.shape[1],batch_size=F).to(dev)
Wt=np.load(SX,allow_pickle=True)['weights']; part=Wt.argmax(1)
def subv(idx): idx=np.asarray(idx); return idx[np.linspace(0,len(idx)-1,min(NSUB,len(idx))).astype(int)]
Lh=np.where(np.isin(part,[20]+list(range(25,40))))[0]; Rh=np.where(np.isin(part,[21]+list(range(40,55))))[0]
Lf=np.where(part==18)[0]; Rf=np.where(part==19)[0]
Lt=torch.tensor(subv(np.concatenate([Lf,Lh])),device=dev); Rt=torch.tensor(subv(np.concatenate([Rf,Rh])),device=dev)
ARM_IDS=[15,16,17,18,19,20]; arm_idx_t=torch.tensor(ARM_IDS,device=dev)
body=bp0.clone().to(dev).requires_grad_(True); bp_fit=bp0.clone().to(dev)
mask=torch.zeros(21,1,device=dev)
for i in ARM_IDS: mask[i]=1.0
RID_S=[21,40,43,46,49,52]; RID_W=[112+i for i in [0,5,9,17,13,2]]; LID_S=[20,25,28,31,34,37]; LID_W=[91+i for i in [0,5,9,17,13,2]]
def forward(bp): return m(betas=betas,body_pose=bp.reshape(F,63),global_orient=go,transl=tr,left_hand_pose=lhp,right_hand_pose=rhp)
def proj(J3):
    z=J3[...,2].clamp(min=1e-4); u=Kt[:,0,0:1]*J3[...,0]/z+Kt[:,0,2:3]; v=Kt[:,1,1:2]*J3[...,1]/z+Kt[:,1,2:3]; return torch.stack([u,v],-1)
def gmof(r2): return (SIGMA**2*r2)/(SIGMA**2+r2)
def aa2R(aa):
    th=aa.norm(dim=-1,keepdim=True).clamp(min=1e-8); ax=aa/th; x,y,z=ax[...,0],ax[...,1],ax[...,2]
    c=torch.cos(th)[...,0]; s=torch.sin(th)[...,0]; C=1-c
    return torch.stack([c+x*x*C,x*y*C-z*s,x*z*C+y*s,y*x*C+z*s,c+y*y*C,y*z*C-x*s,z*x*C-y*s,z*y*C+x*s,c+z*z*C],-1).reshape(aa.shape[:-1]+(3,3))
def geo(R1,R2):
    Rr=R1.transpose(-1,-2)@R2; t_=Rr[...,0,0]+Rr[...,1,1]+Rr[...,2,2]; return torch.acos(((t_-1)/2).clamp(-1+1e-6,1-1e-6)).mean()

opt=torch.optim.Adam([body],lr=0.01)
for it in range(ITERS):
    bp=bp_fit*(1-mask)+body*mask; out=forward(bp); Vv=out.vertices; J=out.joints
    # 손 2D (강, 충돌 프레임에서 평면이탈 방지)
    Lhand=0.0
    for SID,WID in [(RID_S,RID_W),(LID_S,LID_W)]:
        p=proj(J[:,SID]); o=WB[:,WID,:2]; cf=WB[:,WID,2:3].clamp(0,1)
        c=o.mean(1,keepdim=True); mn=o.min(1,keepdim=True)[0]; mx=o.max(1,keepdim=True)[0]
        sc=torch.sqrt(((mx-mn)**2).sum(-1,keepdim=True)).clamp(min=1.0)
        r2=(((p-c)/sc-(o-c)/sc)**2).sum(-1,keepdim=True)
        Lhand=Lhand+(cf**2*gmof(r2*sc*sc)).mean()/1000.0
    D=torch.cdist(Vv[:,Lt],Vv[:,Rt]); mindL=D.min(2)[0]; mindR=D.min(1)[0]
    Lcol=(torch.relu(MARGIN-mindL).sum(1)+torch.relu(MARGIN-mindR).sum(1)).mean()
    # per-frame anchor: 비충돌=lock, 충돌=free
    da=((body[:,arm_idx_t]-bp_fit[:,arm_idx_t])**2).sum(-1).sum(-1)   # (F,)
    Lanch=(anch_wt*da).mean()
    Ra=aa2R(body[:,arm_idx_t]); Ltc=geo(Ra[:-1],Ra[1:])
    loss=W_HAND2D*Lhand+W_COL*Lcol+Lanch+W_TC*Ltc
    opt.zero_grad(); loss.backward()
    with torch.no_grad(): body.grad*=mask
    opt.step()
    if it%50==0 or it==ITERS-1:
        with torch.no_grad(): npen=int((mindL<MARGIN).sum().item())
        print('  co it%3d hand=%.4f col=%.5f anch=%.3f tc=%.4f 근접<%dmm=%d'%(it,float(Lhand),Lcol.item(),Lanch.item(),float(Ltc),int(MARGIN*1000),npen))

with torch.no_grad(): bp_final=(bp_fit*(1-mask)+body*mask).detach().cpu()
# 떨림 제거: 보정량(delta=collonly-fit2d)을 시간축 gaussian smooth (비충돌 delta≈0 → 영향 없음)
from scipy.ndimage import gaussian_filter1d as _g1
bpf_np=bp_fit.cpu().numpy(); delta=bp_final.numpy()-bpf_np
for i in ARM_IDS:
    for c in range(3):
        delta[:,i,c]=_g1(delta[:,i,c],2.0,mode='nearest')
bp_final=torch.from_numpy(bpf_np+delta).float()
print(f'[collonly] delta 시간축 평활(sigma=2) 적용')
out_dict=copy.deepcopy(src)
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp in out_dict:
        b=out_dict[sp]['body_pose'].reshape(F,21,3).clone()
        for i in ARM_IDS: b[:,i]=bp_final[:,i]
        out_dict[sp]['body_pose']=b.reshape(F,21,3) if out_dict[sp]['body_pose'].dim()==3 else b.reshape(F,63)
torch.save(out_dict,out_pt)
print(f'[collonly_v4] saved {out_pt} ({time.perf_counter()-t0:.1f}s)')
