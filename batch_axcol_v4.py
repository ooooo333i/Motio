"""
batch_axcol_v4 (사용자 #2): 충돌을 '손목 회전(axial=2D invisible)'으로 해소. 팔/위치 고정.
 - 손목 [19,20]만 최적화. 강한 손2D(swing 보존) → 옵티마이저가 2D 안 건드리는 축(axial)으로 손을 돌려 분리.
 - collision 반발 + 강 hand2D + anchor(현 손목) + temporal.
입력: <clip>/{armik_v4.pt, collision_armik_v4.json, preprocess/wholebody_kp2d.npy}
출력: <clip>/armik_axcol.pt
usage: python batch_axcol_v4.py <clip_dir> [iters]
"""
import sys, numpy as np, copy, time, json
sys.path.insert(0,'/workspace/capstone_deploy')
import torch, smplx
from scipy.ndimage import binary_dilation, gaussian_filter1d
clip=sys.argv[1].rstrip('/'); ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 250
BASE=sys.argv[3] if len(sys.argv)>3 else 'armik_v4.pt'; STEM=BASE.replace('.pt','')
dev='cuda'; out_pt=f'{clip}/{STEM}_col.pt'; SX='/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz'
W_HAND2D=15.0   # 강 (swing/투영 보존 → 분리는 invisible axial로)
W_COL=300.0; W_ANCH=8.0; W_TC=300.0; MARGIN=0.018; SIGMA=100.0; NSUB=250
t0=time.perf_counter()
src=torch.load(f'{clip}/{BASE}',map_location='cpu'); inc=src['smpl_params_incam']
bp0=inc['body_pose'].reshape(-1,21,3).clone(); F=bp0.shape[0]
K=src['K_fullimg']; K=K.unsqueeze(0).expand(F,-1,-1) if K.dim()==2 else K; Kt=K.float().to(dev)
WB=torch.tensor(np.load(f'{clip}/preprocess/wholebody_kp2d.npy'),dtype=torch.float32,device=dev)
col=json.load(open(f'{clip}/collision_{STEM}.json')); bad=np.zeros(F,bool)
for k in ('L손-R손','L팔-R팔'):
    for f in col[k]['bad_frames']: bad[f]=True
bad=binary_dilation(bad,iterations=6)  # 전파 넓힘
r=gaussian_filter1d(bad.astype(np.float64),4.0); r=np.clip(r/max(r.max(),1e-6),0,1)
anch_w=torch.tensor(np.exp(np.log(3000.0)*(1-r)+np.log(W_ANCH)*r).astype(np.float32),device=dev)  # 비충돌 잠금
print(f'[axcol] 충돌(+dilate) 프레임={int(bad.sum())}/{F}')
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
wr_R=bp0[:,20].clone().to(dev).requires_grad_(True); wr_L=bp0[:,19].clone().to(dev).requires_grad_(True)
wr_R0=bp0[:,20].clone().to(dev); wr_L0=bp0[:,19].clone().to(dev); bp_fixed=bp0.clone().to(dev)
RID_S=[21,40,43,46,49,52]; RID_W=[112+i for i in [0,5,9,17,13,2]]; LID_S=[20,25,28,31,34,37]; LID_W=[91+i for i in [0,5,9,17,13,2]]
def proj(J3):
    z=J3[...,2].clamp(min=1e-4); u=Kt[:,0,0:1]*J3[...,0]/z+Kt[:,0,2:3]; v=Kt[:,1,1:2]*J3[...,1]/z+Kt[:,1,2:3]; return torch.stack([u,v],-1)
def gmof(r2): return (SIGMA**2*r2)/(SIGMA**2+r2)
opt=torch.optim.Adam([wr_R,wr_L],lr=0.01)
_prev_npen=None; _stall=0   # early-stop 상태(충돌 수렴 감지)
for it in range(ITERS):
    bp=bp_fixed.clone(); bp[:,20]=wr_R; bp[:,19]=wr_L
    out=m(betas=betas,body_pose=bp.reshape(F,63),global_orient=go,transl=tr,left_hand_pose=lhp,right_hand_pose=rhp); Vv=out.vertices; J=out.joints
    Lhand=0.0
    for SID,WID in [(RID_S,RID_W),(LID_S,LID_W)]:
        p=proj(J[:,SID]); o=WB[:,WID,:2]; cf=WB[:,WID,2:3].clamp(0,1)
        c=o.mean(1,keepdim=True); mn=o.min(1,keepdim=True)[0]; mx=o.max(1,keepdim=True)[0]
        sc=torch.sqrt(((mx-mn)**2).sum(-1,keepdim=True)).clamp(min=1.0)
        r2=(((p-c)/sc-(o-c)/sc)**2).sum(-1,keepdim=True); Lhand=Lhand+(cf**2*gmof(r2*sc*sc)).mean()/1000.0
    D=torch.cdist(Vv[:,Lt],Vv[:,Rt]); Lcol=(torch.relu(MARGIN-D.min(2)[0]).sum(1)+torch.relu(MARGIN-D.min(1)[0]).sum(1)).mean()
    da=((wr_R-wr_R0)**2).sum(-1)+((wr_L-wr_L0)**2).sum(-1)
    Lanch=(anch_w*da).mean()
    Ltc=((wr_R[1:]-wr_R[:-1])**2).mean()+((wr_L[1:]-wr_L[:-1])**2).mean()
    loss=W_HAND2D*Lhand+W_COL*Lcol+Lanch+W_TC*Ltc
    opt.zero_grad(); loss.backward(); opt.step()
    if it%25==0 or it==ITERS-1:
        with torch.no_grad(): npen=int((D.min(2)[0]<MARGIN).sum().item())
        print('  axcol it%3d hand=%.4f col=%.5f anch=%.4f 근접<%dmm=%d'%(it,float(Lhand),float(Lcol),float(Lanch),int(MARGIN*1000),npen))
        # early-stop: 충돌 수렴(npen=0) 또는 2회 연속 개선없음 + 최소 80 iter 후 (품질 동일, 속도↑)
        if it>=80:
            _stall = _stall+1 if (_prev_npen is not None and npen>=_prev_npen) else 0
            _prev_npen=npen
            if npen==0 or _stall>=2:
                print('  axcol early-stop @it%d (npen=%d, 수렴)'%(it,npen)); break
        else:
            _prev_npen=npen
# delta 시간축 평활(떨림 방지)
bpf=bp_fixed.cpu().numpy(); rN=wr_R.detach().cpu().numpy(); lN=wr_L.detach().cpu().numpy()
dR=rN-bpf[:,20]; dL=lN-bpf[:,19]
for c in range(3): dR[:,c]=gaussian_filter1d(dR[:,c],4.0,mode='nearest'); dL[:,c]=gaussian_filter1d(dL[:,c],4.0,mode='nearest')
bpn=bp0.clone(); bpn[:,20]=torch.from_numpy(bpf[:,20]+dR).float(); bpn[:,19]=torch.from_numpy(bpf[:,19]+dL).float()
out_dict=copy.deepcopy(src)
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp in out_dict: out_dict[sp]['body_pose']=bpn.reshape(F,21,3) if out_dict[sp]['body_pose'].dim()==3 else bpn.reshape(F,63)
torch.save(out_dict,out_pt); print(f'[axcol] saved {out_pt} ({time.perf_counter()-t0:.1f}s)')
