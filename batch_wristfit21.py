"""
batch_wristfit21 (사용자 의도): grip(손가락 pose)은 HaMeR 그대로 고정, 손목 각도만 (ROM 내에서)
 조정해 손 전체(WB 손목+15 손가락 joint=16점)를 2D keyframe에 정렬. 강체 손 회전으로 손가락까지 맞춤.
 - 손목 [19,20]만 최적화. 팔=arm_prefit, 손가락=_post(HaMeR) 고정.
 - loss: scale-norm gmof 2D(16점) + WSM 평활 + ROM soft-limit(dev45/flex90).
입력: <clip>/{smplx_merged_hamer_post.pt, smplx_arm_prefit.pt, preprocess/wholebody_kp2d.npy}
출력: <clip>/wristfit21.pt
usage: python batch_wristfit21.py <clip_dir> [iters]
"""
import sys, numpy as np, copy, time
sys.path.insert(0,'/workspace/capstone_deploy')
import torch, smplx
clip=sys.argv[1].rstrip('/'); ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 300
dev='cuda'; out_pt=f'{clip}/wristfit21.pt'; SX='/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz'
W2D=1.0; WSM=20.0; W_ROM=15.0; SIGMA=100.0; DEV=np.radians(45.); FLEX=np.radians(90.)
t0=time.perf_counter()
src=torch.load(f'{clip}/smplx_merged_hamer_post.pt',map_location='cpu'); inc=src['smpl_params_incam']
arm=torch.load(f'{clip}/smplx_arm_prefit.pt',map_location='cpu')['smpl_params_incam']['body_pose'].reshape(-1,21,3)
bp0=inc['body_pose'].reshape(-1,21,3).clone(); F=bp0.shape[0]
for i in [12,13,15,16,17,18,19,20]: bp0[:,i]=arm[:,i]
K=src['K_fullimg']; K=K.unsqueeze(0).expand(F,-1,-1) if K.dim()==2 else K; K=K.float().numpy()
WB=np.load(f'{clip}/preprocess/wholebody_kp2d.npy')
def fix(t):
    t=(t if torch.is_tensor(t) else torch.tensor(t)).float()
    if t.dim()==1:t=t.unsqueeze(0)
    if t.shape[0]==1 and F>1:t=t.expand(F,-1).contiguous()
    return t.to(dev)
betas=fix(inc['betas']);go=fix(inc['global_orient']);tr=fix(inc['transl']);lhp=fix(inc['left_hand_pose']);rhp=fix(inc['right_hand_pose'])
m=smplx.create(SX,model_type='smplx',gender='neutral',use_pca=False,flat_hand_mean=False,num_betas=betas.shape[1],batch_size=F).to(dev)
wr_R=bp0[:,20].clone().to(dev).requires_grad_(True); wr_L=bp0[:,19].clone().to(dev).requires_grad_(True)
bp_fixed=bp0.clone().to(dev)
# 16점: 손목 + 15 손가락 joint (thumb/index/middle/pinky/ring × MCP/PIP/DIP)
def fmap(ws,bw,wW):  # ws=손목 smplx, bw=손가락 smplx base, wW=WB hand base
    sj=[ws]; wj=[wW+0]
    for so,wo in [(0,1),(3,5),(6,9),(9,17),(12,13)]:
        for k in range(3): sj.append(bw+so+k); wj.append(wW+wo+k)
    return sj,wj
RID,RW=fmap(21,40,112); LID,LW=fmap(20,25,91)
Kt=torch.tensor(K,dtype=torch.float32,device=dev); WBt=torch.tensor(WB,dtype=torch.float32,device=dev)
def proj(J):
    z=J[...,2].clamp(min=1e-4); u=Kt[:,0,0:1]*J[...,0]/z+Kt[:,0,2:3]; v=Kt[:,1,1:2]*J[...,1]/z+Kt[:,1,2:3]; return torch.stack([u,v],-1)
def gmof(r2): return (SIGMA**2*r2)/(SIGMA**2+r2)
def loss2d(J,SID,WID):
    pj=proj(J[:,SID]); obs=WBt[:,WID,:2]; conf=WBt[:,WID,2:3].clamp(0,1)
    c=obs.mean(1,keepdim=True); mn=obs.min(1,keepdim=True)[0]; mx=obs.max(1,keepdim=True)[0]
    sc=torch.sqrt(((mx-mn)**2).sum(-1,keepdim=True)).clamp(min=1.0)
    res2=(((pj-c)/sc-(obs-c)/sc)**2).sum(-1,keepdim=True)
    return (conf**2*gmof(res2)).mean()
opt=torch.optim.Adam([wr_R,wr_L],lr=0.02)
for it in range(ITERS):
    bp=bp_fixed.clone(); bp[:,20]=wr_R; bp[:,19]=wr_L
    out=m(betas=betas,body_pose=bp.reshape(F,63),global_orient=go,transl=tr,left_hand_pose=lhp,right_hand_pose=rhp); J=out.joints
    l2d=loss2d(J,RID,RW)+loss2d(J,LID,LW)
    sm=((wr_R[1:]-wr_R[:-1])**2).mean()+((wr_L[1:]-wr_L[:-1])**2).mean()
    rom=(torch.relu(wr_R[:,1].abs()-DEV)**2+torch.relu(wr_R[:,2].abs()-FLEX)**2+torch.relu(wr_L[:,1].abs()-DEV)**2+torch.relu(wr_L[:,2].abs()-FLEX)**2).mean()
    loss=W2D*l2d*100+WSM*sm+W_ROM*rom
    opt.zero_grad(); loss.backward(); opt.step()
    if it%50==0 or it==ITERS-1: print('  wf21 it%3d 2d=%.4f sm=%.4f'%(it,float(l2d),float(sm)))
out_dict=copy.deepcopy(src)
bpn=bp0.clone(); bpn[:,20]=wr_R.detach().cpu(); bpn[:,19]=wr_L.detach().cpu()
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp in out_dict: out_dict[sp]['body_pose']=bpn.reshape(F,21,3) if out_dict[sp]['body_pose'].dim()==3 else bpn.reshape(F,63)
torch.save(out_dict,out_pt); print(f'[wristfit21] saved {out_pt} ({time.perf_counter()-t0:.1f}s)')
