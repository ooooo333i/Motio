"""
batch_zanchor_v4 (사용자 아이디어): 2D 따라가기 + GVHMR z(깊이) 소프트 앵커 + temporal.
  + (옵션) collision 항.  충돌 검출/프레임잠금 없는 '일반' 형태.
- 어깨/팔꿈치/손목 최적화. 손2D(WB)+팔2D(ViTPose) 따라감.
- z-anchor: elbow/wrist/hand 관절의 cam-z를 GVHMR(_post)에 soft 앵커 (깊이 흔들림/flip 방지).
- temporal: 인접 프레임 회전 평활.
usage: python batch_zanchor_v4.py <clip_dir> [iters] [w_col]   # w_col=0 → 순수 z앵커(A), >0 → +충돌(B)
출력: <clip>/zanchor_v4.pt
"""
import sys, numpy as np, copy, time
sys.path.insert(0, '/workspace/capstone_deploy')
import torch, smplx

clip = sys.argv[1].rstrip('/'); ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
W_COL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
dev='cuda'; out_pt=f'{clip}/zanchor_v4.pt'; SX='/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz'
W_HAND2D=8.0; W_BODY2D=1.0; W_Z=80.0; W_TC=250.0; W_ROTPRIOR=1.0
MARGIN=0.015; SIGMA=100.0; NSUB=250

t0=time.perf_counter()
warm=torch.load(f'{clip}/fit2d.pt',map_location='cpu'); inc=warm['smpl_params_incam']
gv=torch.load(f'{clip}/smplx_merged_hamer_post.pt',map_location='cpu')['smpl_params_incam']
bp0=inc['body_pose'].reshape(-1,21,3).clone(); F=bp0.shape[0]
bp_gv=gv['body_pose'].reshape(F,21,3).clone()
K=warm['K_fullimg']; K=K.unsqueeze(0).expand(F,-1,-1) if K.dim()==2 else K; Kt=K.float().to(dev)
WB=torch.tensor(np.load(f'{clip}/preprocess/wholebody_kp2d.npy'),dtype=torch.float32,device=dev)
vp=torch.load(f'{clip}/preprocess/vitpose.pt',map_location='cpu'); vp=(vp.numpy() if torch.is_tensor(vp) else np.array(vp)).astype(np.float32)
VPt=torch.tensor(vp,device=dev)
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
body=bp0.clone().to(dev).requires_grad_(True); bp_fit=bp0.clone().to(dev); bp_gvd=bp_gv.to(dev)
mask=torch.zeros(21,1,device=dev)
for i in ARM_IDS: mask[i]=1.0
VP_BODY=[(5,16),(6,17),(7,18),(8,19),(9,20),(10,21)]; VPI=[c for c,_ in VP_BODY]; SJ=[s for _,s in VP_BODY]; SJt=torch.tensor(SJ,device=dev)
# z-anchor 대상 관절(out.joints idx): elbow18/19, wrist20/21, + 손 joint(25~54 중 일부)
ZJ=[18,19,20,21]+list(range(25,55)); ZJt=torch.tensor(ZJ,device=dev)
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
with torch.no_grad():
    z_gv=forward(bp_gvd*mask+bp_fit*(1-mask)).joints[:,ZJt,2].detach()  # GVHMR 팔 기준 z (몸=fit2d=gvhmr 동일)
    # 주의: 몸통은 fit2d=gvhmr 동일하므로 사실상 GVHMR 팔의 z
    z_gv=forward(bp_gvd).joints[:,ZJt,2].detach()

opt=torch.optim.Adam([body],lr=0.01)
for it in range(ITERS):
    bp=bp_fit*(1-mask)+body*mask; out=forward(bp); Vv=out.vertices; J=out.joints; uv=proj(J)
    Lhand=0.0
    for SID,WID in [(RID_S,RID_W),(LID_S,LID_W)]:
        p=proj(J[:,SID]); o=WB[:,WID,:2]; cf=WB[:,WID,2:3].clamp(0,1)
        c=o.mean(1,keepdim=True); mn=o.min(1,keepdim=True)[0]; mx=o.max(1,keepdim=True)[0]
        sc=torch.sqrt(((mx-mn)**2).sum(-1,keepdim=True)).clamp(min=1.0)
        r2=(((p-c)/sc-(o-c)/sc)**2).sum(-1,keepdim=True)
        Lhand=Lhand+(cf**2*gmof(r2*sc*sc)).mean()/1000.0
    pj=uv[:,SJt,:]; tg=VPt[:,VPI,:2]; cf2=VPt[:,VPI,2:3].clamp(0,1); cf2=torch.where(cf2>=0.3,cf2,torch.zeros_like(cf2))
    Lbody=(cf2*((pj-tg)**2).sum(-1,keepdim=True)).mean()/1000.0
    Lz=((J[:,ZJt,2]-z_gv)**2).mean()                       # ★ GVHMR z 소프트 앵커
    Lrp=(((body[:,arm_idx_t]-bp_gvd[:,arm_idx_t])**2).sum(-1)).mean()
    Ra=aa2R(body[:,arm_idx_t]); Ltc=geo(Ra[:-1],Ra[1:])
    loss=W_HAND2D*Lhand+W_BODY2D*Lbody+W_Z*Lz+W_ROTPRIOR*Lrp+W_TC*Ltc
    if W_COL>0:
        D=torch.cdist(Vv[:,Lt],Vv[:,Rt]); mindL=D.min(2)[0]; mindR=D.min(1)[0]
        Lcol=(torch.relu(MARGIN-mindL).sum(1)+torch.relu(MARGIN-mindR).sum(1)).mean()
        loss=loss+W_COL*Lcol
    else:
        with torch.no_grad():
            D=torch.cdist(Vv[:,Lt],Vv[:,Rt]); Lcol=torch.relu(MARGIN-D.min(2)[0]).sum(1).mean()
    opt.zero_grad(); loss.backward()
    with torch.no_grad(): body.grad*=mask
    opt.step()
    if it%50==0 or it==ITERS-1:
        print('  z it%3d hand=%.4f body=%.4f z=%.5f tc=%.4f col=%.5f'%(it,float(Lhand),Lbody.item(),Lz.item(),float(Ltc),float(Lcol)))

with torch.no_grad(): bp_final=(bp_fit*(1-mask)+body*mask).detach().cpu()
out_dict=copy.deepcopy(warm)
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp in out_dict:
        b=out_dict[sp]['body_pose'].reshape(F,21,3).clone()
        for i in ARM_IDS: b[:,i]=bp_final[:,i]
        out_dict[sp]['body_pose']=b.reshape(F,21,3) if out_dict[sp]['body_pose'].dim()==3 else b.reshape(F,63)
torch.save(out_dict,out_pt)
print(f'[zanchor_v4] saved {out_pt} (w_col={W_COL}) ({time.perf_counter()-t0:.1f}s)')
