"""
batch_adaptive_v4 (실험3): 신뢰도-게이트 손목.
원하는 거동: 손/손목이 잘 보이는 구간엔 2D 손을 바짝 따라가고(=fit2d 충실도),
            갑자기 안 보이거나 튀는 구간엔 전완을 추적하며 손목은 GVHMR로 잡아둠(=armwrist_v4 안정).

trust[f] (per hand): WB 손 conf 높음 ∧ fit2d 손목 안튐(vel) ∧ fit2d 손목 ROM 정상.
  fit2d.pt = 이미 만들어진 '공격적 2D 추적' 솔루션 → 그게 튀고/과굴절하는 프레임 = 신뢰 못함.
  → bad 프레임 dilate+gaussian ramp, trust = (1-bad)*conf.
per-frame:
  + 손2D 가중 = W_HAND2D * trust            (좋은 구간만 손 바짝 추적)
  + 손목→GVHMR full-local anchor = W_ANCH*(1-trust)  (나쁜 구간만 전완으로 잡아둠)
  + 팔(collar/sh/elbow): 항상 body2D+prior+temporal+depth (안정 전완)
  + wrist swing Huber + axial→GVHMR + ROM soft-limit(안전캡)
입력: <clip>/{smplx_merged_hamer_post.pt, fit2d.pt, preprocess/{vitpose.pt, wholebody_kp2d.npy}}
출력: <clip>/adaptive_v4.pt
usage: python batch_adaptive_v4.py <clip_dir> [iters]
"""
import sys, numpy as np, copy, time
sys.path.insert(0, '/workspace/capstone_deploy')
import torch, smplx
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d, binary_dilation
from physical_limits import decompose_wrist_local, JOINT_LIMITS

clip = sys.argv[1].rstrip('/'); ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
dev = 'cuda'; out_pt = f'{clip}/adaptive_v4.pt'

# weights
W_BODY2D=1.0; W_HAND2D=12.0; SIGMA=100.0
W_PRIOR=3.0; W_TC=300.0; W_DEPTH=50.0
W_SWING=6.0; HUBER_DELTA=0.04; W_AXIAL_SM=20.0; W_AXIAL_ANCHOR=4.0
W_ANCH=10.0          # 나쁜 프레임 손목 full-local → GVHMR
W_ROM=15.0; DEV_LIM=np.radians(45.0); FLEX_LIM=np.radians(90.0)
# trust 판정 임계
CONF_LO=0.40; VEL_BAD=28.0

t0=time.perf_counter()
src=torch.load(f'{clip}/smplx_merged_hamer_post.pt',map_location='cpu'); inc=src['smpl_params_incam']
bp0=inc['body_pose'].reshape(-1,21,3).clone(); F=bp0.shape[0]
K=src['K_fullimg']; K=K.unsqueeze(0).expand(F,-1,-1) if K.dim()==2 else K; Kt=K.float().to(dev)
WB=np.load(f'{clip}/preprocess/wholebody_kp2d.npy'); WBt=torch.tensor(WB,dtype=torch.float32,device=dev)
vp=torch.load(f'{clip}/preprocess/vitpose.pt',map_location='cpu'); vp=(vp.numpy() if torch.is_tensor(vp) else np.array(vp)).astype(np.float32)
VPt=torch.tensor(vp,device=dev)
fit2d=torch.load(f'{clip}/fit2d.pt',map_location='cpu')['smpl_params_incam']['body_pose'].reshape(F,21,3).numpy()

RID_S=[21,40,43,46,49,52]; RID_W=[112+i for i in [0,5,9,17,13,2]]
LID_S=[20,25,28,31,34,37]; LID_W=[91+i for i in [0,5,9,17,13,2]]

# ---- trust[f] per hand (R=20/RID_W, L=19/LID_W) ----
def build_trust(wr, hand_w):
    conf=WB[:,hand_w,2].mean(1)                     # (F,) 평균 손 conf
    Rr=Rotation.from_rotvec(fit2d[:,wr]).as_matrix()
    vel=np.array([0]+[np.degrees(np.linalg.norm(Rotation.from_matrix(Rr[f-1].T@Rr[f]).as_rotvec())) for f in range(1,F)])
    lim=JOINT_LIMITS['wrist']
    rom_bad=np.array([(not(lim['deviation']['hard_neg']<=decompose_wrist_local(fit2d[f,wr])['deviation']<=lim['deviation']['hard_pos']))
                      or (not(lim['flexion']['hard_neg']<=decompose_wrist_local(fit2d[f,wr])['flexion']<=lim['flexion']['hard_pos']))
                      for f in range(F)])
    # 속도는 '신뢰도까지 낮을 때만' bad (진짜 빠른 스윙=고conf는 신뢰). ROM위반/저conf는 항상 bad.
    bad=(conf<CONF_LO)|rom_bad|((vel>VEL_BAD)&(conf<0.6))
    bad=binary_dilation(bad,iterations=2)
    bad_ramp=gaussian_filter1d(bad.astype(float),2.0); bad_ramp=np.clip(bad_ramp/max(bad_ramp.max(),1e-6),0,1)
    trust=(1-bad_ramp)*np.clip(conf/0.6,0,1)
    trust=gaussian_filter1d(trust,1.5,mode='nearest'); trust=np.clip(trust,0,1)
    return trust, int(bad.sum())
trustR,nbR=build_trust(20,RID_W); trustL,nbL=build_trust(19,LID_W)
print(f'[trust] R: bad={nbR}f mean_trust={trustR.mean():.2f}  L: bad={nbL}f mean_trust={trustL.mean():.2f}')
tR=torch.tensor(trustR,dtype=torch.float32,device=dev); tL=torch.tensor(trustL,dtype=torch.float32,device=dev)

def fix(t):
    t=(t if torch.is_tensor(t) else torch.tensor(t)).float()
    if t.dim()==1:t=t.unsqueeze(0)
    if t.shape[0]==1 and F>1:t=t.expand(F,-1).contiguous()
    return t.to(dev)
betas=fix(inc['betas']);go=fix(inc['global_orient']);tr=fix(inc['transl']);lhp=fix(inc['left_hand_pose']);rhp=fix(inc['right_hand_pose'])
m=smplx.create('/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz',model_type='smplx',gender='neutral',use_pca=False,flat_hand_mean=False,num_betas=betas.shape[1],batch_size=F).to(dev)

ARM_IDS=[12,13,15,16,17,18,19,20]; WR_IDS=[19,20]
body=bp0.clone().to(dev).requires_grad_(True); bp_orig=bp0.clone().to(dev)
mask=torch.zeros(21,1,device=dev)
for i in ARM_IDS: mask[i]=1.0
VP_BODY=[(5,16),(6,17),(7,18),(8,19),(9,20),(10,21)]; VPI=[c for c,_ in VP_BODY]; SJ=[s for _,s in VP_BODY]; SJt=torch.tensor(SJ,device=dev)
arm_idx_t=torch.tensor(ARM_IDS,device=dev)
def forward(bp): return m(betas=betas,body_pose=bp.reshape(F,63),global_orient=go,transl=tr,left_hand_pose=lhp,right_hand_pose=rhp)
def proj(J3):
    z=J3[...,2].clamp(min=1e-4); u=Kt[:,0,0:1]*J3[...,0]/z+Kt[:,0,2:3]; v=Kt[:,1,1:2]*J3[...,1]/z+Kt[:,1,2:3]; return torch.stack([u,v],-1)
def gmof(r2): return (SIGMA**2*r2)/(SIGMA**2+r2)
def huber(d,delta): a=d.abs(); return torch.where(a<delta,0.5*d**2,delta*(a-0.5*delta))
def aa2R(aa):
    th=aa.norm(dim=-1,keepdim=True).clamp(min=1e-8); ax=aa/th; x,y,z=ax[...,0],ax[...,1],ax[...,2]
    c=torch.cos(th)[...,0]; s=torch.sin(th)[...,0]; C=1-c
    return torch.stack([c+x*x*C,x*y*C-z*s,x*z*C+y*s,y*x*C+z*s,c+y*y*C,y*z*C-x*s,z*x*C-y*s,z*y*C+x*s,c+z*z*C],-1).reshape(aa.shape[:-1]+(3,3))
def geo(R1,R2):
    Rr=R1.transpose(-1,-2)@R2; t_=Rr[...,0,0]+Rr[...,1,1]+Rr[...,2,2]; return torch.acos(((t_-1)/2).clamp(-1+1e-6,1-1e-6)).mean()
with torch.no_grad():
    z_gv=forward(bp_orig).joints[:,SJ,2].detach()
gv_wrist=bp_orig[:,WR_IDS,:].detach()    # (F,2,3) full local
trust_h={20:tR,19:tL}

opt=torch.optim.Adam([body],lr=0.02)
for it in range(ITERS):
    bp=bp_orig*(1-mask)+body*mask; out=forward(bp); J=out.joints; uv=proj(J)
    pj=uv[:,SJt,:]; tgt=VPt[:,VPI,:2]; conf=VPt[:,VPI,2:3].clamp(0,1); conf=torch.where(conf>=0.3,conf,torch.zeros_like(conf))
    Lbody=(conf*((pj-tgt)**2).sum(-1,keepdim=True)).mean()/1000.0
    # hand2D, trust 게이트 (per hand)
    Lhand=0.0
    for SID,WID,th in [(RID_S,RID_W,tR),(LID_S,LID_W,tL)]:
        p=proj(J[:,SID]); obs=WBt[:,WID,:2]; cf=WBt[:,WID,2:3].clamp(0,1)
        c=obs.mean(1,keepdim=True); mn=obs.min(1,keepdim=True)[0]; mx=obs.max(1,keepdim=True)[0]
        sc=torch.sqrt(((mx-mn)**2).sum(-1,keepdim=True)).clamp(min=1.0)
        r2=(((p-c)/sc-(obs-c)/sc)**2).sum(-1,keepdim=True)
        per=(cf**2*gmof(r2*sc*sc)).mean(dim=(1,2))/1000.0   # (F,)
        Lhand=Lhand+(per*th).mean()                          # trust 가중
    Lprior=(((body[:,arm_idx_t]-bp_orig[:,arm_idx_t])**2).sum(-1)).mean()
    Ra=aa2R(body[:,arm_idx_t]); Ltc=geo(Ra[:-1],Ra[1:])
    Ldepth=((J[:,SJt,2]-z_gv)**2).mean()
    sw=0.0; axsm=0.0; axan=0.0; rom=0.0; anch=0.0
    for k,wi in enumerate(WR_IDS):
        w=body[:,wi]; th=trust_h[wi]
        sw=sw+huber(w[1:,1:]-w[:-1,1:],HUBER_DELTA).mean()
        axsm=axsm+((w[1:,0]-w[:-1,0])**2).mean()
        axan=axan+((w[:,0]-gv_wrist[:,k,0])**2).mean()
        rom=rom+(torch.relu(w[:,1].abs()-DEV_LIM)**2).mean()+(torch.relu(w[:,2].abs()-FLEX_LIM)**2).mean()
        # 나쁜 프레임(1-trust): 손목 full-local → GVHMR
        anch=anch+(((w-gv_wrist[:,k,:])**2).sum(-1)*(1-th)).mean()
    loss=(W_BODY2D*Lbody+W_HAND2D*Lhand+W_PRIOR*Lprior+W_TC*Ltc+W_DEPTH*Ldepth
          +W_SWING*sw+W_AXIAL_SM*axsm+W_AXIAL_ANCHOR*axan+W_ANCH*anch+W_ROM*rom)
    opt.zero_grad(); loss.backward()
    with torch.no_grad(): body.grad*=mask
    opt.step()
    if it%80==0 or it==ITERS-1:
        print('  adp it%3d body=%.3f hand=%.4f anch=%.4f tc=%.4f depth=%.4f rom=%.4f'%(it,Lbody.item(),float(Lhand),float(anch),float(Ltc),Ldepth.item(),float(rom)))

with torch.no_grad(): bp_final=(bp_orig*(1-mask)+body*mask).detach().cpu()
out_dict=copy.deepcopy(src)
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp in out_dict:
        b=out_dict[sp]['body_pose'].reshape(F,21,3).clone()
        for i in ARM_IDS: b[:,i]=bp_final[:,i]
        out_dict[sp]['body_pose']=b.reshape(F,21,3) if out_dict[sp]['body_pose'].dim()==3 else b.reshape(F,63)
torch.save(out_dict,out_pt)
print(f'[adaptive_v4] saved {out_pt} ({time.perf_counter()-t0:.1f}s)')
# 진단
lim=JOINT_LIMITS['wrist']; bpc=bp_final.numpy()
for nm,wr in [('R',20),('L',19)]:
    Rr=Rotation.from_rotvec(bpc[:,wr]).as_matrix()
    v=np.array([0]+[np.degrees(np.linalg.norm(Rotation.from_matrix(Rr[f-1].T@Rr[f]).as_rotvec())) for f in range(1,F)])
    rb=sum(1 for f in range(F) if not(lim['deviation']['hard_neg']<=decompose_wrist_local(bpc[f,wr])['deviation']<=lim['deviation']['hard_pos'] and lim['flexion']['hard_neg']<=decompose_wrist_local(bpc[f,wr])['flexion']<=lim['flexion']['hard_pos']))
    print(f'[diag {nm}] vel max={v.max():.0f} >33:{int((v>33.3).sum())}  ROM위반:{rb}f')
