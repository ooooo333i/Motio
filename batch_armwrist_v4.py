"""
batch_armwrist_v4 (실험2): 전완(collar/shoulder/elbow)+손목 동시 최적화.
가설: 손목만 fit하면 고정 GVHMR 전완 오차 때문에 손목이 deviation을 인체한계(±45°)
      너머로 꺾어 2D를 맞춘다(실험1에서 확인). 전완을 풀어주면 손목이 자연 범위로 돌아오는가?
에너지:
  + body 2D (ViTPose 어깨/팔꿈치/손목, COCO17)             ← 전완 방향 고정
  + hand 2D (WB 손목+5MCP, scale-norm gmof)               ← 손 swing
  + GVHMR prior(L2) + temporal(geodesic)                  ← depth flip(충돌) 방지
  + depth anchor (elbow/wrist cam-z → GVHMR)              ← 깊이 고정
  + wrist swing Huber + axial→GVHMR anchor                ← 스파이크/invisible 노이즈
  + ROM soft-limit (deviation>45°, flexion>90° 패널티)     ← 과굴절 방지
입력: <clip>/smplx_merged_hamer_post.pt, preprocess/{vitpose.pt, wholebody_kp2d.npy}
출력: <clip>/armwrist_v4.pt
usage: python batch_armwrist_v4.py <clip_dir> [iters]
"""
import sys, numpy as np, copy, time
sys.path.insert(0, '/workspace/capstone_deploy')
import torch, smplx

clip = sys.argv[1].rstrip('/'); ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
dev = 'cuda'
out_pt = f'{clip}/armwrist_v4.pt'

# ---- weights ----
W_BODY2D = 1.0       # ViTPose 팔 2D (px^2 /1000)
W_HAND2D = 6.0       # WB 손 2D (scale-norm)
SIGMA = 100.0
W_PRIOR = 3.0        # GVHMR arm pose L2 (depth flip 방지)
W_TC = 300.0         # temporal geodesic
W_DEPTH = 50.0       # cam-z anchor → GVHMR
W_SWING = 8.0; HUBER_DELTA = 0.04
W_AXIAL_SM = 30.0; W_AXIAL_ANCHOR = 4.0
W_ROM = 15.0         # deviation/flexion soft-limit
DEV_LIM = np.radians(45.0); FLEX_LIM = np.radians(90.0)

t0 = time.perf_counter()
src = torch.load(f'{clip}/smplx_merged_hamer_post.pt', map_location='cpu')
inc = src['smpl_params_incam']
bp0 = inc['body_pose'].reshape(-1, 21, 3).clone(); F = bp0.shape[0]
K = src['K_fullimg']; K = K.unsqueeze(0).expand(F, -1, -1) if K.dim() == 2 else K
Kt = K.float().to(dev)
WB = torch.tensor(np.load(f'{clip}/preprocess/wholebody_kp2d.npy'), dtype=torch.float32, device=dev)
vp = torch.load(f'{clip}/preprocess/vitpose.pt', map_location='cpu')
vp = (vp.numpy() if torch.is_tensor(vp) else np.array(vp)).astype(np.float32)
VPt = torch.tensor(vp, device=dev)   # (F,17,3)

def fix(t):
    t = (t if torch.is_tensor(t) else torch.tensor(t)).float()
    if t.dim() == 1: t = t.unsqueeze(0)
    if t.shape[0] == 1 and F > 1: t = t.expand(F, -1).contiguous()
    return t.to(dev)
betas = fix(inc['betas']); go = fix(inc['global_orient']); tr = fix(inc['transl'])
lhp = fix(inc['left_hand_pose']); rhp = fix(inc['right_hand_pose'])
m = smplx.create('/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz', model_type='smplx',
                 gender='neutral', use_pca=False, flat_hand_mean=False, num_betas=betas.shape[1], batch_size=F).to(dev)

# 최적화: 팔 chain rows. body_pose idx
ARM_IDS = [12, 13, 15, 16, 17, 18, 19, 20]
WR_IDS = [19, 20]
body = bp0.clone().to(dev).requires_grad_(True)
bp_orig = bp0.clone().to(dev)
mask = torch.zeros(21, 1, device=dev)
for i in ARM_IDS: mask[i] = 1.0

# 2D 대응
VP_BODY = [(5,16),(6,17),(7,18),(8,19),(9,20),(10,21)]   # (coco, smplx out joint)
VPI = [c for c,_ in VP_BODY]; SJ = [s for _,s in VP_BODY]
RID_S=[21,40,43,46,49,52]; RID_W=[112+i for i in [0,5,9,17,13,2]]
LID_S=[20,25,28,31,34,37]; LID_W=[91+i for i in [0,5,9,17,13,2]]

def forward(bp):
    return m(betas=betas, body_pose=bp.reshape(F,63), global_orient=go, transl=tr,
             left_hand_pose=lhp, right_hand_pose=rhp)
def proj(J3):
    z = J3[...,2].clamp(min=1e-4)
    u = Kt[:,0,0:1]*J3[...,0]/z + Kt[:,0,2:3]; v = Kt[:,1,1:2]*J3[...,1]/z + Kt[:,1,2:3]
    return torch.stack([u,v],-1)
def gmof(r2): return (SIGMA**2*r2)/(SIGMA**2+r2)
def huber(d,delta):
    a=d.abs(); return torch.where(a<delta, 0.5*d**2, delta*(a-0.5*delta))
def aa2R(aa):
    th=aa.norm(dim=-1,keepdim=True).clamp(min=1e-8); ax=aa/th
    x,y,z=ax[...,0],ax[...,1],ax[...,2]; c=torch.cos(th)[...,0]; s=torch.sin(th)[...,0]; C=1-c
    return torch.stack([c+x*x*C,x*y*C-z*s,x*z*C+y*s, y*x*C+z*s,c+y*y*C,y*z*C-x*s,
                        z*x*C-y*s,z*y*C+x*s,c+z*z*C],-1).reshape(aa.shape[:-1]+(3,3))
def geo(R1,R2):
    Rr=R1.transpose(-1,-2)@R2; tr_=Rr[...,0,0]+Rr[...,1,1]+Rr[...,2,2]
    return torch.acos(((tr_-1)/2).clamp(-1+1e-6,1-1e-6)).mean()

# GVHMR 원본 cam-z (depth anchor 기준)
with torch.no_grad():
    J0 = forward(bp_orig).joints
    z_gv = J0[:, SJ, 2].detach()   # (F,6)
SJt = torch.tensor(SJ, device=dev)
arm_idx_t = torch.tensor(ARM_IDS, device=dev)
gv_axial = bp_orig[:, WR_IDS, 0].detach()   # (F,2)

opt = torch.optim.Adam([body], lr=0.02)
for it in range(ITERS):
    bp = bp_orig*(1-mask) + body*mask
    out = forward(bp); J = out.joints
    uv = proj(J)
    # body 2D
    pj = uv[:, SJt, :]; tgt = VPt[:, VPI, :2]; conf = (VPt[:, VPI, 2:3]).clamp(0,1)
    conf = torch.where(conf>=0.3, conf, torch.zeros_like(conf))
    Lbody = (conf*((pj-tgt)**2).sum(-1,keepdim=True)).mean()/1000.0
    # hand 2D (scale-norm gmof)
    Lhand = 0.0
    for SID,WID in [(RID_S,RID_W),(LID_S,LID_W)]:
        p = proj(J[:,SID]); obs=WB[:,WID,:2]; cf=WB[:,WID,2:3].clamp(0,1)
        c=obs.mean(1,keepdim=True); mn=obs.min(1,keepdim=True)[0]; mx=obs.max(1,keepdim=True)[0]
        sc=torch.sqrt(((mx-mn)**2).sum(-1,keepdim=True)).clamp(min=1.0)
        r2=(((p-c)/sc-(obs-c)/sc)**2).sum(-1,keepdim=True)
        Lhand = Lhand + (cf**2*gmof(r2*sc*sc)).mean()/1000.0   # 다시 px 스케일로
    # prior + temporal
    Lprior = (((body[:,arm_idx_t]-bp_orig[:,arm_idx_t])**2).sum(-1)).mean()
    Ra = aa2R(body[:,arm_idx_t]); Ltc = geo(Ra[:-1],Ra[1:])
    # depth anchor
    Ldepth = ((J[:,SJt,2]-z_gv)**2).mean()
    # wrist swing/axial
    sw=0.0; axsm=0.0; axan=0.0; rom=0.0
    for k,wi in enumerate(WR_IDS):
        w=body[:,wi]
        sw = sw + huber(w[1:,1:]-w[:-1,1:], HUBER_DELTA).mean()
        axsm = axsm + ((w[1:,0]-w[:-1,0])**2).mean()
        axan = axan + ((w[:,0]-gv_axial[:,k])**2).mean()
        rom = rom + (torch.relu(w[:,1].abs()-DEV_LIM)**2).mean() + (torch.relu(w[:,2].abs()-FLEX_LIM)**2).mean()
    loss = (W_BODY2D*Lbody + W_HAND2D*Lhand + W_PRIOR*Lprior + W_TC*Ltc + W_DEPTH*Ldepth
            + W_SWING*sw + W_AXIAL_SM*axsm + W_AXIAL_ANCHOR*axan + W_ROM*rom)
    opt.zero_grad(); loss.backward()
    with torch.no_grad():
        body.grad *= mask
    opt.step()
    if it%80==0 or it==ITERS-1:
        print('  aw it%3d  body=%.3f hand=%.3f prior=%.4f tc=%.4f depth=%.4f swing=%.4f rom=%.4f'
              %(it,Lbody.item(),Lhand.item(),Lprior.item(),float(Ltc),Ldepth.item(),sw.item(),float(rom)))

with torch.no_grad():
    bp_final = (bp_orig*(1-mask)+body*mask).detach().cpu()
out_dict = copy.deepcopy(src)
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp in out_dict:
        b = out_dict[sp]['body_pose'].reshape(F,21,3).clone()
        for i in ARM_IDS: b[:,i] = bp_final[:,i]
        out_dict[sp]['body_pose'] = b.reshape(F,21,3) if out_dict[sp]['body_pose'].dim()==3 else b.reshape(F,63)
torch.save(out_dict, out_pt)
print(f'[armwrist_v4] saved {out_pt} ({time.perf_counter()-t0:.1f}s)')

# 진단
from scipy.spatial.transform import Rotation
from physical_limits import decompose_wrist_local, JOINT_LIMITS
lim=JOINT_LIMITS['wrist']; bpc=bp_final.numpy()
for nm,wr in [('R',20),('L',19)]:
    Rr=Rotation.from_rotvec(bpc[:,wr]).as_matrix()
    v=np.array([0]+[np.degrees(np.linalg.norm(Rotation.from_matrix(Rr[f-1].T@Rr[f]).as_rotvec())) for f in range(1,F)])
    dev_bad=sum(1 for f in range(F) if not(lim['deviation']['hard_neg']<=decompose_wrist_local(bpc[f,wr])['deviation']<=lim['deviation']['hard_pos']))
    print(f'[diag {nm}] vel max={v.max():.0f} >33:{int((v>33.3).sum())}  deviation>±45: {dev_bad}f')
