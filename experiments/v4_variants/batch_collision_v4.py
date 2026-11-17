"""
batch_collision_v4: fit2d 위에 손-손/팔-팔 관통만 최소 보정 (overlay 보존).
- 반발항: L팔 정점과 R팔 정점이 MARGIN보다 가까우면 (MARGIN-d)^2로 밀어냄 (normal 불필요, 미분가능).
- anchor(L2 to fit2d, 강함): 충돌 없는 곳은 안 건드림.
- 2D(hand+body, 가볍게): overlay 유지.  + temporal 평활.
입력: <clip>/fit2d.pt, preprocess/{vitpose.pt, wholebody_kp2d.npy}
출력: <clip>/collision_v4.pt
usage: python batch_collision_v4.py <clip_dir> [iters]
"""
import sys, numpy as np, copy, time
sys.path.insert(0, '/workspace/capstone_deploy')
import torch, smplx

clip = sys.argv[1].rstrip('/'); ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
dev = 'cuda'; out_pt = f'{clip}/collision_v4.pt'
SX = '/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz'
# weights
MARGIN = 0.015        # 15mm(손잡이 반경)보다 가까운 좌우팔 정점 = 밀어냄
W_COL = 300.0         # 반발 (프레임당 위반량 합, 선형 hinge)
W_ANCH = 30.0         # fit2d 유지 (충돌 없는 곳 고정)
W_2D = 0.5            # overlay 약하게 유지
W_TC = 200.0          # temporal
SIGMA = 100.0
NSUB = 250            # 좌우팔 정점 서브샘플

t0 = time.perf_counter()
src = torch.load(f'{clip}/fit2d.pt', map_location='cpu'); inc = src['smpl_params_incam']
bp0 = inc['body_pose'].reshape(-1, 21, 3).clone(); F = bp0.shape[0]
K = src['K_fullimg']; K = K.unsqueeze(0).expand(F, -1, -1) if K.dim() == 2 else K; Kt = K.float().to(dev)
WB = torch.tensor(np.load(f'{clip}/preprocess/wholebody_kp2d.npy'), dtype=torch.float32, device=dev)
vp = torch.load(f'{clip}/preprocess/vitpose.pt', map_location='cpu'); vp = (vp.numpy() if torch.is_tensor(vp) else np.array(vp)).astype(np.float32)
VPt = torch.tensor(vp, device=dev)
def fix(t):
    t = (t if torch.is_tensor(t) else torch.tensor(t)).float()
    if t.dim() == 1: t = t.unsqueeze(0)
    if t.shape[0] == 1 and F > 1: t = t.expand(F, -1).contiguous()
    return t.to(dev)
betas = fix(inc['betas']); go = fix(inc['global_orient']); tr = fix(inc['transl']); lhp = fix(inc['left_hand_pose']); rhp = fix(inc['right_hand_pose'])
m = smplx.create(SX, model_type='smplx', gender='neutral', use_pca=False, flat_hand_mean=False, num_betas=betas.shape[1], batch_size=F).to(dev)

# 부위 vertex (스키닝 argmax) + 서브샘플
W = np.load(SX, allow_pickle=True)['weights']; part = W.argmax(1)
def sub(idx):
    idx = np.asarray(idx);
    return idx[np.linspace(0, len(idx)-1, min(NSUB, len(idx))).astype(int)]
Lhand = np.where(np.isin(part, [20]+list(range(25,40))))[0]; Rhand = np.where(np.isin(part, [21]+list(range(40,55))))[0]
Lfore = np.where(part==18)[0]; Rfore = np.where(part==19)[0]
Larm = sub(np.concatenate([Lfore, Lhand])); Rarm = sub(np.concatenate([Rfore, Rhand]))
Lt = torch.tensor(Larm, device=dev); Rt = torch.tensor(Rarm, device=dev)

ARM_IDS = [15,16,17,18,19,20]   # shoulder/elbow/wrist (손 측면이동 위해 어깨도)
arm_idx_t = torch.tensor(ARM_IDS, device=dev)
body = bp0.clone().to(dev).requires_grad_(True); bp_orig = bp0.clone().to(dev)
mask = torch.zeros(21,1,device=dev)
for i in ARM_IDS: mask[i]=1.0
VP_BODY=[(5,16),(6,17),(7,18),(8,19),(9,20),(10,21)]; VPI=[c for c,_ in VP_BODY]; SJ=[s for _,s in VP_BODY]; SJt=torch.tensor(SJ,device=dev)
RID_S=[21,40,43,46,49,52]; RID_W=[112+i for i in [0,5,9,17,13,2]]; LID_S=[20,25,28,31,34,37]; LID_W=[91+i for i in [0,5,9,17,13,2]]
def forward(bp): return m(betas=betas, body_pose=bp.reshape(F,63), global_orient=go, transl=tr, left_hand_pose=lhp, right_hand_pose=rhp)
def proj(J3):
    z=J3[...,2].clamp(min=1e-4); u=Kt[:,0,0:1]*J3[...,0]/z+Kt[:,0,2:3]; v=Kt[:,1,1:2]*J3[...,1]/z+Kt[:,1,2:3]; return torch.stack([u,v],-1)
def aa2R(aa):
    th=aa.norm(dim=-1,keepdim=True).clamp(min=1e-8); ax=aa/th; x,y,z=ax[...,0],ax[...,1],ax[...,2]
    c=torch.cos(th)[...,0]; s=torch.sin(th)[...,0]; C=1-c
    return torch.stack([c+x*x*C,x*y*C-z*s,x*z*C+y*s,y*x*C+z*s,c+y*y*C,y*z*C-x*s,z*x*C-y*s,z*y*C+x*s,c+z*z*C],-1).reshape(aa.shape[:-1]+(3,3))
def geo(R1,R2):
    Rr=R1.transpose(-1,-2)@R2; t_=Rr[...,0,0]+Rr[...,1,1]+Rr[...,2,2]; return torch.acos(((t_-1)/2).clamp(-1+1e-6,1-1e-6)).mean()

opt = torch.optim.Adam([body], lr=0.01)
for it in range(ITERS):
    bp = bp_orig*(1-mask)+body*mask; out=forward(bp); Vv=out.vertices; J=out.joints
    VL=Vv[:,Lt]; VR=Vv[:,Rt]                          # (F,NSUB,3)
    D=torch.cdist(VL,VR)                              # (F,NSUB,NSUB)
    mindL=D.min(dim=2)[0]                             # 각 L정점의 최근접 R거리
    mindR=D.min(dim=1)[0]                             # 각 R정점의 최근접 L거리
    # 프레임당 위반량 합(선형 hinge), 양방향 대칭 → 두 팔 다 밀어냄
    Lcol=(torch.relu(MARGIN-mindL).sum(1)+torch.relu(MARGIN-mindR).sum(1)).mean()
    mind=mindL
    Lanch=(((body[:,arm_idx_t]-bp_orig[:,arm_idx_t])**2).sum(-1)).mean()
    # 2D (hand+body, 가볍게)
    uv=proj(J); pj=uv[:,SJt,:]; tgt=VPt[:,VPI,:2]; cf=VPt[:,VPI,2:3].clamp(0,1); cf=torch.where(cf>=0.3,cf,torch.zeros_like(cf))
    Lbody=(cf*((pj-tgt)**2).sum(-1,keepdim=True)).mean()/1000.0
    Lhand2d=0.0
    for SID,WID in [(RID_S,RID_W),(LID_S,LID_W)]:
        p=proj(J[:,SID]); o=WB[:,WID,:2]; c=WB[:,WID,2:3].clamp(0,1)
        Lhand2d=Lhand2d+(c*((p-o)**2).sum(-1,keepdim=True)).mean()/1000.0
    Ra=aa2R(body[:,arm_idx_t]); Ltc=geo(Ra[:-1],Ra[1:])
    loss=W_COL*Lcol + W_ANCH*Lanch + W_2D*(Lbody+Lhand2d) + W_TC*Ltc
    opt.zero_grad(); loss.backward()
    with torch.no_grad(): body.grad*=mask
    opt.step()
    if it%40==0 or it==ITERS-1:
        with torch.no_grad():
            npen=int((mind<MARGIN).sum().item())
        print('  col it%3d  col=%.6f anch=%.5f 2d=%.4f tc=%.4f  근접쌍<%dmm=%d'%(it,Lcol.item(),Lanch.item(),float(Lbody+Lhand2d),float(Ltc),int(MARGIN*1000),npen))

with torch.no_grad(): bp_final=(bp_orig*(1-mask)+body*mask).detach().cpu()
out_dict=copy.deepcopy(src)
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp in out_dict:
        b=out_dict[sp]['body_pose'].reshape(F,21,3).clone()
        for i in ARM_IDS: b[:,i]=bp_final[:,i]
        out_dict[sp]['body_pose']=b.reshape(F,21,3) if out_dict[sp]['body_pose'].dim()==3 else b.reshape(F,63)
torch.save(out_dict,out_pt)
print(f'[collision_v4] saved {out_pt} ({time.perf_counter()-t0:.1f}s)')
