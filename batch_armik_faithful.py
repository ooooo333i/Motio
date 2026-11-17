"""
batch_armik_faithful: armik 팔(IK) 위에 Dyn-HaMR '충실' 손목 fit (overlay 최상).
 - 팔(collar/shoulder/elbow) = arm_prefit (손목 위치 교정됨).
 - 손가락 = _post (HaMeR 후처리).
 - 손목 회전 = 01_fit_faithful 방식: scale-norm 2D(hand-bbox center/scale, gmof σ100, conf²)
              + scale-norm joints3d_smooth. 2-stage LBFGS. (WSM 대신 충실판)
입력: <clip>/{smplx_merged_hamer_post.pt, smplx_arm_prefit.pt, preprocess/wholebody_kp2d.npy}
출력: <clip>/armik_faithful.pt
usage: python batch_armik_faithful.py <clip_dir>
"""
import sys, numpy as np, copy, time
sys.path.insert(0,'/workspace/capstone_deploy')
import torch, smplx
from scipy.spatial.transform import Rotation
clip=sys.argv[1].rstrip('/'); WSM=float(sys.argv[2]) if len(sys.argv)>2 else 0.0; dev='cuda'; out_pt=f'{clip}/armik_faithful.pt'
SX='/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz'
t0=time.perf_counter()
src=torch.load(f'{clip}/smplx_merged_hamer_post.pt',map_location='cpu'); inc=src['smpl_params_incam']
arm=torch.load(f'{clip}/smplx_arm_prefit.pt',map_location='cpu')['smpl_params_incam']['body_pose'].reshape(-1,21,3)
bp0=inc['body_pose'].reshape(-1,21,3).clone(); F=bp0.shape[0]
for i in [12,13,15,16,17,18,19,20]: bp0[:,i]=arm[:,i]   # 교정된 팔+손목 init
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
RID_S=[21,40,43,46,49,52]; RID_W=[112+i for i in [0,5,9,17,13,2]]; LID_S=[20,25,28,31,34,37]; LID_W=[91+i for i in [0,5,9,17,13,2]]
Kt=torch.tensor(K,dtype=torch.float32,device=dev); WBt=torch.tensor(WB,dtype=torch.float32,device=dev); SIGMA=100.0
def proj(J):
    z=J[...,2].clamp(min=1e-4); u=Kt[:,0,0:1]*J[...,0]/z+Kt[:,0,2:3]; v=Kt[:,1,1:2]*J[...,1]/z+Kt[:,1,2:3]; return torch.stack([u,v],-1)
def gmof(r2): return (SIGMA**2*r2)/(SIGMA**2+r2)
def loss_2d_side(J,SID,WID):           # scale-norm 2D (크기불변) + gmof + conf^2
    pj=proj(J[:,SID]); obs=WBt[:,WID,:2]; conf=WBt[:,WID,2:3].clamp(0,1)
    c=obs.mean(1,keepdim=True); mn=obs.min(1,keepdim=True)[0]; mx=obs.max(1,keepdim=True)[0]
    sc=torch.sqrt(((mx-mn)**2).sum(-1,keepdim=True)).clamp(min=1.0)
    res2=(((pj-c)/sc-(obs-c)/sc)**2).sum(-1,keepdim=True)
    return (conf**2*gmof(res2)).mean()
def loss_j3ds_side(J,SID):             # scale-norm 3D joint smoothness
    j=J[:,SID]; mn=j.min(1,keepdim=True)[0]; mx=j.max(1,keepdim=True)[0]
    hs=torch.sqrt(((mx-mn)**2).sum(-1,keepdim=True)).clamp(min=1e-3,max=1.0)
    d=(j/hs)[1:]-(j/hs)[:-1]; return 0.5*(d**2).mean()
def run_stage(iters,w_j3ds,lr):
    opt=torch.optim.LBFGS([wr_R,wr_L],max_iter=20,lr=lr,line_search_fn='strong_wolfe')
    for it in range(iters):
        def closure():
            opt.zero_grad(); bp=bp_fixed.clone(); bp[:,20]=wr_R; bp[:,19]=wr_L
            out=m(betas=betas,body_pose=bp.reshape(F,63),global_orient=go,transl=tr,left_hand_pose=lhp,right_hand_pose=rhp); J=out.joints
            l2d=loss_2d_side(J,RID_S,RID_W)+loss_2d_side(J,LID_S,LID_W)
            lj3=loss_j3ds_side(J,RID_S)+loss_j3ds_side(J,LID_S)
            sm=((wr_R[1:]-wr_R[:-1])**2).mean()+((wr_L[1:]-wr_L[:-1])**2).mean()
            loss=10000.0*l2d+w_j3ds*lj3+WSM*sm; loss.backward(); return loss
        opt.step(closure)
print('=== Stage0 Root (j3ds=1000, 50it) ==='); run_stage(50,1000.0,1.0)
print('=== Stage1 Smooth (j3ds=10000, 100it) ==='); run_stage(100,10000.0,1.0)
out_dict=copy.deepcopy(src)
bpn=bp0.clone(); bpn[:,20]=wr_R.detach().cpu(); bpn[:,19]=wr_L.detach().cpu()
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp in out_dict:
        out_dict[sp]['body_pose']=bpn.reshape(F,21,3) if out_dict[sp]['body_pose'].dim()==3 else bpn.reshape(F,63)
torch.save(out_dict,out_pt)
print(f'[armik_faithful] saved {out_pt} ({time.perf_counter()-t0:.1f}s)')
