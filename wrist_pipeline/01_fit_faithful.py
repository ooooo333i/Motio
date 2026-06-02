"""
Dyn-HaMR 충실 재현 (실제 활성 구성만):
  - joints2d = 10000 (hand bbox로 정규화 + gmof robust sigma=100)   ← 지배
  - joints3d_smooth = 1000(stage0) -> 10000(stage1) (hand 3D scale 정규화)
  - 2-stage: Root(wrist만,50it,j3ds=1000) -> Smooth(wrist만,300it,j3ds=10000)
  - 회전평활/bone_length/joint_consistency/motion_prior(HMP) 전부 없음 (원본 코드도 비활성)
  - 손가락은 HaMeR(pp) 고정 (pose_prior가 손가락 pin하는 것과 동등, grip 유지)
실험 목적: 원본 그대로면 우리 케이스(작은 손)에서 손목이 잡히는가?
"""
import sys, numpy as np, copy
sys.path.insert(0,'/workspace/capstone_deploy')
import torch, smplx
from scipy.spatial.transform import Rotation

dev='cuda'
base='/workspace/capstone_deploy/outputs/cap_pipeline/input/'
src=torch.load(base+'cmp_baseline_hamer.pt',map_location='cpu')
inc=src['smpl_params_incam']
F=inc['body_pose'].shape[0]
K=src['K_fullimg'].numpy()
WB=np.load(base+'preprocess/wholebody_kp2d.npy')   # (F,133,3)

m=smplx.create('/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz',model_type='smplx',
  gender='neutral',use_pca=False,flat_hand_mean=False,num_betas=10,batch_size=F).to(dev)
betas=inc['betas'].to(dev); go=inc['global_orient'].to(dev); tr=inc['transl'].to(dev)
lhp=inc['left_hand_pose'].to(dev); rhp=inc['right_hand_pose'].to(dev)
bp0=inc['body_pose'].reshape(F,21,3).clone()
wr_R=bp0[:,20].clone().to(dev).requires_grad_(True)
wr_L=bp0[:,19].clone().to(dev).requires_grad_(True)
bp_fixed=bp0.clone().to(dev)

# 대응: wrist + 5 MCP (palm orientation 결정) — 2D & 3D_smooth 공통
RID_S=[21,40,43,46,49,52]; RID_W=[112+i for i in [0,5,9,17,13,2]]
LID_S=[20,25,28,31,34,37]; LID_W=[91+i for i in [0,5,9,17,13,2]]
Kt=torch.tensor(K,dtype=torch.float32,device=dev)
WBt=torch.tensor(WB,dtype=torch.float32,device=dev)
SIGMA=100.0
def proj(J):
    z=J[...,2].clamp(min=1e-4)
    u=Kt[:,0,0:1]*J[...,0]/z+Kt[:,0,2:3]; v=Kt[:,1,1:2]*J[...,1]/z+Kt[:,1,2:3]
    return torch.stack([u,v],-1)
def gmof(res2): return (SIGMA**2*res2)/(SIGMA**2+res2)

def loss_2d_side(J,SID,WID):
    pj=proj(J[:,SID])                  # (F,6,2)
    obs=WBt[:,WID,:2]; conf=WBt[:,WID,2:3].clamp(0,1)   # (F,6,1)
    # hand bbox scale 정규화 (obs 기준, pred도 동일 center/scale)
    c=obs.mean(1,keepdim=True)
    mn=obs.min(1,keepdim=True)[0]; mx=obs.max(1,keepdim=True)[0]
    sc=torch.sqrt(((mx-mn)**2).sum(-1,keepdim=True)).clamp(min=1.0)  # (F,1,1)
    pn=(pj-c)/sc; on=(obs-c)/sc
    res2=((pn-on)**2).sum(-1,keepdim=True)             # (F,6,1)
    return (conf**2*gmof(res2)).mean()

def loss_j3ds_side(J,SID):
    j=J[:,SID]                          # (F,6,3)
    mn=j.min(1,keepdim=True)[0]; mx=j.max(1,keepdim=True)[0]
    hs=torch.sqrt(((mx-mn)**2).sum(-1,keepdim=True)).clamp(min=1e-3,max=1.0)
    jn=j/hs
    d=jn[1:]-jn[:-1]
    return 0.5*(d**2).mean()

def run_stage(iters,w_j3ds,lr):
    opt=torch.optim.LBFGS([wr_R,wr_L],max_iter=20,lr=lr,line_search_fn='strong_wolfe')
    for it in range(iters):
        def closure():
            opt.zero_grad()
            bp=bp_fixed.clone(); bp[:,20]=wr_R; bp[:,19]=wr_L
            out=m(betas=betas,body_pose=bp.reshape(F,63),global_orient=go,transl=tr,
                  left_hand_pose=lhp,right_hand_pose=rhp)
            J=out.joints
            l2d=loss_2d_side(J,RID_S,RID_W)+loss_2d_side(J,LID_S,LID_W)
            lj3=loss_j3ds_side(J,RID_S)+loss_j3ds_side(J,LID_S)
            loss=10000.0*l2d + w_j3ds*lj3
            loss.backward()
            return loss
        opt.step(closure)
        if it%10==0 or it==iters-1:
            with torch.no_grad():
                bp=bp_fixed.clone(); bp[:,20]=wr_R; bp[:,19]=wr_L
                out=m(betas=betas,body_pose=bp.reshape(F,63),global_orient=go,transl=tr,
                      left_hand_pose=lhp,right_hand_pose=rhp)
                J=out.joints
                l2d=(loss_2d_side(J,RID_S,RID_W)+loss_2d_side(J,LID_S,LID_W)).item()
                lj3=(loss_j3ds_side(J,RID_S)+loss_j3ds_side(J,LID_S)).item()
            print('  it%3d  2d=%.4f j3ds=%.5f'%(it,l2d,lj3))

print('=== Stage 0 (Root): wrist, 50 it, j3ds=1000 ===')
run_stage(50,1000.0,1.0)
print('=== Stage 1 (Smooth): wrist, 300 it -> 100 it, j3ds=10000 ===')
run_stage(100,10000.0,1.0)

out_pt=copy.deepcopy(src)
bpn=bp0.clone(); bpn[:,20]=wr_R.detach().cpu(); bpn[:,19]=wr_L.detach().cpu()
for sp in ('smpl_params_incam','smpl_params_global'):
    out_pt[sp]['body_pose']=bpn.reshape(F,21,3)
torch.save(out_pt,base+'cmp_dynhamr_faithful.pt')
print('saved cmp_dynhamr_faithful.pt')

# 진단: axial twist + velocity
def stw(aa):
    q=Rotation.from_rotvec(aa).as_quat();d=np.degrees(2*np.arctan2(q[0],q[3]));return (d+180)%360-180
bpc=bpn.numpy()
def vmax(wr):
    Rr=Rotation.from_rotvec(bpc[:,wr]).as_matrix()
    v=[np.degrees(np.linalg.norm(Rotation.from_matrix(Rr[f-1].T@Rr[f]).as_rotvec())) for f in range(1,F)]
    return max(v),sum(1 for x in v if x>33)
print('f245 axial: R=%.0f L=%.0f (HaMeR R=-19/L=-83, GVHMR정답 R=-11/L=2)'%(stw(bpc[245,20]),stw(bpc[245,19])))
print('velocity: R max=%.0f >33:%d | L max=%.0f >33:%d'%(*vmax(20),*vmax(19)))
ham=src['smpl_params_incam']['body_pose'].reshape(F,21,3).numpy()
print('axial range R: faithful=[%.0f,%.0f] HaMeR=[%.0f,%.0f]'%(
  min(stw(bpc[f,20]) for f in range(200,430)),max(stw(bpc[f,20]) for f in range(200,430)),
  min(stw(ham[f,20]) for f in range(200,430)),max(stw(ham[f,20]) for f in range(200,430))))
