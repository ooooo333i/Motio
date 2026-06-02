"""
일반화 fit2d: 손목 강한 2D fit + wrist-local 회전 평활(WSM=20) + 손가락 고정.
검증된 단일영상 방식 그대로. K/WB 동적 (영상별).
usage: python batch_fit2d.py <clip_dir>
  in : <clip_dir>/smplx_merged_hamer_post.pt , <clip_dir>/preprocess/wholebody_kp2d.npy
  out: <clip_dir>/fit2d.pt
"""
import sys, numpy as np, copy, time
sys.path.insert(0,'/workspace/capstone_deploy')
import torch, smplx

clip_dir=sys.argv[1].rstrip('/')
base_pt=f'{clip_dir}/smplx_merged_hamer_post.pt'
out_pt=f'{clip_dir}/fit2d.pt'
dev='cuda'
t0=time.perf_counter()
src=torch.load(base_pt,map_location='cpu')
inc=src['smpl_params_incam']
def g(k):
    v=inc[k]; return v
bp_raw=inc['body_pose']; F=bp_raw.shape[0]
bp0=bp_raw.reshape(F,21,3).clone()
K=src['K_fullimg']
if K.dim()==2: K=K.unsqueeze(0).expand(F,-1,-1)
K=K.float().numpy()
WB=np.load(f'{clip_dir}/preprocess/wholebody_kp2d.npy')   # (F,133,3)
assert WB.shape[0]==F, f'WB frames {WB.shape[0]} != pose frames {F}'

def fix(t,d):
    t=t if torch.is_tensor(t) else torch.tensor(t)
    t=t.float()
    if t.dim()==1: t=t.unsqueeze(0)
    if t.shape[0]==1 and F>1: t=t.expand(F,-1).contiguous()
    return t.to(dev)
betas=fix(inc['betas'],10); go=fix(inc['global_orient'],3); tr=fix(inc['transl'],3)
lhp=fix(inc['left_hand_pose'],45); rhp=fix(inc['right_hand_pose'],45)

m=smplx.create('/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz',model_type='smplx',
  gender='neutral',use_pca=False,flat_hand_mean=False,num_betas=betas.shape[1],batch_size=F).to(dev)

wr_R=bp0[:,20].clone().to(dev).requires_grad_(True)
wr_L=bp0[:,19].clone().to(dev).requires_grad_(True)
bp_fixed=bp0.clone().to(dev)
RID_S=[21,40,43,46,49,52]; RID_W=[112+i for i in [0,5,9,17,13,2]]
LID_S=[20,25,28,31,34,37]; LID_W=[91+i for i in [0,5,9,17,13,2]]
Kt=torch.tensor(K,dtype=torch.float32,device=dev)
WBt=torch.tensor(WB,dtype=torch.float32,device=dev)
def proj(J):
    z=J[...,2].clamp(min=1e-4)
    u=Kt[:,0,0:1]*J[...,0]/z+Kt[:,0,2:3]; v=Kt[:,1,1:2]*J[...,1]/z+Kt[:,1,2:3]
    return torch.stack([u,v],-1)

opt=torch.optim.Adam([wr_R,wr_L],lr=0.02)
W2D=1.0; WSM=20.0   # 강한 2D + wrist-local 회전 평활
for it in range(250):
    bp=bp_fixed.clone(); bp[:,20]=wr_R; bp[:,19]=wr_L
    out=m(betas=betas,body_pose=bp.reshape(F,63),global_orient=go,transl=tr,
          left_hand_pose=lhp,right_hand_pose=rhp)
    J=out.joints
    loss_2d=0.0
    for SID,WID in [(RID_S,RID_W),(LID_S,LID_W)]:
        pj=proj(J[:,SID]); tgt=WBt[:,WID,:2]; conf=WBt[:,WID,2:3].clamp(0,1)
        loss_2d=loss_2d+((conf*(pj-tgt)**2).mean())
    sm=((wr_R[1:]-wr_R[:-1])**2).mean()+((wr_L[1:]-wr_L[:-1])**2).mean()
    loss=W2D*loss_2d/1000.0+WSM*sm
    opt.zero_grad(); loss.backward(); opt.step()
    if it%50==0:
        print('  fit2d it%3d  2d_rmse=%.1fpx  wsm=%.4f'%(it,(loss_2d.item()/(6*2*2))**0.5,sm.item()))

out_dict=copy.deepcopy(src)
bpn=bp0.clone(); bpn[:,20]=wr_R.detach().cpu(); bpn[:,19]=wr_L.detach().cpu()
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp in out_dict:
        out_dict[sp]['body_pose']=bpn.reshape(F,21,3) if out_dict[sp]['body_pose'].dim()==3 else bpn.reshape(F,63)
torch.save(out_dict,out_pt)
print(f'[fit2d] saved {out_pt}  ({time.perf_counter()-t0:.1f}s)  FIT2D_SEC={time.perf_counter()-t0:.2f}')
