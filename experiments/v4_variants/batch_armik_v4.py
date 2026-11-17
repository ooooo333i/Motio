"""
batch_armik_v4: 사용자 아이디어 = 'HaMeR 손목 위치 추종 + 자연스러운 팔 IK'.
 Stage A (외부): cap_arm_prefit.py → smplx_arm_prefit.pt (어깨/팔꿈치/손목을 ViTPose 2D에 IK, 자연prior+temporal)
 Stage B (이 스크립트): _post(HaMeR 손가락) 위에 arm_prefit의 팔(collar/shoulder/elbow)을 얹고,
                        손목 회전만 WB 손 2D에 fit (위치는 A가 줌, 방향만 여기서).
입력: <clip>/{smplx_merged_hamer_post.pt, smplx_arm_prefit.pt, preprocess/wholebody_kp2d.npy}
출력: <clip>/armik_v4.pt
usage: python batch_armik_v4.py <clip_dir> [iters]
"""
import sys, numpy as np, copy, time
sys.path.insert(0,'/workspace/capstone_deploy')
import torch, smplx
clip=sys.argv[1].rstrip('/'); ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 250
dev='cuda'; out_pt=f'{clip}/armik_v4.pt'; SX='/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz'
W2D=1.0; WSM=20.0
t0=time.perf_counter()
src=torch.load(f'{clip}/smplx_merged_hamer_post.pt',map_location='cpu'); inc=src['smpl_params_incam']
arm=torch.load(f'{clip}/smplx_arm_prefit.pt',map_location='cpu')['smpl_params_incam']['body_pose'].reshape(-1,21,3)
bp0=inc['body_pose'].reshape(-1,21,3).clone(); F=bp0.shape[0]
# ★ 교정된 팔(collar/shoulder/elbow) 주입 (손목은 아래서 fit, 손가락은 _post 유지)
for i in [12,13,15,16,17,18]: bp0[:,i]=arm[:,i]
bp0[:,19]=arm[:,19]; bp0[:,20]=arm[:,20]   # 손목 init도 arm_prefit(coarse)에서
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
Kt=torch.tensor(K,dtype=torch.float32,device=dev); WBt=torch.tensor(WB,dtype=torch.float32,device=dev)
def proj(J):
    z=J[...,2].clamp(min=1e-4); u=Kt[:,0,0:1]*J[...,0]/z+Kt[:,0,2:3]; v=Kt[:,1,1:2]*J[...,1]/z+Kt[:,1,2:3]; return torch.stack([u,v],-1)
opt=torch.optim.Adam([wr_R,wr_L],lr=0.02)
for it in range(ITERS):
    bp=bp_fixed.clone(); bp[:,20]=wr_R; bp[:,19]=wr_L
    out=m(betas=betas,body_pose=bp.reshape(F,63),global_orient=go,transl=tr,left_hand_pose=lhp,right_hand_pose=rhp); J=out.joints
    l2d=0.0
    for SID,WID in [(RID_S,RID_W),(LID_S,LID_W)]:
        pj=proj(J[:,SID]); tgt=WBt[:,WID,:2]; conf=WBt[:,WID,2:3].clamp(0,1); l2d=l2d+((conf*(pj-tgt)**2).mean())
    sm=((wr_R[1:]-wr_R[:-1])**2).mean()+((wr_L[1:]-wr_L[:-1])**2).mean()
    loss=W2D*l2d/1000.0+WSM*sm
    opt.zero_grad(); loss.backward(); opt.step()
    if it%50==0: print('  armik it%3d 2d_rmse=%.1fpx wsm=%.4f'%(it,(l2d.item()/(6*2*2))**0.5,sm.item()))
out_dict=copy.deepcopy(src)
bpn=bp0.clone(); bpn[:,20]=wr_R.detach().cpu(); bpn[:,19]=wr_L.detach().cpu()
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp in out_dict:
        out_dict[sp]['body_pose']=bpn.reshape(F,21,3) if out_dict[sp]['body_pose'].dim()==3 else bpn.reshape(F,63)
torch.save(out_dict,out_pt)
print(f'[armik_v4] saved {out_pt} ({time.perf_counter()-t0:.1f}s)')
