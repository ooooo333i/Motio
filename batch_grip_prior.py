"""
batch_grip_prior (v3, IoU 겹침 게이트 + 앞/뒤 손 구분): 손 그립을 confidence + 손겹침으로 게이트.
 ⚠️ 절대 규칙: 손가락(left/right_hand_pose)만 수정. body_pose(손목/전완/팔) 절대 안 건드림.
 게이트(손별):
   bad = (WB손conf 낮음)  OR  (두 손 영역 겹침(IoU) AND 이 손이 뒤(cam-z 큼))
   - 앞 손(카메라 가까이, 잘 보임) → 겹쳐도 HaMeR 유지 (실제로 잘 봄).
   - 뒤 손(가려짐, HaMeR가 앞손 따라가 망가짐) → 양옆 비겹침 그립으로 보간 홀드.
 IoU = WB 손 키포인트(conf>0.3) bbox 겹침 — 손목·손 위치는 신뢰됨(오염 안 됨).
 cam-z = SMPL-X 손목 joint(20/21) z (작을수록 앞). 팔(body_pose) 기반이라 손가락 오염과 무관.
입력: <clip>/<base.pt>, preprocess/wholebody_kp2d.npy, fit2d.pt(betas/go/transl)
출력: <clip>/<base>_grip.pt
usage: python batch_grip_prior.py <clip_dir> <base.pt> [conf_th=0.7] [smooth=1.5] [iou_th=0.1]
"""
import sys, os, numpy as np, copy
import torch, smplx
from scipy.ndimage import gaussian_filter1d, binary_dilation
clip=sys.argv[1].rstrip('/'); base=sys.argv[2]
CONF_TH=float(sys.argv[3]) if len(sys.argv)>3 else 0.7
SMOOTH=float(sys.argv[4]) if len(sys.argv)>4 else 1.5
IOU_TH=float(sys.argv[5]) if len(sys.argv)>5 else 0.1
SPIKE=0.5
SMPLX='/workspace/hmp_test/data/body_models/smplx/SMPLX_NEUTRAL.npz'
out_pt=f'{clip}/{base.replace(".pt","")}_grip.pt'
src=torch.load(f'{clip}/{base}',map_location='cpu'); inc=src['smpl_params_incam']
F=inc['body_pose'].shape[0]
body_before=inc['body_pose'].clone()
WB=np.load(f'{clip}/preprocess/wholebody_kp2d.npy')
conf={'L':WB[:,91:112,2].mean(1),'R':WB[:,112:133,2].mean(1)}

# --- 손 영역 겹침(IoU) ---
def hbox(idx):
    pts=WB[:,idx,:2]; cf=WB[:,idx,2]; b=np.full((F,4),np.nan)
    for f in range(F):
        v=cf[f]>0.3; p=pts[f][v]
        if len(p)>=3: b[f]=[p[:,0].min(),p[:,1].min(),p[:,0].max(),p[:,1].max()]
    return b
Rbox=hbox(range(112,133)); Lbox=hbox(range(91,112)); iou=np.zeros(F)
for f in range(F):
    if np.isnan(Rbox[f,0]) or np.isnan(Lbox[f,0]): continue
    x1=max(Rbox[f,0],Lbox[f,0]);y1=max(Rbox[f,1],Lbox[f,1]);x2=min(Rbox[f,2],Lbox[f,2]);y2=min(Rbox[f,3],Lbox[f,3])
    inter=max(0,x2-x1)*max(0,y2-y1); ar=(Rbox[f,2]-Rbox[f,0])*(Rbox[f,3]-Rbox[f,1]); br=(Lbox[f,2]-Lbox[f,0])*(Lbox[f,3]-Lbox[f,1])
    iou[f]=inter/(ar+br-inter+1e-6)
ovl=binary_dilation(iou>IOU_TH,iterations=2)   # 겹침 구간(여유 dilate)

# --- 앞/뒤 손 (SMPL-X 손목 cam-z) ---
_f2=f'{clip}/fit2d.pt'  # betas/go/transl = 클립상수 → fit2d 있으면 쓰고, 없으면 base pt에서 (fit2d 의존 제거)
f2=torch.load(_f2,map_location='cpu')['smpl_params_incam'] if os.path.exists(_f2) else inc
def fix(t):
    t=(t if torch.is_tensor(t) else torch.tensor(t)).float()
    if t.dim()==1:t=t.unsqueeze(0)
    if t.shape[0]==1:t=t.expand(F,-1).contiguous()
    return t
betas=fix(f2['betas']);go=fix(f2['global_orient']);trn=fix(f2['transl'])
m=smplx.create(SMPLX,model_type='smplx',gender='neutral',use_pca=False,flat_hand_mean=False,num_betas=betas.shape[1],batch_size=F)
with torch.no_grad():
    J=m(betas=betas,body_pose=fix(inc['body_pose']).reshape(F,63),global_orient=go,transl=trn,
        left_hand_pose=fix(inc['left_hand_pose']).reshape(F,45),right_hand_pose=fix(inc['right_hand_pose']).reshape(F,45)).joints
zL=J[:,20,2].numpy(); zR=J[:,21,2].numpy()
is_back={'R':zR>zL,'L':zL>zR}   # 뒤 손 = cam-z 큼

def stabilize(hp, cf, back):   # hp:(F,45), cf:손conf, back:이 손이 뒤(F,)
    hp=hp.copy(); conf_sm=gaussian_filter1d(cf,3.0)
    bad_conf=binary_dilation(conf_sm<CONF_TH,iterations=2)
    bad_ovl=ovl & back                                    # 겹침 + 이 손이 뒤 → 홀드
    bad=bad_conf | bad_ovl
    good=[f for f in range(F) if not bad[f]]; filled=hp.copy()
    for f in np.where(bad)[0]:                             # 양옆 good(비겹침·고conf) 그립으로 보간 홀드
        lo=max([g for g in good if g<f],default=None); hi=min([g for g in good if g>f],default=None)
        if lo is None and hi is None: continue
        if lo is None: filled[f]=hp[hi]; continue
        if hi is None: filled[f]=hp[lo]; continue
        t=(f-lo)/(hi-lo); filled[f]=(1-t)*hp[lo]+t*hp[hi]
    d=np.array([0.0]+[np.linalg.norm(filled[f]-filled[f-1]) for f in range(1,F)])  # per-frame spike
    bad2=binary_dilation(d>SPIKE,iterations=1); good2=[f for f in range(F) if not bad2[f]]
    for f in np.where(bad2)[0]:
        lo=max([g for g in good2 if g<f],default=None); hi=min([g for g in good2 if g>f],default=None)
        if lo is None or hi is None: continue
        t=(f-lo)/(hi-lo); filled[f]=(1-t)*filled[lo]+t*filled[hi]
    sm=np.stack([gaussian_filter1d(filled[:,c],SMOOTH,mode='nearest') for c in range(45)],1)
    return sm, int(bad_conf.sum()), int(bad_ovl.sum())
lh=inc['left_hand_pose'].reshape(F,45).numpy(); rh=inc['right_hand_pose'].reshape(F,45).numpy()
lh2,nlc,nlo=stabilize(lh,conf['L'],is_back['L']); rh2,nrc,nro=stabilize(rh,conf['R'],is_back['R'])
print(f'겹침 프레임 {int(ovl.sum())}f (IoU>{IOU_TH})')
print(f'L손 홀드: 저conf {nlc}f + 겹침뒤손 {nlo}f / R손 홀드: 저conf {nrc}f + 겹침뒤손 {nro}f')
out=copy.deepcopy(src)
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp in out:
        out[sp]['left_hand_pose']=torch.from_numpy(lh2).float().reshape(out[sp]['left_hand_pose'].shape)
        out[sp]['right_hand_pose']=torch.from_numpy(rh2).float().reshape(out[sp]['right_hand_pose'].shape)
torch.save(out,out_pt)
assert torch.equal(out['smpl_params_incam']['body_pose'], body_before), "body_pose 변함! (금지)"
print('[검증] body_pose(손목/전완/팔) 완전 불변 ✓')
print(f'[grip_prior v3 IoU게이트] saved {out_pt}')
