"""
batch_rom_clamp: 손목이 인체 ROM(가동범위)을 넘는 프레임만 한계 안으로 클램프.
 편위(dev)±45, 굴곡/신전(flex)±90, 비틀림(axial) 우[-90,160]/좌[-160,90](부호반대).
 위반 프레임만 수정 후 경계 약하게 평활.
입력: <clip>/<base.pt>   출력: <clip>/<base>_rom.pt
usage: python batch_rom_clamp.py <clip_dir> <base.pt>
"""
import sys, numpy as np, copy, torch
from scipy.spatial.transform import Rotation as Rt
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0,'/workspace/capstone_deploy')
from physical_limits import JOINT_LIMITS
clip=sys.argv[1].rstrip('/'); base=sys.argv[2]; out_pt=f'{clip}/{base.replace(".pt","")}_rom.pt'
lim=JOINT_LIMITS['wrist']
DEV=(lim['deviation']['hard_neg'],lim['deviation']['hard_pos'])      # -45,45
FLEX=(lim['flexion']['hard_neg'],lim['flexion']['hard_pos'])         # -90,90
AXR=(lim['axial']['hard_neg'],lim['axial']['hard_pos'])              # -90,160 (right)
AXL=(-lim['axial']['hard_pos'],-lim['axial']['hard_neg'])            # 좌: 부호반대
src=torch.load(f'{clip}/{base}',map_location='cpu'); inc=src['smpl_params_incam']
bp=inc['body_pose'].reshape(-1,21,3).numpy().copy(); F=len(bp)
for nm,wr,ax in [('R',20,AXR),('L',19,AXL)]:
    nclamp=0; new=bp[:,wr].copy()
    for f in range(F):
        e=Rt.from_rotvec(bp[f,wr]).as_euler('XYZ',degrees=True)  # axial,dev,flex
        c=e.copy()
        c[0]=np.clip(e[0],ax[0],ax[1]); c[1]=np.clip(e[1],DEV[0],DEV[1]); c[2]=np.clip(e[2],FLEX[0],FLEX[1])
        if not np.allclose(c,e,atol=0.5):
            nclamp+=1; new[f]=Rt.from_euler('XYZ',c,degrees=True).as_rotvec()
    # 클램프 경계 약한 quat 평활(불연속 방지)
    q=Rt.from_rotvec(new).as_quat()
    for f in range(1,F):
        if np.dot(q[f],q[f-1])<0: q[f]=-q[f]
    qs=np.stack([gaussian_filter1d(q[:,i],0.8,mode='nearest') for i in range(4)],1); qs/=np.linalg.norm(qs,axis=1,keepdims=True)
    bp[:,wr]=Rt.from_quat(qs).as_rotvec()
    print(f'{nm}: ROM 클램프 {nclamp}f')
out=copy.deepcopy(src)
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp in out: out[sp]['body_pose']=torch.from_numpy(bp).float().reshape(out[sp]['body_pose'].shape)
torch.save(out,out_pt)
# 검증
from physical_limits import decompose_wrist_local
for nm,wr in [('R',20),('L',19)]:
    rb=sum(1 for f in range(F) if not(DEV[0]<=decompose_wrist_local(bp[f,wr])['deviation']<=DEV[1] and FLEX[0]<=decompose_wrist_local(bp[f,wr])['flexion']<=FLEX[1]))
    print(f'  [{nm}] 클램프 후 dev/flex 위반: {rb}f')
print(f'[rom_clamp] saved {out_pt}')
