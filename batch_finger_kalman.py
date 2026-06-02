"""
손가락(hand_pose)에만 Kalman 적용. 손목 axial 등 body_pose는 절대 미적용.
각 손가락 관절(15개/손)을 회전 Kalman으로 평활.
usage: python batch_finger_kalman.py <clip_dir>
  in : <clip_dir>/fit2d.pt
  out: <clip_dir>/fit2d_fingerkalman.pt
"""
import sys, numpy as np, copy, time
sys.path.insert(0,'/workspace/capstone_deploy')
import torch
from cap_kalman_wrist import kalman_smooth_rotation

clip_dir=sys.argv[1].rstrip('/')
src=torch.load(f'{clip_dir}/fit2d.pt',map_location='cpu')
out_pt=f'{clip_dir}/fit2d_fingerkalman.pt'
t0=time.perf_counter()
# 손가락 Kalman 파라미터 (jitter 제거, 빠른 손가락 동작 허용 위해 gate 넉넉)
QN,RN,GATE,GF=0.001,0.2,60.0,30.0
out=copy.deepcopy(src)
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp not in out: continue
    for key in ('left_hand_pose','right_hand_pose'):
        hp=out[sp][key]
        F=hp.shape[0]; aa=hp.reshape(F,-1,3).numpy()   # (F,15,3)
        for j in range(aa.shape[1]):
            aa[:,j]=kalman_smooth_rotation(aa[:,j],QN,RN,GATE,GF)
        out[sp][key]=torch.from_numpy(aa.reshape(F,-1)).float()
torch.save(out,out_pt)
# 검증: 손가락 jitter 감소량
def fjit(d):
    hp=d['smpl_params_incam']['right_hand_pose']
    v=(hp[1:]-hp[:-1]).abs().max(1).values
    return float(v.max()),float(v.mean())
b=fjit(src); a=fjit(out)
print(f'[finger_kalman] R-hand Δ/frame  before max={b[0]:.3f} mean={b[1]:.3f}  ->  after max={a[0]:.3f} mean={a[1]:.3f}')
print(f'[finger_kalman] saved {out_pt}  FINGERKALMAN_SEC={time.perf_counter()-t0:.2f}')
