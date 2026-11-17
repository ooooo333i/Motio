"""
batch_wrist_kalman (Stage F, 마지막 평활): 손목 회전에만 Kalman 적용.
 진단: jitter는 Stage B(HaMeR 손목 swing)에서 들어오고 이후 단계가 청소 안 해 출력까지 샘.
       손가락 포즈는 이미 매끈(grip σ1.5) → 떨리는 손목에 붙어 흔들리는 것.
 → 손목 [19,20] 회전을 Kalman(게이트)으로 평활. 게이트가 진짜 빠른 swing은 보존, 고주파 jitter만 제거.
 ⚠️ 손목(body_pose[19,20])만. 팔/손가락은 절대 불변(assert).
입력: <clip>/<base.pt>   출력: <clip>/<base>_wk.pt
usage: python batch_wrist_kalman.py <clip_dir> <base.pt> [gate_deg=45]
"""
import sys, numpy as np, copy
sys.path.insert(0,'/workspace/capstone_deploy')
import torch
from cap_kalman_wrist import kalman_smooth_rotation
clip=sys.argv[1].rstrip('/'); base=sys.argv[2]
GATE=float(sys.argv[3]) if len(sys.argv)>3 else 45.0
QN,RN,GF=0.001,0.2,30.0     # 손가락 칼만과 동일 계열(jitter 제거, 빠른 동작 허용)
out_pt=f'{clip}/{base.replace(".pt","")}_wk.pt'
src=torch.load(f'{clip}/{base}',map_location='cpu')
out=copy.deepcopy(src)
F=src['smpl_params_incam']['body_pose'].reshape(-1,21,3).shape[0]
arm_before=src['smpl_params_incam']['body_pose'].reshape(F,21,3)[:, [12,13,15,16,17,18]].clone()
fing_before=(src['smpl_params_incam']['left_hand_pose'].clone(), src['smpl_params_incam']['right_hand_pose'].clone())
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp not in out: continue
    bp=out[sp]['body_pose'].reshape(F,21,3).numpy().copy()
    for j in (19,20):   # L, R 손목
        bp[:,j]=kalman_smooth_rotation(bp[:,j],QN,RN,GATE,GF)
    out[sp]['body_pose']=torch.from_numpy(bp).float().reshape(out[sp]['body_pose'].shape)
torch.save(out,out_pt)
# 불변 검증
a=out['smpl_params_incam']['body_pose'].reshape(F,21,3)[:, [12,13,15,16,17,18]]
assert torch.equal(a, arm_before), "팔 변함! (금지)"
assert torch.equal(out['smpl_params_incam']['left_hand_pose'], fing_before[0]) and torch.equal(out['smpl_params_incam']['right_hand_pose'], fing_before[1]), "손가락 변함! (금지)"
print(f'[검증] 팔·손가락 완전 불변 ✓ (손목만 칼만)')
print(f'[wrist_kalman] saved {out_pt} (gate={GATE}°)')
