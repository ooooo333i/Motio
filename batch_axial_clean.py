"""
batch_axial_clean: armik_faithful(swing 최고, axial 스파이크) → axial만 정리.
 02_adaptive_axial 로직: swing-twist 분해, swing=faithful 유지,
 axial(twist)은 'sudden' 구간만 GVHMR(arm_prefit)로 블렌딩 + gaussian 평활.
 (swing=관측, axial=GVHMR 분담. axial은 invisible이라 overlay 손해 0.)
입력: <clip>/{armik_faithful.pt, smplx_arm_prefit.pt}
출력: <clip>/armik_clean.pt
usage: python batch_axial_clean.py <clip_dir>
"""
import sys, numpy as np, copy, torch
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d, binary_dilation, median_filter
clip=sys.argv[1].rstrip('/')
B=torch.load(f'{clip}/armik_faithful.pt',map_location='cpu')
gv=torch.load(f'{clip}/smplx_arm_prefit.pt',map_location='cpu')['smpl_params_incam']['body_pose'].reshape(-1,21,3).numpy()
F=len(gv); RESID=30.0; MEDW=7; DIL=2
def quat(aa): return Rotation.from_rotvec(aa).as_quat()
def aa(q): return Rotation.from_quat(q).as_rotvec()
def qmul(a,b): return (Rotation.from_quat(a)*Rotation.from_quat(b)).as_quat()
def twist_x(q):
    x,y,z,w=q; n=np.hypot(x,w); return np.array([0,0,0,1.0]) if n<1e-9 else np.array([x/n,0,0,w/n])
def swing_twist(q): t=twist_x(q); s=qmul(q,t*np.array([-1,-1,-1,1])); return s,t
def sang(q): d=np.degrees(2*np.arctan2(q[0],q[3])); return (d+180)%360-180
def twist_q(deg): r=np.radians(deg)/2; return np.array([np.sin(r),0,0,np.cos(r)])
def wrap(d): return ((d+180)%360)-180
out=copy.deepcopy(B)
for side,wr in [('R',20),('L',19)]:
    bp=B['smpl_params_incam']['body_pose'].reshape(F,21,3).numpy()
    swings=[]; a_base=np.zeros(F); a_gv=np.zeros(F)
    for f in range(F):
        s,t=swing_twist(quat(bp[f,wr])); swings.append(s); a_base[f]=sang(t)
        _,tg=swing_twist(quat(gv[f,wr])); a_gv[f]=sang(tg)
    # §2.1: axial은 GVHMR만 신뢰 → 전 구간 GVHMR axial 사용(swing은 faithful 유지). invisible이라 overlay 손해 0.
    med=median_filter(a_base,size=MEDW,mode='nearest'); resid=np.abs(wrap(a_base-med))
    bad=binary_dilation(resid>RESID,iterations=DIL)
    # GVHMR axial을 unwrap 평활(연속성) 후 사용
    a_gv_un=np.degrees(np.unwrap(np.radians(a_gv)))
    a_tgt=gaussian_filter1d(a_gv_un,1.5,mode='nearest')
    new=np.zeros((F,3))
    for f in range(F): new[f]=aa(qmul(swings[f],twist_q(a_tgt[f])))
    for sp in ('smpl_params_incam','smpl_params_global'):
        b=out[sp]['body_pose'].reshape(F,21,3).numpy().copy(); b[:,wr]=new; out[sp]['body_pose']=torch.from_numpy(b).reshape(F,21,3)
    print(f'{side}: sudden axial 구간={100*bad.mean():.1f}% GVHMR로 블렌딩')
torch.save(out,f'{clip}/armik_clean.pt')
bpn=out['smpl_params_incam']['body_pose'].reshape(F,21,3).numpy()
for nm,wr in [('R',20),('L',19)]:
    Rr=Rotation.from_rotvec(bpn[:,wr]).as_matrix()
    v=[np.degrees(np.linalg.norm(Rotation.from_matrix(Rr[f-1].T@Rr[f]).as_rotvec())) for f in range(1,F)]
    print(f'  {nm} 손목속도 max={max(v):.0f} >33:{sum(1 for x in v if x>33)}f')
print(f'saved {clip}/armik_clean.pt')
