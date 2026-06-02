"""
HaMeR(2D정제=faithful) 따라가다가, '갑자기 트위스트되는 구간'만 GVHMR axial로.
- base = cmp_dynhamr_faithful.pt (실제 손 따라감, swing/flex/dev 보존)
- 검출: sudden = |Δaxial/frame|>VEL  또는  GVHMR이탈 |axial-gv|>DEV  -> dilate -> gaussian ramp
- 교체: axial만 GVHMR로 부드럽게 blend. swing(flex/dev)은 항상 HaMeR(faithful) 유지.
"""
import numpy as np, copy, torch
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d, binary_dilation, median_filter
base='/workspace/capstone_deploy/outputs/cap_pipeline/input/'
B=torch.load(base+'cmp_dynhamr_faithful.pt',map_location='cpu')   # HaMeR(2D정제) base
gv=torch.load(base+'smplx_arm_prefit.pt',map_location='cpu')['smpl_params_incam']['body_pose'].reshape(-1,21,3).numpy()
F=len(gv)
RESID=30.0  # deg: 자기 local-median에서 튀면 'sudden 트위스트 artifact'
MEDW=7      # median window
DIL=2       # 검출 구간 양옆 확장

def quat(aa): return Rotation.from_rotvec(aa).as_quat()
def aa(q): return Rotation.from_quat(q).as_rotvec()
def qmul(a,b): return (Rotation.from_quat(a)*Rotation.from_quat(b)).as_quat()
def twist_x(q):
    x,y,z,w=q; n=np.hypot(x,w)
    return np.array([0,0,0,1.0]) if n<1e-9 else np.array([x/n,0,0,w/n])
def swing_twist(q):
    t=twist_x(q); s=qmul(q,t*np.array([-1,-1,-1,1])); return s,t
def sang(q): d=np.degrees(2*np.arctan2(q[0],q[3])); return (d+180)%360-180
def twist_q(deg): r=np.radians(deg)/2; return np.array([np.sin(r),0,0,np.cos(r)])
def wrap(d): return ((d+180)%360)-180

out=copy.deepcopy(B)
rep={}
for side,wr in [('R',20),('L',19)]:
    bp=B['smpl_params_incam']['body_pose'].reshape(F,21,3).numpy()
    swings=[]; a_base=np.zeros(F); a_gv=np.zeros(F)
    for f in range(F):
        s,t=swing_twist(quat(bp[f,wr])); swings.append(s); a_base[f]=sang(t)
        _,tg=swing_twist(quat(gv[f,wr])); a_gv[f]=sang(tg)
    # sudden 검출: HaMeR axial이 자기 local-median에서 튀는 곳 (꾸준한 차이는 정상)
    med=median_filter(a_base,size=MEDW,mode='nearest')
    resid=np.abs(wrap(a_base-med))
    bad=resid>RESID
    bad=binary_dilation(bad,iterations=DIL)
    w=gaussian_filter1d(bad.astype(float),2.0)   # 0..1 부드러운 ramp
    w=np.clip(w/max(w.max(),1e-6),0,1)
    rep[side]=bad.mean()
    # target axial: base->gv 블렌딩
    a_tgt=a_base+w*wrap(a_gv-a_base)
    a_tgt=gaussian_filter1d(a_tgt,1.2,mode='nearest')
    new=np.zeros((F,3))
    for f in range(F):
        new[f]=aa(qmul(swings[f],twist_q(a_tgt[f])))
    for sp in ('smpl_params_incam','smpl_params_global'):
        b=out[sp]['body_pose'].reshape(F,21,3).numpy().copy()
        b[:,wr]=new; out[sp]['body_pose']=torch.from_numpy(b).reshape(F,21,3)
    print('%s: sudden구간=%.1f%% (GVHMR로 교체)'%(side,100*rep[side]))

torch.save(out,base+'cmp_adaptive_v2.pt')
print('saved cmp_adaptive_v2.pt')
# 진단
bpn=out['smpl_params_incam']['body_pose'].reshape(F,21,3).numpy()
def vmax(wr):
    Rr=Rotation.from_rotvec(bpn[:,wr]).as_matrix()
    v=[np.degrees(np.linalg.norm(Rotation.from_matrix(Rr[f-1].T@Rr[f]).as_rotvec())) for f in range(1,F)]
    return max(v),sum(1 for x in v if x>33)
print('velocity after: R max=%.0f >33:%d | L max=%.0f >33:%d'%(*vmax(20),*vmax(19)))
