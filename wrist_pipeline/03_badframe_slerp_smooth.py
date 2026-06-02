"""
최종본 A: cmp_adaptive_v2 -> cmp_adaptive_v3
 1) bad_frames 손목 slerp 보간 (양끝 good 프레임)
 2) bad 구간 경계 스무딩 (bad 근처만 gaussian quat smooth, clean 구간은 보존)
"""
import numpy as np, copy, torch, json
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import gaussian_filter1d, binary_dilation
base='/workspace/capstone_deploy/outputs/cap_pipeline/input/'
hmp='/workspace/hmp_test/data/rgb_data/in_the_wild/'
badR=set(json.load(open(hmp+'tennis_full/hamer_out/bad_frames.json')))
badL=set(json.load(open(hmp+'tennis_full_left/hamer_out/bad_frames.json')))
out=torch.load(base+'cmp_adaptive_v2.pt',map_location='cpu')
F=out['smpl_params_incam']['body_pose'].reshape(-1,21,3).shape[0]

def slerp_fill(aa, bad):
    good=[f for f in range(F) if f not in bad]; aa=aa.copy()
    for f in sorted(bad):
        lo=max([g for g in good if g<f],default=None); hi=min([g for g in good if g>f],default=None)
        if lo is None and hi is None: continue
        if lo is None: aa[f]=aa[hi]; continue
        if hi is None: aa[f]=aa[lo]; continue
        key=Rotation.from_rotvec([aa[lo],aa[hi]]); aa[f]=Slerp([0,1],key)([(f-lo)/(hi-lo)]).as_rotvec()[0]
    return aa

def quat_smooth(aa, sigma):
    q=Rotation.from_rotvec(aa).as_quat()
    for f in range(1,len(q)):
        if np.dot(q[f],q[f-1])<0: q[f]=-q[f]
    qs=np.stack([gaussian_filter1d(q[:,i],sigma,mode='nearest') for i in range(4)],1)
    qs/=np.linalg.norm(qs,axis=1,keepdims=True)
    return Rotation.from_quat(qs).as_rotvec()

def process(aa, bad):
    filled=slerp_fill(aa,bad)
    sm=quat_smooth(filled,1.5)
    # bad 근처 가중치 (경계만 스무딩, clean 보존)
    m=np.zeros(F,bool); m[list(bad)]=True; m=binary_dilation(m,iterations=2)
    w=gaussian_filter1d(m.astype(float),2.0); w=np.clip(w/max(w.max(),1e-6),0,1)
    qf=Rotation.from_rotvec(filled).as_quat(); qs=Rotation.from_rotvec(sm).as_quat()
    res=np.zeros((F,3))
    for f in range(F):
        if np.dot(qf[f],qs[f])<0: qs[f]=-qs[f]
        key=Rotation.from_quat([qf[f],qs[f]]); res[f]=Slerp([0,1],key)([w[f]]).as_rotvec()[0]
    return res

bp=out['smpl_params_incam']['body_pose'].reshape(F,21,3).numpy().copy()
bp[:,20]=process(bp[:,20],badR); bp[:,19]=process(bp[:,19],badL)
for sp in ('smpl_params_incam','smpl_params_global'):
    b=out[sp]['body_pose'].reshape(F,21,3).numpy().copy(); b[:,20]=bp[:,20]; b[:,19]=bp[:,19]
    out[sp]['body_pose']=torch.from_numpy(b).reshape(F,21,3)
torch.save(out,base+'cmp_adaptive_v3.pt')

def vel(a,f):
    R0=Rotation.from_rotvec(a[f-1]).as_matrix();R1=Rotation.from_rotvec(a[f]).as_matrix()
    return np.degrees(np.linalg.norm(Rotation.from_matrix(R0.T@R1).as_rotvec()))
vR=[vel(bp[:,20],f) for f in range(1,F)]; vL=[vel(bp[:,19],f) for f in range(1,F)]
print('saved cmp_adaptive_v3.pt')
print('R wrist vel: max=%.0f  >33:%d frames'%(max(vR),sum(1 for x in vR if x>33)))
print('L wrist vel: max=%.0f  >33:%d frames'%(max(vL),sum(1 for x in vL if x>33)))
print('f408-415 R vel:',[round(vel(bp[:,20],f)) for f in range(408,416)])
