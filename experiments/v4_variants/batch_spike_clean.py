"""
batch_spike_clean: 손목 속도 스파이크(>33°/f, 비물리적=bad detection)만 SLERP 보간 + 경계 평활.
 generalized 03_badframe_slerp_smooth: 외부 JSON 대신 속도>임계로 bad 자동 검출.
 good 프레임(faithful swing)은 보존, 스파이크만 양끝 good에서 slerp + bad 근처만 quat gaussian blend.
입력: <clip>/armik_faithful.pt  출력: <clip>/armik_final.pt
usage: python batch_spike_clean.py <clip_dir> [vel_thresh=33]
"""
import sys, numpy as np, copy, torch
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import gaussian_filter1d, binary_dilation
clip=sys.argv[1].rstrip('/'); VTH=float(sys.argv[2]) if len(sys.argv)>2 else 33.0
out=torch.load(f'{clip}/armik_faithful.pt',map_location='cpu')
F=out['smpl_params_incam']['body_pose'].reshape(-1,21,3).shape[0]
def vel(aa):
    R=Rotation.from_rotvec(aa).as_matrix()
    return np.array([0]+[np.degrees(np.linalg.norm(Rotation.from_matrix(R[f-1].T@R[f]).as_rotvec())) for f in range(1,len(aa))])
def slerp_fill(aa,bad):
    good=[f for f in range(F) if not bad[f]]; aa=aa.copy()
    for f in np.where(bad)[0]:
        lo=max([g for g in good if g<f],default=None); hi=min([g for g in good if g>f],default=None)
        if lo is None and hi is None: continue
        if lo is None: aa[f]=aa[hi]; continue
        if hi is None: aa[f]=aa[lo]; continue
        key=Rotation.from_rotvec([aa[lo],aa[hi]]); aa[f]=Slerp([0,1],key)([(f-lo)/(hi-lo)]).as_rotvec()[0]
    return aa
def quat_smooth(aa,sigma):
    q=Rotation.from_rotvec(aa).as_quat()
    for f in range(1,len(q)):
        if np.dot(q[f],q[f-1])<0: q[f]=-q[f]
    qs=np.stack([gaussian_filter1d(q[:,i],sigma,mode='nearest') for i in range(4)],1); qs/=np.linalg.norm(qs,axis=1,keepdims=True)
    return Rotation.from_quat(qs).as_rotvec()
def process(aa):
    v=vel(aa); bad=v>VTH; bad=binary_dilation(bad,iterations=1)
    nbad=int(bad.sum())
    filled=slerp_fill(aa,bad); sm=quat_smooth(filled,1.5)
    w=gaussian_filter1d(bad.astype(float),2.0); w=np.clip(w/max(w.max(),1e-6),0,1)
    qf=Rotation.from_rotvec(filled).as_quat(); qs=Rotation.from_rotvec(sm).as_quat(); res=np.zeros((F,3))
    for f in range(F):
        if np.dot(qf[f],qs[f])<0: qs[f]=-qs[f]
        key=Rotation.from_quat([qf[f],qs[f]]); res[f]=Slerp([0,1],key)([w[f]]).as_rotvec()[0]
    return res,nbad
bp=out['smpl_params_incam']['body_pose'].reshape(F,21,3).numpy().copy()
r,nR=process(bp[:,20]); l,nL=process(bp[:,19]); bp[:,20]=r; bp[:,19]=l
for sp in ('smpl_params_incam','smpl_params_global'):
    b=out[sp]['body_pose'].reshape(F,21,3).numpy().copy(); b[:,20]=bp[:,20]; b[:,19]=bp[:,19]; out[sp]['body_pose']=torch.from_numpy(b).reshape(F,21,3)
torch.save(out,f'{clip}/armik_final.pt')
print(f'spike 프레임 R={nR} L={nL} (>{VTH:.0f}deg/f) slerp 보간')
for nm,wr in [('R',20),('L',19)]:
    v=vel(bp[:,wr]); print(f'  {nm} 손목속도 max={v.max():.0f} >33:{int((v>33).sum())}f')
print(f'saved {clip}/armik_final.pt')
