"""
batch_hamer_wrist: 손목 = swing(HaMeR) + axial(전완/GVHMR=arm_prefit) + 속도클램프.
 - 팔=arm_prefit(IK 위치), 손가락=_post(HaMeR), 손목 swing=HaMeR(overlay 충실), axial=전완(물리적).
 - HaMeR 없는 프레임=arm_prefit 손목 fallback. 마지막에 속도>33°/f slerp + 평활.
입력: <clip>/{smplx_merged_hamer_post.pt, smplx_arm_prefit.pt, hamer_out_clean/}
출력: <clip>/hamer_wrist.pt
usage: python batch_hamer_wrist.py <clip_dir>
"""
import sys, numpy as np, copy, time
from pathlib import Path
sys.path.insert(0,'/workspace/capstone_deploy')
import torch
from scipy.spatial.transform import Rotation as Rt, Slerp
from scipy.ndimage import gaussian_filter1d, binary_dilation
from cap_wrist_refine import load_hamer_detections_full
clip=sys.argv[1].rstrip('/'); AXIAL=sys.argv[2] if len(sys.argv)>2 else 'gv'; SMOOTH=float(sys.argv[3]) if len(sys.argv)>3 else 1.8; t0=time.perf_counter()
src=torch.load(f'{clip}/smplx_merged_hamer_post.pt',map_location='cpu'); inc=src['smpl_params_incam']
arm=torch.load(f'{clip}/smplx_arm_prefit.pt',map_location='cpu')['smpl_params_incam']['body_pose'].reshape(-1,21,3).numpy()
bp=inc['body_pose'].reshape(-1,21,3).numpy().copy(); F=bp.shape[0]
go=inc['global_orient'].numpy();
if go.shape[0]==1: go=np.repeat(go,F,0)
for i in [12,13,15,16,17,18]: bp[:,i]=arm[:,i]   # IK 팔(손목 제외)
det=load_hamer_detections_full(Path(f'{clip}/hamer_out_clean'),F)
WB=np.load(f'{clip}/preprocess/wholebody_kp2d.npy')   # (F,133,3) 가려짐 검출용
HCONF={'right':WB[:,112:133,2].mean(1),'left':WB[:,91:112,2].mean(1)}
CONF_TH=0.55
RCH=[2,5,8,13,16,18]; LCH=[2,5,8,12,15,17]
def Gelbow(f,chain):
    Rm=Rt.from_rotvec(go[f]).as_matrix()
    for j in chain: Rm=Rm@Rt.from_rotvec(bp[f,j]).as_matrix()
    return Rm
def quat(aa): return Rt.from_rotvec(aa).as_quat()
def qmul(a,b): return (Rt.from_quat(a)*Rt.from_quat(b)).as_quat()
def twist_x(q):
    x,y,z,w=q; n=np.hypot(x,w); return np.array([0,0,0,1.]) if n<1e-9 else np.array([x/n,0,0,w/n])
def swing_of(q): t=twist_x(q); return qmul(q,t*np.array([-1,-1,-1,1]))
def sang(q): d=np.degrees(2*np.arctan2(q[0],q[3])); return (d+180)%360-180
def twist_q(deg): r=np.radians(deg)/2; return np.array([np.sin(r),0,0,np.cos(r)])
def vel(aa,f):
    R0=Rt.from_rotvec(aa[f-1]).as_matrix();R1=Rt.from_rotvec(aa[f]).as_matrix()
    return np.degrees(np.linalg.norm(Rt.from_matrix(R0.T@R1).as_rotvec()))
for side,wr,chain in [('right',20,RCH),('left',19,LCH)]:
    new=arm[:,wr].copy()   # fallback=arm_prefit 손목
    a_gv=np.array([sang(twist_x(quat(arm[f,wr]))) for f in range(F)])  # 전완/GVHMR axial
    for f in range(F):
        if f not in det[side]: continue
        Ge=Gelbow(f,chain)
        Rl_ham=Ge.T@det[side][f]['wrist_R']          # HaMeR 손목 local
        if AXIAL=='hamer':
            new[f]=Rt.from_matrix(Rl_ham).as_rotvec()    # 순수 HaMeR 손목(swing+axial)
        else:
            sw=swing_of(Rt.from_matrix(Rl_ham).as_quat())
            new[f]=Rt.from_quat(qmul(sw,twist_q(a_gv[f]))).as_rotvec()
    # bad = 속도 스파이크(비물리적) OR 가려짐/뭉개짐(WB conf 낮음) → good HaMeR에서 slerp + 경계평활
    v=np.array([0]+[vel(new,f) for f in range(1,F)])
    bad=(v>33)|(HCONF[side]<CONF_TH)
    bad=binary_dilation(bad,iterations=2)
    nbad=int(bad.sum())
    good=[f for f in range(F) if not bad[f]]
    filled=new.copy()
    for f in np.where(bad)[0]:
        lo=max([g for g in good if g<f],default=None); hi=min([g for g in good if g>f],default=None)
        if lo is None and hi is None: continue
        if lo is None: filled[f]=new[hi]; continue
        if hi is None: filled[f]=new[lo]; continue
        key=Rt.from_rotvec([new[lo],new[hi]]); filled[f]=Slerp([0,1],key)([(f-lo)/(hi-lo)]).as_rotvec()[0]
    # 경계만 가우시안 블렌딩(good HaMeR 보존)
    q=Rt.from_rotvec(filled).as_quat()
    for f in range(1,F):
        if np.dot(q[f],q[f-1])<0: q[f]=-q[f]
    qs=np.stack([gaussian_filter1d(q[:,i],1.5,mode='nearest') for i in range(4)],1); qs/=np.linalg.norm(qs,axis=1,keepdims=True)
    sm=Rt.from_quat(qs).as_rotvec()
    w=gaussian_filter1d(bad.astype(float),2.0); w=np.clip(w/max(w.max(),1e-6),0,1)
    qf=Rt.from_rotvec(filled).as_quat(); qS=Rt.from_quat(qs).as_quat()
    new=np.zeros((F,3))
    for f in range(F):
        if np.dot(qf[f],qS[f])<0: qS[f]=-qS[f]
        key=Rt.from_quat([qf[f],qS[f]]); new[f]=Slerp([0,1],key)([w[f]]).as_rotvec()[0]
    print(f'  {side}: bad(가려짐+스파이크)={nbad}f → good HaMeR slerp')
    bp[:,wr]=new
    print(f'{side}: HaMeR swing+전완 axial 적용 (HaMeR 검출 {len(det[side])}f)')
# 전역 가벼운 quat 평활 (HaMeR per-frame jitter 제거, overlay 유지)
if SMOOTH>0:
    for wr in [19,20]:
        q=Rt.from_rotvec(bp[:,wr]).as_quat()
        for f in range(1,F):
            if np.dot(q[f],q[f-1])<0: q[f]=-q[f]
        qs=np.stack([gaussian_filter1d(q[:,i],SMOOTH,mode='nearest') for i in range(4)],1); qs/=np.linalg.norm(qs,axis=1,keepdims=True)
        bp[:,wr]=Rt.from_quat(qs).as_rotvec()
    print(f'  전역 평활 sigma={SMOOTH}')
out=copy.deepcopy(src)
for sp in ('smpl_params_incam','smpl_params_global'):
    if sp in out: out[sp]['body_pose']=torch.from_numpy(bp).float().reshape(out[sp]['body_pose'].shape)
torch.save(out,f'{clip}/hamer_wrist_{AXIAL}.pt')
# 진단
from physical_limits import decompose_wrist_local,JOINT_LIMITS
lim=JOINT_LIMITS['wrist']
for nm,wr in [('R',20),('L',19)]:
    v=[vel(bp[:,wr],f) for f in range(1,F)]
    rb=sum(1 for f in range(F) if not(lim['deviation']['hard_neg']<=decompose_wrist_local(bp[f,wr])['deviation']<=lim['deviation']['hard_pos'] and lim['flexion']['hard_neg']<=decompose_wrist_local(bp[f,wr])['flexion']<=lim['flexion']['hard_pos']))
    print(f'[diag {nm}] 속도 max={max(v):.0f} >33:{sum(1 for x in v if x>33)}f  ROM위반:{rb}f')
print(f'[hamer_wrist] saved {clip}/hamer_wrist_{AXIAL}.pt ({time.perf_counter()-t0:.1f}s)')
