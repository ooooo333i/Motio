"""4분할 + 선수 줌인 (손목 비틀림 보이게). usage: <clip> <s> <e> <m1> <m2> <m3> <out> [L1 L2 L3]"""
import os, sys
from pathlib import Path
import torch, cv2, numpy as np
from tqdm import tqdm
PROJ=Path('/workspace/capstone_deploy'); sys.path.insert(0,str(PROJ)); os.chdir(str(PROJ))
clip_dir=Path(sys.argv[1]); start=int(sys.argv[2]); end=int(sys.argv[3])
pts=[sys.argv[4],sys.argv[5],sys.argv[6]]; out_mp4=sys.argv[7]
labs=[sys.argv[8] if len(sys.argv)>8 else 'M1',sys.argv[9] if len(sys.argv)>9 else 'M2',sys.argv[10] if len(sys.argv)>10 else 'M3']
from hmr4d.utils.video_io_utils import get_video_fps, get_video_lwh, get_video_reader, get_writer
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.net_utils import to_cuda
from hmr4d.utils.vis.renderer import Renderer
def filt(p):
    a={"betas","body_pose","global_orient","transl","left_hand_pose","right_hand_pose","jaw_pose","leye_pose","reye_pose","expression"}
    return {k:v for k,v in p.items() if k in a}
vin=clip_dir/"0_input_video.mp4"; m=make_smplx("supermotion_fullhands").cuda(); faces=m.faces
fps=get_video_fps(str(vin)); _,W,H=get_video_lwh(str(vin))
def verts(pt):
    pred=torch.load(clip_dir/pt,map_location="cpu"); inc=pred["smpl_params_incam"]; F=inc["body_pose"].shape[0]
    p=filt(inc); p={k:(v[start:end] if torch.is_tensor(v) and v.shape[0]==F else v) for k,v in p.items()}
    with torch.no_grad(): out=m(**to_cuda(p)); V=out.vertices; J=out.joints
    K=pred["K_fullimg"]; K=K[0] if K.dim()==3 else K
    return V,J,K
Vs=[];K=None;Jref=None
for i,pt in enumerate(pts):
    V,J,K=verts(pt); Vs.append(V)
    if i==2: Jref=J  # Ours 관절로 줌 박스
r=Renderer(W,H,device="cuda",faces=faces,K=K,bin_size=0)
# 줌 박스: Ours 상체+팔 관절 2D bbox (전 프레임 union) + 마진. 정사각 비슷하게.
Kc=K.cpu().numpy()
JB=Jref[:, :25].cpu().numpy()  # 몸+손 일부
z=np.clip(JB[:,:,2],1e-4,None); u=Kc[0,0]*JB[:,:,0]/z+Kc[0,2]; v=Kc[1,1]*JB[:,:,1]/z+Kc[1,2]
x0,x1=np.percentile(u,1),np.percentile(u,99); y0,y1=np.percentile(v,1),np.percentile(v,99)
cx,cy=(x0+x1)/2,(y0+y1)/2; half=max(x1-x0,y1-y0)*0.85  # 마진
X0=int(max(0,cx-half)); X1=int(min(W,cx+half)); Y0=int(max(0,cy-half*1.1)); Y1=int(min(H,cy+half*1.1))
print(f'줌 박스: x[{X0}:{X1}] y[{Y0}:{Y1}] (원본 {W}x{H})')
rd=get_video_reader(str(vin)); w=get_writer(str(clip_dir/out_mp4),fps=fps,crf=18)
cols=[(0.75,0.75,0.78),(1.0,0.8,0.5),(0.5,1.0,0.7)]
def lab(img,t,c=(0,255,255)):
    cv2.putText(img,t,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),5); cv2.putText(img,t,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,c,2); return img
for i,img in enumerate(tqdm(rd,total=end)):
    if i>=end: break
    if i<start: continue
    j=i-start
    raw=lab(img[Y0:Y1,X0:X1].copy(),f"f{i} RAW")
    panels=[raw]
    for k in range(3):
        o=r.render_mesh(Vs[k][j].cuda(),img,cols[k])[Y0:Y1,X0:X1]; panels.append(lab(o,labs[k]))
    row=np.concatenate(panels,axis=1)
    w.write_frame(row)
w.close(); print(f"[zoom] saved {clip_dir/out_mp4}  frames {start}-{end}")
