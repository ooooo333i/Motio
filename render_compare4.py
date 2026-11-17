"""
4분할 비교 렌더: [raw | M1 | M2 | M3] 가로 스택.
usage: python render_compare4.py <clip_dir> <start> <end> <ptM1> <ptM2> <ptM3> <out_mp4> [L1 L2 L3]
"""
import os, sys
from pathlib import Path
import torch, cv2, numpy as np
from tqdm import tqdm
PROJ = Path('/workspace/capstone_deploy'); sys.path.insert(0, str(PROJ)); os.chdir(str(PROJ))
clip_dir = Path(sys.argv[1]); start = int(sys.argv[2]); end = int(sys.argv[3])
pts = [sys.argv[4], sys.argv[5], sys.argv[6]]; out_mp4 = sys.argv[7]
labs = [sys.argv[8] if len(sys.argv)>8 else 'M1', sys.argv[9] if len(sys.argv)>9 else 'M2', sys.argv[10] if len(sys.argv)>10 else 'M3']
from hmr4d.utils.video_io_utils import get_video_fps, get_video_lwh, get_video_reader, get_writer
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.net_utils import to_cuda
from hmr4d.utils.vis.renderer import Renderer
def filt(p):
    a={"betas","body_pose","global_orient","transl","left_hand_pose","right_hand_pose","jaw_pose","leye_pose","reye_pose","expression"}
    return {k:v for k,v in p.items() if k in a}
vin = clip_dir/"0_input_video.mp4"
m = make_smplx("supermotion_fullhands").cuda(); faces=m.faces
fps=get_video_fps(str(vin)); _,W,H=get_video_lwh(str(vin))
def verts(pt):
    pred=torch.load(clip_dir/pt,map_location="cpu"); inc=pred["smpl_params_incam"]; F=inc["body_pose"].shape[0]
    p=filt(inc); p={k:(v[start:end] if torch.is_tensor(v) and v.shape[0]==F else v) for k,v in p.items()}
    with torch.no_grad(): V=m(**to_cuda(p)).vertices
    K=pred["K_fullimg"]; K=K[0] if K.dim()==3 else K
    return V,K
Vs=[]; K=None
for pt in pts:
    V,K=verts(pt); Vs.append(V)
r=Renderer(W,H,device="cuda",faces=faces,K=K,bin_size=0)
rd=get_video_reader(str(vin)); SCALE=0.5
w=get_writer(str(clip_dir/out_mp4),fps=fps,crf=15)
cols=[(0.75,0.75,0.78),(1.0,0.8,0.5),(0.5,1.0,0.7)]  # M1 회색 M2 주황 M3 초록
def lab(img,t,c=(0,255,255)):
    cv2.putText(img,t,(15,40),cv2.FONT_HERSHEY_SIMPLEX,0.95,(0,0,0),5)
    cv2.putText(img,t,(15,40),cv2.FONT_HERSHEY_SIMPLEX,0.95,c,2); return img
for i,img in enumerate(tqdm(rd,total=end)):
    if i>=end: break
    if i<start: continue
    j=i-start
    raw=lab(img.copy(),f"f{i} RAW")
    panels=[raw]
    for k in range(3):
        o=r.render_mesh(Vs[k][j].cuda(),img,cols[k]); panels.append(lab(o,labs[k]))
    row=np.concatenate(panels,axis=1)
    row=cv2.resize(row,(int(row.shape[1]*SCALE),int(row.shape[0]*SCALE)))
    w.write_frame(row)
w.close(); print(f"[cmp4] saved {clip_dir/out_mp4}  frames {start}-{end}")
