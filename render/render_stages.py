import os,sys
from pathlib import Path
import torch,cv2,numpy as np
from tqdm import tqdm
PROJ=Path('/workspace/capstone_deploy');sys.path.insert(0,str(PROJ));os.chdir(str(PROJ))
from hmr4d.utils.video_io_utils import get_video_fps,get_video_lwh,get_video_reader,get_writer
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.net_utils import to_cuda
from hmr4d.utils.vis.renderer import Renderer
cd=Path(sys.argv[1]); out=sys.argv[2]
def filt(p):
    a={"betas","body_pose","global_orient","transl","left_hand_pose","right_hand_pose","jaw_pose","leye_pose","reye_pose","expression"}
    return {k:v for k,v in p.items() if k in a}
m=make_smplx("supermotion_fullhands").cuda();faces=m.faces
vin=cd/'0_input_video.mp4';fps=get_video_fps(str(vin));_,W,H=get_video_lwh(str(vin))
stages=[('1.fit2d','fit2d.pt'),('2.armik(pos fix)','armik_v4.pt'),('3.+collision','armik_v4_col.pt')]
Vs=[];K=None
for lab,pt in stages:
    pr=torch.load(cd/pt,map_location='cpu');inc=pr['smpl_params_incam'];F=inc['body_pose'].shape[0]
    with torch.no_grad(): V=m(**to_cuda(filt(inc))).vertices
    Vs.append((lab,V)); K=pr['K_fullimg'];K=K[0] if K.dim()==3 else K
r=Renderer(W,H,device="cuda",faces=faces,K=K,bin_size=0)
rd=get_video_reader(str(vin));w=get_writer(str(cd/out),fps=fps,crf=23);SC=0.34
def lab(im,t):
    cv2.putText(im,t,(12,36),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),5);cv2.putText(im,t,(12,36),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2);return im
for i,img in enumerate(tqdm(rd,total=F)):
    if i>=F:break
    cells=[lab(img.copy(),f'f{i} RAW')]
    for l,V in Vs: cells.append(lab(r.render_mesh(V[i].cuda(),img,(1.,.9,.6)),l))
    row=np.concatenate(cells,axis=1);row=cv2.resize(row,(int(row.shape[1]*SC),int(row.shape[0]*SC)))
    w.write_frame(row)
w.close();print(f'[stages] saved {cd/out}')
