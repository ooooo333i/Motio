import os,sys
from pathlib import Path
import torch, cv2
from tqdm import tqdm
PROJ=Path('/workspace/capstone_deploy'); sys.path.insert(0,str(PROJ)); os.chdir(str(PROJ))
PT=sys.argv[1]; OUT=sys.argv[2]; A=int(sys.argv[3]); B=int(sys.argv[4])
def filt(p):
    a={"betas","body_pose","global_orient","transl","left_hand_pose","right_hand_pose","jaw_pose","leye_pose","reye_pose","expression"}
    return {k:v for k,v in p.items() if k in a}
from hmr4d.utils.video_io_utils import get_video_fps,get_video_lwh,get_video_reader,get_writer
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.net_utils import to_cuda
from hmr4d.utils.vis.renderer import Renderer
d=PROJ/"outputs/cap_pipeline/input"; vin=d/"0_input_video.mp4"
m=make_smplx("supermotion_fullhands").cuda(); faces=m.faces
fps=get_video_fps(str(vin)); _,W,H=get_video_lwh(str(vin))
pred=torch.load(d/PT,map_location="cpu")
with torch.no_grad(): V=m(**to_cuda(filt(pred["smpl_params_incam"]))).vertices
K=pred["K_fullimg"][0]; r=Renderer(W,H,device="cuda",faces=faces,K=K,bin_size=0)
rd=get_video_reader(str(vin)); w=get_writer(str(d/OUT),fps=fps,crf=23)
for i,img in tqdm(enumerate(rd),total=B):
    if i<A: continue
    if i>=B: break
    out=r.render_mesh(V[i].cuda(),img,(1.0,0.9,0.6))
    cv2.putText(out,f"f{i}",(20,H-30),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),3)
    w.write_frame(out)
w.close(); print("saved",OUT)
