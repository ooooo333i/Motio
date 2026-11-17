"""
클립 렌더: SMPL-X mesh overlay (K 동적, 영상별 해상도).
usage: python batch_render.py <clip_dir> <pt_name> <out_mp4>
"""
import os,sys
from pathlib import Path
import torch, cv2
from tqdm import tqdm
PROJ=Path('/workspace/capstone_deploy'); sys.path.insert(0,str(PROJ)); os.chdir(str(PROJ))
clip_dir=Path(sys.argv[1]); pt_name=sys.argv[2]; out_mp4=sys.argv[3]
def filt(p):
    a={"betas","body_pose","global_orient","transl","left_hand_pose","right_hand_pose","jaw_pose","leye_pose","reye_pose","expression"}
    return {k:v for k,v in p.items() if k in a}
from hmr4d.utils.video_io_utils import get_video_fps,get_video_lwh,get_video_reader,get_writer
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.net_utils import to_cuda
from hmr4d.utils.vis.renderer import Renderer
# 입력 영상: argv[4] 명시 > clip_dir/0_input_video.mp4
if len(sys.argv)>4:
    vin=Path(sys.argv[4])
else:
    vin=clip_dir/"0_input_video.mp4"
    if not vin.exists():
        cands=list(clip_dir.glob("*input*.mp4"))+list(clip_dir.glob("*.mp4"))
        vin=cands[0]
m=make_smplx("supermotion_fullhands").cuda(); faces=m.faces
fps=get_video_fps(str(vin)); _,W,H=get_video_lwh(str(vin))
pred=torch.load(clip_dir/pt_name,map_location="cpu")
inc=pred["smpl_params_incam"]
F=inc["body_pose"].shape[0]
# body_pose (F,21,3)->(F,63) 등 make_smplx 호환
p=filt(inc)
with torch.no_grad(): V=m(**to_cuda(p)).vertices
K=pred["K_fullimg"]; K=K[0] if K.dim()==3 else K
r=Renderer(W,H,device="cuda",faces=faces,K=K,bin_size=0)
rd=get_video_reader(str(vin)); w=get_writer(str(clip_dir/out_mp4),fps=fps,crf=15)
for i,img in tqdm(enumerate(rd),total=F):
    if i>=F: break
    out=r.render_mesh(V[i].cuda(),img,(1.0,0.9,0.6))
    cv2.putText(out,f"f{i}",(20,H-30),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),3)
    w.write_frame(out)
w.close(); print(f"[render] saved {clip_dir/out_mp4}")
