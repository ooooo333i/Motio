"""학회톤 포스터 그림 4종 생성 (matplotlib). 영문 라벨."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
OUT = "/workspace/capstone_deploy/poster/"

# ── 학회톤 팔레트 (muted) ──
INK="#1f2430"; LINE="#8a8f99"
BLU="#dfe7f1"; BLU_E="#4a6491"      # structure (API/job)
GRN="#e2ece1"; GRN_E="#4f7a5b"      # preprocessing
ORG="#f3e3d2"; ORG_E="#b06b39"      # v4 (contribution)
ORG_HI="#ecd3b6"
PUR="#e6e1ee"; PUR_E="#6b5b8a"      # export
RED="#f0dada"; RED_E="#b05050"      # final/result

def box(ax,x,y,w,h,title,sub="",fc="#eee",ec=INK,fs=9,sfs=7.2,bold=True,tc=INK):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.35,rounding_size=1.6",
                fc=fc,ec=ec,lw=1.1))
    ax.text(x,y+(h*0.17 if sub else 0),title,ha="center",va="center",fontsize=fs,
            fontweight="bold" if bold else "normal",color=tc)
    if sub: ax.text(x,y-h*0.21,sub,ha="center",va="center",fontsize=sfs,color="#3a3f4a")

def arr(ax,x1,y1,x2,y2,color=LINE,lw=1.3,style="-|>"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=12,
                lw=lw,color=color,shrinkA=1,shrinkB=1))

def group(ax,x,y,w,h,label,ec):  # 점선 그룹 박스
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.5,rounding_size=2",
                fc="none",ec=ec,lw=1.3,ls=(0,(5,3))))
    if label: ax.text(x+2.5,y+h-2.0,label,ha="left",va="top",fontsize=8.5,
                      fontweight="bold",color=ec)

# ════════════════════════════════════════════════════════════════════
# FIG 1 — 메인 세로 (학회톤)
# ════════════════════════════════════════════════════════════════════
def fig_main():
    fig,ax=plt.subplots(figsize=(8.5,13)); ax.set_xlim(0,100); ax.set_ylim(0,150); ax.axis("off")
    ax.text(50,147,"Motio — Monocular Hand–Body Recovery",ha="center",fontsize=13,fontweight="bold",color=INK)
    ax.text(50,143.5,"service pipeline (default: quality / adaptive v4)",ha="center",fontsize=8.5,color="#555")
    box(ax,50,138,28,5,"Input video","",BLU,BLU_E)
    arr(ax,50,135.5,50,132)
    box(ax,50,128,62,6.5,"API  (main.py)","upload · clip · detect · select-target · job",BLU,BLU_E,sfs=7.5)
    arr(ax,50,124.7,50,121.5)
    box(ax,50,118,62,5.5,"Job pipeline  (jobs.py · GPU lock)","run_extraction → export",BLU,BLU_E,sfs=7.5)
    arr(ax,50,115.2,50,111)
    group(ax,8,21,84,88,"Inference  (entry_point_opt.py · --mode quality)",GRN_E)
    ax.text(50,102.5,"[1]  Preprocessing & merge  (entry_point_hamer)",ha="center",fontsize=8.8,fontweight="bold",color=GRN_E)
    for x,t,s in [(25,"GVHMR","body SMPL-X + cam"),(50,"ViTPose","arm 17 / hand 133"),(75,"HaMeR","hand MANO")]:
        box(ax,x,96,23,7,t,s,GRN,GRN_E,sfs=7)
    for x in (25,50,75): arr(ax,x,92.5,50,89.5)
    box(ax,50,86,48,6,"Merge: GVHMR body + HaMeR fingers","→ smplx_merged_hamer_post.pt",GRN,GRN_E,sfs=7)
    arr(ax,50,83,50,80)
    ax.text(50,78.3,"[2]  Adaptive v4 refinement  (_refine_quality)",ha="center",fontsize=8.8,fontweight="bold",color=ORG_E)
    steps=[("A  arm IK (cap_arm_prefit)","depth anchor + ViTPose conf gating"),
           ("A2  arm smoothing","conf-gated; removes occlusion jitter"),
           ("B  wrist = HaMeR angle","swing + confidence gate + smoothing"),
           ("C  ROM clamp","forearm-axis biomechanical limits"),
           ("D  collision resolve (axcol)","2D-invariant axial rotation"),
           ("E  grip prior (fingers only)","IoU + conf gate; body_pose fixed")]
    y=73
    for i,(t,s) in enumerate(steps):
        box(ax,50,y,64,5.3,t,s,ORG_HI,ORG_E,sfs=7)
        if i<5: arr(ax,50,y-2.65,50,y-4.0)
        y-=6.6
    arr(ax,50,y+1.2,50,y-0.6)
    box(ax,50,y-3.6,58,5.6,"FINAL  =  …_rom_col_grip.pt","indep. MediaPipe 2D↑ best · twist/collision↓ lowest",RED,RED_E,sfs=7)
    arr(ax,50,18.5,50,15)
    box(ax,50,11.5,60,6,"Export  (exporters.py · Blender)","postprocessed.npz → FBX / BVH",PUR,PUR_E,sfs=7.5)
    arr(ax,50,8.5,50,5.5)
    box(ax,50,2.7,30,4.5,"output.fbx / .bvh","",PUR,PUR_E)
    plt.savefig(OUT+"fig1_main.png",dpi=200,bbox_inches="tight",facecolor="white"); plt.close()

# ════════════════════════════════════════════════════════════════════
# FIG 2 — 전처리 내부 상세 (세로)
# ════════════════════════════════════════════════════════════════════
def fig_preproc():
    fig,ax=plt.subplots(figsize=(9,11)); ax.set_xlim(0,100); ax.set_ylim(0,118); ax.axis("off")
    ax.text(50,115,"[1] Preprocessing internals  (entry_point_hamer.py)",ha="center",fontsize=12,fontweight="bold",color=INK)
    ax.text(50,111.5,"GVHMR (body+camera) + ViTPose (2D) + HaMeR (hands) → graft",ha="center",fontsize=8.5,color="#555")
    box(ax,50,105,30,5,"Input video frames","",BLU,BLU_E)
    arr(ax,50,102.5,50,99)
    box(ax,50,95,46,6,"Person detection & tracking","YOLO tracks → person_track_id (CAP_REUSE_BBX_NPZ)",BLU,BLU_E,sfs=7)
    # 3 갈래
    arr(ax,46,92,25,79); arr(ax,50,92,50,86); arr(ax,54,92,75,86)
    # GVHMR 컬럼
    group(ax,6,40,30,38,"GVHMR (world body + camera)",GRN_E)
    box(ax,21,73,26,6,"SLAM camera","SimpleVO/DPVO → w2c, K",GRN,GRN_E,sfs=6.8)
    arr(ax,21,69.7,21,66.5)
    box(ax,21,63,26,6,"focal = image diagonal","≈ √(W²+H²) = 2202  (avoids SLAM blow-up)",GRN,GRN_E,sfs=6.6)
    arr(ax,21,59.7,21,56.5)
    box(ax,21,53,26,6,"HMR4D regressor","ViT feats → SMPL-X body",GRN,GRN_E,sfs=6.8)
    arr(ax,21,49.7,21,46.5)
    box(ax,21,43,26,5.5,"hmr4d_results.pt","body (world+incam)",GRN,GRN_E,sfs=6.8)
    # ViTPose 컬럼 (가운데)
    group(ax,38,46,24,38,"ViTPose 2D",GRN_E)
    box(ax,50,78,20,6,"COCO-17","arm/body 2D",GRN,GRN_E,sfs=6.8)
    arr(ax,50,75,50,72)
    box(ax,50,68.5,20,5.5,"vitpose.pt","wrist/elbow conf",GRN,GRN_E,sfs=6.8)
    box(ax,50,60,20,6,"WholeBody-133","hand 2D (person crop)",GRN,GRN_E,sfs=6.8)
    arr(ax,50,57,50,54)
    box(ax,50,50.5,20,5.5,"wholebody_kp2d.npy","21 pts/hand + conf",GRN,GRN_E,sfs=6.6)
    # HaMeR 컬럼
    group(ax,64,46,30,38,"HaMeR (hands)",GRN_E)
    box(ax,79,78,26,6,"hand bbox","from wrist / WB-133 box",GRN,GRN_E,sfs=6.8)
    arr(ax,79,75,79,72)
    box(ax,79,68.5,26,6,"crop → MANO","per-hand pose (rescale 2.0)",GRN,GRN_E,sfs=6.8)
    arr(ax,79,65.5,79,62.5)
    box(ax,79,59,26,5.5,"bbox temporal smooth","σ frames (stabilize crop)",GRN,GRN_E,sfs=6.6)
    arr(ax,79,56.2,79,53.5)
    box(ax,79,50.5,26,5.5,"hamer_out_clean/","wrist+finger detections",GRN,GRN_E,sfs=6.6)
    # Merge
    arr(ax,21,40,50,33); arr(ax,50,47.7,50,33); arr(ax,79,47.7,50,33)
    box(ax,50,29,64,8,"Merge / graft","GVHMR body  +  HaMeR fingers (mirror L, wrist handling)\n→ smplx_merged_hamer_post.pt   (\"naive graft\")",GRN,GRN_E,sfs=7)
    arr(ax,50,24.8,50,21.5)
    box(ax,50,17.5,52,6,"→ to [2] Adaptive v4 refinement","",ORG_HI,ORG_E)
    plt.savefig(OUT+"fig2_preproc.png",dpi=200,bbox_inches="tight",facecolor="white"); plt.close()

# ════════════════════════════════════════════════════════════════════
# FIG 3 — 가로형 (전처리 상세 포함)
# ════════════════════════════════════════════════════════════════════
def fig_horizontal():
    fig,ax=plt.subplots(figsize=(17,8)); ax.set_xlim(0,200); ax.set_ylim(0,92); ax.axis("off")
    ax.text(100,89,"Motio pipeline (horizontal)  —  preprocessing detail + adaptive v4",ha="center",fontsize=13,fontweight="bold",color=INK)
    # Input
    box(ax,12,46,16,7,"Input\nvideo","detect/track",BLU,BLU_E,sfs=6.5)
    arr(ax,20,46,30,46)
    # Preprocessing column (detailed)
    group(ax,30,10,64,72,"[1] Preprocessing  (entry_point_hamer)",GRN_E)
    # 3 stacked sub-pipelines
    box(ax,46,72,28,7,"GVHMR","SLAM cam · focal=2202 · SMPL-X body",GRN,GRN_E,sfs=6.3)
    box(ax,46,62,28,7,"ViTPose","COCO-17 arm · WholeBody-133 hand",GRN,GRN_E,sfs=6.3)
    box(ax,46,52,28,7,"HaMeR","hand bbox → crop → MANO",GRN,GRN_E,sfs=6.3)
    for yy in (72,62,52): arr(ax,60,yy,74,40)
    box(ax,80,40,22,9,"Merge / graft","GVHMR body\n+ HaMeR fingers\n→ merged_post.pt",GRN,GRN_E,sfs=6.2)
    box(ax,62,20,52,7,"outputs: hmr4d_results.pt · vitpose.pt · wholebody_kp2d.npy · hamer_out_clean/","",GRN,GRN_E,bold=False,fs=6.6)
    arr(ax,91,40,103,46)
    # v4 column
    group(ax,103,8,70,76,"[2] Adaptive v4 refinement  (_refine_quality)",ORG_E)
    vs=[("A  arm IK","depth anchor + conf gate (wrist 14→3.5px)"),
        ("A2  arm smoothing","conf-gated occlusion-jitter removal"),
        ("B  wrist = HaMeR angle","swing + confidence gate"),
        ("C  ROM clamp","forearm-axis biomech limits"),
        ("D  collision resolve","2D-invariant axial rotation"),
        ("E  grip prior (fingers)","IoU + conf gate; body fixed")]
    yy=76
    for i,(t,s) in enumerate(vs):
        box(ax,138,yy,62,7.4,t,s,ORG_HI,ORG_E,fs=8,sfs=6.4)
        if i<5: arr(ax,138,yy-3.7,138,yy-5.0)
        yy-=11.0
    arr(ax,169,46,180,46)
    # final + export
    box(ax,189,52,18,9,"FINAL\n…grip.pt","",RED,RED_E,sfs=6)
    arr(ax,189,47.5,189,40)
    box(ax,189,34,18,8,"Export","FBX/BVH\n(Blender)",PUR,PUR_E,sfs=6.3)
    plt.savefig(OUT+"fig3_horizontal.png",dpi=200,bbox_inches="tight",facecolor="white"); plt.close()

# ════════════════════════════════════════════════════════════════════
# FIG 4 — 평가 결과 3종
# ════════════════════════════════════════════════════════════════════
def fig_eval():
    fig,axs=plt.subplots(1,3,figsize=(15,5.2))
    fig.suptitle("Evaluation  —  non-circular & physical metrics (Ours = adaptive v4)",fontsize=13,fontweight="bold",color=INK,y=1.0)
    OURS=ORG_E; OTH="#9aa0ab"
    # (a) Independent MediaPipe 2D PCK, 8 clips
    ax=axs[0]
    m=["GVHMR/\nHaMeR","fit2d\n(v3)","Naive\ngraft","Ours\nv4"]; v=[0.455,0.566,0.620,0.623]
    cols=[OTH,OTH,OTH,OURS]
    ax.bar(m,v,color=cols,edgecolor=INK,lw=0.8,width=0.62)
    for i,x in enumerate(v): ax.text(i,x+0.008,f"{x:.3f}",ha="center",fontsize=8.5,fontweight="bold" if i==3 else "normal")
    ax.set_title("(a) Independent MediaPipe 2D PCK@0.5\n(8 clips · non-circular)",fontsize=9.5)
    ax.set_ylim(0,0.72); ax.set_ylabel("PCK@0.5 (↑)"); ax.grid(axis="y",ls=":",alpha=0.5)
    # (b) GT-free physical: twist violation & collision (Ours lowest)
    ax=axs[1]
    labels=["twist viol.\n(%)","collision\n(frames, ÷40)"]
    naive=[7.07,243/40]; fit2d=[6.61,391/40]; ours=[5.16,109/40]
    x=np.arange(2); w=0.26
    ax.bar(x-w,naive,w,label="Naive graft",color=OTH,edgecolor=INK,lw=0.7)
    ax.bar(x,fit2d,w,label="fit2d",color="#c4c9d2",edgecolor=INK,lw=0.7)
    ax.bar(x+w,ours,w,label="Ours v4",color=OURS,edgecolor=INK,lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8.5)
    ax.set_title("(b) GT-free physical plausibility\n(lower = better · Ours lowest)",fontsize=9.5)
    ax.set_ylabel("value (↓)"); ax.legend(fontsize=7.5); ax.grid(axis="y",ls=":",alpha=0.5)
    # (c) Dyn-HaMR comparison (input clip, MediaPipe)
    ax=axs[2]
    m=["Naive\ngraft","Dyn-HaMR\n(SOTA)","Ours\nv4"]; v=[0.140,0.199,0.287]
    cols=[OTH,"#7d8da0",OURS]
    ax.bar(m,v,color=cols,edgecolor=INK,lw=0.8,width=0.6)
    for i,x in enumerate(v): ax.text(i,x+0.006,f"{x:.3f}",ha="center",fontsize=8.5,fontweight="bold" if i==2 else "normal")
    ax.set_title("(c) vs Dyn-HaMR — indep. MediaPipe PCK\n(input clip · 442f · published SOTA)",fontsize=9.5)
    ax.set_ylim(0,0.34); ax.set_ylabel("PCK@0.5 (↑)"); ax.grid(axis="y",ls=":",alpha=0.5)
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(OUT+"fig4_eval.png",dpi=200,bbox_inches="tight",facecolor="white"); plt.close()

fig_main(); print("fig1 ok")
fig_preproc(); print("fig2 ok")
fig_horizontal(); print("fig3 ok")
fig_eval(); print("fig4 ok")
