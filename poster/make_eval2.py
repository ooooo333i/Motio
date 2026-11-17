"""평가 그림 재생성 (4 베이스라인: GVHMR raw / Wrist+smooth / Naive graft / Ours). 숫자 실측."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
plt.rcParams.update({"font.family":"DejaVu Sans"})
OUT="/workspace/capstone_deploy/poster/"
INK="#1b2230"; GOLD="#d9821f"; GOLDF="#f4d9b0"; GREY="#aab0bb"; GREY2="#c4c9d2"; DHM="#5f7088"

# 실측 (8클립 독립 MediaPipe 11점 / Dyn-HaMR input 442f / twist 8클립)
MP_lab=["GVHMR\nraw","Wrist+\nsmooth","Naive\ngraft","Ours\nv4"]
MP_v=[0.180,0.203,0.204,0.235]
DH_lab=["Naive\ngraft","Dyn-HaMR\n(SOTA)","Ours\nv4"]; DH_v=[0.140,0.199,0.287]
TW_lab=["Wrist+smooth\n(no limit)","Ours v4\n(forearm-axis ROM)"]; TW_v=[3.41,2.11]

def starbar(ax,labs,vals,star,title,ylab,ymax,fmt="{:.3f}",basec=GREY):
    cols=[basec]*len(vals); cols[star]=GOLD
    ax.bar(labs,vals,color=cols,edgecolor=INK,lw=1.0,width=0.62)
    for i,v in enumerate(vals):
        ax.text(i,v+ymax*0.015,fmt.format(v),ha="center",
                fontsize=11 if i==star else 9.3,fontweight="bold" if i==star else "normal",
                color=GOLD if i==star else INK)
    ax.set_title(title,fontsize=10.5,fontweight="bold",color=INK,pad=8)
    ax.set_ylabel(ylab,fontsize=9.3); ax.set_ylim(0,ymax); ax.grid(axis="y",ls=":",alpha=0.5)
    for s in ("top","right"): ax.spines[s].set_visible(False)

# ───────── FIG 4 (학회톤, 3패널) ─────────
fig,axs=plt.subplots(1,3,figsize=(15,5))
fig.suptitle("Evaluation — independent & physical metrics (Ours = adaptive v4)",
             fontsize=13,fontweight="bold",color=INK,y=1.0)
starbar(axs[0],MP_lab,MP_v,3,"(a) Independent MediaPipe 2D PCK\n(8 clips · non-circular · 11 kpts)","PCK@0.5 (↑)",0.27)
axs[0].text(1.5,0.255,"Ours best",fontsize=9,color=GOLD,fontweight="bold",ha="center")
starbar(axs[1],DH_lab,DH_v,2,"(b) vs Dyn-HaMR — published SOTA\n(input clip · 442f · independent)","PCK@0.5 (↑)",0.33)
axs[1].annotate("+44%",xy=(2,0.287),xytext=(1.05,0.30),fontsize=12,color=GOLD,fontweight="bold",
                arrowprops=dict(arrowstyle="->",color=GOLD,lw=1.6))
# (c) twist — 정직 프레이밍
ax=axs[2]
ax.bar(TW_lab,TW_v,color=[GREY,GOLD],edgecolor=INK,lw=1.0,width=0.5)
for i,v in enumerate(TW_v): ax.text(i,v+0.08,f"{v:.2f}%",ha="center",fontsize=10,
                                    fontweight="bold" if i==1 else "normal",color=GOLD if i==1 else INK)
ax.set_title("(c) Forearm-twist: physical when performed\n(GVHMR/Naive = 0% only by NOT twisting)",fontsize=10.5,fontweight="bold",color=INK,pad=8)
ax.set_ylabel("axial ROM violation % (↓)",fontsize=9.3); ax.set_ylim(0,4.2); ax.grid(axis="y",ls=":",alpha=0.5)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.text(0.5,3.9,"both perform HaMeR twist; Ours keeps it physical",fontsize=8,style="italic",color="#666",ha="center")
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(OUT+"fig4_eval.png",dpi=200,bbox_inches="tight",facecolor="white"); plt.close()
print("fig4 ok")

# ───────── FIG 5 (임팩트) ─────────
fig=plt.figure(figsize=(15,8.2)); fig.patch.set_facecolor("white")
fig.text(0.5,0.955,"Adaptive v4 — Headline Results",ha="center",fontsize=20,fontweight="bold",color=INK)
fig.text(0.5,0.912,"scored by an independent (non-circular) referee — MediaPipe Hands · baselines isolate our wrist contribution",
         ha="center",fontsize=10,color="#555")
axb=fig.add_axes([0.04,0.60,0.92,0.25]); axb.set_xlim(0,100); axb.set_ylim(0,20); axb.axis("off")
def badge(x,big,lab):
    axb.add_patch(FancyBboxPatch((x-11,2),22,16,boxstyle="round,pad=0.4,rounding_size=2.2",fc=GOLDF,ec=GOLD,lw=2.0))
    axb.text(x,13.2,big,ha="center",va="center",fontsize=29,fontweight="bold",color=INK)
    axb.text(x,5.2,lab,ha="center",va="center",fontsize=9.0,color="#2a2f3a")
badge(13,"#1","beats GVHMR · Naive ·\nWrist-merge · Dyn-HaMR\non independent 2D")
badge(38,"+44%","2D PCK vs Dyn-HaMR\n(published SOTA)")
badge(63,"+31%","2D PCK vs\nGVHMR raw")
badge(88,"physical","performs forearm-twist\n& keeps it in ROM\n(naive: no twist)")
gs=fig.add_gridspec(1,2,left=0.07,right=0.95,top=0.50,bottom=0.10,wspace=0.22)
ax=fig.add_subplot(gs[0,0]); starbar(ax,MP_lab,MP_v,3,"Independent 2D accuracy (8 clips)","PCK@0.5 (↑)",0.27)
ax.text(1.5,0.255,"best of all",fontsize=9.5,color=GOLD,fontweight="bold",ha="center")
ax=fig.add_subplot(gs[0,1]); starbar(ax,DH_lab,DH_v,2,"vs Dyn-HaMR — published SOTA","PCK@0.5 (↑)",0.33)
ax.annotate("+44%",xy=(2,0.287),xytext=(1.05,0.30),fontsize=13,color=GOLD,fontweight="bold",
            arrowprops=dict(arrowstyle="->",color=GOLD,lw=1.8))
fig.text(0.5,0.03,"All values measured on this project's pipeline output; non-circular / physical metrics.",
         ha="center",fontsize=8,color="#888",style="italic")
plt.savefig(OUT+"fig5_impact.png",dpi=200,bbox_inches="tight",facecolor="white"); plt.close()
print("fig5 ok")
