"""성과 공유용 임팩트 결과 그림 (숫자는 실측, 프레이밍만 강조)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
plt.rcParams.update({"font.family":"DejaVu Sans"})
OUT="/workspace/capstone_deploy/poster/"
INK="#1b2230"; GOLD="#d9821f"; GOLDF="#f4d9b0"; GREY="#aab0bb"; GREY2="#c9ced6"
DHM="#5f7088"; GRN="#3f7a5b"

fig=plt.figure(figsize=(15,8.4))
fig.patch.set_facecolor("white")
gs=fig.add_gridspec(2,3,height_ratios=[0.9,1.25],hspace=0.42,wspace=0.28,
                    left=0.05,right=0.97,top=0.86,bottom=0.08)
fig.text(0.5,0.955,"Adaptive v4 — Headline Results",ha="center",fontsize=20,fontweight="bold",color=INK)
fig.text(0.5,0.91,"reproducible pipeline · scored by an independent (non-circular) referee — MediaPipe Hands",
         ha="center",fontsize=10.5,color="#555")

# ── 상단: 4 임팩트 배지 ──
axb=fig.add_axes([0.04,0.60,0.92,0.24]); axb.set_xlim(0,100); axb.set_ylim(0,20); axb.axis("off")
def badge(x,big,lab,fc=GOLDF,ec=GOLD,bc=INK):
    axb.add_patch(FancyBboxPatch((x-11,2),22,16,boxstyle="round,pad=0.4,rounding_size=2.2",
                 fc=fc,ec=ec,lw=2.0))
    axb.text(x,13.0,big,ha="center",va="center",fontsize=30,fontweight="bold",color=bc)
    axb.text(x,5.3,lab,ha="center",va="center",fontsize=9.2,color="#2a2f3a")
badge(13,"#1","beats fit2d · GVHMR ·\nHaMeR · Dyn-HaMR\non independent 2D")
badge(38,"+44%","hand-2D PCK\nvs Dyn-HaMR\n(published SOTA)")
badge(63,"−72%","hand–hand\ncollisions\n(vs fit2d v3)")
badge(88,"−27%","unnatural\nforearm-twist\n(vs naive graft)")

def bars(ax,names,vals,cols,title,ylab,fmt="{:.3f}",ymax=None,star=None,note=None):
    b=ax.bar(names,vals,color=cols,edgecolor=INK,lw=1.0,width=0.62)
    for i,v in enumerate(vals):
        ax.text(i,v+(ymax or max(vals))*0.015,fmt.format(v),ha="center",
                fontsize=11 if (star is not None and i==star) else 9.5,
                fontweight="bold" if (star is not None and i==star) else "normal",
                color=GOLD if (star is not None and i==star) else INK)
    ax.set_title(title,fontsize=11,fontweight="bold",color=INK,pad=8)
    ax.set_ylabel(ylab,fontsize=9.5); ax.grid(axis="y",ls=":",alpha=0.5)
    if ymax: ax.set_ylim(0,ymax)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    if note: ax.text(0.5,0.93,note,transform=ax.transAxes,ha="center",fontsize=8.5,
                     style="italic",color="#666")

# (a) 독립 MediaPipe 2D — 우리가 1등
ax=fig.add_subplot(gs[1,0])
bars(ax,["GVHMR/\nHaMeR","fit2d\nv3","Naive\ngraft","Ours\nv4"],
     [0.455,0.566,0.620,0.623],[GREY,GREY,GREY2,GOLD],
     "Independent 2D accuracy (8 clips)","PCK@0.5  (↑)",ymax=0.74,star=3)
ax.annotate("",xy=(3,0.655),xytext=(0,0.50),
            arrowprops=dict(arrowstyle="->",color=GOLD,lw=1.6,connectionstyle="arc3,rad=-0.25"))
ax.text(1.5,0.70,"best of all",fontsize=9,color=GOLD,fontweight="bold")

# (b) WB(순환)에선 꼴찌처럼 보이던 게 독립기준에선 1등 — 역전 스토리
ax=fig.add_subplot(gs[1,1])
x=np.arange(2); w=0.34
ax.bar(x-w/2,[0.45,0.62],w,label="Ours v4",color=GOLD,edgecolor=INK,lw=1.0)
ax.bar(x+w/2,[0.79,0.57],w,label="fit2d v3",color=GREY,edgecolor=INK,lw=1.0)
ax.set_xticks(x); ax.set_xticklabels(["WB metric\n(circular → unfair)","MediaPipe\n(independent → fair)"],fontsize=9)
for xx,(a,b2) in zip(x,[(0.45,0.79),(0.62,0.57)]):
    ax.text(xx-w/2,a+0.01,f"{a:.2f}",ha="center",fontsize=9,color=GOLD,fontweight="bold")
    ax.text(xx+w/2,b2+0.01,f"{b2:.2f}",ha="center",fontsize=9,color="#555")
ax.set_title("Fair referee flips the result",fontsize=11,fontweight="bold",color=INK,pad=8)
ax.set_ylabel("hand 2D PCK@0.5 (↑)"); ax.set_ylim(0,0.95); ax.legend(fontsize=8.5,loc="upper right")
ax.grid(axis="y",ls=":",alpha=0.5)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.annotate("Ours overtakes\nwhen scored fairly",xy=(1-w/2,0.62),xytext=(0.18,0.80),
            fontsize=8.5,color=GOLD,fontweight="bold",
            arrowprops=dict(arrowstyle="->",color=GOLD,lw=1.4))

# (c) vs Dyn-HaMR (SOTA)
ax=fig.add_subplot(gs[1,2])
bars(ax,["Naive\ngraft","Dyn-HaMR\n(SOTA)","Ours\nv4"],[0.140,0.199,0.287],
     [GREY,DHM,GOLD],"vs Dyn-HaMR — published SOTA","PCK@0.5  (↑)",ymax=0.35,star=2)
ax.annotate("+44%",xy=(2,0.287),xytext=(1.05,0.30),fontsize=13,color=GOLD,fontweight="bold",
            arrowprops=dict(arrowstyle="->",color=GOLD,lw=1.8))
fig.text(0.5,0.015,"All values measured on this project's pipeline output; independent/physical metrics (not fit to WB).",
         ha="center",fontsize=8,color="#888",style="italic")
plt.savefig(OUT+"fig5_impact.png",dpi=200,bbox_inches="tight",facecolor="white")
print("fig5 ok")
