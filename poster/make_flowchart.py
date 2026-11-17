"""Motio 서비스 플로우차트 PNG 생성 (matplotlib). 영문 라벨(포스터용)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 15))
ax.set_xlim(0, 100); ax.set_ylim(0, 150); ax.axis("off")

C = dict(api="#cfe2f3", pre="#d9ead3", v4="#fce5cd", v4hi="#f9cb9c",
         exp="#d9d2e9", fin="#ea9999", line="#666666")

def box(x, y, w, h, title, sub="", fc="#eee", ec="#333", bold=True, fs=10, sfs=8):
    p = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.4,rounding_size=2",
                       fc=fc, ec=ec, lw=1.4)
    ax.add_patch(p)
    ax.text(x, y+(h*0.16 if sub else 0), title, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal")
    if sub:
        ax.text(x, y-h*0.22, sub, ha="center", va="center", fontsize=sfs, color="#333")

def arrow(x1, y1, x2, y2, style="-|>", lw=1.6, color=C["line"]):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=14, lw=lw, color=color, shrinkA=2, shrinkB=2))

ax.text(50, 147, "Motio  —  Monocular Hand–Body Recovery Pipeline",
        ha="center", fontsize=14, fontweight="bold")
ax.text(50, 143.5, "(service flow, latest:  default mode = quality / adaptive v4)",
        ha="center", fontsize=9, color="#555")

# ── Upload ──
box(50, 138, 30, 5, "Upload video", "user", C["api"])
arrow(50, 135.5, 50, 132)

# ── API layer ──
box(50, 128, 64, 7, "A.  API layer   (motio_backend/main.py)",
    "upload → clip(trim) → detect people → select target → create job", C["api"], fs=10, sfs=8)
arrow(50, 124.5, 50, 121)

# ── Job pipeline header ──
box(50, 117.5, 64, 6, "B.  Job pipeline   (jobs.py · GPU lock)",
    "run_extraction  →  export", C["api"], fs=10)
arrow(50, 114.5, 50, 110.5)

# ── Inference container ──
ax.add_patch(FancyBboxPatch((7, 20), 86, 88, boxstyle="round,pad=0.6,rounding_size=3",
             fc="#ffffff", ec="#888", lw=1.6, ls="--"))
ax.text(50, 105.5, "Inference   (entry_point_opt.py,  --mode quality)",
        ha="center", fontsize=11, fontweight="bold", color="#333")

# ── [1] Preprocessing ──
ax.text(50, 101.5, "[1]  Preprocessing & merge   (entry_point_hamer.py)",
        ha="center", fontsize=9.5, fontweight="bold", color="#38761d")
box(24, 95, 24, 8, "GVHMR", "body SMPL-X\n+ camera", C["pre"], fs=10, sfs=7.5)
box(50, 95, 24, 8, "ViTPose", "arm COCO17\nhand WB-133", C["pre"], fs=10, sfs=7.5)
box(76, 95, 24, 8, "HaMeR", "hand MANO\ndetections", C["pre"], fs=10, sfs=7.5)
for x in (24, 50, 76): arrow(x, 91, 50, 88)
box(50, 85, 50, 6.5, "Merge  =  GVHMR body  +  HaMeR fingers",
    "→ smplx_merged_hamer_post.pt   (\"naive graft\" baseline)", C["pre"], fs=9.5, sfs=7.5)
arrow(50, 81.7, 50, 78.5)

# ── [2] v4 chain ──
ax.text(50, 77, "[2]  Adaptive v4 refinement   (_refine_quality)",
        ha="center", fontsize=9.5, fontweight="bold", color="#b45f06")
steps = [
    ("A   arm IK  (cap_arm_prefit)", "depth anchor + ViTPose conf gating   → wrist 14→3.5px"),
    ("A2  arm smoothing", "conf-gated; removes occlusion jitter, keeps visible motion"),
    ("B   wrist = HaMeR angle", "swing from HaMeR + confidence gate + smoothing"),
    ("C   ROM clamp", "forearm-axis biomechanical wrist limits"),
    ("D   collision resolve  (axcol)", "detect → 2D-invariant axial rotation (early-stop)"),
    ("E   grip prior  (fingers only)", "IoU-overlap + conf gate;  body_pose unchanged"),
]
y = 72
for i, (t, s) in enumerate(steps):
    box(50, y, 66, 5.6, t, s, C["v4hi"], ec="#b45f06", fs=9.5, sfs=7.3)
    if i < len(steps)-1: arrow(50, y-2.8, 50, y-4.2)
    y -= 7.0

# final pt  (y == 30 here)
arrow(50, y+1.0, 50, y-1.0)
box(50, y-4.2, 60, 6, "FINAL  =  hamer_wrist_hamer_rom_col_grip.pt",
    "independent MediaPipe 2D ↑ best · twist/collision ↓ lowest", C["fin"], fs=9.5, sfs=7.3)

# ── Export ──
yE = 13
arrow(50, 19.5, 50, yE+3.5)
box(50, yE, 64, 7, "C.  Export   (exporters.py · Blender)",
    "final.pt → postprocessed.npz → FBX (Unreal Z-up) / BVH", C["exp"], fs=10, sfs=8)
arrow(50, yE-3.5, 50, yE-6.5)
box(50, yE-9.5, 34, 5.5, "output.fbx  /  .bvh", "download", C["exp"], fs=10)

plt.tight_layout()
out = "/workspace/capstone_deploy/poster/service_flow.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", out)
