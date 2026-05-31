"""
analyze_axial_flip.py

wrist LOCAL rotation을 Euler XYZ로 분해:
  X = axial (pronation/supination, forearm axis)
  Y = flexion/extension
  Z = ulnar/radial deviation

총각도(geodesic)가 144°이지만 axial 성분이 ±150°+로 튀는 flip 프레임을 수치로 입증.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJ_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJ_ROOT / "outputs/cap_pipeline/input/wrist_refine_debug/axial_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUSPICIOUS_FRAMES = list(range(355, 376)) + list(range(400, 425))
NORMAL_FRAMES = [190, 195, 200, 205, 210]
AXIAL_FLIP_THRESH = 150.0   # deg — per-axis constraint candidate


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def aa_to_rotmat_np(aa: np.ndarray) -> np.ndarray:
    """(3,) axis-angle → (3,3) rotation matrix (Rodrigues)."""
    theta = float(np.linalg.norm(aa))
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64)
    axis = aa / theta
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    t = 1.0 - c
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ], dtype=np.float64)


def rotmat_to_euler_xyz(R: np.ndarray) -> tuple[float, float, float]:
    """
    R = Rx(ex) @ Ry(ey) @ Rz(ez)  (intrinsic XYZ, 손목 로컬 관례)
    반환: (ex, ey, ez) in degrees.
    Gimbal lock (ey ≈ ±90°) 시 ez=0 으로 고정.
    """
    # R[2,0] = -sin(ey)
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    ey = np.degrees(np.arcsin(sy))          # Y: flexion/ext
    cy = np.cos(np.arcsin(sy))

    if abs(cy) > 1e-6:
        ex = np.degrees(np.arctan2(R[2, 1], R[2, 2]))   # X: axial
        ez = np.degrees(np.arctan2(R[1, 0], R[0, 0]))   # Z: deviation
    else:
        # gimbal lock: ez 축 자유도 사라짐 → ez=0 처리
        ex = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        ez = 0.0

    return float(ex), float(ey), float(ez)


def geodesic_deg_from_aa(aa: np.ndarray) -> float:
    """axis-angle → geodesic angle (deg) from identity."""
    return float(np.degrees(np.linalg.norm(aa)))


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────────────────────
data_path = PROJ_ROOT / "outputs/cap_pipeline/input/smplx_merged_stage4.pt"
print(f"[axial] Loading {data_path} ...")
merged = torch.load(data_path, map_location="cpu")
params = merged["smpl_params_incam"]

body_pose = params["body_pose"].numpy()   # (F, 21, 3)
F = body_pose.shape[0]
print(f"[axial] Total frames: {F}")

# SMPL-X body_pose index → full joint:  body_pose[i] = full_joint(i+1)
# left_wrist  = full joint 20 → body_pose index 19
# right_wrist = full joint 21 → body_pose index 20
LEFT_WR_IDX  = 19
RIGHT_WR_IDX = 20


# ──────────────────────────────────────────────────────────────────────────────
# 전체 프레임 Euler 분해
# ──────────────────────────────────────────────────────────────────────────────
print("[axial] Decomposing wrist rotations ...")

results = {side: {"axial": [], "flex": [], "dev": [], "geodesic": []}
           for side in ("left", "right")}

for fidx in range(F):
    for side, bpidx in [("left", LEFT_WR_IDX), ("right", RIGHT_WR_IDX)]:
        aa = body_pose[fidx, bpidx]          # (3,) local axis-angle
        R  = aa_to_rotmat_np(aa)
        ex, ey, ez = rotmat_to_euler_xyz(R)
        geo = geodesic_deg_from_aa(aa)
        results[side]["axial"].append(ex)
        results[side]["flex"].append(ey)
        results[side]["dev"].append(ez)
        results[side]["geodesic"].append(geo)


# ──────────────────────────────────────────────────────────────────────────────
# 수치 보고: 의심 구간 vs 정상 구간
# ──────────────────────────────────────────────────────────────────────────────
SIDE_IDX = {"left": LEFT_WR_IDX, "right": RIGHT_WR_IDX}

print("\n" + "="*80)
print("WRIST LOCAL ROTATION — Euler XYZ decomposition")
print("  X = axial (pronation/supination)  Y = flexion/ext  Z = deviation")
print("  Geodesic = ||axis-angle|| (same as rotation angle from rest pose)")
print("="*80)

for side in ("left", "right"):
    ax = np.array(results[side]["axial"])
    fl = np.array(results[side]["flex"])
    dv = np.array(results[side]["dev"])
    geo = np.array(results[side]["geodesic"])

    print(f"\n{'─'*60}")
    print(f"  {side.upper()} WRIST")
    print(f"{'─'*60}")
    print(f"  {'Frame':>6}  {'Axial(X)':>10}  {'Flex(Y)':>9}  {'Dev(Z)':>8}  {'Geodesic':>10}")

    print(f"\n  [NORMAL frames]")
    for fidx in NORMAL_FRAMES:
        if fidx < F:
            print(f"  {fidx:>6}  {ax[fidx]:>+10.1f}°  {fl[fidx]:>+9.1f}°  {dv[fidx]:>+8.1f}°  {geo[fidx]:>9.1f}°")

    print(f"\n  [SUSPICIOUS frames — f355-375 swing area]")
    for fidx in range(355, 376):
        if fidx < F:
            flag = " *** FLIP?" if abs(ax[fidx]) > AXIAL_FLIP_THRESH else ""
            print(f"  {fidx:>6}  {ax[fidx]:>+10.1f}°  {fl[fidx]:>+9.1f}°  {dv[fidx]:>+8.1f}°  {geo[fidx]:>9.1f}°{flag}")

    print(f"\n  [SUSPICIOUS frames — f400-420]")
    for fidx in range(400, 421):
        if fidx < F:
            flag = " *** FLIP?" if abs(ax[fidx]) > AXIAL_FLIP_THRESH else ""
            print(f"  {fidx:>6}  {ax[fidx]:>+10.1f}°  {fl[fidx]:>+9.1f}°  {dv[fidx]:>+8.1f}°  {geo[fidx]:>9.1f}°{flag}")

    # 전체 flip 프레임 목록
    flip_frames = np.where(np.abs(ax) > AXIAL_FLIP_THRESH)[0]
    print(f"\n  [All frames with |axial| > {AXIAL_FLIP_THRESH}°]")
    print(f"  Count: {len(flip_frames)}")
    if len(flip_frames) > 0:
        # 연속 구간으로 그룹핑
        groups = []
        cur_start = flip_frames[0]
        cur_end = flip_frames[0]
        for f in flip_frames[1:]:
            if f == cur_end + 1:
                cur_end = f
            else:
                groups.append((cur_start, cur_end))
                cur_start = cur_end = f
        groups.append((cur_start, cur_end))
        for gs, ge in groups:
            if gs == ge:
                print(f"    f{gs:04d}  axial={ax[gs]:+.1f}°  geo={geo[gs]:.1f}°")
            else:
                axials_in = ax[gs:ge+1]
                print(f"    f{gs:04d}-f{ge:04d}  axial=[{axials_in.min():+.1f}°..{axials_in.max():+.1f}°]  "
                      f"geo=[{geo[gs]:.1f}°..{geo[ge]:.1f}°]")

    # 정상 구간 axial 통계
    normal_ax = [ax[f] for f in NORMAL_FRAMES if f < F]
    if normal_ax:
        print(f"\n  Normal frames axial range: {min(normal_ax):+.1f}° .. {max(normal_ax):+.1f}°")
        print(f"  Normal frames geodesic range: "
              f"{min(geo[f] for f in NORMAL_FRAMES if f < F):.1f}° .. "
              f"{max(geo[f] for f in NORMAL_FRAMES if f < F):.1f}°")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1: 전체 시계열 (axial + geodesic)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[axial] Generating plots ...")

fig, axes = plt.subplots(2, 2, figsize=(20, 10))
fig.suptitle("Wrist Local Rotation — Euler Axial (X) vs Geodesic Total", fontsize=14)

frames = np.arange(F)

for row, side in enumerate(("left", "right")):
    ax_arr  = np.array(results[side]["axial"])
    fl_arr  = np.array(results[side]["flex"])
    dv_arr  = np.array(results[side]["dev"])
    geo_arr = np.array(results[side]["geodesic"])

    # --- 전체 시계열 ---
    ax_plot = axes[row, 0]
    ax_plot.fill_between(frames, -AXIAL_FLIP_THRESH, AXIAL_FLIP_THRESH,
                         alpha=0.08, color="green", label=f"safe zone ±{AXIAL_FLIP_THRESH}°")
    ax_plot.plot(frames, ax_arr,  color="red",    lw=1.0, label="Axial (X)")
    ax_plot.plot(frames, fl_arr,  color="blue",   lw=0.7, alpha=0.6, label="Flex (Y)")
    ax_plot.plot(frames, dv_arr,  color="purple", lw=0.7, alpha=0.6, label="Dev (Z)")
    ax_plot.plot(frames, geo_arr, color="black",  lw=1.0, ls="--", alpha=0.8, label="Geodesic total")
    ax_plot.axhline(AXIAL_FLIP_THRESH,  color="red", ls=":", lw=1.5)
    ax_plot.axhline(-AXIAL_FLIP_THRESH, color="red", ls=":", lw=1.5)
    # 의심 구간 강조
    for gs, ge in [(355, 375), (400, 420)]:
        ax_plot.axvspan(gs, ge, alpha=0.12, color="orange", label="_suspicious")
    ax_plot.set_title(f"{side.upper()} wrist — all frames")
    ax_plot.set_xlabel("Frame")
    ax_plot.set_ylabel("Degrees")
    ax_plot.set_ylim(-200, 200)
    ax_plot.legend(fontsize=8, loc="upper left")
    ax_plot.grid(True, alpha=0.3)

    # --- 확대: swing 구간 f300-450 ---
    ax_zoom = axes[row, 1]
    zs, ze = 300, min(450, F)
    zframes = frames[zs:ze]
    ax_zoom.fill_between(zframes, -AXIAL_FLIP_THRESH, AXIAL_FLIP_THRESH,
                         alpha=0.08, color="green")
    ax_zoom.plot(zframes, ax_arr[zs:ze],  color="red",    lw=1.5, label="Axial (X)")
    ax_zoom.plot(zframes, fl_arr[zs:ze],  color="blue",   lw=1.0, alpha=0.7, label="Flex (Y)")
    ax_zoom.plot(zframes, dv_arr[zs:ze],  color="purple", lw=1.0, alpha=0.7, label="Dev (Z)")
    ax_zoom.plot(zframes, geo_arr[zs:ze], color="black",  lw=1.5, ls="--", alpha=0.8, label="Geodesic total")
    ax_zoom.axhline(AXIAL_FLIP_THRESH,  color="red", ls=":", lw=2)
    ax_zoom.axhline(-AXIAL_FLIP_THRESH, color="red", ls=":", lw=2, label=f"±{AXIAL_FLIP_THRESH}° threshold")
    for gs, ge in [(355, 375), (400, 420)]:
        ax_zoom.axvspan(gs, ge, alpha=0.15, color="orange", label="_suspicious")
    # 정상 구간 표시
    ax_zoom.axvspan(190, 215, alpha=0.12, color="cyan", label="normal (f190-215)")

    # 의심 프레임 어노테이션
    for fidx in range(zs, ze):
        if abs(ax_arr[fidx]) > AXIAL_FLIP_THRESH:
            ax_zoom.annotate(f"f{fidx}\n{ax_arr[fidx]:+.0f}°",
                             xy=(fidx, ax_arr[fidx]),
                             xytext=(fidx + 2, ax_arr[fidx] + np.sign(ax_arr[fidx]) * 10),
                             fontsize=7, color="red",
                             arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    ax_zoom.set_title(f"{side.upper()} wrist — zoom f300-450")
    ax_zoom.set_xlabel("Frame")
    ax_zoom.set_ylabel("Degrees")
    ax_zoom.set_ylim(-200, 200)
    ax_zoom.legend(fontsize=8, loc="upper left")
    ax_zoom.grid(True, alpha=0.3)

plt.tight_layout()
out_plot = OUT_DIR / "axial_euler_timeseries.png"
plt.savefig(out_plot, dpi=120, bbox_inches="tight")
plt.close()
print(f"[axial] Saved → {out_plot}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2: f200(normal) vs flip 프레임 — bar chart 비교
# ──────────────────────────────────────────────────────────────────────────────
# 대표 프레임 선정: f200(normal), flip 프레임 중 axial 최대값인 것
for side in ("left", "right"):
    ax_arr  = np.array(results[side]["axial"])
    fl_arr  = np.array(results[side]["flex"])
    dv_arr  = np.array(results[side]["dev"])
    geo_arr = np.array(results[side]["geodesic"])

    # flip 프레임 집합
    flip_mask = np.abs(ax_arr) > AXIAL_FLIP_THRESH
    flip_frames_arr = np.where(flip_mask)[0]

    # swing 구간 (f300-450)에서 axial 절댓값 최대 프레임
    swing_range = range(300, min(450, F))
    swing_axials = [(abs(ax_arr[f]), f) for f in swing_range]
    swing_axials.sort(reverse=True)
    top_swing_frames = [f for _, f in swing_axials[:5]]

    compare_frames = NORMAL_FRAMES[:3] + top_swing_frames[:5]
    compare_labels = [f"f{f}\n({'NORMAL' if f in NORMAL_FRAMES else 'SWING'})" for f in compare_frames]

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    x = np.arange(len(compare_frames))
    w = 0.2
    bar_axial = ax2.bar(x - w,   [ax_arr[f] for f in compare_frames], w, label="Axial (X)", color="red", alpha=0.8)
    bar_flex  = ax2.bar(x,       [fl_arr[f] for f in compare_frames], w, label="Flex (Y)",  color="blue", alpha=0.8)
    bar_dev   = ax2.bar(x + w,   [dv_arr[f] for f in compare_frames], w, label="Dev (Z)",   color="purple", alpha=0.8)
    ax2.plot(x + 3*w, [geo_arr[f] for f in compare_frames], "k^--", markersize=8, label="Geodesic", lw=1.5)

    ax2.axhline(AXIAL_FLIP_THRESH,  color="red", ls=":", lw=2, label=f"+{AXIAL_FLIP_THRESH}° axial limit")
    ax2.axhline(-AXIAL_FLIP_THRESH, color="red", ls=":", lw=2)
    ax2.axhspan(-AXIAL_FLIP_THRESH, AXIAL_FLIP_THRESH, alpha=0.05, color="green")

    # 정상 vs swing 구분선
    ax2.axvline(len(NORMAL_FRAMES[:3]) - 0.5, color="gray", ls="--", lw=1.5)
    ax2.text(len(NORMAL_FRAMES[:3]) - 0.5 - 0.3, 160, "← NORMAL", fontsize=9, ha="right", color="gray")
    ax2.text(len(NORMAL_FRAMES[:3]) - 0.5 + 0.1, 160, "SWING →", fontsize=9, ha="left", color="orange")

    ax2.set_xticks(x)
    ax2.set_xticklabels(compare_labels, fontsize=9)
    ax2.set_ylim(-200, 200)
    ax2.set_ylabel("Degrees")
    ax2.set_title(f"{side.upper()} Wrist — Normal vs Swing: Euler XYZ decomposition\n"
                  f"(Axial threshold = ±{AXIAL_FLIP_THRESH}°, Geodesic = black markers)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    out2 = OUT_DIR / f"axial_bar_compare_{side}.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[axial] Saved → {out2}")


# ──────────────────────────────────────────────────────────────────────────────
# 최종 요약
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY — per-axis constraint analysis")
print("="*80)
for side in ("left", "right"):
    ax_arr  = np.array(results[side]["axial"])
    geo_arr = np.array(results[side]["geodesic"])

    flip_frames_list = np.where(np.abs(ax_arr) > AXIAL_FLIP_THRESH)[0]
    print(f"\n{side.upper()} WRIST:")
    print(f"  axial |>150°| frames: {len(flip_frames_list)} / {F}")
    if len(flip_frames_list):
        print(f"  axial max: {ax_arr[flip_frames_list].max():+.1f}°  min: {ax_arr[flip_frames_list].min():+.1f}°")
        print(f"  geodesic at those frames: min={geo_arr[flip_frames_list].min():.1f}°  max={geo_arr[flip_frames_list].max():.1f}°")
        overlap_swing = [f for f in flip_frames_list if 355 <= f <= 420]
        print(f"  overlap with swing region f355-420: {len(overlap_swing)} frames {overlap_swing[:10]}")

    # 정상 구간
    norm_ax = ax_arr[[f for f in NORMAL_FRAMES if f < F]]
    norm_geo = geo_arr[[f for f in NORMAL_FRAMES if f < F]]
    print(f"  normal (f190-210) axial: [{norm_ax.min():+.1f}°..{norm_ax.max():+.1f}°]  "
          f"geodesic: [{norm_geo.min():.1f}°..{norm_geo.max():.1f}°]")
    print(f"  → geodesic constraint (144°) {'WOULD MISS' if (geo_arr[flip_frames_list] < 144).any() else 'would catch'} some flip frames")

print(f"\n[axial] All outputs → {OUT_DIR}")
print("[axial] Done.")
