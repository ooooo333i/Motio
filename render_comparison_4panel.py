"""
render_comparison_4panel.py

4-panel comparison video (2×2 grid, h264):
  ┌─────────────────────┬──────────────────────┐
  │ 원본 (Raw input)    │ Stage 1: Arm prefit  │
  ├─────────────────────┼──────────────────────┤
  │ Stage 2: Merge      │ Stage 3: Final       │
  │ smplx_merged_prefit │ smplx_merged_stage4  │
  └─────────────────────┴──────────────────────┘

Output resolution = source resolution (1080×1920), each cell = 540×960.
Frame number shown in every cell.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PANELS = [
    # (video filename, label text, BGR label colour)
    ("0_input_video.mp4",               "Raw input",              (200, 200, 200)),
    ("render_stage1_arm_prefit_incam.mp4", "Stage 1: Arm prefit", (255, 200,  80)),
    ("render_merged_prefit_incam.mp4",  "Stage 2: Merge (prefit)", (80, 220, 120)),
    ("render_stage4_final_incam.mp4",   "Stage 3: Final",         (80, 180, 255)),
]

LABEL_H = 48   # height of the label bar at top of each cell
FRAME_FONT_SCALE = 0.55
LABEL_FONT_SCALE = 0.70


def put_label(cell: np.ndarray, text: str, color_bgr: tuple) -> np.ndarray:
    """Prepend a dark label bar with coloured text above `cell`."""
    w = cell.shape[1]
    bar = np.zeros((LABEL_H, w, 3), dtype=np.uint8)
    # dim tint of label colour as background
    bar[:] = (int(color_bgr[0] * 0.22),
              int(color_bgr[1] * 0.22),
              int(color_bgr[2] * 0.22))
    cv2.putText(bar, text, (10, LABEL_H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
                color_bgr, 2, cv2.LINE_AA)
    return np.concatenate([bar, cell], axis=0)


def put_frame_num(cell: np.ndarray, fidx: int) -> np.ndarray:
    """Draw frame number in bottom-right corner of `cell` (in-place, returns cell)."""
    h, w = cell.shape[:2]
    text = f"f{fidx}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                  FRAME_FONT_SCALE, 1)
    x = w - tw - 8
    y = h - 8
    cv2.putText(cell, text, (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, FRAME_FONT_SCALE,
                (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(cell, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, FRAME_FONT_SCALE,
                (240, 240, 240), 1, cv2.LINE_AA)
    return cell


def main():
    parser = argparse.ArgumentParser(description="4-panel comparison video")
    parser.add_argument("--input-dir", default="outputs/cap_pipeline/input")
    parser.add_argument("--output",    default="outputs/cap_pipeline/input/render_comparison_4panel_h264.mp4")
    parser.add_argument("--crf",       type=int, default=20)
    args = parser.parse_args()

    d = Path(args.input_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Open all captures
    caps = []
    for fname, _, _ in PANELS:
        p = d / fname
        if not p.exists():
            print(f"[4panel] ERROR: missing {p}", file=sys.stderr)
            sys.exit(1)
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            print(f"[4panel] ERROR: cannot open {p}", file=sys.stderr)
            sys.exit(1)
        caps.append(cap)

    # Source resolution from first panel
    src_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = caps[0].get(cv2.CAP_PROP_FPS)
    total = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

    cell_w = src_w // 2   # 540
    cell_h = src_h // 2   # 960

    # cell_h includes the label bar, so actual image area = cell_h - LABEL_H
    img_h = cell_h - LABEL_H

    out_w = src_w    # 1080
    out_h = src_h    # 1920

    print(f"[4panel] Source: {src_w}×{src_h}  {fps:.2f}fps  {total} frames")
    print(f"[4panel] Cell:   {cell_w}×{cell_h} (img {cell_w}×{img_h} + label {LABEL_H}px)")
    print(f"[4panel] Output: {out_w}×{out_h}")

    # Pipe raw BGR frames into ffmpeg for h264 encoding
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(args.crf),
        "-preset", "fast",
        str(out),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"[4panel] Encoding → {out}  (crf={args.crf})")
    fidx = 0
    while True:
        cells = []
        all_ok = True
        for i, (cap, (_, label, color)) in enumerate(zip(caps, PANELS)):
            ok, frame = cap.read()
            if not ok:
                all_ok = False
                break
            # Resize to cell image area (width=cell_w, height=img_h)
            img_cell = cv2.resize(frame, (cell_w, img_h),
                                  interpolation=cv2.INTER_AREA)
            put_frame_num(img_cell, fidx)
            # Add label bar on top → final cell is cell_w × cell_h
            img_cell = put_label(img_cell, label, color)
            cells.append(img_cell)

        if not all_ok or len(cells) < 4:
            break

        # Compose 2×2 grid: [[TL, TR], [BL, BR]]
        top_row    = np.concatenate([cells[0], cells[1]], axis=1)  # (cell_h, out_w, 3)
        bottom_row = np.concatenate([cells[2], cells[3]], axis=1)
        grid       = np.concatenate([top_row, bottom_row], axis=0)  # (out_h, out_w, 3)

        assert grid.shape == (out_h, out_w, 3), \
            f"grid shape mismatch: {grid.shape} vs ({out_h},{out_w},3)"

        proc.stdin.write(grid.tobytes())
        fidx += 1

        if fidx % 100 == 0:
            print(f"[4panel]  {fidx}/{total}")

    proc.stdin.close()
    proc.wait()

    for cap in caps:
        cap.release()

    print(f"[4panel] Done — {fidx} frames written → {out}")


if __name__ == "__main__":
    main()
