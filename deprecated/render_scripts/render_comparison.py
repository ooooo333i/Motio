"""
render_comparison.py

4개 단계 영상을 2×2 그리드로 합쳐 비교 영상을 생성합니다.

레이아웃 (각 셀 1/3 scale = 360×640):
  [Baseline (GVHMR)]    [Stage 1: Arm pre-fit]
  [Stage 2: Wrist ins]  [Stage 3: Palm merge]

출력: outputs/cap_pipeline/input/render_comparison_2x2.mp4
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent
os.chdir(str(PROJ_ROOT))

CELL_SCALE = 1 / 3   # 1080→360  1920→640

PANELS = [
    ("render_baseline_incam.mp4",              "Baseline (GVHMR)",     (200, 200, 200)),
    ("render_stage1_arm_prefit_incam.mp4",     "Stage 1: Arm pre-fit", (80, 200, 255)),
    ("render_stage2_wrist_inserted_incam.mp4", "Stage 2: Wrist ins.",  (80, 220, 120)),
    ("render_merged_prefit_incam.mp4",         "Stage 3: Palm merge",  (255, 180, 80)),
]

LABEL_H = 40   # pixels for label bar at top of each cell


def put_label(cell: np.ndarray, text: str, color: tuple) -> np.ndarray:
    h, w = cell.shape[:2]
    bar = np.zeros((LABEL_H, w, 3), dtype=np.uint8)
    # background tint
    bar[:] = (int(color[2]*0.3), int(color[1]*0.3), int(color[0]*0.3))
    cv2.putText(bar, text, (8, LABEL_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return np.concatenate([bar, cell], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/cap_pipeline/input")
    parser.add_argument("--scale", type=float, default=CELL_SCALE)
    parser.add_argument("--crf", type=int, default=23)
    args = parser.parse_args()

    d = Path(args.input_dir)
    scale = args.scale

    # Open all video captures
    caps = []
    for fname, label, color in PANELS:
        path = d / fname
        if not path.exists():
            print(f"[comp] 파일 없음: {path}")
            raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(path))
        caps.append((cap, label, color))
        print(f"[comp] opened {fname}  "
              f"{int(cap.get(3))}×{int(cap.get(4))}  "
              f"{cap.get(5):.2f}fps  {int(cap.get(7))}f")

    fps        = caps[0][0].get(cv2.CAP_PROP_FPS)
    n_frames   = int(caps[0][0].get(cv2.CAP_PROP_FRAME_COUNT))
    orig_h     = int(caps[0][0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w     = int(caps[0][0].get(cv2.CAP_PROP_FRAME_WIDTH))

    cell_w = int(orig_w * scale)
    cell_h = int(orig_h * scale)
    grid_w = cell_w * 2
    grid_h = (cell_h + LABEL_H) * 2

    out_path = d / "render_comparison_2x2.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (grid_w, grid_h))

    print(f"[comp] Grid: {grid_w}×{grid_h}  {fps:.2f}fps  {n_frames}frames")
    print(f"[comp] Writing → {out_path}")

    for fidx in range(n_frames):
        cells = []
        for cap, label, color in caps:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            small = cv2.resize(frame, (cell_w, cell_h),
                               interpolation=cv2.INTER_AREA)
            cells.append(put_label(small, label, color))

        row0 = np.concatenate([cells[0], cells[1]], axis=1)  # Baseline | Stage1
        row1 = np.concatenate([cells[2], cells[3]], axis=1)  # Stage2   | Stage3
        grid = np.concatenate([row0, row1], axis=0)
        writer.write(grid)

        if (fidx + 1) % 200 == 0 or fidx == n_frames - 1:
            print(f"  {fidx+1}/{n_frames}")

    writer.release()
    for cap, _, _ in caps:
        cap.release()

    sz = out_path.stat().st_size / 1e6
    print(f"[comp] 완료 → {out_path}  ({sz:.1f}MB)")


if __name__ == "__main__":
    main()
