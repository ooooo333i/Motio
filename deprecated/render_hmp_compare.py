"""
render_hmp_compare.py

smplx_hmp_injected.pt vs smplx_kalman_B.pt 비교 렌더링.

1. 두 파일 전체 incam 영상 렌더링
2. swing 구간(f320-430) 클립 + 나란히 2열 비교 영상 생성

출력:
  outputs/cap_pipeline/input/render_kalmanB_incam.mp4
  outputs/cap_pipeline/input/render_hmp_injected_incam.mp4
  outputs/cap_pipeline/input/render_hmp_compare_swing.mp4   ← 핵심 비교
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))
os.chdir(str(PROJ_ROOT))

SWING_START = 320
SWING_END   = 430


def filter_smplx_params(params: dict) -> dict:
    allowed = {"betas", "body_pose", "global_orient", "transl",
               "left_hand_pose", "right_hand_pose",
               "jaw_pose", "leye_pose", "reye_pose", "expression"}
    return {k: v for k, v in params.items() if k in allowed}


def render_pt_to_video(pt_path: Path, video_in: Path, video_out: Path,
                       *, mesh_color=(0.8, 0.8, 0.8), crf: int = 23,
                       frame_start: int = 0, frame_end: int = None):
    from hmr4d.utils.video_io_utils import get_video_fps, get_video_lwh, get_video_reader, get_writer
    from hmr4d.utils.smplx_utils import make_smplx
    from hmr4d.utils.net_utils import to_cuda
    from hmr4d.utils.vis.renderer import Renderer

    print(f"[render] {pt_path.name} → {video_out.name}")
    pred    = torch.load(pt_path, map_location="cpu")
    smplx_m = make_smplx("supermotion_fullhands").cuda()
    faces   = smplx_m.faces

    F = pred["smpl_params_incam"]["body_pose"].shape[0]
    if frame_end is None:
        frame_end = F

    print(f"  SMPL-X forward {F} frames (rendering {frame_start}~{frame_end})...")
    with torch.no_grad():
        smplx_out = smplx_m(**to_cuda(filter_smplx_params(pred["smpl_params_incam"])))
    verts_incam = smplx_out.vertices   # (F, V, 3)

    fps         = get_video_fps(str(video_in))
    _, W, H     = get_video_lwh(str(video_in))
    K           = pred["K_fullimg"][0]

    renderer = Renderer(W, H, device="cuda", faces=faces, K=K, bin_size=0)
    reader   = get_video_reader(str(video_in))

    video_out.parent.mkdir(parents=True, exist_ok=True)
    writer = get_writer(str(video_out), fps=fps, crf=crf)

    for i, img_raw in tqdm(enumerate(reader), total=frame_end, desc=video_out.name):
        if i < frame_start:
            continue
        if i >= frame_end or i >= F:
            break
        img = renderer.render_mesh(verts_incam[i].cuda(), img_raw, list(mesh_color))
        writer.write_frame(img)

    writer.close()
    print(f"  Saved → {video_out}")
    return video_out


def make_side_by_side(left_mp4: Path, right_mp4: Path, out_mp4: Path,
                      label_l: str, label_r: str,
                      scale: float = 0.5, crf: int = 23):
    """두 영상을 나란히(left|right) 합쳐 비교 영상 생성."""
    cap_l = cv2.VideoCapture(str(left_mp4))
    cap_r = cv2.VideoCapture(str(right_mp4))
    fps   = cap_l.get(cv2.CAP_PROP_FPS)
    W0    = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0    = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W     = int(W0 * scale)
    H     = int(H0 * scale)
    out_w = W * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (out_w, H))

    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    fs    = max(0.5, H / 640)

    print(f"[side-by-side] {left_mp4.name} | {right_mp4.name} → {out_mp4.name}")
    nf = int(min(cap_l.get(cv2.CAP_PROP_FRAME_COUNT),
                 cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))

    for i in tqdm(range(nf), desc="side-by-side"):
        ok_l, fl = cap_l.read()
        ok_r, fr = cap_r.read()
        if not ok_l or not ok_r:
            break
        fl = cv2.resize(fl, (W, H))
        fr = cv2.resize(fr, (W, H))
        frame = np.concatenate([fl, fr], axis=1)
        # 레이블
        cv2.putText(frame, label_l, (10, 30), FONT, fs, (255, 255, 100), 2)
        cv2.putText(frame, label_r, (W + 10, 30), FONT, fs, (100, 255, 150), 2)
        # 프레임 번호
        vid_f = SWING_START + i
        cv2.putText(frame, f"f{vid_f}", (out_w - 80, H - 10), FONT, fs * 0.8, (200, 200, 200), 1)
        writer.write(frame)

    cap_l.release(); cap_r.release(); writer.release()
    # mp4v → h264 재인코딩
    tmp = out_mp4.with_suffix(".tmp.mp4")
    out_mp4.rename(tmp)
    os.system(f"ffmpeg -y -i {tmp} -vcodec libx264 -crf {crf} -pix_fmt yuv420p {out_mp4} -loglevel error")
    tmp.unlink(missing_ok=True)
    print(f"  Saved → {out_mp4}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/cap_pipeline/input")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--swing-only", action="store_true",
                        help="swing 구간(f320-430)만 렌더링 (빠름)")
    args = parser.parse_args()

    d        = Path(args.input_dir)
    video_in = d / "0_input_video.mp4"

    f_start = SWING_START if args.swing_only else 0
    f_end   = SWING_END + 1 if args.swing_only else None

    # ── 1. Kalman-B baseline ─────────────────────────────────────────
    out_kb = d / "render_kalmanB_incam.mp4"
    if not out_kb.exists():
        render_pt_to_video(d / "smplx_kalman_B.pt", video_in, out_kb,
                           mesh_color=(0.75, 0.75, 0.85),
                           crf=args.crf, frame_start=f_start, frame_end=f_end)
    else:
        print(f"[skip] {out_kb.name} 이미 존재")

    # ── 2. HMP 주입 결과 ──────────────────────────────────────────────
    out_hmp = d / "render_hmp_injected_incam.mp4"
    render_pt_to_video(d / "smplx_hmp_injected.pt", video_in, out_hmp,
                       mesh_color=(0.6, 0.9, 0.7),   # 초록빛
                       crf=args.crf, frame_start=f_start, frame_end=f_end)

    # ── 3. 나란히 비교 ────────────────────────────────────────────────
    out_cmp = d / "render_hmp_compare_swing.mp4"
    make_side_by_side(out_kb, out_hmp, out_cmp,
                      label_l="Kalman-B (before)",
                      label_r="HMP injected (after)",
                      scale=0.5, crf=args.crf)

    print("\n완료.")
    print(f"  baseline : {out_kb}")
    print(f"  hmp      : {out_hmp}")
    print(f"  비교영상  : {out_cmp}")


if __name__ == "__main__":
    main()
