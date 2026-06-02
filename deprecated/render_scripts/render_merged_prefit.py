"""
render_merged_prefit.py

smplx_merged_prefit.pt (Stage 1+2+3 결과)를 incam 영상으로 렌더링합니다.
기존 render_incam 로직을 standalone으로 구현.

출력:
  outputs/cap_pipeline/input/render_merged_prefit_incam.mp4     ← 최종 결과
  outputs/cap_pipeline/input/render_baseline_incam.mp4          ← 비교용 GVHMR baseline

사용법:
  python render_merged_prefit.py
  python render_merged_prefit.py --compare          # baseline도 함께 렌더링
  python render_merged_prefit.py --no-compare       # 최종 결과만 렌더링
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))
os.chdir(str(PROJ_ROOT))


def filter_smplx_params(params: dict) -> dict:
    allowed = {"betas", "body_pose", "global_orient", "transl",
               "left_hand_pose", "right_hand_pose",
               "jaw_pose", "leye_pose", "reye_pose", "expression"}
    return {k: v for k, v in params.items() if k in allowed}


def render_pt_to_video(
    pt_path: Path,
    video_in: Path,
    video_out: Path,
    *,
    mesh_color: tuple = (0.8, 0.8, 0.8),
    crf: int = 23,
):
    """SMPL-X incam render: pt_path의 smpl_params_incam → video_out."""
    from hmr4d.utils.video_io_utils import get_video_fps, get_video_lwh, get_video_reader, get_writer
    from hmr4d.utils.smplx_utils import make_smplx
    from hmr4d.utils.net_utils import to_cuda
    from hmr4d.utils.vis.renderer import Renderer

    print(f"[render] Loading {pt_path.name}...")
    pred      = torch.load(pt_path, map_location="cpu")
    smplx_m   = make_smplx("supermotion_fullhands").cuda()
    faces     = smplx_m.faces

    F = pred["smpl_params_incam"]["body_pose"].shape[0]
    print(f"[render] Running SMPL-X forward for {F} frames...")
    with torch.no_grad():
        smplx_out = smplx_m(**to_cuda(filter_smplx_params(pred["smpl_params_incam"])))
    verts_incam = smplx_out.vertices   # (F, V, 3)

    fps             = get_video_fps(str(video_in))
    length, W, H    = get_video_lwh(str(video_in))
    K               = pred["K_fullimg"][0]   # (3, 3) use first frame K

    print(f"[render] Video: {W}×{H}  {fps:.2f}fps  {length} frames")
    renderer = Renderer(W, H, device="cuda", faces=faces, K=K, bin_size=0)
    reader   = get_video_reader(str(video_in))

    video_out.parent.mkdir(parents=True, exist_ok=True)
    writer = get_writer(str(video_out), fps=fps, crf=crf)

    for i, img_raw in tqdm(enumerate(reader), total=length,
                            desc=f"→ {video_out.name}"):
        if i >= F:
            break
        img = renderer.render_mesh(verts_incam[i].cuda(), img_raw, list(mesh_color))
        writer.write_frame(img)

    writer.close()
    print(f"[render] Saved → {video_out}")
    return video_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/cap_pipeline/input")
    parser.add_argument("--compare",    dest="compare", action="store_true",  default=True,
                        help="Also render baseline for comparison (default True)")
    parser.add_argument("--no-compare", dest="compare", action="store_false")
    parser.add_argument("--crf", type=int, default=23,
                        help="Video CRF quality (17=lossless, 23=default, 28=smaller)")
    args = parser.parse_args()

    d         = Path(args.input_dir)
    video_in  = d / "0_input_video.mp4"

    # ── 1. 최종 Stage 1+2+3 결과 ──────────────────────────────────────────────
    render_pt_to_video(
        d / "smplx_merged_prefit.pt",
        video_in,
        d / "render_merged_prefit_incam.mp4",
        mesh_color=(0.7, 0.85, 1.0),   # 파란빛 — 새 결과 구분
        crf=args.crf,
    )

    # ── 2. 비교용 GVHMR baseline ────────────────────────────────────────────────
    if args.compare:
        render_pt_to_video(
            d / "hmr4d_results.pt",
            video_in,
            d / "render_baseline_incam.mp4",
            mesh_color=(0.8, 0.8, 0.8),   # 기본 회색
            crf=args.crf,
        )

    print("\n[render] 완료.")
    print(f"  최종: {d}/render_merged_prefit_incam.mp4")
    if args.compare:
        print(f"  비교: {d}/render_baseline_incam.mp4")


if __name__ == "__main__":
    main()
