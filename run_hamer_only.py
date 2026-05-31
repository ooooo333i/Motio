#!/usr/bin/env python3
"""Run only the HaMeR inference step, skipping GVHMR entirely."""
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))
import entry_point as cap

FRAME_DIR   = PROJ_ROOT / "outputs/cap_pipeline/input/hamer_frames"
VITPOSE_PT  = PROJ_ROOT / "outputs/cap_pipeline/input/preprocess/vitpose.pt"
OUT_DIR     = PROJ_ROOT / "outputs/cap_pipeline/input/hamer_out_clean"
HAMER_ROOT  = PROJ_ROOT / "external/hamer"

RESCALE_FACTOR   = 2.0
MIN_CONF         = 0.3
BATCH_SIZE       = 16
FORCE            = True
HAND_BBOX_SOURCE = "wholebody133"
BBX_PT           = PROJ_ROOT / "outputs/cap_pipeline/input/preprocess/bbx.pt"

if __name__ == "__main__":
    import time
    import entry_point_hamer as cap_h
    tic = time.perf_counter()
    cap_h.run_hamer_from_gvhmr_keypoints_debug(
        frame_dir=FRAME_DIR,
        vitpose_path=VITPOSE_PT,
        out_dir=OUT_DIR,
        hamer_root=HAMER_ROOT,
        checkpoint=None,
        batch_size=BATCH_SIZE,
        rescale_factor=RESCALE_FACTOR,
        min_conf=MIN_CONF,
        force=FORCE,
        verbose=False,
        save_crops=False,
        hand_bbox_source=HAND_BBOX_SOURCE,
        bbx_path=BBX_PT,
    )
    elapsed = time.perf_counter() - tic
    npz_count = len(list(OUT_DIR.glob('*.npz')))
    print(f"\n[DONE] HaMeR outputs saved to: {OUT_DIR}")
    print(f"  npz count: {npz_count}")
    print(f"  total time: {elapsed:.1f}s")
