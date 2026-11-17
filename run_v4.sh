#!/bin/bash
# v4 전체 체인 + 단계별 타이밍. usage: bash run_v4.sh <clip_dir>
clip=${1%/}
PY=/workspace/envs/cap-deploy/bin/python
export PYOPENGL_PLATFORM=
export CAP_ARM_DEPTH=100000
log=$clip/v4_timing.txt
: > $log
F=$($PY -c "import torch;print(torch.load('$clip/fit2d.pt',map_location='cpu')['smpl_params_incam']['body_pose'].shape[0])" 2>/dev/null)
echo "clip=$(basename $clip) frames=$F" | tee -a $log
S=$(date +%s%N)
step(){ local name="$1"; shift; local a=$(date +%s%N); "$@" > $clip/_v4_$name.log 2>&1; local rc=$?; local b=$(date +%s%N); awk -v n="$name" -v a=$a -v b=$b -v r=$rc 'BEGIN{printf "%-12s %6.1fs  (rc=%d)\n",n,(b-a)/1e9,r}' | tee -a $log; return $rc; }
step arm     $PY cap_arm_prefit.py --input-dir $clip --w-tc 5000 --target-px 3 --debug-frames 0 || exit 1
step armsm  $PY batch_arm_smooth.py $clip 1.0 3.0 || exit 1
step hamerw  $PY batch_hamer_wrist.py $clip hamer 1.8 || exit 1
step rom     $PY batch_rom_clamp.py $clip hamer_wrist_hamer.pt || exit 1
step coldet  $PY collision_detect.py $clip hamer_wrist_hamer_rom.pt || exit 1
step axcol   $PY batch_axcol_v4.py $clip 250 hamer_wrist_hamer_rom.pt || exit 1
step grip    $PY batch_grip_prior.py $clip hamer_wrist_hamer_rom_col.pt 0.7 1.5 0.1 || exit 1
E=$(date +%s%N)
awk -v s=$S -v e=$E -v f="$F" 'BEGIN{printf "TOTAL(opt)   %6.1fs  for %s frames  (%.3fs/frame)\n",(e-s)/1e9,f,(e-s)/1e9/f}' | tee -a $log
