#!/bin/bash
PY=/workspace/envs/cap-deploy/bin/python
export PYOPENGL_PLATFORM=
OUT=/workspace/capstone_deploy/v4_generalization
mkdir -p $OUT
cd /workspace/capstone_deploy
for c in front_1 front_2 front_3 front_4 side_1 side_2 side_3 side_4; do
  clip=outputs/cap_pipeline/$c
  fin=$clip/hamer_wrist_hamer_rom_col_grip.pt
  if [ ! -f "$fin" ]; then echo "[skip] $c (산출물 없음)"; continue; fi
  F=$($PY -c "import torch;print(torch.load('$clip/fit2d.pt',map_location='cpu')['smpl_params_incam']['body_pose'].shape[0])")
  echo "########## $c (F=$F) v3 vs v4 HQ $(date +%T) ##########"
  $PY render_ab_hq.py $clip 0 $((F-1)) fit2d.pt hamer_wrist_hamer_rom_col_grip.pt $OUT/${c}_v3_vs_v4_HQ.mp4 "v3 fit2d" "v4 ours" > $clip/_v3v4.log 2>&1
  echo "[done] $c $(date +%T)"
done
echo "########## v3vs4 전체 완료 $(date +%T) ##########"
ls -la $OUT/*v3_vs_v4_HQ.mp4 2>/dev/null
