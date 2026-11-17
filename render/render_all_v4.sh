#!/bin/bash
PY=/workspace/envs/cap-deploy/bin/python
export PYOPENGL_PLATFORM=
OUT=/workspace/capstone_deploy/v4_generalization
mkdir -p $OUT
# 파이프라인 배치 완료 대기
echo "[render] 배치 완료 대기..."
while ! grep -q "배치 완료" /tmp/v4_batch.log 2>/dev/null; do sleep 15; done
echo "[render] 시작 $(date +%T)"
for c in front_1 front_2 front_3 front_4 side_1 side_2 side_3 side_4; do
  clip=outputs/cap_pipeline/$c
  fin=$clip/hamer_wrist_hamer_rom_col_grip.pt
  if [ ! -f "$fin" ]; then echo "[render] $c 산출물 없음, 스킵"; continue; fi
  F=$($PY -c "import torch;print(torch.load('$clip/fit2d.pt',map_location='cpu')['smpl_params_incam']['body_pose'].shape[0])")
  echo "[render] $c (F=$F) v4단일..."
  $PY render/batch_render.py $clip hamer_wrist_hamer_rom_col_grip.pt $OUT/${c}_v4.mp4 $clip/0_input_video.mp4 > $clip/_v4_render.log 2>&1
  echo "[render] $c v3-vs-v4 나란히..."
  $PY render/render_ab_range.py $clip 0 $((F-1)) fit2d.pt hamer_wrist_hamer_rom_col_grip.pt $OUT/${c}_v3_vs_v4.mp4 "v3 fit2d" "v4" >> $clip/_v4_render.log 2>&1
  echo "[render] $c 완료 $(date +%T)"
done
echo "[render] 전체 완료 $(date +%T)"
ls -la $OUT/*.mp4
