#!/bin/bash
PY=/workspace/envs/cap-deploy/bin/python
export PYOPENGL_PLATFORM=
OUT=/workspace/capstone_deploy/v4_generalization
echo "[hq] 수정 재실행 완료 대기..."
while ! grep -q "수정 재실행 완료" /tmp/v4_fix.log 2>/dev/null; do sleep 15; done
echo "[hq] 렌더 시작 $(date +%T)"
for c in front_1 front_2 front_3 front_4 side_1 side_2 side_3 side_4; do
  clip=outputs/cap_pipeline/$c
  fin=$clip/hamer_wrist_hamer_rom_col_grip.pt; old=$clip/grip_preArmFix.pt
  [ ! -f "$fin" ] && continue
  F=$($PY -c "import torch;print(torch.load('$clip/fit2d.pt',map_location='cpu')['smpl_params_incam']['body_pose'].shape[0])")
  if [ -f "$old" ]; then
    echo "[hq] $c before/after (수정전 떨림 vs 수정후) F=$F"
    $PY render_ab_hq.py $clip 0 $((F-1)) grip_preArmFix.pt hamer_wrist_hamer_rom_col_grip.pt $OUT/${c}_ARMFIX_before_after.mp4 "before(jitter)" "after(fixed)" > $clip/_hq.log 2>&1
  fi
  echo "[hq] $c 단일 고화질"
  $PY batch_render_hq.py $clip hamer_wrist_hamer_rom_col_grip.pt $OUT/${c}_v4_fixed_HQ.mp4 $clip/0_input_video.mp4 >> $clip/_hq.log 2>&1
  echo "[hq] $c 완료 $(date +%T)"
done
echo "[hq] 전체 완료 $(date +%T)"; ls -la $OUT/*ARMFIX* $OUT/*_HQ.mp4 2>/dev/null
