#!/bin/bash
for c in front_1 front_3 front_4 side_1 side_2 side_3 side_4; do
  clip=outputs/cap_pipeline/$c
  # 옛(떨림) grip 보존
  [ -f $clip/hamer_wrist_hamer_rom_col_grip.pt ] && cp $clip/hamer_wrist_hamer_rom_col_grip.pt $clip/grip_preArmFix.pt
  echo "########## $c 재실행 $(date +%T) ##########"
  bash run_v4.sh $clip
done
echo "########## 수정 재실행 완료 $(date +%T) ##########"
