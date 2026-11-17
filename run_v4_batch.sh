#!/bin/bash
for c in front_2 front_4 side_1 side_2 side_3 side_4; do
  echo "########## $c 시작 $(date +%T) ##########"
  bash run_v4.sh outputs/cap_pipeline/$c
done
echo "########## 배치 완료 $(date +%T) ##########"
