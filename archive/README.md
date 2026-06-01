# archive/ — 보관 (현재 파이프라인 미사용)

import 체인에 묶이지 않은 일회성/검증용 자료. 필요 시 루트로 복원 가능.

## scripts/ — 일회성 디버그 스크립트
- check_wrist_orientation.py : 손목 방향 디버그 (구 휴리스틱기)
- projection_check.py        : incam 투영 점검
- cap_smooth_wrist.py        : wrist median smoother (PIPELINE_STATUS상 비활성, smplx_final_smoothed 생성용)

## vis/ — HMP 검증 이미지 (jpg, 재생성 가능)
- vis_hmp_left/right         : 좌/우 HMP projection 시각 확인
- vis_compare_hmp(2)         : KB vs HMP swing 비교
- vis_lambda30               : lambda_orient=30 비교
- vis_wrist_hmp_grip_hamer   : wrist=HMP + grip=HaMeR 비교
- vis_vitpose                : ViTPose WB 키포인트 오버레이
- vis_full                   : 전체 1495 렌더 검증 프레임

생성 코드는 루트의 vis_hmp_*.py, render_*.py 로 유지됨 (도구는 보존).
