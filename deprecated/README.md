# deprecated/ — 격리 (현재 파이프라인 미사용)

완전 삭제 대신 격리. 필요 시 루트로 복원 가능. 근거: docs/PIPELINE_MAP.md

## render_scripts/ — 휴리스틱기 시각화 스크립트 (DEAD)
import 체인에 없음. 옛 비교 영상 생성용.
- render_comparison.py, render_comparison_4panel.py : 2×2 / 4패널 비교 영상
- render_merged_prefit.py : smplx_merged_prefit incam 렌더 (현재 render_hmp_compare가 자체 함수로 대체)
- render_hamer_incam.py : HaMeR incam 렌더
- render_vitpose_skeleton.py : ViTPose 스켈레톤 오버레이

현재 렌더는 render_hmp_compare.py / render_vitpose_wb.py / vis_hmp_*.py 사용.
