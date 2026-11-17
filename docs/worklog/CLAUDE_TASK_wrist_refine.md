# Claude Task: Wrist Refine Implementation

This document is the execution request for Claude Code CLI.
The design source of truth is `docs/HANDOFF_for_claude_code.md`.
Do not treat this file as a replacement for the handoff document.

현재 작업 디렉터리는 /workspace/capstone_deploy 입니다.
이 repo root 기준으로 파일을 읽고 수정해주세요.

먼저 handoff 문서를 읽고, 이 문서를 source of truth로 취급해주세요.

문서 경로:
docs/HANDOFF_for_claude_code.md

함께 확인할 파일:
- cap_wrist_refine.py
- entry_point_hamer.py
- entry_point.py는 참고용으로만 확인

앞으로 모든 설명, 진행 상황, 결과 보고는 한국어로 작성해주세요.

코드 변경 전에 먼저 백업 파일을 생성해주세요.

백업 파일:
- entry_point_hamer.py → entry_point_hamer.py.bak_before_wrist_refine
- entry_point.py → entry_point.py.bak_reference_only
- cap_wrist_refine.py → cap_wrist_refine.py.bak_before_wrist_refine

구현에 들어가기 전에, 백업 생성 완료 여부와 함께
‘수정할 파일 / 수정할 함수 / 실행 예정 검증’에 대한 짧은 작업 계획을 한국어로 먼저 5~10줄 내로 보여준 뒤 바로 구현을 진행하세요.

중요 제약사항:
- entry_point.py는 절대 수정하지 마세요.
- 실제 파이프라인 연결은 entry_point_hamer.py에만 진행하세요.
- cap_wrist_refine.py는 repo root에서 runtime module로 사용하세요.
- docs 또는 DOCS 폴더 안의 Python 파일을 runtime code로 import하지 마세요.
- smplx_merged_hamer_post.pt를 덮어쓰지 마세요.
- baseline / wrist_only / wrist_refined 결과를 각각 분리해서 저장하세요.
- overlay gate를 건너뛰지 마세요.
- DeVI 논문을 다시 해석하거나 설계를 바꾸지 마세요. 설계는 handoff 문서에 이미 확정되어 있습니다.
- HaMeR build/environment가 바뀌지 않았다면 HaMeR out.keys() 디버그 확인을 반복하지 마세요.
- 현재 RunPod build에서는 이미 아래가 확인되었습니다:
  - pred_keypoints_3d: torch.Size([1, 21, 3])
  - pred_keypoints_2d: torch.Size([1, 21, 2])

구현 작업:
1. entry_point_hamer.py의 HaMeR npz 저장 단계에 아래를 추가하세요.
   - keypoints_3d=out["pred_keypoints_3d"][n].detach().cpu().numpy()
   - keypoints_2d=out["pred_keypoints_2d"][n].detach().cpu().numpy()

2. cap_wrist_refine.py를 수정하세요.
   - load_hamer_detections_full()에서 keypoints_3d와 keypoints_2d를 읽도록 수정
   - build_hand_targets_option1_hamer() 구현
   - dummy target을 반환하거나 NotImplementedError를 우회하지 말 것
   - target uv는 debug_target_overlay로 반드시 확인 가능해야 함

3. entry_point_hamer.py에 run_wrist_refine()을 연결하세요.
   - merge_hamer_hands_into_gvhmr_postprocessed()가 smplx_merged_hamer_post.pt를 만든 이후에 실행
   - Stage A only 결과:
     smplx_merged_hamer_wrist_only.pt
   - Stage A + Option 1 refine 결과:
     smplx_merged_hamer_wrist_refined.pt
   - baseline smplx_merged_hamer_post.pt는 절대 덮어쓰지 말 것

4. 적용 전/후 비교 결과를 생성하세요.
   - baseline 영상
   - wrist_only 영상
   - wrist_refined 영상
   각각 다른 mp4 파일명으로 저장하세요.

5. 구현 후 아래 검사를 실행하세요.
   - python -m py_compile entry_point_hamer.py cap_wrist_refine.py
   - python - <<'PY'
   import cap_wrist_refine
   print("cap_wrist_refine import OK")
   PY
6. 기존 HaMeR npz에 keypoints_3d/keypoints_2d가 없으면 HaMeR npz만 삭제하고 재생성하세요.
   - --force는 꼭 필요한 경우가 아니면 사용하지 마세요.
   - 기존 GVHMR 결과는 가능하면 재사용하세요.

7. overlay 결과를 생성하세요.
   - before
   - after_wrist
   - target_uv
   - after_refine

8. 마지막에 한국어로 아래를 정리해서 보고해주세요.
   - 생성한 백업 파일
   - 수정한 파일
   - 핵심 수정 내용
   - syntax/import check 결과
   - 생성된 .pt 파일 경로
   - 생성된 mp4 파일 경로
   - 생성된 overlay 디렉터리
   - 실패한 gate 또는 warning
   - 추가로 사용자가 직접 확인해야 할 부분

자동으로 디버깅하고 고칠 수 있는 단순 코드 오류는 직접 수정하세요.
필요하다면 기존 repo 안의 overlay/render/debug 관련 코드를 참고하거나 재사용 가능한 유틸을 활용해도 됩니다.
단, entry_point.py는 reference only이며 수정하지 마세요.
실제 overlay gate는 handoff 문서와 cap_wrist_refine.py의 debug_overlay / debug_target_overlay 흐름을 우선 기준으로 유지하세요.
overlay 결과는 before / after_wrist / target_uv / after_refine 단계로 분리해서 저장하세요.
기존 코드를 참고하더라도 handoff 문서의 overlay gate 순서를 바꾸거나 생략하지 마세요.

단, 아래 상황에서는 작업을 멈추고 사용자에게 확인을 요청하세요.
- entry_point.py 수정이 필요해지는 경우
- SMPL-X kintree가 문서의 가정과 다르게 나오는 경우
- Option 1 target overlay가 손 위에 찍히지 않고 계속 크게 어긋나는 경우
- 좌표계/손 joint mapping에 대해 확정할 수 없는 설계 판단이 필요한 경우
- dummy target이나 임시 우회 없이는 진행할 수 없는 경우