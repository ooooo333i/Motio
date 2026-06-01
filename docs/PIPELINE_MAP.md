# PIPELINE_MAP.md — 실사용 파이프라인 분석 (2026-06-01)

> STEP 1 산출물. 코드 미수정, 읽기 분석만. 리팩토링 판단의 근거.

## ⚠️ 핵심 발견: 병렬 파이프라인 2개 (서로 다른 결과)

이 repo에는 **서로 다른 결과를 내는 두 파이프라인**이 공존한다:

- **P1 = `entry_point_hamer.main()`** ("공식" 디버그 엔트리) — 옛 **DeVI식 Stage A/B 손목 refine**
- **P2 = 수동 실험 체인** (이번 세션 작업) — **ViTPose 팔 + HMP 손목 + 손가락 후처리** → 현재 최선 결과 B

**`entry_point_hamer.py`를 실행하면 P1(옛 방식)이 돌아가고, 현재 최선 결과(B)는 P2(수동)로만 나온다.** 둘이 끊겨 있음 = 정리의 핵심 쟁점.

---

## P1: entry_point_hamer.main() 실제 호출 체인

```
cap.run_gvhmr()                         → hmr4d_results.pt, preprocess/{vitpose.pt, bbx.pt, wholebody_kp2d.npy}
 → cap.extract_video_frames()           → hamer_frames/
 → run_hamer_from_gvhmr_keypoints_debug() (wholebody133 bbox) → hamer_out_clean/*.npz
 → merge_hamer_hands_into_gvhmr_postprocessed()  → smplx_merged_hamer_post.pt   [손가락만 병합+후처리, 손목=GVHMR]
 → cap.render_merged_result_video()
 → cap_wrist_refine.run_wrist_refine(do_refine=False)  → smplx_merged_hamer_wrist_only.pt   [Stage A]
 → cap_wrist_refine.run_wrist_refine(do_refine=True, option1) → smplx_merged_hamer_wrist_refined.pt  [Stage A+B]
 → tools.demo 로 3 variant 렌더
```

## P2: 수동 실험 체인 (현재 최선 결과 경로)

```
cap_arm_prefit.py    : hmr4d_results.pt + preprocess/vitpose.pt → smplx_arm_prefit.pt   [ViTPose 팔 fit, arm RMS 3.27px]
cap_mano_prefit.py   : (+smplx_stage1a.pt optional) + hamer_out_clean → hamer_mano_prefit.pt
cap_merge_opt.py     : smplx_arm_prefit.pt + hamer_mano_prefit.pt + vitpose.pt → smplx_merged_prefit.pt
cap_kalman_wrist.py  : smplx_merged_prefit.pt → smplx_kalman_B.pt   [baseline 비교용]
HMP (외부 /workspace/hmp_test) : run_hamer init + fitting_app.py → recon_000_30fps.npz (tennis_full / tennis_full_left)
   └ cap_wrist_refine.load_hamer_detections_full 사용
cap_inject_hmp.py    : smplx_arm_prefit.pt(B) + HMP npz + HaMeR 손가락(후처리)
                       → smplx_B_hmpwrist_ppfinger.pt   [★ 현재 최선 = ViTPose팔 + HMP손목 + 후처리손가락]
                       (A 버전: GVHMR 손목 유지 = smplx_A_gvhmrwrist_ppfinger.pt)
```

손가락 후처리 = `entry_point_hamer.postprocess_hand_sequence` (temporal gate + outlier + fill + gaussian smooth) 를 cap_inject_hmp가 import.

---

## 파일/함수 분류

### LIVE — 현재 최선 결과(B)에 실제 기여
| 파일/함수 | 역할 |
|---|---|
| `entry_point.py` (cap) | GVHMR 실행, HaMeR 실행, load_hamer_hand_detections 등 기반 유틸 |
| `entry_point_hamer.py`: run_hamer_from_gvhmr_keypoints_debug, merge_*, postprocess_hand_sequence + HandPostprocessConfig + smooth_sequence/detect_*/fill_missing | GVHMR→HaMeR, 손가락 병합·후처리 |
| `cap_arm_prefit.py` | ViTPose 팔 fit → smplx_arm_prefit.pt (B의 base) |
| `cap_wrist_refine.py`: **load_hamer_detections_full** | HaMeR npz 로더 (inject + HMP run_hamer가 사용) |
| `cap_inject_hmp.py` | HMP 손목 + 후처리 손가락 주입 → B |
| `physical_limits.py` | 검증 기준 |
| HMP 외부 (`/workspace/hmp_test`) | wrist motion prior |

### UNSURE — 확인 필요 (임의 판단 안 함)
| 항목 | 쟁점 |
|---|---|
| `entry_point_hamer.main()`의 run_wrist_refine ×2 (Stage A/B) | 공식 엔트리가 실행하지만 HMP로 대체됨. main을 P2(HMP)로 재배선할지? |
| `cap_wrist_refine`: run_wrist_refine, replace_wrists, refine_2d, build_hand_targets_option1, verify_wrist_chains | P1 Stage A/B 전용. P1 버리면 DEAD. load_hamer_detections_full만 LIVE |
| `cap_mano_prefit.py`, `cap_merge_opt.py`, `cap_kalman_wrist.py` | smplx_kalman_B(baseline 비교) 생성. 최선 B는 안 씀. baseline으로 유지? |
| `smplx_merged_prefit.pt`, `smplx_kalman_B.pt` | baseline 산출물 |

### DEAD — 버려진 시도 (PIPELINE_STATUS 근거)
| 항목 | 버린 이유 |
|---|---|
| `cap_wrist_refine._slerp_fill_invalid` (SLERP 보간) | 잘못된 프레임을 이웃으로 smoothing, use_slerp=False 기본 |
| `cap_wrist_refine.build_hand_targets_option2_detector` | option2 detector 미사용 (option1만 씀) |
| cap_merge_opt 내 **B1/B2 refine** | B1=관통 13%, B2=효과없음 → 비활성 (merge_refine_meta: disabled) |
| cap_mano_prefit 내 **Z-anchor / conf-gating** (smplx_stage1a 경로) | 효과 미미로 제거됨 |
| 옛 렌더 스크립트: render_comparison.py, render_comparison_4panel.py, render_merged_prefit.py, render_hamer_incam.py, render_vitpose_skeleton.py | 휴리스틱기 시각화, 현재 미사용 |
| 이미 archive됨: check_wrist_orientation, projection_check, cap_smooth_wrist | (이전 정리) |

> 주의: B1/B2, Z-anchor, SLERP은 **별도 파일이 아니라 LIVE 스크립트 내부의 죽은 코드 블록**.
> 파일 이동이 아니라 함수 내 편집이 필요 → 회귀 위험. 현재는 비활성 플래그로 안 돌므로 **격리보다 그대로 두는 게 안전**.

---

## 절대 삭제 금지
- .pt: hmr4d_results, smplx_arm_prefit, smplx_kalman_B, smplx_B_hmpwrist_ppfinger, smplx_A_gvhmrwrist_ppfinger, smplx_hmp_injected_full, hamer_mano_prefit, smplx_merged_prefit
- .py: cap_inject_hmp, physical_limits, cap_wrist_refine, cap_arm_prefit, entry_point, entry_point_hamer
- 외부: /workspace/hmp_test 전체
- 백업: outputs/cap_pipeline/backup_pre_refactor/
