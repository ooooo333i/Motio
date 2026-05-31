# Pipeline Status — Heuristic Phase Complete

**Date**: 2026-05-31  
**Branch**: runpod-wrist-refine-debug  
**Status**: Heuristic pipeline complete. Transitioning to HMP (motion prior) for wrist correction.

---

## 1. 파이프라인 구조

```
입력: 테니스 단일 비디오 (1495 frames, 60fps, 1080×1920)
      GVHMR(body), HaMeR(hand MANO), ViTPose COCO17(body 2D), WB133(wholebody 2D)

Stage 1: cap_arm_prefit.py
  입력: hmr4d_results.pt (GVHMR baseline)
  출력: smplx_arm_prefit.pt
  내용: GVHMR body pose → ViTPose arm keypoint fitting (shoulder/elbow/wrist)
  성능: arm RMS ~3.3px (target_px 달성)
  메타키: cap_arm_prefit_meta

Stage 2: cap_mano_prefit.py
  입력: smplx_arm_prefit.pt + hamer_out_clean/ (HaMeR npz)
  출력: hamer_mano_prefit.pt
  내용: HaMeR MANO 결과를 SMPL-X incam 좌표계로 rigid 보정
  제거됨: Z-anchor, conf-gating (효과 미미)

Stage 3: cap_merge_opt.py
  입력: smplx_arm_prefit.pt (body) + hamer_mano_prefit.pt (hand) + hamer_out_clean/
  출력: smplx_merged_prefit.pt
  내용:
    - Stage A: replace_wrists() — HaMeR wrist_R → SMPL-X local wrist rotation
      (G_e^{-1} @ R_hamer. 검증: FK wrist = HaMeR wrist_R, geodesic ≈ 0° 전 프레임)
    - Stage A+: hand_pose 직접 주입 (HaMeR mano_params → axis-angle, - hand_mean)
    - B1/B2: 의도적으로 비활성화 (아래 이유 참고)
  메타키: cap_merge_meta

후처리: cap_kalman_wrist.py (Kalman-B)
  입력: smplx_merged_prefit.pt
  출력: smplx_kalman_B.pt  ← 현재 최종 산출물
  내용: wrist rotation에 constant-velocity Kalman filter
    Config B: q_noise=0.0005, r_noise=0.2, gate_deg=33°, gate_factor=50×
  + ROM 160° clip (SLERP)
```

---

## 2. 각 .pt 파일 목록

| 파일 | 단계 | 설명 | 보존 |
|---|---|---|---|
| `hmr4d_results.pt` | 입력 | GVHMR baseline | 필수 |
| `smplx_arm_prefit.pt` | Stage 1 | arm 최적화 완료 body | 필수 |
| `hamer_mano_prefit.pt` | Stage 2 | HaMeR hand prefit | 필수 |
| `smplx_merged_prefit.pt` | Stage 3 | wrist_R + hand_pose 주입 | 필수 |
| `smplx_kalman_B.pt` | 후처리 | **현재 최종 baseline** | 필수 |
| `smplx_kalman_A.pt` | 실험 | Kalman A (q=0.001, r=0.1) | 삭제 가능 |
| `smplx_kalman_C.pt` | 실험 | Kalman C (q=0.001, r=0.3) | 삭제 가능 |
| `smplx_final_smoothed.pt` | 실험 | median spike smoother | 삭제 가능 |
| `smplx_merged_stage4.pt` | 구버전 | 구 파이프라인 (B1 활성) | 구버전 |
| `smplx_merged_stage4_axfix.pt` | 구버전 실험 | 잘못된 base에 axial fix | 삭제 가능 |
| `smplx_merged_prefit_expa/b/c.pt` | 실험 | Stage 3 실험 variants | 삭제 가능 |
| `smplx_merged_prefit_exp_noanchor.pt` | 실험 | anchor 없는 variant | 삭제 가능 |
| `smplx_stage1a.pt` | 구버전 | 구 Stage 1 | 삭제 가능 |

---

## 3. 의도적으로 비활성화된 것들 (이유 포함)

### B1 arm refine (cap_merge_opt.py)
- **비활성화 이유**: B1이 arm을 ViTPose 2D에 억지로 맞추면서 관통 13.1% → 팔이 몸통을 뚫음
- B1 활성 시: arm RMS wrist 1.0px→8.5px 악화
- B1 비활성 시: 관통 3.4%로 감소
- `merge_refine_meta: {'b1': 'disabled', 'b2': 'disabled'}`

### B2 hand refine (cap_merge_opt.py)
- **비활성화 이유**: B2 자체가 효과 없음 (즉시 plateau), 오히려 노이즈 추가

### SLERP outlier 보간 (cap_wrist_refine.py)
- **제거 이유**: SLERP이 잘못된 HaMeR 프레임의 wrist rotation을 이웃으로 smoothing
  → 손목이 HaMeR를 안 따라가는 문제. `use_slerp=False`가 기본값

### Z-anchor / conf-gating (cap_mano_prefit.py)
- **제거 이유**: 효과 미미. palm p50 개선이 통계적으로 유의미하지 않음

---

## 4. 미해결 문제 및 HMP 전환 근거

### 진단 결론 (2026-05-31)
Stage A wrist 주입은 완벽하다 (FK wrist = HaMeR wrist_R, geodesic ≈ 0° 전 프레임).
문제는 HaMeR 신호 자체의 한계다.

#### 문제 A: 정상 구간(f200) 손 묻힘
- 현상: SMPL-X 렌더에서 손/손가락이 몸통 안에 들어가 보이지 않음
- 수치: f200 LEFT wrist axial = -52° (강한 supination)
  → 팔이 앞에 교차된 자세에서 손가락이 몸통 방향을 향함
- 2D 손목 위치 오차: SMPL-X (542, 898)px vs HaMeR cam_t (543, 956)px = **Y축 58px 오차**
- 원인: HaMeR가 단일 프레임 독립적으로 추정한 wrist axial이 부정확
  smplx_arm_prefit.pt(GVHMR arm 위치)와 HaMeR wrist_R의 조합이 관통 발생

#### 문제 B: 스윙 구간 axial flip
- 현상: 스윙 중 손이 180° 뒤집힌 것처럼 보임
- smplx_merged_prefit.pt에서: f365-391 axial -82°~-100° (Kalman 후 더 완만하게 퍼짐)
- Kalman이 flip을 smoothing하면서 앞뒤 프레임 오염 → velocity 기반 감지 불가
- **단순 flip 제거로는 "올바른 axial값"을 알 수 없음** — motion prior 필요

#### 왜 휴리스틱으로 해결 불가
| 시도 | 문제점 |
|---|---|
| Kalman smoothing | flip 신호를 부드럽게 만들지만 올바른 값으로 수렴 안 함 |
| ROM clip (±80°) | clip 대상의 "올바른 axial"을 모름 |
| SLERP interp | anchor도 -78°~-80°로 이미 오염, 수렴 지점 없음 |
| B1 재활성화 | 관통 + arm 비틀림 재발 |

**결론: motion prior(HMP)가 biomechanically plausible한 wrist axial을 제공해야 함.**

---

## 5. HMP 전환 계획

### Baseline
- `smplx_kalman_B.pt`: HMP 결과와 비교할 heuristic baseline
- 백업: `outputs/cap_pipeline/backup_pre_hmp/`

### HMP STEP 0 (조사)
- AMASS 필요 여부
- MANO prior 재활용 가능성
- prior checkpoint 확인
- 별도 환경: `/workspace/hmp_test`

---

## 6. 핵심 수치 요약

```
Stage 1 (arm_prefit):
  arm RMS: mean=3.27px, p50=3.27px (target ≤ 3.5px ✓)

Stage 3 (merged_prefit):
  Stage A geo(FK wrist, HaMeR wrist_R): mean=0.0°, max=0.1° ✓
  2D wrist position error: LEFT mean=147.6px, RIGHT mean=156.4px (깊이 스케일 차이, 렌더 정렬에는 무관)
  hand_pose injection: 1495/1495 frames both hands ✓

Kalman-B:
  L_wrist velocity: mean=1.53°/f, p95=4.36°/f, max=33.5°/f
  R_wrist velocity: mean=2.16°/f, max=358.8°/f (Euler wrap artifact, 실제 회전은 smooth)
  ROM >160°: 0 frames ✓
```

---

## 7. 재현 명령

```bash
# Stage 1
python cap_arm_prefit.py

# Stage 2  
python cap_mano_prefit.py

# Stage 3
python cap_merge_opt.py

# Kalman-B
python cap_kalman_wrist.py \
  --input-pt smplx_merged_prefit.pt \
  --gate-deg 33.0 --gate-factor 50.0
# Config B (q=0.0005, r=0.2) → smplx_kalman_B.pt
```
