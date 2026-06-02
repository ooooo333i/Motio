# 손목/손 모션 파이프라인 — 전체 정리 (capstone_deploy)

> 작성 2026-06-01. 테니스 포핸드 영상(GVHMR 몸 + HaMeR 손 → SMPL-X)에서
> **손목 품질**을 개선하기 위한 파이프라인과 그동안의 최적화·수정 과정을 세세히 기록.
> 언제든 이 문서만 보고 재현/이해 가능하도록 작성.

---

## 0. TL;DR (제일 좋은 결과 = 무엇을 쓰면 되나)

- **단일 영상 최선**: `cmp_adaptive_v3.pt`
  = HaMeR 2D정제(faithful) 손목 + 갑작스러운 트위스트 구간만 GVHMR axial + bad_frames slerp 보간 + 경계 스무딩.
  생성 스크립트: `wrist_pipeline/01_fit_faithful.py → 02_adaptive_axial.py → 03_badframe_slerp_smooth.py`
- **여러 영상 일반화(검증됨, 4개 영상)**: `batch_fit2d.py → batch_finger_kalman.py → batch_render.py`
  = 손목 강한 2D fit + wrist-local 회전평활(WSM=20) + 손가락 고정, 그 뒤 손가락만 Kalman.
  (단일영상의 adaptive/bad_frames 단계를 뺀 **간소·일반화 버전**. 영상별 K 동적.)
- **환경**: `/workspace/envs/cap-deploy/bin/python` (HMP만 `/workspace/envs/hmp`).
- **절대 금지**: 손목 **axial(전완축 비틀림)** 에 Kalman/2D-fit/HMP로 “고치려” 들지 말 것 — 작은 손에선 관측 불가(아래 §2.1). axial이 필요하면 **GVHMR**에서만 가져온다.

---

## 1. 전체 파이프라인 (데이터 흐름)

```
영상(mp4)
  │
  ├─[1] GVHMR  ─────────────→ hmr4d_results.pt   (몸/팔 SMPL-X, K_fullimg, global/incam)
  │                            preprocess/{vitpose.pt(coco17), bbx.pt}
  │
  ├─[2] HaMeR(wholebody133) ─→ hamer_out_clean/*.npz  (손 MANO: 손목 R + 손가락 45D)
  │     + ViTPose-H wholebody → preprocess/wholebody_kp2d.npy  (133 keypoints 2D, 손 포함)
  │
  ├─[3] merge ──────────────→ smplx_merged_hamer_post.pt
  │                            (GVHMR 몸 + HaMeR 손가락(후처리), 손목=GVHMR init)
  │
  ├─[4] 손목 파이프라인 (우리 핵심) ─→ fit2d.pt / cmp_adaptive_v3.pt
  │       body_pose[19]=좌손목, body_pose[20]=우손목 만 최적화/교정
  │
  ├─[5] 손가락만 스무딩(Kalman) ─→ fit2d_fingerkalman.pt
  │       left/right_hand_pose(각 15관절)에만. 손목 axial 미적용.
  │
  └─[6] 렌더 ───────────────→ render_*.mp4  (SMPL-X mesh overlay, K 동적)
```

**SMPL-X 손목 규약**: `body_pose[19]`=left wrist(joint20), `body_pose[20]`=right wrist(joint21). axis-angle.
**손목 local 축(경험적 검증)**: X=axial(전완축 비틀림/회내·회외), Y=요/척側 편위, Z=굴곡/신전.
**손목 global↔local**: `G_wrist = G_elbow @ R_local`. 즉 `R_local = G_elbow⁻¹ @ G_wrist`.
FK 엘보까지: 우=`[2,5,8,13,16,18]`, 좌=`[2,5,8,12,15,17]` (body_pose 인덱스).

---

## 2. 핵심 문제와 근본 원인 (왜 이렇게 했나 — 수정 과정의 논리)

### 2.1 ★ 손목 axial(꽈배기/비틀림)은 작은 손에서 “관측 불가”
- 테니스 영상의 손은 라켓을 쥔 채 ~30px, 전완축 방향으로 보임.
- 손목+5 MCP 키포인트는 **전완축 회전(axial)에 거의 불변(null space)**.
- 결과:
  - **2D reprojection**: axial이 바뀌어도 2D 오차 거의 동일 → 2D-fit이 axial을 못 잡음(임의값).
  - **joints3d_smooth**: 6점이 회전축 근처 동일평면 → axial 평활 페널티도 미미.
  - **HaMeR**: 팔이 없어 axial 모호(원시 출력이 ±90~180° 튐).
  - **Kalman**: 빠른 jitter는 줄여도 sustained/static 오류(잘못된 axial 고정)는 못 고침. → 오히려 악화.
- **유일한 신뢰 소스 = GVHMR**: 전체 팔 chain(어깨–팔꿈치–손목) 기하 제약이 있어 손목 local axial이 안정적·물리적으로 올바름(전완 twist는 팔 chain에 속함).
- **양날의 검**: axial이 2D에 안 보인다 = axial 오류도 **렌더 overlay에 거의 안 보인다**.
  → 수치(velocity 180°/frame)로 “그로테스크”라 판정하면 오판. **반드시 렌더로 확인**.

### 2.2 GVHMR vs HaMeR 역할 분담
- **swing(굴곡/편위)** = 실제 손 방향. HaMeR(2D 정제)가 잘 따라감 → overlay 충실. → **base로 사용**.
- **axial(비틀림)** = HaMeR 못 잡음, GVHMR만 안정. → **sudden 구간만 GVHMR에서 차용**.
- 합성: swing-twist 분해 후 `new_wrist = swing(HaMeR) ⊗ twist(GVHMR, 필요 구간만)`.

### 2.3 Dyn-HaMR “원본 충실 재현” → 우리 케이스에선 실패
- Dyn-HaMR `optim.yaml`엔 `motion_prior(HMP)=0.075`, `bone_length=2000`, `joint_consistency=100`이 적혀있으나
  **실제 코드(`losses.py`/`optimizers.py`)에선 전부 주석 처리(비활성)**. `MotionOptimizer`(HMP 단계) 클래스도 통째 주석.
  → 실제로 도는 건 **joints2d(10000) + joints3d_smooth + pose_prior**, 변수는 root_orient+trans+betas+latent_pose.
  → 두 핵심 트릭: **2D loss와 joints3d_smooth를 hand-bbox/scale로 정규화**(size-invariant).
- 충실 재현해도 **우리 손목 axial은 못 잡음**(HaMeR 노이즈 그대로). 이유 = §2.1(작은 손 null space).
  Dyn-HaMR이 되는 건 “손이 크고 다각도로 잡힌 hand-centric 영상”이라 2D가 3DOF 전부를 잡기 때문.
- 결론: **“HMP가 마법”은 거짓**. Dyn-HaMR 성공요인은 강한 2D 전처리(큰 손 전제)지 HMP가 아님.

### 2.4 HMP(NeMF 손 모션 prior)는 우리 케이스에 불필요
- HMP는 **손 전용**(root_orient + 손 pose만, 팔 없음). 손의 **절대(global)** 방향을 “그럴듯한 손 움직임”으로 만들 뿐.
- 우리가 보는 손목각 = **전완 대비 상대(local)**. 홀로 멀쩡한 손도 팔에 붙으면 비틀림/안쪽 꺾임 발생.
- 실험(`cmp_adaptive_v2` → HMP, anchor=30): axial 변화 평균 R 8.5°/L 16.5°로 작고 **렌더상 거의 동일**(axial invisible).
  - “각도 제한(axial-only lock)”도 가능하나, 제한할수록 HMP가 줄 게 없어져 **사실상 무효**.
- 결론: **HMP 제외**(가시 이득 0). axial은 GVHMR, swing은 HaMeR로 충분.

### 2.5 bad_frames (HaMeR 검출 불량 구간)
- 빠른 스윙·모션블러 구간에서 HaMeR 손 메쉬 불량 → per-frame fit이 **180°/frame 스파이크**(예: f410-413).
- ViTPose 2D는 신뢰도 높아도 3D 방향이 degenerate해서 손이 “몸 안쪽으로 꺾임”.
- **해결**: bad_frames(우손/좌손 각 json) 구간 손목을 **양끝 good 프레임에서 slerp 보간** + 경계 가우시안 스무딩.
  (per-frame 불량 데이터를 믿지 않고 이웃으로 메움.) f408-413: 109~180 → **<13°/frame**.

### 2.6 손가락(hand_pose)은 별도, 손목과 독립
- 손가락 jitter(원시 Δ 최대 ~3.9 rad/frame)는 손목과 무관.
- `entry_point_hamer.postprocess_hand_sequence`(이상치+결측보간+가우시안)로 1차 평활 → Δ~0.57.
- 추가로 **손가락 15관절에만 회전 Kalman** 적용 가능(`batch_finger_kalman.py`). **손목 axial엔 절대 미적용**.

---

## 3. 손목 파이프라인 상세 — 단일영상 최선 (cmp_adaptive_v3)

base = `cmp_baseline_hamer.pt` (arm-prefit 몸 + HaMeR 손목 + 후처리 손가락).

1. **`wrist_pipeline/01_fit_faithful.py`** → `cmp_dynhamr_faithful.pt`
   - 손목(body_pose[19/20])만 최적화.
   - loss: **scale-normalized 2D**(WB 손목+5MCP, weight 10000) + **scale-normalized joints3d_smooth**(1000→10000).
   - 2-stage LBFGS(Root 50it → Smooth 100it). 회전평활/HMP/bone_length 없음(Dyn-HaMR 실제 활성구성 충실).
   - 효과: 실제 손 추적(swing) 양호. **단, axial은 여전히 HaMeR 노이즈**(§2.1).
2. **`wrist_pipeline/02_adaptive_axial.py`** → `cmp_adaptive_v2.pt`
   - swing-twist 분해. **swing = faithful 유지**, **axial(twist)** 은 “sudden” 구간만 GVHMR로 블렌딩.
   - sudden 검출 = HaMeR axial이 자기 **local-median(window7)에서 30° 이상 튀는 곳**(꾸준한 차이는 정상 추적이라 건드리지 않음). dilate=2, 가우시안 ramp.
   - axial target = `a_base + w·wrap(a_gv − a_base)`, gaussian σ1.2.
3. **`wrist_pipeline/03_badframe_slerp_smooth.py`** → `cmp_adaptive_v3.pt`  ★최종
   - bad_frames(우 `bad_frames_full_R.json`/좌 `_L.json`, 또는 hmp_test 쪽 json) 손목을 양끝 good에서 **slerp 보간**.
   - bad 근처만 quat 가우시안(σ1.5) 블렌딩(경계 점프 제거), clean 구간 보존.
   - 결과: f408-415 변화량 109~180 → ≤13°/frame, 손목 속도 >33°/frame 128→84(나머지는 invisible axial).

렌더: `wrist_pipeline/render_region.py <pt> <out.mp4> <a> <b>` (정면, 구간 또는 전체).

---

## 4. 일반화 배치 파이프라인 — 여러 영상 (검증: 4개)

단일영상의 adaptive/bad_frames 단계를 뺀 **간소·일반화 버전**(영상별 K 동적, 추가 튜닝 불필요).

per clip:
1. **GVHMR+HaMeR+merge**: `entry_point_hamer.py --video <v> --output-root outputs/cap_pipeline --auto-person --no-interactive --skip-result-video` (env `CAP_SKIP_WRIST_REFINE=1`)
   → `<clip>/{hmr4d_results.pt, smplx_merged_hamer_post.pt, preprocess/wholebody_kp2d.npy}`
2. **손목 fit2d**: `batch_fit2d.py <clip_dir>` → `fit2d.pt`
   - 손목만 Adam 250it. loss = **강한 2D**(픽셀, W2D=1.0, /1000) + **WSM=20 × wrist-local aa 회전평활**. 손가락 고정.
   - K는 `K_fullimg`에서 동적, WB는 `preprocess/wholebody_kp2d.npy`.
3. **손가락 Kalman**: `batch_finger_kalman.py <clip_dir>` → `fit2d_fingerkalman.pt`
   - 좌/우 hand_pose 15관절 각각 회전 Kalman(q=0.001,r=0.2,gate=60,factor=30). **손목 미적용**.
4. **렌더**: `batch_render.py <clip_dir> <pt> <out.mp4> <원본영상>` → `render_*.mp4`

오케스트레이터 예시: `tmp/batch_run.sh`(4클립 순차 + 타이밍). 산출 = `outputs/cap_pipeline/front_{n}/render_front_{n}_{fit2d,fingerkalman}.mp4`.

---

## 5. 파일/스크립트 맵

| 파일 | 역할 |
|---|---|
| `entry_point.py` | GVHMR 실행, HaMeR 유틸, run_gvhmr, K_fullimg 산출 |
| `entry_point_hamer.py` | GVHMR→HaMeR(wholebody133)→merge. **wholebody_kp2d.npy 저장**(추가됨), `CAP_SKIP_WRIST_REFINE`(추가됨) |
| `external/hamer/.../models/__init__.py` | `load_hamer(init_renderer=False)` (pyrender 우회, 추가됨) |
| `cap_arm_prefit.py` | ViTPose 팔 fit → smplx_arm_prefit.pt (단일영상 base의 팔) |
| `cap_kalman_wrist.py` | 회전 Kalman(QuatKalman, kalman_smooth_rotation) — 손가락 Kalman이 재사용 |
| `wrist_pipeline/01_fit_faithful.py` | scale-norm 2D + joints3d_smooth 손목 fit (단일) |
| `wrist_pipeline/02_adaptive_axial.py` | sudden 구간 axial→GVHMR (단일) |
| `wrist_pipeline/03_badframe_slerp_smooth.py` | bad_frames slerp + 경계 스무딩 (단일 최종) |
| `wrist_pipeline/render_region.py` | 단일영상 구간/전체 렌더 |
| `batch_fit2d.py` | 일반화 손목 fit (강한2D+WSM=20+손가락고정) |
| `batch_finger_kalman.py` | 손가락만 Kalman |
| `batch_render.py` | 일반화 렌더(K 동적, 원본영상 명시) |
| `physical_limits.py` | ROM·각속도 기준(MAX_WRIST_DEG_PER_FRAME=33.3), swing-twist 분해, 축 검증 |
| `docs/PIPELINE_MAP.md` | LIVE/DEAD 분류, P1(entry_point_hamer)/P2(수동) 구분 |

산출 .pt(단일, `outputs/cap_pipeline/input/`): `cmp_baseline_hamer`, `cmp_dynhamr_faithful`, `cmp_adaptive_v2`, **`cmp_adaptive_v3`(최종)**, (분석용) `cmp_hmp_v2`, `cmp_hmp_axlock_v2`.

---

## 6. 실행 방법

```bash
PY=/workspace/envs/cap-deploy/bin/python
export PYOPENGL_PLATFORM=          # 프로필의 egl 상속 방지 (필수)

# (A) 단일 영상 — 최선(adaptive_v3) 재현 (input 클립 기준)
$PY wrist_pipeline/01_fit_faithful.py          # → cmp_dynhamr_faithful.pt
$PY wrist_pipeline/02_adaptive_axial.py        # → cmp_adaptive_v2.pt
$PY wrist_pipeline/03_badframe_slerp_smooth.py # → cmp_adaptive_v3.pt
$PY wrist_pipeline/render_region.py cmp_adaptive_v3.pt render_adaptive_v3_FULL.mp4 0 9999

# (B) 새 영상 1개 일반화
CAP_SKIP_WRIST_REFINE=1 $PY entry_point_hamer.py --video <video.mp4> \
    --output-root outputs/cap_pipeline --auto-person --no-interactive --skip-result-video
CLIP=outputs/cap_pipeline/<stem>
$PY batch_fit2d.py "$CLIP"
$PY batch_finger_kalman.py "$CLIP"
$PY batch_render.py "$CLIP" fit2d.pt           render_fit2d.mp4        <video.mp4>
$PY batch_render.py "$CLIP" fit2d_fingerkalman.pt render_fingerkalman.mp4 <video.mp4>
```

---

## 7. 일반화를 위한 수정 (하드코딩 제거 / 환경)

1. **K 동적**: 모든 스크립트가 클립별 `K_fullimg`를 읽음(하드코딩 없음). 2202.9는 GVHMR이 1080×1920에서 산출한 focal일 뿐(해상도 다르면 자동 변경).
2. **EGL/pyrender 우회**: 프로필 `PYOPENGL_PLATFORM=egl` + libEGL.so 부재로 HaMeR import/OffscreenRenderer 크래시.
   → `export PYOPENGL_PLATFORM=`(빈값) + `load_hamer(init_renderer=False)`. (검출에 렌더러 불필요. 복구하려면 `CAP_HAMER_RENDERER=1`.)
3. **wholebody_kp2d.npy 캐싱**: entry_point_hamer가 메모리에만 두던 WB133을 `<clip>/preprocess/`에 저장(fit2d 입력).
4. **레거시 단계 차단**: `CAP_SKIP_WRIST_REFINE=1` → merge 이후 Stage A/B + 3변종 렌더 건너뜀.

---

## 8. 타이밍 (4영상 배치, 1080×1920/60fps)

| 클립 | 프레임 | GVHMR | HaMeR | wholebody | fit2d | 손가락Kalman | 렌더(2종) | 총합 |
|---|---|---|---|---|---|---|---|---|
| front_1 | 1297 | (재사용) | 150.9 | 22.4 | 18.7 | 3.8 | 415 | 597 |
| front_2 | 1891 | (재사용) | 205.3 | 33.6 | 25.7 | 5.4 | 498 | 743 |
| front_3 | 1495 | 123.3 | 166.5 | 25.8 | 21.0 | 4.1 | 335 | 752 |
| front_4 | 1183 | 96.5 | 149.1 | 20.5 | 18.0 | 3.3 | 246 | 595 |

신규 GVHMR ≈ 100~125s. fit2d 2D RMSE 모두 ~3.7~4px. 렌더가 최대 병목(전체 프레임 ×2).

---

## 9. 주의 / 금지

- ✗ 손목 **axial**에 Kalman/2D-fit/HMP로 직접 교정 시도 금지(작은 손 null space). axial은 GVHMR에서만.
- ✗ 수치(velocity 등)만으로 품질 판정 금지 — axial은 invisible. **반드시 렌더로 확인**.
- ✗ 손목에 맞춰 GVHMR 팔을 역으로 fitting하지 말 것.
- ✓ 손가락 스무딩과 손목 처리는 독립(중복 없이).
- ✓ 기존 tennis 결과(`outputs/cap_pipeline/input/`, `cmp_adaptive_v2/v3`)는 보존.

관련 메모리: `final-wrist-pipeline`, `batch-generalization`, `dynhamr-faithful-fails`, `arm-vs-bbox-rootcause`, `smplx-wrist-axes`, `anomaly-regions-vitpose`.
