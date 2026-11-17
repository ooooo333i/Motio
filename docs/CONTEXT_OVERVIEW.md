# CONTEXT_OVERVIEW — capstone_deploy 전체 맥락 (2026-06-02)

> 새 세션/협업자가 **이 파일 하나만 읽고** 프로젝트의 목적·현재 정답 경로·열린 이슈를 파악하도록 만든 최상위 오리엔테이션 문서.
> 상세 근거는 `WRIST_PIPELINE.md`(최신·정답), 코드 맵은 본 문서 §5 참고.
> ⚠️ `PIPELINE_STATUS.md`(5/31)·`PIPELINE_MAP.md`(6/1)는 **HMP 시대의 옛 결론**으로 일부 stale — §6 참고.

---

## 1. 한 줄 요약

단안(monocular) **테니스 포핸드 영상**에서 **GVHMR(전신) + HaMeR(손)** 를 융합해 SMPL-X 3D 모션을 복원하되,
두 모델의 이음새인 **손목(wrist) 관절**을 물리적으로 자연스럽게 보정하는 것이 핵심인 배포용 프로젝트.

---

## 2. 목표와 핵심 문제

**목표**: 단일 테니스 영상 → 렌더 오버레이에서 손목/손이 물리적으로 올바르게 보이는 깨끗한 SMPL-X 복원.

**구성**
- 몸체 = GVHMR (COCO17 ViTPose 2D로 구동되는 전신 SMPL-X)
- 손가락 = HaMeR (프레임별 MANO: 손목 글로벌 R + 45D 손가락 포즈)
- **문제 관절** = 손목: `body_pose[19]`=왼손, `body_pose[20]`=오른손 (axis-angle)

**근본 원인 (가장 중요한 사실 — WRIST_PIPELINE §2.1)**
- 손목 **축회전(axial = 전완 비틀림/회내·회외, SMPL-X 손목 로컬 X축)** 은 작은 손(~30px, 라켓 쥐고 정면)에서 **관측 불가(null space)**.
- 손목+5 MCP 키포인트가 회전축 근처 → 2D 재투영 오차·joints3d_smooth가 축회전에 거의 불변, HaMeR는 ±90~180° 임의값.
- **결론: 어떤 2D 기반·손-단독 방법으로도 축회전 복원 불가. 오직 GVHMR(어깨→팔꿈치→손목 전체 팔 체인 기하)만 신뢰 가능.**

**양날의 검**: 축회전이 2D에 안 보인다 = 축회전 오류도 렌더에 거의 안 보인다.
→ **수치(속도 180°/f 등)만으로 품질 판정 금지. 반드시 렌더로 확인.**

**손목 로컬↔글로벌 규약**
- `G_wrist = G_elbow @ R_local` → `R_local = G_elbow⁻¹ @ G_wrist`
- 로컬 축: **X=axial(비틀림), Y=요/척측 편위(deviation), Z=굴곡/신전(flexion)** (경험적 검증, `physical_limits.py`가 권위 기준)
- 팔꿈치까지 FK 체인(body_pose 인덱스): 오른쪽=`[2,5,8,13,16,18]`, 왼쪽=`[2,5,8,12,15,17]`

---

## 3. 전체 파이프라인 흐름

```
영상(mp4)
  → YOLO 트래킹            → preprocess/bbx.pt
  → ViTPose COCO17 + VO    → preprocess/{vitpose.pt, vit_features.pt, slam_results.pt}
  → GVHMR (CUDA 필수)      → hmr4d_results.pt  (전신 SMPL-X global/incam, K_fullimg)
  → HaMeR (WholeBody133)   → hamer_out_clean/####_<pid>.npz + preprocess/wholebody_kp2d.npy
  → merge                  → smplx_merged_hamer_post.pt  (몸+손가락, 손목은 아직 GVHMR)
  → [손목 보정: 아래 두 경로 중 하나]
```

### 경로 A — 단일영상 BEST (adaptive_v3)  ※ `outputs/cap_pipeline/input/` (tennis 1495f)

```
hmr4d_results.pt + preprocess/vitpose.pt
  → cap_arm_prefit.py                        → smplx_arm_prefit.pt
  → cap_compare_hmp_modes.py                 → cmp_baseline_hamer.pt   (LIVE base — 없으면 재현 불가)
  → wrist_pipeline/01_fit_faithful.py        → cmp_dynhamr_faithful.pt
  → wrist_pipeline/02_adaptive_axial.py      → cmp_adaptive_v2.pt
  → wrist_pipeline/03_badframe_slerp_smooth.py → cmp_adaptive_v3.pt   ★FINAL
  → wrist_pipeline/render_region.py <pt> <out.mp4> <a> <b>
```

### 경로 B — 배치/일반화 (clip stem별, 동적 K, 4클립+ 검증)

```
smplx_merged_hamer_post.pt + preprocess/wholebody_kp2d.npy
  → batch_fit2d.py          → fit2d.pt    (강한 2D + 손목-로컬 회전평활 WSM=20, 손가락 동결)
  → batch_finger_kalman.py  → fit2d_fingerkalman.pt  (손가락 15관절만 Kalman; 손목 축회전엔 절대 미적용)
  → batch_render.py <clip> <pt> <out.mp4> <video>  → render_*.mp4
```
배치는 단일영상의 adaptive-axial / bad_frames 단계를 **의도적으로 생략**한 간소·일반화 버전.

---

## 4. 핵심 알고리즘 (adaptive_v3) — "HaMeR를 신뢰하되 선택적 수리"

**역할 분담**: swing(굴곡+편위 = 실제 보이는 손 방향)은 HaMeR 유지, axial(비틀림)은 sudden 구간만 GVHMR 차용.
→ `new_wrist = swing(HaMeR) ∘ twist(GVHMR, 필요한 곳에서만)`

1. **arm prefit** (`cap_arm_prefit.py`): 팔 8개 row(collar 12/13, shoulder 15/16, elbow 17/18, wrist 19/20)만 Adam으로 ViTPose 2D 피팅. `L = L_2d + 5·L_prior(원본 GVHMR L2) + 1e3·L_temporal(geodesic 평활)`. 비-팔 gradient 마스킹. 손목은 여기선 coarse.
2. **base build** (`cap_compare_hmp_modes.py`): `R_w = G_e⁻¹ @ HaMeR_wrist_R`(왼손 `M=diag(-1,1,1)` 미러)를 손목 row에 기록. 손가락은 `postprocess_hand_sequence`(temporal gate + MAD outlier + gap interp + gaussian) 후 hand_mean 차감해 동결. baseline(HaMeR)/full(HMP)/selective 3변형 중 **baseline만 LIVE**.
3. **2D faithful fit** (`01_fit_faithful.py`): 손목 row(20,19)만 LBFGS. `L = 10000·L_2d + w_j3ds·L_j3ds`. L_2d는 손목+5MCP 투영 후 hand-bbox 중심/스케일 정규화(크기 불변), gmof robust(σ=100), conf² 가중. 2단계(Root 50it / Smooth 100it). swing 충실, but 축회전 아티팩트 발생 가능.
4. **adaptive axial** (`02_adaptive_axial.py`): 전완 X축 swing-twist 분해 → 축회전각 `a_base`. **sudden twist 검출** = `|wrap(a_base − median(a_base, MEDW=7))| > 30°`, dilate(2)+gaussian ramp. 검출 구간만 `a_tgt = a_base + w·wrap(a_gv − a_base)`로 GVHMR 블렌딩, 재합성. **swing은 항상 HaMeR.**
5. **badframe slerp** (`03_badframe_slerp_smooth.py`, FINAL): HaMeR `bad_frames.json`(R/L) 구간 손목을 양끝 good에서 SLERP 보간 → sign-aligned gaussian(σ=1.5) quat 평활을 **bad 경계 근처에서만** SLERP 블렌딩(clean 보존). f408–415: 109~180 → ≤13°/f.

---

## 5. 코드 지도 (LIVE vs DEAD)

### LIVE — 진입점/서비스
| 파일 | 역할 |
|---|---|
| `entry_point.py` | baseline 단일영상 CLI(COCO17 bbox, raw merge → `smplx_merged_hamer.pt`). 배포 baseline + 공유 헬퍼 |
| `entry_point_hamer.py` | 강화 CLI. WholeBody133 bbox, bbox 평활, 손 postprocess → `smplx_merged_hamer_post.pt`. `CAP_SKIP_WRIST_REFINE=1`이면 레거시 Stage A/B 차단. `postprocess_hand_sequence`/`HandPostprocessConfig` 제공 |
| `service.py` | FastAPI. `POST /runs` 업로드 → `entry_point.py` subprocess(강제 `--auto-person --no-interactive`). ⚠️ baseline 산출물만 노출 |
| `batch_entry_point.py` | 폴더 배치 러너(`quality_labels.csv` accept 필터) |

### LIVE — 손목 알고리즘 / 배치
| 파일 | 역할 |
|---|---|
| `cap_arm_prefit.py` | ViTPose 팔 fit → `smplx_arm_prefit.pt` |
| `cap_compare_hmp_modes.py` | base 생성 → `cmp_baseline_hamer.pt` (adaptive_v3 진입점) |
| `wrist_pipeline/01_fit_faithful.py` | scale-norm 2D + j3d_smooth 손목 fit |
| `wrist_pipeline/02_adaptive_axial.py` | sudden 구간 axial→GVHMR |
| `wrist_pipeline/03_badframe_slerp_smooth.py` | bad_frames slerp + 경계 평활 (단일 최종) |
| `physical_limits.py` | 축맵·ROM 표·`MAX_WRIST_DEG_PER_FRAME=33.33`·swing-twist·검증기 (런타임 스테이지 아님) |
| `batch_fit2d.py` / `batch_finger_kalman.py` / `batch_render.py` | 일반화 손목 fit / 손가락 Kalman / 동적 K 렌더 |

### LIVE — 렌더/뷰어
| 파일 | 역할 |
|---|---|
| `wrist_pipeline/render_region.py` | input/ 고정 구간 렌더(close inspection) |
| `batch_render.py` | 일반 렌더(동적 K, 프레임 번호 burn-in) |
| `viewer.py` | **NEW(6/2, 미커밋)** 인터랙티브 3D pyrender 뷰어(2D 오버레이 아님). 기본 `_post.pt → smplx_merged_hamer.pt`; input/ 클립엔 `_post.pt`가 없어 `--merged cmp_adaptive_v3.pt` 명시 필요 |

### LIVE — GVHMR/HaMeR 런타임
- `hmr4d/`: `build_gvhmr.py`, `model/gvhmr/{gvhmr_pl_demo, pipeline/gvhmr_pipeline}.py`, `network/gvhmr/relative_transformer.py`(RoPE Transformer, max_len=120), `utils/{endecoder(out dim=151), postprocess}.py`
- `tools/demo/demo.py`(전처리·예측·렌더 내부), `tools/visual_critic_loop.py`(HaMeR 투영/bbox 디버그)
- `external/hamer/hamer/models/hamer.py`(ViT-H + MANO head, 오른손 예측·왼손 미러)

### DEAD / 격리 (삭제 금지)
| 항목 | 비고 |
|---|---|
| `cap_wrist_refine.py` (52KB) | 원조 DeVI 2-stage(replace + refine_2d). 대부분 DEAD — **단 `load_hamer_detections_full`만 LIVE**(cap_compare가 import). B1/B2 의도적 disabled |
| `cap_inject_hmp.py` + HMP 경로 | P2 HMP 주입 체인. **가시 이득 0으로 폐기**(superseded by adaptive_v3) |
| `cmp_full_hmp.pt`, `cmp_selective_hmp.pt` | 비교용·DEAD(삭제 금지) |
| `deprecated/`, `archive/` | 휴리스틱기 렌더/분석 스크립트·이미지(격리됨) |

---

## 6. 현재 상태 & 진화 (⚠️ 문서 stale 주의)

시간순으로 접근법이 바뀌었고, **오래된 문서가 stale** 하다:

| 문서 | 날짜 | 그 시점의 결론 | 현재 유효성 |
|---|---|---|---|
| `PIPELINE_STATUS.md` | 5/31 | 휴리스틱(Kalman-B) 완료 → "HMP 전환 예정" | **stale** (HMP 폐기됨) |
| `PIPELINE_MAP.md` | 6/1 | P1(공식) vs P2(수동), "최선 = HMP 손목 B" | **stale** (HMP 폐기됨) |
| `WRIST_PIPELINE.md` | 6/1 후반 | **HMP 폐기**, adaptive_v3(단일)+batch(다중)로 확정 | ✅ **현재 정답** |
| `README.md` | 6/2 | adaptive_v3 재현 + 배치 일반화 | ✅ 현재 |

→ PIPELINE_STATUS/MAP의 "HMP가 최선" 서술은 **이미 폐기된 옛 결론**. 현재 정답 = **adaptive_v3(단일) / batch(다중)**.

### 열린 이슈
- **배포 단절**: `service.py`·`entry_point.py`는 baseline `smplx_merged_hamer.pt`만 노출 → **배포 서비스가 최선 결과(adaptive_v3 / `_post.pt`)를 내보내지 않음.** 공식 엔트리(P1)와 최선 결과 경로(P2)가 끊겨 있음 → 정리/재배선 쟁점.
- **side-view 일반화 진행 중(정황)**: 6/2에 side 클립 4개 + `viewer.py` 신규 추가(미커밋). side-view 일반화 검증 / 3D 점검 단계로 추정.
- git 미커밋 다수: 영상(front_#*, side_#*, input.mp4), `viewer.py`, `batch_inputs/`, 로그 등.

---

## 7. 주의사항 (gotchas)

- **`numpy==1.23.5` 고정 필수** (2.x → pandas/바이너리 충돌).
- 체크포인트는 git 미포함 — 직접 배치(README "체크포인트 배치" 절). **GPU/CUDA 필수**(GVHMR는 CUDA 없으면 실패).
- 단일영상 스크립트(`wrist_pipeline/*`, `cap_compare_hmp_modes.py`)는 `F=1495`·input 클립·`/workspace/hmp_test` **절대경로 하드코딩** → 새 영상은 반드시 `batch_*.py`(경로 동적) 사용.
- `export PYOPENGL_PLATFORM=`(빈값) + `load_hamer(init_renderer=False)` 필요(EGL/pyrender 우회). 복구는 `CAP_HAMER_RENDERER=1`.
- 환경: `/workspace/envs/cap-deploy/bin/python` (HMP만 `/workspace/envs/hmp`).
- **손목 axial에 Kalman/2D-fit/HMP 직접 교정 금지** — 작은 손 null space. axial은 GVHMR에서만.
- **수치만으로 품질 판정 금지 — 반드시 렌더로 확인** (axial invisible).
- `MAX_WRIST_DEG_PER_FRAME`은 60fps 기준(33.3°/f)이나 일부 클립이 30fps → gate가 ~2배 느슨.
- 보존: `outputs/cap_pipeline/input/`(tennis 결과), `backup_pre_refactor/`, `backup_pre_hmp/`.

---

## 8. 실행 빠른 참조

```bash
PY=/workspace/envs/cap-deploy/bin/python
export PYOPENGL_PLATFORM=          # 프로필의 egl 상속 방지 (필수)

# (A) 단일영상 — adaptive_v3 재현 (input 클립)
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
$PY batch_render.py "$CLIP" fit2d.pt              render_fit2d.mp4        <video.mp4>
$PY batch_render.py "$CLIP" fit2d_fingerkalman.pt render_fingerkalman.mp4 <video.mp4>
```
