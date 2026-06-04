# 현재 파이프라인(Ours) 전체 설명 + 한계 분석

> 2026-06-03. 서비스가 쓰는 batch 체인(=Ours, fast mode) 기준. 후처리 고도화(2.1~2.3)와
> 논문 포스터 비교의 토대 문서. 관련: `WRIST_PIPELINE.md`, `CONTEXT_OVERVIEW.md`, `MOTIO_SPEED_FLAGS.md`.

## 0. 한눈에
단안 테니스 영상 → 사람 검출/선택 → **GVHMR(전신 SMPL-X) + HaMeR(손 MANO) 융합** →
손목 2D fit + 손가락 평활 → FBX. "이음새인 손목"을 보정하는 게 핵심.

## 1. 단계별 (entry_point_opt mode=fast → entry_point_hamer)

| # | 단계 | 입력→출력 | 핵심 |
|---|---|---|---|
| 0 | 검출/선택 | clip → person_track_id | YOLO track + 선수필터(면적/지속). bbx 재사용 옵션(CAP_REUSE_DETECTION). |
| a | prepare_selected_person_track | clip → `preprocess/bbx.pt` | 선택 선수의 per-frame bbox(xyxy,xys). |
| b | **GVHMR preprocess** | clip,bbx → vitpose·feature·VO | ViTPose-2D(크롭 img_ds=0.5→256×192) + ViT features + 카메라 SimpleVO. |
| c | **GVHMR predict** | features → `hmr4d_results.pt` | SMPL-X 전신(global+incam). 손은 거친 상태. |
| d | **HaMeR** | clip,kp → 손별 npz | Pass1 wholebody ViTPose(133kp)→L/R 손 bbox(시계열 평활). Pass2 ViT-H 크롭 per-hand(프레임 가로질러 배치)→MANO 손 포즈. **좌/우 완전 분리.** |
| e | **merge** | gvhmr+손 → `smplx_merged_hamer_post.pt` | HaMeR 손을 SMPL-X에 주입. **hand_mean 차감**, **R_cam_align=I transplant**, side별 시계열 후처리(게이팅·gap보간·outlier reject·smoothing). |
| f | **batch_fit2d** | merged → `fit2d.pt` | **L/R 손목(body_pose[19],[20])만** wholebody 2D 손목/손가락기저 키포인트에 fit(Adam 250it). `W2D=1, WSM=20`(강한 2D + 강한 손목 평활). 손가락·몸 고정. |
| g | **batch_finger_kalman** | fit2d → `fit2d_fingerkalman.pt`(=Ours 최종) | 손가락 15관절/손만 회전 Kalman 평활. **손목·body 미적용.** |
| h | export | final.pt → `output.fbx` | rotation/transl 평활 npz → Blender SMPL-X addon FBX(keep-fps) + 접지. |

핵심 불변식: ① HaMeR(pose2rot=False)→SMPL-X 주입 시 side별 `hand_mean` 차감.
② GVHMR incam과 HaMeR global_orient이 동일 카메라 프레임 → `R_w=G_e⁻¹@G_w` transplant(왼손 미러).
③ 손목 **축회전(axial)은 2D에 안 보임 → GVHMR에서만**, 품질은 렌더로 판정.

## 2. 한계 (후처리 고도화 대상)

### 2.1 손목/팔 깊이 겹침 (다리 X — 사용자 정정 2026-06-03)
- **부위**: 다리 self-collision 이 아니라 **손목/팔**. 오버레이에서 전완·손이 몸 쪽으로 접혀/겹쳐 보이는 구간.
- **원인**: `batch_fit2d` 가 손목(body_pose[19],[20])을 **wholebody 2D 키포인트에 fit** 할 때, 2D(xy)만 제약하고
  **깊이(z)는 제약이 없음** → fit 과정에서 손목/전완의 z 가 어긋나 팔이 몸 쪽으로 말려/겹친다.
- **사용자 아이디어(유효, 비교적 구현 쉬움)**: 2D 투영을 1순위로 맞추되 **GVHMR 의 z(팔/손목 깊이)를 신뢰·보존**.
  → batch_fit2d 손실에 **z-anchor 항**(현재 wr_R/wr_L fit 결과 관절의 z 를 GVHMR 원본 z 에 묶는 패널티) 추가.
  손목만 fit 하므로 영향 범위가 작아 **다리 충돌(LBS·고비용)과 달리 batch_fit2d 한 곳 수정으로 가능**.
- 한계: 단안 z 자체가 ambiguous + GVHMR z 도 완벽치 않음. z 보존을 너무 강하게 하면 2D fit 이 약해짐(가중치 튜닝 필요).

### 2.2 손 좌/우 처리 (검토 결과)
- **코드상 완전 분리 확인**: HaMeR per-hand, merge side별, fit2d `wr_L/wr_R` 독립 파라미터, finger_kalman side별.
  → 공유 파라미터 버그 **없음**.
- "손목 각도가 너무 비슷" 의심의 실제 원인 후보: (a) **그립 시 양손이 붙어 wholebody 2D 손목 키포인트가
  모호/근접** → 양쪽 fit이 비슷한 타깃으로 수렴, (b) **강한 WSM=20 평활**이 좌우 모두 비슷하게 눌러버림,
  (c) 실제로 두 손이 비슷한 포즈(양손 그립). → 검증: 클립별 L/R 손목 각 상관 + wholebody L/R 손목 conf 확인.

### 2.3 손목 튐 / 그립 갑자기 펴짐
- **원인**: HaMeR는 per-frame 추론 → 시간 일관성 없음. 오클루전·모션블러·작은 손에서 한 프레임 오검 →
  손목이 튀거나 그립이 갑자기 펴짐. merge에 게이팅/outlier-reject가 있으나 작은 손 axial은 관측 불가(null space)라
  잔여 스파이크가 남음.
- 방향: HaMeR 신뢰도 기반 시계열 게이팅 강화 + (튐 구간) GVHMR로 폴백/블렌딩 + 그립 상태 일관성(연속성) 제약.

### 2.4 adaptive 추종의 lag/이상치 trade-off
- **현상**: 오버레이 시 빠른 손목 동작을 못 따라가는 구간 존재.
- **원인**: batch_fit2d의 `WSM=20`(강한 손목 평활)이 **안정성↔반응성 trade-off**에서 안정성 쪽 →
  빠른 스윙에서 under-fit/lag. 반대로 평활 낮추면 HaMeR 튐이 그대로 들어옴.
- **목표(미해결 과제)**: HaMeR의 위치/각도를 최대한 유지하면서 튐 없이 따라가기. 후보:
  (i) **신뢰도 적응형 평활**(고신뢰=반응, 저신뢰=평활), (ii) 속도-인지 스무딩(One-Euro류로 빠른 구간 lag↓),
  (iii) HaMeR raw에 가깝게 두되 outlier만 제거(현재는 평활이 과함), (iv) 2D fit weight를 구간별 가변.

## 3. 포스터 비교군 (이 파이프라인 관점에서)
- (A) GVHMR 손 직결: 단계 c까지(HaMeR 없음).
- (B) GVHMR-fit + HaMeR 손 각도: 단계 e/f(현재 핵심).
- (C) 실패 알고리즘 사례: 폐기된 HMP/heuristic, velocity 이상탐지 오판 등(아카이브 메모리 참조).
- (D) Dyn-HaMR(카메라 VIPE) 결과.
- (E) Ours = 최종 `fit2d_fingerkalman.pt`.
