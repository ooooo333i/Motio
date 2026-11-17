# ADAPTIVE_V4_DESIGN — 손목 파이프라인 v4 설계 노트 (2026-06-03)

> 목적: 서비스(배치 일반화)에서 돌릴 **adaptive_v4** 설계. 현재 알고리즘을 코드 기준으로
> 정밀 재정리하고, 한계를 사용자가 관측한 3대 문제(전완각/overlay 정확도, 충돌, 튐·그립)와
> 매핑한 뒤, 오프라인 영상이라는 이점을 살린 v4 방향을 적는다.
> 선행 문서: `WRIST_PIPELINE.md`(정답), `CONTEXT_OVERVIEW.md`(오리엔테이션).

---

## 0. 가장 중요한 규약 (모든 분석의 뿌리)

**절대 손 방향 = `G_elbow ∘ R_local_wrist`.**
손목 local row(`body_pose[19]`=좌, `[20]`=우)는 *전완(G_elbow) 위에 얹히는 상대 회전*이다.
- `G_wrist = G_elbow @ R_local` → `R_local = G_elbow⁻¹ @ G_wrist`
- 손목 local 축(검증됨, `physical_limits.py`): **X=axial(전완 비틀림, invisible)**, Y=deviation, Z=flexion.
- axial은 작은 손(~30px)에서 **2D 관측 불가(null space)** → 오류도 렌더에 거의 안 보임(양날의 검).

---

## 1. ★ 배포 경로 ≠ 단일영상 best (가장 먼저 짚을 것)

두 경로의 **손목 처리가 완전히 다르다.** v4는 **배치 경로 위에** 올린다.

| | 단일영상 best (input/) | **배치/서비스 (배포)** |
|---|---|---|
| base pt | `cmp_baseline_hamer.pt` (arm_prefit 몸) | `smplx_merged_hamer_post.pt` (**raw GVHMR 몸, arm_prefit 없음**) |
| 손목 | `01 faithful → 02 adaptive_axial → 03 badframe_slerp` | `batch_fit2d.py` 한 방 |
| swing | 2D fit (smooth 없음) | 2D fit + **WSM=20 (전 축 평활)** |
| axial | sudden 구간만 GVHMR 블렌딩 | **없음** (HaMeR 노이즈 그대로, invisible이라 방치) |
| bad frame | 외부 JSON → SLERP 전면교체 | **없음** |
| 전완 | arm_prefit(2D fit) | **raw GVHMR (시계열 모델)** |

→ **서비스에는 arm_prefit / adaptive_axial / badframe_slerp가 전혀 없다.** 배포 손목 = `batch_fit2d`
의 *손목 local 2D fit + WSM=20 전축 평활* + `batch_finger_kalman`(손가락만). 단일영상 기법들은
"메뉴"로만 참고. v4는 배치 경로(`smplx_merged_hamer_post.pt` 기반)를 출발점으로 한다.

---

## 2. 현재 알고리즘 정밀 재정리 (코드 기준)

### 단일영상 (참고용)
1. **`01_fit_faithful.py`** — 손목 local row만 LBFGS.
   `L = 10000·L2d(손목+5MCP, hand-bbox center/scale 정규화=크기불변, gmof σ=100, conf²) + w_j3ds·L_j3ds_smooth(3D도 hand-scale 정규화)`. 2-stage(Root 50 / Smooth 100). **회전평활·prior 없음.** → swing 충실, axial·jitter 미처리.
2. **`02_adaptive_axial.py`** — swing-twist 분해, **axial만** 손봄. `a_base`(HaMeR axial)가 local-median(`MEDW=7`)에서 **>30°(`RESID`)** 튀면 sudden → dilate(2) → gaussian ramp `w`. `a_tgt=a_base+w·wrap(a_gv−a_base)`. **`a_gv`는 `smplx_arm_prefit.pt`에서** (02:12). swing은 항상 HaMeR.
3. **`03_badframe_slerp_smooth.py`** ★ — 외부 정적 JSON 구간의 **손목 전 축**을 양끝 good에서 SLERP 전면교체 + 경계 quat gaussian(σ1.5).

### 배치/서비스 (v4 출발점)
- **`batch_fit2d.py`**: 손목 Adam 250it. `L = 강한2D(/1000) + WSM=20·Σ(wr[1:]−wr[:-1])²`.
  평활을 **local axis-angle 전 축(swing+axial)** 에 동일 적용. base=`smplx_merged_hamer_post.pt`(raw GVHMR 몸).
- **`batch_finger_kalman.py`**: 좌/우 hand_pose 15관절만 회전 Kalman. 손목 미적용.

---

## 3. 사용자 관측 3대 문제 — 정확한 원인 (코드 매핑)

### 문제① "전완 각도 때문에 overlay 정확도 ↓ (대신 jitter는 막음)"

> ⚠️ **정정**: 이전 설명에서 "arm_prefit이 forearm twist를 오염시켜 a_gv가 틀어진다"고 했는데
> **부정확했다. 사용자 지적이 맞다.** 이유:
> - axial 기준 `a_gv`는 **GVHMR의 시계열 prior**라 대체로 물리적으로 말이 된다.
> - 게다가 COCO17 손목은 **점 하나**라 손목을 2D에 fit해도 **axial(전완축 비틀림)은 못 움직인다**
>   (점을 회전축에서 돌려도 그 점은 안 움직임). → 2D fit이 axial을 건드리지 못하고 prior(w=5)가
>   GVHMR 값 근처로 잡아둔다. **즉 a_gv ≈ GVHMR의 합리적 forearm-twist prior. 오염 아님.**

**그럼 실제 overlay 정확도 저하의 원인은 (영향 큰 순):**
1. **`03` badframe slerp**: bad 구간 **손목 전 축을 기하 보간으로 전면 교체** → 빠른 스윙에서 "그럴듯하나 실제와 다른" 포즈 → **"못 따라가는 구간"의 최대 주범**. (단, 배치엔 없음.)
2. **배치 `WSM=20` 전축 평활**: swing까지 평활 → **빠른 스윙 temporal lag**. 이게 배포에서의 핵심 정확도 손실.
3. **전완 jitter/depth 누수(구조적)**: 절대 손 = `G_elbow ∘ R_local`. GVHMR 팔의 **per-frame jitter·depth**가 절대 손에 그대로 새어든다. 그런데 `batch_fit2d`의 WSM 평활은 **local 공간**에서 일어나므로
   (a) 전완에서 기원한 jitter는 **못 잡고**, (b) 동시에 swing을 lag시킨다.
   → "전완 각도 때문에 정확도가 떨어진다"는 **전완 twist 오염이 아니라, 전완의 jitter/depth가 절대 손에 누수되고 local-space 평활이 그걸 못 잡으면서 swing만 뭉개는** 구조 때문.

### 문제② 충돌 — 활성 코드에 충돌/깊이 항 **전무** (확인 완료)
관측: **허벅지는 안 파고듦. 몸통은 가끔, 팔끼리·손끼리 겹침.** → 두 종류:
- **(a) 몸통 침투**: raw GVHMR 팔의 **depth-flip 모호성**(단안). 전완 depth가 흔들리면 팔/손이 몸통으로. + 손목 2D fit이 depth는 제약 안 함.
- **(b) 팔-팔 / 손-손 self-intersection**: 좌우 사지가 각자는 맞는 depth라도 **서로 겹침**. depth anchor만으론 부족 → **inter-segment(좌↔우) 충돌 항**이 필요.
- ※ 과거 `deprecated/cap_mano_prefit._apply_z_anchor`는 "HaMeR 손 depth→GVHMR" 용도였고 효과미미로 폐기 — **이번 arm depth anchor와는 별개 목적.**

### 문제③ 튐 / 그립 갑자기 펴짐 / 못 따라감
- **튐**: 단일 `01`은 temporal smooth=0. 배치는 WSM이 있으나 local·전축이라 전완 jitter는 통과.
- **그립 펴짐**: 손가락은 `postprocess_hand_sequence`(gate+MAD+gaussian)+Kalman뿐, **grip-closedness prior 없음** → HaMeR가 순간 "손 펴짐" 예측 시 통과. (라켓 grip은 sustained인데 그 사전지식 부재.)
- **구조적**: 이상치 탐지가 외부 정적 JSON(tennis 전용) → 배치는 아예 버림. `RESID/MEDW/σ` 매직넘버. 순차 .pt 파이프라인 → 단일 에너지로 trade-off 불가.

---

## 4. adaptive_v4 설계 (배치/서비스 + 오프라인 이점)

**설계 철학(한 줄)**: *주파수·축으로 분해 — swing 저주파=HaMeR, swing 고주파=제거,
axial=GVHMR(invisible이라 손해0), bad=downweight(교체 아님), 충돌=soft-hinge(평소0).*
이산 교체(region switch / slerp replace)를 버리고 **단일 robust 최적화 + 연속 가중**으로.

### 4.1 단일 손목 에너지 (batch_fit2d 대체)
```
E = w2d·L2d(robust, scale-norm)
  + wsw·L_swing_smooth(Huber, swing축만)      # 빠른 스윙 보존, jitter만
  + wax·L_axial_prior(GVHMR, 연속가중)         # invisible → fidelity 손해 0
  + wcol·L_collision                            # 평소 0, 침투 시만
  + wgrip·L_grip                                # 그립 펴짐 방지
  + wdep·L_depth_anchor(GVHMR cam-z)            # 충돌(a) 직격
```
- **swing 평활을 축 분리**: WSM=20 전축평활 → **swing축(Y,Z)만 Huber**. axial(X)은 따로.
  → 빠른 스윙 lag 제거(문제①-2). robust(Huber)라 스파이크는 죽이되 진짜 빠른 동작은 보존.
- **axial = GVHMR 연속 soft prior**: 가중치 `= sigmoid(HaMeR axial의 local-median 이탈 + (1−2D conf))`.
  이산 30° threshold의 **경계 튐 제거 + 일반화**. axial은 invisible이라 overlay 손해 0.

### 4.2 이상치/bad_frame을 온라인·연속으로 (외부 JSON 폐기)
- per-frame **confidence** = f(2D conf, reproj residual, wrist velocity) 자동 산출.
- robust loss가 자동 downweight + 이웃 prior로 채움. **SLERP 전면교체 금지** → 실제 손에서 덜 벗어남(문제①-1, ③).
- 일반화됨(클립별 동적, 매직넘버 제거).

### 4.3 Depth anchor — 충돌(a) + 전완 누수(①-3) 동시 완화 [최우선·최저비용]
- `L_depth = Σ_joint ρ(z_joint − z_gvhmr_orig)` (elbow·wrist cam-z, focal 동일 → 원본 z에 soft anchor).
- 단안 depth-flip 제거 → 팔이 몸통으로 파고드는 침투 직격.
- **주의**: GVHMR depth 자체가 살짝 틀릴 수 있으니 hard가 아닌 **soft(robust)** 로. 2D fidelity와 균형.

### 4.4 Collision term — 충돌(b: 팔-팔/손-손) [오프라인이라 감당 가능]
관측상 **사지 self-intersection**이 주이므로 inter-segment 우선:
- **경량(권장 시작)**: capsule–capsule 거리. 좌전완↔우전완, 좌손↔우손, (전완·손)↔몸통축 capsule.
  거리 `< margin`일 때만 soft-hinge penalty(평소 0). 빠르고 미분가능.
- **정밀**: SMPL-X 손/팔 vertices vs 상대편 vertices/몸통 **SDF penetration loss**. 무겁지만 확실.
- **detect→repair 2-pass(오프라인 강점)**: 1차 최적화 → 메쉬 교차/렌더로 침투 프레임 자동 검출
  → 그 프레임만 collision 항 켜고 재최적화. **HaMeR 최대 보존**(침투 없는 구간은 안 건드림).

### 4.5 Grip prior — 그립 펴짐 방지
- 손가락에 **mean-grip anchor**(시퀀스 median 또는 라켓 grip 레퍼런스) 약하게 + curl 하한.
- grip은 저주파 sustained라 안전. HaMeR의 순간 "펴짐" 아웃라이어를 끌어내림(문제③).

### 4.6 (선택) 절대 방향 fit — 전완 누수 차단
- 손목을 **절대(global) orientation으로 2D fit** 후 최종에 `R_local=G_elbow⁻¹G_wrist` 환원.
- 전완 jitter가 swing으로 새는 걸 차단. axial만 GVHMR local에서.

---

## 5. 오프라인(비실시간) 이점 활용

서비스가 실시간이 아니므로 **인과성(causal) 제약 없음** → 적극 활용:
- **양방향 전 시퀀스 최적화**(과거+미래 동시) — 이미 batch는 full-seq. robust outlier 판정에 미래 프레임도 사용.
- **detect→repair 다중 패스**(§4.4): 렌더/메쉬교차로 침투·스파이크 자동 검출 후 해당 프레임만 재최적화.
- **무거운 collision(mesh SDF)** 감당 가능.
- **render-in-loop 검증**: axial invisible 원칙상 수치만으론 판정 금지 → 자동 렌더 후 침투/lag 메트릭으로 자기 검증 루프.

---

## 6. 권장 우선순위 (배치/서비스 기준)

1. **Depth anchor (§4.3)** — 가장 싸고 충돌(a)·전완 누수(①-3) 직격. **여기부터.**
2. **swing/axial 축분리 + 연속가중 통합 최적화 (§4.1–4.2)** — WSM 전축평활 → swing Huber + axial 연속 prior. ①-2, ③ 해결, 일반화.
3. **Collision inter-segment (§4.4 경량)** — 팔-팔/손-손. 필요시 detect→repair.
4. **Grip prior (§4.5)** — 그립 펴짐.
5. (선택) 절대 방향 fit (§4.6), render-in-loop 자기검증 (§5).

## 7. 열린 질문
- swing/axial 분해를 batch에 넣을 때 forearm(G_elbow)을 매 프레임 GVHMR에서 읽어 분해 — 비용·정확도 확인.
- collision capsule 반경/margin 캘리브레이션(영상별 betas 의존).
- mean-grip 레퍼런스: 시퀀스 median vs 고정 라켓 grip pose 중 무엇이 일반화 좋은지.
- depth anchor robust ρ의 scale(GVHMR depth 신뢰도) 튜닝.

---

## 8. 실험 결과 (2026-06-03, front_3 F=1495)

검증 스크립트: `batch_fit2d_v4.py`(손목만), `batch_armwrist_v4.py`(전완+손목),
`render_ab_range.py`(A/B 구간 렌더, 프레임번호 burn-in).

### 8.1 ★ 핵심 발견 — 손목만 fit하면 인체불가 각도로 과굴절 (전완이 병목)
손 2D 재투영 RMSE(px, overlay 정확도 직접 지표) + 손목 물리:

| 방법 | R손(px) | L손(px) | 손목 ROM위반 | 각속도 스파이크(>33°/f) |
|---|---|---|---|---|
| GVHMR원본(_post) | 20.3 | 24.0 | 0 | 0 |
| fit2d (현 배포) | 10.5 | 11.5 | **R27/L79f** | 0 |
| v4-wristonly | 9.4 | 10.9 | **R54/L134f** | **R29/L28f** |
| **armwrist_v4** | **8.0** | **9.4** | **0** | **0** |

- ROM 위반은 거의 전부 **deviation(±45° 한계)·flexion(±90°)** = 보이는 swing축에서 발생.
  divergence 프레임 분해: fit2d는 f728/730 **flex=124°**, f852 **dev=86°**, f427 dev=60° 등 **인체 불가**.
  armwrist_v4는 같은 프레임에서 flex≈20°, dev≈−3° 자연.
- 원인: **고정된 GVHMR 전완이 살짝 틀려, 2D 손을 맞추려면 손목을 한계 너머로 비틀어야 함.**
  2D를 더 충실히 따를수록(v4-wristonly) 더 악화(L 134f).
- **→ 손목만으론 벽. 전완(collar/shoulder/elbow)을 같이 풀어야 함.** (사용자 직감 = 정답.)

### 8.2 armwrist_v4 = 셋 다 개선 (정확도·물리·안정 동시)
`batch_armwrist_v4.py`: 팔 chain rows 동시 최적화. 에너지 = body2D(ViTPose 어깨/팔꿈치/손목)
+ hand2D(WB 손목+5MCP, scale-norm gmof) + **GVHMR prior(L2,w=3) + temporal(geo,w=300) +
depth anchor(cam-z→GVHMR,w=50)**(=충돌/depth-flip 방지) + wrist swing Huber + axial→GVHMR
+ ROM soft-limit(dev45/flex90).
- **overlay 가장 정확(8.0/9.4px, 배포比 ~25%↓)** + 손목 물리위반 0 + 스파이크 0.
- "전완을 풀면 뻣뻣/부정확해질 것"이란 우려는 기우 — 전완이 제대로 일을 해 손목이 자연 중립이 되고
  결과적으로 손 추적이 **더** 정확. depth anchor+prior가 GVHMR 전완을 *soft하게* 잡아 충돌 방지.

### 8.2b ★ adaptive_v4 = 신뢰도-게이트 (사용자 핵심 요구 반영)
사용자 피드백(A/B 영상 후): "armwrist_v4가 더 안정적이지만, **갑자기 튀는 구간 빼곤 기존(fit2d)이 손을 더 잘 따라감.**
잘 보이는 구간엔 바짝 따라가되, 안 보이거나 튀는 구간엔 전완 추적으로 잡아둬라."
→ armwrist_v4를 전역 적용한 게 문제(anchor를 전역으로 세게 걸어 good 구간 충실도까지 죽임).

`batch_adaptive_v4.py`: **프레임별 trust 게이트**.
- `trust[f]`(per hand) = `fit2d.pt`(공격적 2D 추적 솔루션)가 그 프레임에서 **안 튀고(vel) + ROM 정상 + WB conf 높음**.
  → bad = `conf<0.4 | ROM위반 | (vel>28 & conf<0.6)` (속도는 저conf일 때만 bad → **진짜 빠른 스윙 보존**), dilate(2)+gaussian ramp.
- per-frame: `손2D가중 = W_HAND2D·trust` (good 구간만 손 바짝), `손목 full-local→GVHMR anchor = W_ANCH·(1−trust)` (bad 구간만 전완으로 잡음). 팔은 항상 body2D+prior+depth(안정 전완).
- 결과(front_3): R bad=52f/L bad=120f(=fit2d ROM위반 dilate, 진짜 spike/가림), 손목 ROM위반 0, vel max 4.

**good/bad 분리 RMSE(px)** — adaptive_v4의 의도 검증:

| | R good/bad | L good/bad |
|---|---|---|
| fit2d(배포) | 9.4 / 42.0 | 10.0 / 30.3 |
| armwrist_v4 | 7.0 / 36.9 | 8.4 / 21.5 |
| **adaptive_v4** | **6.2** / 47.3 | **7.0** / 40.8 |

- **good 프레임에서 최고(6.2/7.0, fit2d보다도 정확)** — 전완+손목 둘 다 쓰니 fit2d(손목만)보다 나음.
- bad 프레임 RMSE 높음은 **의도** — 거긴 2D 타겟이 노이즈라 안 쫓고 GVHMR로 잡음(=사용자 요구).
- 정지프레임 검증: good 구간(f518-522)=fit2d와 동일 추적, spike 구간(f743)=더 자연/안정.
- 튜닝 노브: `CONF_LO/VEL_BAD`(bad 판정 민감도), `W_ANCH`(bad 구간 잡는 세기), `W_HAND2D`(good 구간 추적 세기).
- 전체 A/B: `front_3/ab_full_fit2d_vs_adaptivev4.mp4`.

### 8.4 충돌 처리 (사용자 우선순위 전환 → collision first)
사용자 결정: 손목 추적은 fit2d가 더 자연스럽다(전완 안 움직여 팔 실루엣이 영상과 맞음) →
**손목/팔은 fit2d 유지, 충돌 처리로 전환.** 몸 충돌은 이 최적화 이전 과거 문제(현재 ≈0).
목표 = **손-손/손-팔 겹침**(작아도 "통과처럼" 보임). 예: front_3 ~9초(f504~586).

**검출기 `collision_detect.py`**: 스키닝 argmax로 부위 분리, 부호거리(정점이 상대표면 안쪽=관통)
+ **거리게이트(nearest<40mm)**로 false positive 제거(초기 219mm는 멀리 떨어진 팔의 오검출).
8클립 스캔(fit2d): 주 충돌 = **손-손/전완 겹침**(0~17%, ~16-26mm; side_1 18%, side_4 12%, front_3 9%).
팔→몸 관통은 front_1(1.9%,39mm)·side_3(2.7%,32mm)뿐 — **렌더 확인 결과 "쉬는중/걷는중 팔을 옆구리에 붙인 것"으로 시각적으론 정상.** 즉 진짜 거슬리는 건 손-손/전완 겹침(특히 포핸드에서 한 손은 자유인데 메쉬가 두 손을 붙여 통과처럼 보임).

**해결 `batch_collision_v4.py`**: fit2d 위 최소 보정. L팔↔R팔 정점 거리-마진 반발(MARGIN=15mm=손잡이반경,
선형 hinge **프레임당 위반량 합**, 양방향 대칭) + anchor(L2→fit2d) + 2D(가볍게) + temporal.
어깨/팔꿈치/손목(15,16,17,18,19,20) 최적화.
- ⚠️ 첫 시도 실패: 손실을 전체쌍 평균하면 희석돼 col=0(그래디언트 0). → **프레임당 위반 합**으로 고쳐야 작동.
- 결과(front_3): **손-손 9.0%→0%, 팔-팔 9.7%→0.1%** 근접쌍 9009→0. 2D손실 0.68→1.0(미미, anchor가 잡음).
- 산출: `collision_v4.pt`. A/B: `ab_collision_9s.mp4`(f480-620), `ab_full_collision_v4.mp4`.
- 튜닝: `MARGIN`(분리 간격), `W_COL`(반발), `W_ANCH`(fit2d 유지 세기).
- 남음: 모션 검증, 8클립 일반화, 서비스 배선(fit2d→collision_v4 단계 추가).

### 8.5 ★★ 통합 2D+collision (사용자 통찰 → 최종 후보) `batch_fitcol_v4.py`
사용자 통찰: "2D는 depth 모호(null space). 2D fit과 collision을 **한 최적화**에 넣으면,
옵티마이저가 충돌을 **2D-거의-불변인 깊이방향**으로 풀어 2D 손해 ~0." → post-hoc(8.4)보다 옳음.

검증: post-hoc collision_v4는 충돌은 0됐지만 **손 2D 악화(10.5→15.6px)** — 손을 평면으로 밀어서.
통합 fitcol_v4는: 손2D 강하게(주목적)+collision+GVHMR prior/temporal, 어깨/팔꿈치/손목 풀어
**깊이 분리 자유도** 확보, fit2d warm start.

**2D RMSE(px) — 결정적 비교 (충돌은 collision_v4/fitcol_v4 모두 0%):**

| | R손 | L손 | 팔 | 충돌 |
|---|---|---|---|---|
| GVHMR | 20.3 | 24.0 | 14.9 | — |
| fit2d(배포) | 10.5 | 11.5 | 14.9 | 9% |
| collision_v4(post-hoc) | 15.6 | 16.8 | 15.8 | 0% |
| **fitcol_v4(통합)** | **8.2** | **9.0** | 15.0 | **0%** |

- **fitcol_v4가 손 2D 최고(8.2/9.0, fit2d보다도↓)** + 팔 2D 유지(15.0=영상에 그대로) + 충돌 0.
- 충돌을 깊이 null-space로 풀어 2D 손해 0, 팔 자유도로 손이 2D에 더 잘 붙음. armwrist의 "팔 어긋남" 문제 없음(팔 2D 불변).
- 렌더(9s f509/540/557): fit2d는 전완 겹침 → fitcol_v4는 전완 분리되며 영상과 더 일치.
- 산출 `fitcol_v4.pt`, 영상 `ab_full_fitcol_v4.mp4`. **현재 최선 후보.**
- 튜닝: `W_HAND2D`(손추적), `W_COL/MARGIN`(분리), `W_PRIOR/W_TC`(팔 매끄러움·GVHMR 근접).
- 남음: 모션 전수검증, 8클립 일반화, 서비스 배선(`entry_point_hamer→fit2d→fitcol_v4`).

### 8.6 ⚠️ fitcol_v4는 "충돌 처리만"이 아니다 → `batch_collonly_v4.py`
사용자 지적: "fitcol_v4가 정말 충돌 처리만 한 게 맞나?" → **아니다.** 검증:
fit2d→fitcol_v4 관절 회전변화가 **충돌f와 비충돌f에서 동일**(어깨 8°, 팔꿈치 8°, 손목 ~20°, 비충돌 1346f도).
즉 fitcol_v4는 어깨/팔꿈치/손목을 **전 프레임 2D 재최적화** — 2D 개선(8.2)도 충돌이 아니라 이 전역 re-fit 때문.
사용자가 원한 건 **fit2d 유지 + 충돌만**.

**`batch_collonly_v4.py` = 진짜 충돌 처리만**: 충돌 프레임(json∪dilate3)만 풀고 나머지는
per-frame anchor로 fit2d에 **잠금**(W_LOCK=3000 vs W_FREE=15). 손2D 강(평면이탈 방지→분리는 깊이로)+collision+temporal.
- **비충돌 1346f 변화 ≈ 0**(팔꿈치 0.16°, 손목 0.08°) = fit2d와 사실상 동일.
- **충돌 149f만** 최소 변화(팔꿈치 4.5°, 손목 5.6°)로 깊이 분리. 충돌 0%(손-손/팔-팔).
- 손 2D 보존: R 10.5→10.1, L 11.5→10.8 (fit2d 동급, 살짝↑).
- 산출 `collonly_v4.pt`, 영상 `ab_full_collonly_v4.mp4`.

**세 방식 비교 (목적에 따라 선택):**
| | 손2D R/L | 충돌 | fit2d 보존? |
|---|---|---|---|
| collision_v4 (post-hoc) | 15.6/16.8 | 0 | 부분(2D 악화) |
| fitcol_v4 (통합 전역) | 8.2/9.0 | 0 | ✗ 전 프레임 재fit |
| **collonly_v4 (충돌만)** | 10.1/10.8 | 0 | ✓ 비충돌 거의 그대로 |

→ "fit2d 유지+충돌만"이면 **collonly_v4**, "전반적 2D까지 더 좋게"면 fitcol_v4. 사용자 요구=전자.

### 8.7 z-anchor 실험 (사용자 아이디어: 2D+GVHMR z소프트앵커+temporal) `batch_zanchor_v4.py`
사용자: "GVHMR z를 소프트하게 따라가며(prev/next gradient) 2D 따라가는 게 맞지 않나?"
→ depth anchor(§4.3)의 직접 구현. **(A) 순수 z앵커 vs (B) z앵커+충돌항** 비교 (front_3):

| 구성 | 손-손 충돌 | 팔-팔 |
|---|---|---|
| fit2d | 9.0% | 9.7% |
| (A) 순수 z앵커 | **11.6%** | 11.8% |
| (B) z앵커+충돌항 | **0%** | 0% |

★ **핵심(negative) 결과: z앵커 단독으론 충돌 안 줄어(오히려↑).** 이유: 충돌이 **GVHMR 깊이 자체에 내재**(fit2d 손위치=GVHMR 팔) → z를 GVHMR에 앵커하면 *겹친 깊이 유지*하고 분리를 **저항**. 충돌 제거엔 **명시적 충돌항 필수**.
- z앵커 = 좋은 *정규화*(2D-fit이 새로 만드는 depth-flip 예방, 일반화) 이지만 ≠ 충돌제거.
- (B) z앵커+충돌(W_Z=80)은 결국 **fitcol_v4와 동일 거동**: 손2D 8.3/9.0(↑) BUT **손목 평균 20.3° 변함**(전역 재fit). z앵커가 2D항을 못 이겨 fit2d 보존 실패.

**최종 트레이드오프 (front_3, vs fit2d):**
| | 손2D R/L | wrist변화 | 충돌 | fit2d유지 |
|---|---|---|---|---|
| collonly_v4 | 10.1/10.8 | 1.1° | 0% | ✓ |
| zanchor_B / fitcol_v4 | 8.3/9.0 | 20-23° | 0% | ✗ |

→ **(가) fit2d 유지+충돌만 = collonly_v4** / **(나) 2D↑ 전역재fit = fitcol_v4**. 사용자는 과거 전역재fit(armwrist)을 "별로"라 함 → (가) 선호 추정. **모션 렌더로 사용자 최종 결정 대기**(`ab_full_collonly_v4.mp4` vs `ab_full_fitcol_v4.mp4`).

### 8.8 collonly_v4 떨림 제거 (사용자 지적: 충돌경계 바들바들) — 확정 버전
사용자 결정: **collonly_v4 채택**(=fit2d 이미 GVHMR z 따라가니 "fit2d+z+충돌"=충돌만; zanchor는 전역재fit이라 0:04 무충돌인데도 손목 26~40° 달라짐을 사용자가 포착).
사용자 지적: 충돌 경계에서 손목이 fit2d↔보정 계단식 전환 → 바들바들.

**원인**: binary anchor(LOCK3000↔FREE15) 경계 불연속 + 충돌구간 내 per-frame 반발 jitter.
**해결 2가지**(`batch_collonly_v4.py`):
1. anchor를 **로그공간 gaussian ramp**(RAMP_SIGMA=3): 충돌=자유, 양옆 서서히 잠금복귀 → 보정 fade in/out.
2. **보정량(delta=collonly−fit2d) 시간축 gaussian smooth(sigma=2)**: 충돌구간 내 jitter까지 제거. 비충돌 delta≈0이라 무영향.
+ W_TC 250→400, MARGIN 15→20mm(평활로 줄어드는 peak 보상).

**결과(front_3):** 떨림 acc R/L 0.35/0.43 = **fit2d와 동일(급증 0f)** + 비충돌 변화 0.14° + 충돌 9%→**0.5%**(8f, ≤14mm 얕은 잔여). 마지막 0.5%를 0으로 밀면 떨림 재발 위험 → 현 지점이 떨림0+충돌≈0 균형.
산출 `collonly_v4.pt`, 영상 `ab_full_collonly_v4_smooth.mp4`. **= 확정 후보.**
남음: 8클립 일반화(side_1/4 충돌 많음, 각 클립 collision_detect→collonly), 서비스 배선.

### 8.9 ★★★ 사용자 핵심 아이디어 = 팔 IK (GVHMR 손목위치 틀림→HaMeR 위치 추종) `batch_armik_v4.py`
사용자(방법론 확정): "GVHMR이 손목 위치를 주는 게 틀렸으니 **HaMeR(관측) 손목 위치를 따라가고**, 그 위치에
**자연스러운 팔꿈치·팔 각도가 나오게 IK 최적화**하라." (이전 '위치 고정' 입장을 뒤집음.)

**= 역기구학(IK) 문제.** 구현 = 기존 `cap_arm_prefit.py`(이미 튜닝됨) 재사용:
- Stage A `cap_arm_prefit.py --input-dir <clip>`: 어깨/팔꿈치/손목을 ViTPose 2D에 IK + GVHMR-자연prior(L2)
  + temporal(geo 1e3). → 손목 위치 16.9→**3.3px**. (배치 클립에서 잘 돎.)
- Stage B `batch_armik_v4.py`: `_post`(HaMeR 손가락) 위에 arm_prefit의 collar/shoulder/elbow 주입,
  손목 회전만 WB 손2D에 fit(WSM=20). → 손 방향 얹기.

**결과(front_3, vs fit2d):**
| | 손목위치 | 팔꿈치 | 손방향 | 손목속도 |
|---|---|---|---|---|
| GVHMR/fit2d | 14.0 | 17.7 | 11.0 | 16 |
| **armik_v4** | **1.1** | **1.4** | 10.5 | 15 |

- **손목 위치 14→1.1px**(GVHMR 틀린 위치 → 관측 위치로 교정) + 팔꿈치 1.4px(자연 IK) + 손방향 유지 + 매끄러움(15°/f).
- 렌더(f240/557/852/1024): 팔 자연스럽고 손/팔이 영상 위치에 정확. **예전 armwrist와 차이**: armwrist는 손목 위치를 안 고쳤음(15px 그대로, 헛움직임)→ "별로". armik는 위치 교정이 목적(1.1px)이라 다름.
- ⚠️ HaMeR 3D 깊이는 직접 사용 X(과거 z-anchor 효과미미). 위치 타겟=2D(ViTPose 손목 conf93%), 깊이=GVHMR-자연prior로 가지 선택.
- 산출 `armik_v4.pt`, 영상 `ab_full_armik_v4.mp4`. **= 사용자 아이디어의 직접 실현, 강력한 후보.**
- 파이프라인: `entry_point_hamer → cap_arm_prefit(StageA) → batch_armik_v4(StageB) → collision_detect → batch_collonly_v4(충돌)`.
- 남음: 모션 검증, 충돌 패스 재적용(arm 위치 바뀌었으니 재검출), 8클립 일반화, 서비스 배선.

### 8.10 ★★★ v4 FINAL 파이프라인 (front_3 완성 시도) — 4-stage
"오늘 front_3 하나를 완벽하게" 목표. 사용자 IK 아이디어(8.9) + Dyn-HaMR 충실 swing + 스파이크/충돌 정리.

**파이프라인:**
1. `cap_arm_prefit.py --input-dir <clip>` → `smplx_arm_prefit.pt` — 팔 IK(어깨/팔꿈치/손목→ViTPose2D + GVHMR자연prior + temporal). 손목위치 16.9→**3.3px**.
2. `batch_armik_faithful.py <clip> 2000` → `armik_faithful.pt` — 교정팔 위에 **Dyn-HaMR 충실 손목**(scale-norm gmof 2D + j3d_smooth) + WSM=2000 평활. swing 9.0px.
3. `batch_spike_clean.py <clip> 33` → `armik_final.pt` — 손목속도>33°/f(비물리적, 대부분 invisible axial 노이즈) **slerp 보간 + 경계 평활**(generalized 03_badframe). swing 9.4px, 속도 ≤33, 스파이크 **0f**.
4. `collision_detect.py <clip> armik_final.pt` + `batch_collonly_v4.py <clip> 250 armik_final.pt` → `armik_final_col.pt` — 충돌 7.6%→**2.3%**.

**vs fit2d(배포):**
| | 손목위치 | swing | 속도스파이크 | 충돌 |
|---|---|---|---|---|
| fit2d | 14.0px | 11.0 | 낮음 | 9% |
| **v4 FINAL** | **1.1px** | **9.4** | **0f** | **2.3%** |

- 영상 `ab_full_FINAL.mp4`.
- ⚠️ **약점=충돌 잔여 2.3%**: armik이 팔을 2D 정위치로 옮기며 좌우 팔이 더 깊이(35mm) 겹침 → collonly(delta평활)가 완전히 못 풀음. 더 조이려면 collonly MARGIN/W_COL↑ 또는 delta평활↓(떨림 주의). 
- 스크립트: `cap_arm_prefit / batch_armik_faithful / batch_spike_clean / collision_detect / batch_collonly_v4`(BASE 인자로 임의 pt 충돌처리 가능).

### 8.11 ★★★★ v4 CLEAN 파이프라인 (사용자 4점 피드백 전부 해결) — 확정
사용자 피드백(v4 FINAL 후): #1 손정합 fit2d 근소우위, #2 충돌 손에서 심함→투영 안벗어나는 선에서 손목각도로 제어,
#3 slerp 구간 명시, #4 떨림 존재→원인 서술.

**진단(#4 떨림 = 두 원인):** 단계별 mean-acc로 특정 —
- 팔 떨림: GVHMR/fit2d 0.35/0.54 → arm_prefit(StageA) **1.62/2.91**. 원인=ViTPose 2D를 프레임별로 따라가 ViTPose jitter 상속(fit2d는 GVHMR 팔 고정이라 매끄러웠음).
- 손목 떨림: arm_prefit 0.28 → armik_faithful(StageB) **5.83**. 원인=faithful scale-norm 2D가 손 2D를 너무 충실히 추종(평활 약함).
→ **faithful 우회가 #1·#4 둘 다의 원인.** 버림.

**해결 = CLEAN 3-stage:**
1. `cap_arm_prefit.py --w-tc 5000 --target-px 3` + **arm 회전(collar/sh/elbow) 시간축 gaussian σ1.5** → 손목위치 3.5px, 팔 떨림 0.24~0.47(GVHMR 이하).
2. `batch_armik_v4.py`(WSM raw 2D 손목, =fit2d 정합도) → 손정합 10.6px(≥fit2d 11.0), 손목떨림 0.32(≤fit2d).
3. `batch_axcol_v4.py`(#2: 손목만 최적화+강 hand2D+collision → invisible **axial 회전**으로 분리, delta 평활) → 충돌 7.5%→**1.5%**, 손정합 10.6→10.8(보존), 떨림 보존.

**결과(front_3) — 4점 전부 해결:**
| 항목 | fit2d | v4 CLEAN |
|---|---|---|
| 손목 위치 | 14px | 3.5px |
| #1 손 정합 | 11.0 | 10.6 (≥fit2d) |
| #2 충돌 | 9% | 1.5% (axial해소, 2D보존) |
| #3 slerp떨림 | — | 없음(faithful 폐기) |
| #4 떨림 | 0.35 | ≤fit2d 전 관절 |

- 산출 `armik_axcol.pt`, 영상 `ab_full_v4clean.mp4`. **= 확정 파이프라인.**
- #3 참고: faithful판의 spike-clean slerp 구간(이제 미사용) R 109f/L 85f은 batch_spike_clean 로그에 기록.
- 잔여 충돌 1.5%(≤17mm)=axial로 못 푸는 palm/wrist 위치겹침(소수, 그립자세). 더 줄이려면 미세 depth 허용 또는 MARGIN↑.
- 파이프라인: `entry_hamer → arm_prefit(w_tc5000)+arm평활 → batch_armik_v4 → axcol`. 남음: 8클립 일반화·서비스 배선.

### 8.12 손가락 2D fit (사용자 피드백 #2 손가락 안맞음 / #4 19초 털기) `batch_finger_fit.py`
사용자(v4_clean 후): #1 팔꿈치 약간 안맞음, #2 손가락이 실제 HaMeR 손과 안맞음, 손목위치 완벽, #4 19초쯤 오른손 "털기".
진단: #2·#4 둘 다 **손가락 pose**(지금까지 _post HaMeR 그대로, 2D fit 안함). 손가락 변화 mean 0.29 max 1.40rad/f,
f1176(≈19.6s) 1.23rad spike = "털기". WB 손 21점 conf 88~92% → 손가락 2D fit 가능.
#1 팔꿈치는 사실 이미 2.7px(fit2d 19px의 7배 정확, 빗나감 0~1f) — 시각상 "off"는 극소수f or depth.

**`batch_finger_fit.py`**: smplx 손가락 joint(L25-39/R40-54, 3/finger: thumb/index/middle/pinky/ring 순) ↔ WB 21점(MCP/PIP/DIP) 매핑. 손가락(hand_pose 45)만 최적화. scale-norm gmof 2D + HaMeR prior(그립유지) + temporal + 후 gaussian평활(σ1.5).
- 가중치: W2D=3, W_PRIOR=15, W_TC=800 (2D 노이즈 vs 평활 균형; 약하면 손가락 flail).
- 결과: 손가락 2D **21.3→13.6px**, 변화 mean 0.29→**0.21**(HaMeR보다 부드러움), f1176 1.23→**0.35**(털기 제거), >0.8rad 33→5f.

**= 완성 v4 4-stage 파이프라인:**
`arm_prefit(w_tc5000)+arm평활σ1.5 → batch_armik_v4(WSM손목) → batch_axcol_v4(axial충돌) → batch_finger_fit`
산출 `armik_finger.pt`, 영상 `ab_full_v4final.mp4`. 사용자 4점(#1 팔꿈치 OK, #2 손가락, #3 slerp없음, #4 떨림/털기) 처리.

### 8.13 ★★★★★ HaMeR 손목 + 신뢰도 게이트 (사용자: 손가락은 HaMeR 각도 따라가야) — 유력 최종
사용자 핵심 정정: 손목을 WB 2D에 fit한 게 아니라 **HaMeR 손목 각도**를 따라가야 함(그게 "완벽한 overlay").
측정: Y(WB-fit) vs HaMeR 손목 = 전체 66/75° 차이(swing 16~19°, **axial 55~66°**). → Y는 HaMeR가 아니었음.

**`batch_hamer_wrist.py`**: 팔=arm_prefit(IK), 손가락=_post(HaMeR), **손목=HaMeR 글로벌→local(`G_elbow⁻¹@HaMeR_wrist`)**.
- ⚠️ 시행착오: axial을 전완(GVHMR)으로 *교체*(AXIAL=gv) → 66° 비틀려 손 망가짐. **교체 아니고 HaMeR 그대로(AXIAL=hamer)가 맞음**(ROM위반도 R38→R5로 적음).
- 순수 HaMeR(AXIAL=hamer): 손 잘 보이는 구간 **완벽**(사용자 확인), but 가려짐/스윙 뭉개짐 구간 실패(HaMeR 검출 불량).
- v4_clean(WB-fit): 나쁜 구간 OK, 좋은 구간 못 따라감. → **둘의 장점 합치기 = 신뢰도 게이트.**

**신뢰도 게이트** (batch_hamer_wrist에 추가): bad = (HaMeR 손목 속도>33 비물리적) ∪ (**WB 손 conf<0.55 = 가려짐/뭉개짐**), dilate2 → 양옆 good HaMeR에서 **SLERP coast** + 경계 가우시안 블렌딩.
- 결과(front_3): bad R336/L333f(22%), 속도 max27/32 스파이크0, ROM위반 R0/L20f.
- **좋은 구간(78%)=순수 HaMeR(완벽) + 나쁜 구간(22%)=good 이웃 slerp(부드러운 coast)** = 사용자 요구 "둘 다".
- 산출 `hamer_wrist_hamer.pt`, 영상 `render_hamer_gated.mp4`. CONF_TH=0.55 튜닝가능.

**현재 유력 손목 파이프라인:** `arm_prefit(w_tc5000)+arm평활 → batch_hamer_wrist(HaMeR swing+axial, conf게이트) → [충돌 axcol]`.
남음: 사용자 모션 검증, ROM 클램프(L20f), 충돌 재적용, 8클립.

### 8.14 jitter 평활 + ROM 클램프 + 충돌 단계분리 (사용자 요청)
- **jitter**: 게이트HaMeR가 HaMeR per-frame 노이즈로 떨림 acc 2.5(fit2d 0.35). → 전역 quat gaussian **σ1.8** → acc 0.45(fit2d급), HaMeR 이탈 3°(제거된 노이즈만). `batch_hamer_wrist.py` SMOOTH 인자. 속도 max 11/13.
- **ROM 클램프** `batch_rom_clamp.py`: 손목 dev±45/flex±90/axial(우[-90,160]좌반대) 넘는 프레임만 클램프+경계평활. R16/L11f 클램프 → 위반 R0/L8f. 산출 `<base>_rom.pt`.
- **충돌 재적용** `batch_axcol_v4.py`(BASE 인자화): ROM 결과 충돌 4.3%→0.7%(axial 분리). 산출 `<base>_col.pt`.
- 단계분리 렌더: `render_step1_rom.mp4`, `render_step2_rom_col.mp4` (문제 격리용).
- ⚠️ axcol의 W_HAND2D=15가 HaMeR 손목을 WB쪽으로 당길 수 있음 → HaMeR 기반엔 충돌프레임만 영향(비충돌은 anchor lock). 검증 필요.

### 8.16 ★ 손목 위치 깜빡임 = arm_prefit 깊이 미앵커 (사용자가 정확히 지적) → depth anchor로 해결
사용자: "손목 위치가 잠깐 깜빡이는데 원래(fit2d) 없었다" → 측정: 우리 최종 L손목이 fit2d(GVHMR팔)比 평균5cm/최대23cm 어긋, f685 급변 9cm/f 등 = **arm_prefit이 ViTPose 2D fit하며 깊이 미앵커라 단안 모호 프레임서 z가 GVHMR에서 벗어났다 돌아옴(깜빡)**. fit2d엔 없음(GVHMR팔 직접). 제 "GVHMR 문제" 진단은 틀렸음.
**해결 = §4.3 depth anchor**: `cap_arm_prefit.py`에 `CAP_ARM_DEPTH` env(기본0=off) 추가 — 팔관절 cam-z를 GVHMR에 soft 앵커(`L_depth=mean((z-z_gv)²)`). **CAP_ARM_DEPTH=100000** → f684 깜빡 16→4cm, 최대급변 9.0→3.8cm, 팔2D 3.5→4.5px(거의유지). 영상 `render_depthfix.mp4`.
**확정 arm 생성**: `CAP_ARM_DEPTH=100000 cap_arm_prefit.py --w-tc 5000 --target-px 3` + arm회전 gaussian σ1.5.
※ z-깊이 스윙(0.94m)은 GVHMR도 동일=진짜 모션(arm_prefit이 추가한 건 per-frame jitter뿐, gaussian이 제거). Kalman도 같은 일(노이즈제거)이라 실제모션은 못 줄임(=lag).

### 8.15 그립 prior (사용자: 빼지 말 것 — 손 그립 갑자기 펴짐)
> ⚠️⚠️ **절대 규칙(사용자 강조): 그립 건드릴 땐 오직 손가락(left/right_hand_pose)만. 손목·전완·body_pose 전부 금지. 무조건 손가락만.**
손가락 grip이 갑자기 펴지는 구간 방지. §4.5 참고: mean-grip(시퀀스 median 또는 라켓 grip 레퍼런스) 약한 anchor + curl 하한 + 손가락 temporal. 현재 손가락=_post(HaMeR) 고정이라 HaMeR가 순간 펴짐 예측하면 통과. ROM/충돌 단계 후 추가 예정.

### 8.3 남은 검증/할 일

### 8.3 남은 검증/할 일
- **모션 확인 필수**(정지프레임은 둘 다 비슷·plausible로 보임): `ab_full_fit2d_vs_armwristv4.mp4`(front_3 전체 A/B) 스크럽.
- 충돌(팔-팔/손-손)은 아직 inter-segment 항 미적용 — armwrist_v4는 depth anchor로 몸통침투만 완화. §4.4 capsule 추가 필요.
- 가중치 일반화: front_3 단일 검증. front/side 8클립 + betas 다양성에서 W_DEPTH/W_PRIOR/ROM 캘리브.
- 그립 펴짐(§4.5) 미적용. 손가락은 여전히 _post 고정.
- 서비스 배선: `batch_fit2d` → `batch_armwrist_v4`로 교체 검토(단, 일반화 검증 후).

> 관련: `WRIST_PIPELINE.md`, `physical_limits.py`, 메모리 `final-wrist-pipeline`, `batch-generalization`, `motio-dynhamr-comparison`, `adaptive-v4-design`.
</content>
</invoke>
