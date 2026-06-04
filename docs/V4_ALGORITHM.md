# adaptive_v4 — 핵심 알고리즘 & 최적화 수식 (2026-06-03)

> 단안 테니스 영상에서 손목/손/팔을 영상에 극한까지 정합시키는 후처리 파이프라인.
> **상태: front_3에서 확정. 8클립 일반화 비교 후 가중치/임계가 수정될 수 있음.**
> 발전 서사 = [V4_JOURNEY.md](V4_JOURNEY.md), 실험 로그 = [ADAPTIVE_V4_DESIGN.md](ADAPTIVE_V4_DESIGN.md).

---

## 0. 문제 정의

세 모델의 출력을 융합해 SMPL-X 한 벌로 만든다:
- **GVHMR** — 전신 시계열 SMPL-X (몸/팔이 안정적, 가려짐에 강함, but 손목 *위치*가 부정확).
- **HaMeR** — 단일 손 전문가 MANO (잘 보이면 손/손목 각도가 완벽, but 프레임별 독립 → 지터, 가려짐/겹침에 환각).
- **ViTPose** — COCO17(17점, 팔/손목) + WholeBody(133점, 손가락). 프레임별·관절별 **신뢰도(conf)** 보유.

핵심 통찰(사용자): **"GVHMR 손목 위치가 틀렸다 → HaMeR/관측 손목을 따라가되, 자연스러운 팔각도가 나오게 역기구학(IK)으로 풀자."**
→ 손목 *위치*는 ViTPose 2D로 IK, 손목 *각도*는 HaMeR, 팔 깊이/안정은 GVHMR, 손가락은 HaMeR + 신뢰도 게이트.

기호: `J` = SMPL-X joint(3D, cam), `proj(·)` = 핀홀 투영, `θ` = axis-angle, `R` = 회전행렬, `q` = quaternion.
`conf_{f,j}` = ViTPose 프레임 f·관절 j 신뢰도. `z(·)` = cam-z(깊이).

---

## A. 팔 IK + 안정화 — `cap_arm_prefit.py`

**최적화 변수**: 양팔 8관절(collar·shoulder·elbow ×2)의 회전 `θ_arm`. (Adam, early-stop)

| 항 | 수식 | 역할 |
|---|---|---|
| `L_2d` | `mean_f Σ_j conf_{f,j} · ‖proj(J_{f,j}) − vp_{f,j}‖²` (j=어깨/팔꿈치/손목×2) | 손목·팔꿈치를 **관측된 ViTPose 2D 위치**로 IK. **conf 가중**이라 블러/가려짐 프레임은 자동 약화. 손목은 base 0.6 + outlier 게이팅으로 더 약하게. |
| `L_prior` | `mean_f w_prior(f) · ‖θ_arm(f) − θ_arm^{GVHMR}(f)‖²` | 팔을 **GVHMR 자세로 끌어당김**(2D가 못 정하는 자유도를 자연스럽게). |
| `L_tc` | `geodesic(R_arm[:−1], R_arm[1:])` | 인접 프레임 **시간 평활**(ViTPose jitter 억제). |
| `L_depth` | `mean ‖z(J_arm) − z^{GVHMR}_arm‖²` | **깊이 앵커**: 팔관절 cam-z를 GVHMR에 soft 고정. 단안 깊이 모호로 인한 **위치 깜빡임 제거**(2D는 보존, 깊이 방향만). |

**총손실**: `L = L_2d + L_prior + w_tc·L_tc + W_DEPTH·L_depth`  (front_3: `w_tc=5000, W_DEPTH=100000, target_px=3`)

> 핵심 원리 — **단안에서 2D는 깊이를 모른다**. 그래서 위치 교정(2D)과 안정(깊이 앵커·prior)을 직교 방향으로 나눠 서로 손해 없이 푼다. 손목 위치 14px→3.5px, 깊이 깜빡임 9cm→3.8cm.

---

## B. 손목 = HaMeR + 신뢰도 게이트 — `batch_hamer_wrist.py`

손목 회전을 **swing-twist 분해**로 둘로 나눈다: `q_wrist = q_swing · q_twist(x축=전완 축)`.

- **swing(편위+굴곡) = HaMeR** : `q_swing ← swing( G_elbow⁻¹ · R^{HaMeR}_{wrist,global} )` — "잘 보이면 완벽한" HaMeR 손목 각도를 직접 이식(WB 2D fit 아님; WB는 HaMeR와 swing 16°/axial 55° 다름).
- **twist(축 회전) = 전완(arm_prefit/GVHMR)** : 2D에 안 보이는 축은 물리적 전완값 유지(HaMeR axial 교체하면 66° 비틀려 망가짐).

**신뢰도 게이트**(손별): `bad_f = (각속도 > 33°/f) ∪ (conf^{WB손}_f < 0.55)` → dilate2 → bad 구간은 양옆 good HaMeR에서 **slerp coast** + 경계 평활.
**전역 평활**: quaternion gaussian `σ=1.8` (HaMeR 프레임지터 제거, 실모션 이탈 3°).

> 역할 — 잘 보이는 구간(완벽 HaMeR) + 가려짐/스윙뭉개짐(부드러운 coast)의 **장점 둘 다**. HaMeR는 시계열 모델이 없어 6.7Hz 지터·80°/f 급변이 있는데, σ1.8 평활이 절반으로 줄임.

---

## C. ROM 클램프 — `batch_rom_clamp.py`

손목 local 회전을 (편위 dev, 굴곡 flex, 축 axial)로 분해 → `dev∈±45°, flex∈±90°`로 clamp.
**역할**: 해부학적 손목 한계 강제(안전캡). 비위반 프레임 불변.

---

## D. 충돌 해소 (축 회전) — `batch_axcol_v4.py`

**최적화 변수**: 손목 [19,20] 회전만. (다른 모든 것 고정)

| 항 | 수식 | 역할 |
|---|---|---|
| `L_hand2d` | scale-normalized `gmof(‖proj(J_손)−WB_손‖²)`, conf² 가중 (W=15, 강) | 손 2D **투영·swing 보존** → 옵티마이저가 2D 안 건드리는 축으로만 움직이게 강제. |
| `L_col` | `mean_f Σ relu(MARGIN − min‖V^L_손 − V^R_손‖)` (MARGIN=18mm) | 손-손 메쉬 **관통을 프레임당 위반합**으로 밀어냄(평균 쓰면 희석돼 grad=0 — 주의). |
| `L_anch` | `mean (anch_w · ‖θ_wrist − θ_wrist^0‖²)`, `anch_w`=비충돌 3000→충돌 8 로그램프 | **비충돌 프레임 잠금**, 충돌 프레임만 자유. |
| `L_tc` | `‖θ_wrist[1:]−θ_wrist[:−1]‖²` | 시간 평활(충돌 경계 떨림 방지). |

**총손실**: `L = 15·L_hand2d + 300·L_col + L_anch + 300·L_tc`. bad는 dilate6 + delta σ4 **전파**.

> 핵심 원리 — 충돌을 **2D에 안 보이는 axial(축) 회전**으로 푼다. 손을 평면으로 밀면(post-hoc) 2D가 깨지지만, 축으로 돌리면 overlay 손해 0 + 충돌 9%→1%.

---

## E. 그립 게이트 (손가락만) — `batch_grip_prior.py`

**⚠️ 절대 규칙: 손가락(`left/right_hand_pose`)만 수정. `body_pose`(손목/전완/팔) 불변 — assert로 보장.**

손별 `bad`:
```
bad_f = (conf^{WB손}_f < 0.7, dilate2)                      # 순수 가려짐
      ∪ (손영역겹침 IoU_f > 0.1  AND  이 손이 뒤(cam-z 큼))   # 겹침: 뒤 손만
```
- **고conf** → HaMeR 그립 그대로 (잘 봄).
- **bad** → 양옆 good(비겹침·고conf) 프레임 그립으로 **선형보간 홀드** + per-frame 스파이크 보간 + σ1.5 평활.
- **겹침 시 앞 손(cam-z 작음, 잘 보임)은 HaMeR 유지**, 뒤 손(HaMeR가 앞손 따라가 망가짐)만 교정.

| 신호 | 계산 | 왜 |
|---|---|---|
| 손영역 IoU | WB 손 21점(conf>0.3) bbox 겹침 | **오염 안 됨**(손목·손 위치는 신뢰). conf는 겹쳐도 0.9 유지(과신)라 못 씀. |
| 앞/뒤 | SMPL-X 손목 joint cam-z 비교 | 팔(body_pose) 기반이라 손가락 오염과 무관. |

> 역할 — HaMeR 그립 지터·겹침환각을 신뢰도·기하로 게이트. IoU는 기하 기반이라 **8클립 일반화 기대**.

---

## 최종 파이프라인 & 파라미터 (front_3)

```
영상 → [전처리] GVHMR(hmr4d) + HaMeR(hamer_out_clean) + ViTPose(COCO17, WholeBody)
  A  cap_arm_prefit.py   --w-tc 5000 --target-px 3,  env CAP_ARM_DEPTH=100000   → smplx_arm_prefit.pt
  B  batch_hamer_wrist.py  <clip> hamer 1.8                                      → hamer_wrist_hamer.pt
  C  batch_rom_clamp.py    <clip> hamer_wrist_hamer.pt                           → …_rom.pt
  D  collision_detect.py + batch_axcol_v4.py <clip> 250 …_rom.pt                 → …_rom_col.pt
  E  batch_grip_prior.py   <clip> …_rom_col.pt 0.7 1.5 0.1                       → …_rom_col_grip.pt  (최종)
```
> 주의: arm_gate(가려짐 팔 게이트)는 **제거됨**(v4_10). un-gated IK가 이미 GVHMR≈4~7cm라, 회전 blend가 손목 위치를 비선형으로 17~34cm 밀어내 오히려 악화시켰음. 가려짐 팔은 GVHMR 충실로 둠.

## 설계 원칙 (요약)
1. **2D는 깊이를 모른다** → 위치(2D)와 안정(깊이/축)을 직교 분해해 손해 없이 푼다 (A의 depth anchor, D의 axial 충돌).
2. **소스별 강점**: 위치=ViTPose IK, 각도=HaMeR, 깊이/가려짐=GVHMR, 손가락=HaMeR+게이트.
3. **신뢰도·기하 게이트**: conf는 가려짐엔 정직하나 겹침엔 과신 → 겹침은 IoU(기하)로 잡는다.
4. **국소·가역**: 비대상 프레임 불변(anchor lock/assert), best는 덮어쓰지 않고 버전 누적.
5. **렌더 검증 필수**: axial 등 2D-불변 변화는 수치만으론 판정 금지 — 항상 렌더로 확인.

## 알려진 한계 (일반화 시 점검)
- 임팩트 스윙(예 19.8초)의 손목 "확 변함"은 **85% 진짜 모션**(GVHMR 일치). ViTPose 블러 스냅(15%, ~4px)은 제거해도 미미 → 근본해결=라켓 비전.
- 가중치·임계(`w_tc, W_DEPTH, conf 0.55/0.7, IoU 0.1, MARGIN 18mm`)는 front_3 튜닝값. **betas/카메라/충돌량 다른 8클립에서 재검증 필요**.
