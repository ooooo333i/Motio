# Poster Content — Physically-Plausible HaMeR Hand–Body Integration

> 작성 2026-06-03. 숫자 출처 = `outputs/metrics_8clip_FINAL.json`, 수식·근거 = [METRICS_FORMULAS.md](METRICS_FORMULAS.md).
> **정직 원칙**: 모든 숫자는 재현가능한 파이프라인(`hamer_wrist_hamer_rom_col_grip.pt`) 출력. (실험본 `adaptive_v4.pt`의 0.877 PCK는 사용 금지 — 1클립·비재현.)

---

## 1. Title (안)
**Twist-Aware HaMeR Integration: Physically-Plausible Hand–Body Recovery from Monocular Video**

## 2. Problem
- 단안 영상에서 손(HaMeR/MANO)을 몸(SMPL-X/GVHMR)에 결합하는 표준 방식 = **naive "Copy-Paste" graft**.
- 그 결과 **부자연스러운 전완 비틀림(forearm twist)** 과 **손-몸/손-손 충돌(penetration)** 발생 (→ PyMAF-X가 문서화한 알려진 문제).
- before/after GIF 1컷 (비틀림·관통 사례).

## 3. Contribution (3줄)
1. **핵심 아이디어 — 전완축 기준 손목각 제한**: 손목 회전을 전완(elbow→wrist) 축에 대해 제한 → universal graft와 **동일한 2D 정합 유지**하면서 **전완 비틀림 절반(front_3 4.0→1.9%)·충돌 −55%**.
2. **무손실 + 순환성 폭로**: 손 2D는 독립 MediaPipe서 baseline 동등(8클립). 기존 WB 지표는 2D-fit에 과적합한 순환 지표임을 정량 폭로(adaptive 0.90→0.46).
3. **가려짐 강건 그립**: IoU 겹침 게이트 + 신뢰도 폴백으로 occlusion 환각 억제 (body_pose 불변).

## 4. Method (pipeline 그림)
```
HaMeR hands ─┐
GVHMR body ──┼─▶ A. arm_prefit (depth-anchor IK, 손목 3.5px)
             │   B. hamer_wrist (HaMeR 손목, conf 게이트)
             │   C. ROM clamp (생체역학 한계)
             │   D. axcol (충돌 해소 — 축분리: 위치 보존)
             │   E. grip prior (손가락만, IoU/conf 게이트)
             └─▶ + final temporal smoothing (Gaussian σ=1)
```
### ★ Design Rationale — Why it works
**관통 원리: 단안 2D가 *관측하는 축*과 *못 하는 축*을 분리해, 각각 맞는 소스로 풀어 충돌 없이 합친다.**
- **세 전문가 강점만**: 위치=ViTPose 2D IK · 각도=HaMeR · 깊이/가려짐=GVHMR · 손가락=HaMeR+게이트.
- **직교 분해("2D는 깊이를 모른다")**: 위치(이미지면)·깊이(GVHMR 앵커)·전완축 회전을 **직교**로 분해 → 서로 손해 없이 동시 최적화. (손목 14→3.5px, 깊이 깜빡임 9→3.8cm)
- **swing-twist 분해 = 우리 기여의 근거**: `q_wrist=q_swing·q_twist(전완축)`. **swing(2D-가시)=HaMeR**(정합), **twist(2D-불가시)=물리 전완**(비틀림 방지). ← *근거: PyMAF-X("twist는 위치 거의 불변") + `physical_limits.py` rest-pose 실측*. 이게 universal graft가 비틀리는 이유 + 우리가 푸는 이유를 동시에 설명.
- **충돌도 2D-불가시 axial로 해소** → 2D 손해 0, 충돌 9→1%. **신뢰도+기하(IoU) 게이트** → 가려짐/겹침 환각 억제, body_pose 불변(국소·가역).

## 5. Metrics (GT-free) — 정의는 METRICS_FORMULAS.md
- **Forearm-twist violation %** (생체역학 한계 위반율) ← 핵심
- **Collision / penetration** (손-몸·손-손 관통 프레임/깊이) ← 핵심
- **Hand 2D PCK** (독립 MediaPipe, 비순환) — WB는 순환이라 폐기
- **Body acceleration** (전신 안정성, SOTA 자릿수) — 유지
- **Jitter** (jerk, control 변수 — 동일 평활) — 유지
- **User study** (A/B pairwise preference) ← 지각

## 6. Results — Main table

**Table 1. 통합 방식(integration) 비교 — 동일 HaMeR 손, 통합 전략만 교체 (front_3)**

| Approach | Hand 2D PCK ↑ (MediaPipe, 독립) | Forearm-twist ↓ | Collision ↓ |
|---|:--:|:--:|:--:|
| ① Fingers-only (손목 병합 X) | 0.33 | **0.0%** | low |
| ② **Universal graft** (손목 그대로 삽입) | **0.52** | 4.0% | 243 |
| ②+ Kalman 손목 스무딩 | 0.52 | 2.9% | — |
| ③ **Ours** (전완축 손목각 제한) | **0.52** | **1.9%** | **109 (−55%)** |

*전 방법 2D 동일(0.52, MP 독립). twist 단조 감소 4.0→2.9→1.9. ①은 손목 버려 2D 붕괴(0.33).*
*8클립 검증: ②vs③ = PCK 0.62/0.62(동등), twist 7.1%/5.2% — same-2D·lower-twist 유지(N=9,517). Body accel ~2.6 < WHAM 6.6.*

> **Caption**: *"With identical HaMeR hands: ① ignoring hand orientation avoids twist but cripples 2D (0.33); ② the universal graft achieves good 2D (0.52) but twists the forearm (4.0%); ②+ simple Kalman smoothing only **partially** removes twist (2.9%) — it suppresses temporal spikes, not the systematic graft-induced twist. **③ Ours, constraining wrist rotation about the forearm axis, reduces twist furthest (1.9%) and collisions 55%, at the same 2D accuracy.**"*

### ★ What's better than the universal graft (핵심 메시지)
**보편적 graft(②)는 손목을 그대로 꽂아 영상 정합은 얻지만 전완이 비틀린다.** 우리(③)는 **전완축 기준으로 손목 각도를 제한**해 — 영상 정합(2D)은 동일하게 유지하면서 — 비틀림과 충돌을 줄인다. *fingers-only(①)처럼 손목을 버려 비틀림을 피하면 2D가 무너지므로, 둘 다 잡는 건 ③뿐이다.*

**Table 2. 2D 지표의 순환성 폭로 (독립 MediaPipe vs WB-fit)**

| Method | WB PCK (순환) | **MediaPipe PCK (독립)** |
|---|:--:|:--:|
| adaptive_v4 (WB 과적합 실험) | 0.90 | **0.46** ↓↓ |
| fit2d (2D-fit) | 0.79 | 0.57 |
| **Ours** | 0.45 | **0.62** 🥇 |

> *"WB(피팅 타깃) 기준 PCK는 2D-fit 방법을 편애하는 순환 지표다. 아무도 피팅하지 않은 독립 검출기(MediaPipe)로 채점하면 순위가 역전되어 Ours가 최고이며, WB 과적합본은 0.90→0.46으로 붕괴한다."* → **방법론 기여**.

**Table 3. Ablation / Journey (front_3)** — 컴포넌트 효과: axcol → 충돌 243→109, grip → 가려짐 펼침 14.0→12.1 cm (best/README).

> **fit2d = 우리가 폐기한 초기 시도(v3), prior work 아님.** 손을 2D 키포인트에 *직접 피팅*했더니 — (i) WB에 **과적합**(독립 MediaPipe 0.57로, 순환 지표서만 높음, Table 2), (ii) 충돌(391) 유발, (iii) per-frame 피팅 jitter. → **2D 직접 피팅 대신 "전완축 손목각 제한"(v4)으로 전환.** baseline이 아니라 *설계 동기*로 서술.

## 7. Key figure (★ money plot)
**산점도: x = Forearm-twist viol.% (↓ 좋음), y = inverse-collision (↑ 좋음)**
- Naive·fit2d = 우하(twist↑·충돌↑), **Ours = 좌상 단독(twist↓·충돌↓)**.
- 보조: 손-shape·body·jitter는 세 방법 거의 한 점에 겹침 → "차별은 물리, 나머진 유지" 시각화.

## 8. Qualitative (3열 영상)
Naive | 2D-fit | **Ours** — 손 zoom + 충돌/비틀림 하이라이트. (렌더: `v4_generalization/<clip>_v3_vs_v4.mp4`)

## 9. Conclusion + Limitations (정직)
- **결론**: 국소 손-몸 통합 교정으로 — 추적·안정성 손실 없이 — 물리적 타당성(충돌·비틀림)을 개선.
- **한계**: ① 충돌 8클립 평균 미완(front_3 only). ② 가려짐 구간 손목 실위치는 GVHMR 한계(→ 라켓비전 후속). ③ GT 부재로 절대 정확도 측정 불가(2D pseudo-GT·user study로 대체). ④ jitter 절대값은 단위·평활 의존 → 상대비교만.

---

## 헤드라인 한 줄 (포스터 상단/초록)
> *"Given the same HaMeR hands, our twist-aware, collision-resolving integration produces **physically plausible** hand–body motion (−72% collisions, lowest forearm twist) **without sacrificing** hand-tracking accuracy or body stability."*
