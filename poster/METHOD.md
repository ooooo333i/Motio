# Method — Observability-Structured Wrist Fusion for Monocular Hand–Body Recovery

> 정식 method 서술 (논문/발표용). 수치·하이퍼파라미터는 [../docs/V4_ALGORITHM.md](../docs/V4_ALGORITHM.md) 부록.
> 용어: "손 전문가" 대신 **HaMeR(손 크롭만 입력받는 단일 손 MANO 추정 모델)**, **GVHMR(전신 SMPL-X 시계열 추정 모델)**, **ViTPose(2D 키포인트 검출, 프레임별 신뢰도)**.

---

## 1. Problem formulation
단안 영상에서 expressive 3D human(SMPL-X)을 복원하려면 **전신 모델(GVHMR)**과 **손 모델(HaMeR)**의 출력을 하나의 SMPL-X로 합성해야 한다. 그러나 HaMeR는 **손 크롭만 보고(in isolation)** 손과 손목 global orientation을 추정하므로, 그 손목 추정은 **자신이 어느 전완(elbow→wrist)에 연결되는지 모른다 — 즉 전완 운동학과 분리(decoupled)되어 있다.**

표준 합성(손목 회전을 그대로 이식하는 *copy-paste graft*)은 이 손-국소 손목 추정을 신체에 직접 전이한다. 손목의 yaw/roll이 전완 bone에 의해 구속되지 않으므로, 합성 결과는 **해부학적으로 부적절한 전완 회내/회외(forearm axial twist)**를 보인다 (PyMAF-X도 동일 문제를 보고).

## 2. Key observation — 관측 가능성(observability) 구조
단안 2D 관측은 **image-plane projection만 구속**하고 다음에 **불변(invariant)**이다:
- (i) **depth** (광선 방향)
- (ii) **카메라-광선/전완 축에 대한 회전(axial)**

따라서 손목 회전의 자유도는 관측 가능성에 따라 자연히 분리된다:
- **swing**(굴곡 flexion + 편위 deviation) — 이미지에서 **관측 가능**, well-constrained.
- **twist**(전완 축 회전) — 2D에서 **관측 불가능**, 손-국소 추정에는 **ill-posed**(HaMeR axial을 그대로 쓰면 ~66° 붕괴).

## 3. Proposed — observability-structured constrained fusion

표기: 단위 quaternion $q=(x,y,z,w)$, $\otimes$=quaternion 곱, $R(\cdot)$=rotation, $\pi(\cdot)$=핀홀 투영, $z(\cdot)$=cam-depth, $d_{\mathrm{geo}}$=geodesic 거리.

### 3.1 Swing–twist 연산자 (전완 x-축 기준)
손목 회전을 전완 축에 대한 **swing**(2D 가시)과 **twist**(2D 불가시)로 분해한다:
$$q^{\mathrm{tw}}(q)=\frac{(x,0,0,w)}{\sqrt{x^2+w^2}},\qquad q^{\mathrm{sw}}(q)=q\otimes \big(q^{\mathrm{tw}}\big)^{-1},\qquad \alpha(q)=2\,\mathrm{atan2}(x,w)$$
이면 $q=q^{\mathrm{sw}}\otimes q^{\mathrm{tw}}$. ($\alpha$=전완축 회전각/twist.)

### 3.2 **Core — 손목 병합 (swing=HaMeR, twist=전완)** ★
팔 chain $\mathcal C$(collar→shoulder→elbow)로 elbow global rotation:
$$G_e(f)=R(\phi^{\mathrm{root}}_f)\textstyle\prod_{j\in\mathcal C}R(\theta_{j,f})$$
HaMeR 손목을 **전완-local** 프레임으로 가져온 뒤,
$$R^{\mathrm{ham,loc}}_w(f)=G_e(f)^{\top}\,R^{\mathrm{HaMeR}}_w(f)$$
**swing은 HaMeR에서, twist는 신체(전완)의 물리 axial $\alpha^{\mathrm{body}}(f)$에서** 취해 합성:
$$\boxed{\,q_w(f)=\underbrace{q^{\mathrm{sw}}\!\big(R^{\mathrm{ham,loc}}_w(f)\big)}_{\text{image-observable (HaMeR)}}\otimes\underbrace{q^{\mathrm{tw}}\!\big(\alpha^{\mathrm{body}}(f)\big)}_{\text{unobservable axial (forearm)}}\,}$$
$$\alpha^{\mathrm{body}}(f)=\alpha\!\big(q^{\mathrm{tw}}(R^{\mathrm{arm}}_w(f))\big)\quad(\text{arm-IK 손목의 twist})$$
→ **손목을 손 단독 변수가 아닌 전완과 연결된 관절로 재정의.** well-constrained swing(손)과 ill-constrained axial(전완)을 분리해, **2D 정합 손실 없이**(독립 MediaPipe서 naive와 동등 정합) graft-induced 비틀림을 **전완 물리값으로 대체**한다. ($q_w$=`body_pose[wrist]`로 출력, HaMeR 미검출 프레임은 arm-IK 손목으로 fallback.)

### 3.3 신뢰도·물리 게이트 + 시계열 평활
비물리적 급변·가려짐 프레임을 양옆 good HaMeR에서 SLERP 보간:
$$\mathrm{bad}(f)=\mathbb 1\!\big[\omega(f)>\tau_\omega\big]\ \lor\ \mathbb 1\!\big[c^{\mathrm{WB}}_{\mathrm{hand}}(f)<\tau_c\big],\quad \omega(f)=\big\|\log\!\big(R(f{-}1)^{\top}R(f)\big)\big\|$$
$$q_w(f)\!\leftarrow\!\mathrm{Slerp}\big(q^{\mathrm{lo}}_w,q^{\mathrm{hi}}_w;\tfrac{f-\mathrm{lo}}{\mathrm{hi}-\mathrm{lo}}\big)\ (\text{bad}),\qquad q_w\!\leftarrow\!\mathrm{Slerp}\big(q_w,\mathcal G_\sigma[q_w]; w\big)\ (\text{경계 블렌딩})$$
($\tau_\omega=33°/\text{f},\ \tau_c=0.55$; $\mathcal G_\sigma$=quaternion gaussian, $w$=dilated bad mask. 전역 $\sigma=1.8$.)

### 3.4 직교 subspace — 위치·깊이는 arm-IK로 (stage A)
손목/팔 **위치는 2D로, 깊이·자세는 prior/앵커로** — 직교 방향이라 상충 없음:
$$\min_{\theta_{\mathrm{arm}}}\ \underbrace{\tfrac1F\!\sum_{f,j}c_{f,j}\|\pi(J_{f,j})-x^{\mathrm{vp}}_{f,j}\|^2}_{L_{2d}\ (\text{2D IK})}+\underbrace{\tfrac1F\!\sum_f w^{\mathrm p}_f\|\theta_{\mathrm{arm},f}-\theta^{\mathrm{GV}}_{\mathrm{arm},f}\|^2}_{L_{\mathrm{prior}}}+w_{tc}\underbrace{\textstyle\sum_f d_{\mathrm{geo}}(R_f,R_{f{+}1})}_{L_{tc}}+W_d\underbrace{\tfrac1F\!\sum_f\|z(J_{\mathrm{arm}})-z^{\mathrm{GV}}\|^2}_{L_{\mathrm{depth}}}$$

### 3.5 충돌도 2D-불변 axial subspace에서 (stage D) — 손목 회전만 최적화
$$\min_{\theta_{19,20}}\ 15\,L_{\mathrm{hand2d}}+300\,L_{\mathrm{col}}+L_{\mathrm{anch}}+300\,L_{tc}$$
$$L_{\mathrm{col}}=\tfrac1F\!\sum_f\mathrm{relu}\!\Big(m-\min_{u,v}\|V^{L}_u-V^{R}_v\|\Big),\quad L_{\mathrm{anch}}=\tfrac1F\!\sum_f a_f\|\theta_w(f)-\theta^0_w(f)\|^2$$
$L_{\mathrm{hand2d}}$(강하게)이 swing/2D를 고정 → 옵티마이저가 **2D 불변 axial로만** 손을 돌려 관통을 해소한다(reprojection 손실 0; 본 측정 penetration −55%, front_3). $a_f$: 비충돌 3000 → 충돌 8 (충돌 프레임만 자유), $m{=}18\,$mm.

### 3.6 ROM clamp (stage C)
손목 local을 (dev, flex, axial)로 분해 → $\mathrm{dev}\in[-45°,45°],\ \mathrm{flex}\in[-90°,90°]$ 해부학적 clamp(비위반 불변).

### 3.7 Total objective — staged composition
본 방법은 **단일 end-to-end joint loss가 아니라**, 관측 가능성에 따라 분해된 하위문제를 **순차 합성(staged composition)**으로 푼다 — 각 단계가 직교 subspace를 다루므로 순차 적용이 서로를 깨지 않는다. **두 개의 미분가능 최적화**(stage A, D)와 **세 개의 closed-form 연산자**(B 병합, C clamp, E grip)로 구성된다:

$$\Theta^{\star}\;=\;\Pi_{E}\;\circ\;\Big[\arg\min_{\theta_{w}}\,L_{D}(\theta_w)\Big]\;\circ\;\Pi_{C}\;\circ\;\mathcal F_{B}\;\circ\;\Big[\arg\min_{\theta_{\mathrm{arm}}}\,L_{A}(\theta_{\mathrm{arm}})\Big]\,(\Theta_{0})$$

| 단계 | 형태 | 목적함수 / 연산 |
|---|---|---|
| **A** arm IK | $\arg\min$ | $L_A=L_{2d}+L_{\mathrm{prior}}+w_{tc}L_{tc}+W_{d}L_{\mathrm{depth}}$  *(코드: `cap_arm_prefit.py:500`; $w_{tc}{=}5000,\,W_d{=}10^5$)* |
| **B** 손목 병합 | closed-form $\mathcal F_B$ | $q_w=q^{\mathrm{sw}}(G_e^{\top}R^{\mathrm{HaMeR}}_w)\otimes q^{\mathrm{tw}}(\alpha^{\mathrm{body}})$ + SLERP 게이트 (§3.2–3.3) |
| **C** ROM clamp | projection $\Pi_C$ | $\mathrm{dev}\!\in\!\pm45°,\ \mathrm{flex}\!\in\!\pm90°$ |
| **D** 충돌 | $\arg\min$ | $L_D=15L_{\mathrm{hand2d}}+300L_{\mathrm{col}}+L_{\mathrm{anch}}+300L_{tc}$ |
| **E** grip | closed-form $\Pi_E$ | 손가락만, conf·IoU 게이트 보간 hold (body_pose 불변) |

> **설계 정당화**: 단계들이 *직교 관측 subspace*(위치/깊이/swing/axial)를 다루므로 joint optimization 없이도 상호 간섭이 없다. 하드 제약(swing-twist, RoM, grip)은 soft loss 대신 **closed-form projection**으로 풀어 joint 최적화의 불안정성을 피한다. $L_A,L_D$만 미분가능 최적화(Adam, early-stop).

### 3.8 Efficiency
구조 자체가 빠르다 — 두 가지 이유:
1. **Closed-form 단계(B·C·E)는 반복이 없다** — swing-twist 병합·RoM clamp·grip 보간은 1-pass 행렬/quaternion 연산이라 거의 즉시(합쳐 $\sim$0.007 s/frame).
2. **좋은 초기값 → 빠른 수렴**: 최적화 단계도 강한 prior에서 출발한다. stage A는 **GVHMR 자세**에서(이미 정답 근처라 위치만 미세 IK), stage D는 **이미 병합·clamp된 손목**에서(충돌 프레임만 국소 보정) 시작 → 둘 다 최적점 부근 초기화라 early-stop으로 소수 iteration에 수렴.

→ 전체 오프라인 정제가 $\sim$0.04–0.07 s/frame 수준. *(상세 단계별 타이밍·baseline 대비는 별도 비교 문서.)*

## 4. Offline, confidence-adaptive fusion
본 방법은 **오프라인 후처리 정제**이므로 전체 시퀀스를 활용한다. frame·joint별 **신뢰도 적응적 융합**:
- HaMeR 손 관측이 고신뢰 → 그대로 신뢰.
- **가려짐(occlusion)** → 검출 신뢰도가 정직하게 하락 → 시계열 일관 신체 prior로 부드럽게 fallback.
- **손–손 겹침(overlap)** → 검출 신뢰도는 과신(겹쳐도 0.9 유지) → **기하(IoU)로 판정**, 뒤 손만 교정.

이 적응성(잘 보이면 손, 안 보이면 팔·시계열)은 **streaming/실시간 방식엔 불가능**한, 오프라인의 구조적 이점이다.

## 5. Contributions
1. **Observability-structured wrist fusion** — swing–twist 분해로 image-observable DOF(손)와 unobservable axial DOF(전완)를 분리. 손목을 *손 단독 변수*가 아닌 *전완과 연결된 관절*로 모델링하여 copy-paste graft의 전완 비틀림을 **원리적으로 감소**(graft-induced axial을 전완 물리값으로 대체).
2. **Orthogonal-subspace refinement** — 위치(2D IK)·깊이(prior anchor)·충돌(axial)을 비간섭 subspace에서 풀어, 상호 손실 없이 physical plausibility를 향상.
3. **Non-circular evaluation** — 피팅 타깃 기반 2D 지표의 circularity를 정량 폭로하고, held-out detector(MediaPipe)에서 baseline 동등 정합 + 우월한 physical plausibility(충돌·전완 비틀림)를 입증.

> **주제 연결 (AI 성능 개선)**: 두 SOTA 모델(GVHMR·HaMeR)의 단순 합성은 출력이 물리적으로 붕괴한다(전완 비틀림·관통). 본 연구는 **관측 가능성 구조에 기반한 학습-불필요(post-hoc) 정제**로, 추가 학습/데이터 없이 composed pipeline의 출력 충실도(physical fidelity)를 개선한다.

---

## English abstract (CVPR-style, 한 단락)
*Monocular expressive human recovery composes a whole-body model (GVHMR) with a hand model (HaMeR) that regresses the hand from cropped hand images **in isolation**, decoupled from the forearm. Naively grafting HaMeR's wrist orientation onto the body — the de-facto standard — transfers a hand-frame estimate whose axial rotation is unconstrained by the elbow–wrist bone, yielding anatomically implausible forearm twist and hand–body inter-penetration. We observe that monocular 2D evidence is invariant to depth and to rotation about the forearm axis; the wrist's degrees of freedom thus separate by observability into an image-observable **swing** and an unobservable **twist**. We reformulate wrist estimation as an observability-structured constrained fusion: a swing–twist decomposition takes swing from the hand model and constrains twist to the body's physical forearm value under anatomical limits, treating the wrist as a kinematic joint rather than a free hand variable. The same principle factorizes position, depth, and collision into orthogonal, non-interfering subspaces, resolving inter-penetration in the 2D-invariant axial subspace at zero reprojection cost. As an offline, confidence-adaptive refinement, the method trusts the hand when visible and falls back to temporally-coherent body priors under occlusion (geometric IoU gating for overlap). Under a **held-out, non-circular** 2D hand metric (MediaPipe), our method matches the naive graft's hand alignment while reducing forearm-twist violations and hand–hand penetration, requiring **no additional training**.*
