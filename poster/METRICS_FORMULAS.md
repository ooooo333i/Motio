# Evaluation Metrics — Formulas & Sources (GT-free)

> 작성 2026-06-03. 구현 = `cap_metrics.py`. 모두 **3D GT 불필요** — 출력 궤적/메쉬 + 2D 검출(pseudo-GT)만 사용.
> 표기: 3D 관절 $X_{j,t}\in\mathbb{R}^3$ (관절 $j$, 프레임 $t$, $T$ 프레임), 2D 검출 $x_{j,t}\in\mathbb{R}^2$ (신뢰도 $c_{j,t}$), $\Delta t=1/\text{fps}$, 카메라 $K$, 투영 $\pi([u,v,w])=[u/w,\,v/w]$.

---

## 1. Forearm-twist / wrist joint-limit violation (%)
손목 로컬 회전을 (axial=전완 비틀림, deviation, flexion) 으로 분해해 생체역학 한계 $[\theta^{\min},\theta^{\max}]$ 위반 프레임 비율.
$$\text{Viol} = \frac{1}{ST}\sum_{s\in\{L,R\}}\sum_{t}\mathbb{1}\!\big[\theta^{\text{axial}}_{s,t}\notin[\theta^{\min},\theta^{\max}]\big]$$
- 구현: `physical_limits.check_wrist_rotation` (SMPL-X 축 규약 실측 검증).
- **개념·동기 출처**: PyMAF-X — *"naive Copy-Paste integration → unnatural wrist poses"*, twist(=elbow→wrist 축 회전)는 관절 위치 거의 불변. → 비틀림 보정의 정당성.
  - Zhang et al., *PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images*, TPAMI 2023. arXiv:2207.06400.
- 보조: Moon et al., *Accurate 3D Hand Pose Estimation for Whole-Body 3D Human Mesh Estimation (Hand4Whole)*, CVPRW 2022 — SMPL-X 손목 yaw/pitch 부정확 문제.

## 2. Collision / penetration
부위쌍(손-몸, 손-손, 팔-몸) 메쉬 관통. 침투 프레임 수 + 평균/최대 침투 깊이(mm).
$$\text{Pen\%}=\frac{1}{|V|T}\sum_{t,v}\mathbb{1}[\Phi(v_t)<0],\qquad \text{Depth}=\frac{1}{N}\sum \max(0,-\Phi(v_t))$$
($\Phi$ = signed distance, 내부 음수.) 구현: `collision_detect.py` → `collision_*.json`.
- **출처**: Hassan et al., *Resolving 3D Human Pose Ambiguities with 3D Scene Constraints (PROX)*, ICCV 2019 — vertex-SDF 기반 inter-penetration 깊이, contact = SDF<0.

## 3. 2D hand accuracy — **독립 검출기(MediaPipe), 비순환**
3D 손 관절(손목+MCP 5점/손)을 재투영 후, **어느 방법도 피팅하지 않은 제3 검출기 MediaPipe Hands** 2D와 PCK 비교.
$$\hat{x}_{j,t}=\pi(K\,\tilde{X}_{j,t}),\qquad
\text{PCK@}\alpha=\frac{1}{|H|\,|\mathcal{D}|}\sum_{t\in\mathcal{D}}\sum_{j\in H}\mathbb{1}\!\big[\lVert\hat{x}_{j,t}-x^{\text{MP}}_{j,t}\rVert_2\le \alpha\,s_t\big]$$
$\mathcal{D}$=MP 검출 프레임, $s_t$=손크기(손목→중지MCP), $\alpha=0.5$, $H$=6점(손목+5 MCP). SMPL-X↔MP/WB 대응 = batch_fit2d 규약(`[20,25,28,31,34,37]`↔손목·index·middle·ring·pinky·thumb MCP). L/R = 투영 손목 최근접 매칭. 구현: `cap_mediapipe_detect.py`(검출) + `cap_metrics`(평가).
- **왜 MediaPipe(독립)인가 — 순환성**: fit2d/adaptive는 **WB(`wholebody_kp2d.npy`)에 피팅** → WB-PCK는 자기 목적함수(순환) + 컨벤션 offset로 편향. 실제로 adaptive_v4는 **WB 0.90 → MP 0.46** 붕괴(과적합 폭로). 독립 MP에선 Ours 최고. ⇒ **헤드라인 2D는 MediaPipe만 사용.**
- **정직 caveat**: 빠른 테니스 손이라 MP 검출률 2–88%(클립차 큼) → 절대 PCK 낮음(~0.45–0.62). N-weighted pooled(9,517 손샘플) 기준 보고. v4 ≈ naive graft(동률) = **무손실**(독립 기준).
- **PCK 정의 출처**: Andriluka et al., *2D Human Pose Estimation: New Benchmark and SOTA Analysis (MPII/PCKh)*, CVPR 2014.

## 4. Jitter (smoothness, control 변수)
관절 위치의 3차 미분(jerk) 평균 크기. (root-relative: pelvis 차감 → articulation만.)
$$j_{j,t}=\frac{X_{j,t+2}-2X_{j,t+1}+2X_{j,t-1}-X_{j,t-2}}{2\Delta t^{3}},\qquad
\text{Jitter}=\frac{1}{|H|(T-4)}\sum_{j\in H}\sum_t\lVert j_{j,t}\rVert_2$$
- **출처**: Yi et al., *TransPose* (SIGGRAPH 2021) / *Physical Inertial Poser (PIP)* (CVPR 2022) — jitter = 전관절 jerk 평균(원전 단위 km/s³).
- **주의**: 절대값은 단위·평활·fps 의존 → **상대 비교만**. 본 연구는 전 방법 동일 σ=1 평활 후 비교(=control).

## 5. Body acceleration (전신 안정성)
몸통 관절(0–21) 2차 미분 평균 크기 (m/s²). SOTA HMR 의 Accel 자릿수와 비교용 sanity.
$$\text{Accel}=\frac{1}{|B|(T-2)}\sum_{j\in B}\sum_t\Big\lVert\frac{X_{j,t+1}-2X_{j,t}+X_{j,t-1}}{\Delta t^{2}}\Big\rVert_2$$
- **앵커 출처**: Shin et al., *WHAM*, CVPR 2024 — 3DPW Accel 6.6 m/s² (ViT). 원 Accel(예측−GT 차) 정의: Kanazawa et al., *Learning 3D Human Dynamics from Video (HMMR)*, CVPR 2019.
- 주의: 본 연구 값은 GT-free 크기(오차값 아님) → "자릿수가 SOTA 이하" sanity 로만 사용.

## 6. User study (perceptual)
A/B pairwise preference: $N$명 rater × 클립, blind·랜덤순서. 선호율 + 이항검정.
$$\text{Pref}=\frac{\#\{\text{ours 선택}\}}{N_{\text{trials}}},\qquad p=\text{Binomial}(k;N,0.5)\ \text{(귀무: 무선호)}$$
- 표준 관행(GT 부재 시 perceptual eval). 손/팔/전체 3축으로 분리 권장.

---

## 요약: 무엇이 GT를 대체하나
| 지표 | GT 대체물 |
|---|---|
| twist viol / jitter / accel / collision | 없음 — 출력 궤적·메쉬만 |
| hand 2D / (옵션)PCK | 2D 검출기(COCO-WholeBody) = pseudo-GT |
| user study | 사람 perceptual 판단 |

**핵심 정직 원칙**: 최적화한 양(2D fit·평활)은 '주장 검증/control'로만, '독립 우월성'은 최적화 안 한 신호(충돌·twist·offset제거 shape)로 주장.
