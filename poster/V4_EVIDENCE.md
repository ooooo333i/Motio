# v4 Evidence Pack — 측정 결과 & 포스터 근거 (2026-06-04)

> 재현 파이프라인 = `hamer_wrist_hamer_rom_col_grip.pt` (run_v4.sh). 수식·출처 = [METRICS_FORMULAS.md](METRICS_FORMULAS.md), 포스터 본문 = [POSTER_CONTENT.md](POSTER_CONTENT.md).
> 숫자 출처: `outputs/metrics_mediapipe_8clip.json`, `outputs/metrics_8clip_FINAL.json`.
> **정직 원칙**: 실험본 `adaptive_v4.pt`의 PCK 0.877(WB·1클립·비재현) **사용 금지**. WB 기반 2D는 fit2d에 순환이라 헤드라인 부적합 → **독립 검출기(MediaPipe)로만** 2D 우월성 주장.

---

## 0. 한 줄 결론
**SOTA 손(HaMeR) + SOTA 몸(GVHMR)을 "물리적으로 타당하게" 융합 — 트위스트·충돌은 최저, 독립 기준 2D는 최고, 그 외는 무손실.**

---

## 1. 핵심 발견 — WB 순환성의 역전 (이 연구의 방법론적 기여)
손 2D 정확도를 **무엇으로 채점하느냐**에 따라 승자가 바뀐다:

| 채점 기준 | Ours v4 | fit2d | 성격 |
|---|:--:|:--:|---|
| WholeBody (WB) PCK | 0.45 | **0.79** | fit2d가 WB에 피팅 → **순환(자가채점)** |
| **MediaPipe (독립) PCK** | **0.62** 🥇 | 0.57 | 아무도 안 쓴 제3 검출기 → **공정** |

→ **WB는 fit2d를 편애하는 순환 지표.** 독립 심판(MediaPipe)으로 바꾸면 **Ours가 역전 승**. (front_3에서 발견 → 8클립 9,517 hands에서 재현됨.)

---

## 2. Main Table — 8클립 평균 (mean ± std)

| Metric | Naive graft | HaMeR+2Dfit (fit2d) | **Ours v4** | 판정 |
|---|:--:|:--:|:--:|---|
| **Forearm-twist viol.** ↓ (%) | 7.07 ± 2.75 | 6.61 ± 3.69 | **5.16 ± 3.01** | **최저 ✅ 물리** |
| **MediaPipe 2D PCK** ↑ (독립) | 0.620 | 0.566 | **0.623** | **최고 ✅ 공정 2D** |
| MediaPipe 2D err ↓ (px) | 16.03 | 17.53 | **15.94** | 최저 ✅ |
| Hand-shape 2D err ↓ (px, offset제거) | 6.23 ± 1.37 | 6.00 ± 1.11 | 6.14 ± 1.31 | =무손실 |
| Body accel (m/s²) | 2.60 ± 0.74 | 2.56 ± 0.74 | 2.60 ± 0.74 | =유지 |
| Jitter (jerk, control) | 87.7 ± 19 | 69.1 ± 15 | 82.4 ± 19 | naive보다↓, fit2d보다↑ |

> MediaPipe pooled (9,517 hands): GVHMR/HaMeR-merged = 0.455 (최하). Ours가 컴포넌트(HaMeR raw 0.455)보다도 **융합으로 +0.17 향상.**

## 3. Collision (front_3, mesh penetration frames ↓)
| Naive(pre) | fit2d | **Ours v4** |
|:--:|:--:|:--:|
| 243 | 391 | **109** |

→ Ours가 손-손/손-몸 관통 **최소** (fit2d 대비 −72%, naive 대비 −55%). 축분리(axcol)로 2D 보존하며 해소.

---

## 4. 세 줄 기여 (포스터 Contribution)
1. **물리적 타당성** — forearm-twist 위반 최저(5.16%) + 충돌 최저(109). naive Copy-Paste graft의 알려진 문제(PyMAF-X)를 정량 해소.
2. **공정 2D 무손실/우위** — 독립 MediaPipe 기준 2D PCK 최고(0.623). WB 순환성을 드러내고 비순환 기준으로 입증.
3. **가려짐 강건 그립** — IoU 겹침 게이트 + 신뢰도 폴백으로 occlusion·두손겹침 환각 억제 (손가락만, body_pose 불변 assert).

## 5. 정직한 caveat (리뷰 방어)
- **Jitter**: Ours(82) < naive(88)지만 > fit2d(69). fit2d가 매끈한 건 GVHMR 전완을 그대로 둬서(=트위스트 포기)임 → **smoothness↔twist 트레이드오프**로 정직히 서술. 동일 σ=1 평활 control 하에 비교.
- **MediaPipe 절대값 낮음(0.62)**: 테니스 손이 작고 빨라 MP도 ~14%만 검출 → **절대 정확도가 아니라 *상대 순위(비순환)***가 근거. 프레임별 최근접 손 매칭 + 중앙값.
- **Collision은 front_3 단일** — 8클립 확장 측정 필요(side는 충돌 더 많음).
- **3D GT 없음** — MPJPE는 in-the-wild라 불가. 아래 보강 계획 참조.

---

## 6. 평가 설계 (4축) & 보강 계획
| 축 | 지표 | 상태 |
|---|---|---|
| 손 품질 | HaMeR FreiHAND/HO3D **인용** ("SOTA 손 위 구축") | 인용 |
| 통합 정확도(3D) | **ARCTIC** MPJPE (풀바디+양손 GT 비디오) | ⏳ 등록·다운 필요 (PoC 몇 시퀀스) |
| 트위스트(도메인) | **라켓축** 정렬오차 (RacketVision, 비순환 GT) | ⏳ RacketVision 진행 |
| 안정/물리 | twist·collision·jitter·accel | ✅ 완료(위 표) |
| 2D(공정) | 독립 MediaPipe PCK | ✅ 완료(위 표) |

> ARCTIC = "GT 신뢰성 +1" (no-regression / 손목 트위스트 GT). 라켓축 = "도메인 트위스트 압도". 둘 다 옵션이며, **현재 표만으로도 "물리 타당 + 공정 2D 무손실"** 방어 가능.

## 7. 정성 근거 (영상)
- 8클립 `v4_generalization/<clip>_v3_vs_v4_HQ.mp4` (RAW|v3|v4)
- `DECISION_front3_adaptiveV4_vs_gripchain.mp4`, `front_1_ARMFIX_before_after.mp4`
- before/after: 트위스트·관통·가려짐 그립 환각 사례 컷.

---

## 8. 베이스라인 정의 (재현용)
- **Naive graft** = HaMeR 손을 SMPL-X에 Copy-Paste (표준 관행, PyMAF-X가 문제 지적). 동일 HaMeR 손이라 우리와 손 품질 동일 → 차이는 *통합 방식*.
- **HaMeR+2Dfit (fit2d)** = 배포 v3. 손목을 WB 2D에 피팅(전완=GVHMR 유지).
- **GVHMR / HaMeR-merged** = 컴포넌트 원본.
- **Ours v4** = arm_prefit IK → hamer_wrist(conf게이트) → ROM → axcol(축분리) → grip(IoU/conf) [+ σ=1 평활].
