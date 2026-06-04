# Evaluation — Adaptive v4 (poster section, 2026-06-04)

> 모든 숫자는 재현 파이프라인(`hamer_wrist_hamer_rom_col_grip.pt`) 출력 실측.
> 비교군은 **우리 기여(손목 통합)를 격리**하도록 선택 — fit2d(별도 배포물) 대신 *naive 통합 변형 + GVHMR 원본*.
> 수식·출처 = [METRICS_FORMULAS.md](METRICS_FORMULAS.md).

---

## 1. 비교군 (baselines) — 우리 핵심 아이디어를 안 준 공정 대조
| 약칭 | 정의 | 손목 처리 |
|---|---|---|
| **GVHMR raw** | GVHMR 원본 SMPL-X (손=GVHMR 자체 추정) | GVHMR 손목, 손가락 미정밀 |
| **Naive graft** | GVHMR 바디 + HaMeR 손가락 (손목 병합 X) | GVHMR 손목 (트위스트 평평) |
| **Wrist+smooth** | 순수 HaMeR 손목 직접 병합 + Kalman 평활 | HaMeR 손목, **제한·게이트 없음** |
| **Ours v4** | 전완축 손목각 제한(ROM) + 신뢰도 게이트 + 축분리 충돌 | HaMeR 손목 + **물리·신뢰도 제약** |

> 핵심: baseline 3종 모두 **우리 아이디어(전완축 제한/게이트)를 안 줌** → 차이 = 순수 우리 기여.

---

## 2. ★ 메인 결과 — 독립 2D 정확도 (non-circular)
**심판 = MediaPipe Hands** (우리도 baseline도 학습/피팅에 안 쓴 제3 검출기). 8클립 pooled, 손목+MCP+끝 11점.

| 방법 | **PCK@0.5 ↑** | median err (px) ↓ |
|---|:--:|:--:|
| GVHMR raw | 0.180 | 29.9 |
| Wrist+smooth | 0.203 | 29.8 |
| Naive graft | 0.204 | 29.5 |
| **Ours v4** | **0.235** | **25.1** |

**→ Ours가 전 baseline 압도:** vs Naive **+15%**, vs Wrist+smooth **+16%**, vs GVHMR raw **+31%** (상대).

> 절대값이 낮은(~0.2) 이유: 테니스 손이 작고 빨라 MediaPipe도 ~14%만 검출 → **상대 순위(비순환)가 근거.** 프레임별 최근접 손 매칭 + 중앙값.

---

## 3. vs Dyn-HaMR — published SOTA
Dyn-HaMR(동적 손 SOTA)을 **GVHMR 카메라 주입으로 공정 graft** 후 비교 (input 클립 442f, 독립 MediaPipe).

| 방법 | PCK@0.5 ↑ |
|---|:--:|
| Naive graft | 0.140 |
| **Dyn-HaMR (SOTA)** | 0.199 |
| **Ours v4** | **0.287** |

**→ Ours가 SOTA Dyn-HaMR 대비 +44% (상대).**

---

## 4. 물리적 타당성 (보조)
**Forearm-twist(axial) 위반율** — *정직한 해석 주의*:

| 방법 | axial 위반% | 해석 |
|---|:--:|---|
| GVHMR raw / Naive | 0.00 | 위반 0 = **트위스트를 *안 함*** (GVHMR 평평 손목 → 틀린 트위스트) |
| Wrist+smooth | 3.41 | HaMeR 트위스트 따라가나 **제한 없어 비물리적 초과** |
| **Ours v4** | **2.11** | HaMeR 트위스트 추종 + **제일 물리적**(전완축 ROM 제한) |

> ⚠️ "위반 0"은 우월이 아님 — naive는 *트위스트 자체를 포기*한 것. **트위스트를 실제로 수행하는 방법(Wrist+smooth, Ours) 중 Ours가 가장 물리적.** (트위스트 정확도 정량 = 라켓축, 진행 중.)

충돌(손-손 관통): front_3에선 4방법 동률(≈3f) — 이 baseline들은 손가락이 유사해 비구분. (충돌 우위는 fit2d 등 2D-피팅 대비에서 두드러짐.)

---

## 5. 정직성 원칙
- **최적화한 양(2D fit·평활)은 control/주장검증에만.** 독립 우월성은 **최적화 안 한 신호**(독립 MediaPipe, 물리 제약)로 주장.
- 절대 PCK 낮음 → **상대 순위**가 근거. 표본·검출률 명시.
- twist "0%"의 함정 명시 (트위스트 미수행).
- 트위스트 *정확도* 정량은 라켓축(RacketVision)·ARCTIC(3D GT)로 보강 예정.

---

## 6. 한 줄 요약
**독립 심판(MediaPipe)에서 Ours가 naive 통합·GVHMR 원본·SOTA(Dyn-HaMR)를 모두 이기고(+15~44%), 트위스트를 *수행하면서* 가장 물리적이다.**

### 데이터 출처
- MediaPipe npy: `outputs/cap_pipeline/<clip>/mediapipe_hands.npy` (8클립 + input)
- 산출: `hamer_wrist_hamer_rom_col_grip.pt`(ours) / `smplx_merged_hamer_post.pt`(naive) / `hmr4d_results.pt`(GVHMR) / `baseline_m2_kalman.pt`(wrist+smooth) / `cmp_dynhamr_grafted.pt`(Dyn-HaMR)
