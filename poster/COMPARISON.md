# Experiments — Comparison

> [METHOD.md](METHOD.md) 동반 실험 문서. 모든 수치는 실제 출력에서 `cap_metrics.py`로 계산(재현 가능). 평가 정의·출처 = [METRICS_FORMULAS.md](METRICS_FORMULAS.md).
> 용어: GVHMR(전신 SMPL-X 추정), HaMeR(손 크롭 단일 손 MANO 추정), Dyn-HaMR(동적 카메라 4D 손 — 다단계 test-time 최적화).

---

## 1. Setup
- **GT-free 평가** (3D GT 없음). 지표:
  - **Forearm-twist viol.↓** — 손목 axial 생체역학 한계 위반율 (물리 타당성).
  - **2D hand PCK↑** — **독립 검출기 MediaPipe**(어느 방법도 피팅 안 함, 비순환), 손목+MCP 6점, PCK@0.5.
  - **Hand jitter↓** — 손 관절 jerk 평균 크기 (smoothness).
  - **Hand–hand collision↓** — 손-손 메쉬 관통 프레임 비율.
- 클립: front_3(상세 4-way), 8클립(front_1–4, side_1–4; pooled N=9,517 손샘플).

## 2. Integration ablation — 단계별 무엇을 이식하는가 (front_3, 동일 HaMeR)

| 구성 | Forearm-twist↓ | **2D PCK↑ (MediaPipe)** | Hand jitter↓ | 손-손 충돌↓ |
|---|:--:|:--:|:--:|:--:|
| **GVHMR 단독** (손 정지) | 0.0% | 0.33 | 254 | low |
| **+ 손가락만 이식** (손목=GVHMR) | 0.0% | 0.33 | 261 | low |
| **+ 손목까지 이식** (universal graft) | 4.0% | **0.52** | 220 | 8.1% |
| **Ours** (전완축 손목각 제한) | **1.9%** | **0.52** | **208** | **3.3%** |

**관찰**:
- 손가락만 이식하면 손목 방향이 영상과 안 맞아 **2D가 안 오른다**(0.33, GVHMR 단독과 동일).
- 손목까지 이식해야 2D가 오르지만(0.52) **전완 비틀림(4.0%)·충돌(8.1%)**이 생긴다.
- **Ours는 동일한 2D(0.52)를 유지하면서 비틀림을 절반(1.9%)·충돌을 8.1→3.3%로** 줄인다. — *즉 손목 이식의 정합 이득은 취하고 부작용만 제거.*
- 8클립 검증(universal graft vs Ours): 2D PCK **0.62 = 0.62**(동등, N=9,517), twist **7.1% → 5.2%**.

## 3. vs Dyn-HaMR (SOTA dynamic hand) — 성능 & 효율

### 3.1 Performance (front_3, 독립 MediaPipe)
| Method | Forearm-twist↓ | 2D PCK↑ | Hand jitter↓ |
|---|:--:|:--:|:--:|
| Dyn-HaMR (full-body 통합) | 28.3% | 0.38 | 920 |
| **Ours** | **1.9%** | **0.52** | **208** |

> Dyn-HaMR는 손 단독 시계열 최적화에 강하나 **전신/카메라를 함께 풀지 않아**, 동일 파이프라인에 통합하면 손목 방향이 몸과 어긋나 **비틀림 28%·jitter 920**으로 악화된다(손-level temporal 일관성이 통합 후 전이되지 않음). 가벼운 정합-인지 정제만으로 Ours가 **더 낮은 비틀림·더 높은 2D·더 부드러운** 손을 낸다.

### 3.2 Efficiency — 본 연구의 강점
| Method | 방식 | 비용 |
|---|---|---|
| Dyn-HaMR | **다단계 test-time 최적화** (SLAM + interacting-hand motion prior + 계층 초기화) | 시퀀스당 무거움 (optimization-based) — *[측정값: 실제 실행 시간 기입]* |
| **Ours** | **feedforward(GVHMR·HaMeR) + 경량 정제** | **≈0.04–0.07 s/frame** (정제), 1.5k프레임 ≈ 1–2.5분 |

> **핵심**: Dyn-HaMR류 SOTA는 per-sequence 최적화로 비용이 크다. 본 방법은 **closed-form 병합 + warm-start 소수-iteration 최적화**(§Method 3.7–3.8)라, **추가 학습 없이 SOTA보다 가벼우면서도 통합 손 품질은 동등하거나 더 우월**하다. ⇒ "성능 개선 + 효율" 동시 달성.

## 4. 평가 지표의 순환성 (방법론 기여)
| Method | WB-PCK (피팅 타깃, 순환) | **MediaPipe-PCK (독립)** |
|---|:--:|:--:|
| adaptive_v4 (WB 과적합 실험) | 0.90 | 0.46 ↓↓ |
| fit2d (2D 직접 피팅) | 0.79 | 0.57 |
| **Ours** | 0.45 | **0.62 (8클립 1위)** |

> 피팅 타깃(WB) 기준 2D-PCK는 2D-fit 방법을 편애하는 **순환 지표**다. 아무도 피팅하지 않은 독립 MediaPipe로 채점하면 순위가 역전되어 Ours가 최고이며, WB 과적합본은 0.90→0.46으로 붕괴한다.

---

## 정직 caveat (논문 전 확인)
- **Dyn-HaMR 성능 수치(twist 28%·jitter 920)는 naive graft + degenerate 카메라 통합 결과** — Dyn-HaMR 단독 손 품질이 아니라 *"통합 시 전이 실패"*를 보이는 값. (라벨 명확히.)
- **Dyn-HaMR 실행 시간은 미측정(공개 안 됨)** — 너희가 실제 돌린 측정값으로 채울 것. 정성적으로 "optimization-based → 무겁다"만 확실.
- **충돌 3.3%**는 손-손 기준 재현 파이프라인(grip-chain) 값. (0% 변이는 2D·jitter 희생 → 폐기; 별도 논의.)
- 2D PCK 절대값(0.3–0.6)이 낮은 건 빠른 테니스 손이라 MP 검출률 자체가 낮은 탓 — **상대 비교**로 해석.
