# adaptive_v4 발전 기록 (2026-06-03)

> front_3(테니스 포핸드, 1495f)에서 손목/손/팔을 "극한까지" 정합시킨 하루의 기록.
> 사용자(주도적 방향제시)와 Claude(구현·측정·검증)의 협업. 상세 실험은 `ADAPTIVE_V4_DESIGN.md` §8.

---

## 0. 최종 v4 파이프라인 (현재 best = `best/v4_05_armgate_2026-06-03`)

```
영상 → GVHMR(몸) + HaMeR(손) + ViTPose(COCO17/WB 2D)
  │
  ├─[A] 팔 IK + 안정화   cap_arm_prefit.py (CAP_ARM_DEPTH=100000, w_tc=5000)
  │       · 어깨/팔꿈치/손목을 ViTPose 2D에 IK (손목 위치 14px→1px 교정)
  │       · depth anchor: 팔관절 cam-z를 GVHMR에 soft 앵커 (위치 깜빡임 제거)
  │       · arm 회전 gaussian σ1.5 (ViTPose jitter 제거)
  │       · batch_arm_gate.py: 손 가려짐(conf<0.65) 프레임 팔 위치 slerp (가려짐 튐 제거)
  │
  ├─[B] 손목 = HaMeR + 신뢰도 게이트   batch_hamer_wrist.py (AXIAL=hamer, smooth=1.8)
  │       · 손목 = HaMeR 글로벌→local (G_elbow⁻¹@HaMeR_wrist) — "잘 보이면 완벽한" HaMeR 각도 그대로
  │       · 신뢰도 게이트: bad=(속도>33 ∪ WB손conf<0.55) → good HaMeR에서 slerp coast (가려짐/스윙 뭉개짐 처리)
  │       · 전역 quat 평활 σ1.8 (HaMeR per-frame jitter 제거, 이탈 3°)
  │
  ├─[C] ROM 클램프   batch_rom_clamp.py (dev±45/flex±90/axial)
  │
  └─[D] 충돌 해소   batch_axcol_v4.py (손목 axial로 분리, 2D 보존, dilate6+delta σ4 전파)
```

손가락 = `_post`(HaMeR 후처리) 고정. **남은 것 = 그립 prior**(손 갑자기 펴짐).

---

## 1. 발전 타임라인 — 막다른 길과 돌파구

처음엔 "손목 v4 = 더 충실한 2D fit + 충돌/depth anchor"로 시작했으나, **반복적 실험·렌더 검증·사용자 피드백**으로 방향이 크게 바뀜:

1. **fit2d 한계 진단**: 손목만 2D fit하면 고정 GVHMR 전완 탓에 손목이 인체불가 각도(편위 88°)로 과굴절. → **팔(전완)이 병목**.
2. **armwrist/fitcol/zanchor (전역 재fit) 시도**: 2D 숫자는 좋아지나(8px), **팔/손목을 전 프레임 움직여** 사용자가 "기존이 더 잘 따라가"라 거부. → *전역 재fit은 답이 아님.*
3. **collonly (충돌만)**: fit2d 유지+충돌만. 떨림 발생 → ramp+delta평활로 해결. but 손목 충실도 문제 잔존.
4. **사용자 통찰 "z를 따라가며 2D fit 동시에"**: 실험 결과 z앵커 단독은 충돌 못 풀고(GVHMR 깊이에 박힘), 합치면 전역재fit이 됨을 데이터로 규명.
5. **★ 사용자 핵심 정정 "GVHMR 손목 위치가 틀렸다 → HaMeR 위치 추종 + 자연스러운 팔 IK"**: = **역기구학(IK)** 문제로 재정의. `cap_arm_prefit`(기존 튜닝됨) 재사용 → 손목 위치 14→1px. **이게 v4의 방향을 결정.**
6. **손가락 안 맞음 → 사용자 "손목 각도로 맞춰라(grip 고정)"**: rigid 손 한계(18px) 발견. + WB 손 2D가 노이즈라 HaMeR grip 신뢰가 옳음을 규명.
7. **★ 사용자 "HaMeR 손목 각도를 따라가야 (그게 완벽한 overlay)"**: 측정으로 Y(WB fit)가 HaMeR와 swing16°/axial55° 다름을 확인. → **손목 = HaMeR 직접 이식**.
8. **"잘 보이는 구간만 완벽, 가려짐/스윙은 뭉개짐" + "v4clean은 반대"** → **신뢰도 게이트**로 둘의 장점 결합 (HaMeR good + slerp coast bad).
9. **jitter → σ1.8 평활**(savgol·Kalman도 동등, 실제모션은 못줄임=lag 규명).
10. **★ 사용자 "위치 깜빡임, 원래 없었는데" (내 'GVHMR탓' 진단 반박)** → arm_prefit 깊이 미앵커가 원인 규명 → **depth anchor**(§4.3, 처음 적어두고 안 넣었던 것) 추가 → 깜빡 9→3.8cm.
11. **★ 사용자 "19-20초 팔 위치 튐, ViTPose fit 문제 아니냐"** → 가려짐 프레임에서 팔 위치 튐(실관측+5px인데 +32px) 규명 → **팔 위치 신뢰도 게이트** → +32→+8px.

---

## 2. 기여 (누가 무엇을)

### 사용자 (방향·핵심 통찰)
- **"전완이 문제다"** — 손목만 보지 말고 팔을 봐야 함을 직관.
- **"GVHMR 손목 위치가 틀렸으니 HaMeR 위치를 IK로 추종"** — v4의 결정적 재정의.
- **"HaMeR 손목 각도를 따라가야 완벽한 overlay"** — WB 2D fit이 아니라 HaMeR 직접 사용.
- **"손가락 grip은 고정, 손목 각도로 맞춰라"** — 노이즈한 WB 손가락 안 쫓는 게 옳음.
- **"위치 깜빡임은 원래 없었다", "가려짐 팔 튐은 ViTPose fit 탓 아니냐"** — Claude의 오진을 두 번 정확히 반박, depth anchor·arm gate로 이어짐.
- **"잘 보이는 구간 + 안 보이는 구간 장점 둘 다"** — 신뢰도 게이트의 동기.
- **"막 지우지 마", "best는 기록해라"** — 작업 위생.

### Claude (구현·측정·검증)
- 모든 단계를 **렌더 + 수치(2D RMSE, 속도, ROM, 깊이, 깜빡임)로 검증** (axial-invisible 원칙상 수치만으론 판정 금지 준수).
- 사용자 가설을 **데이터로 검증/반증** (z앵커 단독 무효, ViTPose-fit 무관 등).
- 신뢰도 게이트·depth anchor·arm gate·충돌 전파·jitter 평활 구현.
- 오진 2회(GVHMR탓) → 사용자 반박 후 즉시 재측정·수정.

---

## 3. 만든 스크립트
| 파일 | 역할 |
|---|---|
| `batch_hamer_wrist.py` | 손목=HaMeR + 신뢰도 게이트 + 평활 (★핵심) |
| `batch_rom_clamp.py` | 손목 ROM 클램프 |
| `batch_axcol_v4.py` | 충돌을 손목 axial로 해소 (2D 보존, 전파) |
| `batch_arm_gate.py` | 팔 위치 신뢰도 게이트 (가려짐 튐 제거) |
| `cap_arm_prefit.py` | (수정) `CAP_ARM_DEPTH` depth anchor 추가 |
| `collision_detect.py` | 메쉬 self-penetration 검출 (거리게이트) |
| `render_ab_range.py` | 구간 A/B 비교 렌더 |
| (실험/폐기) `batch_fit2d_v4 / armwrist_v4 / fitcol_v4 / zanchor_v4 / collonly_v4 / wristfit21 / finger_fit / armik_*` | 발전 과정의 시도들 |

## 4. 남은 일
- **그립 prior** (손가락 갑자기 펴짐) — 유일한 미해결.
- 8클립 일반화 (side_1/4 충돌 많음, betas 다양성에서 가중치 검증).
- 서비스 배선.
</content>
