# Motio 추출 속도 최적화 — 플래그/되돌리기 가이드

> 2026-06-03. Tier 1 최적화의 환경변수 토글 + 백업 위치. 언제든 되돌릴 수 있게 정리.

추출은 백엔드가 `entry_point_opt.py`(→`entry_point_hamer.py`/`entry_point.py`)와
`motio_backend/run_export.py` 를 **subprocess** 로 부르므로, 아래 env 만 바꾸면 즉시 반영(서버 재기동 불필요).

## 환경변수 토글

| 변수 | 기본 | 의미 / 되돌리기 |
|---|---|---|
| `CAP_HAMER_BATCH` | `64` | HaMeR 를 프레임 가로질러 큰 배치로 일괄 추론. 작게 주면 배치↓. |
| `CAP_HAMER_LEGACY` | (off) | `=1` 이면 **옛 per-frame HaMeR 루프**(배치 이전)로 복귀. |
| `CAP_HAMER_WORKERS` | `4` | HaMeR 크롭 준비(cv2.imread+리사이즈) 스레드 병렬 수. `1`/`0` = 순차(되돌리기). bit-exact. |
| `CAP_AMP` | `0`(OFF) | `=1` 이면 HaMeR+wholebody ViTPose **fp16** autocast. ⚠️ 손 포즈 max~2.9° 변하고 이득 ~8s뿐이라 기본 OFF. GVHMR 전처리는 cv2 비호환으로 적용 불가. |
| `CAP_GROUND` | `1` | FBX 접지(발을 Z=0). `=0` 이면 접지 생략(공중부양 원본, 2차 Blender 생략). |

## 측정(5s·301f 클립 기준)

- 배치 적용: HaMeR 51s → ~23s (2.25×). 채택.
- 크롭 병렬(workers=4): HaMeR ~23s → ~19s. bit-exact. 채택.
- fp16(CAP_AMP=1): before_render 71→63.5s 이지만 손 max 2.9°/p99 0.6° 변화 → 기본 OFF.
- 접지: 정확성 위해 완성 FBX import 후 grounding(`ground_fbx.py`, ~15s). in-proc 시도는 축 어긋나 폐기.
- 현재 기본값 전체 E2E: 추출 ~66s(before_render) + export+접지 ~15s.

## 보류(미적용)

- **트래킹/ViTPose 중복 제거(#3)**: 추출 내부 YOLO 는 이미 1회(캐시 재사용). 검출↔추출 중복만 존재하나
  검출 bbox 주입은 GVHMR 입력 person 이 바뀌어 모션 회귀 위험. 가장 큰 GVHMR 전처리 26s
  (ViTPose+ViT features+VO)는 모두 필수라 중복제거로 못 줄임. → 위험>이득, 보류.

## fps / 프레임 보존 수정 (2026-06-03)

증상: 60fps 원본인데 결과 FBX 가 30fps(슬로모션)에다 프레임이 깎임(600→250). 원인 3중 누수:
1. `run_export.stage_b` 가 `export_to_unreal_fbx.py` 에 **`--fps` 미전달** → 기본 30 으로 베이크. → 수정: `--fps {fps}` 전달(stage_b 에 fps 인자 추가, main 에서 args.fps 전달).
2. SMPL-X 애드온 `target_framerate` 기본 30 → `step_size=int(mocap/target)` 로 **다운샘플**. → 수정: 애드온 `operators/animation.py` 에 `SMPLX_KEEP_FPS`(기본 1) — target=mocap 강제(step=1, 프레임 보존). `=0` 이면 원래 동작.
3. 접지 패스 `ground_fbx.py` 재import 시 **scene fps=24 + 기본 frame_end=250** 로 truncation. → 수정: fps 인자 받아 import 전/후 scene fps 고정 + 임포트 액션 길이로 frame_end 확장.

검증: 600프레임 클립 → 최종 output.fbx **601프레임·60fps·10.02s** (손실 0). 기존 30fps 잡은 **재추출해야** 고쳐짐(이미 구워진 FBX 는 자동 안 바뀜).

토글: `SMPLX_KEEP_FPS=0` 이면 애드온 다운샘플(옛 동작). export fps 는 백엔드가 검출 fps 를 자동 전달.

## 선수 검출 필터 + YOLO 재사용 (2026-06-03)

### 선수 필터 (`detect_people._filter_players`, 기본 ON)
배경 관중 제거. `CAP_PLAYER_MIN_AREA`(0.02=화면2%) AND `CAP_PLAYER_MIN_PRESENCE`(0.5) 둘 다 만족만 선수로.
`CAP_PLAYER_MAX`(3, 복식). `CAP_PLAYER_FILTER=0` 끔. side 영상: 40명→1명(선수 14.2%). **단 시간은 ~1s만 절약**
(검출 시간의 99%는 YOLO 추론이라 사람 수와 무관). 가치는 시간이 아니라 정확도/UX(선수 자동 격리).

### YOLO 재추적 재사용 (`CAP_REUSE_DETECTION`, **기본 ON**, 2026-06-03 켬)
검출 YOLO 결과(_tracks.npz)를 추출이 재사용 → `prepare_selected_person_track` 의 YOLO 재추적 스킵.
- 효과: 추출 wall **~20-30s↓** (side 147→116s, front_#2 100→83s).
- bit-exact 아님(두 YOLO 트래킹이 완전 동일치 않아 박스 미세차 → 손 한 축 최악 ~30°, body ~3-4.5°, mean<0.7°).
  **단 side·front 두 영상 오버레이 검증에서 육안 차이 없음** 확인 → 기본 ON 채택(사용자 승인). 되돌리기 `CAP_REUSE_DETECTION=0`.
- 켜면 검출이 _tracks.npz 저장(이미 기본 생성, `CAP_REUSE_DETECTION=0` 으로 생성도 끔) +
  `inference.py` 가 `CAP_REUSE_BBX_NPZ` 전달 + `entry_point.prepare_selected_person_track` 가 재사용.

## 백업 파일

- `entry_point.py.bak_before_fp16`
- `entry_point_hamer.py.bak_before_fp16`
- `entry_point_hamer.py.bak_before_batch` (배치 리팩토링 이전)
- 폐기: `motio_backend/ground_after_export.py` (in-proc 접지, 축 어긋남 — 미사용)
- 애드온 keep-fps 백업: `smplx_blender_addon/operators/animation.py.bak_before_keepfps` (소스 + blender5_userres 설치본 양쪽)
- 검출필터/재사용 백업: `entry_point.py.bak_before_reuse`, `motio_backend/detect_people.py.bak_before_reuse`, `motio_backend/inference.py.bak_before_reuse`
