# MOTIO_INTEGRATION.md — 프론트↔백엔드↔추론서버 연결 (살아있는 문서)

> 작성 2026-06-02. **이 문서는 작업 진행에 따라 계속 갱신한다(Claude 참조용).**
> 목적: motio-frontend(고정) ↔ motio_backend(신규) ↔ 추론서버(capstone_deploy)를 연결해
> "한 줄 명령 → 어디서든 브라우저 접속" 가능한 실동작 웹 서비스를 완성한다.
> 관련: `docs/CONTEXT_OVERVIEW.md`(파이프라인 맥락), `/workspace/motio-frontend/docs/*`(API 명세·어댑터 설계).

---

## 0. 확정된 결정 (사용자, 2026-06-02)

| # | 항목 | 결정 |
|---|---|---|
| D1 | 서비스가 내릴 정제 산출물 | **batch 체인(`fit2d_fingerkalman.pt`)** 우선. = entry_point_hamer → batch_fit2d → batch_finger_kalman. **당장은 서비스 연결 최우선.** axial/bad_frame 일반화는 **추후 반드시** 추가할 것이므로 entry_point_opt를 **확장형(mode 플래그)** 으로 설계. |
| D2 | FBX export | **사용자 제공 `postprocess_pipeline/export_to_unreal_fbx.py`(동결) 연동.** 손대지 않음. BVH는 확장 가능하게 hook만. |
| D3 | 공개 접속 | **RunPod 프록시 URL**(`https://<podid>-<port>.proxy.runpod.net`). 한 줄 스크립트로 백엔드+프론트 기동 후 링크 출력. |
| C1 | 제약 | **cap-deploy conda 환경 변경 금지.** 중요 파일은 무조건 백업. **삭제 금지(격리만).** 기존 결과 동일 재현(회귀 검증). |

---

## 1. 목표 아키텍처

```
브라우저(어디서든)
  → RunPod 프록시 URL
    → 프론트(motio-frontend, Vite/React, 정적 빌드 or dev)
    → motio_backend (FastAPI, 신규)  ── API 명세(motio-demo-api-spec.md) 그대로 구현
        ├─ 검출 어댑터(Tracker 기반)         : select 화면용 사람 track
        ├─ entry_point_opt.py (추론)         : 정제 .pt 생성 (batch 체인)
        ├─ postprocess_pipeline (별도 env)    : .pt → .npz → FBX (동결 스크립트)
        └─ 정적 /media                        : preview.mp4, output.fbx/bvh
```

핵심 매핑: `프론트 targetId(YOLO track_id 문자열) → 백엔드 person_track_id(int) → entry_point_opt --person-track-id`.

플로우: `upload → trim → detect → select-target → extraction-job → poll → result/download`.

---

## 2. 현실 점검 (문서 ≠ 코드, 2026-06-02 검증 완료)

> ⚠️ frontend/docs의 `current_api_flow.md`는 **마치 구현된 듯** 쓰여 있으나 실제로는 대부분 미구현. 아래가 사실.

- ❌ `capstone_deploy/motio_backend/` **없음** → 처음부터 구축.
- ❌ FBX/BVH export: repo에 없었음 → 사용자가 `postprocess_pipeline/export_to_unreal_fbx.py` 추가함(✅). 단 **Blender 미설치**(별도 env 필요).
- ✅ `entry_point_hamer.py`: `--person-track-id/--person-select-ui/--auto-person/--no-interactive/--skip-result-video` 존재. `CAP_SKIP_WRIST_REFINE=1` → merge 직후 정지(= `smplx_merged_hamer_post.pt` + `preprocess/wholebody_kp2d.npy`).
- ✅ 검출 빌딩블록(`entry_point.py`): `Tracker`, `build_person_track_candidates`, `select_person_candidate(person_track_id)`, `prepare_selected_person_track`, `run_gvhmr(person_track_id)`.
- ✅ **batch 체인은 일반화됨**: `batch_fit2d.py <clip>`(K 동적), `batch_finger_kalman.py <clip>`.
- ❌ **adaptive_v3 체인(wrist_pipeline/01·02·03)은 일반화 안 됨**: `base='.../input/'` 하드코딩, CLI 없음, 03은 tennis 전용 `bad_frames.json` 의존 → 임의 영상 불가. (D1의 "추후 일반화" 대상.)
- `Tracker.track()`은 `confidence`를 버림 → **tracker.py 수정 안 하고** 검출 스크립트(`detect_people.py`)가 YOLO track 루프를 자체 복제해 conf 직접 수집(회귀 위험 0).
- **백엔드 실행 환경 = cap-deploy** (fastapi/uvicorn/multipart 이미 존재 → 설치 불필요, C1 준수). ffmpeg=/usr/bin. **API 워커는 torch 미로드**, 검출·추론은 모두 subprocess(단일 GPU 직렬화·VRAM 해제).

---

## 3. 구현 단계 (체크리스트)

- [x] **P1. entry_point_opt.py** ✅ : batch 체인 오케스트레이터, 확장형(mode=fast/quality), manifest 출력. 기존 파일 무수정. (`run_opt()` + CLI)
- [x] **P1-v. 회귀 검증** ✅ : front_1 입력을 임시 dir(`outputs/_regress_opt/`)에 복사해 `_run_batch_refine` 실행 → 기존 `fit2d_fingerkalman.pt`와 **max|Δ|=0 (bit-exact 재현)**. 기존 결과 무손상.
- [x] **P2. motio_backend/** ✅ : FastAPI 전 엔드포인트 구현 + 검출/추론/export 어댑터 + 단일GPU 락 + job 큐 + 에러포맷 + /media + SPA 서빙. **스모크 검증**(upload→clip(trim+preview)→실YOLO검출(%bbox)→select-target→에러포맷) + **실 uvicorn 바인드 검증**(health/docs/SPA/upload) 통과.
- [x] **P2-e. E2E 추출 검증** ✅ : entry_point_opt 풀체인(GVHMR+HaMeR+merge→batch_fit2d 2.6px→finger_kalman→manifest num_frames=121) 2s 클립에서 exit0. = 백엔드 job 이 쓰는 추출 경로 실증.
- [x] **P3. FBX/BVH export** ✅(배선+StageA) : `run_export.py` 슬림 후처리 `.npz`(rotation_postprocess, numpy/scipy, pyvista/torch-smplx 불필요) **검증**(키/shape 계약 일치). Stage B(Blender→FBX) **배선 완료**(env `MOTIO_BLENDER` 게이트, best-effort). BVH hook 열어둠.
  - ⚠️ 실제 FBX 생성은 **Blender + SMPL-X 애드온** 설치 필요(이 서버 미설치 = 런-원스 갭). 미설정 시 npz 까지만, `files.fbx=null`(명세 허용)로 서비스 정상.
- [x] **P4. 프론트 실연결 + 배포 + FBX** ✅(애드온 제외) :
  - Node v20.18.1 `/workspace/node` 설치 → 프론트 **실 API 재빌드**(`VITE_USE_MOCK_API=false`, 단일오리진). 누락 mock 자산(asset/bvh,fbx)은 dist 번들본에서 복구.
  - **라이브 서비스 8000 기동** + **전체 API E2E 통과**: upload→clip→detect→select→job(queued→running→complete ~70s)→result `{frames:121, poseAccuracy:96.5}`. (버그 2개 수정: extraction-jobs async, inference 인자.)
  - **Blender 5.1.2 `/workspace/blender5` 설치** → 템플릿 읽기 OK + export 파이프라인이 `smplx_add_animation` 호출 지점까지 도달. FBX 풀배선(절대경로 수정).
  - ⚠️ **남은 1개**: **SMPL-X Blender 애드온**(`smplx_add_animation`, smpl-x.is.tue.mpg.de 로그인 월·사용자 자산)을 Blender5.1 에 설치하면 FBX 자동 생성. 미설치 시 npz 까지·`files.fbx=null`(서비스 정상).
  - 한 줄 실행: `bash run_motio.sh` (Blender5.1 자동감지, 프록시 URL 출력).

---

## 4. P1 설계 — entry_point_opt.py

위치: `capstone_deploy/entry_point_opt.py` (신규, 기존 무수정).

- **CLI**: `--video`(필수), `--output-root`, `--person-track-id`, `--auto-person`, `--no-interactive`, `--mode {fast,quality}`(기본 fast), `--skip-result-video`, `--hamer-root`, `--force`, `--manifest`.
- **함수 진입점**: `run_opt(video, output_root, person_track_id=None, mode="fast", ...) -> dict(manifest)` — 백엔드가 import 호출 가능. CLI도 동일 호출.
- **mode=fast (D1, 현재 서비스 경로)**:
  1. subprocess: `python entry_point_hamer.py --video V --output-root R --person-track-id ... --auto-person --no-interactive --skip-result-video` + env `CAP_SKIP_WRIST_REFINE=1`, `PYOPENGL_PLATFORM=''`.
  2. stdout에서 `smplx_merged_hamer_post.pt` 경로 파싱 → `clip_dir = parent`.
  3. subprocess: `python batch_fit2d.py <clip_dir>` → `fit2d.pt`.
  4. subprocess: `python batch_finger_kalman.py <clip_dir>` → `fit2d_fingerkalman.pt` (= 최종).
- **mode=quality**: 미구현 — `NotImplementedError`(명시적). 추후 02/03 일반화를 끼울 hook 위치 주석으로 표시.
- **출력 manifest(JSON)**: `{status, mode, clip_dir, outputs:{merged_post, fit2d, final}, num_frames}` → stdout `[OPT_MANIFEST] {...}` + `<clip_dir>/opt_manifest.json`. (backend가 final_pt/num_frames 파싱.)
- **회귀 안전**: 기존 스크립트를 *그대로* subprocess 호출 → 산출물이 수동 실행과 100% 동일. 각 단계 fresh process로 VRAM 해제.

> 확장 메모: D1대로 axial/bad_frame 일반화 시 `mode=quality`에서 (a) batch_fit2d 대신 일반화 faithful-fit, (b) 일반화 adaptive-axial(GVHMR 참조), (c) 자동 bad_frame 검출 후 slerp 를 끼운다. bad_frame은 영상별 생성 필요(현재 tennis 전용 json 의존).

---

## 5. 미해결/갭 (계속 갱신)

- Blender 미설치 → P3 전에 설치(별도 경로, cap-deploy 불가침). `export_to_unreal_fbx.py`의 정확한 입출력 계약은 P3에서 정독.
- postprocess_pipeline 의존(pyvista, smplx, blender)이 cap-deploy와 충돌 가능 → **별도 env** 필요(어떤 env로 돌릴지 P3에서 결정).
- 세션 토큰/업로드 상한/검출 동기·비동기/bbox 단위(%) 등 명세 §6 "협의 필요" 항목 → 백엔드 구현 시 기본값 채택(% 좌표, 검출 동기 MVP 등) 후 문서화.
- RunPod 8000 포트 노출 여부(대시보드) — 프록시 접속 전제. URL: `https://<POD_ID>-8000.proxy.runpod.net`.
- ✅ Blender 5.1.2 설치(`/workspace/blender5`) + Node v20(`/workspace/node`) 설치 완료.
- ⚠️ **SMPL-X Blender 애드온만 미설치** = FBX 유일 블로커. 사용자가 애드온 zip 제공 시 `/workspace/blender5` extensions 에 설치 → FBX 자동(run_export.py 배선 완료).
- ⚠️ **RunPod 대시보드에서 8000 HTTP 포트 노출** 필요(프록시 URL 접속 전제).
- 라이브 서버: 백그라운드 uvicorn(0.0.0.0:8000), 로그 `outputs/motio_server.log`. 재기동은 `bash run_motio.sh`.
- 검증 산출 스크래치: `outputs/_motio_smoke*`, `_motio_live`, `_motio_e2e`, `_regress_opt`, `_export_*`(삭제 안 함, 정리 시 격리).

---

## 6. 변경 로그

- 2026-06-02: 문서 생성. 결정 D1~D3/C1 기록. 현실 점검 완료. P1(entry_point_opt) 착수.
- 2026-06-02: **P1 완료** — `entry_point_opt.py` 생성(기존 파일 무수정). 구조검증(import/--help) + **P1-v 회귀검증 bit-exact 통과**(front_1 재현 max|Δ|=0, 기존 결과 보존). 다음=P2(motio_backend).
- 2026-06-02: **P2 완료** — `motio_backend/`(12파일, FastAPI) 구축. 검출은 tracker.py 무수정(detect_people.py 가 conf 자체수집). 스모크+실uvicorn 검증 통과. **P2-e E2E 추출** exit0(2s클립). cap-deploy 무변경(C1).
- 2026-06-02: **P3 완료(배선+StageA)** — `run_export.py` 슬림 후처리 npz 검증(계약 일치). Stage B(Blender) 배선·게이트. 갭: Blender+SMPL-X 애드온 설치(FBX 실제생성), node/npm(프론트 실연결). 다음=P4.
- 2026-06-03: **P4 완료(애드온 제외)** — Node20 설치+프론트 실API 재빌드, **라이브 8000 + 전체 API E2E 통과**(result frames121/poseAcc96.5), Blender5.1.2 설치(템플릿 OK·FBX 풀배선). 유일 잔여=SMPL-X 애드온(사용자 자산) + RunPod 8000 노출. **서비스 접속 가능 상태 완성.**
- 2026-06-03: **잔여 2개 모두 해소 = 서비스 완전 완성.**
  - ① **SMPL-X Blender 애드온 설치+FBX 검증 완료.** 사용자가 `/workspace/smplx_blender_addon`(신형 extension, `blender_manifest.toml`)에 업로드. `extension build`→`install-file --repo user_default --enable`로 설치하되, **유저리소스를 `/workspace/blender5_userres`(영속)로 고정**(`/root`는 휘발성). build/install 왕복에서 대용량 모델 blend(`smplx_model_lh_20230302.blend`, 345MB)가 truncate돼 "Missing DNA block" → **원본에서 data/* 재복사로 해결**(업로드가 진행 중이었어서 lsof로 완료 대기 후 복사). 백그라운드 모드(`blender -b -P`)에서 `smplx_add_animation` 등록 확인. **run_export.py 풀체인 실행 → output.fbx 161MB 정상 생성**(front_1 fit2d_fingerkalman.pt, T=1297). `run_motio.sh`+`/root/.bashrc`에 `BLENDER_USER_RESOURCES` export 추가(백엔드 서브프로세스 상속·재시작 대비).
  - ② **RunPod 8000 포트 노출 완료(사용자)** + **라이브 서버 재기동·검증.** 프록시 URL `https://n4zzj8bie0ydsi-8000.proxy.runpod.net` 외부 접속 = `/`(SPA 200), `/docs`(200), `/healthz`({ok:true} 200), `/assets/index-*.js`(1.14MB 200) 전부 통과. 로컬 8000도 동일.
  - ⇒ **남은 작업 없음.** 추후 과제는 D1의 axial/bad_frame 일반화(entry_point_opt mode=quality)뿐.
- 2026-06-03: **타이밍/병목 계측 추가 + 실측.** 무침투 계측 3종: ① `gpu.run_logged` 가 pipeline/export 로그 각 줄에 `[HH:MM:SS.mmm]` 부착(반환 문자열은 원본 유지→`[OPT_MANIFEST]` 파싱 보존). ② `main.py` http 미들웨어가 업로드/클립/검출/선택/잡생성 소요를 `outputs/motio_service/access_trace.log` 에 기록. ③ 신규 `motio_backend/trace.py` (`report`/`watch`): 산출물 mtime+잡 타임스탬프+access_trace 로 단계별 표+병목 지목. 사용: `python -m motio_backend.trace watch|report [--job ID]`.
  - **실측(front_#1.mp4, 5s/301프레임, 단일 사람) 전구간 106s, FBX 149MB/poseAcc 96.5 정상.** 병목 = **포즈추출(GVHMR+HaMeR+merge) 96s = 76%.** 그 안을 로그로 쪼개면 setup+YOLO트래킹 ~20s, GVHMR(ViTPose 포함) ~30s, **HaMeR(손)+merge ~51s(최대 단일 비용)**. 이후 batch_fit2d ~8s, finger_kalman ~2s, export(npz+Blender FBX) ~8s, 사람검출(HTTP) 12s. ⇒ 최적화 1순위 = HaMeR(배치/해상도/프레임 스킵), 2순위 GVHMR.
- 2026-06-03: **UX 3건 수정(사용자 피드백).**
  - ① **검출 추적 부드럽게.** 백엔드 `detect_people.py` `_smooth_xyxy`(박스 중심·크기 이동평균, win=5, 프레임 수 보존)로 YOLO jitter 완화. 프론트 `DetectionVideo.tsx`는 `onTimeUpdate`(~4Hz) 대신 **requestAnimationFrame(~60Hz)**로 currentTime 추적 + `currentFrame` 분수화(반올림 제거) → BBoxOverlay 보간이 매 프레임 부드럽게.
  - ② **processing 92% 가짜 → 실제 진행률.** 기존 프론트는 `Math.min(92, +4/900ms)` 가짜 애니메이션이라 92%에서 멈춘 듯 보였음. 백엔드 `jobs.compute_progress()` 신설: 산출물(.pt/.npz/.fbx) 단계 바닥 + `pipeline.log` tqdm 카운터로 **실제 0~100% + 단계명**(status 응답에 `progress`,`stage` 추가). HaMeR 구간은 프레임수(≥50)짜리 막대만 잡아 직전 완료막대 오인 방지. 프론트 `ProcessingPage.tsx`는 가짜 인터벌 제거하고 실제 progress 사용(단조 보정 `Math.max`), 단계명 배지 표시. 실측: 사람트래킹→전신포즈→HaMeR(45→80)→2D피팅82→손가락89→FBX92→100 매끄럽게.
  - ③ **FBX 뷰어 누움 보정.** export 는 Unreal 프리셋 **Z-up**(다운로드 FBX 는 그대로 둠=정답), 웹뷰어 three.js 는 **Y-up** 이라 캐릭터가 옆으로 누워 보였음. `FbxViewer.tsx` 에서 로드 직후 루트 `rotation.set(-π/2,0,0)`(Z-up→Y-up) 보정 — bbox/센터링/바닥은 회전 후 계산해 정합. (export_to_unreal_fbx.py 는 동결 유지.)
  - 프론트 재빌드(Node20) + 백엔드 재기동 완료. 검증: 진행률 실측 OK, 검출 평활화 OK, 뷰어 회전은 브라우저 육안 확인 필요(누움 원인=Y/Z up 불일치 확정).
- 2026-06-03: **결과 FBX 재생속도·접지 수정(사용자 피드백 "느림 + 공중부양").**
  - ④ **재생 느림 = fps 불일치.** manifest 에 fps 가 없어 export 가 30fps 로 베이크 → 원본 59.9fps 영상이 절반 속도 슬로모션. `jobs.py` 에서 export fps 를 `manifest.fps → 검출 JSON fps(storage.load_detection) → 30` 순으로 결정하도록 수정. 이제 원본 fps 로 구워 실시간 재생.
  - ⑤ **공중부양 = 단안 추정이 지면 기준 아님(카메라 공간).** 실측: FBX 발 바닥 Z=0.41m 떠 있음. SMPL-X 공간 접지는 애드온/AMASS 변환의 수직 오프셋 때문에 최종 FBX 와 불일치(실험 확인). → 신규 `motio_backend/ground_fbx.py`: export 후 Blender 로 **완성 FBX 를 다시 열어 실제 메시 월드 Z 최저값(프레임 샘플, 하위 5%)을 구해 루트(armature.delta_location.z)를 내려 발을 Z=0 에** 앉히고 동일 Unreal 설정으로 재출력. `run_export.stage_b` 가 output.fbx 생성 직후 호출(실패 시 원본 유지). 검증: 발 Z 0.41→0.02, 키 1.77m 정상. 다운로드 FBX 도 접지되어 Unreal 에서도 바닥에 섬.
  - (export_to_unreal_fbx.py 는 계속 동결. 접지는 별도 후처리로 분리.)
- 2026-06-03: **HaMeR 배치 추론 최적화(병목 단축, 사용자 승인).**
  - 원인: `entry_point_hamer.py` HaMeR 루프가 프레임당 1~2 손 크롭을 `--hamer-batch-size`(기본 1) 로 순차 추론 → ViT-Huge forward 가 크롭 하나씩 → GPU 거의 idle(5.9fps). `--hamer-batch-size` 를 올려도 프레임당 손이 1~2개뿐이라(ViTDetDataset 이 프레임 단위) 효과 없음 = 구조 문제.
  - 수정: **모든 프레임의 손 크롭을 먼저 수집 → 프레임 가로질러 큰 배치(`CAP_HAMER_BATCH`, 기본 64)로 일괄 추론.** 크롭 계산(ViTDetDataset)·저장 npz 포맷은 그대로. `CAP_HAMER_LEGACY=1` 로 기존 per-frame 경로 복귀(폴백). 백업: `entry_point_hamer.py.bak_before_batch`.
  - 동치성: ViT=LayerNorm 이라 배치 구성 무관 → 출력 동일해야 정상. 검증(동일 클립 602 손 npz + 최종 merged 손포즈): max|Δ| 정점 1.5e-6, mano 2.7e-5, **최종 left/right_hand_pose 5e-6 rad(≈0.0003°)**, global/body/transl=0 정확. = fp32 누적순서 반올림뿐, 품질 변화 아님. 사용자 승인 후 채택.
  - 속도: **HaMeR 51.2s→22.7s(2.25×), 추출(render전) 99s→71s(~28%).** 남은 병목: 이제 HaMeR 22.7s 의 대부분은 프레임별 cv2.imread+크롭(CPU single-thread), 그리고 GVHMR 전처리 26s 가 단일 최대. 다음 최적화 후보 = 크롭 로딩 병렬화 + GVHMR 전처리.
