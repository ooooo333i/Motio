# Motio 서비스 전체 흐름 (코드 기준, 최신 2026-06-04)

> 영상 업로드 → 3D 손-팔-몸 재구성(v4) → FBX/BVH. 실제 레포 코드의 호출 순서 그대로.
> **최신 변경**: 추론 기본 모드 = `quality`(= adaptive v4). 되돌리기 `CAP_OPT_MODE=fast`.

---

## 0. 한눈에 (API → 추론 → 익스포트)

```
[사용자] 영상 업로드
   │  motio_backend/main.py  (FastAPI)
   ▼
┌─────────────────────────────────────────────────────────────┐
│ A. API 레이어 (main.py)                                       │
│   POST /api/videos        → 업로드                            │
│   POST .../clip           → 구간 트림 (media.py)              │
│   POST .../detect         → 사람 검출 (detection.py)          │
│   POST .../select-target  → 추적 대상 person_track_id 선택    │
│   POST /api/extraction-jobs → 잡 생성 → 백그라운드 실행        │
│   GET  .../{id}           → 폴링 / 결과·다운로드              │
└─────────────────────────────────────────────────────────────┘
   │  jobs.py : _process_job(job_id)   (asyncio 백그라운드)
   ▼
┌─────────────────────────────────────────────────────────────┐
│ B. 잡 파이프라인 (jobs.py)                                    │
│   1) inference.run_extraction(clip, track_id) → manifest      │
│   2) exporters.export_artifacts_async(final_pt) → FBX/BVH     │
│   3) status=complete                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. 추론 (inference.py → entry_point_opt.py)

```
inference.run_extraction
   │  cmd = entry_point_opt.py --video --output-root --mode $CAP_OPT_MODE
   │        (CAP_OPT_MODE 기본 "quality";  CAP_REUSE_BBX_NPZ 로 YOLO 재추적 스킵)
   │  async with GPU_LOCK:  (단일 GPU 직렬화)
   ▼
entry_point_opt.run_opt(mode="quality")
   │
   ├─[1] _run_hamer_merge → entry_point_hamer.py  (env CAP_SKIP_WRIST_REFINE=1)
   │      = 전처리/검출/병합 (레거시 손목정제는 스킵, 검출물은 보존)
   │      ┌──────────────────────────────────────────────┐
   │      │ · GVHMR  : 전신 SMPL-X + 카메라 (hmr4d_results.pt)
   │      │ · ViTPose: COCO17 팔 (preprocess/vitpose.pt)
   │      │            WholeBody 133 손 (preprocess/wholebody_kp2d.npy)
   │      │ · HaMeR  : 손 MANO 검출 (hamer_out_clean/)
   │      │ · merge  : GVHMR 바디 + HaMeR 손가락
   │      │            → smplx_merged_hamer_post.pt   (= "Naive graft")
   │      └──────────────────────────────────────────────┘
   │
   └─[2] _refine_quality → ★ adaptive v4 체인 (run_v4.sh 와 동일)
          A  cap_arm_prefit      팔 IK + depth앵커 + ViTPose conf 게이팅 → smplx_arm_prefit.pt
                                 (손목 위치 14→3.5px)        env CAP_ARM_DEPTH=100000
          A2 batch_arm_smooth    팔 회전 conf-게이트 평활 (가려짐 떨림 제거, 가시 보존)
          B  batch_hamer_wrist   손목 = HaMeR 각도 + 신뢰도 게이트 + 평활 → hamer_wrist_hamer.pt
          C  batch_rom_clamp     전완축 기준 생체역학 손목 한계 → …_rom.pt
          D  collision_detect    손-손/손-몸 메쉬 관통 검출 → collision_*.json
             batch_axcol_v4      충돌을 2D-불변 축회전으로 해소(early-stop) → …_rom_col.pt
          E  batch_grip_prior    손가락만: IoU 겹침 + conf 게이트 (body_pose 불변) → …_rom_col_grip.pt
          ────────────────────────────────────────────────────────────
          최종 = hamer_wrist_hamer_rom_col_grip.pt   (manifest.outputs.final)
```

> 입력 의존: A는 hmr4d+vitpose, B는 merged_post+arm_prefit+hamer_out_clean+wholebody,
> D/E는 wholebody. (fit2d 불필요 — grip의 betas는 base pt에서 폴백.)

---

## 2. 익스포트 (exporters.py → Blender)

```
exporters.export_artifacts_async(final_pt, fps)
   │  · 슬림 후처리 → postprocessed.npz
   │  · Blender 구동 → FBX (Unreal Z-up) / BVH
   │  · Blender 미설정 시 npz 까지만 (fbx=None)
   ▼
산출: output.fbx / .bvh  (job 결과로 다운로드)
```

---

## 3. 데이터 산출물 맵 (clip_dir = output_root/<clip>/)

| 단계 | 파일 | 생성자 |
|---|---|---|
| 전처리[1] | `hmr4d_results.pt`, `preprocess/vitpose.pt`, `preprocess/wholebody_kp2d.npy`, `hamer_out_clean/`, `smplx_merged_hamer_post.pt` | entry_point_hamer |
| v4-A | `smplx_arm_prefit.pt` | cap_arm_prefit (+arm_smooth) |
| v4-B~D | `hamer_wrist_hamer{,_rom,_rom_col}.pt`, `collision_*.json` | hamer_wrist / rom_clamp / axcol |
| v4-E(최종) | `hamer_wrist_hamer_rom_col_grip.pt` | grip_prior |
| 익스포트 | `postprocessed.npz`, `output.fbx/.bvh` | exporters |

---

## 4. 모드 스위치
| 모드 | 경로 | 산출 final | 용도 |
|---|---|---|---|
| **quality (기본)** | _refine_quality (v4 체인) | `hamer_wrist_hamer_rom_col_grip.pt` | adaptive v4 (정확·물리타당) |
| fast (레거시) | _run_batch_refine | `fit2d_fingerkalman.pt` | fit2d+finger Kalman (빠름) |

전환: `CAP_OPT_MODE=fast` (inference.py) 또는 `--mode fast` (CLI).

---

## 핵심 파일 위치
- API: `motio_backend/main.py`, 잡: `motio_backend/jobs.py`, 추론어댑터: `motio_backend/inference.py`, 익스포트: `motio_backend/exporters.py`
- 오케스트레이터: `entry_point_opt.py` (`run_opt` / `_run_hamer_merge` / `_refine_quality`)
- 전처리: `entry_point_hamer.py`
- v4 체인: `cap_arm_prefit.py`, `batch_arm_smooth.py`, `batch_hamer_wrist.py`, `batch_rom_clamp.py`, `collision_detect.py`, `batch_axcol_v4.py`, `batch_grip_prior.py`
- 알고리즘 상세: `docs/V4_ALGORITHM.md` / 근거: `poster/V4_EVIDENCE.md`
