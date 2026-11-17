# motio_backend — Motio 백엔드 어댑터

프론트(motio-frontend) ↔ 추론서버(capstone_deploy) 연결 FastAPI 어댑터.
프론트 명세(`motio-frontend/docs/motio-demo-api-spec.md`)를 그대로 구현하고, 내부에서
검출(`detect_people.py`)·추론(`entry_point_opt.py`)·후처리/FBX(`run_export.py`)를 **subprocess**로 호출한다.
설계·결정: `../docs/MOTIO_INTEGRATION.md`.

## 실행

```bash
# 한 줄 실행(권장): 백엔드 + (있으면)프론트 단일 오리진 + 공개 URL 출력
bash /workspace/capstone_deploy/run_motio.sh

# 또는 직접
cd /workspace/capstone_deploy
/workspace/envs/cap-deploy/bin/python -m uvicorn motio_backend.main:app --host 0.0.0.0 --port 8000
```

- 백엔드는 **cap-deploy 환경**에서 실행(추가 설치 불필요; fastapi/uvicorn 이미 존재).
- API 워커는 torch 를 로드하지 않음 — 검출/추론은 fresh subprocess(단일 GPU 직렬화·VRAM 해제).
- 공개 접속: RunPod 프록시 `https://<POD_ID>-8000.proxy.runpod.net` (대시보드에서 8000 HTTP 포트 노출 필요).

## 환경변수

| 변수 | 기본 | 설명 |
|---|---|---|
| `MOTIO_PORT` | 8000 | 포트 |
| `MOTIO_SERVICE_ROOT` | `outputs/motio_service` | 데이터/산출 루트(`/media` 로 공개) |
| `MOTIO_MAX_UPLOAD_BYTES` | 2GB | 업로드 상한 |
| `MOTIO_CORS_ORIGINS` | `*` | (단일 오리진 서빙 시 불필요) |
| `MOTIO_BLENDER` | (없음) | Blender 실행파일. **설정 시 결과 FBX 생성**, 미설정 시 후처리 npz 까지만 |
| `CAP_PYTHON` | cap-deploy python | 체인 단계 인터프리터 |

## 엔드포인트 (명세 그대로)

`POST /api/videos` · `POST /api/videos/{vid}/clip` · `POST|GET /api/videos/{vid}/clip/{cid}/detect`
`POST /api/videos/{vid}/clip/{cid}/select-target` · `POST /api/extraction-jobs`
`GET /api/extraction-jobs/{jid}` · `GET .../result` · `GET .../download?format=FBX|BVH` · `GET /healthz`

플로우: upload → clip(트림+preview) → detect(동기) → select-target → extraction-job(비동기·폴링) → result/download.
매핑: 프론트 `targetId`(YOLO track_id str) → `person_track_id`(int) → `entry_point_opt --person-track-id`.

## FBX/BVH (export)

- 흐름: 정제 `.pt` → `run_export.py` 슬림 후처리 `.npz`(rotation_postprocess, numpy/scipy) → **Blender** `export_to_unreal_fbx.py` → `output.fbx`.
- **Blender 가 이 서버에 설치돼 있지 않으면 npz 까지만 만들고 `files.fbx=null`**(명세상 null 허용, 서비스는 정상 완료).
- FBX 를 실제로 생성하려면: Blender + SMPL-X Blender 애드온(`smplx_add_animation`) 설치 후 `MOTIO_BLENDER=<blender>` 설정.
  - `postprocess_pipeline/export_to_unreal_fbx.py`(동결)와 `smplx_template.blend` 는 준비됨. **SMPL-X 애드온은 별도 설치 필요**.
- BVH: 확장 지점(현재 미생성). `run_export.py`/`exporters.py` 에 hook 만 열려 있음.

## 모듈

| 파일 | 역할 |
|---|---|
| `main.py` | FastAPI 앱(엔드포인트, CORS, /media, SPA 서빙, 에러핸들러) |
| `config.py` `errors.py` `schemas.py` | 설정 / 에러포맷 / pydantic 모델 |
| `storage.py` | JSON 상태 저장(videos/clips/detections/selections/jobs) + id 생성 |
| `media.py` | ffmpeg 트림(무손실 -c copy)/preview/probe |
| `detect_people.py` | (subprocess) YOLO 트래킹→검출 JSON(conf 수집, % bbox). **tracker.py 무수정** |
| `detection.py` | 검출 subprocess 어댑터 |
| `inference.py` | `entry_point_opt.py` subprocess 어댑터(manifest 파싱) |
| `gpu.py` | 단일 GPU 락 + async subprocess 실행기 |
| `jobs.py` | job 큐/상태/결과(파생 poseAccuracy)/재시작 복구 |
| `run_export.py` | (subprocess) 슬림 후처리 npz → (Blender)FBX |
| `exporters.py` | export 훅(async) |
