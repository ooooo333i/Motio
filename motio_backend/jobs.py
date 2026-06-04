"""추출 job 큐/상태/결과. 단일 GPU 워커(gpu.GPU_LOCK 으로 직렬화)."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from . import config, errors, exporters, inference, storage


# ── 생성/처리 ───────────────────────────────────────────
def create_job(video_id: str, clip_id: str, target_id: str,
               person_track_id: int | None, clip_path: str | Path) -> dict:
    job_id = storage.new_id("job")
    job = {
        "jobId": job_id,
        "status": "queued",
        "created_at": time.time(),
        "videoId": video_id,
        "clipId": clip_id,
        "targetId": target_id,
        "person_track_id": person_track_id,
        "clip_path": str(clip_path),
    }
    storage.save_job(job_id, job)
    asyncio.create_task(_process_job(job_id))  # 백그라운드 실행(엔드포인트 async 컨텍스트)
    return job


async def _process_job(job_id: str) -> None:
    job = storage.load_job(job_id)
    if job is None:
        return
    job["status"] = "running"
    job["started_at"] = time.time()
    storage.save_job(job_id, job)
    try:
        manifest = await inference.run_extraction(
            job_id, job["clip_path"], job.get("person_track_id")
        )
        job["manifest"] = manifest

        # FBX/BVH export (슬림 후처리 npz → Blender FBX). Blender 미설정 시 npz 까지만(fbx=None).
        final_pt = manifest["outputs"]["final"]
        out_dir = Path(final_pt).parent
        export_log = storage.job_dir(job_id) / "export.log"
        # 원본 영상 fps 로 FBX 를 구워야 재생 속도가 맞다(기본 30 으로 구우면 60fps 영상이
        # 절반 속도 슬로우모션으로 보임). manifest → 검출 JSON(fps) → 30 순으로 결정.
        det_meta = storage.load_detection(job.get("clipId", "")) or {}
        fps = float(manifest.get("fps") or det_meta.get("fps") or 30.0)
        try:
            files = await exporters.export_artifacts_async(final_pt, out_dir, export_log, fps=fps)
        except Exception as ex:
            files = {"fbx": None, "bvh": None}
            job["export_warning"] = str(ex)[:300]
        job["files_internal"] = files

        job["status"] = "complete"
        job["finished_at"] = time.time()
        storage.save_job(job_id, job)
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = {"code": errors.INFERENCE_FAILED, "message": str(exc)[:500]}
        job["finished_at"] = time.time()
        storage.save_job(job_id, job)


# ── 상태/결과 ───────────────────────────────────────────
def _recent_log_lines(job: dict, n: int = 12, tail_bytes: int = 32768) -> list[str]:
    """pipeline.log 끝부분에서 사람이 읽을 최근 줄 n개(연속 tqdm 진행막대는 최신만 유지)."""
    import re
    log = storage.job_dir(job["jobId"]) / "pipeline.log"
    if not log.exists():
        return []
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            sz = f.tell()
            f.seek(max(0, sz - tail_bytes))
            raw = f.read().decode("utf-8", "ignore")
    except Exception:
        return []
    raw = raw.replace("\r", "\n")
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    is_bar = lambda s: ("it/s" in s) or ("%|" in s) or ("█" in s)  # noqa: E731
    out: list[str] = []
    for ln in raw.split("\n"):
        ln = ansi.sub("", ln).rstrip()
        if not ln.strip():
            continue
        if is_bar(ln) and out and is_bar(out[-1]):
            out[-1] = ln  # 연속 진행막대는 최신 1개로 접음
        else:
            out.append(ln)
    return out[-n:]


def status_payload(job: dict) -> dict:
    out = {"jobId": job["jobId"], "status": job["status"]}
    if job["status"] == "failed" and job.get("error"):
        out["error"] = job["error"]
    pct, stage = compute_progress(job)
    out["progress"] = pct
    out["stage"] = stage
    out["log"] = _recent_log_lines(job)
    return out


# 추출 잡 단계 바닥(산출물 존재 → 최소 진행률·단계명). pct 오름차순; 존재하는 것 중 최댓값 채택.
#   포즈추출(GVHMR+HaMeR+merge) 0→82. 이후 정제는 mode 별로 산출물이 다름:
#   quality(adaptive_v4): arm_prefit→hamer_wrist→rom→col→grip,  fast(legacy): fit2d→fingerkalman.
#   두 모드 산출물을 모두 나열(한 잡엔 한 세트만 존재) + 공통 export.
_STAGE_FLOOR = [
    ("smplx_merged_hamer_post.pt",        82, "Hand merge"),
    # ── quality (adaptive_v4): 손목·손 정합 단계들 ──
    ("smplx_arm_prefit.pt",               84, "Wrist align"),
    ("hamer_wrist_hamer.pt",              86, "Wrist align"),
    ("hamer_wrist_hamer_rom.pt",          88, "Wrist align"),
    ("hamer_wrist_hamer_rom_col.pt",      90, "Wrist align"),
    ("hamer_wrist_hamer_rom_col_grip.pt", 93, "Finger cleanup"),
    # ── fast (legacy fit2d) ──
    ("fit2d.pt",                          88, "Wrist align"),
    ("fit2d_fingerkalman.pt",             92, "Finger cleanup"),
    # ── 공통 export ──
    ("postprocessed.npz",                 95, "FBX export"),
    ("output.fbx",                        99, "Finalizing"),
]
# 포즈추출(0~82) 내부 세부 단계: (로그 마커, sub바닥, sub천장, 라벨, min_total)
#   min_total: 이 phase 의 진행막대로 인정할 tqdm total 최소값.
#   HaMeR 구간은 프레임 수(수백)짜리 per-frame 막대만 잡아야 함 — 직전 단계의
#   완료된 작은 막대(예: ViTPose 19/19)가 last-line 에 남아 100%로 튀는 것 방지.
_POSE_PHASES = [
    ("YoloV8 Tracking",          0.05, 0.12, "Player tracking", 1),
    ("Running GVHMR preprocess", 0.12, 0.35, "Body pose",       1),
    ("Running GVHMR prediction", 0.40, 0.50, "Body pose",       1),
    ("for HaMeR",                0.55, 0.97, "Hand pose",       50),
]


def _latest_counter(lines: list[str], min_total: int = 1) -> float | None:
    """로그 뒤에서부터 tqdm 진행막대 'cur/total' 을 찾아 비율(0~1) 반환.

    total < min_total 인 작은 막대는 무시(엉뚱한 막대 오인 방지).
    완료막대(cur==tot)가 끝에 남아도 그 phase 의 막대라면 정상(100% 의미).
    """
    import re
    for ln in reversed(lines[-25:]):
        m = re.search(r"(\d+)\s*/\s*(\d+)", ln)
        if m:
            cur, tot = int(m.group(1)), int(m.group(2))
            if tot >= min_total and tot > 0:
                return max(0.0, min(1.0, cur / tot))
    return None


def compute_progress(job: dict) -> tuple[int, str]:
    """산출물 + pipeline.log 기반 *실제* 진행률(0~100)과 사람이 읽는 단계명."""
    st = job.get("status")
    if st == "complete":
        return 100, "Complete"
    if st in (None, "queued"):
        return 0, "Queued"
    if st == "failed":
        return int(job.get("_last_progress", 0)), "Failed"

    out_root = storage.job_output_dir(job["jobId"])
    # 1) 완료된 산출물로 단계 바닥 결정(역순: 가장 진행된 것 우선)
    floor, stage = 0, "Initializing"
    for name, pct, label in _STAGE_FLOOR:
        if next(out_root.glob(f"**/{name}"), None) is not None:
            floor, stage = pct, label
    if floor >= 82:
        return min(floor, 99), stage

    # 2) 포즈추출(0~82) 내부 — pipeline.log 마커 + tqdm 카운터로 세부 진행
    log = storage.job_dir(job["jobId"]) / "pipeline.log"
    sub, label = 0.03, "Initializing / loading models"
    if log.exists():
        try:
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            lines = []
        cur_phase = None
        # 가장 최근에 등장한 phase 마커 선택
        last_idx = -1
        for marker, lo, hi, lab, min_tot in _POSE_PHASES:
            idx = max((i for i, ln in enumerate(lines) if marker in ln), default=-1)
            if idx > last_idx:
                last_idx, cur_phase = idx, (lo, hi, lab, idx, min_tot)
        if cur_phase:
            lo, hi, lab, idx, min_tot = cur_phase
            frac = _latest_counter(lines[idx:], min_tot) or 0.0
            sub, label = lo + (hi - lo) * frac, lab
    pct = int(min(81, max(1, round(sub * 82))))
    return pct, label


def _hand_coverage(clip_dir: Path, fallback_frames: int) -> tuple[float, int]:
    """smplx_merged_hamer_post.json 에서 (hand coverage 0~1, num_frames)."""
    report = clip_dir / "smplx_merged_hamer_post.json"
    if report.exists():
        try:
            r = json.loads(report.read_text(encoding="utf-8"))
            nf = int(r.get("num_frames") or fallback_frames or 0)
            if nf > 0:
                lf = float(r.get("left_hand_frames", 0))
                rf = float(r.get("right_hand_frames", 0))
                return (min(lf / nf, 1.0) + min(rf / nf, 1.0)) / 2.0, nf
        except Exception:
            pass
    return 0.0, int(fallback_frames or 0)


def build_result(job: dict) -> dict:
    """명세 4.8 result 응답 구성. (frames + 파생 poseAccuracy)"""
    manifest = job.get("manifest") or {}
    clip_dir = Path(manifest.get("clip_dir", ""))
    num_frames = int(manifest.get("num_frames") or 0)

    hand_cov, nf2 = _hand_coverage(clip_dir, num_frames)
    if not num_frames:
        num_frames = nf2

    # 선택 track 의 평균 confidence
    det = storage.load_detection(job["clipId"]) or {}
    track_conf = 0.0
    for o in det.get("objects", []):
        if str(o.get("id")) == str(job.get("targetId")):
            track_conf = float(o.get("confidence", 0.0))
            break

    pose_acc = max(0.0, min(100.0, 100.0 * (0.7 * track_conf + 0.3 * hand_cov)))

    files = job.get("files_internal") or {"fbx": None, "bvh": None}
    fbx_url = config.media_url(files["fbx"]) if files.get("fbx") else None
    bvh_url = config.media_url(files["bvh"]) if files.get("bvh") else None

    return {
        "jobId": job["jobId"],
        "files": {"fbx": fbx_url, "bvh": bvh_url},
        "metrics": {"frames": int(num_frames), "poseAccuracy": round(pose_acc, 1)},
    }


def recover_on_startup() -> None:
    """서버 재시작 시 떠 있던 job 을 failed 로(가짜 완료 방지)."""
    for j in storage.iter_jobs():
        if j.get("status") in ("queued", "running"):
            j["status"] = "failed"
            j["error"] = {"code": errors.INFERENCE_FAILED, "message": "server restarted during job"}
            storage.save_job(j["jobId"], j)
