"""모션 추출 파이프라인 end-to-end 타이밍/병목 분석기 (stdlib 전용).

업로드 → 클립 → 검출 → 선택 → 추출잡(GVHMR+HaMeR+merge → 2D fit → finger kalman)
→ export(npz → Blender FBX) 의 각 단계 벽시계 소요를 모아 표로 보여주고,
가장 오래 걸린 단계(병목)를 짚는다.

데이터 출처(모두 무침투, 백엔드가 자연히 남기는 산출물):
  - access_trace.log         : HTTP 단계(업로드/클립/검출/선택/잡생성) 소요  (main.py 미들웨어)
  - jobs/<id>/status.json    : 잡 created/started/finished 타임스탬프
  - jobs/<id>/output/**/*.pt : 단계 산출물 mtime = 단계 경계
  - jobs/<id>/pipeline.log   : 타임스탬프 찍힌 추론 로그 (gpu.run_logged)
  - jobs/<id>/export.log     : export 로그

사용법:
  python -m motio_backend.trace report            # 최신 잡 사후 분석
  python -m motio_backend.trace report --job JOBID
  python -m motio_backend.trace watch             # 라이브(잡 진행 실시간 추적 + 병목표)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import config, storage

# 잡 output 안에서 찾을 단계 산출물(순서 = 파이프라인 순서). (라벨, glob)
JOB_ARTIFACTS = [
    ("포즈추출(GVHMR+HaMeR+merge)", "**/smplx_merged_hamer_post.pt"),
    ("2D fit",                      "**/fit2d.pt"),
    ("손가락 Kalman(최종 .pt)",      "**/fit2d_fingerkalman.pt"),
    ("export: 후처리 npz",          "**/postprocessed.npz"),
    ("export: Blender FBX",         "**/output.fbx"),
]

# access_trace.log 의 path 패턴 → 사람이 읽는 단계명
HTTP_STAGES = [
    ("업로드",      "POST", "/api/videos", lambda p: p == "/api/videos"),
    ("클립/트림",   "POST", "/clip",       lambda p: p.endswith("/clip")),
    ("사람 검출",   "POST", "/detect",     lambda p: p.endswith("/detect")),
    ("타겟 선택",   "POST", "/select-target", lambda p: p.endswith("/select-target")),
    ("잡 생성",     "POST", "/api/extraction-jobs", lambda p: p == "/api/extraction-jobs"),
]


def _fmt(sec: float | None) -> str:
    if sec is None:
        return "   —   "
    if sec < 90:
        return f"{sec:6.1f}s"
    return f"{sec/60:6.1f}m"


def _bar(sec: float, total: float, width: int = 28) -> str:
    if total <= 0:
        return ""
    n = int(round(width * sec / total))
    return "█" * n + "·" * (width - n)


def _latest_job() -> str | None:
    jobs = storage.iter_jobs()
    if not jobs:
        return None
    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return jobs[0].get("jobId")


def _artifact_mtime(out_root: Path, pattern: str) -> float | None:
    cands = sorted(out_root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return cands[-1].stat().st_mtime if cands else None


def _read_access_trace() -> list[dict]:
    rows: list[dict] = []
    p = config.SERVICE_ROOT / "access_trace.log"
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) >= 6:
            rows.append({"ts_str": parts[0], "t": float(parts[1]), "method": parts[2],
                         "path": parts[3], "status": parts[4], "dur": parts[5]})
    return rows


def _http_breakdown() -> list[tuple[str, float, str]]:
    """access_trace.log → [(단계명, 소요초, status)] (각 단계 마지막 호출 기준)."""
    rows = _read_access_trace()
    out: list[tuple[str, float, str]] = []
    for label, method, _key, match in HTTP_STAGES:
        hits = [r for r in rows if r["method"] == method and match(r["path"])]
        if hits:
            last = hits[-1]
            sec = float(last["dur"].rstrip("ms")) / 1000.0
            out.append((label, sec, last["status"]))
    return out


def _job_breakdown(job_id: str) -> tuple[dict, list[tuple[str, float | None]]]:
    job = storage.load_job(job_id) or {}
    out_root = storage.job_output_dir(job_id)
    created = job.get("created_at")
    started = job.get("started_at")
    finished = job.get("finished_at")

    # 단계 경계 시각: 시작 = started_at, 이후 각 산출물 mtime
    marks: list[tuple[str, float | None]] = []
    prev = started
    if created and started:
        marks.append(("큐 대기", started - created))
    for label, pattern in JOB_ARTIFACTS:
        mt = _artifact_mtime(out_root, pattern)
        if mt is not None and prev is not None:
            marks.append((label, mt - prev))
            prev = mt
        else:
            marks.append((label, None))
    return job, marks


def report(job_id: str | None) -> int:
    job_id = job_id or _latest_job()
    if not job_id:
        print("분석할 잡이 없습니다 (아직 추출을 한 번도 안 돌림). 먼저 영상→추출을 실행하세요.")
        # HTTP 단계만이라도 보여줌
        http = _http_breakdown()
        if http:
            print("\n[HTTP 단계(잡 이전)]")
            for label, sec, status in http:
                print(f"  {label:24s} {_fmt(sec)}  (HTTP {status})")
        return 1

    job, marks = _job_breakdown(job_id)
    http = _http_breakdown()

    print(f"\n══════ 파이프라인 타이밍 분석 — job {job_id} ══════")
    print(f"상태: {job.get('status','?')}", end="")
    if job.get("created_at") and job.get("finished_at"):
        print(f"   총(큐포함): {_fmt(job['finished_at'] - job['created_at'])}")
    else:
        print()

    segments: list[tuple[str, float]] = []
    if http:
        print("\n── 잡 이전 (HTTP 요청) ──")
        for label, sec, status in http:
            print(f"  {label:26s} {_fmt(sec)}   HTTP {status}")
            segments.append((label, sec))

    print("\n── 추출 잡 ──")
    for label, sec in marks:
        segments.append((label, sec if sec else 0.0))

    total = sum(s for _, s in segments) or 1.0
    print(f"\n{'단계':<28}{'소요':>9}   {'비중':<30}")
    print("─" * 74)
    for label, sec in segments:
        pct = 100.0 * sec / total
        print(f"{label:<28}{_fmt(sec):>9}   {_bar(sec, total)} {pct:4.1f}%")
    print("─" * 74)
    if segments:
        bn = max(segments, key=lambda x: x[1])
        print(f"⮕ 병목: 「{bn[0]}」 {_fmt(bn[1])} (전체의 {100.0*bn[1]/total:.0f}%)")

    if job.get("status") == "failed":
        print(f"\n⚠ 실패: {job.get('error',{}).get('message','?')}")
    if job.get("export_warning"):
        print(f"⚠ export 경고: {job['export_warning']}")
    print(f"\n로그: {storage.job_dir(job_id)}/pipeline.log , export.log")
    return 0


def _gpu_sample() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        u, m = r.stdout.strip().split(",")
        return f"GPU {u.strip():>3}% / {int(m):>5}MiB"
    except Exception:
        return "GPU   ?"


def _running_stage() -> str:
    try:
        r = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=5)
        txt = r.stdout
        for key, name in [("blender", "Blender/FBX"), ("hamer", "HaMeR"),
                          ("gvhmr", "GVHMR"), ("batch_finger_kalman", "finger-kalman"),
                          ("batch_fit2d", "2D-fit"), ("entry_point", "추론체인"),
                          ("detect_people", "검출"), ("ffmpeg", "ffmpeg/트림")]:
            if key in txt.lower() if key in ("blender", "hamer", "gvhmr") else key in txt:
                return name
    except Exception:
        pass
    return "—"


def watch(job_id: str | None, poll: float = 1.5) -> int:
    print("라이브 추적 시작. 영상 업로드→추출을 진행하세요. (Ctrl-C 종료)\n")
    seen_job: str | None = job_id
    seen_artifacts: set[str] = set()
    seen_http = 0
    start_wall = time.time()
    last_state = None

    while True:
        try:
            # 새 잡 자동 감지
            jid = seen_job or _latest_job()
            now = time.time()
            wall = time.strftime("%H:%M:%S")

            # HTTP 단계 새 이벤트 출력
            rows = _read_access_trace()
            for r in rows[seen_http:]:
                print(f"[{wall}] HTTP  {r['method']} {r['path']}  → {r['status']} ({r['dur']})")
            seen_http = len(rows)

            if jid:
                if jid != seen_job:
                    seen_job = jid
                    print(f"[{wall}] ▶ 잡 감지: {jid}")
                job = storage.load_job(jid) or {}
                state = job.get("status")
                if state != last_state:
                    print(f"[{wall}] 잡 상태: {last_state} → {state}")
                    last_state = state
                out_root = storage.job_output_dir(jid)
                for label, pattern in JOB_ARTIFACTS:
                    if label in seen_artifacts:
                        continue
                    if _artifact_mtime(out_root, pattern) is not None:
                        seen_artifacts.add(label)
                        el = now - (job.get("started_at") or start_wall)
                        print(f"[{wall}] ✓ 단계 완료: {label}  (잡 시작 +{el:.1f}s)")
                if state in ("complete", "failed"):
                    print(f"\n[{wall}] 잡 종료({state}). 분석표 ↓")
                    report(jid)
                    return 0
                # 진행 중이면 현재 무엇을 돌리는지 한 줄 상태
                if state == "running":
                    print(f"[{wall}]   …진행중  {_gpu_sample()}  단계≈{_running_stage()}", end="\r")

            time.sleep(poll)
        except KeyboardInterrupt:
            print("\n중단.")
            return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="모션 추출 파이프라인 타이밍/병목 분석")
    sub = ap.add_subparsers(dest="cmd")
    pr = sub.add_parser("report", help="완료/진행 잡 사후 분석표")
    pr.add_argument("--job", default=None)
    wa = sub.add_parser("watch", help="라이브 추적")
    wa.add_argument("--job", default=None)
    wa.add_argument("--poll", type=float, default=1.5)
    args = ap.parse_args(argv)

    if args.cmd == "watch":
        return watch(args.job, args.poll)
    return report(getattr(args, "job", None))


if __name__ == "__main__":
    raise SystemExit(main())
