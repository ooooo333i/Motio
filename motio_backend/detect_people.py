#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""detect_people.py — 클립에서 사람 검출/트래킹 → Motio 검출 JSON (cap-deploy 환경 subprocess).

백엔드가 이 스크립트를 subprocess 로 호출한다(API 워커에 torch 미로드, VRAM 격리).
tracker.py 는 *수정하지 않고*, YOLO track 루프를 여기서 복제해 confidence 를 직접 수집한다.

usage: python detect_people.py --clip <clip.mp4> --out <detection.json> [--max-people N]
출력: {"fps", "objects":[{id,label,type,confidence,track:[{frame,bbox{x,y,w,h}}]}]}  (bbox=퍼센트 0~100)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


def _filter_players(candidates, total_frames):
    """선수 후보만 남긴다(배경 관중 제거). 비싼 dense-track 생성 *이전*에 적용.

    기준(둘 다 만족): ① 화면 점유 평균 면적 avg_area ≥ CAP_PLAYER_MIN_AREA(기본 0.02=2%),
    ② 등장 지속성 frame_count/total ≥ CAP_PLAYER_MIN_PRESENCE(기본 0.5). 짧게 스치는 사람 제외.
    아무도 통과 못하면 점수(면적×지속성) 최상위 1명은 보존(빈 결과 방지). 최대 CAP_PLAYER_MAX(기본 3)명.
    CAP_PLAYER_FILTER=0 이면 필터 비활성(기존 동작).
    """
    if os.environ.get("CAP_PLAYER_FILTER", "1") == "0" or not candidates:
        return candidates
    total = max(int(total_frames or 0), max((c.frame_count for c in candidates), default=1), 1)
    min_area = float(os.environ.get("CAP_PLAYER_MIN_AREA", "0.02"))
    min_pres = float(os.environ.get("CAP_PLAYER_MIN_PRESENCE", "0.5"))
    max_players = int(os.environ.get("CAP_PLAYER_MAX", "3"))

    def score(c):
        return c.avg_area * (c.frame_count / total)

    kept = [c for c in candidates
            if c.avg_area >= min_area and (c.frame_count / total) >= min_pres]
    if not kept:
        kept = [max(candidates, key=score)]
    kept.sort(key=score, reverse=True)
    return kept[:max_players]

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _video_fps(clip_path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    return round(float(fps), 3) if fps and fps > 0 else 30.0


def _smooth_xyxy(full_np, win: int = 5):
    """프레임별 bbox 의 YOLO jitter 를 줄인다(중심·크기 이동평균). 길이/프레임 수 보존.

    코너(x1,y1,x2,y2) 대신 중심(cx,cy)·크기(w,h)를 평활화해야 박스가 자연스럽게 따라간다.
    """
    import numpy as np
    F = full_np.shape[0]
    if F < 3:
        return full_np
    x1, y1, x2, y2 = full_np[:, 0], full_np[:, 1], full_np[:, 2], full_np[:, 3]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1), (y2 - y1)
    k = max(1, min(win, F))
    kernel = np.ones(k) / k

    def _ma(a):
        # edge 패딩 후 컨볼루션 → 양끝도 길이 보존하며 부드럽게
        pad = k // 2
        ap = np.pad(a, (pad, pad), mode="edge")
        return np.convolve(ap, kernel, mode="valid")[:F]

    cxs, cys, ws, hs = _ma(cx), _ma(cy), _ma(w), _ma(h)
    out = np.stack([cxs - ws / 2.0, cys - hs / 2.0, cxs + ws / 2.0, cys + hs / 2.0], axis=1)
    return out.astype(full_np.dtype)


# 서버 in-process 호출 시 YOLO 모델을 1회만 로딩(매 호출 재로딩 제거). 서브프로세스 1회 호출엔 영향 없음.
_TRACKER = None


def _get_tracker():
    global _TRACKER
    if _TRACKER is None:
        from hmr4d.utils.preproc.tracker import Tracker
        _TRACKER = Tracker()
    return _TRACKER


def detect(clip_path: str, out_path: str, max_people: int | None = None) -> dict:
    import numpy as np
    from hmr4d.utils.preproc.tracker import Tracker  # 정적메서드(sort_track_length 등)용 — 모델 로딩 아님
    from hmr4d.utils.video_io_utils import get_video_lwh
    import entry_point as cap  # build_person_track_candidates

    tracker = _get_tracker()  # 인스턴스(YOLO 모델)는 캐시 재사용 → 매 호출 재로딩 제거

    # ── YOLO 트래킹 (track() 복제 + conf 수집) ──────────────────
    _t_yolo = time.time()
    cfg = {"device": "cuda", "conf": 0.5, "classes": 0, "verbose": False, "stream": True}
    results = tracker.yolo.track(str(clip_path), **cfg)
    track_history = []
    id_to_conf: dict[int, list[float]] = defaultdict(list)
    for result in results:
        if result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            bbx = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            frame = [{"id": ids[i], "bbx_xyxy": bbx[i]} for i in range(len(ids))]
            for i in range(len(ids)):
                id_to_conf[ids[i]].append(float(conf[i]))
        else:
            frame = []
        track_history.append(frame)

    length, width, height = get_video_lwh(str(clip_path))
    width = max(int(width), 1)
    height = max(int(height), 1)

    _yolo_sec = time.time() - _t_yolo

    id_to_frame_ids, id_to_bbx_xyxys, id_sorted = Tracker.sort_track_length(track_history, str(clip_path))
    candidates = cap.build_person_track_candidates(id_to_frame_ids, id_to_bbx_xyxys, id_sorted, width, height)
    if max_people:
        candidates = candidates[:max_people]
    _n_raw = len(candidates)
    candidates = _filter_players(candidates, length)  # 선수만(배경 관중 제거) — dense track 전에

    _t_loop = time.time()
    objects = []
    raw_tracks = {}  # track_id → 원본(스무딩 前) dense bbx_xyxy. 추출이 YOLO 재추적 대신 재사용.
    for c in candidates:
        # 프레임 정렬된 전체 track (결측 보간) — (F,4) xyxy 픽셀
        full = Tracker.build_one_track(
            str(clip_path), id_to_frame_ids[c.track_id], id_to_bbx_xyxys[c.track_id]
        )
        full_np = full.cpu().numpy() if hasattr(full, "cpu") else np.asarray(full)
        raw_tracks[f"t{int(c.track_id)}"] = full_np.astype(np.float32).copy()  # 스무딩 前 = 추출 build_one_track 과 동일
        full_np = _smooth_xyxy(full_np)  # 프레임 간 떨림 완화(부드러운 추적, JSON/UI 용)
        confs = id_to_conf.get(c.track_id, [])
        conf_avg = round(float(np.mean(confs)), 3) if confs else 0.0

        track = []
        for f in range(full_np.shape[0]):
            x1, y1, x2, y2 = [float(v) for v in full_np[f]]
            track.append({
                "frame": f,
                "bbox": {
                    "x": round(x1 / width * 100.0, 3),
                    "y": round(y1 / height * 100.0, 3),
                    "w": round((x2 - x1) / width * 100.0, 3),
                    "h": round((y2 - y1) / height * 100.0, 3),
                },
            })
        objects.append({
            "id": str(c.track_id),
            "label": f"Person {c.track_id}",
            "type": "human",
            "confidence": conf_avg,
            "track": track,
        })

    _loop_sec = time.time() - _t_loop
    payload = {"fps": _video_fps(clip_path), "objects": objects}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    # 재사용용 raw 트랙 저장(추출이 YOLO 재추적을 스킵하도록). CAP_REUSE_DETECTION=0 이면 생략.
    if os.environ.get("CAP_REUSE_DETECTION", "1") != "0" and raw_tracks:
        try:
            import numpy as _np
            tp = Path(out_path).with_name(Path(out_path).stem + "_tracks.npz")
            _np.savez(str(tp), **raw_tracks)
        except Exception as _e:
            print(f"[DETECT] raw 트랙 저장 실패(무시): {_e}")
    print(f"[DETECT] yolo={_yolo_sec:.1f}s track-build={_loop_sec:.1f}s "
          f"raw_tracks={_n_raw} kept={len(objects)} -> {out_path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Motio 사람 검출 어댑터 (Tracker -> 검출 JSON).")
    p.add_argument("--clip", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-people", type=int, default=None)
    args = p.parse_args(argv)
    try:
        detect(args.clip, args.out, args.max_people)
        return 0
    except Exception as exc:
        print(f"[DETECT_ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
