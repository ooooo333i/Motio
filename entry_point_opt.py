#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entry_point_opt.py — Motio 서비스용 추론 오케스트레이터 (배포 정제 산출물 일원화)

목적
  CONTEXT_OVERVIEW의 "열린 이슈"(배포가 손목 정제된 결과를 안 내림)를 해결한다.
  entry_point_hamer.py 를 *전혀 수정하지 않고* 그 뒤에 일반화 손목 정제(batch 체인)를
  자동으로 이어붙여, 임의 업로드 영상 1건 → 최종 정제 .pt 까지 한 번에 만든다.

체인 (mode="fast", 현재 기본 = 서비스 경로 · docs/MOTIO_INTEGRATION.md D1)
  [1] entry_point_hamer.py   (env CAP_SKIP_WRIST_REFINE=1)
        -> <clip>/smplx_merged_hamer_post.pt , <clip>/preprocess/wholebody_kp2d.npy
  [2] batch_fit2d.py <clip>            -> <clip>/fit2d.pt           (손목 강한2D + WSM 평활)
  [3] batch_finger_kalman.py <clip>    -> <clip>/fit2d_fingerkalman.pt  ← 서비스가 내리는 최종본
  세 스크립트를 *그대로* subprocess 로 호출하므로 산출물은 수동 실행과 100% 동일(회귀 안전).
  각 단계는 fresh process 라 단계 사이 GPU VRAM 이 해제된다(단일 GPU 서버 안정).

mode="quality" (미구현 · 확장 지점)
  추후 wrist_pipeline 02(adaptive axial)/03(bad_frame slerp)의 *일반화* 버전을 여기 끼운다.
  현재는 명시적으로 NotImplementedError. 구조(분기/hook)만 열어둔다.

백엔드 연동 ("언제든 열릴 수 있는 형태")
  - CLI 그대로 subprocess 호출 가능. 또는 `from entry_point_opt import run_opt` 로 직접 호출.
  - 종료 시 manifest(JSON)를 stdout('[OPT_MANIFEST] {...}') 과 <clip>/opt_manifest.json 에 출력.
    backend 는 여기서 final_pt / num_frames 등을 읽는다.

주의
  - GVHMR/HaMeR/SMPL-X 단계는 GPU·CUDA 필요.
  - HaMeR import 시 PYOPENGL_PLATFORM='' 필요(프로필 egl 상속 방지) — 자동 설정.
  - 이 파일은 기존 어떤 파일도 수정/삭제하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# 체인 단계가 쓰는 산출물 파일명 (스크립트들의 고정 출력명과 일치)
MERGED_POST_NAME = "smplx_merged_hamer_post.pt"
FIT2D_NAME = "fit2d.pt"
FINAL_FAST_NAME = "fit2d_fingerkalman.pt"
FINAL_QUALITY_NAME = "hamer_wrist_hamer_rom_col_grip.pt"  # v4(adaptive) 최종 산출
WHOLEBODY_NPY = "preprocess/wholebody_kp2d.npy"


# ──────────────────────────────────────────────────────────────────────
# subprocess 유틸
# ──────────────────────────────────────────────────────────────────────
def _base_env() -> dict:
    """체인 단계 공통 환경. HaMeR egl 우회 + 레거시 손목정제 차단."""
    env = dict(os.environ)
    env.setdefault("PYOPENGL_PLATFORM", "")  # 빈 값 강제(프로필 egl 상속 방지)
    return env


def _run(cmd: list[str], env: dict, stage: str, stream: bool = True) -> str:
    """cmd 실행. stdout 를 (스트리밍하며) 모아 반환. 실패 시 RuntimeError."""
    print(f"\n[OPT] >>> {stage}: {' '.join(str(c) for c in cmd)}", flush=True)
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        if stream:
            sys.stdout.write(line)
            sys.stdout.flush()
    proc.wait()
    if proc.returncode != 0:
        tail = "".join(lines[-40:])
        raise RuntimeError(f"[OPT] stage '{stage}' failed (exit {proc.returncode}).\n--- tail ---\n{tail}")
    return "".join(lines)


def _find_clip_dir(hamer_stdout: str, output_root: Path) -> Path:
    """entry_point_hamer 출력에서 clip_dir(= smplx_merged_hamer_post.pt 의 부모)를 찾는다.
    1순위: stdout 에 찍힌 절대경로 파싱. 2순위: output_root 하위 glob 중 최신."""
    for line in hamer_stdout.splitlines():
        if MERGED_POST_NAME in line:
            for tok in line.replace(":", " ").split():
                if tok.endswith(MERGED_POST_NAME):
                    p = Path(tok)
                    if p.is_file():
                        return p.parent
    # fallback: output_root 아래에서 가장 최근 생성된 _post.pt
    cands = sorted(
        output_root.glob(f"*/{MERGED_POST_NAME}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if cands:
        return cands[0].parent
    raise FileNotFoundError(
        f"[OPT] {MERGED_POST_NAME} 를 찾지 못함. entry_point_hamer 가 merge 단계까지 갔는지 확인."
    )


def _num_frames(clip_dir: Path) -> int | None:
    """smplx_merged_hamer_post.json(report) 에서 num_frames 를 읽는다(없으면 None)."""
    report = clip_dir / "smplx_merged_hamer_post.json"
    if report.is_file():
        try:
            data = json.loads(report.read_text())
            for k in ("num_frames", "frames", "n_frames"):
                if isinstance(data.get(k), int):
                    return data[k]
        except Exception:
            pass
    return None


# ──────────────────────────────────────────────────────────────────────
# 체인 단계
# ──────────────────────────────────────────────────────────────────────
def _run_hamer_merge(
    video: Path,
    output_root: Path,
    person_track_id: int | None,
    auto_person: bool,
    no_interactive: bool,
    skip_result_video: bool,
    hamer_root: str | None,
    force: bool,
    python_exe: str,
) -> Path:
    """[1] entry_point_hamer (CAP_SKIP_WRIST_REFINE=1) -> clip_dir(_post.pt + wholebody_kp2d.npy)."""
    env = _base_env()
    env["CAP_SKIP_WRIST_REFINE"] = "1"  # merge 직후 정지(레거시 Stage A/B + 3변종 렌더 스킵)

    cmd = [python_exe, str(REPO_ROOT / "entry_point_hamer.py"),
           "--video", str(video),
           "--output-root", str(output_root),
           "--person-select-ui", "auto"]
    if person_track_id is not None:
        cmd += ["--person-track-id", str(person_track_id)]
    if auto_person:
        cmd += ["--auto-person"]
    if no_interactive:
        cmd += ["--no-interactive"]
    if skip_result_video:
        cmd += ["--skip-result-video"]
    if hamer_root:
        cmd += ["--hamer-root", hamer_root]
    if force:
        cmd += ["--force"]

    out = _run(cmd, env, "1/3 entry_point_hamer (merge)")
    clip_dir = _find_clip_dir(out, output_root)

    # 입력 사전조건 확인 (batch_fit2d 가 요구)
    post_pt = clip_dir / MERGED_POST_NAME
    wb_npy = clip_dir / WHOLEBODY_NPY
    if not post_pt.is_file():
        raise FileNotFoundError(f"[OPT] 누락: {post_pt}")
    if not wb_npy.is_file():
        raise FileNotFoundError(
            f"[OPT] 누락: {wb_npy} (batch_fit2d 입력). entry_point_hamer 가 wholebody_kp2d 를 저장했는지 확인."
        )
    print(f"[OPT] clip_dir = {clip_dir}")
    return clip_dir


def _run_batch_refine(clip_dir: Path, python_exe: str) -> Path:
    """[2][3] batch_fit2d -> batch_finger_kalman. 최종 fit2d_fingerkalman.pt 반환."""
    env = _base_env()
    _run([python_exe, str(REPO_ROOT / "batch_fit2d.py"), str(clip_dir)], env, "2/3 batch_fit2d")
    fit2d = clip_dir / FIT2D_NAME
    if not fit2d.is_file():
        raise FileNotFoundError(f"[OPT] 누락: {fit2d} (batch_fit2d 산출 실패)")

    _run([python_exe, str(REPO_ROOT / "batch_finger_kalman.py"), str(clip_dir)], env, "3/3 batch_finger_kalman")
    final = clip_dir / FINAL_FAST_NAME
    if not final.is_file():
        raise FileNotFoundError(f"[OPT] 누락: {final} (batch_finger_kalman 산출 실패)")
    return final


def _refine_quality(clip_dir: Path, python_exe: str) -> Path:
    """[quality 모드] adaptive_v4 체인 — 일반화 검증 완료(2026-06-04).

    파이프라인 (run_v4.sh 와 동일, 전 단계 영상별 자동·하드코딩 없음):
      A  cap_arm_prefit (CAP_ARM_DEPTH 깊이앵커 + ViTPose conf 게이팅 폴백) → smplx_arm_prefit.pt
      A2 batch_arm_smooth (가려짐 구간 conf-게이트 평활) → 손목위치 떨림 제거
      B  batch_hamer_wrist (손목=HaMeR 각도 + 신뢰도 게이트 + 평활)
      C  batch_rom_clamp (전완축 기준 생체역학 손목 한계)
      D  collision_detect + batch_axcol_v4 (충돌을 2D-불변 축회전으로 해소)
      E  batch_grip_prior (손가락만; IoU 겹침 + conf 게이트, body_pose 불변)
    근거: poster/V4_EVIDENCE.md (독립 MediaPipe 2D 최고, twist·충돌 최저).

    입력 전제(entry_point_hamer 가 [1]단계에서 생성): hmr4d_results.pt, preprocess/vitpose.pt,
      preprocess/wholebody_kp2d.npy, hamer_out_clean/. (fit2d 불필요 — betas 는 base pt 에서.)
    """
    # ── 입력 전제 확인 (없으면 cryptic 실패 대신 명확한 에러) ──
    need = {
        "hmr4d_results.pt": clip_dir / "hmr4d_results.pt",
        "preprocess/vitpose.pt": clip_dir / "preprocess" / "vitpose.pt",
        WHOLEBODY_NPY: clip_dir / WHOLEBODY_NPY,
        "hamer_out_clean/": clip_dir / "hamer_out_clean",
        MERGED_POST_NAME: clip_dir / MERGED_POST_NAME,
    }
    miss = [k for k, p in need.items() if not p.exists()]
    if miss:
        raise FileNotFoundError(
            f"[OPT-quality] 입력 누락: {miss}. entry_point_hamer([1]단계)가 GVHMR/ViTPose/HaMeR "
            f"전처리를 모두 생성했는지 확인 (CAP_SKIP_WRIST_REFINE=1 은 검출은 보존)."
        )

    env = _base_env()
    arm_env = dict(env)
    arm_env["CAP_ARM_DEPTH"] = "100000"  # 깊이 앵커(단안 위치 깜빡임 제거)
    clip = str(clip_dir)
    R = REPO_ROOT

    # (fit2d 불필요: grip 의 betas/go/transl 은 base pt 에 동일하게 있음 → batch_grip_prior 가 자동 폴백)
    # A) 팔 IK + 깊이앵커 + conf 게이팅
    _run([python_exe, R / "cap_arm_prefit.py", "--input-dir", clip,
          "--w-tc", "5000", "--target-px", "3", "--debug-frames", "0"], arm_env, "QA cap_arm_prefit")
    # A2) 팔 회전 평활 (가려짐 떨림 제거, 가시 구간 보존)
    _run([python_exe, R / "batch_arm_smooth.py", clip, "1.0", "3.0"], env, "QA2 batch_arm_smooth")
    # B) 손목 = HaMeR + 신뢰도 게이트
    _run([python_exe, R / "batch_hamer_wrist.py", clip, "hamer", "1.8"], env, "QB batch_hamer_wrist")
    # C) ROM clamp
    _run([python_exe, R / "batch_rom_clamp.py", clip, "hamer_wrist_hamer.pt"], env, "QC batch_rom_clamp")
    # D) 충돌 해소 (검출 → 축분리 최적화)
    _run([python_exe, R / "collision_detect.py", clip, "hamer_wrist_hamer_rom.pt"], env, "QD collision_detect")
    _run([python_exe, R / "batch_axcol_v4.py", clip, "250", "hamer_wrist_hamer_rom.pt"], env, "QD batch_axcol_v4")
    # E) 그립 게이트 (손가락만)
    _run([python_exe, R / "batch_grip_prior.py", clip, "hamer_wrist_hamer_rom_col.pt",
          "0.7", "1.5", "0.1"], env, "QE batch_grip_prior")

    final = clip_dir / FINAL_QUALITY_NAME
    if not final.is_file():
        raise FileNotFoundError(f"[OPT-quality] 누락: {final} (grip prior 산출 실패)")
    return final


# ──────────────────────────────────────────────────────────────────────
# 공개 진입점
# ──────────────────────────────────────────────────────────────────────
def run_opt(
    video: str | Path,
    output_root: str | Path,
    person_track_id: int | None = None,
    mode: str = "quality",  # 기본=adaptive v4 (fast=레거시 fit2d). 근거: poster/V4_EVIDENCE.md
    auto_person: bool = True,
    no_interactive: bool = True,
    skip_result_video: bool = True,
    hamer_root: str | None = None,
    force: bool = False,
    python_exe: str | None = None,
    manifest_path: str | Path | None = None,
) -> dict:
    """임의 영상 1건 → 최종 정제 .pt. manifest(dict) 반환.

    backend 는 이 함수를 import 호출하거나 CLI 를 subprocess 로 호출할 수 있다.
    """
    video = Path(video).resolve()
    output_root = Path(output_root)
    python_exe = python_exe or sys.executable
    if not video.is_file():
        raise FileNotFoundError(f"[OPT] 입력 영상 없음: {video}")
    if mode not in ("fast", "quality"):
        raise ValueError(f"[OPT] 알 수 없는 mode: {mode} (fast|quality)")

    clip_dir = _run_hamer_merge(
        video, output_root, person_track_id, auto_person, no_interactive,
        skip_result_video, hamer_root, force, python_exe,
    )

    if mode == "fast":
        final_pt = _run_batch_refine(clip_dir, python_exe)
    else:  # quality
        final_pt = _refine_quality(clip_dir, python_exe)

    manifest = {
        "status": "complete",
        "mode": mode,
        "video": str(video),
        "clip_dir": str(clip_dir),
        "outputs": {
            "merged_post": str(clip_dir / MERGED_POST_NAME),
            "fit2d": str(clip_dir / FIT2D_NAME) if (clip_dir / FIT2D_NAME).is_file() else None,
            "final": str(final_pt),
        },
        "num_frames": _num_frames(clip_dir),
    }

    # manifest 기록 (clip_dir + 선택 경로)
    (clip_dir / "opt_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    if manifest_path:
        Path(manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n[OPT_MANIFEST] {json.dumps(manifest, ensure_ascii=False)}", flush=True)
    print(f"[OPT] DONE. final = {final_pt}", flush=True)
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Motio 서비스용 추론 오케스트레이터 (entry_point_hamer + batch 정제 체인).",
    )
    p.add_argument("--video", required=True, help="입력 영상 경로(임의 클립).")
    p.add_argument("--output-root", default="outputs/motio_service/jobs", help="출력 루트(클립별 하위폴더 생성).")
    p.add_argument("--person-track-id", type=int, default=None, help="추출 대상 사람(YOLO track id). 프론트 targetId.")
    p.add_argument("--mode", choices=("fast", "quality"), default="quality",
                   help="quality=adaptive v4(기본, 권장). fast=레거시 fit2d+finger_kalman.")
    p.add_argument("--no-auto-person", dest="auto_person", action="store_false",
                   help="largest-person 자동선택 끄기(기본 켜짐).")
    p.add_argument("--interactive", dest="no_interactive", action="store_false",
                   help="누락값에 프롬프트 허용(기본은 비대화=fail-fast).")
    p.add_argument("--render-result-video", dest="skip_result_video", action="store_false",
                   help="merged 결과 mp4 렌더(기본 스킵).")
    p.add_argument("--hamer-root", default=None, help="HaMeR repo 경로(기본 entry_point 기본값).")
    p.add_argument("--force", action="store_true", help="가능하면 산출물 재계산.")
    p.add_argument("--python", dest="python_exe", default=None, help="체인 단계 실행 인터프리터(기본 현재 python).")
    p.add_argument("--manifest", default=None, help="manifest JSON 추가 저장 경로(기본 <clip>/opt_manifest.json).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_opt(
            video=args.video,
            output_root=args.output_root,
            person_track_id=args.person_track_id,
            mode=args.mode,
            auto_person=args.auto_person,
            no_interactive=args.no_interactive,
            skip_result_video=args.skip_result_video,
            hamer_root=args.hamer_root,
            force=args.force,
            python_exe=args.python_exe,
            manifest_path=args.manifest,
        )
        return 0
    except Exception as exc:  # 백엔드가 비정상 종료를 감지할 수 있게 비0 반환
        print(f"\n[OPT_ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
