"""
cap_smooth_wrist.py  (v2 — spike-only)

Median-based outlier detection + SLERP 보간.
단일 프레임 스파이크만 제거, 빠른 실제 스윙은 유지.

알고리즘 (wrist rotation):
  1. v_t = geodesic(R[t-1], R[t])  (deg/frame)
  2. 윈도우 w의 이웃 median/MAD 계산
  3. score_t = |v_t - median| / (MAD + 1e-6)
  4. score > threshold → 스파이크 마킹
  5. 스파이크 프레임 → 앞뒤 valid 프레임으로 SLERP 보간

hand_pose: 동일 로직, 성분별 RMS 속도 → 스파이크 검출 → 선형 보간
"""
from __future__ import annotations
import argparse, copy
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


# ──────────────────────────────────────────────────────────────
# 1. 속도 / 스파이크 검출
# ──────────────────────────────────────────────────────────────
def rotation_velocity(aa_seq: np.ndarray) -> np.ndarray:
    """(F, 3) axis-angle → (F,) geodesic deg/frame.  v[0] = v[1] (경계 복사)."""
    R = Rotation.from_rotvec(aa_seq.reshape(-1, 3))
    vel = np.zeros(len(R), dtype=np.float32)
    vel[1:] = np.degrees(np.abs((R[:-1].inv() * R[1:]).magnitude()))
    vel[0]  = vel[1]
    return vel


def pose_velocity(pose_seq: np.ndarray) -> np.ndarray:
    """(F, D) axis-angle → (F,) RMS deg/frame.  v[0] = v[1]."""
    diff = np.zeros(pose_seq.shape[0], dtype=np.float32)
    diff[1:] = np.degrees(
        np.sqrt(((pose_seq[1:] - pose_seq[:-1]) ** 2).mean(axis=-1))
    )
    diff[0] = diff[1]
    return diff


def detect_spikes(vel: np.ndarray, window: int = 5,
                  threshold: float = 5.0) -> np.ndarray:
    """(F,) velocity → (F,) boolean spike mask.

    각 프레임 t에서 half = window//2 크기의 이웃(자신 제외)으로
    median과 MAD를 계산하고, score = |v_t - median| / (MAD + 1e-6).
    score > threshold 이면 스파이크.
    """
    F    = len(vel)
    half = window // 2
    mask = np.zeros(F, dtype=bool)

    for t in range(F):
        lo  = max(0, t - half)
        hi  = min(F, t + half + 1)
        nbr = np.concatenate([vel[lo:t], vel[t+1:hi]])
        if len(nbr) == 0:
            continue
        med  = np.median(nbr)
        mad  = np.median(np.abs(nbr - med))
        score = abs(vel[t] - med) / (mad + 1e-6)
        if score > threshold:
            mask[t] = True

    return mask


# ──────────────────────────────────────────────────────────────
# 2. SLERP / 선형 보간 채우기
# ──────────────────────────────────────────────────────────────
def slerp_fill_rotations(aa_seq: np.ndarray,
                         spike_mask: np.ndarray) -> np.ndarray:
    """스파이크 프레임을 앞뒤 valid SLERP로 채운다. (F,3) in/out."""
    out    = aa_seq.copy()
    F      = len(aa_seq)
    valid  = np.where(~spike_mask)[0]

    if len(valid) == 0:
        return out   # 전부 스파이크면 포기

    R_all  = Rotation.from_rotvec(aa_seq)

    for t in np.where(spike_mask)[0]:
        prev_idx = valid[valid < t]
        next_idx = valid[valid > t]

        if len(prev_idx) == 0 and len(next_idx) == 0:
            continue
        elif len(prev_idx) == 0:
            out[t] = aa_seq[next_idx[0]]
        elif len(next_idx) == 0:
            out[t] = aa_seq[prev_idx[-1]]
        else:
            p, n  = prev_idx[-1], next_idx[0]
            alpha = (t - p) / (n - p)
            times = np.array([float(p), float(n)])
            slerp = Slerp(times, Rotation.concatenate([R_all[p], R_all[n]]))
            out[t] = slerp(float(t)).as_rotvec().astype(np.float32)

    return out


def linear_fill(pose_seq: np.ndarray,
                spike_mask: np.ndarray) -> np.ndarray:
    """hand_pose (F,D): 스파이크 프레임을 선형 보간으로 채운다."""
    out   = pose_seq.copy()
    F     = len(pose_seq)
    valid = np.where(~spike_mask)[0]

    if len(valid) == 0:
        return out

    for t in np.where(spike_mask)[0]:
        prev_idx = valid[valid < t]
        next_idx = valid[valid > t]

        if len(prev_idx) == 0 and len(next_idx) == 0:
            continue
        elif len(prev_idx) == 0:
            out[t] = pose_seq[next_idx[0]]
        elif len(next_idx) == 0:
            out[t] = pose_seq[prev_idx[-1]]
        else:
            p, n  = prev_idx[-1], next_idx[0]
            alpha = (t - p) / (n - p)
            out[t] = (1 - alpha) * pose_seq[p] + alpha * pose_seq[n]

    return out.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 3. 메인 처리
# ──────────────────────────────────────────────────────────────
def run_smooth(
    input_path:   Path,
    output_path:  Path,
    *,
    window:        int   = 5,
    threshold:     float = 5.0,
    abs_threshold: float = 90.0,   # 절대값 기준 (deg/frame), 물리적 불가 구간
    smooth_wrist:  bool  = True,
    smooth_hand:   bool  = True,
    verbose:       bool  = True,
) -> dict:
    """Returns spike_stats dict."""
    print(f"[smooth] window={window}  threshold={threshold}  abs_threshold={abs_threshold}°")

    pred   = torch.load(input_path, map_location="cpu")
    result = copy.deepcopy(pred)
    params = result["smpl_params_incam"]
    F      = params["body_pose"].shape[0]

    body_pose = params["body_pose"].reshape(F, 21, 3).numpy()
    lhp       = params["left_hand_pose"].reshape(F, 45).numpy()
    rhp       = params["right_hand_pose"].reshape(F, 45).numpy()

    stats = {}

    # ── wrist rotation ─────────────────────────────────────
    if smooth_wrist:
        for name, bp_idx in [("L_wrist", 19), ("R_wrist", 20)]:
            raw  = body_pose[:, bp_idx, :]
            vel  = rotation_velocity(raw)

            mask_mad = detect_spikes(vel, window=window, threshold=threshold)
            mask_abs = vel > abs_threshold                  # 절대값 기준
            mask     = mask_mad | mask_abs                  # OR 결합

            n_mad   = int(mask_mad.sum())
            n_abs   = int(mask_abs.sum())
            n_total = int(mask.sum())
            if verbose:
                print(f"[smooth] {name}: MAD={n_mad}  abs>{abs_threshold}°={n_abs}  "
                      f"total={n_total} ({n_total/F*100:.1f}%)")
            body_pose[:, bp_idx, :] = slerp_fill_rotations(raw, mask)
            stats[name] = {"n_spike": n_total, "pct": n_total/F*100,
                           "mask": mask, "vel": vel,
                           "n_mad": n_mad, "n_abs": n_abs}

    # hand_pose는 필터링하지 않음 — HaMeR 원본 그대로 유지

    # ── 저장 ────────────────────────────────────────────────
    params["body_pose"]       = torch.from_numpy(body_pose).reshape(F, 21, 3)
    params["left_hand_pose"]  = torch.from_numpy(lhp).reshape(F, 45)
    params["right_hand_pose"] = torch.from_numpy(rhp).reshape(F, 45)
    result["smpl_params_incam"] = params

    if "smpl_params_global" in result:
        g  = result["smpl_params_global"]
        gb = g["body_pose"].reshape(F, 21, 3)
        gb[:, 19] = params["body_pose"].reshape(F, 21, 3)[:, 19]
        gb[:, 20] = params["body_pose"].reshape(F, 21, 3)[:, 20]
        g["body_pose"]        = gb.reshape(F, 21, 3)
        g["left_hand_pose"]   = params["left_hand_pose"].clone()
        g["right_hand_pose"]  = params["right_hand_pose"].clone()
        result["smpl_params_global"] = g

    result.setdefault("cap_smooth_meta", {}).update({
        "method": "median_spike", "window": window, "threshold": threshold,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    print(f"[smooth] saved → {output_path}")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",   default="outputs/cap_pipeline/input")
    p.add_argument("--input-pt",    default="smplx_merged_prefit.pt")
    p.add_argument("--output-pt",   default="smplx_final_smoothed.pt")
    p.add_argument("--window",         type=int,   default=5)
    p.add_argument("--threshold",      type=float, default=5.0)
    p.add_argument("--abs-threshold",  type=float, default=90.0)
    p.add_argument("--no-wrist",       action="store_true")
    p.add_argument("--no-hand",        action="store_true")
    args = p.parse_args()

    d = Path(args.input_dir)
    run_smooth(
        input_path    = d / args.input_pt,
        output_path   = d / args.output_pt,
        window        = args.window,
        threshold     = args.threshold,
        abs_threshold = args.abs_threshold,
        smooth_wrist  = not args.no_wrist,
        smooth_hand   = not args.no_hand,
    )


if __name__ == "__main__":
    main()
