#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cap_metrics.py — GT-free 정량 비교 (여러 메서드 .pt 를 같은 기준으로).

학회 포스터용. GT(3D 정답)가 없으므로 **출력 궤적만으로 계산되는** 지표만 담는다.
지금 바로 되는 것:
  - Jitter (mean jerk, 3차 미분 크기)        : 부드러움. 너희 직접 주장.
  - Acceleration (mean accel, 2차 미분 크기)
  - Bone-length 일관성 (변동계수 CoV)         : 스켈레톤 안정성(강체면 0).
  - Wrist joint-limit 위반율 (physical_limits) : 전완 비틀림(axial)·deviation·flexion.
모두 부위별(full-body / arm-chain / hand)로 쪼개 **전파 스토리**를 본다.

예약(2단계, 외부 2D 필요): held-out 2D PCKh 재투영 → reproject_pckh() 훅만 열어둠.

표준 호출:
    python cap_metrics.py \
        --m gvhmr=outputs/cap_pipeline/input/hmr4d_results.pt \
        --m dynhamr=outputs/cap_pipeline/input/cmp_dynhamr_grafted.pt \
        --m ours=outputs/cap_pipeline/input/cmp_adaptive_v3.pt \
        --space global --fps 30 --out outputs/metrics_compare

데이터: .pt 는 SMPL-X 파라미터(+K_fullimg). make_smplx("supermotion_fullhands")로
3D 관절(127개)을 forward 한 뒤 지표 계산. body_pose 는 (T,63)/(T,21,3) 둘 다 흡수.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# ── SMPL-X 관절 인덱스 (out.joints, 앞 55개 = SMPL-X 규약) ───────────────────
# shoulder L/R=16/17, elbow=18/19, wrist=20/21 (cap_arm_prefit 과 동일)
PELVIS = 0
ARM_CHAIN = [16, 17, 18, 19, 20, 21]                 # 어깨·팔꿈치·손목
HAND = [20, 21] + list(range(25, 55))                # 손목 + 양손 손가락(L 25-39, R 40-54)
BODY = list(range(22))                               # 주요 몸통 22개
REGIONS = {"full": BODY, "arm": ARM_CHAIN, "hand": HAND}

# Bone-length CoV 용 뼈 (팔 체인) : (parent_jid, child_jid)
BONES = [(16, 18), (18, 20),   # L: 어깨→팔꿈치→손목
         (17, 19), (19, 21),   # R
         (13, 16), (14, 17)]   # collar→shoulder

# body_pose 안 손목 인덱스 (body_pose[i] = SMPL-X joint i+1) : physical_limits 와 동일
WRIST_BP = {"L": 19, "R": 20}

_MODEL = None


def get_model(device="cpu"):
    global _MODEL
    if _MODEL is None:
        from hmr4d.utils.smplx_utils import make_smplx
        _MODEL = make_smplx("supermotion_fullhands").to(device).eval()
    return _MODEL


# ── 로딩 ────────────────────────────────────────────────────────────────────
def load_params(pt_path: str, space: str = "global"):
    """메서드 .pt → 정규화된 SMPL-X 파라미터 dict + K(있으면). body_pose→(T,63)."""
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    key = f"smpl_params_{space}"
    if key not in d:  # dynhamr_grafted 처럼 incam 만 있는 경우 fallback
        key = "smpl_params_incam" if "smpl_params_incam" in d else "smpl_params_global"
    p = d[key]
    bp = p["body_pose"].float()
    bp = bp.reshape(bp.shape[0], -1)                  # (T,21,3)|(T,63) → (T,63)
    T = bp.shape[0]

    def g(name, dim):
        v = p[name].float()
        v = v.reshape(T, -1) if v.shape[0] == T else v.reshape(1, -1).expand(T, -1)
        return v

    params = dict(
        body_pose=bp,
        global_orient=g("global_orient", 3),
        transl=g("transl", 3),
        betas=g("betas", 10),
        left_hand_pose=g("left_hand_pose", 45),
        right_hand_pose=g("right_hand_pose", 45),
    )
    for k in ("jaw_pose", "leye_pose", "reye_pose"):
        if k in p:
            params[k] = g(k, 3)
    K = d.get("K_fullimg")
    return params, (K.float() if K is not None else None), key


@torch.no_grad()
def forward_joints(params, device="cpu", chunk=400) -> np.ndarray:
    """SMPL-X forward → 관절 (T, 127, 3) numpy. 메모리 위해 청크 처리."""
    m = get_model(device)
    T = params["body_pose"].shape[0]
    out = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        kw = {k: v[s:e].to(device) for k, v in params.items()}
        out.append(m.forward(**kw).joints.cpu())
    return torch.cat(out, 0).numpy()                  # (T, 127, 3)


# ── 지표 ────────────────────────────────────────────────────────────────────
def _finite_diff(x, order):
    """x:(T,J,3) → order차 유한차분 (T-order, J, 3)."""
    for _ in range(order):
        x = x[1:] - x[:-1]
    return x


def jitter(joints, idx, fps, root_rel=True):
    """mean jerk magnitude. 단위 m/s³ (입력이 m면). root_rel=True → pelvis 빼서 관절 articulation만."""
    j = joints[:, idx, :].astype(np.float64)
    if root_rel:
        j = j - joints[:, PELVIS:PELVIS + 1, :].astype(np.float64)
    jerk = _finite_diff(j, 3) * (fps ** 3)            # (T-3, |idx|, 3)
    return float(np.linalg.norm(jerk, axis=-1).mean())


def accel(joints, idx, fps, root_rel=True):
    """mean acceleration magnitude. 단위 m/s² (=> mm/s² 는 ×1000)."""
    j = joints[:, idx, :].astype(np.float64)
    if root_rel:
        j = j - joints[:, PELVIS:PELVIS + 1, :].astype(np.float64)
    a = _finite_diff(j, 2) * (fps ** 2)
    return float(np.linalg.norm(a, axis=-1).mean())


def bone_length_cov(joints):
    """팔 체인 뼈 길이의 변동계수(std/mean) 평균. 강체면 0 → 낮을수록 안정."""
    covs = []
    for a, b in BONES:
        ell = np.linalg.norm(joints[:, a, :] - joints[:, b, :], axis=-1)
        m = ell.mean()
        if m > 1e-6:
            covs.append(ell.std() / m)
    return float(np.mean(covs)) if covs else float("nan")


def wrist_limit_violation(params, level="hard"):
    """양손 손목 회전을 physical_limits 로 검사 → 성분별 위반율(%) + 전체."""
    import physical_limits as PL
    bp = params["body_pose"].numpy().reshape(-1, 21, 3)   # (T,21,3) axis-angle
    comps = ["axial", "deviation", "flexion"]
    viol = {c: 0 for c in comps}
    any_v = 0
    T = bp.shape[0]
    for t in range(T):
        bad_any = False
        for side in ("L", "R"):
            aa = bp[t, WRIST_BP[side]]
            res = PL.check_wrist_rotation(aa, level=level)
            for c in comps:
                if not res[c]["in_range"]:
                    viol[c] += 1
                    bad_any = True
        if bad_any:
            any_v += 1
    denom = T * 2  # 양손
    out = {f"wrist_{c}_viol%": 100.0 * viol[c] / denom for c in comps}
    out["wrist_any_viol%"] = 100.0 * any_v / T
    return out


# ── 보조: naive smoothing 대조군 생성 ───────────────────────────────────────
def naive_smooth_joints(joints, sigma=2.0):
    """3D 관절을 시간축 Gaussian 으로 뭉갠 '반칙 대조군'. jitter는 줄지만 정합은 깎임."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(joints, sigma=sigma, axis=0, mode="nearest")


# ── 2D 손 재투영 (독립 정답 = COCO-WholeBody wholebody_kp2d.npy) ─────────────
# SMPL-X 손관절 ↔ wholebody 손kp 대응 (batch_fit2d.py 와 동일, 손목+MCP 5개 = 6점/손).
# wholebody 손: 왼손 91~111, 오른손 112~132. kp0=손목, kp9=중지 MCP.
LID_S = [20, 25, 28, 31, 34, 37]; LID_W = [91 + i for i in [0, 5, 9, 17, 13, 2]]
RID_S = [21, 40, 43, 46, 49, 52]; RID_W = [112 + i for i in [0, 5, 9, 17, 13, 2]]


def hand_reproj(pt_path, wb_path, conf_th=0.3, alpha=0.5):
    """incam 3D 손관절을 K로 재투영 → wholebody 2D 손kp와 비교.
    반환: 손별/전체 reproj 오차(px) + PCK@alpha(손크기=손목→중지MCP 로 정규화).
    ※ wholebody 에 피팅한 방법(fit2d 등)엔 부분순환 — 비순환 baseline(HaMeR-fingers/GVHMR/Dyn)에서 공정."""
    params, K, _ = load_params(pt_path, space="incam")
    joints = forward_joints(params)                    # (T,127,3) incam (단위 m)
    K = K.numpy()
    wb = np.load(wb_path)                               # (F,133,3) 픽셀+conf
    T = min(len(joints), len(wb))
    joints, K, wb = joints[:T], K[:T], wb[:T]
    z = np.clip(joints[..., 2], 1e-4, None)
    fx, fy = K[:, 0, 0][:, None], K[:, 1, 1][:, None]
    cx, cy = K[:, 0, 2][:, None], K[:, 1, 2][:, None]
    u = fx * joints[..., 0] / z + cx
    v = fy * joints[..., 1] / z + cy
    uv = np.stack([u, v], -1)                           # (T,127,2)

    out = {}; all_err = []; all_pck = []
    for side, SID, WID in [("L", LID_S, LID_W), ("R", RID_S, RID_W)]:
        proj = uv[:, SID, :]                            # (T,6,2)
        gt = wb[:, WID, :2]; c = wb[:, WID, 2]          # (T,6),(T,6)
        d = np.linalg.norm(proj - gt, axis=-1)          # (T,6) px
        scale = np.clip(np.linalg.norm(gt[:, 0] - gt[:, 2], axis=-1), 1e-3, None)  # 손목→중지MCP
        m = c > conf_th
        if m.sum() == 0:
            continue
        out[f"hand_px_{side}"] = float(d[m].mean())
        out[f"hand_pck_{side}"] = float(((d <= alpha * scale[:, None]) & m).sum() / m.sum())
        all_err.append(d[m]); all_pck.append(out[f"hand_pck_{side}"])
    if all_err:
        out["hand_px"] = float(np.concatenate(all_err).mean())
        out["hand_pck@%.1f" % alpha] = float(np.mean(all_pck))
    out["reproj_frames"] = int(T)
    return out


# ── 드라이버 ────────────────────────────────────────────────────────────────
def evaluate(pt_path, space, fps, device, do_smooth=False, wb_path=None):
    params, K, used = load_params(pt_path, space)
    joints = forward_joints(params, device=device)        # (T,127,3) — 단위 m
    T = joints.shape[0]
    row = {"frames": T, "space": used}
    for rname, idx in REGIONS.items():
        row[f"jitter_{rname}"] = jitter(joints, idx, fps)
        row[f"accel_{rname}"] = accel(joints, idx, fps)
    row["bone_cov"] = bone_length_cov(joints)
    try:
        row.update(wrist_limit_violation(params))
    except Exception as e:
        row["wrist_viol_err"] = f"{type(e).__name__}: {e}"
    if wb_path:  # 2D 손 재투영 (독립 정답). 프레임 안 맞으면 min 만큼만.
        try:
            row.update(hand_reproj(pt_path, wb_path))
        except Exception as e:
            row["hand_reproj_err"] = f"{type(e).__name__}: {e}"
    if do_smooth:
        sj = naive_smooth_joints(joints, sigma=2.0)
        for rname, idx in REGIONS.items():
            row[f"jitter_{rname}"] = jitter(sj, idx, fps)
    return row


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description="GT-free 메서드 비교 (jitter/bone/joint-limit).")
    ap.add_argument("--m", action="append", default=[], metavar="label=path",
                    help="메서드. 반복 가능. 예: --m ours=.../cmp_adaptive_v3.pt")
    ap.add_argument("--space", default="global", choices=["global", "incam"])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--wb", default=None, metavar="wholebody_kp2d.npy",
                    help="2D 손 재투영용 COCO-WholeBody kp(같은 클립). 주면 hand_px/hand_pck 추가.")
    ap.add_argument("--smooth-baseline", default=None, metavar="label=path",
                    help="이 메서드의 naive-smoothing 대조군 행을 추가(반칙 대조군).")
    ap.add_argument("--out", default=None, help="결과 prefix (.json/.csv 저장).")
    args = ap.parse_args(argv)

    specs = []
    for item in args.m:
        label, path = item.split("=", 1)
        specs.append((label, path, False))
    if args.smooth_baseline:
        label, path = args.smooth_baseline.split("=", 1)
        specs.append((f"{label}+smooth", path, True))

    if not specs:
        ap.error("최소 하나의 --m label=path 필요")

    rows = {}
    for label, path, sm in specs:
        print(f"[eval] {label:16s} ← {path}{'  (naive smooth)' if sm else ''}")
        rows[label] = evaluate(path, args.space, args.fps, args.device, do_smooth=sm, wb_path=args.wb)

    # 표 출력
    cols = ["frames", "jitter_hand", "wrist_axial_viol%", "wrist_any_viol%"]
    if args.wb:
        cols += ["hand_px", "hand_pck@0.5"]
    else:
        cols += ["jitter_arm", "bone_cov"]
    w = max(len(l) for l in rows) + 2
    print("\n" + "method".ljust(w) + "".join(c.rjust(15) for c in cols))
    print("-" * (w + 15 * len(cols)))
    for label, r in rows.items():
        print(label.ljust(w) + "".join(_fmt(r.get(c, "")).rjust(15) for c in cols))
    print("\n* jitter/accel 단위 m/s³·m/s² (root-relative, 낮을수록 부드러움)")
    print("* bone_cov=팔뼈 길이 변동계수(낮을수록 안정), wrist_*_viol%=손목 한계 위반율")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.with_suffix(".json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        allk = sorted({k for r in rows.values() for k in r})
        lines = ["method," + ",".join(allk)]
        for label, r in rows.items():
            lines.append(label + "," + ",".join(_fmt(r.get(k, "")) for k in allk))
        outp.with_suffix(".csv").write_text("\n".join(lines))
        print(f"\n저장: {outp.with_suffix('.json')} , {outp.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
