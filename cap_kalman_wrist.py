"""
cap_kalman_wrist.py

Constant-velocity Kalman filter on wrist rotation (quaternion space).
Outlier gating: frames > gate_deg° are treated as unreliable measurements.

Only processes left_wrist (body_pose[19]) and right_wrist (body_pose[20]).
hand_pose / other body joints: untouched.

파라미터:
  q_noise   : process noise (작을수록 부드러운 궤적 선호)
  r_noise   : measurement noise (클수록 측정값 불신, 예측 따름)
  gate_deg  : 이 각도 초과하면 r_noise *= gate_factor
  gate_factor: outlier 프레임에서 R 팽창 배수
"""
from __future__ import annotations
import argparse, copy
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


# ──────────────────────────────────────────────────────────────
# 1. Quaternion Kalman (constant-velocity, 8D state)
# ──────────────────────────────────────────────────────────────
class QuatKalman:
    """Constant-velocity Kalman for one wrist quaternion sequence.

    State  x = [q(4), dq(4)]  (quaternion + quaternion velocity)
    Transition F:  q_new = q + dq*dt,  dq_new = dq
    Measurement H: observe q directly
    """

    def __init__(self, q_noise: float, r_noise: float, dt: float = 1.0,
                 gate_deg: float = 33.0, gate_factor: float = 50.0):
        self.dt           = dt
        self.gate_rad     = np.radians(gate_deg)
        self.gate_factor  = gate_factor

        # Transition
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4) * dt

        # Measurement
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        # Noise
        self.Q = np.eye(8) * q_noise
        self.Q[4:, 4:] *= 0.1          # velocity process noise smaller
        self.R_base = np.eye(4) * r_noise

        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None

    def _init(self, q0: np.ndarray):
        self.x = np.zeros(8)
        self.x[:4] = q0
        self.P = np.eye(8) * 0.1

    def step(self, q_meas: np.ndarray) -> np.ndarray:
        """Feed one measurement, return filtered quaternion (4,)."""
        q_meas = np.asarray(q_meas, dtype=np.float64)

        if self.x is None:
            self._init(q_meas)
            return q_meas.copy()

        # Sign consistency
        if np.dot(q_meas, self.x[:4]) < 0:
            q_meas = -q_meas

        # ── Predict ──
        x_p = self.F @ self.x
        norm = np.linalg.norm(x_p[:4])
        if norm > 1e-8:
            x_p[:4] /= norm
        P_p = self.F @ self.P @ self.F.T + self.Q

        # ── Outlier gating ──
        try:
            R_pred = Rotation.from_quat(x_p[:4])
            R_meas = Rotation.from_quat(q_meas)
            angle  = abs((R_pred.inv() * R_meas).magnitude())
        except Exception:
            angle = 0.0
        R_eff = self.R_base * (self.gate_factor if angle > self.gate_rad else 1.0)

        # ── Update ──
        S = self.H @ P_p @ self.H.T + R_eff
        K = P_p @ self.H.T @ np.linalg.solve(S.T, np.eye(4)).T
        innov = q_meas - self.H @ x_p
        self.x = x_p + K @ innov
        self.P = (np.eye(8) - K @ self.H) @ P_p

        norm = np.linalg.norm(self.x[:4])
        if norm > 1e-8:
            self.x[:4] /= norm

        return self.x[:4].copy()


# ──────────────────────────────────────────────────────────────
# 2. 시퀀스 처리
# ──────────────────────────────────────────────────────────────
def kalman_smooth_rotation(aa_seq: np.ndarray,
                            q_noise: float, r_noise: float,
                            gate_deg: float, gate_factor: float) -> np.ndarray:
    """(F,3) axis-angle → Kalman → (F,3) axis-angle."""
    F = len(aa_seq)
    R_all = Rotation.from_rotvec(aa_seq.reshape(-1, 3))
    quats  = R_all.as_quat()                    # (F, 4) [x,y,z,w]

    # Sign consistency
    for i in range(1, F):
        if np.dot(quats[i], quats[i-1]) < 0:
            quats[i] = -quats[i]

    kf  = QuatKalman(q_noise=q_noise, r_noise=r_noise,
                     gate_deg=gate_deg, gate_factor=gate_factor)
    out = np.zeros_like(quats)
    for i in range(F):
        out[i] = kf.step(quats[i])

    return Rotation.from_quat(out).as_rotvec().astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 3. 평가 헬퍼
# ──────────────────────────────────────────────────────────────
def angular_velocity_stats(aa_seq: np.ndarray, label: str):
    R   = Rotation.from_rotvec(aa_seq.reshape(-1, 3))
    vel = np.degrees(np.abs((R[:-1].inv() * R[1:]).magnitude()))
    n90 = int((vel > 90).sum())
    print(f"  {label:22s}: p50={np.percentile(vel,50):.1f}° "
          f"p95={np.percentile(vel,95):.1f}° max={vel.max():.1f}° "
          f">90°:{n90}프레임")


# ──────────────────────────────────────────────────────────────
# 4. 메인
# ──────────────────────────────────────────────────────────────
CONFIGS = {
    "A": dict(q_noise=0.001,  r_noise=0.1),
    "B": dict(q_noise=0.0005, r_noise=0.2),
    "C": dict(q_noise=0.001,  r_noise=0.3),
}

def run(input_path: Path, output_dir: Path,
        gate_deg: float = 33.0, gate_factor: float = 50.0):

    pred = torch.load(input_path, map_location="cpu")
    F    = pred["smpl_params_incam"]["body_pose"].shape[0]
    bp_raw = pred["smpl_params_incam"]["body_pose"].reshape(F, 21, 3).numpy()

    print(f"\n[kalman] Input: {input_path.name}  F={F}")
    print(f"[kalman] gate_deg={gate_deg}°  gate_factor={gate_factor}×\n")

    # 원본 통계
    print("각속도 통계 (L_wrist / R_wrist):")
    angular_velocity_stats(bp_raw[:, 19, :], "HaMeR raw (L)")
    angular_velocity_stats(bp_raw[:, 20, :], "HaMeR raw (R)")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for cfg_name, cfg in CONFIGS.items():
        result = copy.deepcopy(pred)
        params = result["smpl_params_incam"]
        bp     = bp_raw.copy()

        for name, idx in [("L_wrist", 19), ("R_wrist", 20)]:
            bp[:, idx, :] = kalman_smooth_rotation(
                bp_raw[:, idx, :],
                q_noise=cfg["q_noise"], r_noise=cfg["r_noise"],
                gate_deg=gate_deg, gate_factor=gate_factor,
            )

        params["body_pose"] = torch.from_numpy(bp).reshape(F, 21, 3)
        result["smpl_params_incam"] = params

        if "smpl_params_global" in result:
            g  = result["smpl_params_global"]
            gb = g["body_pose"].reshape(F, 21, 3)
            gb[:, 19] = params["body_pose"].reshape(F, 21, 3)[:, 19]
            gb[:, 20] = params["body_pose"].reshape(F, 21, 3)[:, 20]
            g["body_pose"] = gb.reshape(F, 21, 3)
            result["smpl_params_global"] = g

        out_path = output_dir / f"smplx_kalman_{cfg_name}.pt"
        torch.save(result, out_path)

        print(f"\n  Config {cfg_name} (Q={cfg['q_noise']} R={cfg['r_noise']}):")
        angular_velocity_stats(bp[:, 19, :], f"  Kalman-{cfg_name} L")
        angular_velocity_stats(bp[:, 20, :], f"  Kalman-{cfg_name} R")
        results[cfg_name] = out_path
        print(f"  saved → {out_path}")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",    default="outputs/cap_pipeline/input")
    p.add_argument("--input-pt",     default="smplx_merged_prefit.pt")
    p.add_argument("--output-dir",   default="outputs/cap_pipeline/input")
    p.add_argument("--gate-deg",     type=float, default=33.0)
    p.add_argument("--gate-factor",  type=float, default=50.0)
    args = p.parse_args()

    run(
        input_path  = Path(args.input_dir) / args.input_pt,
        output_dir  = Path(args.output_dir),
        gate_deg    = args.gate_deg,
        gate_factor = args.gate_factor,
    )


if __name__ == "__main__":
    main()
