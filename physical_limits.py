"""
physical_limits.py — 팔/손목 관절의 물리적 한계 레퍼런스.

이후 모든 단계(HMP 주입, 검증, 게이팅)의 검증 기준(ground truth).
코드에서 import 해서 사용:

    from physical_limits import JOINT_LIMITS, WRIST_AXIS_MAP, MAX_WRIST_DEG_PER_FRAME
    from physical_limits import decompose_wrist_local, check_wrist_rotation

================================================================================
SMPL-X 좌표축 대응 (2026-06-01 경험적 검증 — rest pose에서 직접 확인)
================================================================================
검증 방법: SMPL-X rest pose에서 body_pose[wrist]에 각 로컬 축으로 +30° 회전을
적용하고, 손 관절 centroid가 손바닥 평면(palm normal=+Y)에 대해 어느 방향으로
움직이는지 측정.

forearm 방향 (elbow→wrist): L=[+0.996,...]≈+X, R=[-0.998,...]≈-X  → forearm 축 = X
palm normal (손 PCA 최소축): ≈ +Y (양손)

결과 (양손 동일, 부호만 반대):
  rot about X → 손 centroid 거의 안 움직임 (|Δ|≈0.007)  → X = AXIAL (회내/회외, twist)
  rot about Y → deviation 방향 이동 (dev≈0.055)         → Y = RADIAL/ULNAR DEVIATION
  rot about Z → flexion 방향 이동 (flex≈0.055)          → Z = FLEXION/EXTENSION

⚠️ 주의: 초기 비공식 메모에 'Y=flex/ext, Z=deviation'으로 적혀 있었으나
   경험적 검증 결과 Y와 Z가 반대였음. 이 파일이 정확한 기준임.

부호 규약:
  - axial(X) 부호는 좌/우 손에서 반대 (forearm이 +X vs -X). 크기 한계는 동일.
  - 검증은 가능하면 부호가 아닌 '범위/크기'로 수행.
================================================================================
"""
from __future__ import annotations
import numpy as np

try:
    from scipy.spatial.transform import Rotation as _R
except Exception:  # scipy 없는 환경 대비
    _R = None


# ──────────────────────────────────────────────────────────────────────────
# SMPL-X body_pose 인덱스 (body_pose[i] = full SMPL-X joint i+1, root 제외)
# ──────────────────────────────────────────────────────────────────────────
BODY_POSE_IDX = {
    "L_shoulder": 16, "R_shoulder": 17,
    "L_elbow": 18,    "R_elbow": 19,
    "L_wrist": 19 - 0,  # placeholder, 아래에서 명시
}
# 명시적 (혼동 방지)
BODY_POSE_IDX = {
    "L_collar": 11, "R_collar": 12,
    "L_shoulder": 13, "R_shoulder": 14,
    "L_elbow": 17, "R_elbow": 18,
    "L_wrist": 19, "R_wrist": 20,
}
# FK chain (pelvis→elbow) body_pose 인덱스 — cap_inject_hmp.py와 동일
LEFT_CHAIN_BP  = [2, 5, 8, 12, 15, 17]   # to L elbow (joint 18)
RIGHT_CHAIN_BP = [2, 5, 8, 13, 16, 18]   # to R elbow (joint 19)


# ──────────────────────────────────────────────────────────────────────────
# 로컬 축 → 해부학적 운동 대응 (경험적 검증됨)
# ──────────────────────────────────────────────────────────────────────────
WRIST_AXIS_MAP = {
    "X": "axial",       # pronation / supination (forearm twist)
    "Y": "deviation",   # radial / ulnar deviation
    "Z": "flexion",     # flexion / extension
}
WRIST_AXIS_INDEX = {"axial": 0, "deviation": 1, "flexion": 2}  # axis-angle / euler XYZ 성분

ELBOW_AXIS_NOTE = (
    "Elbow는 사실상 1-DOF 힌지. X(forearm twist)는 ~0이어야 정상 "
    "(elbow에서 비틀림은 비물리적). 굴곡은 Y/Z 회전으로 나타남 "
    "(rest에서 upper-arm·forearm 모두 X축, 굴곡 시 wrist가 -X(몸쪽)로 당겨짐)."
)
SHOULDER_AXIS_NOTE = (
    "Shoulder는 3-DOF 볼조인트. 모든 축 회전이 유효 (flexion/abduction/rotation). "
    "축별 단일 라벨 부여 부적절 — ROM 범위로만 검증."
)


# ──────────────────────────────────────────────────────────────────────────
# 관절 ROM (정상 가동범위 normal + 하드 한계 margin, degree)
#   normal: 일반적 능동 가동범위
#   hard  : 이 범위를 넘으면 비물리적 → 노이즈/오류로 판정
#   부호: (-)=한 방향, (+)=반대 방향. 손목 axial은 좌우 부호 반대이므로
#         abs 기반 검증 권장.
# ──────────────────────────────────────────────────────────────────────────
JOINT_LIMITS = {
    "wrist": {
        # forearm pronation/supination — SMPL-X에선 wrist X축 twist로 나타남
        "axial": {
            "smplx_axis": "X",
            "normal_neg": -60.0,   # pronation
            "normal_pos": 140.0,   # supination
            "hard_neg": -90.0,
            "hard_pos": 160.0,
            "note": "forearm 회내(-)/회외(+). SMPL-X엔 별도 radioulnar 관절 없어 wrist X에 합쳐짐. 좌/우 부호 반대.",
        },
        # radial/ulnar deviation — wrist Y축
        "deviation": {
            "smplx_axis": "Y",
            "normal_neg": -19.0,   # radial deviation
            "normal_pos": 33.0,    # ulnar deviation
            "hard_neg": -45.0,
            "hard_pos": 45.0,
            "note": "노뼈쪽(-)/자뼈쪽(+) 편향. margin ±45°.",
        },
        # flexion/extension — wrist Z축
        "flexion": {
            "smplx_axis": "Z",
            "normal_neg": -71.0,   # extension (뒤로 젖힘)
            "normal_pos": 73.0,    # flexion (앞으로 굽힘)
            "hard_neg": -90.0,
            "hard_pos": 90.0,
            "note": "굴곡(+)/신전(-). margin ±90°.",
        },
    },
    "elbow": {
        "flexion": {
            "smplx_axis": "Y/Z (hinge plane)",
            "normal_neg": 0.0,     # 완전 신전 (과신전 거의 0)
            "normal_pos": 150.0,   # 굴곡
            "hard_neg": -10.0,     # 약간의 과신전 허용
            "hard_pos": 160.0,
            "note": ELBOW_AXIS_NOTE,
        },
        "axial_twist": {
            "smplx_axis": "X",
            "normal_neg": -5.0,
            "normal_pos": 5.0,
            "hard_neg": -15.0,
            "hard_pos": 15.0,
            "note": "elbow에서 forearm twist는 비물리적. ~0이어야 함. 초과 시 오류.",
        },
    },
    "shoulder": {
        # 3-DOF — 축별 라벨 대신 운동별 ROM
        "flexion": {     # 앞으로 들기
            "normal_neg": -60.0,   # extension (뒤로)
            "normal_pos": 180.0,
            "hard_neg": -70.0,
            "hard_pos": 185.0,
            "note": SHOULDER_AXIS_NOTE,
        },
        "abduction": {   # 옆으로 벌리기
            "normal_neg": 0.0,
            "normal_pos": 180.0,
            "hard_neg": -30.0,     # adduction
            "hard_pos": 185.0,
            "note": SHOULDER_AXIS_NOTE,
        },
        "rotation": {    # 내/외회전
            "normal_neg": -90.0,   # internal
            "normal_pos": 90.0,    # external
            "hard_neg": -100.0,
            "hard_pos": 100.0,
            "note": SHOULDER_AXIS_NOTE,
        },
    },
}


# ──────────────────────────────────────────────────────────────────────────
# 각속도 한계
#   인간 손목 최대 각속도 ~2000°/s (테니스 서브 등 극단). 그 이상 = 비물리적.
# ──────────────────────────────────────────────────────────────────────────
FPS_DEFAULT = 60.0
MAX_WRIST_DEG_PER_SEC = 2000.0
MAX_WRIST_DEG_PER_FRAME = MAX_WRIST_DEG_PER_SEC / FPS_DEFAULT   # 33.33°/frame @60fps

# 관절별 최대 각속도 (deg/s) — 손목이 가장 빠름
MAX_DEG_PER_SEC = {
    "wrist": 2000.0,
    "elbow": 1500.0,
    "shoulder": 1000.0,
}

def max_deg_per_frame(joint: str = "wrist", fps: float = FPS_DEFAULT) -> float:
    """관절·fps별 프레임당 최대 허용 각도 변화."""
    return MAX_DEG_PER_SEC[joint] / fps


# ──────────────────────────────────────────────────────────────────────────
# 헬퍼: wrist 로컬 회전 분해 / 검증
# ──────────────────────────────────────────────────────────────────────────
def decompose_wrist_local(aa_or_rotmat) -> dict:
    """wrist 로컬 회전(axis-angle (3,) 또는 rotmat (3,3))을
    (axial, deviation, flexion) degree로 분해.

    SMPL-X 검증 결과 X=axial, Y=deviation, Z=flexion 이므로
    intrinsic euler 'XYZ' 1·2·3 성분이 각각 axial·deviation·flexion에 대응.
    (대각도에서는 euler coupling이 있어 근사치 — 게이팅/판정용.)
    """
    if _R is None:
        raise RuntimeError("scipy 필요")
    arr = np.asarray(aa_or_rotmat, dtype=np.float64)
    R = _R.from_rotvec(arr) if arr.shape == (3,) else _R.from_matrix(arr)
    ax, dev, flex = R.as_euler("XYZ", degrees=True)
    return {"axial": float(ax), "deviation": float(dev), "flexion": float(flex)}


def check_wrist_rotation(aa_or_rotmat, level: str = "hard") -> dict:
    """wrist 로컬 회전을 분해해 각 성분이 한계 내인지 검사.
    level: 'normal' 또는 'hard'.
    반환: {component: {"value":deg, "in_range":bool, "lo":, "hi":}}
    부호가 좌/우에 따라 다른 axial은 hard 범위(대칭 확장)로 판정 권장.
    """
    comp = decompose_wrist_local(aa_or_rotmat)
    lim = JOINT_LIMITS["wrist"]
    out = {}
    for name, val in comp.items():
        lo = lim[name][f"{level}_neg"]
        hi = lim[name][f"{level}_pos"]
        out[name] = {"value": val, "lo": lo, "hi": hi, "in_range": lo <= val <= hi}
    return out


def check_angular_velocity(rotvec_seq, fps: float = FPS_DEFAULT, joint: str = "wrist") -> dict:
    """연속 프레임 wrist 로컬 회전 시퀀스의 프레임간 geodesic 각속도 검사.
    rotvec_seq: (F,3) axis-angle. 반환: per-frame deg, 초과 프레임 인덱스.
    """
    if _R is None:
        raise RuntimeError("scipy 필요")
    Rs = _R.from_rotvec(np.asarray(rotvec_seq, dtype=np.float64))
    mats = Rs.as_matrix()
    limit = max_deg_per_frame(joint, fps)
    vels = np.zeros(len(mats))
    for i in range(1, len(mats)):
        d = mats[i - 1].T @ mats[i]
        vels[i] = np.degrees(np.linalg.norm(_R.from_matrix(d).as_rotvec()))
    bad = np.where(vels > limit)[0].tolist()
    return {"vel_deg_per_frame": vels, "limit": limit, "bad_frames": bad}


if __name__ == "__main__":
    print("SMPL-X wrist 축 대응 (검증됨):", WRIST_AXIS_MAP)
    print("손목 최대 각속도:", MAX_WRIST_DEG_PER_FRAME, "deg/frame @", FPS_DEFAULT, "fps")
    print()
    import numpy as _np
    # 예시: 30° flexion(Z) 단일 회전
    aa = _np.array([0, 0, _np.radians(30)])
    print("예시 30° about Z:", decompose_wrist_local(aa))
    print("check(hard):", {k: v["in_range"] for k, v in check_wrist_rotation(aa).items()})
