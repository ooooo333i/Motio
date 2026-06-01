"""
render_vitpose_wb.py

HaMeR에 들어가는 ViTPose WholaBody 133 keypoints를 원본 영상에 렌더링.

좌손 kp[91:112], 우손 kp[112:133] 을 원본 프레임에 겹쳐 그림.
HaMeR hand bbox (wholebody 기반) 도 함께 표시.

출력:
  outputs/cap_pipeline/input/vitpose_wb_overlay.mp4   (전체 영상)
  outputs/cap_pipeline/input/vitpose_wb_swing.mp4     (f320-430 클립)
"""
import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))
os.chdir(str(PROJ_ROOT))

# ── 관절 연결 (COCO wholebody hand, 0-based 내부 인덱스) ──────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),           # 검지
    (0, 9), (9, 10), (10, 11), (11, 12),      # 중지
    (0, 13), (13, 14), (14, 15), (15, 16),    # 약지
    (0, 17), (17, 18), (18, 19), (19, 20),    # 소지
]

# 133 keypoints 중 body (0..90), left_hand (91..111), right_hand (112..132)
L_HAND_START = 91
R_HAND_START = 112
HAND_LEN     = 21

# COCO17 body (vitpose.pt 구조 확인용)
COCO17_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),           # head
    (5,7),(7,9),(6,8),(8,10),          # arms
    (5,6),(5,11),(6,12),(11,12),       # torso
    (11,13),(13,15),(12,14),(14,16),   # legs
]


def draw_hand(img, kps, color_pt, color_line, conf_thr=0.2, radius=5, thickness=2):
    """kps: (21, 3) [x, y, conf]"""
    for i, j in HAND_CONNECTIONS:
        if kps[i, 2] > conf_thr and kps[j, 2] > conf_thr:
            p1 = (int(kps[i, 0]), int(kps[i, 1]))
            p2 = (int(kps[j, 0]), int(kps[j, 1]))
            cv2.line(img, p1, p2, color_line, thickness)
    for i in range(21):
        if kps[i, 2] > conf_thr:
            pt = (int(kps[i, 0]), int(kps[i, 1]))
            cv2.circle(img, pt, radius, color_pt, -1)
            # 손목(0번)은 더 크게
            if i == 0:
                cv2.circle(img, pt, radius + 3, color_pt, 2)


def compute_hand_bbox(kps, conf_thr=0.2, margin_ratio=0.3):
    """유효 키포인트에서 bbox 계산. kps: (21,3)"""
    valid = kps[kps[:, 2] > conf_thr]
    if len(valid) < 3:
        return None
    x1, y1 = valid[:, 0].min(), valid[:, 1].min()
    x2, y2 = valid[:, 0].max(), valid[:, 1].max()
    w, h = x2 - x1, y2 - y1
    margin = max(w, h) * margin_ratio
    return int(x1 - margin), int(y1 - margin), int(x2 + margin), int(y2 + margin)


def render_frames(wb_kp2d, frame_dir, out_path, fps, f_start=0, f_end=None,
                  conf_thr=0.2, show_bbox=True):
    frames = sorted(Path(frame_dir).glob("*.jpg"))
    if f_end is None:
        f_end = len(frames)
    frames = frames[f_start:f_end]

    if not frames:
        print(f"[warn] no frames found in {frame_dir}")
        return

    sample = cv2.imread(str(frames[0]))
    H, W = sample.shape[:2]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp = str(out_path).replace(".mp4", "_tmp.mp4")
    writer = cv2.VideoWriter(tmp, fourcc, fps, (W, H))

    for i, fpath in tqdm(enumerate(frames), total=len(frames), desc=Path(out_path).name):
        vid_f = f_start + i
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        if vid_f >= len(wb_kp2d):
            break

        kp = wb_kp2d[vid_f]  # (133, 3)
        lh = kp[L_HAND_START: L_HAND_START + HAND_LEN]  # (21,3)
        rh = kp[R_HAND_START: R_HAND_START + HAND_LEN]  # (21,3)

        # 오른손 (초록)
        draw_hand(img, rh, (0, 220, 80), (0, 160, 60), conf_thr=conf_thr)
        # 왼손 (파랑)
        draw_hand(img, lh, (80, 120, 255), (50, 80, 200), conf_thr=conf_thr)

        if show_bbox:
            rbox = compute_hand_bbox(rh, conf_thr=conf_thr)
            lbox = compute_hand_bbox(lh, conf_thr=conf_thr)
            if rbox:
                cv2.rectangle(img, (rbox[0], rbox[1]), (rbox[2], rbox[3]), (0, 220, 80), 2)
                cv2.putText(img, "R-hand bbox", (rbox[0], rbox[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2)
            if lbox:
                cv2.rectangle(img, (lbox[0], lbox[1]), (lbox[2], lbox[3]), (80, 120, 255), 2)
                cv2.putText(img, "L-hand bbox", (lbox[0], lbox[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 120, 255), 2)

        # 프레임 번호
        cv2.putText(img, f"f{vid_f}  GREEN=right  BLUE=left",
                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 100), 2)

        writer.write(img)

    writer.release()
    # h264 재인코딩
    os.system(f"ffmpeg -y -i {tmp} -vcodec libx264 -crf 20 -pix_fmt yuv420p {out_path} -loglevel error")
    os.remove(tmp)
    print(f"  saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  default="outputs/cap_pipeline/input")
    parser.add_argument("--frame-dir",  default="outputs/cap_pipeline/input/hamer_frames")
    parser.add_argument("--conf-thr",   type=float, default=0.2)
    parser.add_argument("--swing-only", action="store_true",
                        help="swing 구간(f320-430)만 렌더링")
    parser.add_argument("--fps",        type=float, default=None)
    args = parser.parse_args()

    d       = Path(args.input_dir)
    wb_path = d / "preprocess" / "wholebody_kp2d.npy"
    if not wb_path.exists():
        print(f"[error] {wb_path} 없음")
        return

    wb_kp2d = np.load(str(wb_path))   # (F, 133, 3)
    print(f"wholebody_kp2d: {wb_kp2d.shape}  conf_thr={args.conf_thr}")

    # FPS
    fps = args.fps
    if fps is None:
        import subprocess, json
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
             str(d / "0_input_video.mp4")],
            capture_output=True, text=True)
        info = json.loads(r.stdout)
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        num, den = map(int, v["r_frame_rate"].split("/"))
        fps = num / den
    print(f"FPS: {fps:.2f}")

    if args.swing_only:
        render_frames(wb_kp2d, args.frame_dir,
                      str(d / "vitpose_wb_swing.mp4"),
                      fps, f_start=320, f_end=431,
                      conf_thr=args.conf_thr)
    else:
        # 전체 + swing 클립 둘 다
        render_frames(wb_kp2d, args.frame_dir,
                      str(d / "vitpose_wb_overlay.mp4"),
                      fps, conf_thr=args.conf_thr)
        render_frames(wb_kp2d, args.frame_dir,
                      str(d / "vitpose_wb_swing.mp4"),
                      fps, f_start=320, f_end=431,
                      conf_thr=args.conf_thr)

    print("완료.")


if __name__ == "__main__":
    main()
