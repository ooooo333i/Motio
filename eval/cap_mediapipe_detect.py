#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cap_mediapipe_detect.py — 독립 손 2D 검출(MediaPipe Hands) → mediapipe_hands.npy.

비순환 2D 평가용 제3 검출기. 출력 포맷 = (T, 2, 21, 3) full-image px (+z), 미검출=0.
hmp env 에서: /workspace/envs/hmp/bin/python cap_mediapipe_detect.py <clip_dir>
"""
import sys, numpy as np, cv2
import mediapipe as mp


def detect(clip_dir):
    import os
    vid = os.path.join(clip_dir, "0_input_video.mp4")
    cap = cv2.VideoCapture(vid)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                     min_detection_confidence=0.3, min_tracking_confidence=0.3)
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        f = np.zeros((2, 21, 3), np.float32)
        if res.multi_hand_landmarks:
            for h, lm in enumerate(res.multi_hand_landmarks[:2]):
                for k, p in enumerate(lm.landmark):
                    f[h, k] = [p.x * W, p.y * H, p.z]
        out.append(f)
    cap.release(); hands.close()
    arr = np.stack(out, 0).astype(np.float32)            # (T,2,21,3)
    op = os.path.join(clip_dir, "mediapipe_hands.npy")
    np.save(op, arr)
    det = (np.abs(arr).reshape(len(arr), -1).sum(1) > 0).sum()
    print(f"{clip_dir}: {len(arr)}f, 검출 {det}f ({det/len(arr)*100:.0f}%) -> {op}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        detect(d)
