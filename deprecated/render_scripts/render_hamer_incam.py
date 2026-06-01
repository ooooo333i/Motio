#!/usr/bin/env python3
"""Render HaMeR MANO meshes onto original video frames (full-frame projection).

Uses cam_t_full + per-frame focal_length from hamer_out_clean npz files.
Writes output to outputs/cap_pipeline/input/hamer_incam_wb133.mp4
"""
import os, sys, time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pyrender
import trimesh

PROJ_ROOT  = Path(__file__).resolve().parent
FRAME_DIR  = PROJ_ROOT / "outputs/cap_pipeline/input/hamer_frames"
HAMER_DIR  = PROJ_ROOT / "outputs/cap_pipeline/input/hamer_out_clean"
VIDEO_IN   = PROJ_ROOT / "outputs/cap_pipeline/input/0_input_video.mp4"
VIDEO_OUT  = PROJ_ROOT / "outputs/cap_pipeline/input/hamer_incam_wb133.mp4"

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# ── MANO faces ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(PROJ_ROOT / "external/hamer"))
os.chdir(str(PROJ_ROOT / "external/hamer"))
from hamer.models import DEFAULT_CHECKPOINT, load_hamer  # noqa: E402
_hamer_model, _hamer_cfg = load_hamer(DEFAULT_CHECKPOINT)
_mano_faces_right = _hamer_model.mano.faces.copy()   # (1538, 3)
_mano_faces_left  = _mano_faces_right[:, [0, 2, 1]]  # flip winding for left
del _hamer_model
os.chdir(str(PROJ_ROOT))

# Weld seam faces (same as HaMeR Renderer)
_SEAM_FACES = np.array([
    [92,38,234],[234,38,239],[38,122,239],[239,122,279],[122,118,279],
    [279,118,215],[118,117,215],[215,117,214],[117,119,214],[214,119,121],
    [119,120,121],[121,120,78],[120,108,78],[78,108,79]], dtype=np.int32)
FACES_RIGHT = np.concatenate([_mano_faces_right, _SEAM_FACES], axis=0)
FACES_LEFT  = np.concatenate([_mano_faces_left,  _SEAM_FACES[:,[0,2,1]]], axis=0)

# Colours
COLOR_RIGHT = (0.9, 0.75, 0.6)   # skin-ish
COLOR_LEFT  = (0.6, 0.75, 0.9)   # blue-ish


def create_lights():
    thetas = np.pi * np.array([1/6, 5/6, 3/2])
    phis   = np.pi * np.array([0,   0,   1/2])
    nodes  = []
    for theta, phi in zip(thetas, phis):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z  = np.array([xp, yp, zp])
        z  = z / np.linalg.norm(z)
        x  = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x  = x / np.linalg.norm(x)
        y  = np.cross(z, x)
        M  = np.eye(4); M[:3, :3] = np.stack([x, y, z], axis=1)
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=M))
    return nodes


def render_hands_on_frame(
    img_bgr: np.ndarray,
    detections: list,          # list of dict: vertices, cam_t_full, focal_length, is_right
) -> np.ndarray:
    """Render all detected hands onto img_bgr using pyrender."""
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)
    scene    = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))

    for d in detections:
        verts     = d['vertices'].copy()   # (778, 3)
        cam_t     = d['cam_t_full'].copy() # (3,)
        focal     = float(d['focal_length'])
        is_right  = bool(d['is_right'])

        # Flip left-hand vertices
        if not is_right:
            verts[:, 0] *= -1

        # Build trimesh in camera space
        faces  = FACES_RIGHT if is_right else FACES_LEFT
        color  = COLOR_RIGHT  if is_right else COLOR_LEFT
        mesh_t = trimesh.Trimesh(verts + cam_t, faces)
        # pyrender expects Y-down camera → rotate 180° around X
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh_t.apply_transform(rot)

        mat  = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0,
                                                   alphaMode='OPAQUE',
                                                   baseColorFactor=(*color, 1.0))
        scene.add(pyrender.Mesh.from_trimesh(mesh_t, mat))

    # Camera at origin (vertices already in cam space)
    # Use focal_length from the last detection (they share the same intrinsics per frame)
    focal = float(detections[-1]['focal_length']) if detections else 5000.0
    cam_t_flip = np.array([0.0, 0.0, 0.0])
    cam_pose   = np.eye(4)
    cam_pose[:3, 3] = cam_t_flip
    camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal,
                                       cx=W / 2.0, cy=H / 2.0, zfar=1e12)
    scene.add(camera, pose=cam_pose)

    for node in create_lights():
        scene.add_node(node)

    color_r, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    mask   = (color_r[:, :, 3:].astype(np.float32) / 255.0)
    rend   = color_r[:, :, :3].astype(np.float32) / 255.0
    result = rend * mask + img_rgb * (1 - mask)
    return (result * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1]  # → BGR


# ── Load all npz and group by frame id ────────────────────────────────────────
print("Loading HaMeR detections...", end=" ", flush=True)
frame_dets: dict[str, list] = defaultdict(list)
for npz_path in sorted(HAMER_DIR.glob("*.npz")):
    frame_id = npz_path.stem.rsplit("_", 1)[0]
    d = np.load(str(npz_path), allow_pickle=True)
    frame_dets[frame_id].append({
        'vertices':    np.asarray(d['vertices'],    dtype=np.float32),
        'cam_t_full':  np.asarray(d['cam_t_full'],  dtype=np.float32).reshape(3),
        'focal_length': float(np.asarray(d['focal_length']).reshape(-1)[0]),
        'is_right':    bool(float(np.asarray(d['is_right']).reshape(-1)[0])),
    })
print(f"{len(frame_dets)} frames with detections")

# ── Setup video writer ────────────────────────────────────────────────────────
from hmr4d.utils.video_io_utils import get_video_fps
fps = get_video_fps(str(VIDEO_IN))
frame_paths = sorted(FRAME_DIR.glob("*.jpg"))
print(f"Rendering {len(frame_paths)} frames at {fps:.2f} fps → {VIDEO_OUT.name}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
sample = cv2.imread(str(frame_paths[0]))
H, W   = sample.shape[:2]
writer = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (W, H))

tic = time.perf_counter()
for i, fp in enumerate(frame_paths):
    frame_id = fp.stem                    # "0200"
    img = cv2.imread(str(fp))
    dets = frame_dets.get(frame_id, [])
    if dets:
        img = render_hands_on_frame(img, dets)
    writer.write(img)
    if (i + 1) % 100 == 0:
        elapsed = time.perf_counter() - tic
        fps_so_far = (i + 1) / elapsed
        remain    = (len(frame_paths) - i - 1) / fps_so_far
        print(f"  {i+1}/{len(frame_paths)}  {fps_so_far:.1f}fps  ETA {remain:.0f}s")

writer.release()
elapsed = time.perf_counter() - tic
print(f"\nDone in {elapsed:.1f}s  ({len(frame_paths)/elapsed:.1f} fps)")
print(f"Saved: {VIDEO_OUT}")
