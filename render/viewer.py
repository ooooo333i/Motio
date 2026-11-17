#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.utils.net_utils import to_cuda
from hmr4d.utils.smplx_utils import make_smplx


SMPLX_ALLOWED_KEYS = {
    "betas",
    "body_pose",
    "global_orient",
    "transl",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "expression",
}


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def filter_smplx_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k in SMPLX_ALLOWED_KEYS}


def move_to_start_point_face_z(verts: torch.Tensor, joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    verts = verts.clone()
    joints = joints.clone()
    offset = joints[0, 0].clone()
    offset[1] = verts[:, :, 1].min()
    verts = verts - offset
    joints = joints - offset
    transform = compute_T_ayfz2ay(joints[[0]], inverse=True)
    verts = apply_T_on_points(verts, transform)
    joints = apply_T_on_points(joints, transform)
    return verts, joints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open an interactive 3D viewer for merged SMPL-X results."
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        help="CAP run directory, e.g. outputs/cap_pipeline/4k_tennis",
    )
    parser.add_argument(
        "--merged",
        type=Path,
        default=None,
        help="Merged .pt path. Defaults to BASE/smplx_merged_hamer_post.pt, then BASE/smplx_merged_hamer.pt",
    )
    parser.add_argument(
        "--space",
        choices=("global", "incam"),
        default="global",
        help="Which SMPL-X parameter space to visualize.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to show when --play is off.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the sequence in the viewer. Without this, only one frame is shown.",
    )
    parser.add_argument("--fps", type=float, default=24.0, help="Playback FPS when --play is enabled.")
    parser.add_argument("--stride", type=int, default=1, help="Playback frame stride.")
    parser.add_argument("--body-color", type=float, nargs=3, default=(0.78, 0.82, 0.86))
    parser.add_argument("--bg-color", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--wireframe", action="store_true")
    parser.add_argument("--no-align", action="store_true", help="Do not re-center / face-Z align global output.")
    parser.add_argument("--view-rot-x", type=float, default=90.0, help="Viewer-space mesh rotation around X in degrees.")
    parser.add_argument("--view-rot-y", type=float, default=0.0, help="Viewer-space mesh rotation around Y in degrees.")
    parser.add_argument("--view-rot-z", type=float, default=0.0, help="Viewer-space mesh rotation around Z in degrees.")
    return parser.parse_args()


def resolve_merged_path(base_dir: Path, merged_arg: Path | None) -> Path:
    if merged_arg is not None:
        merged = merged_arg.resolve()
        if not merged.is_file():
            raise FileNotFoundError(f"Merged result not found: {merged}")
        return merged

    candidates = [
        base_dir / "smplx_merged_hamer_post.pt",
        base_dir / "smplx_merged_hamer.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"No merged result found under: {base_dir}")


def load_vertices(merged_path: Path, space: str, align: bool) -> tuple[np.ndarray, np.ndarray]:
    pred = torch_load(merged_path)
    param_key = "smpl_params_global" if space == "global" else "smpl_params_incam"
    if param_key not in pred:
        raise KeyError(f"{merged_path} has no {param_key}")

    params = filter_smplx_params(pred[param_key])
    smplx = make_smplx("supermotion_fullhands").cuda()
    with torch.no_grad():
        smplx_out = smplx(**to_cuda(params))

    verts = smplx_out.vertices
    joints = smplx_out.joints
    if space == "global" and align:
        verts, joints = move_to_start_point_face_z(verts, joints)
    return verts.detach().cpu().numpy().astype(np.float32), np.asarray(smplx.faces, dtype=np.int32)


def make_mesh(
    pyrender: Any,
    trimesh: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: tuple[float, float, float],
    wireframe: bool,
    view_rot_xyz: tuple[float, float, float],
) -> Any:
    vertex_colors = np.tile(np.asarray([*color, 1.0], dtype=np.float32), (vertices.shape[0], 1))
    mesh_trimesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy(), vertex_colors=vertex_colors, process=False)
    for angle_deg, axis in zip(view_rot_xyz, ([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        if abs(float(angle_deg)) > 1e-6:
            mesh_trimesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(angle_deg), axis))
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1,
        roughnessFactor=0.8,
        baseColorFactor=(*color, 1.0),
        alphaMode="OPAQUE",
    )
    return pyrender.Mesh.from_trimesh(mesh_trimesh, material=material, smooth=True, wireframe=wireframe)


def run_viewer(verts: np.ndarray, faces: np.ndarray, args: argparse.Namespace) -> None:
    import pyrender
    import trimesh

    scene = pyrender.Scene(bg_color=[*args.bg_color, 1.0], ambient_light=(0.35, 0.35, 0.35))
    frame_idx = max(0, min(int(args.frame), len(verts) - 1))
    view_rot_xyz = (float(args.view_rot_x), float(args.view_rot_y), float(args.view_rot_z))
    mesh_node_ref = [
        scene.add(
            make_mesh(
                pyrender,
                trimesh,
                verts[frame_idx],
                faces,
                tuple(args.body_color),
                args.wireframe,
                view_rot_xyz,
            ),
            name="smplx_body",
        )
    ]

    viewer = pyrender.Viewer(
        scene,
        use_raymond_lighting=True,
        run_in_thread=bool(args.play),
        viewport_size=(1400, 1000),
        title="SMPL-X Interactive Viewer",
    )

    if not args.play:
        return

    stop_event = threading.Event()

    def animate() -> None:
        idx = frame_idx
        delay = 1.0 / max(args.fps, 1e-6)
        stride = max(1, int(args.stride))
        while viewer.is_active and not stop_event.is_set():
            next_idx = (idx + stride) % len(verts)
            mesh = make_mesh(
                pyrender,
                trimesh,
                verts[next_idx],
                faces,
                tuple(args.body_color),
                args.wireframe,
                view_rot_xyz,
            )
            with viewer.render_lock:
                scene.remove_node(mesh_node_ref[0])
                mesh_node_ref[0] = scene.add(mesh, name="smplx_body")
            idx = next_idx
            time.sleep(delay)

    thread = threading.Thread(target=animate, daemon=True)
    thread.start()
    try:
        while viewer.is_active:
            time.sleep(0.1)
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        viewer.close_external()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    merged_path = resolve_merged_path(base_dir, args.merged)
    verts, faces = load_vertices(merged_path, args.space, align=not args.no_align)
    run_viewer(verts, faces, args)


if __name__ == "__main__":
    main()
