"""[DEPRECATED / 미사용] in-proc 접지 시도 — 축 어긋남으로 폐기.

  export 의 bake_space_transform/축변환은 export 시점에 적용되므로, export 직후의 라이브 씬은
  최종 FBX 좌표계가 아니다(검증: 발이 -1.4m 로 가라앉음). 정확한 접지는 *완성 FBX 를 import*
  하는 ground_fbx.py(2-pass)가 담당한다. 이 파일은 참고용으로만 남김(run_export 에서 미사용).

--- 이하 원래 설명 ---
ground_after_export.py — export_to_unreal_fbx.py 와 *같은 Blender 프로세스*에서
두 번째 -P 스크립트로 실행돼, 이미 만들어진 씬을 그대로 써서 발을 지면(Z=0)에 앉힌다.

별도 Blender 기동 + 모델 빌드 + 149MB FBX import(~10s)를 생략한다(ground_fbx.py 대비).
씬에는 export 스크립트가 만든 아머처(SMPLXRig)·메시가 이미 로드돼 있다.
접지 후 동일 Unreal 설정으로 `{base}_smplx_unreal.fbx` 를 덮어쓴다(다운로드 FBX 도 접지).

호출: blender -b <template> -P export_to_unreal_fbx.py -P ground_after_export.py -- <npz> <out_dir>
(export 스크립트와 동일한 -- 인자를 공유한다.)
"""
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
input_npz = os.path.abspath(argv[0])
output_dir = os.path.abspath(argv[1]) if len(argv) > 1 else os.path.dirname(input_npz)
base = os.path.splitext(os.path.basename(input_npz))[0]
out_fbx = os.path.join(output_dir, f"{base}_smplx_unreal.fbx")

if os.environ.get("CAP_GROUND", "1") != "1":
    sys.stderr.write("[GROUND] CAP_GROUND!=1 → 접지 생략\n")
    sys.exit(0)

meshes = [o for o in bpy.data.objects if o.type == "MESH"]
arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
if not meshes or arm is None:
    sys.stderr.write("[GROUND] 메시/아머처 없음 — 접지 생략\n")
    sys.exit(0)

sc = bpy.context.scene
f0, f1 = sc.frame_start, sc.frame_end
step = max(1, (f1 - f0) // 40)
mins = []
for f in range(f0, f1 + 1, step):
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    fmin = None
    for o in meshes:
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        mw = ev.matrix_world
        for v in me.vertices:
            z = (mw @ v.co).z
            if fmin is None or z < fmin:
                fmin = z
        ev.to_mesh_clear()
    if fmin is not None:
        mins.append(fmin)

if not mins:
    sys.stderr.write("[GROUND] 정점 없음 — 접지 생략\n")
    sys.exit(0)

mins.sort()
floor_z = mins[max(0, int(len(mins) * 0.05))]  # 하위 5%(이상치 강건)
arm.delta_location.z += -floor_z
sys.stderr.write(f"[GROUND] 최저 발 Z={floor_z:.3f}(샘플 {len(mins)}) → 루트 Z {-floor_z:+.3f}\n")

bpy.ops.object.select_all(action="DESELECT")
arm.select_set(True)
for m in meshes:
    m.select_set(True)
bpy.context.view_layer.objects.active = arm

bpy.ops.export_scene.fbx(
    filepath=out_fbx, use_selection=True, global_scale=1.0, apply_unit_scale=True,
    apply_scale_options="FBX_SCALE_NONE", object_types={"ARMATURE", "MESH"},
    use_mesh_modifiers=True, mesh_smooth_type="FACE", use_subsurf=False,
    add_leaf_bones=False, primary_bone_axis="Y", secondary_bone_axis="X",
    axis_forward="X", axis_up="Z", bake_space_transform=True, bake_anim=True,
    bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False, bake_anim_force_startend_keying=True,
    bake_anim_step=1.0, bake_anim_simplify_factor=0.0, path_mode="COPY", embed_textures=False,
)
sys.stderr.write(f"[GROUND] 접지 재출력(in-proc) -> {out_fbx}\n")
