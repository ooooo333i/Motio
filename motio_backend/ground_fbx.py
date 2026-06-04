"""ground_fbx.py — 이미 만들어진 FBX 를 Blender 에서 실제 메시 기준으로 지면에 앉힌다.

단안(GVHMR) 모션은 지면 기준이 아니라 카메라 공간이라 캐릭터가 통째로 공중에 뜬다.
SMPL-X 공간에서 접지해도 애드온/AMASS 변환이 수직 오프셋을 더해 최종 FBX 와 불일치하므로,
*완성된 FBX 의 실제 메시 월드 Z* 최저값을 프레임에 걸쳐 구해 루트를 그만큼 내려 발을 Z=0 에 둔다.

사용: blender -b -P ground_fbx.py -- <fbx_in> <fbx_out>
export_to_unreal_fbx.py(동결) 의 FBX 설정과 동일하게 다시 내보낸다(Unreal Z-up 유지).
"""
import sys
import bpy
import mathutils

argv = sys.argv[sys.argv.index("--") + 1:]
fbx_in, fbx_out = argv[0], argv[1]
fps = float(argv[2]) if len(argv) > 2 else 0.0

bpy.ops.wm.read_factory_settings(use_empty=True)
# ⚠️ FBX import 는 scene fps 로 프레임을 환산한다. 기본 24fps 씬에 60fps FBX 를 넣으면
#    600프레임 → ~240프레임으로 다운샘플돼 프레임이 손실된다. import 전에 원본 fps 로 고정.
sc = bpy.context.scene
if fps > 0:
    sc.render.fps = int(round(fps))
    sc.render.fps_base = 1.0
bpy.ops.import_scene.fbx(filepath=fbx_in)
if fps > 0:  # import 가 바꿨을 수 있어 재출력 전 다시 고정
    sc.render.fps = int(round(fps))
    sc.render.fps_base = 1.0

# ⚠️ import 후 scene.frame_end 가 Blender 기본값(250)에 머물러, bake_anim 재출력이 1~250 만
#    굽고 나머지 프레임을 잘라먹는다. 임포트된 액션 길이로 frame_end 를 확장(프레임 손실 방지).
_end = 1
for _o in bpy.data.objects:
    if _o.animation_data and _o.animation_data.action:
        _end = max(_end, int(round(_o.animation_data.action.frame_range[1])))
sc.frame_start = 1
sc.frame_end = _end
sys.stderr.write(f"[GROUND] frame_end 확장 → {_end} (프레임 손실 방지)\n")

meshes = [o for o in bpy.data.objects if o.type == "MESH"]
arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
if not meshes or arm is None:
    sys.stderr.write("[GROUND] 메시/아머처를 찾지 못함 — 접지 생략, 원본 복사\n")
    bpy.ops.export_scene.fbx(filepath=fbx_out, use_selection=False)
    sys.exit(0)

depsgraph = bpy.context.evaluated_depsgraph_get()

# 프레임 전체를 균등 샘플(최대 ~40개)해 메시 월드 Z 최저값을 구함(전 프레임 평가는 과함)
f0, f1 = sc.frame_start, sc.frame_end
total = max(1, f1 - f0)
step = max(1, total // 40)
mins = []
for f in range(f0, f1 + 1, step):
    sc.frame_set(f)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    frame_min = None
    for o in meshes:
        ev = o.evaluated_get(depsgraph)
        me = ev.to_mesh()
        mw = ev.matrix_world
        for v in me.vertices:
            z = (mw @ v.co).z
            if frame_min is None or z < frame_min:
                frame_min = z
        ev.to_mesh_clear()
    if frame_min is not None:
        mins.append(frame_min)

if not mins:
    sys.stderr.write("[GROUND] 정점 없음 — 접지 생략\n")
    dz = 0.0
else:
    mins.sort()
    # 단일 프레임 이상치(발 dip)에 강건하도록 하위 5% 지점을 지면으로 사용
    floor_z = mins[max(0, int(len(mins) * 0.05))]
    dz = -floor_z
    sys.stderr.write(f"[GROUND] 최저 발 Z={floor_z:.3f} (샘플 {len(mins)}프레임) → 루트 Z {dz:+.3f}\n")

# 루트(아머처)를 dz 만큼 내려 접지. delta_location 은 애니메이션 키프레임 위에 더해지므로
# 베이크된 루트 모션을 보존하면서 전체를 평행이동한다.
arm.delta_location.z += dz

# export_to_unreal_fbx.py 와 동일 설정으로 재출력(Unreal Z-up 유지)
bpy.ops.object.select_all(action="DESELECT")
arm.select_set(True)
for m in meshes:
    m.select_set(True)
bpy.context.view_layer.objects.active = arm

bpy.ops.export_scene.fbx(
    filepath=fbx_out,
    use_selection=True,
    global_scale=1.0,
    apply_unit_scale=True,
    apply_scale_options="FBX_SCALE_NONE",
    object_types={"ARMATURE", "MESH"},
    use_mesh_modifiers=True,
    mesh_smooth_type="FACE",
    use_subsurf=False,
    add_leaf_bones=False,
    primary_bone_axis="Y",
    secondary_bone_axis="X",
    axis_forward="X",
    axis_up="Z",
    bake_space_transform=True,
    bake_anim=True,
    bake_anim_use_all_bones=True,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False,
    bake_anim_force_startend_keying=True,
    bake_anim_step=1.0,
    bake_anim_simplify_factor=0.0,
    path_mode="COPY",
    embed_textures=False,
)
sys.stderr.write(f"[GROUND] 접지 FBX -> {fbx_out}\n")
