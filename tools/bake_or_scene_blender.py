#!/usr/bin/env python3

"""Bake OBJ scene transforms in Blender and export a normalized OBJ.

Run with:
  blender --background --python tools/bake_or_scene_blender.py -- \
      OUTPUT.obj INPUT_A.obj INPUT_B.obj \
      --scale 4.8 \
      --rot-quat 0.6721136569976807 0.2843584716320038 0.3799154460430145 -0.5683904886245728 \
      --floor-to-z 0 \
      --origin-mode center_xy
"""

import argparse
import math
import os
import sys
import traceback
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []

    parser = argparse.ArgumentParser(description="Bake OBJ scale/rotation/alignment and export a new OBJ.")
    parser.add_argument("output", type=str, help="Output OBJ path.")
    parser.add_argument("inputs", nargs="+", type=str, help="Input OBJ path(s).")
    parser.add_argument(
        "--scale",
        nargs="+",
        type=float,
        default=[1.0],
        help="Uniform scale or XYZ scale. Defaults to 1.",
    )
    parser.add_argument(
        "--rot-quat",
        nargs=4,
        type=float,
        metavar=("W", "X", "Y", "Z"),
        default=[1.0, 0.0, 0.0, 0.0],
        help="Rotation quaternion in Isaac/USD WXYZ order.",
    )
    parser.add_argument(
        "--translation",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, 0.0],
        help="Translation applied after scale and rotation, before final alignment.",
    )
    parser.add_argument(
        "--floor-to-z",
        type=float,
        default=0.0,
        help="Shift the baked mesh so its minimum Z lands on this value. Use --no-floor-align to disable.",
    )
    parser.add_argument(
        "--no-floor-align",
        action="store_true",
        default=False,
        help="Do not shift the baked mesh vertically after applying transforms.",
    )
    parser.add_argument(
        "--origin-mode",
        choices=["keep_xy", "center_xy"],
        default="center_xy",
        help="Keep baked XY coordinates or center the bounding box around XY origin.",
    )
    return parser.parse_args(argv)


def _scale_vector(values: list[float]) -> Vector:
    if len(values) == 1:
        return Vector((values[0], values[0], values[0]))
    if len(values) == 3:
        return Vector(values)
    raise ValueError("--scale expects either one value or three XYZ values.")


def _import_obj(path: str) -> list[bpy.types.Object]:
    before = set(bpy.context.scene.objects)
    kwargs = {
        "filepath": path,
        "forward_axis": "Y",
        "up_axis": "Z",
        "use_split_objects": True,
        "use_split_groups": False,
    }
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(**kwargs)
    else:
        bpy.ops.import_scene.obj(filepath=path, axis_forward="Y", axis_up="Z")

    imported = [obj for obj in bpy.context.scene.objects if obj not in before and obj.type == "MESH"]
    if not imported:
        raise RuntimeError(f"No mesh objects imported from {path}")
    return imported


def _export_obj(path: str, objects: list[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=os.path.abspath(path),
            check_existing=False,
            export_selected_objects=True,
            export_uv=True,
            export_normals=True,
            export_materials=True,
            path_mode="COPY",
            forward_axis="Y",
            up_axis="Z",
            global_scale=1.0,
        )
    else:
        bpy.ops.export_scene.obj(
            filepath=os.path.abspath(path),
            check_existing=False,
            use_selection=True,
            use_materials=True,
            path_mode="COPY",
            axis_forward="Y",
            axis_up="Z",
            global_scale=1.0,
        )


def _apply_bake_transform(objects: list[bpy.types.Object], scale: Vector, rot_quat_wxyz: list[float], translation: Vector) -> None:
    quat = Quaternion((rot_quat_wxyz[0], rot_quat_wxyz[1], rot_quat_wxyz[2], rot_quat_wxyz[3]))
    quat_length = math.sqrt(quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z)
    if math.isclose(quat_length, 0.0, abs_tol=1e-12):
        raise ValueError("--rot-quat must not be a zero quaternion.")
    if not math.isclose(quat_length, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        quat.normalize()

    transform = Matrix.Translation(translation) @ quat.to_matrix().to_4x4() @ Matrix.Diagonal((scale.x, scale.y, scale.z, 1.0))
    for obj in objects:
        obj.data.transform(transform)
        obj.data.update()
        obj.matrix_world.identity()


def _bbox(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    mins = Vector((math.inf, math.inf, math.inf))
    maxs = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for vertex in obj.data.vertices:
            co = obj.matrix_world @ vertex.co
            mins.x = min(mins.x, co.x)
            mins.y = min(mins.y, co.y)
            mins.z = min(mins.z, co.z)
            maxs.x = max(maxs.x, co.x)
            maxs.y = max(maxs.y, co.y)
            maxs.z = max(maxs.z, co.z)
    if math.isinf(mins.z):
        raise RuntimeError("No vertices found in imported meshes.")
    return mins, maxs


def _shift_vertices(objects: list[bpy.types.Object], shift: Vector) -> None:
    if shift.length_squared == 0.0:
        return
    for obj in objects:
        for vertex in obj.data.vertices:
            vertex.co += shift
        obj.data.update()


def _format_vec(vec: Vector) -> str:
    return f"({vec.x:.9g}, {vec.y:.9g}, {vec.z:.9g})"


def main() -> None:
    args = parse_args()
    scale = _scale_vector(args.scale)
    translation = Vector(args.translation)

    for input_path in args.inputs:
        if not os.path.isfile(input_path):
            raise FileNotFoundError(input_path)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    imported_objects: list[bpy.types.Object] = []
    for input_path in args.inputs:
        imported_objects.extend(_import_obj(os.path.abspath(input_path)))

    _apply_bake_transform(imported_objects, scale, args.rot_quat, translation)
    before_align_min, before_align_max = _bbox(imported_objects)

    shift = Vector((0.0, 0.0, 0.0))
    if args.origin_mode == "center_xy":
        shift.x = -0.5 * (before_align_min.x + before_align_max.x)
        shift.y = -0.5 * (before_align_min.y + before_align_max.y)
    if not args.no_floor_align:
        shift.z = args.floor_to_z - before_align_min.z
    _shift_vertices(imported_objects, shift)

    after_min, after_max = _bbox(imported_objects)
    _export_obj(args.output, imported_objects)

    print(f"Baked OBJ: {Path(args.output).resolve()}")
    print("Input OBJs:")
    for input_path in args.inputs:
        print(f"  {Path(input_path).resolve()}")
    print(f"Applied scale: {_format_vec(scale)}")
    print(f"Applied rotation WXYZ: ({args.rot_quat[0]:.9g}, {args.rot_quat[1]:.9g}, {args.rot_quat[2]:.9g}, {args.rot_quat[3]:.9g})")
    print(f"Applied translation before alignment: {_format_vec(translation)}")
    print(f"Alignment shift: {_format_vec(shift)}")
    print(f"Bounds before alignment: min={_format_vec(before_align_min)} max={_format_vec(before_align_max)}")
    print(f"Bounds after alignment: min={_format_vec(after_min)} max={_format_vec(after_max)}")
    print(f"Objects: {len(imported_objects)}")
    print(f"Vertices: {sum(len(obj.data.vertices) for obj in imported_objects)}")
    print(f"Faces: {sum(len(obj.data.polygons) for obj in imported_objects)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
