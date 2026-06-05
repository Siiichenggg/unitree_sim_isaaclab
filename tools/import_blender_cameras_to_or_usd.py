#!/usr/bin/env python3

"""Import camera poses from a source Blender scene into a baked OR USD.

Run with Blender:
  blender --background source.blend --python tools/import_blender_cameras_to_or_usd.py -- \
      output.usd input_a.obj input_b.obj --scale 4.8 --rot-quat W X Y Z
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import traceback
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector
from pxr import Gf, Sdf, Usd, UsdGeom


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd_path", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path, help="Raw room OBJ(s) used by the bake step.")
    parser.add_argument("--scale", nargs="+", type=float, default=[1.0])
    parser.add_argument("--rot-quat", nargs=4, type=float, metavar=("W", "X", "Y", "Z"), default=[1.0, 0.0, 0.0, 0.0])
    parser.add_argument("--translation", nargs=3, type=float, metavar=("X", "Y", "Z"), default=[0.0, 0.0, 0.0])
    parser.add_argument("--floor-to-z", type=float, default=0.0)
    parser.add_argument("--no-floor-align", action="store_true", default=False)
    parser.add_argument("--origin-mode", choices=["keep_xy", "center_xy"], default="center_xy")
    parser.add_argument(
        "--source-root-object",
        default="",
        help="Blender object whose local coordinates match the raw OBJ inputs used by the bake step.",
    )
    return parser.parse_args(argv)


def _scale_vector(values: list[float]) -> Vector:
    if len(values) == 1:
        return Vector((values[0], values[0], values[0]))
    if len(values) == 3:
        return Vector(values)
    raise ValueError("--scale expects either one value or three XYZ values.")


def _import_obj(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.context.scene.objects)
    kwargs = {
        "filepath": str(path),
        "forward_axis": "Y",
        "up_axis": "Z",
        "use_split_objects": True,
        "use_split_groups": False,
    }
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(**kwargs)
    else:
        bpy.ops.import_scene.obj(filepath=str(path), axis_forward="Y", axis_up="Z")
    return [obj for obj in bpy.context.scene.objects if obj not in before and obj.type == "MESH"]


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
        raise RuntimeError("No vertices found while computing room bake alignment.")
    return mins, maxs


def _alignment_shift(objects: list[bpy.types.Object], origin_mode: str, floor_to_z: float, floor_align: bool) -> Vector:
    before_min, before_max = _bbox(objects)
    shift = Vector((0.0, 0.0, 0.0))
    if origin_mode == "center_xy":
        shift.x = -0.5 * (before_min.x + before_max.x)
        shift.y = -0.5 * (before_min.y + before_max.y)
    if floor_align:
        shift.z = floor_to_z - before_min.z
    return shift


def _safe_identifier(name: str, used: set[str]) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not identifier or identifier[0].isdigit():
        identifier = f"Camera_{identifier}"
    base = identifier
    suffix = 1
    while identifier in used:
        suffix += 1
        identifier = f"{base}_{suffix}"
    used.add(identifier)
    return identifier


def _camera_data(obj: bpy.types.Object, source_to_baked_matrix: Matrix) -> dict:
    matrix = source_to_baked_matrix @ obj.matrix_world
    loc, quat, _scale = matrix.decompose()
    cam = obj.data
    return {
        "name": obj.name,
        "matrix": matrix,
        "loc": loc,
        "quat": quat,
        "focal_length": float(cam.lens),
        "horizontal_aperture": float(cam.sensor_width),
        "vertical_aperture": float(cam.sensor_height),
        "clipping_range": (float(cam.clip_start), float(cam.clip_end)),
    }


def _write_camera(stage: Usd.Stage, path: Sdf.Path, data: dict) -> None:
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr().Set(data["focal_length"])
    camera.CreateHorizontalApertureAttr().Set(data["horizontal_aperture"])
    camera.CreateVerticalApertureAttr().Set(data["vertical_aperture"])
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(*data["clipping_range"]))

    loc = data["loc"]
    quat = data["quat"]
    xformable = UsdGeom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(loc.x, loc.y, loc.z))
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(quat.w, quat.x, quat.y, quat.z))
    xformable.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _delete_if_exists(stage: Usd.Stage, path: Sdf.Path) -> None:
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)


def main() -> None:
    args = parse_args()
    usd_path = args.usd_path.expanduser().resolve()
    input_paths = [path.expanduser().resolve() for path in args.inputs]
    for input_path in input_paths:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

    source_cameras = [obj for obj in bpy.context.scene.objects if obj.type == "CAMERA"]
    if not source_cameras:
        raise RuntimeError(f"No cameras found in Blender scene: {bpy.data.filepath}")
    source_camera_matrices = {obj.name: obj.matrix_world.copy() for obj in source_cameras}
    source_root_matrix = Matrix.Identity(4)
    if args.source_root_object:
        source_root = bpy.data.objects.get(args.source_root_object)
        if source_root is None:
            raise RuntimeError(f"Source root object not found in Blender scene: {args.source_root_object}")
        source_root_matrix = source_root.matrix_world.copy()

    scale = _scale_vector(args.scale)
    quat = Quaternion(tuple(args.rot_quat))
    quat_length = math.sqrt(quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z)
    if math.isclose(quat_length, 0.0, abs_tol=1e-12):
        raise ValueError("--rot-quat must not be a zero quaternion.")
    if not math.isclose(quat_length, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        quat.normalize()
    translation = Vector(args.translation)
    bake_matrix = Matrix.Translation(translation) @ quat.to_matrix().to_4x4() @ Matrix.Diagonal((scale.x, scale.y, scale.z, 1.0))

    imported_objects: list[bpy.types.Object] = []
    for input_path in input_paths:
        imported_objects.extend(_import_obj(input_path))
    for obj in imported_objects:
        obj.data.transform(bake_matrix)
        obj.data.update()
        obj.matrix_world.identity()
    shift = _alignment_shift(imported_objects, args.origin_mode, args.floor_to_z, not args.no_floor_align)
    source_to_baked_matrix = Matrix.Translation(shift) @ bake_matrix @ source_root_matrix.inverted()

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError(f"USD stage has no default prim: {usd_path}")

    cameras_path = default_prim.GetPath().AppendChild("Cameras")
    blender_cameras_path = cameras_path.AppendChild("Blender")
    stage.DefinePrim(cameras_path, "Xform")
    _delete_if_exists(stage, blender_cameras_path)
    stage.DefinePrim(blender_cameras_path, "Xform")

    used_names: set[str] = set()
    camera_data_by_name: dict[str, dict] = {}
    for obj in source_cameras:
        obj.matrix_world = source_camera_matrices[obj.name]
        data = _camera_data(obj, source_to_baked_matrix)
        camera_data_by_name[obj.name] = data
        prim_name = _safe_identifier(obj.name, used_names)
        _write_camera(stage, blender_cameras_path.AppendChild(prim_name), data)

    _delete_if_exists(stage, cameras_path.AppendChild("EnvironmentCamera"))
    _delete_if_exists(stage, cameras_path.AppendChild("RobotFrontCamera"))

    stage.GetRootLayer().Save()
    print(f"Imported Blender cameras into: {usd_path}")
    print(f"Source blend: {bpy.data.filepath}")
    print(f"Camera root: {blender_cameras_path}")
    print(f"Cameras imported: {len(source_cameras)}")
    print(f"Alignment shift: ({shift.x:.9g}, {shift.y:.9g}, {shift.z:.9g})")
    print(f"Source root object: {args.source_root_object or '<scene world>'}")
    for obj in source_cameras:
        data = camera_data_by_name[obj.name]
        loc = data["loc"]
        print(f"  {obj.name}: ({loc.x:.6f}, {loc.y:.6f}, {loc.z:.6f})")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
