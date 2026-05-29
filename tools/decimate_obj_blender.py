#!/usr/bin/env python3

"""Decimate an OBJ mesh with Blender.

Run with:
  blender --background --python decimate_obj_blender.py -- INPUT.obj OUTPUT.obj --ratio 0.3
  blender --background --python decimate_obj_blender.py -- INPUT.obj OUTPUT.obj --target-faces 10000
"""

import argparse
import math
import os
import sys

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []

    parser = argparse.ArgumentParser(description="Decimate an OBJ mesh with Blender.")
    parser.add_argument("input", type=str, help="Input OBJ path.")
    parser.add_argument("output", type=str, help="Output OBJ path.")
    parser.add_argument("--ratio", type=float, default=0.3, help="Collapse decimation ratio in (0, 1].")
    parser.add_argument(
        "--planar-angle-deg",
        type=float,
        default=0.0,
        help="Dissolve near-coplanar faces before collapse. Default 0 disables it because it can damage UV atlases.",
    )
    parser.add_argument(
        "--no-preserve-boundaries",
        action="store_true",
        default=False,
        help="Allow collapse decimation across UV/material/seam/sharp/normal boundaries for more aggressive simplification.",
    )
    parser.add_argument("--target-faces", type=int, default=None, help="Maximum target face count.")
    return parser.parse_args(argv)


def import_obj(path: str) -> None:
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=path)
    else:
        bpy.ops.import_scene.obj(filepath=path)


def export_obj(path: str) -> None:
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(filepath=path, export_materials=True)
    else:
        bpy.ops.export_scene.obj(
            filepath=path,
            check_existing=False,
            axis_forward="Y",
            axis_up="Z",
            global_scale=1,
            path_mode="RELATIVE",
        )


def mesh_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def polygon_count(objects: list[bpy.types.Object]) -> int:
    return sum(len(obj.data.polygons) for obj in objects)


def apply_planar_dissolve(objects: list[bpy.types.Object], angle_deg: float) -> None:
    if angle_deg <= 0.0:
        return

    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        modifier = obj.modifiers.new(name="codex_planar_dissolve", type="DECIMATE")
        modifier.decimate_type = "DISSOLVE"
        modifier.angle_limit = math.radians(angle_deg)
        modifier.use_dissolve_boundaries = False
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        obj.select_set(False)


def apply_collapse_decimate(objects: list[bpy.types.Object], ratio: float, preserve_boundaries: bool) -> None:
    if ratio >= 1.0:
        return

    for obj in objects:
        if len(obj.data.polygons) <= 3:
            continue
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        modifier = obj.modifiers.new(name="codex_collapse_decimate", type="DECIMATE")
        modifier.ratio = ratio
        if preserve_boundaries:
            modifier.delimit = {"UV", "MATERIAL", "SEAM", "SHARP", "NORMAL"}
        modifier.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        obj.select_set(False)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.ratio <= 1.0:
        raise ValueError(f"--ratio must be in (0, 1], got {args.ratio}")
    if args.planar_angle_deg < 0.0:
        raise ValueError(f"--planar-angle-deg must be >= 0, got {args.planar_angle_deg}")
    if args.target_faces is not None and args.target_faces < 4:
        raise ValueError(f"--target-faces must be >= 4, got {args.target_faces}")
    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_obj(os.path.abspath(args.input))

    objects = mesh_objects()
    if not objects:
        raise RuntimeError(f"No mesh objects imported from {args.input}")

    before = polygon_count(objects)
    apply_planar_dissolve(objects, args.planar_angle_deg)
    after_planar = polygon_count(objects)

    target_ratio = args.ratio
    if args.target_faces is not None:
        target_ratio = min(target_ratio, args.target_faces / after_planar)

    preserve_boundaries = not args.no_preserve_boundaries
    apply_collapse_decimate(objects, target_ratio, preserve_boundaries)

    # Blender's decimator can land slightly above the target. Tighten iteratively.
    if args.target_faces is not None:
        for _ in range(4):
            current = polygon_count(objects)
            if current <= args.target_faces:
                break
            apply_collapse_decimate(objects, max(0.001, args.target_faces / current * 0.98), preserve_boundaries)

    after = polygon_count(objects)
    export_obj(os.path.abspath(args.output))

    print(f"Decimated OBJ: {args.input} -> {args.output}")
    print(
        f"Faces: {before} -> {after_planar} after planar dissolve"
        f" -> {after} after collapse (ratio={after / before:.3f})"
    )
    print(f"Preserve UV/material/seam/sharp/normal boundaries: {preserve_boundaries}")
    if args.target_faces is not None:
        print(f"Target faces: <= {args.target_faces}")


if __name__ == "__main__":
    main()
