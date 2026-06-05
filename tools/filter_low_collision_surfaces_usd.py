#!/usr/bin/env python3

"""Remove low, broad, near-horizontal faces from OR collision USD meshes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom


def _vec3(value) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _triangle_area_and_normal(points: list[tuple[float, float, float]]) -> tuple[float, tuple[float, float, float]]:
    if len(points) < 3:
        return 0.0, (0.0, 0.0, 0.0)

    p0 = points[0]
    normal = [0.0, 0.0, 0.0]
    area = 0.0
    for pa, pb in zip(points[1:-1], points[2:]):
        ux, uy, uz = pa[0] - p0[0], pa[1] - p0[1], pa[2] - p0[2]
        vx, vy, vz = pb[0] - p0[0], pb[1] - p0[1], pb[2] - p0[2]
        cx = uy * vz - uz * vy
        cy = uz * vx - ux * vz
        cz = ux * vy - uy * vx
        length = math.sqrt(cx * cx + cy * cy + cz * cz)
        area += 0.5 * length
        normal[0] += cx
        normal[1] += cy
        normal[2] += cz

    normal_length = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    if normal_length <= 1e-12:
        return area, (0.0, 0.0, 0.0)
    return area, (normal[0] / normal_length, normal[1] / normal_length, normal[2] / normal_length)


def _remove_geom_subsets(mesh_prim: Usd.Prim) -> int:
    stage = mesh_prim.GetStage()
    removed = 0
    for child in list(mesh_prim.GetChildren()):
        if child.GetTypeName() == "GeomSubset":
            stage.RemovePrim(child.GetPath())
            removed += 1
    return removed


def _update_extent(mesh: UsdGeom.Mesh, points) -> None:
    extent = UsdGeom.PointBased.ComputeExtent(points)
    mesh.GetExtentAttr().Set(extent)


def _filter_mesh(mesh_prim: Usd.Prim, args: argparse.Namespace) -> tuple[int, int]:
    mesh = UsdGeom.Mesh(mesh_prim)
    points = [_vec3(point) for point in (mesh.GetPointsAttr().Get() or [])]
    counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    if not points or not counts or not indices:
        return 0, 0

    normal_z_min = math.cos(math.radians(args.max_slope_deg))
    new_counts: list[int] = []
    new_indices: list[int] = []
    removed = 0
    cursor = 0
    for count in counts:
        face_indices = indices[cursor : cursor + count]
        cursor += count
        face_points = [points[index] for index in face_indices]
        xs = [point[0] for point in face_points]
        ys = [point[1] for point in face_points]
        zs = [point[2] for point in face_points]
        z_min, z_max = min(zs), max(zs)
        xy_span = max(max(xs) - min(xs), max(ys) - min(ys))
        area, normal = _triangle_area_and_normal(face_points)

        is_low_surface = args.min_z <= z_min and z_max <= args.max_z
        is_broad = area >= args.min_area or xy_span >= args.min_xy_span
        is_near_horizontal = abs(normal[2]) >= normal_z_min
        should_remove = is_low_surface and is_broad and is_near_horizontal

        if should_remove:
            removed += 1
            continue
        new_counts.append(count)
        new_indices.extend(face_indices)

    if removed and not args.dry_run:
        mesh.GetFaceVertexCountsAttr().Set(new_counts)
        mesh.GetFaceVertexIndicesAttr().Set(new_indices)
        _update_extent(mesh, points)
        _remove_geom_subsets(mesh_prim)
    return removed, len(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd_path", type=Path)
    parser.add_argument("--collision-root-name", default="lowpoly_collision")
    parser.add_argument("--min-z", type=float, default=0.04)
    parser.add_argument("--max-z", type=float, default=0.20)
    parser.add_argument("--min-area", type=float, default=0.04)
    parser.add_argument("--min-xy-span", type=float, default=0.60)
    parser.add_argument("--max-slope-deg", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    usd_path = args.usd_path.expanduser().resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    total_removed = 0
    total_faces = 0
    mesh_count = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        path = str(prim.GetPath())
        if args.collision_root_name and args.collision_root_name not in path:
            schemas = set(prim.GetAppliedSchemas())
            if "PhysicsCollisionAPI" not in schemas:
                continue
        removed, faces = _filter_mesh(prim, args)
        if faces:
            mesh_count += 1
            total_removed += removed
            total_faces += faces
            print(f"{prim.GetPath()}: removed {removed} / {faces} faces")

    if not args.dry_run:
        stage.GetRootLayer().Save()

    print(f"Filtered USD: {usd_path}")
    print(f"Meshes processed: {mesh_count}")
    print(f"Faces removed: {total_removed} / {total_faces}")
    print(
        "Filter: "
        f"z=[{args.min_z}, {args.max_z}], min_area={args.min_area}, "
        f"min_xy_span={args.min_xy_span}, max_slope_deg={args.max_slope_deg}"
    )


if __name__ == "__main__":
    main()
