#!/usr/bin/env python3

"""Attach a hidden low-poly collision USD to a visual USD stage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom


def _relative_asset_path(asset_path: Path, anchor_file: Path) -> str:
    try:
        return os.path.relpath(asset_path, start=anchor_file.parent)
    except ValueError:
        return str(asset_path)


def _remove_collision_from_visual_meshes(stage: Usd.Stage, collision_root_path: Sdf.Path) -> int:
    removed = 0
    for prim in stage.Traverse():
        if prim.GetPath().HasPrefix(collision_root_path):
            continue
        if prim.GetTypeName() != "Mesh":
            continue
        for schema in ("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"):
            if schema in prim.GetAppliedSchemas():
                prim.RemoveAPI(schema)
                removed += 1
        for attr_name in ("physics:approximation", "physics:collisionEnabled"):
            attr = prim.GetAttribute(attr_name)
            if attr:
                prim.RemoveProperty(attr_name)
    return removed


def _flatten_stage_to_file(stage: Usd.Stage, output_path: Path) -> None:
    flattened_layer = stage.Flatten(addSourceFileComment=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    flattened_layer.Export(str(tmp_path))
    os.replace(tmp_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("visual_usd", type=Path, help="USD stage containing the visual mesh.")
    parser.add_argument("collision_usd", type=Path, help="USD stage containing the low-poly collision mesh.")
    parser.add_argument(
        "--flatten",
        action="store_true",
        default=False,
        help="Flatten the collision reference into visual_usd so the final USD has no external collision USD dependency.",
    )
    parser.add_argument(
        "--prim-name",
        default="lowpoly_collision",
        help="Child prim name under the visual stage default prim. Default: lowpoly_collision.",
    )
    args = parser.parse_args()

    visual_usd = args.visual_usd.expanduser().resolve()
    collision_usd = args.collision_usd.expanduser().resolve()
    if not visual_usd.is_file():
        raise FileNotFoundError(visual_usd)
    if not collision_usd.is_file():
        raise FileNotFoundError(collision_usd)

    stage = Usd.Stage.Open(str(visual_usd))
    if stage is None:
        raise RuntimeError(f"Failed to open visual USD: {visual_usd}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError(f"Visual USD has no default prim: {visual_usd}")

    collision_root_path = default_prim.GetPath().AppendChild(args.prim_name)
    removed = _remove_collision_from_visual_meshes(stage, collision_root_path)

    collision_root = stage.DefinePrim(collision_root_path, "Xform")
    collision_root.GetReferences().ClearReferences()
    collision_root.GetReferences().AddReference(_relative_asset_path(collision_usd, visual_usd))

    imageable = UsdGeom.Imageable(collision_root)
    imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)

    stage.GetRootLayer().Save()
    if args.flatten:
        _flatten_stage_to_file(stage, visual_usd)

    print(f"Attached hidden low-poly collision USD: {collision_usd}")
    print(f"Visual USD: {visual_usd}")
    print(f"Collision prim: {collision_root_path}")
    print(f"Flattened into visual USD: {args.flatten}")
    print(f"Removed visual collision API/property entries: {removed}")


if __name__ == "__main__":
    main()
