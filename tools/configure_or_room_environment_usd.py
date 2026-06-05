#!/usr/bin/env python3

"""Add environment-level lights to an OR room USD."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux


def _visible_bbox(stage: Usd.Stage):
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("USD stage has no default prim")
    bbox_root = default_prim.GetChild("geometry")
    if not bbox_root:
        bbox_root = default_prim
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
    )
    return cache.ComputeWorldBound(bbox_root).ComputeAlignedBox()


def _set_xform_translate(prim: Usd.Prim, xyz: tuple[float, float, float]) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*xyz))


def _delete_if_exists(stage: Usd.Stage, path: Sdf.Path) -> None:
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)


def _define_dome_light(stage: Usd.Stage, path: Sdf.Path, intensity: float) -> None:
    light = UsdLux.DomeLight.Define(stage, path)
    light.CreateIntensityAttr().Set(float(intensity))
    light.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    if hasattr(light, "CreateVisibleInPrimaryRayAttr"):
        light.CreateVisibleInPrimaryRayAttr().Set(False)


def _define_key_light(stage: Usd.Stage, path: Sdf.Path, bbox, intensity: float, radius: float) -> None:
    center = bbox.GetMidpoint()
    size = bbox.GetSize()
    pos = (
        float(center[0]),
        float(center[1] - max(size[1] * 0.35, 1.2)),
        float(center[2] + max(size[2] * 0.35, 1.2)),
    )
    light = UsdLux.SphereLight.Define(stage, path)
    light.CreateIntensityAttr().Set(float(intensity))
    light.CreateRadiusAttr().Set(float(radius))
    light.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.96, 0.90))
    _set_xform_translate(light.GetPrim(), pos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd_path", type=Path)
    parser.add_argument("--dome-intensity", type=float, default=800.0)
    parser.add_argument("--key-intensity", type=float, default=3500.0)
    parser.add_argument("--key-radius", type=float, default=1.2)
    args = parser.parse_args()

    usd_path = args.usd_path.expanduser().resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError(f"USD stage has no default prim: {usd_path}")

    bbox = _visible_bbox(stage)
    env_path = default_prim.GetPath().AppendChild("Environment")
    stage.DefinePrim(env_path, "Xform")

    _define_dome_light(stage, env_path.AppendChild("AmbientDomeLight"), args.dome_intensity)
    _define_key_light(stage, env_path.AppendChild("KeySphereLight"), bbox, args.key_intensity, args.key_radius)
    _delete_if_exists(stage, env_path.AppendChild("PreviewCamera"))

    stage.GetRootLayer().Save()
    print(f"Configured OR environment USD: {usd_path}")
    print(f"Environment prim: {env_path}")
    print(f"Visible bbox min={tuple(round(v, 3) for v in bbox.GetMin())} max={tuple(round(v, 3) for v in bbox.GetMax())}")


if __name__ == "__main__":
    main()
