#!/usr/bin/env python3

"""Open a USD stage from an Isaac Sim --exec startup script."""

from __future__ import annotations

import sys
from pathlib import Path

import omni.usd
from pxr import Usd, UsdGeom, UsdLux


def _stage_bbox(stage: Usd.Stage):
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        return None
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
    )
    return cache.ComputeWorldBound(default_prim).ComputeAlignedBox()


def _add_viewer_lights(stage: Usd.Stage, bbox) -> None:
    dome = UsdLux.DomeLight.Define(stage, "/ViewerDomeLight")
    dome.CreateIntensityAttr().Set(1200.0)
    dome.CreateColorAttr().Set((1.0, 1.0, 1.0))

    center = bbox.GetMidpoint()
    size = bbox.GetSize()
    sphere = UsdLux.SphereLight.Define(stage, "/ViewerKeyLight")
    sphere.CreateIntensityAttr().Set(6000.0)
    sphere.CreateRadiusAttr().Set(2.0)
    sphere.CreateColorAttr().Set((1.0, 0.96, 0.9))
    xform = UsdGeom.Xformable(sphere)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set((center[0], center[1] - max(size[1], 2.0), center[2] + max(size[2], 2.0)))


def _stage_has_light(stage: Usd.Stage) -> bool:
    return any("Light" in prim.GetTypeName() for prim in stage.Traverse())


def _frame_viewport(stage: Usd.Stage, bbox) -> None:
    try:
        import numpy as np
        from isaacsim.core.utils.viewports import set_camera_view
    except Exception as exc:
        print(f"[open-usd] viewport framing unavailable: {exc}", flush=True)
        return

    center = bbox.GetMidpoint()
    size = bbox.GetSize()
    distance = max(size[0], size[1], size[2], 1.0) * 1.4
    eye = np.array([center[0] + distance, center[1] - distance, center[2] + distance * 0.55])
    target = np.array([center[0], center[1], center[2] * 0.55])
    set_camera_view(eye=eye, target=target)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: open_usd_stage.py /path/to/scene.usd")

    usd_path = Path(sys.argv[1]).expanduser().resolve()
    if not usd_path.is_file():
        raise SystemExit(f"USD file not found: {usd_path}")

    result = omni.usd.get_context().open_stage(str(usd_path))
    stage = omni.usd.get_context().get_stage()
    bbox = _stage_bbox(stage) if stage is not None else None
    if bbox is not None:
        if not _stage_has_light(stage):
            _add_viewer_lights(stage, bbox)
        _frame_viewport(stage, bbox)
        print(
            "[open-usd] bbox "
            f"min={tuple(round(v, 3) for v in bbox.GetMin())} "
            f"max={tuple(round(v, 3) for v in bbox.GetMax())}",
            flush=True,
        )
    print(f"[open-usd] opened={result} path={usd_path}", flush=True)


main()
