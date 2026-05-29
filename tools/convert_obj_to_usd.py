#!/usr/bin/env python3

"""Convert an OBJ mesh to USD with Isaac Lab."""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Convert an OBJ mesh to USD using Isaac Lab's MeshConverter.")
parser.add_argument("input", type=str, help="Path to the input OBJ file.")
parser.add_argument("output", type=str, nargs="?", help="Path to the output USD file. Defaults to INPUT.usd.")
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=False,
    help="Make the converted geometry instanceable.",
)
parser.add_argument(
    "--collision-approx",
    choices=[
        "disabled",
        "triangle-mesh",
        "mesh-simplification",
        "convex-hull",
        "convex-decomposition",
        "bounding-cube",
    ],
    default="disabled",
    help=(
        "Collision approximation to author into the USD. "
        "Use mesh-simplification for coarse static scene collision. Default: disabled."
    ),
)
parser.add_argument(
    "--collision-simplification-metric",
    type=float,
    default=None,
    help="Optional PhysX mesh-simplification accuracy for --collision-approx mesh-simplification.",
)
parser.add_argument(
    "--unlit-textures",
    action="store_true",
    default=False,
    help="Display diffuse textures through OmniPBR emission so the mesh is not shaded by scene lights.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg  # noqa: E402
from isaaclab.sim.schemas import schemas_cfg  # noqa: E402
from isaaclab.utils.assets import check_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdShade  # noqa: E402


def _resolve_output_path(input_path: str, output_path: str | None) -> str:
    if output_path:
        return os.path.abspath(output_path)
    return os.path.abspath(str(Path(input_path).with_suffix(".usd")))


def _mesh_collision_props() -> schemas_cfg.MeshCollisionPropertiesCfg | None:
    match args_cli.collision_approx:
        case "disabled":
            return None
        case "triangle-mesh":
            return schemas_cfg.TriangleMeshPropertiesCfg()
        case "mesh-simplification":
            return schemas_cfg.TriangleMeshSimplificationPropertiesCfg(
                simplification_metric=args_cli.collision_simplification_metric
            )
        case "convex-hull":
            return schemas_cfg.ConvexHullPropertiesCfg()
        case "convex-decomposition":
            return schemas_cfg.ConvexDecompositionPropertiesCfg()
        case "bounding-cube":
            return schemas_cfg.BoundingCubePropertiesCfg()
        case _:
            raise ValueError(f"Unsupported collision approximation: {args_cli.collision_approx}")


def _set_input(shader: UsdShade.Shader, name: str, value_type: Sdf.ValueTypeName, value) -> None:
    shader.CreateInput(name, value_type).Set(value)


def _make_textured_materials_unlit(usd_path: str) -> int:
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    patched = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        shader = UsdShade.Shader(prim)
        source_asset = shader.GetPrim().GetAttribute("info:mdl:sourceAsset").Get()
        sub_identifier = shader.GetPrim().GetAttribute("info:mdl:sourceAsset:subIdentifier").Get()
        if str(source_asset) != "@OmniPBR.mdl@" and sub_identifier != "OmniPBR":
            continue

        diffuse_texture_input = shader.GetInput("diffuse_texture")
        if diffuse_texture_input is None:
            continue
        diffuse_texture = diffuse_texture_input.Get()
        if not diffuse_texture:
            continue

        _set_input(shader, "enable_emission", Sdf.ValueTypeNames.Bool, True)
        _set_input(shader, "emissive_color", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(1.0, 1.0, 1.0))
        _set_input(shader, "emissive_color_texture", Sdf.ValueTypeNames.Asset, diffuse_texture)
        _set_input(shader, "emissive_intensity", Sdf.ValueTypeNames.Float, 1.0)

        # Keep the diffuse texture as a visible color base. Emission supplies a
        # light-independent baseline; zeroing albedo can make OmniPBR appear gray
        # or black under some viewport/exposure settings.
        _set_input(shader, "diffuse_color_constant", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(1.0, 1.0, 1.0))
        _set_input(shader, "albedo_brightness", Sdf.ValueTypeNames.Float, 1.0)
        _set_input(shader, "albedo_desaturation", Sdf.ValueTypeNames.Float, 0.0)
        _set_input(shader, "metallic_constant", Sdf.ValueTypeNames.Float, 0.0)
        _set_input(shader, "reflection_roughness_constant", Sdf.ValueTypeNames.Float, 1.0)
        patched += 1

    stage.GetRootLayer().Save()
    return patched


def main() -> None:
    mesh_path = os.path.abspath(args_cli.input)
    if not check_file_path(mesh_path):
        raise ValueError(f"Invalid OBJ file path: {mesh_path}")

    dest_path = _resolve_output_path(mesh_path, args_cli.output)
    collision_enabled = args_cli.collision_approx != "disabled"
    mesh_collision_props = _mesh_collision_props()

    mesh_converter_cfg = MeshConverterCfg(
        asset_path=mesh_path,
        force_usd_conversion=True,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        make_instanceable=args_cli.make_instanceable,
        collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=collision_enabled),
        mesh_collision_props=mesh_collision_props,
    )

    print("-" * 80)
    print(f"Input OBJ file: {mesh_path}")
    print("Mesh converter config:")
    print_dict(mesh_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)

    mesh_converter = MeshConverter(mesh_converter_cfg)

    if args_cli.unlit_textures:
        patched = _make_textured_materials_unlit(mesh_converter.usd_path)
        print(f"Unlit textured materials patched: {patched}")

    print("Mesh converter output:")
    print(f"Generated USD file: {mesh_converter.usd_path}")
    print(f"Collision approximation: {args_cli.collision_approx}")
    print("-" * 80)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
