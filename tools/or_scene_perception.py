#!/usr/bin/env python3

"""Perception skill: fixed-camera semantic detections registered to world frame."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.visual_vlm_servo import (
    DEFAULT_MODEL_PATH,
    detect_target,
    enhance_vlm_image,
    first_json_object,
    load_detector,
    load_live_frame,
    normalize_box,
    split_csv,
)

SCENE_CAMERA_NAMES = {
    "halo": (
        "Camera_40795974",
        "Camera_40034694",
        "Camera_46026258",
        "Camera_47661457",
        "PoV_Camera_0",
        "PoV_Camera_1",
        "PoV_Camera_2",
        "PoV_Camera_3",
        "PoV_Camera_4",
        "iso",
        "RobotCam",
    ),
    "pulm": (
        "Camera_45902703L",
        "Camera_46517772L",
        "Camera_44664489L",
        "Camera_41908851L",
        "optimized",
    ),
}


def _float_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _camera_prim_name(scene: str, image_name: str) -> str | None:
    if not image_name.startswith("scene_"):
        return None
    try:
        index = int(image_name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None
    names = SCENE_CAMERA_NAMES.get(scene, ())
    if 0 <= index < len(names):
        return names[index]
    return None


def _load_camera(stage: Usd.Stage, scene: str, image_name: str):
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        return None
    prim_name = _camera_prim_name(scene, image_name)
    if prim_name is None:
        return None
    path = default_prim.GetPath().AppendPath("Cameras/Blender").AppendChild(prim_name)
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return None
    return UsdGeom.Camera(prim)


def _label_key(label: str) -> str:
    return " ".join(label.lower().replace("_", " ").split())


def _target_group_key(instruction: str) -> str:
    # All primary detections in this process answer the same task query. Grouping
    # them by the user instruction is more robust than trusting VLM wording to be
    # byte-identical across camera views.
    return "__target__:" + _label_key(instruction)


def _bbox_center_ray(
    camera: UsdGeom.Camera,
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    u = (x1 + x2) * 0.5
    v = (y1 + y2) * 0.5

    focal = float(camera.GetFocalLengthAttr().Get() or 1.0)
    horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get() or 20.0)
    vertical_aperture = float(camera.GetVerticalApertureAttr().Get() or 20.0)

    # USD camera convention: local -Z is forward, +Y is up.
    sensor_x = (u / max(width, 1) - 0.5) * horizontal_aperture
    sensor_y = (0.5 - v / max(height, 1)) * vertical_aperture
    local_dir = Gf.Vec3d(sensor_x / focal, sensor_y / focal, -1.0).GetNormalized()

    xformable = UsdGeom.Xformable(camera.GetPrim())
    matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    origin = matrix.ExtractTranslation()
    world_dir = matrix.TransformDir(local_dir).GetNormalized()
    return (
        (float(origin[0]), float(origin[1]), float(origin[2])),
        (float(world_dir[0]), float(world_dir[1]), float(world_dir[2])),
    )


def _triangulate_rays(
    rays: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> tuple[tuple[float, float, float], float] | None:
    """Least-squares closest point for calibrated camera rays."""
    if len(rays) < 2:
        return None
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    normalized_rays: list[tuple[np.ndarray, np.ndarray]] = []
    for origin_tuple, direction_tuple in rays:
        origin = np.asarray(origin_tuple, dtype=np.float64)
        direction = np.asarray(direction_tuple, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm < 1e-8:
            continue
        direction = direction / norm
        projector = eye - np.outer(direction, direction)
        a += projector
        b += projector @ origin
        normalized_rays.append((origin, direction))
    if len(normalized_rays) < 2:
        return None
    try:
        point = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        point = np.linalg.lstsq(a, b, rcond=None)[0]
    if not np.all(np.isfinite(point)):
        return None
    errors = []
    for origin, direction in normalized_rays:
        offset = point - origin
        closest = origin + direction * float(np.dot(offset, direction))
        errors.append(float(np.linalg.norm(point - closest)))
    mean_error = float(np.mean(errors)) if errors else math.inf
    if not np.isfinite(mean_error):
        return None
    return (float(point[0]), float(point[1]), float(point[2])), mean_error


def _raw_box_from_value(value) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            return [float(value[key]) for key in ("x1", "y1", "x2", "y2")]
        if all(key in value for key in ("left", "top", "right", "bottom")):
            return [float(value[key]) for key in ("left", "top", "right", "bottom")]
        if all(key in value for key in ("x", "y", "width", "height")):
            x = float(value["x"])
            y = float(value["y"])
            return [x, y, x + float(value["width"]), y + float(value["height"])]
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return [float(item) for item in value[:4]]
    return None


def _obstacle_inventory_prompt(
    instruction: str,
    image_size: tuple[int, int],
    camera_name: str,
    max_obstacles: int,
    obstacle_hint: str,
) -> str:
    hint = obstacle_hint.strip()
    hint_text = f"Optional user/domain hint: {hint!r}. " if hint else ""
    return (
        "You are the perception module for safe robot navigation. "
        f"Camera/view id: {camera_name}. Image size: {image_size[0]}x{image_size[1]} pixels. "
        f"Robot task: {instruction!r}. {hint_text}"
        "Identify visible objects or structures that the robot should avoid while navigating. "
        "Do not use a fixed class list; decide the category from the image. "
        "For each item, estimate its physical ground footprint radius in meters from visual evidence and object type. "
        "Return ONLY one JSON object with schema: "
        '{"obstacles":[{"label":"object name","box":[x1,y1,x2,y2],"confidence":0.0,'
        '"is_obstacle":true,"is_robot":false,"dynamic":false,'
        '"footprint_radius_m":0.0,"size_reason":"short reason"}]}. '
        f"Return at most {max_obstacles} items. "
        "Use pixel coordinates in the original image. If no obstacle is visible, return {\"obstacles\":[]}."
    )


def _qwen_obstacle_inventory(
    processor,
    model,
    image,
    *,
    instruction: str,
    camera_name: str,
    max_obstacles: int,
    obstacle_hint: str,
    max_new_tokens: int,
) -> str:
    import torch

    prompt = _obstacle_inventory_prompt(instruction, image.size, camera_name, max_obstacles, obstacle_hint)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
    return processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def _payload_bool(payload: dict, key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _payload_float(payload: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


MIN_OBSTACLE_FOOTPRINT_RADIUS_M = 0.15
MAX_OBSTACLE_FOOTPRINT_RADIUS_M = 1.25
UNKNOWN_OBSTACLE_FOOTPRINT_RADIUS_M = 0.35


def _clamp_footprint_radius(radius: float) -> float:
    return max(MIN_OBSTACLE_FOOTPRINT_RADIUS_M, min(MAX_OBSTACLE_FOOTPRINT_RADIUS_M, radius))


def _footprint_radius_from_payload(payload: dict) -> tuple[float | None, str]:
    radius = _payload_float(payload, ("footprint_radius_m", "radius_m", "estimated_radius_m"))
    if radius is not None and radius > 0.0:
        return _clamp_footprint_radius(radius), "vlm_estimate"
    width = _payload_float(payload, ("footprint_width_m", "width_m", "diameter_m"))
    depth = _payload_float(payload, ("footprint_depth_m", "depth_m", "length_m"))
    values = [value for value in (width, depth) if value is not None and value > 0.0]
    if values:
        return _clamp_footprint_radius(max(values) * 0.5), "vlm_dimensions"
    return None, "missing"


def _footprint_radius_from_bbox(box: tuple[float, float, float, float], image) -> float:
    width = max(float(getattr(image, "width", 0) or 0), 1.0)
    height = max(float(getattr(image, "height", 0) or 0), 1.0)
    x1, y1, x2, y2 = box
    box_fraction = max(0.0, min(1.0, max((x2 - x1) / width, (y2 - y1) / height)))
    radius = UNKNOWN_OBSTACLE_FOOTPRINT_RADIUS_M + 0.45 * box_fraction
    return _clamp_footprint_radius(radius)


def _obstacles_from_inventory(raw_text: str, image, threshold: float, max_obstacles: int) -> list[dict]:
    payload = first_json_object(raw_text)
    if payload is None:
        return []
    raw_items = payload.get("obstacles", payload.get("objects", payload.get("items", [])))
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        return []
    obstacles: list[dict] = []
    for raw_item in raw_items[: max(0, max_obstacles)]:
        if not isinstance(raw_item, dict):
            continue
        raw_box = None
        for key in ("box", "bbox", "bbox_2d", "bounding_box", "coordinates"):
            raw_box = _raw_box_from_value(raw_item.get(key))
            if raw_box is not None:
                break
        if raw_box is None:
            continue
        box = normalize_box(raw_box, image)
        if box is None:
            continue
        confidence = _payload_float(raw_item, ("confidence", "score"))
        if confidence is None:
            confidence = 1.0
        if confidence > 1.0:
            confidence /= 100.0
        if confidence < threshold:
            continue
        radius, footprint_source = _footprint_radius_from_payload(raw_item)
        size_reason = str(raw_item.get("size_reason") or raw_item.get("reason") or "")[:240]
        if radius is None or radius <= 0.0:
            radius = _footprint_radius_from_bbox(box, image)
            footprint_source = "bbox_safety_fallback"
            if not size_reason:
                size_reason = "VLM omitted footprint radius; parser used conservative bbox-based safety estimate."
        label = str(raw_item.get("label") or raw_item.get("name") or raw_item.get("category") or "obstacle").strip()
        if not label:
            label = "obstacle"
        is_robot = _payload_bool(raw_item, "is_robot", False)
        is_obstacle = _payload_bool(raw_item, "is_obstacle", not is_robot)
        obstacles.append(
            {
                "label": label,
                "confidence": max(0.0, min(1.0, confidence)),
                "box": box,
                "is_obstacle": is_obstacle,
                "is_robot": is_robot,
                "dynamic": _payload_bool(raw_item, "dynamic", False) and not is_robot,
                "footprint_radius_m": radius,
                "footprint_source": footprint_source,
                "size_reason": size_reason,
            }
        )
    return obstacles


def _triangulate_item_groups(
    groups: dict[str, list[tuple[dict, tuple[tuple[float, float, float], tuple[float, float, float]]]]],
    args: argparse.Namespace,
    *,
    pose_source: str,
) -> None:
    for _label, ray_items in groups.items():
        triangulated = _triangulate_rays([ray for _, ray in ray_items])
        if triangulated is None:
            continue
        world_xyz, error_m = triangulated
        if not (
            args.world_bounds[0] <= world_xyz[0] <= args.world_bounds[1]
            and args.world_bounds[2] <= world_xyz[1] <= args.world_bounds[3]
            and args.world_bounds[4] <= world_xyz[2] <= args.world_bounds[5]
        ):
            continue
        if error_m > args.max_triangulation_error:
            continue
        for item, _ray in ray_items:
            item["world_xyz"] = world_xyz
            item["world_xy"] = (world_xyz[0], world_xyz[1])
            item["triangulation_error_m"] = error_m
            item["world_pose_source"] = pose_source


def _wait_for_any_camera(camera_names: list[str], timeout_s: float) -> list[str]:
    from tools.shared_memory_utils import MultiImageReader

    deadline = time.monotonic() + timeout_s
    available: list[str] = []
    reader = MultiImageReader()
    try:
        while time.monotonic() < deadline:
            for name in camera_names:
                if name in available:
                    continue
                if reader.read_single_image(name) is not None:
                    available.append(name)
            if available:
                return available
            time.sleep(0.1)
    finally:
        reader.close()
    return available


def run_perception(args: argparse.Namespace) -> dict:
    print(
        f"[or-perception] start instruction={args.instruction!r} scene={args.scene} device={args.device}",
        file=sys.stderr,
        flush=True,
    )
    stage = Usd.Stage.Open(str(args.room_usd))
    if stage is None:
        raise RuntimeError(f"failed to open room USD: {args.room_usd}")

    print(f"[or-perception] loading model={args.model_path}", file=sys.stderr, flush=True)
    processor, model = load_detector(args.model_path, args.device)
    env_cameras = split_csv(args.env_cameras)[: max(0, args.max_env_cameras)]
    head_cameras = split_csv(args.head_cameras)
    requested_cameras = env_cameras + head_cameras
    print(f"[or-perception] waiting cameras={requested_cameras}", file=sys.stderr, flush=True)
    cameras = _wait_for_any_camera(env_cameras + head_cameras, args.camera_timeout)
    print(f"[or-perception] live cameras={cameras}", file=sys.stderr, flush=True)
    objects: list[dict] = []
    obstacle_objects: list[dict] = []
    head_visible = False
    head_detections: list[dict] = []
    obstacle_hint = args.obstacle_queries.strip()
    target_group_key = _target_group_key(args.instruction)
    target_ray_groups: dict[
        str,
        list[tuple[dict, tuple[tuple[float, float, float], tuple[float, float, float]]]],
    ] = {}
    obstacle_ray_groups: dict[
        str,
        list[tuple[dict, tuple[tuple[float, float, float], tuple[float, float, float]]]],
    ] = {}

    for camera_name in cameras:
        image = None
        try:
            image = load_live_frame(camera_name, args.camera_timeout)
            target_t0 = time.monotonic()
            print(f"[or-perception] target inference camera={camera_name}", file=sys.stderr, flush=True)
            detection, raw_text = detect_target(
                processor,
                model,
                image,
                args.instruction,
                camera_name,
                args.score_threshold,
                args.max_new_tokens,
                args.vlm_image_scale,
                args.vlm_sharpness,
                args.vlm_contrast,
            )
            target_elapsed = time.monotonic() - target_t0
        except Exception as exc:
            print(f"[or-perception] camera={camera_name} failed: {exc}", file=sys.stderr, flush=True)
            continue
        if detection is None:
            detection = None
            print(
                f"[or-perception] target camera={camera_name} not found elapsed={target_elapsed:.2f}s",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[or-perception] target camera={camera_name} label={detection.label} "
                f"score={detection.score:.2f} box={tuple(round(float(v), 1) for v in detection.box)} "
                f"elapsed={target_elapsed:.2f}s",
                file=sys.stderr,
                flush=True,
            )

        camera = _load_camera(stage, args.scene, camera_name) if camera_name.startswith("scene_") else None

        if detection is not None:
            item = {
                "object_id": f"{detection.label}_{camera_name}".replace(" ", "_"),
                "label": detection.label,
                "confidence": detection.score,
                "seen_by": [camera_name],
                "bbox_by_view": {camera_name: tuple(float(v) for v in detection.box)},
                "world_xy": None,
                "head_visible": camera_name.startswith("head"),
                "reachable": True,
                "last_seen_s": time.monotonic(),
                "raw_vlm": raw_text[:500],
                "is_obstacle": False,
            }
            if camera:
                ray = _bbox_center_ray(camera, detection.box, image.size)
                if ray is not None:
                    item["world_ray"] = {
                        "origin": ray[0],
                        "direction": ray[1],
                    }
                    target_ray_groups.setdefault(target_group_key, []).append((item, ray))
            else:
                head_visible = True
                head_detections.append(item)
            objects.append(item)

        if (
            not camera_name.startswith("scene_")
            or camera is None
            or image is None
            or args.max_obstacle_detections <= 0
        ):
            continue
        try:
            camera_index = int(camera_name.split("_", 1)[1])
        except (IndexError, ValueError):
            camera_index = 0
        if camera_index >= args.max_obstacle_cameras:
            continue

        try:
            obstacle_t0 = time.monotonic()
            print(f"[or-perception] obstacle inference camera={camera_name}", file=sys.stderr, flush=True)
            enhanced = enhance_vlm_image(
                image,
                scale=args.vlm_image_scale,
                sharpness=args.vlm_sharpness,
                contrast=args.vlm_contrast,
            )
            obstacle_raw = _qwen_obstacle_inventory(
                processor,
                model,
                enhanced,
                instruction=args.instruction,
                camera_name=camera_name,
                max_obstacles=args.max_obstacle_detections,
                obstacle_hint=obstacle_hint,
                max_new_tokens=args.obstacle_max_new_tokens,
            )
            obstacle_detections = _obstacles_from_inventory(
                obstacle_raw,
                enhanced,
                args.score_threshold,
                args.max_obstacle_detections,
            )
            print(
                f"[or-perception] obstacles camera={camera_name} count={len(obstacle_detections)} "
                f"elapsed={time.monotonic() - obstacle_t0:.2f}s",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            print(
                f"[or-perception] obstacle inventory camera={camera_name} failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue

        sx = image.width / max(float(enhanced.width), 1.0)
        sy = image.height / max(float(enhanced.height), 1.0)
        label_counts: dict[str, int] = {}
        for obstacle in obstacle_detections:
            x1, y1, x2, y2 = obstacle["box"]
            box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
            label_key = _label_key(obstacle["label"])
            occurrence = label_counts.get(label_key, 0)
            label_counts[label_key] = occurrence + 1
            # Keep one ray per label occurrence per camera. This avoids a single
            # view contributing duplicate rays to the same triangulation group.
            group_key = f"{label_key}:{occurrence}"
            ray = _bbox_center_ray(camera, box, image.size)
            if ray is None:
                continue
            item = {
                "object_id": f"{obstacle['label']}_{camera_name}_{occurrence}".replace(" ", "_"),
                "label": obstacle["label"],
                "query": obstacle["label"],
                "confidence": obstacle["confidence"],
                "seen_by": [camera_name],
                "bbox_by_view": {camera_name: tuple(float(v) for v in box)},
                "world_xy": None,
                "head_visible": False,
                "reachable": False,
                "last_seen_s": time.monotonic(),
                "raw_vlm": obstacle_raw[:500],
                "is_obstacle": obstacle["is_obstacle"],
                "is_robot": obstacle["is_robot"],
                "dynamic": obstacle["dynamic"],
                "footprint_radius_m": obstacle["footprint_radius_m"],
                "footprint_source": obstacle["footprint_source"],
                "size_reason": obstacle["size_reason"],
                "world_ray": {
                    "origin": ray[0],
                    "direction": ray[1],
                },
            }
            obstacle_objects.append(item)
            obstacle_ray_groups.setdefault(group_key, []).append((item, ray))

    _triangulate_item_groups(target_ray_groups, args, pose_source="multi_view_triangulation")
    _triangulate_item_groups(obstacle_ray_groups, args, pose_source="multi_view_obstacle_triangulation")
    print(
        f"[or-perception] done objects={len(objects)} obstacles={len(obstacle_objects)} "
        f"head_visible={head_visible}",
        file=sys.stderr,
        flush=True,
    )

    return {
        "instruction": args.instruction,
        "scene": args.scene,
        "objects": objects,
        "obstacle_objects": obstacle_objects,
        "head_visible": head_visible,
        "head_detections": head_detections,
        "timestamp_s": time.monotonic(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction")
    parser.add_argument("--scene", choices=["halo", "pulm"], default="halo")
    parser.add_argument("--room-usd", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--env-cameras", default=",".join(f"scene_{index:02d}" for index in range(11)))
    parser.add_argument("--head-cameras", default="head")
    parser.add_argument("--max-env-cameras", type=int, default=6)
    parser.add_argument(
        "--obstacle-queries",
        default="",
        help="Optional natural-language hint for obstacle inventory; categories are inferred by VLM.",
    )
    parser.add_argument("--max-obstacle-detections", type=int, default=7)
    parser.add_argument("--max-obstacle-cameras", type=int, default=4)
    parser.add_argument("--camera-timeout", type=float, default=5.0)
    parser.add_argument("--score-threshold", type=float, default=0.04)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--vlm-image-scale", type=float, default=_float_from_env("UNITREE_VLM_IMAGE_SCALE", 1.25))
    parser.add_argument("--vlm-sharpness", type=float, default=_float_from_env("UNITREE_VLM_SHARPNESS", 1.25))
    parser.add_argument("--vlm-contrast", type=float, default=_float_from_env("UNITREE_VLM_CONTRAST", 1.05))
    parser.add_argument("--obstacle-max-new-tokens", type=int, default=384)
    parser.add_argument("--max-triangulation-error", type=float, default=0.75)
    parser.add_argument(
        "--world-bounds",
        type=float,
        nargs=6,
        default=(-4.0, 4.0, -4.0, 4.0, -0.2, 2.5),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                args.device = "cpu"
        except Exception:
            args.device = "cpu"
    result = run_perception(args)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
