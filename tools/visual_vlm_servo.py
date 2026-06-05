#!/usr/bin/env python3

"""Camera VLM target detection and visual-servo navigation.

This script reads live Isaac camera frames from DDS, uses Qwen-VL to ground a
natural-language target in the image, and optionally publishes Unitree
wholebody run commands to visually servo toward that target.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "vlm" / "qwen3-vl-2b-instruct"
DEFAULT_DEBUG_IMAGE = PROJECT_ROOT / "logs" / "or_vlm_nav" / "visual_grounding_debug.jpg"
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
ACTION_SET = {"move_forward", "turn_left", "turn_right", "search_left", "search_right", "stop", "wait"}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    @property
    def width(self) -> float:
        x1, _, x2, _ = self.box
        return max(0.0, x2 - x1)

    @property
    def height(self) -> float:
        _, y1, _, y2 = self.box
        return max(0.0, y2 - y1)


@dataclass
class CameraFrame:
    name: str
    image: Image.Image
    role: str
    metadata: str


@dataclass
class PlannerDecision:
    action: str
    target_visible: bool
    best_view: str
    reason: str
    confidence: float


def split_csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _float_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def enhance_vlm_image(
    image: Image.Image,
    *,
    scale: float = 1.0,
    sharpness: float = 1.0,
    contrast: float = 1.0,
) -> Image.Image:
    scale = max(0.25, min(float(scale), 4.0))
    sharpness = max(0.0, min(float(sharpness), 4.0))
    contrast = max(0.25, min(float(contrast), 4.0))
    enhanced = image.convert("RGB")
    if abs(scale - 1.0) > 1e-3:
        size = (max(1, int(round(enhanced.width * scale))), max(1, int(round(enhanced.height * scale))))
        enhanced = enhanced.resize(size, Image.Resampling.LANCZOS)
    if abs(contrast - 1.0) > 1e-3:
        enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast)
    if abs(sharpness - 1.0) > 1e-3:
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness)
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=3))
    return enhanced


def live_camera_sequence(args: argparse.Namespace) -> list[str]:
    requested = split_csv(args.search_cameras)
    if args.camera and args.camera not in requested:
        requested.append(args.camera)
    if not requested:
        requested = [args.camera]
    return requested


def load_live_frame(camera: str, timeout_s: float, reader=None) -> Image.Image:
    from tools.shared_memory_utils import MultiImageReader

    owns_reader = reader is None
    if reader is None:
        reader = MultiImageReader()
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame_bgr = reader.read_single_image(camera)
            if frame_bgr is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
            time.sleep(0.05)
        raise RuntimeError(
            f"No live camera frame received for camera={camera!r}. "
            "Start the simulator with cameras enabled and rendering on; visual mode does not work reliably with --no-render."
        )
    finally:
        if owns_reader:
            reader.close()


def infer_scene_from_room_usd(room_usd: Path | None) -> str:
    if room_usd:
        lowered = str(room_usd).lower()
        if "pulm" in lowered:
            return "pulm"
    return "halo"


def load_room_camera_metadata(room_usd: Path | None) -> dict[str, str]:
    """Return concise camera pose strings keyed by shared-memory image names."""
    if room_usd is None or not room_usd.is_file():
        return {}
    scene = infer_scene_from_room_usd(room_usd)
    camera_names = SCENE_CAMERA_NAMES.get(scene, ())
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(room_usd))
        if stage is None:
            return {}
        default_prim = stage.GetDefaultPrim()
        if not default_prim:
            return {}
        camera_root = default_prim.GetPath().AppendPath("Cameras/Blender")
        metadata: dict[str, str] = {}
        for index, blender_name in enumerate(camera_names):
            prim = stage.GetPrimAtPath(camera_root.AppendChild(blender_name))
            if not prim:
                continue
            xformable = UsdGeom.Xformable(prim)
            matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            pos = matrix.ExtractTranslation()
            metadata[f"scene_{index:02d}"] = (
                f"fixed OR room camera {blender_name}, "
                f"world_pos=({float(pos[0]):.2f},{float(pos[1]):.2f},{float(pos[2]):.2f})"
            )
        return metadata
    except Exception as exc:
        print(f"[visual-vlm] failed to read room camera metadata: {exc}", flush=True)
        return {}


def camera_role(camera_name: str) -> str:
    if camera_name.startswith("scene_"):
        return "fixed_room_context"
    if camera_name.startswith("head"):
        return "robot_egocentric_servo"
    if camera_name in {"left", "right"}:
        return "robot_wrist_context"
    return "unknown"


def frame_from_bgr(camera_name: str, frame_bgr, camera_metadata: dict[str, str]) -> CameraFrame | None:
    if frame_bgr is None:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    metadata = camera_metadata.get(camera_name, "")
    if not metadata:
        metadata = f"view={camera_name}, role={camera_role(camera_name)}"
    return CameraFrame(
        name=camera_name,
        image=Image.fromarray(frame_rgb),
        role=camera_role(camera_name),
        metadata=metadata,
    )


def load_current_frames(
    camera_names: list[str],
    reader,
    camera_metadata: dict[str, str],
    max_frames: int,
) -> list[CameraFrame]:
    frames: list[CameraFrame] = []
    for camera_name in camera_names:
        if len(frames) >= max_frames:
            break
        frame = frame_from_bgr(camera_name, reader.read_single_image(camera_name), camera_metadata)
        if frame is not None:
            frames.append(frame)
    return frames


def select_planner_frames(
    frames: list[CameraFrame],
    servo_cameras: set[str],
    max_frames: int,
    context_offset: int,
) -> tuple[list[CameraFrame], int]:
    ego_frames = [frame for frame in frames if frame.name in servo_cameras]
    context_frames = [frame for frame in frames if frame.name not in servo_cameras]
    if max_frames <= 0:
        return frames, context_offset
    selected = ego_frames[:max_frames]
    remaining = max(0, max_frames - len(selected))
    if remaining and context_frames:
        for step in range(min(remaining, len(context_frames))):
            selected.append(context_frames[(context_offset + step) % len(context_frames)])
        context_offset = (context_offset + remaining) % len(context_frames)
    return selected, context_offset


def format_robot_pose(pose: tuple[float, float, float] | None, age: float) -> str:
    if pose is None:
        return "robot_pose=unknown"
    x, y, yaw = pose
    return f"robot_pose=(x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad, age={age:.2f}s)"


def planner_history_text(history: deque[dict]) -> str:
    if not history:
        return "history=[]"
    items = []
    for item in list(history)[-5:]:
        items.append(
            f"{item['action']} via {item.get('best_view', '')} "
            f"visible={item.get('target_visible', False)} conf={item.get('confidence', 0.0):.2f}"
        )
    return "history=[" + "; ".join(items) + "]"


def high_level_prompt(
    instruction: str,
    frames: list[CameraFrame],
    robot_pose_text: str,
    history: deque[dict],
) -> str:
    view_lines = [
        f"{index}. {frame.name}: role={frame.role}; {frame.metadata}"
        for index, frame in enumerate(frames, start=1)
    ]
    return (
        "You are a lightweight navigation planner for a humanoid robot in an operating room. "
        "Use the labeled views to decide the next high-level action. "
        "Fixed room cameras provide global context only; their image coordinates are not robot steering coordinates. "
        "Robot egocentric head cameras can be used for direct servoing. "
        f"Instruction: {instruction!r}. "
        f"{robot_pose_text}. "
        f"{planner_history_text(history)}. "
        "Views:\n" + "\n".join(view_lines) + "\n"
        "Return ONLY one JSON object with this schema: "
        '{"action":"move_forward|turn_left|turn_right|search_left|search_right|stop|wait",'
        '"target_visible":true,"best_view":"view_name","confidence":0.0,"reason":"short reason"}. '
        "Choose stop only if the robot appears close enough to the target from an egocentric view. "
        "If the target is visible only in fixed room cameras, choose a search/turn action that should help bring it into the robot head view."
    )


def qwen_plan(
    processor,
    model,
    frames: list[CameraFrame],
    instruction: str,
    robot_pose_text: str,
    history: deque[dict],
    max_new_tokens: int,
) -> str:
    content = []
    for frame in frames:
        content.append({"type": "text", "text": f"View {frame.name}: role={frame.role}; {frame.metadata}"})
        content.append({"type": "image", "image": frame.image})
    content.append({"type": "text", "text": high_level_prompt(instruction, frames, robot_pose_text, history)})
    messages = [{"role": "user", "content": content}]
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


def planner_decision_from_qwen(raw_text: str) -> PlannerDecision | None:
    payload = first_json_object(raw_text)
    if payload is None:
        action_match = re.search(r'"?action"?\s*:\s*"?([a-z_]+)"?', raw_text, flags=re.IGNORECASE)
        if action_match is None:
            return None
        payload = {
            "action": action_match.group(1),
            "target_visible": bool(re.search(r'"?target_visible"?\s*:\s*true', raw_text, flags=re.IGNORECASE)),
            "best_view": "",
            "confidence": 0.0,
            "reason": raw_text[:160],
        }
        best_view_match = re.search(r'"?best_view"?\s*:\s*"([^"]+)"', raw_text, flags=re.IGNORECASE)
        if best_view_match:
            payload["best_view"] = best_view_match.group(1)
        confidence_match = re.search(r'"?confidence"?\s*:\s*([0-9.]+)', raw_text, flags=re.IGNORECASE)
        if confidence_match:
            payload["confidence"] = confidence_match.group(1)
    action = str(payload.get("action", "")).strip().lower()
    if action not in ACTION_SET:
        return None
    confidence = payload.get("confidence", 0.0)
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        score = 0.0
    if score > 1.0:
        score /= 100.0
    return PlannerDecision(
        action=action,
        target_visible=bool(payload.get("target_visible", False)),
        best_view=str(payload.get("best_view", "")),
        reason=str(payload.get("reason", "")),
        confidence=max(0.0, min(1.0, score)),
    )


def command_from_planner_action(args: argparse.Namespace, action: str, search_direction: float) -> tuple[float, float, float]:
    if action == "move_forward":
        return args.min_speed, 0.0, 0.0
    if action in {"turn_left", "search_left"}:
        return 0.0, 0.0, abs(args.search_yaw_rate)
    if action in {"turn_right", "search_right"}:
        return 0.0, 0.0, -abs(args.search_yaw_rate)
    if action == "wait":
        return 0.0, 0.0, 0.0
    if action == "stop":
        return 0.0, 0.0, 0.0
    return 0.0, 0.0, abs(args.search_yaw_rate) * search_direction


def load_detector(model_path: Path, device: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    try:
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), local_files_only=True, dtype=dtype)
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), local_files_only=True, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return processor, model


def grounding_prompt(instruction: str, image: Image.Image, camera_name: str) -> str:
    return (
        "You are the vision module for a robot. Look only at the provided image. "
        f"Camera/view id: {camera_name}. "
        f"The image size is {image.width}x{image.height} pixels. "
        f"Robot command: {instruction!r}. "
        "Find the visible target object or place implied by the command. "
        "Return ONLY one JSON object with this schema: "
        '{"found": true, "label": "target name", "box": [x1, y1, x2, y2], "confidence": 0.0}. '
        "Use pixel coordinates in the original image, origin at top-left. "
        "If the target is not visible, return exactly: "
        '{"found": false, "label": "", "box": null, "confidence": 0.0}.'
    )


def qwen_ground(processor, model, image: Image.Image, instruction: str, camera_name: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": grounding_prompt(instruction, image, camera_name)},
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


def first_json_object(text: str) -> dict | None:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def raw_box_from_value(value) -> list[float] | None:
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
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return raw_box_from_value(value[0])
        if len(value) >= 4:
            return [float(item) for item in value[:4]]
    return None


def box_from_payload(payload: dict) -> list[float] | None:
    for key in ("box", "bbox", "bbox_2d", "bounding_box", "coordinates"):
        box = raw_box_from_value(payload.get(key))
        if box is not None:
            return box
    return None


def normalize_box(box: list[float], image: Image.Image) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    max_value = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_value <= 1.5:
        x1, x2 = x1 * image.width, x2 * image.width
        y1, y2 = y1 * image.height, y2 * image.height
    elif max_value <= 1000.0:
        x1, x2 = x1 * image.width / 1000.0, x2 * image.width / 1000.0
        y1, y2 = y1 * image.height / 1000.0, y2 * image.height / 1000.0
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(float(image.width - 1), x1))
    y1 = max(0.0, min(float(image.height - 1), y1))
    x2 = max(0.0, min(float(image.width), x2))
    y2 = max(0.0, min(float(image.height), y2))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return x1, y1, x2, y2


def detection_from_qwen(raw_text: str, image: Image.Image, threshold: float) -> Detection | None:
    payload = first_json_object(raw_text)
    if payload is None:
        return None
    if payload.get("found") is False:
        return None
    box = box_from_payload(payload)
    if box is None:
        return None
    normalized = normalize_box(box, image)
    if normalized is None:
        return None
    confidence = payload.get("confidence", payload.get("score", 1.0))
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        score = 1.0
    if score > 1.0:
        score /= 100.0
    if score < threshold:
        return None
    return Detection(label=str(payload.get("label") or "target"), score=score, box=normalized)


def detect_target(
    processor,
    model,
    image: Image.Image,
    instruction: str,
    camera_name: str,
    threshold: float,
    max_new_tokens: int,
    image_scale: float = 1.0,
    sharpness: float = 1.0,
    contrast: float = 1.0,
) -> tuple[Detection | None, str]:
    original_size = image.size
    enhanced = enhance_vlm_image(image, scale=image_scale, sharpness=sharpness, contrast=contrast)
    raw_text = qwen_ground(processor, model, enhanced, instruction, camera_name, max_new_tokens)
    detection = detection_from_qwen(raw_text, enhanced, threshold)
    if detection is not None and enhanced.size != original_size:
        sx = original_size[0] / max(float(enhanced.width), 1.0)
        sy = original_size[1] / max(float(enhanced.height), 1.0)
        x1, y1, x2, y2 = detection.box
        detection = Detection(
            label=detection.label,
            score=detection.score,
            box=(x1 * sx, y1 * sy, x2 * sx, y2 * sy),
        )
    return detection, raw_text


def draw_debug(image: Image.Image, detection: Detection | None, instruction: str, path: Path) -> None:
    debug = image.copy()
    draw = ImageDraw.Draw(debug)
    if detection is not None:
        x1, y1, x2, y2 = detection.box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
        draw.text((x1 + 4, max(0, y1 - 18)), f"{detection.label} {detection.score:.2f}", fill=(255, 0, 0))
    draw.text((8, 8), "instruction: " + instruction, fill=(255, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    debug.save(path)


def command_from_detection(args: argparse.Namespace, detection: Detection, image_size: tuple[int, int]) -> tuple[float, float, float, bool]:
    width, height = image_size
    cx, _ = detection.center
    center_error = (cx - width * 0.5) / max(width * 0.5, 1.0)
    area_ratio = (detection.width * detection.height) / max(float(width * height), 1.0)
    height_ratio = detection.height / max(float(height), 1.0)
    centered = abs(center_error) <= args.center_tolerance
    close_enough = area_ratio >= args.stop_area_ratio or height_ratio >= args.stop_height_ratio
    # For navigation, the terminal condition is reaching the object vicinity.
    # Requiring image centering as well can make a close side-view target keep
    # rotating until the time budget expires.
    done = close_enough or (centered and area_ratio >= args.stop_area_ratio * 0.75)

    if done:
        return 0.0, 0.0, 0.0, True

    if centered:
        yaw_rate = 0.0
    else:
        yaw_rate = max(-args.max_yaw_rate, min(args.max_yaw_rate, -args.yaw_kp * center_error))
    slow_area_ratio = max(args.slow_area_ratio, args.stop_area_ratio, 1e-6)
    scale = max(0.0, 1.0 - area_ratio / slow_area_ratio)
    if abs(center_error) > args.forward_center_limit or abs(center_error) > args.hard_turn_center_limit:
        vx = 0.0
    else:
        forward_limit = max(args.forward_center_limit, 1e-6)
        alignment = max(0.0, 1.0 - abs(center_error) / forward_limit)
        if centered:
            alignment = 1.0
        if alignment < args.offcenter_forward_scale:
            vx = 0.0
        else:
            vx = args.max_speed * scale * alignment
            vx = max(args.min_speed, vx)
    return vx, 0.0, yaw_rate, False


def target_world_distance(args: argparse.Namespace, pose: tuple[float, float, float] | None) -> float | None:
    target_xy = getattr(args, "target_world_xy", None)
    if pose is None or target_xy is None:
        return None
    try:
        tx, ty = float(target_xy[0]), float(target_xy[1])
        return math.hypot(tx - float(pose[0]), ty - float(pose[1]))
    except (TypeError, ValueError, IndexError):
        return None


def reached_target_world_radius(args: argparse.Namespace, pose_subscriber) -> tuple[bool, float | None]:
    if pose_subscriber is None or getattr(args, "target_arrival_radius", 0.0) <= 0.0:
        return False, None
    if pose_subscriber.age() > getattr(args, "target_pose_max_age", 2.0):
        return False, None
    distance = target_world_distance(args, pose_subscriber.latest())
    return distance is not None and distance <= args.target_arrival_radius, distance


class CommandStreamer:
    """Continuously publish the latest run command so DDS timeout does not zero it between VLM frames."""

    def __init__(self, publisher, height: float, hz: float):
        self._publisher = publisher
        self._height = float(height)
        self._period = 1.0 / max(float(hz), 1.0)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._command = (0.0, 0.0, 0.0, self._height)
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def update(self, vx: float, vy: float, yaw_rate: float, height: float | None = None) -> None:
        if height is None:
            height = self._height
        with self._lock:
            self._command = (float(vx), float(vy), float(yaw_rate), float(height))

    def stop_motion(self, height: float | None = None) -> None:
        self.update(0.0, 0.0, 0.0, self._height if height is None else height)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                vx, vy, yaw_rate, height = self._command
            self._publisher.publish(vx, vy, yaw_rate, height)
            self._stop_event.wait(self._period)

    def close(self) -> None:
        self.stop_motion()
        time.sleep(min(self._period, 0.05))
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._publisher.stop(self._height)


def execute_bbox_servo(args: argparse.Namespace) -> int:
    from tools.astar_nav_dds import RunCommandPublisher, SimPoseSubscriber
    from tools.shared_memory_utils import MultiImageReader

    processor, model = load_detector(args.model_path, args.device)
    publisher = RunCommandPublisher(channel_id=args.channel_id, dds_interface=args.dds_interface)
    pose_subscriber = None
    if args.target_world_xy is not None and args.target_arrival_radius > 0.0:
        try:
            pose_subscriber = SimPoseSubscriber(channel_id=args.channel_id, dds_interface=args.dds_interface)
        except Exception as exc:
            print(f"[visual-vlm] sim_state pose subscriber unavailable: {exc}", flush=True)
    streamer = CommandStreamer(publisher, height=args.height, hz=args.stream_hz)
    streamer.start()
    reader = MultiImageReader()
    camera_names = live_camera_sequence(args)
    servo_cameras = set(split_csv(args.servo_cameras))
    if not servo_cameras:
        servo_cameras = {args.camera}
    print(f"[visual-vlm] visual instruction={args.instruction}")
    print(f"[visual-vlm] search cameras={camera_names}")
    print(f"[visual-vlm] servo cameras={sorted(servo_cameras)}")
    print("[visual-vlm] mode=execute; no fixed XY map is used")

    deadline = time.monotonic() + args.duration
    last_debug = 0.0
    search_direction = -1.0
    next_search_switch = time.monotonic() + args.search_sweep_interval
    camera_index = 0
    try:
        while time.monotonic() < deadline:
            reached_world, target_distance = reached_target_world_radius(args, pose_subscriber)
            if reached_world:
                streamer.stop_motion(args.height)
                print(
                    f"[visual-vlm] reached target world radius "
                    f"distance={target_distance:.2f}m radius={args.target_arrival_radius:.2f}m",
                    flush=True,
                )
                return 0

            camera_name = camera_names[camera_index % len(camera_names)]
            try:
                image = load_live_frame(camera_name, args.camera_timeout, reader=reader)
            except RuntimeError as exc:
                streamer.stop_motion(args.height)
                print(f"[visual-vlm] {exc}", file=sys.stderr)
                return 2

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
            if detection is None:
                camera_index += 1
                now = time.monotonic()
                if now >= next_search_switch:
                    search_direction *= -1.0
                    next_search_switch = now + args.search_sweep_interval
                vx, vy, yaw_rate, done = 0.0, 0.0, abs(args.search_yaw_rate) * search_direction, False
                streamer.update(vx, vy, yaw_rate, args.height)
                print(f"[visual-vlm] camera={camera_name} target not visible; searching qwen={raw_text}", flush=True)
            elif camera_name not in servo_cameras:
                camera_index += 1
                now = time.monotonic()
                if now >= next_search_switch:
                    search_direction *= -1.0
                    next_search_switch = now + args.search_sweep_interval
                streamer.update(0.0, 0.0, abs(args.search_yaw_rate) * search_direction, args.height)
                cx, cy = detection.center
                area_ratio = detection.width * detection.height / max(float(image.width * image.height), 1.0)
                print(
                    f"[visual-vlm] context camera={camera_name} sees {detection.label} "
                    f"score={detection.score:.2f} center=({cx:.0f},{cy:.0f}) area={area_ratio:.3f}; "
                    "waiting for an ego camera before servoing",
                    flush=True,
                )
            else:
                search_direction = -1.0 if detection.center[0] >= image.width * 0.5 else 1.0
                next_search_switch = time.monotonic() + args.search_sweep_interval
                vx, vy, yaw_rate, done = command_from_detection(args, detection, image.size)
                cx, cy = detection.center
                area_ratio = detection.width * detection.height / max(float(image.width * image.height), 1.0)
                print(
                    f"[visual-vlm] camera={camera_name} score={detection.score:.2f} label={detection.label} "
                    f"center=({cx:.0f},{cy:.0f}) area={area_ratio:.3f} "
                    f"cmd=[{vx:.2f},{vy:.2f},{yaw_rate:.2f},{args.height:.2f}]",
                    flush=True,
                )
                if done:
                    streamer.stop_motion(args.height)
                    draw_debug(image, detection, args.instruction, args.debug_image)
                    print("[visual-vlm] reached visual stop condition")
                    return 0
                streamer.update(vx, vy, yaw_rate, args.height)

            now = time.monotonic()
            if now - last_debug >= args.debug_interval:
                draw_debug(image, detection, args.instruction, args.debug_image)
                last_debug = now
            time.sleep(max(0.01, 1.0 / args.command_hz))
    finally:
        streamer.close()
        reader.close()

    print("[visual-vlm] duration expired; stopped")
    return 1


def execute_lightweight_servo(args: argparse.Namespace) -> int:
    from tools.astar_nav_dds import RunCommandPublisher, SimPoseSubscriber
    from tools.shared_memory_utils import MultiImageReader

    processor, model = load_detector(args.model_path, args.device)
    publisher = RunCommandPublisher(channel_id=args.channel_id, dds_interface=args.dds_interface)
    pose_subscriber = None
    try:
        pose_subscriber = SimPoseSubscriber(channel_id=args.channel_id, dds_interface=args.dds_interface)
    except Exception as exc:
        print(f"[visual-vlm] sim_state pose subscriber unavailable: {exc}", flush=True)
    streamer = CommandStreamer(publisher, height=args.height, hz=args.stream_hz)
    streamer.start()
    reader = MultiImageReader()

    room_usd = args.room_usd.expanduser().resolve() if args.room_usd else None
    camera_metadata = load_room_camera_metadata(room_usd)
    camera_names = live_camera_sequence(args)
    servo_cameras = set(split_csv(args.servo_cameras)) or {args.camera}
    history: deque[dict] = deque(maxlen=max(1, args.history_size))
    print(f"[visual-vlm] visual instruction={args.instruction}")
    print(f"[visual-vlm] planner=lightweight")
    print(f"[visual-vlm] search cameras={camera_names}")
    print(f"[visual-vlm] servo cameras={sorted(servo_cameras)}")
    if room_usd:
        print(f"[visual-vlm] room_usd={room_usd}")

    deadline = time.monotonic() + args.duration
    last_debug = 0.0
    search_direction = -1.0
    next_search_switch = time.monotonic() + args.search_sweep_interval
    context_offset = 0
    try:
        while time.monotonic() < deadline:
            all_frames = load_current_frames(camera_names, reader, camera_metadata, len(camera_names))
            if not all_frames:
                streamer.stop_motion(args.height)
                print("[visual-vlm] no live camera frames available", file=sys.stderr)
                return 2
            frames, context_offset = select_planner_frames(all_frames, servo_cameras, args.max_planner_images, context_offset)

            ego_detection = None
            ego_image = None
            ego_camera = ""
            ego_raw = ""
            for frame in frames:
                if frame.name not in servo_cameras:
                    continue
                detection, raw_text = detect_target(
                    processor,
                    model,
                    frame.image,
                    args.instruction,
                    frame.name,
                    args.score_threshold,
                    args.max_new_tokens,
                    args.vlm_image_scale,
                    args.vlm_sharpness,
                    args.vlm_contrast,
                )
                if detection is not None:
                    ego_detection = detection
                    ego_image = frame.image
                    ego_camera = frame.name
                    ego_raw = raw_text
                    break
                ego_raw = raw_text

            if ego_detection is not None and ego_image is not None:
                search_direction = -1.0 if ego_detection.center[0] >= ego_image.width * 0.5 else 1.0
                next_search_switch = time.monotonic() + args.search_sweep_interval
                vx, vy, yaw_rate, done = command_from_detection(args, ego_detection, ego_image.size)
                cx, cy = ego_detection.center
                area_ratio = ego_detection.width * ego_detection.height / max(float(ego_image.width * ego_image.height), 1.0)
                history.append(
                    {
                        "action": "ego_servo",
                        "target_visible": True,
                        "best_view": ego_camera,
                        "confidence": ego_detection.score,
                    }
                )
                print(
                    f"[visual-vlm] ego camera={ego_camera} score={ego_detection.score:.2f} label={ego_detection.label} "
                    f"center=({cx:.0f},{cy:.0f}) area={area_ratio:.3f} "
                    f"cmd=[{vx:.2f},{vy:.2f},{yaw_rate:.2f},{args.height:.2f}]",
                    flush=True,
                )
                if done:
                    streamer.stop_motion(args.height)
                    draw_debug(ego_image, ego_detection, args.instruction, args.debug_image)
                    print("[visual-vlm] reached visual stop condition")
                    return 0
                streamer.update(vx, vy, yaw_rate, args.height)
                if time.monotonic() - last_debug >= args.debug_interval:
                    draw_debug(ego_image, ego_detection, args.instruction, args.debug_image)
                    last_debug = time.monotonic()
            else:
                pose = pose_subscriber.latest() if pose_subscriber is not None else None
                pose_age = pose_subscriber.age() if pose_subscriber is not None else float("inf")
                robot_pose_text = format_robot_pose(pose, pose_age)
                try:
                    raw_plan = qwen_plan(
                        processor,
                        model,
                        frames,
                        args.instruction,
                        robot_pose_text,
                        history,
                        args.planner_max_new_tokens,
                    )
                    decision = planner_decision_from_qwen(raw_plan)
                except Exception as exc:
                    raw_plan = f"planner error: {exc}"
                    decision = None

                now = time.monotonic()
                if now >= next_search_switch:
                    search_direction *= -1.0
                    next_search_switch = now + args.search_sweep_interval

                if decision is None:
                    action = "search_left" if search_direction > 0.0 else "search_right"
                    vx, vy, yaw_rate = command_from_planner_action(args, action, search_direction)
                    streamer.update(vx, vy, yaw_rate, args.height)
                    history.append({"action": action, "target_visible": False, "best_view": "", "confidence": 0.0})
                    print(f"[visual-vlm] planner invalid; fallback {action}; raw={raw_plan}", flush=True)
                else:
                    vx, vy, yaw_rate = command_from_planner_action(args, decision.action, search_direction)
                    streamer.update(vx, vy, yaw_rate, args.height)
                    history.append(
                        {
                            "action": decision.action,
                            "target_visible": decision.target_visible,
                            "best_view": decision.best_view,
                            "confidence": decision.confidence,
                        }
                    )
                    print(
                        f"[visual-vlm] planner action={decision.action} visible={decision.target_visible} "
                        f"best_view={decision.best_view} conf={decision.confidence:.2f} "
                        f"cmd=[{vx:.2f},{vy:.2f},{yaw_rate:.2f},{args.height:.2f}] reason={decision.reason}",
                        flush=True,
                    )
                    if decision.action == "stop":
                        streamer.stop_motion(args.height)
                        return 0

            time.sleep(max(0.01, 1.0 / args.command_hz))
    finally:
        streamer.close()
        reader.close()

    print("[visual-vlm] duration expired; stopped")
    return 1


def execute_servo(args: argparse.Namespace) -> int:
    if args.planner_mode == "bbox":
        return execute_bbox_servo(args)
    return execute_lightweight_servo(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use live camera visual grounding to walk toward an object.")
    parser.add_argument("instruction", help="Target instruction, e.g. 'walk to the cylinder'.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--camera", default="head")
    parser.add_argument("--search-cameras", default="head", help="Comma-separated live camera sweep order.")
    parser.add_argument("--planner-mode", choices=["lightweight", "bbox"], default="lightweight")
    parser.add_argument(
        "--servo-cameras",
        default="head,left,right",
        help="Comma-separated cameras whose bbox coordinates may be used for motion commands.",
    )
    parser.add_argument("--room-usd", type=Path, default=None, help="Current OR room USD, used to label fixed camera poses.")
    parser.add_argument("--camera-discovery-timeout", type=float, default=8.0, help="Seconds to wait for requested DDS camera frames to appear.")
    parser.add_argument("--max-planner-images", type=int, default=8, help="Maximum images sent to the lightweight planner per step.")
    parser.add_argument("--history-size", type=int, default=5, help="Number of recent planner decisions kept in text history.")
    parser.add_argument("--camera-timeout", type=float, default=5.0)
    parser.add_argument("--debug-image", type=Path, default=DEFAULT_DEBUG_IMAGE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-threshold", type=float, default=0.04)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--planner-max-new-tokens", type=int, default=128)
    parser.add_argument("--vlm-image-scale", type=float, default=_float_from_env("UNITREE_VLM_IMAGE_SCALE", 1.25))
    parser.add_argument("--vlm-sharpness", type=float, default=_float_from_env("UNITREE_VLM_SHARPNESS", 1.25))
    parser.add_argument("--vlm-contrast", type=float, default=_float_from_env("UNITREE_VLM_CONTRAST", 1.05))
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--command-hz", type=float, default=2.0)
    parser.add_argument("--stream-hz", type=float, default=10.0, help="DDS command keepalive publish rate while Qwen is running.")
    parser.add_argument("--channel-id", type=int, default=0)
    parser.add_argument("--dds-interface", default="lo")
    parser.add_argument("--height", type=float, default=0.8)
    parser.add_argument("--max-speed", type=float, default=0.28)
    parser.add_argument("--min-speed", type=float, default=0.08)
    parser.add_argument("--max-yaw-rate", type=float, default=0.42)
    parser.add_argument("--yaw-kp", type=float, default=0.55)
    parser.add_argument("--search-yaw-rate", type=float, default=0.25)
    parser.add_argument("--search-sweep-interval", type=float, default=4.0)
    parser.add_argument("--center-tolerance", type=float, default=0.12)
    parser.add_argument("--forward-center-limit", type=float, default=0.35)
    parser.add_argument("--hard-turn-center-limit", type=float, default=0.85)
    parser.add_argument("--offcenter-forward-scale", type=float, default=0.35)
    parser.add_argument("--stop-area-ratio", type=float, default=0.10)
    parser.add_argument("--slow-area-ratio", type=float, default=0.10)
    parser.add_argument("--stop-height-ratio", type=float, default=0.45)
    parser.add_argument("--target-world-xy", nargs=2, type=float, default=None, metavar=("X", "Y"))
    parser.add_argument("--target-arrival-radius", type=float, default=0.0)
    parser.add_argument("--target-pose-max-age", type=float, default=2.0)
    parser.add_argument("--debug-interval", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model_path.exists():
        print(f"[visual-vlm] missing model path: {args.model_path}", file=sys.stderr)
        print(
            "[visual-vlm] download with: hf download Qwen/Qwen3-VL-2B-Instruct "
            "--local-dir models/vlm/qwen3-vl-2b-instruct",
            file=sys.stderr,
        )
        return 2
    return execute_servo(args)


if __name__ == "__main__":
    raise SystemExit(main())
