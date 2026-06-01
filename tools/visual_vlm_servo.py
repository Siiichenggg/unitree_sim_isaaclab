#!/usr/bin/env python3

"""Camera VLM target detection and visual-servo navigation.

This script reads the live Isaac head camera from shared memory, uses Qwen-VL
to ground a natural-language target in the image, and optionally publishes
Unitree wholebody run commands to visually servo toward that target.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "vlm" / "qwen3-vl-2b-instruct"
DEFAULT_DEBUG_IMAGE = PROJECT_ROOT / "logs" / "or_vlm_nav" / "visual_grounding_debug.jpg"

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


def split_csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def live_camera_sequence(args: argparse.Namespace) -> list[str]:
    requested = split_csv(args.search_cameras)
    if args.camera and args.camera not in requested:
        requested.append(args.camera)
    if not requested:
        requested = [args.camera]

    existing: list[str] = []
    for camera in requested:
        shm_path = Path("/dev/shm") / f"isaac_{camera}_image_shm"
        if shm_path.exists() and camera not in existing:
            existing.append(camera)
    if existing:
        return existing
    return [args.camera]


def load_live_frame(camera: str, timeout_s: float, reader=None) -> Image.Image:
    from tools.shared_memory_utils import MultiImageReader

    owns_reader = reader is None
    if reader is None:
        reader = MultiImageReader()
    try:
        deadline = time.monotonic() + timeout_s
        shm_path = Path("/dev/shm") / f"isaac_{camera}_image_shm"
        while time.monotonic() < deadline:
            if not shm_path.exists():
                time.sleep(0.05)
                continue
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


def grounding_prompt(instruction: str, image: Image.Image) -> str:
    return (
        "You are the vision module for a robot. Look only at the provided image. "
        f"The image size is {image.width}x{image.height} pixels. "
        f"Robot command: {instruction!r}. "
        "Find the visible target object or place implied by the command. "
        "Return ONLY one JSON object with this schema: "
        '{"found": true, "label": "target name", "box": [x1, y1, x2, y2], "confidence": 0.0}. '
        "Use pixel coordinates in the original image, origin at top-left. "
        "If the target is not visible, return exactly: "
        '{"found": false, "label": "", "box": null, "confidence": 0.0}.'
    )


def qwen_ground(processor, model, image: Image.Image, instruction: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": grounding_prompt(instruction, image)},
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


def detect_target(processor, model, image: Image.Image, instruction: str, threshold: float, max_new_tokens: int) -> tuple[Detection | None, str]:
    raw_text = qwen_ground(processor, model, image, instruction, max_new_tokens)
    return detection_from_qwen(raw_text, image, threshold), raw_text


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
    done = centered and close_enough

    if done:
        return 0.0, 0.0, 0.0, True

    yaw_rate = max(-args.max_yaw_rate, min(args.max_yaw_rate, -args.yaw_kp * center_error))
    scale = max(0.0, 1.0 - area_ratio / max(args.stop_area_ratio, 1e-6))
    if abs(center_error) > args.hard_turn_center_limit:
        vx = 0.0
    else:
        if abs(center_error) > args.forward_center_limit:
            alignment = args.offcenter_forward_scale
        else:
            alignment = max(args.offcenter_forward_scale, 1.0 - abs(center_error))
        vx = max(args.min_speed, args.max_speed * scale * alignment)
    return vx, 0.0, yaw_rate, False


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


def execute_servo(args: argparse.Namespace) -> int:
    from tools.astar_nav_dds import RunCommandPublisher
    from tools.shared_memory_utils import MultiImageReader

    processor, model = load_detector(args.model_path, args.device)
    publisher = RunCommandPublisher(channel_id=args.channel_id, dds_interface=args.dds_interface)
    streamer = CommandStreamer(publisher, height=args.height, hz=args.stream_hz)
    streamer.start()
    reader = MultiImageReader()
    camera_names = live_camera_sequence(args)
    print(f"[visual-vlm] visual instruction={args.instruction}")
    print(f"[visual-vlm] search cameras={camera_names}")
    print("[visual-vlm] mode=execute; no fixed XY map is used")

    deadline = time.monotonic() + args.duration
    last_debug = 0.0
    search_direction = -1.0
    next_search_switch = time.monotonic() + args.search_sweep_interval
    camera_index = 0
    try:
        while time.monotonic() < deadline:
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
                args.score_threshold,
                args.max_new_tokens,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use live camera visual grounding to walk toward an object.")
    parser.add_argument("instruction", help="Target instruction, e.g. 'walk to the cylinder'.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--camera", choices=["head", "head_up", "head_down", "head_left", "head_right", "left", "right"], default="head")
    parser.add_argument("--search-cameras", default="head,head_left,head_right,head_up,head_down", help="Comma-separated live camera sweep order.")
    parser.add_argument("--camera-timeout", type=float, default=5.0)
    parser.add_argument("--debug-image", type=Path, default=DEFAULT_DEBUG_IMAGE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-threshold", type=float, default=0.04)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--command-hz", type=float, default=2.0)
    parser.add_argument("--stream-hz", type=float, default=10.0, help="DDS command keepalive publish rate while Qwen is running.")
    parser.add_argument("--channel-id", type=int, default=0)
    parser.add_argument("--dds-interface", default="lo")
    parser.add_argument("--height", type=float, default=0.8)
    parser.add_argument("--max-speed", type=float, default=0.45)
    parser.add_argument("--min-speed", type=float, default=0.22)
    parser.add_argument("--max-yaw-rate", type=float, default=0.55)
    parser.add_argument("--yaw-kp", type=float, default=0.65)
    parser.add_argument("--search-yaw-rate", type=float, default=0.25)
    parser.add_argument("--search-sweep-interval", type=float, default=4.0)
    parser.add_argument("--center-tolerance", type=float, default=0.12)
    parser.add_argument("--forward-center-limit", type=float, default=0.60)
    parser.add_argument("--hard-turn-center-limit", type=float, default=0.85)
    parser.add_argument("--offcenter-forward-scale", type=float, default=0.35)
    parser.add_argument("--stop-area-ratio", type=float, default=0.10)
    parser.add_argument("--stop-height-ratio", type=float, default=0.45)
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
