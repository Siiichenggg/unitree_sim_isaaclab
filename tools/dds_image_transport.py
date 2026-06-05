#!/usr/bin/env python3

"""DDS transport for Isaac camera JPEG frames.

The simulator publishes one compressed frame per camera topic:

    rt/isaac/camera/<image_name>

Frames use Unitree SDK2's existing ``Go2FrontVideoData_`` message.  The JPEG
payload is stored in ``video720p`` regardless of camera resolution; dimensions
are recovered by JPEG decoding on the reader side.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import cv2
import numpy as np


DEFAULT_TOPIC_PREFIX = "rt/isaac/camera"
_dds_init_lock = threading.Lock()
_dds_initialized = False
_dds_init_key: tuple[int, str] | None = None


def _split_transport(value: str | None) -> set[str]:
    raw = (value or os.environ.get("UNITREE_CAMERA_TRANSPORT", "dds")).lower()
    raw = raw.replace("+", ",").replace(";", ",").replace(" ", ",")
    tokens = {token.strip() for token in raw.split(",") if token.strip()}
    if not tokens or "auto" in tokens:
        return {"dds"}
    if "both" in tokens:
        tokens.add("dds")
    return tokens


def transport_uses_dds(value: str | None = None) -> bool:
    return "dds" in _split_transport(value)


def camera_transport_description() -> str:
    tokens = sorted(_split_transport(None))
    return ",".join(tokens)


def _dds_channel_id() -> int:
    raw = os.environ.get("UNITREE_DDS_DOMAIN_ID", os.environ.get("UNITREE_DDS_CHANNEL", "0"))
    try:
        return int(raw)
    except ValueError:
        return 0


def _dds_interface() -> str:
    return os.environ.get("UNITREE_DDS_INTERFACE", "lo").strip()


def _initialize_dds_once(channel_id: int, dds_interface: str) -> None:
    global _dds_initialized, _dds_init_key
    key = (int(channel_id), str(dds_interface))
    with _dds_init_lock:
        if _dds_initialized:
            return
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        if dds_interface:
            ChannelFactoryInitialize(channel_id, dds_interface)
        else:
            ChannelFactoryInitialize(channel_id)
        _dds_initialized = True
        _dds_init_key = key


class DDSImagePublisher:
    """Publish latest JPEG frames on per-camera DDS topics."""

    def __init__(
        self,
        *,
        channel_id: Optional[int] = None,
        dds_interface: Optional[str] = None,
        topic_prefix: Optional[str] = None,
        max_hz: Optional[float] = None,
    ):
        self.channel_id = _dds_channel_id() if channel_id is None else int(channel_id)
        self.dds_interface = _dds_interface() if dds_interface is None else str(dds_interface)
        self.topic_prefix = (topic_prefix or os.environ.get("UNITREE_IMAGE_DDS_TOPIC_PREFIX", DEFAULT_TOPIC_PREFIX)).rstrip("/")
        self.max_hz = float(max_hz if max_hz is not None else os.environ.get("UNITREE_IMAGE_DDS_HZ", "5.0"))
        self._publishers = {}
        self._last_pub_s: dict[str, float] = {}
        self._lock = threading.Lock()
        _initialize_dds_once(self.channel_id, self.dds_interface)

        from unitree_sdk2py.idl.unitree_go.msg.dds_ import Go2FrontVideoData_

        self._msg_cls = Go2FrontVideoData_

    def _topic(self, image_name: str) -> str:
        return f"{self.topic_prefix}/{image_name}"

    def _publisher_for(self, image_name: str):
        with self._lock:
            publisher = self._publishers.get(image_name)
            if publisher is not None:
                return publisher
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import Go2FrontVideoData_

            publisher = ChannelPublisher(self._topic(image_name), Go2FrontVideoData_)
            publisher.Init()
            self._publishers[image_name] = publisher
            print(f"[DDSImagePublisher] topic={self._topic(image_name)} max_hz={self.max_hz:g}")
            return publisher

    def publish_jpeg(self, image_name: str, payload: bytes, timestamp_ms: int) -> bool:
        if not payload:
            return False
        now_s = time.monotonic()
        if self.max_hz > 0.0:
            min_interval = 1.0 / self.max_hz
            last = self._last_pub_s.get(image_name, 0.0)
            if last and now_s - last < min_interval:
                return True
        try:
            publisher = self._publisher_for(image_name)
            msg = self._msg_cls(
                time_frame=int(timestamp_ms),
                video720p=bytes(payload),
                video360p=b"",
                video180p=b"",
            )
            publisher.Write(msg)
            self._last_pub_s[image_name] = now_s
            return True
        except Exception as exc:
            print(f"[DDSImagePublisher] failed to publish {image_name}: {exc}")
            return False


class DDSImageReader:
    """Subscribe to per-camera DDS JPEG topics and expose decoded BGR frames."""

    def __init__(
        self,
        *,
        channel_id: Optional[int] = None,
        dds_interface: Optional[str] = None,
        topic_prefix: Optional[str] = None,
    ):
        self.channel_id = _dds_channel_id() if channel_id is None else int(channel_id)
        self.dds_interface = _dds_interface() if dds_interface is None else str(dds_interface)
        self.topic_prefix = (topic_prefix or os.environ.get("UNITREE_IMAGE_DDS_TOPIC_PREFIX", DEFAULT_TOPIC_PREFIX)).rstrip("/")
        self._subscribers = {}
        self._payloads: dict[str, tuple[int, bytes]] = {}
        self._decoded: dict[str, tuple[int, np.ndarray]] = {}
        self._lock = threading.Lock()
        self._available = False
        self._warned = False
        try:
            _initialize_dds_once(self.channel_id, self.dds_interface)
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import Go2FrontVideoData_

            self._msg_cls = Go2FrontVideoData_
            self._available = True
        except Exception as exc:
            self._warned = True
            print(f"[DDSImageReader] DDS unavailable: {exc}")

    def _topic(self, image_name: str) -> str:
        return f"{self.topic_prefix}/{image_name}"

    def _callback(self, image_name: str, msg) -> None:
        payload = msg.video720p or msg.video360p or msg.video180p
        if not payload:
            return
        with self._lock:
            self._payloads[image_name] = (int(msg.time_frame), bytes(payload))

    def ensure_subscriber(self, image_name: str) -> bool:
        if not self._available:
            return False
        if image_name in self._subscribers:
            return True
        try:
            from unitree_sdk2py.core.channel import ChannelSubscriber

            subscriber = ChannelSubscriber(self._topic(image_name), self._msg_cls)
            subscriber.Init(lambda msg, name=image_name: self._callback(name, msg), 1)
            self._subscribers[image_name] = subscriber
            return True
        except Exception as exc:
            if not self._warned:
                print(f"[DDSImageReader] failed to subscribe {self._topic(image_name)}: {exc}")
                self._warned = True
            return False

    def read_single_image(self, image_name: str) -> Optional[np.ndarray]:
        if not self.ensure_subscriber(image_name):
            return None
        with self._lock:
            payload_item = self._payloads.get(image_name)
            decoded_item = self._decoded.get(image_name)
        if payload_item is None:
            return decoded_item[1] if decoded_item is not None else None

        timestamp_ms, payload = payload_item
        if decoded_item is not None and decoded_item[0] == timestamp_ms:
            return decoded_item[1]

        encoded = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            return decoded_item[1] if decoded_item is not None else None
        with self._lock:
            self._decoded[image_name] = (timestamp_ms, image)
        return image

    def close(self) -> None:
        for subscriber in self._subscribers.values():
            for method_name in ("Close", "close", "Stop", "stop"):
                method = getattr(subscriber, method_name, None)
                if method is None:
                    continue
                try:
                    method()
                except Exception:
                    pass
                break
        self._subscribers.clear()
        self._payloads.clear()
        self._decoded.clear()
