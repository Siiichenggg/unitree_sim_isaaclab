# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""DDS-backed multi-camera image transport helpers."""

import time
import numpy as np
import cv2
from typing import Optional, Dict

from tools.dds_image_transport import DDSImageReader, DDSImagePublisher, camera_transport_description, transport_uses_dds


class MultiImageWriter:
    """DDS-only multi-camera image writer."""

    def __init__(self, enable_jpeg: bool = False, jpeg_quality: int = 85, skip_cvtcolor: bool = False):
        """Initialize the multi-image DDS writer.

        Args:
            enable_jpeg: whether to enable JPEG compression
            jpeg_quality: JPEG quality (0-100)
            skip_cvtcolor: whether to skip color conversion
        """
        # 50 FPS 限速（避免高频阻塞主循环）
        self._min_interval_sec = 1.0 / 50.0
        self._last_write_ts_ms = 0

        # 压缩与颜色空间配置（由主进程注入）
        self._enable_jpeg = bool(enable_jpeg)
        self._jpeg_quality = int(jpeg_quality)
        self._skip_cvtcolor = bool(skip_cvtcolor)

        self._dds_publisher = None
        self._dds_publish_disabled = False
        print(f"[MultiImageWriter] Initialized camera_transport={camera_transport_description()}")

    def set_options(self, *, enable_jpeg: Optional[bool] = None, jpeg_quality: Optional[int] = None, skip_cvtcolor: Optional[bool] = None):
        if enable_jpeg is not None:
            self._enable_jpeg = bool(enable_jpeg)
        if jpeg_quality is not None:
            self._jpeg_quality = int(jpeg_quality)
        if skip_cvtcolor is not None:
            self._skip_cvtcolor = bool(skip_cvtcolor)

    def _get_dds_publisher(self):
        if self._dds_publish_disabled:
            return None
        if self._dds_publisher is not None:
            return self._dds_publisher
        try:
            self._dds_publisher = DDSImagePublisher()
            return self._dds_publisher
        except Exception as exc:
            print(f"[MultiImageWriter] DDS image publisher unavailable: {exc}")
            self._dds_publish_disabled = True
            return None

    def write_images(self, images: Dict[str, np.ndarray]) -> bool:
        """Publish multiple images to DDS.

        Args:
            images: the image dictionary, keyed by camera name (for example 'head', 'left', 'scene_00')

        Returns:
            bool: whether the writing is successful
        """
        if not images:
            return False

        # 轻量限速：最多 50 FPS，直接跳过多余写入，避免阻塞主循环
        now_ms = int(time.time() * 1000)
        if self._last_write_ts_ms and (now_ms - self._last_write_ts_ms) < int(self._min_interval_sec * 1000):
            return True

        if not transport_uses_dds():
            return False

        success_count = 0

        for image_name, image in images.items():
            try:
                # 确保连续内存布局，尽量减少拷贝
                if not image.flags['C_CONTIGUOUS']:
                    image = np.ascontiguousarray(image)
                # OpenCV 期望 BGR 格式；可通过配置跳过转换
                if image.ndim == 3 and image.shape[2] == 3:
                    if not self._skip_cvtcolor:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)]
                ok, buffer = cv2.imencode('.jpg', image, encode_params)
                if not ok:
                    print(f"[MultiImageWriter] Failed to encode {image_name} as JPEG")
                    continue
                jpeg_bytes = buffer.tobytes()

                publisher = self._get_dds_publisher()
                if publisher is not None and publisher.publish_jpeg(image_name, jpeg_bytes, now_ms):
                    success_count += 1

            except Exception as e:
                print(f"[MultiImageWriter] Error writing {image_name}: {e}")
                continue

        self._last_write_ts_ms = now_ms
        return success_count > 0

    def close(self):
        """Close writer-owned resources."""
        self._dds_publisher = None


class MultiImageReader:
    """DDS-only multi-camera image reader."""

    def __init__(self):
        """Initialize the multi-image DDS reader."""
        self._dds_reader = DDSImageReader()

    def read_images(self) -> Optional[Dict[str, np.ndarray]]:
        """Read images from DDS.

        Returns:
            Dict[str, np.ndarray]: the image dictionary, the key is the image name, the value is the image array
        """
        images = {}
        image_names = ['head', 'left', 'right'] + [f'scene_{index:02d}' for index in range(11)]

        for image_name in image_names:
            image = self.read_single_image(image_name)
            if image is not None:
                images[image_name] = image

        return images if images else None

    def read_concatenated_image(self) -> Optional[np.ndarray]:
        """Read all images and concatenate them horizontally (for backward compatibility)

        Returns:
            np.ndarray: the concatenated image array; if the reading fails, return None
        """
        images = self.read_images()
        if images is None or not images:
            return None

        try:
            # Concatenate images in a stable diagnostic order.
            image_order = ['head', 'left', 'right']
            frames_to_concat = []

            for image_name in image_order:
                if image_name in images:
                    frames_to_concat.append(images[image_name])

            if not frames_to_concat:
                return None

            if len(frames_to_concat) > 1:
                concatenated_image = cv2.hconcat(frames_to_concat)
            else:
                concatenated_image = frames_to_concat[0]

            return concatenated_image

        except Exception as e:
            print(f"[MultiImageReader] Error concatenating images: {e}")
            return None

    def read_single_image(self, image_name: str) -> Optional[np.ndarray]:
        """Read a single specific image from DDS.

        Args:
            image_name: Name of the image to read, for example "head", "left", or "scene_00"

        Returns:
            np.ndarray: The requested image array, or None if not found or error
        """
        try:
            if self._dds_reader is not None:
                return self._dds_reader.read_single_image(image_name)
            return None

        except Exception as e:
            print(f"[MultiImageReader] Error reading single image {image_name}: {e}")
            return None

    def read_encoded_frame(self, image_name: str = "head") -> Optional[bytes]:
        """DDS reader exposes decoded frames only."""
        return None

    def close(self):
        """Close DDS subscriptions."""
        if self._dds_reader is not None:
            self._dds_reader.close()
            self._dds_reader = None
