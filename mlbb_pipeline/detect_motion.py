from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def detect_motion_intensity(
    proxy_video_path: Path,
    sample_fps: float = 2.0,
    downscale_width: int = 320,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Compute motion score by frame differencing."""
    capture = cv2.VideoCapture(str(proxy_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {proxy_video_path}")

    native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(native_fps / max(sample_fps, 0.1))))

    scores: list[dict[str, Any]] = []
    prev_gray: np.ndarray | None = None
    frame_idx = 0
    sampled_idx = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        resized = _resize_for_motion(frame, downscale_width)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            motion_raw = 0.0
        else:
            diff = cv2.absdiff(gray, prev_gray)
            motion_raw = float(np.mean(diff))

        timestamp = sampled_idx / max(sample_fps, 0.1)
        scores.append(
            {
                "timestamp_sec": float(timestamp),
                "motion_raw": motion_raw,
            }
        )

        prev_gray = gray
        frame_idx += 1
        sampled_idx += 1

    capture.release()

    normalized = _normalize([item["motion_raw"] for item in scores])
    for item, norm in zip(scores, normalized):
        item["score"] = float(norm)

    if logger:
        logger.info("Motion analysis: generated %d scored points", len(scores))
    return scores


def _resize_for_motion(frame: np.ndarray, target_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame
    ratio = target_width / float(width)
    new_h = max(1, int(height * ratio))
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.array(values, dtype=float)
    low = float(np.percentile(arr, 5))
    high = float(np.percentile(arr, 95))
    if high <= low:
        return [0.0 for _ in values]
    arr = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return arr.tolist()

