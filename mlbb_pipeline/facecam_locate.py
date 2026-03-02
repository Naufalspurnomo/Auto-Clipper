from __future__ import annotations

from collections import Counter
import logging
from pathlib import Path
import statistics
from typing import Any

import cv2

BOTTOM_BAND_Y0 = 0.55
BOTTOM_BAND_Y1 = 1.00
MULTICAM_TOP_MARGIN = 0.04
SINGLE_TOP_MARGIN = 0.05
SINGLE_MIN_HEIGHT = 0.24

MODE_MULTI_CAM_STRIP = "MULTI_CAM_STRIP"
MODE_SINGLE_CAM = "SINGLE_CAM"
MODE_NO_FACE = "NO_FACE"


def locate_facecam_roi(
    video_path: Path,
    scan_seconds: int = 60,
    sample_fps: float = 1.0,
    min_hit_rate: float = 0.15,
    face_mode_auto: bool = True,
    multicam_min_faces: int = 3,
    multicam_area_ratio_max: float = 2.5,
    multicam_yvar_max: float = 0.0025,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Classify facecam layout and return a stable crop window."""
    face_cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(face_cascade_path))
    if detector.empty():
        raise RuntimeError(f"Failed to load face cascade: {face_cascade_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / max(sample_fps, 0.1))))
    max_samples = max(5, int(scan_seconds * max(sample_fps, 0.1)))

    samples_seen = 0
    frame_idx = 0
    sample_records: list[dict[str, Any]] = []

    while samples_seen < max_samples:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        faces = _detect_faces_in_bottom_band(frame=frame, detector=detector)
        sample_records.append(
            _classify_sample(
                faces=faces,
                face_mode_auto=face_mode_auto,
                multicam_min_faces=multicam_min_faces,
                multicam_area_ratio_max=multicam_area_ratio_max,
                multicam_yvar_max=multicam_yvar_max,
            )
        )

        samples_seen += 1
        frame_idx += 1

    cap.release()

    if samples_seen == 0:
        return {
            "found": False,
            "reason": "no_frames",
            "roi_name": None,
            "roi_norm": None,
            "mode": MODE_NO_FACE,
            "crop_window_norm": None,
            "snap_anchor": None,
            "raw_mode_counts": _mode_count_dict([]),
            "stable_mode_counts": _mode_count_dict([]),
            "samples": 0,
            "face_hit_rate": 0.0,
        }

    raw_modes = [record["mode"] for record in sample_records]
    stable_modes = _smooth_modes(raw_modes=raw_modes, sample_fps=sample_fps)
    final_mode = _choose_mode(stable_modes)

    stable_face_count = sum(1 for mode in stable_modes if mode != MODE_NO_FACE)
    face_hit_rate = stable_face_count / max(1, len(stable_modes))

    final_anchor: str | None = None
    final_crop = None
    if final_mode != MODE_NO_FACE:
        relevant_records = [
            sample_records[idx]
            for idx, mode in enumerate(stable_modes)
            if mode == final_mode and sample_records[idx].get("crop_window_norm")
        ]
        if final_mode == MODE_SINGLE_CAM:
            final_anchor = _majority_anchor(relevant_records)
            if final_anchor:
                relevant_records = [
                    record for record in relevant_records if record.get("snap_anchor") == final_anchor
                ]
        final_crop = _aggregate_crop_windows(
            [record["crop_window_norm"] for record in relevant_records]
        )

    if final_mode != MODE_NO_FACE and face_hit_rate < min_hit_rate:
        final_mode = MODE_NO_FACE
        final_crop = None
        final_anchor = None

    found = final_mode != MODE_NO_FACE and final_crop is not None
    result = {
        "found": bool(found),
        "mode": final_mode,
        "crop_window_norm": final_crop,
        "roi_norm": final_crop,  # backwards compatibility for existing callers
        "roi_name": _legacy_roi_name(final_mode, final_anchor) if found else None,
        "snap_anchor": final_anchor if found else None,
        "samples": samples_seen,
        "face_hit_rate": round(face_hit_rate, 4),
        "raw_mode_counts": _mode_count_dict(raw_modes),
        "stable_mode_counts": _mode_count_dict(stable_modes),
        "thresholds": {
            "face_mode_auto": bool(face_mode_auto),
            "multicam_min_faces": int(multicam_min_faces),
            "multicam_area_ratio_max": float(multicam_area_ratio_max),
            "multicam_yvar_max": float(multicam_yvar_max),
            "min_hit_rate": float(min_hit_rate),
        },
        "band_norm": [BOTTOM_BAND_Y0, BOTTOM_BAND_Y1],
        "sample_stats": _aggregate_sample_stats(sample_records),
    }
    if not found:
        result["reason"] = "no_stable_facecam_detected"

    if logger:
        logger.info("Facecam locator result: %s", result)
    return result


def _detect_faces_in_bottom_band(frame, detector) -> list[dict[str, float]]:
    frame_h, frame_w = frame.shape[:2]
    y0_px = int(frame_h * BOTTOM_BAND_Y0)
    y1_px = int(frame_h * BOTTOM_BAND_Y1)
    if y1_px <= y0_px:
        return []

    band = frame[y0_px:y1_px, :]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    min_face_px = max(20, int(min(frame_w, y1_px - y0_px) * 0.04))
    detected = detector.detectMultiScale(
        gray,
        scaleFactor=1.12,
        minNeighbors=4,
        minSize=(min_face_px, min_face_px),
    )

    faces: list[dict[str, float]] = []
    frame_area = float(max(1, frame_w * frame_h))
    for x, y, w, h in detected:
        x1 = _clamp(x / frame_w, 0.0, 1.0)
        y1 = _clamp((y + y0_px) / frame_h, 0.0, 1.0)
        x2 = _clamp((x + w) / frame_w, 0.0, 1.0)
        y2 = _clamp((y + h + y0_px) / frame_h, 0.0, 1.0)
        area = (max(1, w) * max(1, h)) / frame_area
        faces.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "area": area,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
            }
        )
    return faces


def _classify_sample(
    faces: list[dict[str, float]],
    face_mode_auto: bool,
    multicam_min_faces: int,
    multicam_area_ratio_max: float,
    multicam_yvar_max: float,
) -> dict[str, Any]:
    face_count = len(faces)
    if face_count == 0:
        return {
            "mode": MODE_NO_FACE,
            "face_count": 0,
            "area_ratio": 0.0,
            "y_var": 0.0,
            "crop_window_norm": None,
            "snap_anchor": None,
        }

    areas = [face["area"] for face in faces]
    y_centers = [face["cy"] for face in faces]
    max_area = max(areas)
    median_area = statistics.median(areas)
    y_var = statistics.pvariance(y_centers) if face_count > 1 else 0.0
    area_ratio = max_area / max(median_area, 1e-9)

    is_multicam = (
        face_mode_auto
        and face_count >= multicam_min_faces
        and area_ratio < multicam_area_ratio_max
        and y_var <= multicam_yvar_max
    )
    if is_multicam:
        min_y = min(face["y1"] for face in faces)
        y0 = _clamp(max(BOTTOM_BAND_Y0, min_y - MULTICAM_TOP_MARGIN), BOTTOM_BAND_Y0, 0.98)
        return {
            "mode": MODE_MULTI_CAM_STRIP,
            "face_count": face_count,
            "area_ratio": area_ratio,
            "y_var": y_var,
            "crop_window_norm": [0.0, y0, 1.0, 1.0],
            "snap_anchor": None,
        }

    dominant_face = max(faces, key=lambda face: face["area"])
    second_area = sorted(areas, reverse=True)[1] if face_count > 1 else 0.0
    dominance_ratio = max_area / max(second_area, 1e-9) if face_count > 1 else float("inf")
    dominant_share = max_area / max(sum(areas), 1e-9)
    has_dominant = (
        face_count == 1
        or dominance_ratio >= 1.35
        or dominant_share >= 0.48
        or area_ratio >= multicam_area_ratio_max
    )
    if not has_dominant:
        return {
            "mode": MODE_NO_FACE,
            "face_count": face_count,
            "area_ratio": area_ratio,
            "y_var": y_var,
            "crop_window_norm": None,
            "snap_anchor": None,
        }

    snap_anchor = "left" if dominant_face["cx"] < 0.5 else "right"
    crop_window = _build_single_cam_window(dominant_face, snap_anchor)
    return {
        "mode": MODE_SINGLE_CAM,
        "face_count": face_count,
        "area_ratio": area_ratio,
        "y_var": y_var,
        "crop_window_norm": crop_window,
        "snap_anchor": snap_anchor,
    }


def _build_single_cam_window(dominant_face: dict[str, float], snap_anchor: str) -> list[float]:
    x1 = 0.0 if snap_anchor == "left" else 0.5
    x2 = 0.5 if snap_anchor == "left" else 1.0
    y1 = _clamp(
        max(BOTTOM_BAND_Y0, dominant_face["y1"] - SINGLE_TOP_MARGIN),
        BOTTOM_BAND_Y0,
        0.98,
    )
    if 1.0 - y1 < SINGLE_MIN_HEIGHT:
        y1 = max(BOTTOM_BAND_Y0, 1.0 - SINGLE_MIN_HEIGHT)
    return [x1, y1, x2, 1.0]


def _smooth_modes(raw_modes: list[str], sample_fps: float) -> list[str]:
    if not raw_modes:
        return []

    window_size = max(3, int(round(max(sample_fps, 0.1) * 3.0)))
    hold_frames = max(2, int(round(max(sample_fps, 0.1) * 2.0)))
    stable: list[str] = []
    last_face_frame = -10_000

    for idx, raw_mode in enumerate(raw_modes):
        previous = stable[-1] if stable else None
        if raw_mode == MODE_NO_FACE and previous in {MODE_SINGLE_CAM, MODE_MULTI_CAM_STRIP}:
            if idx - last_face_frame <= hold_frames:
                stable.append(previous)
                continue

        start_idx = max(0, idx - window_size + 1)
        window_modes = raw_modes[start_idx : idx + 1]
        window_counter = Counter(window_modes)
        top_count = max(window_counter.values())
        candidates = [mode for mode, count in window_counter.items() if count == top_count]

        if previous in candidates:
            chosen = previous
        else:
            chosen = _prefer_face_mode(candidates)
        stable.append(chosen)
        if chosen != MODE_NO_FACE:
            last_face_frame = idx

    return stable


def _choose_mode(stable_modes: list[str]) -> str:
    if not stable_modes:
        return MODE_NO_FACE

    counts = Counter(stable_modes)
    top_count = max(counts.values())
    candidates = [mode for mode, count in counts.items() if count == top_count]
    if len(candidates) == 1:
        return candidates[0]

    # Tie-breaker: prefer latest mode to reduce stale locks.
    for mode in reversed(stable_modes):
        if mode in candidates:
            return mode
    return MODE_NO_FACE


def _majority_anchor(records: list[dict[str, Any]]) -> str | None:
    anchors = [record.get("snap_anchor") for record in records if record.get("snap_anchor")]
    if not anchors:
        return None
    counts = Counter(anchors)
    top_count = max(counts.values())
    candidates = [anchor for anchor, count in counts.items() if count == top_count]
    if len(candidates) == 1:
        return candidates[0]
    for record in reversed(records):
        anchor = record.get("snap_anchor")
        if anchor in candidates:
            return anchor
    return candidates[0]


def _aggregate_crop_windows(windows: list[list[float]]) -> list[float] | None:
    if not windows:
        return None
    coords = []
    for idx in range(4):
        values = [window[idx] for window in windows]
        coords.append(_clamp(float(statistics.median(values)), 0.0, 1.0))

    x1, y1, x2, y2 = coords
    if x2 - x1 < 0.05 or y2 - y1 < 0.05:
        return None
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def _mode_count_dict(modes: list[str]) -> dict[str, int]:
    counts = Counter(modes)
    return {
        MODE_MULTI_CAM_STRIP: int(counts.get(MODE_MULTI_CAM_STRIP, 0)),
        MODE_SINGLE_CAM: int(counts.get(MODE_SINGLE_CAM, 0)),
        MODE_NO_FACE: int(counts.get(MODE_NO_FACE, 0)),
    }


def _aggregate_sample_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "avg_faces": 0.0,
            "median_faces": 0.0,
            "median_area_ratio": 0.0,
            "median_y_var": 0.0,
        }

    face_counts = [record.get("face_count", 0) for record in records]
    area_ratios = [float(record.get("area_ratio", 0.0)) for record in records if record.get("face_count", 0) > 0]
    y_vars = [float(record.get("y_var", 0.0)) for record in records if record.get("face_count", 0) > 1]

    return {
        "avg_faces": round(float(statistics.mean(face_counts)), 4),
        "median_faces": round(float(statistics.median(face_counts)), 4),
        "median_area_ratio": round(float(statistics.median(area_ratios)) if area_ratios else 0.0, 4),
        "median_y_var": round(float(statistics.median(y_vars)) if y_vars else 0.0, 6),
    }


def _legacy_roi_name(mode: str, snap_anchor: str | None) -> str:
    if mode == MODE_MULTI_CAM_STRIP:
        return "multicam_strip"
    if mode == MODE_SINGLE_CAM:
        return f"single_{snap_anchor or 'unknown'}"
    return "no_face"


def _prefer_face_mode(candidates: list[str]) -> str:
    if MODE_MULTI_CAM_STRIP in candidates:
        return MODE_MULTI_CAM_STRIP
    if MODE_SINGLE_CAM in candidates:
        return MODE_SINGLE_CAM
    return MODE_NO_FACE


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
