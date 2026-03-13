from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract


MLBB_KEYWORDS = [
    "SAVAGE",
    "MANIAC",
    "TRIPLE KILL",
    "DOUBLE KILL",
    "MEGA KILL",
    "MONSTER KILL",
    "KILLING SPREE",
    "UNSTOPPABLE",
    "GODLIKE",
    "LEGENDARY",
    "WIPED OUT",
    "WIPE OUT",
    "SHUT DOWN",
    "HAS SLAIN",
    "DESTROYED",
    "TURRET",
    "LORD",
    "TURTLE",
]

EVENT_TYPE_MAP = {
    "SAVAGE": "multi_kill",
    "MANIAC": "multi_kill",
    "TRIPLE KILL": "multi_kill",
    "DOUBLE KILL": "multi_kill",
    "MEGA KILL": "streak",
    "MONSTER KILL": "streak",
    "KILLING SPREE": "streak",
    "UNSTOPPABLE": "streak",
    "GODLIKE": "streak",
    "LEGENDARY": "streak",
    "WIPED OUT": "teamfight",
    "WIPE OUT": "teamfight",
    "SHUT DOWN": "shutdown",
    "HAS SLAIN": "pickoff",
    "DESTROYED": "objective",
    "TURRET": "objective",
    "LORD": "objective",
    "TURTLE": "objective",
}

HIT_WEIGHTS = {
    "SAVAGE": 1.0,
    "MANIAC": 0.92,
    "TRIPLE KILL": 0.82,
    "DOUBLE KILL": 0.65,
    "MEGA KILL": 0.55,
    "MONSTER KILL": 0.62,
    "KILLING SPREE": 0.48,
    "UNSTOPPABLE": 0.58,
    "GODLIKE": 0.72,
    "LEGENDARY": 0.9,
    "WIPED OUT": 0.95,
    "WIPE OUT": 0.95,
    "SHUT DOWN": 0.7,
    "HAS SLAIN": 0.32,
    "DESTROYED": 0.42,
    "TURRET": 0.38,
    "LORD": 0.78,
    "TURTLE": 0.6,
}

DEFAULT_ROIS = {
    "announcement": (0.20, 0.02, 0.80, 0.18),
    "kill_feed": (0.65, 0.02, 0.98, 0.28),
}


def detect_ocr_hits(
    proxy_video_path: Path,
    ocr_lang: str = "eng",
    sample_fps: float = 1.0,
    roi_presets: dict[str, tuple[float, float, float, float]] | None = None,
    auto_roi_scan_seconds: int = 60,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Detect MLBB on-screen keywords and kill-feed burst signals."""
    capture = cv2.VideoCapture(str(proxy_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open proxy video: {proxy_video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    rois = roi_presets or _auto_select_rois(
        proxy_video_path=proxy_video_path,
        width=width,
        height=height,
        native_fps=native_fps,
        sample_fps=sample_fps,
        ocr_lang=ocr_lang,
        scan_seconds=auto_roi_scan_seconds,
        logger=logger,
    )

    stride = max(1, int(round(native_fps / max(sample_fps, 0.1))))
    frame_idx = 0
    sampled_idx = 0
    prev_roi_gray: dict[str, np.ndarray] = {}
    timeline: list[dict[str, Any]] = []
    keyword_events: list[dict[str, Any]] = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        ts = sampled_idx / max(sample_fps, 0.1)
        frame_score = 0.0
        frame_hits: list[str] = []
        frame_burst_max = 0.0

        for roi_name, roi in rois.items():
            crop = _crop_norm(frame, roi)
            prepared = _prepare_ocr_image(crop)
            text_raw = pytesseract.image_to_string(
                prepared,
                lang=ocr_lang,
                config="--oem 3 --psm 6",
            )
            text = " ".join(text_raw.upper().split())
            hits = _extract_hits(text)
            burst = _calc_burst(roi_name, prepared, prev_roi_gray)
            prev_roi_gray[roi_name] = prepared

            hit_score = min(1.0, len(hits) / 2.0)
            burst_score = min(1.0, burst * 4.0)
            roi_weight = 1.0 if roi_name == "announcement" else 0.8
            roi_score = roi_weight * (0.75 * hit_score + 0.25 * burst_score)
            frame_score += roi_score
            frame_burst_max = max(frame_burst_max, burst)

            if hits:
                frame_hits.extend(hits)
                keyword_events.append(
                    {
                        "timestamp_sec": round(ts, 3),
                        "roi": roi_name,
                        "hits": sorted(set(hits)),
                        "event_types": _classify_event_types(hits),
                        "event_strength": round(float(_score_hits(hits)), 4),
                        "text": text[:200],
                        "burst": round(burst, 4),
                    }
                )

        norm_score = frame_score / max(1, len(rois))
        timeline.append(
            {
                "timestamp_sec": round(ts, 3),
                "score": round(float(norm_score), 4),
                "hits": sorted(set(frame_hits)),
                "event_types": _classify_event_types(frame_hits),
                "event_strength": round(float(_score_hits(frame_hits)), 4),
                "burst": round(float(frame_burst_max), 4),
            }
        )

        frame_idx += 1
        sampled_idx += 1

    capture.release()

    if logger:
        logger.info(
            "OCR analysis: %d timeline points, %d keyword events, rois=%s",
            len(timeline),
            len(keyword_events),
            rois,
        )

    return {
        "timeline": timeline,
        "events": keyword_events,
        "rois": {name: [round(v, 4) for v in coords] for name, coords in rois.items()},
    }


def _auto_select_rois(
    proxy_video_path: Path,
    width: int,
    height: int,
    native_fps: float,
    sample_fps: float,
    ocr_lang: str,
    scan_seconds: int,
    logger: logging.Logger | None,
) -> dict[str, tuple[float, float, float, float]]:
    candidates: dict[str, tuple[float, float, float, float]] = {
        "top_left": (0.02, 0.02, 0.45, 0.18),
        "top_center": (0.20, 0.02, 0.80, 0.18),
        "top_right": (0.55, 0.02, 0.98, 0.18),
        "feed_right": (0.65, 0.02, 0.98, 0.30),
    }
    scores = {name: 0 for name in candidates}

    cap = cv2.VideoCapture(str(proxy_video_path))
    if not cap.isOpened():
        return DEFAULT_ROIS.copy()

    max_samples = max(5, int(scan_seconds * max(sample_fps, 0.5)))
    stride = max(1, int(round(native_fps / max(sample_fps, 0.1))))
    stride *= 3  # sparse scan for ROI discovery

    frame_idx = 0
    sample_count = 0
    while sample_count < max_samples:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        for name, roi in candidates.items():
            crop = _prepare_ocr_image(_crop_norm(frame, roi))
            text = pytesseract.image_to_string(
                crop,
                lang=ocr_lang,
                config="--oem 3 --psm 6",
            )
            scores[name] += len(_extract_hits(" ".join(text.upper().split())))
        sample_count += 1
        frame_idx += 1

    cap.release()

    announcement = max(
        ["top_left", "top_center", "top_right"],
        key=lambda name: scores.get(name, 0),
    )
    kill_feed = "feed_right"
    if scores[announcement] <= 0 and scores[kill_feed] <= 0:
        selected = DEFAULT_ROIS.copy()
    else:
        selected = {
            "announcement": candidates[announcement],
            "kill_feed": candidates[kill_feed],
        }

    if logger:
        logger.info("OCR auto-ROI scores: %s", scores)
        logger.info("OCR selected ROIs: %s", selected)
    return selected


def _crop_norm(frame: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = int(max(0.0, min(1.0, roi[0])) * w)
    y1 = int(max(0.0, min(1.0, roi[1])) * h)
    x2 = int(max(0.0, min(1.0, roi[2])) * w)
    y2 = int(max(0.0, min(1.0, roi[3])) * h)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def _prepare_ocr_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _extract_hits(text: str) -> list[str]:
    hits: list[str] = []
    for keyword in MLBB_KEYWORDS:
        if keyword in text:
            hits.append(keyword)
    if "TRIPLE" in text and "KILL" in text and "TRIPLE KILL" not in hits:
        hits.append("TRIPLE KILL")
    if "DOUBLE" in text and "KILL" in text and "DOUBLE KILL" not in hits:
        hits.append("DOUBLE KILL")
    if "WIPE" in text and "OUT" in text and "WIPED OUT" not in hits and "WIPE OUT" not in hits:
        hits.append("WIPED OUT")
    if "LORD" in text and "LORD" not in hits:
        hits.append("LORD")
    if "TURTLE" in text and "TURTLE" not in hits:
        hits.append("TURTLE")
    if "DESTROYED" in text and "DESTROYED" not in hits:
        hits.append("DESTROYED")
    return sorted(set(hits))


def _classify_event_types(hits: list[str]) -> list[str]:
    event_types = {EVENT_TYPE_MAP.get(str(hit).strip().upper(), "misc") for hit in hits}
    event_types.discard("misc")
    return sorted(event_types)


def _score_hits(hits: list[str]) -> float:
    if not hits:
        return 0.0
    strongest = max(HIT_WEIGHTS.get(str(hit).strip().upper(), 0.25) for hit in hits)
    diversity_bonus = min(0.25, max(0, len(_classify_event_types(hits)) - 1) * 0.1)
    count_bonus = min(0.2, max(0, len(set(hits)) - 1) * 0.05)
    return min(1.0, strongest + diversity_bonus + count_bonus)


def _calc_burst(
    roi_name: str,
    current: np.ndarray,
    previous_store: dict[str, np.ndarray],
) -> float:
    prev = previous_store.get(roi_name)
    if prev is None:
        return 0.0
    if prev.shape != current.shape:
        return 0.0
    diff = cv2.absdiff(current, prev)
    return float(np.mean(diff) / 255.0)
