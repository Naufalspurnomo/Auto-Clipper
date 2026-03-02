from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np


def detect_audio_peaks(
    audio_path: Path,
    frame_sec: float = 1.0,
    hop_sec: float = 0.5,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Score audio intensity using RMS + upward slope approximation."""
    if logger:
        logger.info("Audio analysis: loading %s", audio_path)

    y, sr = librosa.load(audio_path.as_posix(), sr=16000, mono=True)
    frame_length = max(1, int(sr * frame_sec))
    hop_length = max(1, int(sr * hop_sec))

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rise = np.diff(rms, prepend=rms[0])
    rise = np.where(rise > 0, rise, 0.0)

    rms_norm = _robust_normalize(rms)
    rise_norm = _robust_normalize(rise)
    score = 0.7 * rms_norm + 0.3 * rise_norm

    timeline = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    results: list[dict[str, Any]] = []
    for ts, rms_v, rise_v, score_v in zip(timeline, rms, rise, score):
        results.append(
            {
                "timestamp_sec": float(ts),
                "rms": float(rms_v),
                "rise": float(rise_v),
                "score": float(score_v),
            }
        )

    if logger:
        logger.info("Audio analysis: generated %d scored points", len(results))
    return results


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.array([])
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 95))
    if high <= low:
        return np.zeros_like(values, dtype=float)
    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)

