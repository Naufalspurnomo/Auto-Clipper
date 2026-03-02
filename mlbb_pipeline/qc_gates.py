from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from mlbb_pipeline.common import run_command
from mlbb_pipeline.template_selector import (
    TEMPLATE_COMEDY_REACTION,
    TEMPLATE_DONATION_QNA,
    TEMPLATE_GAMEPLAY_EPIC,
    TEMPLATE_MULTICAM,
    TEMPLATE_TALK_HOTTAKE,
)


FACE_REQUIRED_TEMPLATES = {
    TEMPLATE_TALK_HOTTAKE,
    TEMPLATE_COMEDY_REACTION,
    TEMPLATE_DONATION_QNA,
    TEMPLATE_MULTICAM,
}

WEIRD_WORD_PATTERNS = (
    r"\bcepek\b",
    r"\bsevej\b",
    r"\bmeniak\b",
    r"\btripel kill\b",
    r"\bdobel kill\b",
)


def choose_template_with_qc(
    requested_template: str,
    facecam_mode: str,
    facecam_found: bool,
) -> dict[str, Any]:
    requested = str(requested_template or TEMPLATE_GAMEPLAY_EPIC).strip().upper()
    mode = str(facecam_mode or "").strip().upper()

    selected = requested
    warnings: list[str] = []
    hard_failures: list[str] = []
    fallback_reason = ""
    fallback_applied = False

    if requested == TEMPLATE_MULTICAM and mode != "MULTI_CAM_STRIP":
        fallback_applied = True
        fallback_reason = "MULTICAM requested but facecam mode is not MULTI_CAM_STRIP."
        selected = TEMPLATE_TALK_HOTTAKE if mode == "SINGLE_CAM" else TEMPLATE_GAMEPLAY_EPIC

    if selected in FACE_REQUIRED_TEMPLATES and not facecam_found:
        fallback_applied = True
        hard_failures.append("face_required_template_without_facecam")
        fallback_reason = (
            fallback_reason or "Template requires face panel but stable facecam was not detected."
        )
        selected = TEMPLATE_GAMEPLAY_EPIC

    if selected in {
        TEMPLATE_TALK_HOTTAKE,
        TEMPLATE_DONATION_QNA,
        TEMPLATE_COMEDY_REACTION,
    } and mode == "MULTI_CAM_STRIP":
        warnings.append("Talk/comedy template on MULTI_CAM_STRIP may reduce face clarity.")

    return {
        "requested_template": requested,
        "selected_template": selected,
        "fallback_applied": bool(fallback_applied),
        "fallback_reason": fallback_reason,
        "warnings": warnings,
        "hard_failures": hard_failures,
    }


def evaluate_srt_quality(
    srt_path: Path,
    expected_template: str,
) -> dict[str, Any]:
    warnings: list[str] = []
    hard_failures: list[str] = []
    weird_hits: set[str] = set()
    entries = 0
    max_text_len = 0
    avg_text_len = 0.0

    if not srt_path.exists():
        return {
            "gate": "subtitle_readability",
            "pass": False,
            "can_burn": False,
            "entries": 0,
            "warnings": [],
            "hard_failures": ["subtitle_file_missing"],
            "weird_hits": [],
            "avg_text_len": 0.0,
            "max_text_len": 0,
        }

    raw = srt_path.read_text(encoding="utf-8", errors="replace")
    blocks = [block for block in re.split(r"\r?\n\r?\n", raw) if block.strip()]
    text_lengths: list[int] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        if "-->" not in lines[1]:
            continue
        entries += 1
        text = " ".join(lines[2:]).strip()
        if not text:
            continue
        text_lengths.append(len(text))
        max_text_len = max(max_text_len, len(text))
        for pattern in WEIRD_WORD_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                weird_hits.add(pattern)

    if text_lengths:
        avg_text_len = sum(text_lengths) / len(text_lengths)

    if entries == 0:
        hard_failures.append("subtitle_entries_empty")
    if weird_hits:
        warnings.append("Detected unclean MLBB words in subtitles.")
    if max_text_len > 84:
        warnings.append("Some subtitle lines are too long for mobile readability.")
    if expected_template in {TEMPLATE_GAMEPLAY_EPIC, TEMPLATE_MULTICAM} and avg_text_len > 56:
        warnings.append("Gameplay/multicam clip has dense subtitle text.")

    return {
        "gate": "subtitle_readability",
        "pass": len(hard_failures) == 0,
        "can_burn": len(hard_failures) == 0,
        "entries": entries,
        "warnings": warnings,
        "hard_failures": hard_failures,
        "weird_hits": sorted(weird_hits),
        "avg_text_len": round(float(avg_text_len), 2),
        "max_text_len": int(max_text_len),
    }


def evaluate_layout_safety(
    layout_info: dict[str, Any],
    expected_template: str,
) -> dict[str, Any]:
    warnings: list[str] = []
    hard_failures: list[str] = []
    layout = layout_info.get("layout", {}) if isinstance(layout_info, dict) else {}
    gameplay_ratio = _as_float(layout.get("gameplay_ratio", 0.0))
    has_facecam = bool(layout_info.get("has_facecam"))

    if expected_template in {TEMPLATE_GAMEPLAY_EPIC, TEMPLATE_MULTICAM} and gameplay_ratio < 0.55:
        warnings.append("Gameplay ratio is low for gameplay-heavy template.")
    if expected_template in {TEMPLATE_TALK_HOTTAKE, TEMPLATE_DONATION_QNA} and gameplay_ratio > 0.55:
        warnings.append("Face panel might be too small for talk/donation template.")
    if expected_template in FACE_REQUIRED_TEMPLATES and not has_facecam:
        hard_failures.append("face_panel_expected_but_missing")

    return {
        "gate": "layout_safety",
        "pass": len(hard_failures) == 0,
        "warnings": warnings,
        "hard_failures": hard_failures,
        "gameplay_ratio": round(gameplay_ratio, 4),
        "has_facecam": has_facecam,
    }


def evaluate_audio_levels(
    media_path: Path,
    ffmpeg_bin: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    warnings: list[str] = []
    hard_failures: list[str] = []

    if not media_path.exists():
        return {
            "gate": "audio_level",
            "pass": False,
            "warnings": [],
            "hard_failures": ["media_file_missing"],
            "max_volume_db": None,
            "mean_volume_db": None,
        }

    result = run_command(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-i",
            str(media_path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "NUL" if os.name == "nt" else "/dev/null",
        ],
        logger=logger,
        allow_failure=True,
    )
    text = "\n".join([(result.stdout or ""), (result.stderr or "")])
    max_volume = _extract_db(text, "max_volume")
    mean_volume = _extract_db(text, "mean_volume")

    if max_volume is None or mean_volume is None:
        warnings.append("Unable to parse ffmpeg volumedetect output.")
    else:
        if max_volume > -0.5:
            warnings.append("Audio peak is close to clipping.")
        if max_volume > 0.1:
            hard_failures.append("audio_clipping_detected")
        if mean_volume < -32.0:
            warnings.append("Audio level is very low.")

    return {
        "gate": "audio_level",
        "pass": len(hard_failures) == 0,
        "warnings": warnings,
        "hard_failures": hard_failures,
        "max_volume_db": max_volume,
        "mean_volume_db": mean_volume,
    }


def _extract_db(text: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}:\s*(-?\d+(?:\.\d+)?)\s*dB", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
