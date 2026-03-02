from __future__ import annotations

import logging
from pathlib import Path

from mlbb_pipeline.common import clamp, run_command

MODE_MULTI_CAM_STRIP = "MULTI_CAM_STRIP"
MODE_SINGLE_CAM = "SINGLE_CAM"
MODE_NO_FACE = "NO_FACE"

TEMPLATE_GAMEPLAY_EPIC = "GAMEPLAY_EPIC"
TEMPLATE_TALK_HOTTAKE = "TALK_HOTTAKE"
TEMPLATE_COMEDY_REACTION = "COMEDY_REACTION"
TEMPLATE_MULTICAM = "MULTICAM"
TEMPLATE_DONATION_QNA = "DONATION_QNA"
TEMPLATE_PRO_ENCOUNTER = "PRO_ENCOUNTER"


def cut_clip(
    source_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    ffmpeg_bin: str,
    logger: logging.Logger,
) -> None:
    """Cut source video to a raw clip with re-encoding for stable timestamps."""
    duration = max(0.5, end_sec - start_sec)
    command = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_path),
    ]
    run_command(command, logger=logger)


def compose_tiktok_clip(
    raw_clip_path: Path,
    output_path: Path,
    ffmpeg_bin: str,
    top_banner_px: int,
    bot_banner_px: int,
    event_image_path: str | None,
    facecam_roi_norm: list[float] | None,
    facecam_mode: str | None,
    edit_template: str | None,
    gameplay_ratio: float,
    importance_score: float | None,
    logger: logging.Logger,
) -> dict:
    """Compose 1080x1920 clip with template-specific layout and packaging."""
    top_h = max(0, int(top_banner_px))
    bot_h = max(0, int(bot_banner_px))
    middle_h = 1920 - top_h - bot_h
    if middle_h <= 200:
        raise ValueError(
            f"Invalid banner sizes: top={top_h}, bottom={bot_h}, middle={middle_h}"
        )

    mode = _normalize_mode(facecam_mode=facecam_mode, has_roi=bool(facecam_roi_norm))
    template = _normalize_template(edit_template=edit_template, facecam_mode=mode)
    profile = _layout_profile(template=template, fallback_ratio=gameplay_ratio)
    legacy_compose_without_bottom = facecam_mode is None and not facecam_roi_norm
    use_face_panel = not legacy_compose_without_bottom
    has_facecam = mode in {MODE_MULTI_CAM_STRIP, MODE_SINGLE_CAM} and bool(facecam_roi_norm)
    face_position = profile["face_position"]
    zoom_factor = _importance_zoom_factor(
        template=template,
        importance_score=importance_score,
    )

    if use_face_panel:
        gameplay_h = max(200, int(middle_h * clamp(profile["gameplay_ratio"], 0.35, 0.9)))
        face_h = middle_h - gameplay_h
        if face_h < 120:
            gameplay_h = middle_h - 120
            face_h = 120
    else:
        gameplay_h = middle_h
        face_h = 0

    filter_parts: list[str] = []
    stack_inputs: list[str] = []

    use_event_banners = bool(
        event_image_path and Path(event_image_path).exists() and (top_h > 0 or bot_h > 0)
    )

    if use_event_banners:
        filter_parts.append(
            "[1:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920[ev]"
        )
        if top_h > 0 and bot_h > 0:
            filter_parts.append("[ev]split=2[evtop][evbot]")
            filter_parts.append(f"[evtop]crop=1080:{top_h}:0:0[topbar]")
            filter_parts.append(f"[evbot]crop=1080:{bot_h}:0:{1920 - bot_h}[botbar]")
        elif top_h > 0:
            filter_parts.append(f"[ev]crop=1080:{top_h}:0:0[topbar]")
        elif bot_h > 0:
            filter_parts.append(f"[ev]crop=1080:{bot_h}:0:{1920 - bot_h}[botbar]")
    else:
        if top_h > 0:
            filter_parts.append(f"color=c=black:s=1080x{top_h}[topbar]")
        if bot_h > 0:
            filter_parts.append(f"color=c=black:s=1080x{bot_h}[botbar]")

    if top_h > 0:
        stack_inputs.append("[topbar]")

    if zoom_factor > 1.001:
        filter_parts.append(
            "[0:v]"
            f"scale=iw*{zoom_factor:.4f}:ih*{zoom_factor:.4f},"
            f"crop=iw/{zoom_factor:.4f}:ih/{zoom_factor:.4f}:x=(iw-ow)/2:y=(ih-oh)/2,"
            f"scale=1080:{gameplay_h}:force_original_aspect_ratio=increase,"
            f"crop=1080:{gameplay_h}[gp]"
        )
    else:
        filter_parts.append(
            f"[0:v]scale=1080:{gameplay_h}:force_original_aspect_ratio=increase,"
            f"crop=1080:{gameplay_h}[gp]"
        )

    face_source = "none"
    if use_face_panel and has_facecam and face_h > 0:
        rx1, ry1, rx2, ry2 = facecam_roi_norm
        rw = max(0.05, clamp(rx2 - rx1, 0.05, 1.0))
        rh = max(0.05, clamp(ry2 - ry1, 0.05, 1.0))
        rx = clamp(rx1, 0.0, 0.95)
        ry = clamp(ry1, 0.0, 0.95)
        filter_parts.append(
            "[0:v]"
            f"crop=iw*{rw:.6f}:ih*{rh:.6f}:iw*{rx:.6f}:ih*{ry:.6f},"
            f"scale=1080:{face_h}:force_original_aspect_ratio=increase,"
            f"crop=1080:{face_h}[face]"
        )
        if face_position == "top":
            filter_parts.append("[face][gp]vstack=inputs=2[mid]")
        else:
            filter_parts.append("[gp][face]vstack=inputs=2[mid]")
        face_source = "face_crop"
    elif use_face_panel and face_h > 0:
        # NO_FACE fallback: keep lower panel filled with blurred gameplay context.
        filter_parts.append(
            "[0:v]"
            f"scale=1080:{face_h}:force_original_aspect_ratio=increase,"
            f"crop=1080:{face_h},boxblur=16:2[face]"
        )
        if face_position == "top":
            filter_parts.append("[face][gp]vstack=inputs=2[mid]")
        else:
            filter_parts.append("[gp][face]vstack=inputs=2[mid]")
        face_source = "blur_gameplay"
    else:
        filter_parts.append("[gp]null[mid]")

    stack_inputs.append("[mid]")
    if bot_h > 0:
        stack_inputs.append("[botbar]")

    if len(stack_inputs) == 1:
        filter_parts.append(f"{stack_inputs[0]}null[outv]")
    else:
        filter_parts.append("".join(stack_inputs) + f"vstack=inputs={len(stack_inputs)}[outv]")

    filter_complex = ";".join(filter_parts)

    command = [ffmpeg_bin, "-y", "-i", str(raw_clip_path)]
    if use_event_banners:
        command.extend(["-loop", "1", "-i", str(event_image_path)])
    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-r",
            "30",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_path),
        ]
    )
    run_command(command, logger=logger)

    return {
        "has_facecam": has_facecam,
        "facecam_mode": mode,
        "edit_template": template,
        "face_panel_source": face_source,
        "face_position": face_position if use_face_panel else "none",
        "importance_zoom_factor": round(float(zoom_factor), 4),
        "facecam_roi_norm": facecam_roi_norm if has_facecam else None,
        "layout": {
            "top_banner_px": top_h,
            "bottom_banner_px": bot_h,
            "middle_height_px": middle_h,
            "gameplay_height_px": gameplay_h,
            "face_height_px": face_h,
            "gameplay_ratio": profile["gameplay_ratio"] if use_face_panel else 1.0,
        },
    }


def _normalize_mode(facecam_mode: str | None, has_roi: bool) -> str:
    if facecam_mode:
        mode = facecam_mode.strip().upper()
        if mode in {MODE_MULTI_CAM_STRIP, MODE_SINGLE_CAM, MODE_NO_FACE}:
            return mode
    return MODE_SINGLE_CAM if has_roi else MODE_NO_FACE


def _normalize_template(edit_template: str | None, facecam_mode: str) -> str:
    if facecam_mode == MODE_MULTI_CAM_STRIP:
        return TEMPLATE_MULTICAM
    if edit_template:
        normalized = edit_template.strip().upper()
        if normalized in {
            TEMPLATE_GAMEPLAY_EPIC,
            TEMPLATE_TALK_HOTTAKE,
            TEMPLATE_COMEDY_REACTION,
            TEMPLATE_MULTICAM,
            TEMPLATE_DONATION_QNA,
            TEMPLATE_PRO_ENCOUNTER,
        }:
            return normalized
    return TEMPLATE_GAMEPLAY_EPIC


def _layout_profile(template: str, fallback_ratio: float) -> dict[str, float | str]:
    defaults = {"gameplay_ratio": clamp(fallback_ratio, 0.55, 0.85), "face_position": "bottom"}
    by_template: dict[str, dict[str, float | str]] = {
        TEMPLATE_GAMEPLAY_EPIC: {"gameplay_ratio": 0.80, "face_position": "bottom"},
        TEMPLATE_PRO_ENCOUNTER: {"gameplay_ratio": 0.74, "face_position": "bottom"},
        TEMPLATE_MULTICAM: {"gameplay_ratio": 0.62, "face_position": "bottom"},
        TEMPLATE_TALK_HOTTAKE: {"gameplay_ratio": 0.40, "face_position": "top"},
        TEMPLATE_COMEDY_REACTION: {"gameplay_ratio": 0.45, "face_position": "top"},
        TEMPLATE_DONATION_QNA: {"gameplay_ratio": 0.42, "face_position": "top"},
    }
    profile = by_template.get(template, defaults)
    return {
        "gameplay_ratio": float(profile.get("gameplay_ratio", defaults["gameplay_ratio"])),
        "face_position": str(profile.get("face_position", defaults["face_position"])),
    }


def _importance_zoom_factor(template: str, importance_score: float | None) -> float:
    try:
        score = float(importance_score if importance_score is not None else 0.0)
    except (TypeError, ValueError):
        score = 0.0
    score = clamp(score, 0.0, 1.0)

    if template in {TEMPLATE_TALK_HOTTAKE, TEMPLATE_DONATION_QNA, TEMPLATE_COMEDY_REACTION}:
        max_zoom = 1.025
    elif template == TEMPLATE_MULTICAM:
        max_zoom = 1.015
    else:
        max_zoom = 1.06
    return 1.0 + (max_zoom - 1.0) * score
