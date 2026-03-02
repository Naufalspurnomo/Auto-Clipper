from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from mlbb_pipeline.common import run_command, seconds_to_srt

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

TEMPLATE_GAMEPLAY_EPIC = "GAMEPLAY_EPIC"
TEMPLATE_TALK_HOTTAKE = "TALK_HOTTAKE"
TEMPLATE_COMEDY_REACTION = "COMEDY_REACTION"
TEMPLATE_MULTICAM = "MULTICAM"
TEMPLATE_DONATION_QNA = "DONATION_QNA"
TEMPLATE_PRO_ENCOUNTER = "PRO_ENCOUNTER"

MLBB_GLOSSARY = {
    r"\b(sav[ae]ge|sevej|cepek)\b": "Savage",
    r"\b(maniac|meniak|maniak)\b": "Maniac",
    r"\b(triple\s*kill|triplekill|tripel kill)\b": "Triple Kill",
    r"\b(double\s*kill|dobel kill|doublekill)\b": "Double Kill",
    r"\b(shut\s*down|shutdown)\b": "Shutdown",
    r"\b(wipe\s*out|wipeout)\b": "Wipe Out",
    r"\b(mobile\s*legend[s]?)\b": "Mobile Legends",
    r"\b(ml)\b": "ML",
}


def transcribe_to_srt(
    media_path: Path,
    srt_path: Path,
    transcript_json_path: Path,
    whisper_model: str,
    api_key: str | None,
    base_url: str | None,
    use_api: bool,
    local_fallback: bool,
    logger: logging.Logger,
    hook_text: str = "",
    hook_duration_sec: float = 2.0,
) -> dict[str, Any]:
    """Create SRT captions via Whisper API with optional local fallback."""
    payload: dict[str, Any] | None = None
    method = ""

    if use_api and api_key and OpenAI is not None:
        method = "whisper_api"
        payload = _transcribe_with_api(
            media_path=media_path,
            whisper_model=whisper_model,
            api_key=api_key,
            base_url=base_url,
        )
    elif use_api and not api_key:
        logger.warning("Whisper API requested but SUMOPOD_API_KEY is empty.")

    if payload is None and local_fallback:
        method = "local_whisper"
        payload = _transcribe_local_whisper(media_path, logger)

    if payload is None:
        return {
            "success": False,
            "method": "none",
            "reason": "No transcription backend available",
            "srt_path": str(srt_path),
        }

    transcript_json_path.parent.mkdir(parents=True, exist_ok=True)
    with transcript_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)

    entries = _build_srt_entries(payload)
    entries = _cleanup_entries(entries)
    entries = _prepend_hook_entry(
        entries=entries,
        hook_text=hook_text,
        hook_duration_sec=hook_duration_sec,
    )
    if not entries:
        return {
            "success": False,
            "method": method,
            "reason": "Transcription returned no timed words/segments",
            "srt_path": str(srt_path),
            "transcript_json_path": str(transcript_json_path),
        }

    srt_path.parent.mkdir(parents=True, exist_ok=True)
    with srt_path.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(entries, 1):
            handle.write(f"{idx}\n")
            handle.write(
                f"{seconds_to_srt(item['start'])} --> {seconds_to_srt(item['end'])}\n"
            )
            handle.write(f"{item['text']}\n\n")

    return {
        "success": True,
        "method": method,
        "entries": len(entries),
        "hook_applied": bool(hook_text),
        "srt_path": str(srt_path),
        "transcript_json_path": str(transcript_json_path),
    }


def burn_srt_captions(
    input_video_path: Path,
    output_video_path: Path,
    srt_path: Path,
    ffmpeg_bin: str,
    logger: logging.Logger,
    margin_v: int = 90,
    caption_template: str | None = None,
) -> None:
    """Burn SRT subtitles into video using TikTok-safe style."""
    if not srt_path.exists():
        logger.warning("Caption burn skipped because SRT does not exist: %s", srt_path)
        shutil.copy2(input_video_path, output_video_path)
        return

    escaped_path = srt_path.resolve().as_posix().replace(":", "\\:")
    style_profile = _caption_style_profile(caption_template=caption_template, fallback_margin=margin_v)
    style = (
        "FontName=Arial,"
        f"FontSize={style_profile['font_size']},"
        "PrimaryColour=&H00FFFFFF&,"
        "OutlineColour=&H00000000&,"
        "BorderStyle=1,"
        f"Outline={style_profile['outline']},"
        "Shadow=0,"
        "Alignment=2,"
        f"MarginV={style_profile['margin_v']}"
    )
    vf = f"subtitles='{escaped_path}':force_style='{style}'"
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_video_path),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-c:a",
        "copy",
        str(output_video_path),
    ]
    run_command(command, logger=logger)


def _transcribe_with_api(
    media_path: Path,
    whisper_model: str,
    api_key: str,
    base_url: str | None,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key, base_url=base_url or None)
    with media_path.open("rb") as handle:
        response = client.audio.transcriptions.create(
            model=whisper_model,
            file=handle,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
    return _to_jsonable(response)


def _transcribe_local_whisper(media_path: Path, logger: logging.Logger) -> dict[str, Any] | None:
    """Optional local fallback if `openai-whisper` is installed."""
    try:
        import whisper
    except ImportError:
        logger.warning("Local whisper fallback requested but package is not installed.")
        return None

    model = whisper.load_model("base")
    result = model.transcribe(str(media_path), word_timestamps=True)
    return _to_jsonable(result)


def _to_jsonable(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if hasattr(payload, "dict"):
        return payload.dict()
    return json.loads(json.dumps(payload, default=str))


def _build_srt_entries(transcript_payload: dict[str, Any]) -> list[dict[str, Any]]:
    words = transcript_payload.get("words", []) or []
    if words:
        return _entries_from_words(words)

    segments = transcript_payload.get("segments", []) or []
    entries = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 1.0))
        text = str(seg.get("text", "")).strip()
        if text:
            entries.append({"start": start, "end": max(end, start + 0.5), "text": text})
    return entries


def _entries_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    chunk: list[str] = []
    start = None
    end = None
    max_words = 6
    max_span = 2.2

    for word_item in words:
        text = str(word_item.get("word", "")).strip()
        if not text:
            continue
        word_start = float(word_item.get("start", 0.0))
        word_end = float(word_item.get("end", word_start + 0.2))
        if start is None:
            start = word_start
        chunk.append(text)
        end = word_end

        span = end - start
        if len(chunk) >= max_words or span >= max_span:
            entries.append({"start": start, "end": end, "text": " ".join(chunk)})
            chunk = []
            start = None
            end = None

    if chunk and start is not None and end is not None:
        entries.append({"start": start, "end": end, "text": " ".join(chunk)})
    return entries


def _cleanup_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for item in entries:
        text = _clean_caption_text(str(item.get("text", "")).strip())
        if not text:
            continue
        cleaned.append(
            {
                "start": float(item.get("start", 0.0)),
                "end": float(item.get("end", 0.0)),
                "text": text,
            }
        )
    return cleaned


def _prepend_hook_entry(
    entries: list[dict[str, Any]],
    hook_text: str,
    hook_duration_sec: float,
) -> list[dict[str, Any]]:
    hook = _clean_caption_text(hook_text)
    if not hook:
        return entries
    safe_duration = max(0.8, min(3.0, float(hook_duration_sec)))
    if entries:
        first_start = float(entries[0].get("start", 0.0))
        allowed = first_start - 0.05
        if allowed < 0.8:
            return entries
        safe_duration = min(safe_duration, allowed)

    hook_entry = {"start": 0.0, "end": round(safe_duration, 3), "text": hook}
    return [hook_entry, *entries]


def _clean_caption_text(text: str) -> str:
    value = " ".join(text.replace("\n", " ").split())
    for pattern, replacement in MLBB_GLOSSARY.items():
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return value.strip()


def _caption_style_profile(caption_template: str | None, fallback_margin: int) -> dict[str, int]:
    base = {
        "font_size": 10,
        "outline": 3,
        "margin_v": max(20, int(fallback_margin)),
    }
    template = (caption_template or "").strip().upper()
    if template in {TEMPLATE_TALK_HOTTAKE, TEMPLATE_DONATION_QNA}:
        base["font_size"] = 12
        base["outline"] = 4
        base["margin_v"] = max(24, int(fallback_margin))
    elif template == TEMPLATE_COMEDY_REACTION:
        base["font_size"] = 11
        base["outline"] = 4
    elif template == TEMPLATE_MULTICAM:
        base["font_size"] = 11
        base["margin_v"] = max(26, int(fallback_margin))
    elif template in {TEMPLATE_GAMEPLAY_EPIC, TEMPLATE_PRO_ENCOUNTER}:
        base["font_size"] = 10
        base["outline"] = 3
    return base


def prepend_hook_to_srt_file(
    srt_path: Path,
    hook_text: str,
    hook_duration_sec: float,
) -> int:
    if not srt_path.exists():
        return 0
    existing = _load_srt_entries(srt_path)
    if not existing:
        return 0
    updated = _prepend_hook_entry(
        entries=existing,
        hook_text=hook_text,
        hook_duration_sec=hook_duration_sec,
    )
    if len(updated) <= len(existing):
        return 0
    with srt_path.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(updated, 1):
            handle.write(f"{idx}\n")
            handle.write(
                f"{seconds_to_srt(item['start'])} --> {seconds_to_srt(item['end'])}\n"
            )
            handle.write(f"{item['text']}\n\n")
    return 1


def _load_srt_entries(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\r?\n\r?\n", raw.strip())
    entries: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_line = lines[1] if "-->" in lines[1] else lines[0]
        if "-->" not in time_line:
            continue
        start_raw, end_raw = [item.strip() for item in time_line.split("-->", 1)]
        start = _srt_to_seconds(start_raw)
        end = _srt_to_seconds(end_raw)
        if end <= start:
            continue
        text_lines = lines[2:] if "-->" in lines[1] else lines[1:]
        text = _clean_caption_text(" ".join(text_lines))
        if not text:
            continue
        entries.append({"start": start, "end": end, "text": text})
    return entries


def _srt_to_seconds(value: str) -> float:
    match = re.match(r"^(\d+):(\d+):(\d+),(\d+)$", value.strip())
    if not match:
        return 0.0
    h, m, s, ms = [int(item) for item in match.groups()]
    return float(h * 3600 + m * 60 + s + (ms / 1000.0))
