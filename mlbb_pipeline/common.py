from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any


def setup_logger(log_path: Path) -> logging.Logger:
    """Create a file + console logger for pipeline runs."""
    logger = logging.getLogger("mlbb_pipeline")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def run_command(
    command: list[str],
    logger: logging.Logger,
    cwd: Path | None = None,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command and raise with stderr context on failure."""
    logger.info("Running command: %s", " ".join(command))
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0 and not allow_failure:
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        detail = stderr or stdout or "Unknown process failure"
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n{detail}"
        )
    return result


def parse_youtube_video_id(url: str) -> str | None:
    """Extract YouTube video ID from common URL patterns."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:[?&].*)?$",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})(?:[?&].*)?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm for logs/json."""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = seconds - (hours * 3600 + minutes * 60)
    whole = int(remaining)
    millis = int(round((remaining - whole) * 1000))
    if millis == 1000:
        millis = 0
        whole += 1
    return f"{hours:02d}:{minutes:02d}:{whole:02d}.{millis:03d}"


def seconds_to_srt(seconds: float) -> str:
    """Convert seconds to SRT clock format HH:MM:SS,mmm."""
    return seconds_to_timestamp(seconds).replace(".", ",")


def ffprobe_video(video_path: Path, logger: logging.Logger) -> dict[str, Any]:
    """Read width/height/fps/duration using ffprobe."""
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ],
        logger=logger,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    format_info = payload.get("format", {})
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    width = int(video_stream.get("width", 0) or 0)
    height = int(video_stream.get("height", 0) or 0)
    fps = _parse_fraction(video_stream.get("avg_frame_rate", "0/1"))
    duration = float(video_stream.get("duration") or format_info.get("duration") or 0.0)
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": duration,
    }


def _parse_fraction(value: str) -> float:
    if not value or "/" not in value:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    num, den = value.split("/", 1)
    try:
        denominator = float(den)
        if denominator == 0:
            return 0.0
        return float(num) / denominator
    except ValueError:
        return 0.0

