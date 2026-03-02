from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mlbb_pipeline.common import ffprobe_video, run_command


def run_ingest(
    url: str,
    run_dir: Path,
    yt_dlp_bin: str,
    ffmpeg_bin: str,
    cookies_path: str | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Download source video and prepare proxy + analysis audio."""
    run_dir.mkdir(parents=True, exist_ok=True)

    source_path = run_dir / "source.mp4"
    proxy_path = run_dir / "proxy.mp4"
    audio_path = run_dir / "audio.wav"
    info_json_path = run_dir / "source.info.json"

    download_video(
        url=url,
        output_path=source_path,
        info_json_path=info_json_path,
        yt_dlp_bin=yt_dlp_bin,
        cookies_path=cookies_path,
        logger=logger,
    )
    generate_proxy(
        source_path=source_path,
        proxy_path=proxy_path,
        ffmpeg_bin=ffmpeg_bin,
        logger=logger,
    )
    extract_audio_wav(
        source_path=source_path,
        audio_path=audio_path,
        ffmpeg_bin=ffmpeg_bin,
        logger=logger,
    )

    probe = ffprobe_video(source_path, logger)
    info_payload = {}
    if info_json_path.exists():
        with info_json_path.open("r", encoding="utf-8") as handle:
            info_payload = json.load(handle)

    return {
        "source_path": str(source_path),
        "proxy_path": str(proxy_path),
        "audio_path": str(audio_path),
        "video_info_path": str(info_json_path),
        "video_id": info_payload.get("id", ""),
        "title": info_payload.get("title", ""),
        "duration_sec": probe.get("duration_sec", 0.0),
        "resolution": {
            "width": probe.get("width", 0),
            "height": probe.get("height", 0),
            "fps": probe.get("fps", 0.0),
        },
    }


def download_video(
    url: str,
    output_path: Path,
    info_json_path: Path,
    yt_dlp_bin: str,
    cookies_path: str | None,
    logger: logging.Logger,
) -> None:
    """Download a YouTube VOD as source.mp4 with yt-dlp."""
    output_tpl = output_path.with_suffix(".%(ext)s")
    command = [
        yt_dlp_bin,
        "--no-playlist",
        "-f",
        "bestvideo+bestaudio/best",
        "--merge-output-format",
        "mp4",
        "--write-info-json",
        "--output",
        str(output_tpl),
        url,
    ]
    if cookies_path:
        command[1:1] = ["--cookies", cookies_path]

    # Keep execution from the project working dir so relative output template
    # resolves to the intended run directory path.
    run_command(command, logger=logger)

    if not output_path.exists():
        downloaded = list(output_path.parent.glob(f"{output_path.stem}.*"))
        mp4_files = [p for p in downloaded if p.suffix.lower() == ".mp4"]
        if mp4_files:
            mp4_files[0].replace(output_path)
    if not output_path.exists():
        raise FileNotFoundError(
            f"Download succeeded but source video not found at: {output_path}"
        )

    generated_info = output_path.with_suffix(".info.json")
    if generated_info.exists() and generated_info != info_json_path:
        generated_info.replace(info_json_path)


def generate_proxy(
    source_path: Path,
    proxy_path: Path,
    ffmpeg_bin: str,
    logger: logging.Logger,
) -> None:
    """Generate 720p-ish proxy for faster CV/OCR analysis."""
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(source_path),
        "-vf",
        "scale=1280:720:force_original_aspect_ratio=decrease",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-an",
        str(proxy_path),
    ]
    run_command(command, logger=logger)


def extract_audio_wav(
    source_path: Path,
    audio_path: Path,
    ffmpeg_bin: str,
    logger: logging.Logger,
) -> None:
    """Extract mono 16k PCM WAV for Whisper and audio scoring."""
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]
    run_command(command, logger=logger)
