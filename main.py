from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mlbb_pipeline.captions import (
    burn_srt_captions,
    prepend_hook_to_srt_file,
    transcribe_to_srt,
)
from mlbb_pipeline.common import (
    load_json,
    parse_youtube_video_id,
    run_command,
    save_json,
    setup_logger,
)
from mlbb_pipeline.compose_tiktok import compose_tiktok_clip, cut_clip
from mlbb_pipeline.detect_audio import detect_audio_peaks
from mlbb_pipeline.detect_motion import detect_motion_intensity
from mlbb_pipeline.detect_ocr import detect_ocr_hits
from mlbb_pipeline.facecam_locate import locate_facecam_roi
from mlbb_pipeline.ingest import run_ingest
from mlbb_pipeline.rank_clips import build_candidates, llm_rank_and_refine
from mlbb_pipeline.qc_gates import (
    choose_template_with_qc,
    evaluate_audio_levels,
    evaluate_layout_safety,
    evaluate_srt_quality,
)
from mlbb_pipeline.template_selector import (
    build_hook_text,
    classify_edit_template,
    extract_transcript_text,
    hook_duration_for_template,
)
from mlbb_pipeline.upload_tiktok import upload_video_to_tiktok


SENSITIVE_KEYS = {
    "sumopod_api_key",
    "tiktok_client_key",
    "tiktok_client_secret",
    "tiktok_access_token",
    "tiktok_refresh_token",
}


@dataclass
class PipelineSettings:
    sumopod_api_key: str
    sumopod_base_url: str
    llm_model: str
    whisper_model: str
    whisper_api_enabled: bool
    whisper_local_fallback: bool
    event_image_path: str
    top_banner_px: int
    bot_banner_px: int
    gameplay_height_px: int
    face_height_px: int
    gameplay_ratio: float
    face_roi_scan_seconds: int
    face_mode_auto: bool
    multicam_min_faces: int
    multicam_area_ratio_max: float
    multicam_yvar_max: float
    ocr_lang: str
    ocr_sample_fps: float
    ocr_auto_scan_seconds: int
    clip_count: int
    clip_min_sec: int
    clip_max_sec: int
    hashtags: str
    mention: str
    tiktok_upload_enabled: bool
    tiktok_mode: str
    tiktok_client_key: str
    tiktok_client_secret: str
    tiktok_access_token: str
    tiktok_refresh_token: str
    tiktok_token_expires_at: int
    tiktok_auto_auth: bool
    tiktok_privacy_level: str
    tiktok_disable_duet: bool
    tiktok_disable_comment: bool
    tiktok_disable_stitch: bool
    output_root: str
    ffmpeg_bin: str
    yt_dlp_bin: str
    cookies_path: str
    ocr_roi_announcement: str
    ocr_roi_kill_feed: str


def main() -> int:
    args = _parse_args()
    load_dotenv()
    settings = _load_settings(args)

    now = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    url_video_id = parse_youtube_video_id(args.url) or f"video_{now}"
    run_dir = Path(settings.output_root) / url_video_id
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / "pipeline.log")
    logger.info("MLBB pipeline started for URL: %s", args.url)
    logger.info("Output directory: %s", run_dir.resolve())

    run_config = {
        "run_started_at_utc": datetime.utcnow().isoformat(),
        "url": args.url,
        "platform": args.platform,
        "settings": _redact_settings(asdict(settings)),
    }
    save_json(run_dir / "run_config.json", run_config)

    ocr_roi = _build_ocr_roi_map(settings, logger)

    try:
        ingest_result = run_ingest(
            url=args.url,
            run_dir=run_dir,
            yt_dlp_bin=settings.yt_dlp_bin,
            ffmpeg_bin=settings.ffmpeg_bin,
            cookies_path=settings.cookies_path or None,
            logger=logger,
        )
        run_config["ingest"] = ingest_result
        save_json(run_dir / "run_config.json", run_config)

        audio_scores = detect_audio_peaks(
            audio_path=Path(ingest_result["audio_path"]),
            frame_sec=1.0,
            hop_sec=0.5,
            logger=logger,
        )
        motion_scores = detect_motion_intensity(
            proxy_video_path=Path(ingest_result["proxy_path"]),
            sample_fps=max(1.0, settings.ocr_sample_fps),
            logger=logger,
        )
        ocr_result = detect_ocr_hits(
            proxy_video_path=Path(ingest_result["proxy_path"]),
            ocr_lang=settings.ocr_lang,
            sample_fps=settings.ocr_sample_fps,
            roi_presets=ocr_roi,
            auto_roi_scan_seconds=settings.ocr_auto_scan_seconds,
            logger=logger,
        )

        candidates = build_candidates(
            duration_sec=float(ingest_result.get("duration_sec", 0.0)),
            audio_scores=audio_scores,
            ocr_scores=ocr_result["timeline"],
            motion_scores=motion_scores,
            clip_min_sec=settings.clip_min_sec,
            clip_max_sec=settings.clip_max_sec,
            max_candidates=50,
        )
        save_json(run_dir / "candidates.json", candidates)

        selected = llm_rank_and_refine(
            candidates=candidates,
            clip_count=settings.clip_count,
            clip_min_sec=settings.clip_min_sec,
            clip_max_sec=settings.clip_max_sec,
            llm_model=settings.llm_model,
            api_key=settings.sumopod_api_key,
            base_url=settings.sumopod_base_url,
            logger=logger,
        )
        save_json(run_dir / "selected_clips.json", selected)

        facecam_result = locate_facecam_roi(
            video_path=Path(ingest_result["source_path"]),
            scan_seconds=settings.face_roi_scan_seconds,
            sample_fps=1.0,
            face_mode_auto=settings.face_mode_auto,
            multicam_min_faces=settings.multicam_min_faces,
            multicam_area_ratio_max=settings.multicam_area_ratio_max,
            multicam_yvar_max=settings.multicam_yvar_max,
            logger=logger,
        )
        run_config["facecam_locator"] = facecam_result
        run_config["ocr"] = {
            "rois": ocr_result.get("rois", {}),
            "events_count": len(ocr_result.get("events", [])),
        }
        save_json(run_dir / "run_config.json", run_config)

        clip_outputs = _process_clips(
            selected=selected,
            source_path=Path(ingest_result["source_path"]),
            clips_dir=clips_dir,
            settings=settings,
            facecam_result=facecam_result,
            logger=logger,
            platform=args.platform,
            run_dir=run_dir,
        )

        transcript_index = {
            "generated_at_utc": datetime.utcnow().isoformat(),
            "clips": [
                {
                    "clip_id": item["clip_id"],
                    "transcript_json": item.get("transcript_json"),
                    "srt_path": item.get("srt_path"),
                }
                for item in clip_outputs
            ],
        }
        save_json(run_dir / "transcript.json", transcript_index)
        save_json(run_dir / "selected_clips.json", clip_outputs)

        run_config["run_finished_at_utc"] = datetime.utcnow().isoformat()
        run_config["generated_clips"] = len(clip_outputs)
        save_json(run_dir / "run_config.json", run_config)

        logger.info("Pipeline completed successfully. Generated clips: %d", len(clip_outputs))
        logger.info("Run folder: %s", run_dir)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed: %s", exc)
        run_config["run_finished_at_utc"] = datetime.utcnow().isoformat()
        run_config["error"] = str(exc)
        save_json(run_dir / "run_config.json", run_config)
        return 1


def _process_clips(
    selected: list[dict[str, Any]],
    source_path: Path,
    clips_dir: Path,
    settings: PipelineSettings,
    facecam_result: dict[str, Any],
    logger,
    platform: str,
    run_dir: Path,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    face_mode = str(facecam_result.get("mode", "NO_FACE")).upper()
    face_roi = facecam_result.get("crop_window_norm") or facecam_result.get("roi_norm")
    if face_mode == "NO_FACE":
        face_roi = None

    for idx, clip in enumerate(selected, 1):
        clip_id = f"clip_{idx:02d}"
        logger.info("Processing %s (%s - %s)", clip_id, clip["start_sec"], clip["end_sec"])

        raw_path = clips_dir / f"{clip_id}_raw.mp4"
        composed_path = clips_dir / f"{clip_id}_composed.mp4"
        final_path = clips_dir / f"{clip_id}_final.mp4"
        srt_path = clips_dir / f"{clip_id}_final.srt"
        transcript_json_path = clips_dir / f"{clip_id}_transcript.json"
        stt_audio_path = clips_dir / f"{clip_id}_stt.mp3"

        cut_clip(
            source_path=source_path,
            output_path=raw_path,
            start_sec=float(clip["start_sec"]),
            end_sec=float(clip["end_sec"]),
            ffmpeg_bin=settings.ffmpeg_bin,
            logger=logger,
        )

        run_command(
            [
                settings.ffmpeg_bin,
                "-y",
                "-i",
                str(raw_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "64k",
                str(stt_audio_path),
            ],
            logger=logger,
        )

        caption_result = transcribe_to_srt(
            media_path=stt_audio_path,
            srt_path=srt_path,
            transcript_json_path=transcript_json_path,
            whisper_model=settings.whisper_model,
            api_key=settings.sumopod_api_key,
            base_url=settings.sumopod_base_url,
            use_api=settings.whisper_api_enabled,
            local_fallback=settings.whisper_local_fallback,
            logger=logger,
        )

        transcript_payload = load_json(transcript_json_path, default={})
        transcript_text = extract_transcript_text(transcript_payload)
        template_decision = classify_edit_template(
            clip=clip,
            facecam_mode=face_mode,
            transcript_text=transcript_text,
        )
        template_qc = choose_template_with_qc(
            requested_template=template_decision.template,
            facecam_mode=face_mode,
            facecam_found=bool(face_roi),
        )
        selected_template = template_qc["selected_template"]
        selected_hook_text = template_decision.hook_text
        selected_hook_duration = template_decision.hook_duration_sec
        if selected_template != template_decision.template:
            selected_hook_text = build_hook_text(
                template=selected_template,
                clip=clip,
                transcript_text=transcript_text,
            )
            selected_hook_duration = hook_duration_for_template(selected_template)

        hook_inserted = prepend_hook_to_srt_file(
            srt_path=srt_path,
            hook_text=selected_hook_text,
            hook_duration_sec=selected_hook_duration,
        )
        caption_result["hook_applied"] = bool(hook_inserted)
        srt_qc = evaluate_srt_quality(
            srt_path=srt_path,
            expected_template=selected_template,
        )

        layout_info = compose_tiktok_clip(
            raw_clip_path=raw_path,
            output_path=composed_path,
            ffmpeg_bin=settings.ffmpeg_bin,
            top_banner_px=settings.top_banner_px,
            bot_banner_px=settings.bot_banner_px,
            event_image_path=settings.event_image_path,
            facecam_roi_norm=face_roi,
            facecam_mode=face_mode,
            edit_template=selected_template,
            gameplay_ratio=settings.gameplay_ratio,
            importance_score=float(clip.get("combined_score", 0.0)),
            logger=logger,
        )
        layout_qc = evaluate_layout_safety(
            layout_info=layout_info,
            expected_template=selected_template,
        )

        if srt_qc.get("can_burn", True):
            burn_srt_captions(
                input_video_path=composed_path,
                output_video_path=final_path,
                srt_path=srt_path,
                ffmpeg_bin=settings.ffmpeg_bin,
                caption_template=selected_template,
                logger=logger,
            )
            caption_result["burn_mode"] = "subtitles"
        else:
            shutil.copy2(composed_path, final_path)
            caption_result["burn_mode"] = "fallback_no_subtitle"
            caption_result["burn_skipped_reason"] = ",".join(
                srt_qc.get("hard_failures", []) or ["subtitle_qc_failed"]
            )

        if composed_path.exists():
            composed_path.unlink()
        if stt_audio_path.exists():
            stt_audio_path.unlink()

        audio_qc = evaluate_audio_levels(
            media_path=final_path,
            ffmpeg_bin=settings.ffmpeg_bin,
            logger=logger,
        )
        hard_failures = (
            list(template_qc.get("hard_failures", []))
            + list(srt_qc.get("hard_failures", []))
            + list(layout_qc.get("hard_failures", []))
            + list(audio_qc.get("hard_failures", []))
        )
        qc_report = {
            "template_gate": template_qc,
            "subtitle_gate": srt_qc,
            "layout_gate": layout_qc,
            "audio_gate": audio_qc,
            "hard_failures": hard_failures,
            "warnings": (
                list(template_qc.get("warnings", []))
                + list(srt_qc.get("warnings", []))
                + list(layout_qc.get("warnings", []))
                + list(audio_qc.get("warnings", []))
            ),
        }
        if hard_failures:
            logger.warning("%s QC hard failures: %s", clip_id, hard_failures)

        hook_text = selected_hook_text or str(clip.get("hook_text", "")).strip()
        if not hook_text:
            hook_text = f"Epic MLBB Moment #{idx}"
        upload_result: dict[str, Any] = {"skipped": True}
        if settings.tiktok_upload_enabled and platform.lower() == "tiktok":
            upload_result = upload_video_to_tiktok(
                video_path=final_path,
                hook_text=hook_text,
                hashtags=settings.hashtags,
                mention=settings.mention,
                run_dir=run_dir,
                settings=asdict(settings),
                logger=logger,
            )

        output_item = {
            **clip,
            "clip_id": clip_id,
            "raw_path": str(raw_path),
            "final_path": str(final_path),
            "srt_path": str(srt_path),
            "transcript_json": str(transcript_json_path),
            "caption_result": caption_result,
            "layout": layout_info,
            "template_decision": {
                "template": selected_template,
                "requested_template": template_decision.template,
                "confidence": template_decision.confidence,
                "reasons": template_decision.reasons,
                "hook_text": selected_hook_text,
                "hook_duration_sec": selected_hook_duration,
            },
            "qc": qc_report,
            "upload_result": upload_result,
        }
        outputs.append(output_item)

    return outputs


def _load_settings(args: argparse.Namespace) -> PipelineSettings:
    clip_min = _env_int("CLIP_MIN_SEC", 20)
    clip_max = _env_int("CLIP_MAX_SEC", 45)
    if clip_min > clip_max:
        clip_min, clip_max = clip_max, clip_min

    top_banner = _env_int("TOP_BANNER_PX", 240)
    bot_banner = _env_int("BOT_BANNER_PX", 240)
    gameplay_height = _env_int("GAMEPLAY_HEIGHT_PX", 0)
    face_height = _env_int("FACE_HEIGHT_PX", 0)
    gameplay_ratio = _compute_gameplay_ratio(top_banner, bot_banner, gameplay_height, face_height)
    multicam_min_faces = max(2, _env_int("MULTICAM_MIN_FACES", 3))
    multicam_area_ratio_max = max(1.1, _env_float("MULTICAM_AREA_RATIO_MAX", 2.5))
    multicam_yvar_max = max(0.00001, _env_float("MULTICAM_YVAR_MAX", 0.0025))

    upload_enabled = _env_bool("TIKTOK_UPLOAD_ENABLED", False)
    if args.dry_run:
        upload_enabled = False

    return PipelineSettings(
        sumopod_api_key=os.getenv("SUMOPOD_API_KEY", "").strip(),
        sumopod_base_url=os.getenv("SUMOPOD_BASE_URL", "https://ai.sumopod.com/v1").strip(),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        whisper_model=os.getenv("WHISPER_MODEL", "whisper-1").strip(),
        whisper_api_enabled=_env_bool("WHISPER_API_ENABLED", True),
        whisper_local_fallback=_env_bool("WHISPER_LOCAL_FALLBACK", True),
        event_image_path=os.getenv("EVENT_IMAGE_PATH", "").strip(),
        top_banner_px=top_banner,
        bot_banner_px=bot_banner,
        gameplay_height_px=gameplay_height,
        face_height_px=face_height,
        gameplay_ratio=gameplay_ratio,
        face_roi_scan_seconds=_env_int("FACE_ROI_SCAN_SECONDS", 60),
        face_mode_auto=_env_bool("FACE_MODE_AUTO", True),
        multicam_min_faces=multicam_min_faces,
        multicam_area_ratio_max=multicam_area_ratio_max,
        multicam_yvar_max=multicam_yvar_max,
        ocr_lang=os.getenv("OCR_LANG", "eng").strip(),
        ocr_sample_fps=_env_float("OCR_SAMPLE_FPS", 1.0),
        ocr_auto_scan_seconds=_env_int("OCR_AUTO_SCAN_SECONDS", 60),
        clip_count=args.clip_count or _env_int("CLIP_COUNT", 5),
        clip_min_sec=clip_min,
        clip_max_sec=clip_max,
        hashtags=os.getenv(
            "HASHTAGS",
            "#MLBBGoldenMonth #MLBBGoldenTurtle #mlbb #mobilelegends",
        ).strip(),
        mention=os.getenv("TIKTOK_MENTION", "").strip(),
        tiktok_upload_enabled=upload_enabled,
        tiktok_mode=os.getenv("TIKTOK_MODE", "sandbox").strip() or "sandbox",
        tiktok_client_key=os.getenv("TIKTOK_CLIENT_KEY", "").strip(),
        tiktok_client_secret=os.getenv("TIKTOK_CLIENT_SECRET", "").strip(),
        tiktok_access_token=os.getenv("TIKTOK_ACCESS_TOKEN", "").strip(),
        tiktok_refresh_token=os.getenv("TIKTOK_REFRESH_TOKEN", "").strip(),
        tiktok_token_expires_at=_env_int("TIKTOK_TOKEN_EXPIRES_AT", 0),
        tiktok_auto_auth=_env_bool("TIKTOK_AUTO_AUTH", False),
        tiktok_privacy_level=os.getenv("TIKTOK_PRIVACY_LEVEL", "SELF_ONLY").strip(),
        tiktok_disable_duet=_env_bool("TIKTOK_DISABLE_DUET", False),
        tiktok_disable_comment=_env_bool("TIKTOK_DISABLE_COMMENT", False),
        tiktok_disable_stitch=_env_bool("TIKTOK_DISABLE_STITCH", False),
        output_root=os.getenv("OUTPUT_ROOT", "outputs").strip(),
        ffmpeg_bin=os.getenv("FFMPEG_BIN", "ffmpeg").strip(),
        yt_dlp_bin=os.getenv("YT_DLP_BIN", "yt-dlp").strip(),
        cookies_path=os.getenv("COOKIES_PATH", "cookies.txt").strip(),
        ocr_roi_announcement=os.getenv("OCR_ROI_ANNOUNCEMENT", "").strip(),
        ocr_roi_kill_feed=os.getenv("OCR_ROI_KILL_FEED", "").strip(),
    )


def _build_ocr_roi_map(
    settings: PipelineSettings,
    logger,
) -> dict[str, tuple[float, float, float, float]] | None:
    announce = _parse_roi(settings.ocr_roi_announcement)
    feed = _parse_roi(settings.ocr_roi_kill_feed)
    if announce and feed:
        roi = {"announcement": announce, "kill_feed": feed}
        logger.info("Using OCR ROI presets from env: %s", roi)
        return roi
    if settings.ocr_roi_announcement or settings.ocr_roi_kill_feed:
        logger.warning(
            "OCR ROI env provided but invalid format. Expected x1,y1,x2,y2 with normalized values."
        )
    return None


def _parse_roi(raw: str) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        return None
    try:
        values = tuple(float(p) for p in parts)
    except ValueError:
        return None
    if not all(0.0 <= value <= 1.0 for value in values):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def _compute_gameplay_ratio(
    top_banner: int,
    bot_banner: int,
    gameplay_height: int,
    face_height: int,
) -> float:
    middle = 1920 - max(0, top_banner) - max(0, bot_banner)
    if middle <= 0:
        return 0.7
    if gameplay_height > 0 and face_height > 0:
        total = gameplay_height + face_height
        if total > 0:
            return gameplay_height / total
    if gameplay_height > 0:
        return min(0.9, max(0.55, gameplay_height / middle))
    return 0.7


def _redact_settings(data: dict[str, Any]) -> dict[str, Any]:
    clean = dict(data)
    for key in SENSITIVE_KEYS:
        if clean.get(key):
            clean[key] = "***REDACTED***"
    return clean


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLBB -> TikTok auto clipper pipeline")
    parser.add_argument("--url", required=True, help="YouTube VOD URL")
    parser.add_argument("--platform", default="tiktok", help="Output platform (default: tiktok)")
    parser.add_argument(
        "--clip-count",
        type=int,
        default=0,
        help="Override CLIP_COUNT from env",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full pipeline but disable TikTok upload",
    )
    return parser.parse_args()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


if __name__ == "__main__":
    sys.exit(main())
