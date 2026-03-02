from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mlbb_pipeline.common import load_json, save_json
from tiktok_uploader import TikTokUploader


class _RuntimeConfigAdapter:
    """Small config adapter so existing TikTokUploader can run in CLI mode."""

    def __init__(self, file_path: Path, initial_tiktok_config: dict[str, Any]) -> None:
        self.file_path = file_path
        self.config = load_json(file_path, default={})
        merged = {**self.config.get("tiktok", {}), **initial_tiktok_config}
        self.config["tiktok"] = merged
        self._persist()

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.config[key] = value
        self._persist()

    def _persist(self) -> None:
        save_json(self.file_path, self.config)


def upload_video_to_tiktok(
    video_path: Path,
    hook_text: str,
    hashtags: str,
    mention: str,
    run_dir: Path,
    settings: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Upload a final clip using existing TikTok uploader implementation."""
    config_adapter = _RuntimeConfigAdapter(
        file_path=run_dir / "tiktok_runtime_config.json",
        initial_tiktok_config={
            "client_key": settings.get("tiktok_client_key", ""),
            "client_secret": settings.get("tiktok_client_secret", ""),
            "mode": settings.get("tiktok_mode", "sandbox"),
            "access_token": settings.get("tiktok_access_token", ""),
            "refresh_token": settings.get("tiktok_refresh_token", ""),
            "token_expires_at": settings.get("tiktok_token_expires_at", 0),
        },
    )

    uploader = TikTokUploader(
        config=config_adapter,
        status_callback=lambda msg: logger.info("[TikTok] %s", msg),
    )

    if not uploader.is_configured():
        return {
            "success": False,
            "error": "TikTok credentials are missing (TIKTOK_CLIENT_KEY/SECRET).",
        }

    if not uploader.is_authenticated():
        if settings.get("tiktok_auto_auth", False):
            try:
                uploader.authenticate()
            except Exception as exc:  # noqa: BLE001
                return {"success": False, "error": f"TikTok auth failed: {exc}"}
        else:
            return {
                "success": False,
                "error": (
                    "TikTok is not authenticated. Set TIKTOK_ACCESS_TOKEN or enable "
                    "TIKTOK_AUTO_AUTH=1 to run OAuth in browser."
                ),
            }

    title = _build_caption(hook_text=hook_text, hashtags=hashtags, mention=mention)
    title = title[:150]

    result = uploader.upload_video(
        video_path=str(video_path),
        title=title,
        description="",
        privacy_level=settings.get("tiktok_privacy_level", "SELF_ONLY"),
        disable_duet=bool(settings.get("tiktok_disable_duet", False)),
        disable_comment=bool(settings.get("tiktok_disable_comment", False)),
        disable_stitch=bool(settings.get("tiktok_disable_stitch", False)),
    )
    return result


def _build_caption(hook_text: str, hashtags: str, mention: str) -> str:
    parts = [hook_text.strip(), hashtags.strip(), mention.strip()]
    return " ".join([part for part in parts if part]).strip()

