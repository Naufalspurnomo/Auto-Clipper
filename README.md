# Auto-Clipper

Auto-Clipper is an AI-assisted video editing pipeline for turning long YouTube VODs into short vertical content (TikTok/Reels/Shorts), with strong support for MLBB-style gaming streams.

This repo currently has two modes:
- `CLI pipeline` (recommended for automation): `main.py`
- `Desktop GUI` (legacy/general workflow): `app.py`

## Why this project

Most clippers only crop and trim. Auto-Clipper is template-driven:
- classifies clip type (gameplay, talk, comedy, multicam, donation, pro encounter)
- applies matching composition template
- generates hook packaging and cleaner captions
- runs QC gates with fallback to reduce broken output

## Core Features

- Automated ingest:
  - Download source with `yt-dlp`
  - Generate proxy and analysis audio
- Candidate detection:
  - Audio intensity (`librosa`)
  - OCR events (`pytesseract`)
  - Motion spikes (`OpenCV`)
- LLM ranking/refinement:
  - Select best moments and refine clip windows
- Facecam intelligence:
  - `MULTI_CAM_STRIP`, `SINGLE_CAM`, `NO_FACE`
  - bottom-band detection + smoothing/hold for stability
- Template-driven editing:
  - `GAMEPLAY_EPIC`
  - `TALK_HOTTAKE`
  - `COMEDY_REACTION`
  - `MULTICAM`
  - `DONATION_QNA`
  - `PRO_ENCOUNTER`
- Adaptive composition:
  - gameplay/face balance by template
  - multicam strip-safe behavior
  - `NO_FACE` fallback with blurred panel
  - subtle importance-based zoom
- Caption packaging:
  - Whisper transcription
  - MLBB glossary cleanup (reduces bad terms like "cepek")
  - hook subtitle at opening
  - style profile by template
- QC gates:
  - template gate
  - subtitle readability gate
  - layout safety gate
  - audio level gate
  - automatic fallback when needed
- Optional TikTok upload support

## Project Structure

```text
.
|- main.py                      # CLI pipeline entrypoint
|- app.py                       # Desktop GUI entrypoint (legacy mode)
|- mlbb_pipeline/
|  |- ingest.py                 # Download, proxy, audio extract
|  |- detect_audio.py           # Audio scoring
|  |- detect_motion.py          # Motion scoring
|  |- detect_ocr.py             # OCR scoring/events
|  |- rank_clips.py             # Candidate fusion + LLM ranking
|  |- facecam_locate.py         # Facecam mode + crop window detection
|  |- template_selector.py      # Clip template classifier + hook logic
|  |- compose_tiktok.py         # Template-aware 1080x1920 composition
|  |- captions.py               # Transcribe, cleanup, subtitle burn
|  |- qc_gates.py               # Quality checks and fallback decisions
|  |- upload_tiktok.py          # TikTok upload integration
|  |- common.py                 # Shared helpers
|- scripts/
|  |- bootstrap_mlbb.sh
|  `- bootstrap_mlbb_windows.ps1
|- .env.example
|- requirements.txt
`- outputs/                     # Generated after runs
```

## Requirements

- Python `3.10+`
- `ffmpeg` in PATH
- `yt-dlp` in PATH
- `tesseract` installed (for OCR)
- API key for LLM/Whisper backend (configured via `.env`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start (CLI Pipeline)

1. Copy env:

```bash
cp .env.example .env
```

2. Fill required values in `.env`:
- `SUMOPOD_API_KEY`
- `EVENT_IMAGE_PATH` (optional but recommended)

3. Run bootstrap (optional helper):

Linux/macOS:

```bash
bash scripts/bootstrap_mlbb.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_mlbb_windows.ps1
```

4. Run pipeline:

Dry-run (no upload):

```bash
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --platform tiktok --dry-run
```

Normal run:

```bash
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --platform tiktok
```

## Important `.env` Settings

Template and facecam behavior:
- `FACE_MODE_AUTO=1`
- `MULTICAM_MIN_FACES=3`
- `MULTICAM_AREA_RATIO_MAX=2.5`
- `MULTICAM_YVAR_MAX=0.0025`
- `FACE_ROI_SCAN_SECONDS=60`

Layout:
- `TOP_BANNER_PX=240`
- `BOT_BANNER_PX=240`
- `GAMEPLAY_HEIGHT_PX=0`
- `FACE_HEIGHT_PX=0`

Detection:
- `OCR_LANG=eng`
- `OCR_SAMPLE_FPS=1.0`
- `OCR_AUTO_SCAN_SECONDS=60`

Selection:
- `CLIP_COUNT=5`
- `CLIP_MIN_SEC=20`
- `CLIP_MAX_SEC=45`

## Output

Each run creates:

```text
outputs/<video_id>/
  source.mp4
  proxy.mp4
  audio.wav
  candidates.json
  selected_clips.json
  transcript.json
  run_config.json
  clips/
    clip_01_raw.mp4
    clip_01_final.srt
    clip_01_final.mp4
    ...
```

`selected_clips.json` includes:
- chosen template per clip
- hook text/duration
- layout metadata
- QC report (`template`, `subtitle`, `layout`, `audio`)

## Desktop GUI Mode

If you want the GUI workflow:

```bash
python app.py
```

## Troubleshooting

- `ffmpeg not found`:
  - install ffmpeg and ensure it is in PATH
- `yt-dlp not found`:
  - install `yt-dlp` and ensure it is in PATH
- OCR not detecting events:
  - verify `tesseract` is installed
  - tune OCR ROI in `.env`
- Bad/missing auth for upload:
  - check `TIKTOK_*` variables
- Python launcher issues on Windows:
  - run with explicit interpreter from active environment

## Roadmap

- Better game boundary detection (match segmentation)
- Richer template-specific motion presets
- Automatic post-QC retry strategy
- Expanded OCR dictionaries and language packs

## License

MIT License. See [LICENSE](LICENSE).
