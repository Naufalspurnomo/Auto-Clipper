# YT-Short-Clipper

[![Discord](https://img.shields.io/badge/Join-Discord-5865F2?logo=discord&logoColor=white)](https://s.id/ytsdiscord)
[![GitHub Stars](https://img.shields.io/github/stars/jipraks/yt-short-clipper?style=social)](https://github.com/jipraks/yt-short-clipper)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ðŸŽ¬ **Automated YouTube to Short-Form Content Pipeline**

Transform long-form YouTube videos (podcasts, interviews, vlogs) into engaging short-form content for TikTok, Instagram Reels, and YouTube Shorts â€” powered by AI.

---

## ï¿½ Getting Started

### For Users (Non-Technical)

Download the desktop app and follow the complete setup guide:

- ðŸ“– **[English Guide](GUIDE.md)** - Complete setup guide with screenshots
- ðŸ“– **[Panduan Indonesia](PANDUAN.md)** - Panduan lengkap dengan screenshot

**What you'll learn:**
1. How to download and run the app
2. Setup required libraries (yt-dlp, FFmpeg, Deno)
3. Setup YouTube cookies for video access
4. Configure AI API (multiple providers supported)
5. Start processing videos

### For Developers

If you want to contribute or run from source:

1. See [Installation](#-installation-for-development) below for development setup
2. See [Contributing](#-contributing) for contribution guidelines
3. See [BUILD.md](BUILD.md) for building the desktop app from source

## âœ¨ Features

- **ðŸŽ¥ Auto Download** - Downloads YouTube videos with Indonesian subtitles using yt-dlp
- **ðŸ” AI Highlight Detection** - Uses GPT-4 to identify the most engaging segments (60-120 seconds)
- **âœ‚ï¸ Smart Clipping** - Automatically cuts video at optimal timestamps
- **ðŸ“± Portrait Conversion** - Converts landscape (16:9) to portrait (9:16) with intelligent speaker tracking
- **ðŸŽ¯ Face Detection** - Two modes available:
  - **OpenCV (Fast)** - Crops to largest face, faster processing
  - **MediaPipe (Smart)** - Tracks active speaker via lip movement detection, more accurate but 2-3x slower
- **ðŸª Hook Generation** - Creates attention-grabbing intro scenes with AI-generated text and TTS voiceover
- **ðŸ“ Auto Captions** - Adds CapCut-style word-by-word highlighted captions using Whisper
- **ðŸ–¼ï¸ Watermark Support** - Add custom watermark with adjustable position, size, and opacity
- **ðŸ“Š SEO Metadata** - Generates optimized titles and descriptions for each clip

## ðŸ—ï¸ Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                        YT-Short-Clipper                         â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                 â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”           â”‚
â”‚  â”‚ YouTube  â”‚â”€â”€â”€â–¶â”‚  Downloader  â”‚â”€â”€â”€â–¶â”‚  Subtitle   â”‚           â”‚
â”‚  â”‚   URL    â”‚    â”‚   (yt-dlp)   â”‚    â”‚   Parser    â”‚           â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜           â”‚
â”‚                                              â”‚                  â”‚
â”‚                                              â–¼                  â”‚
â”‚                                    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”         â”‚
â”‚                                    â”‚ Highlight Finderâ”‚         â”‚
â”‚                                    â”‚    (GPT-4)      â”‚         â”‚
â”‚                                    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜         â”‚
â”‚                                              â”‚                  â”‚
â”‚                                              â–¼                  â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚  â”‚                    Video Processing                       â”‚  â”‚
â”‚  â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚  â”‚
â”‚  â”‚  â”‚   Clipper  â”‚â”€â–¶â”‚  Portrait  â”‚â”€â–¶â”‚  Hook Generator    â”‚  â”‚  â”‚
â”‚  â”‚  â”‚  (FFmpeg)  â”‚  â”‚ Converter  â”‚  â”‚  (TTS + Overlay)   â”‚  â”‚  â”‚
â”‚  â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚  â”‚
â”‚  â”‚                                              â”‚            â”‚  â”‚
â”‚  â”‚                                              â–¼            â”‚  â”‚
â”‚  â”‚                                    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”‚  â”‚
â”‚  â”‚                                    â”‚Caption Generatorâ”‚    â”‚  â”‚
â”‚  â”‚                                    â”‚   (Whisper)    â”‚     â”‚  â”‚
â”‚  â”‚                                    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â”‚  â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â”‚                                              â”‚                  â”‚
â”‚                                              â–¼                  â”‚
â”‚                                    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”         â”‚
â”‚                                    â”‚  Output Clips   â”‚         â”‚
â”‚                                    â”‚  + Metadata     â”‚         â”‚
â”‚                                    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜         â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## ðŸ“‹ Requirements (For Development)

### System Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Runtime |
| FFmpeg | 4.4+ | Video processing |
| yt-dlp | Latest | YouTube downloading |

### Python Dependencies

```
customtkinter>=5.2.0
openai>=1.0.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
google-api-python-client>=2.100.0
google-auth-oauthlib>=1.1.0
google-auth-httplib2>=0.1.1
```

> **Note:** The app uses OpenAI Whisper API instead of local Whisper model.

### API Keys

The app supports **10+ AI providers** including:
- **YT Clip AI** (Recommended) - [https://ai.ytclip.org](https://ai.ytclip.org)
- **OpenAI** - GPT-4, Whisper, TTS
- **Google Gemini** - Free tier available
- **Groq** - Fastest + free
- **Anthropic Claude** - High quality
- And more...

See [GUIDE.md](GUIDE.md) or [PANDUAN.md](PANDUAN.md) for detailed API setup instructions.

---

## ðŸš€ Installation (For Development)

> **Note:** This section is for developers who want to run the app from source code. If you're a regular user, please follow the [User Guide](GUIDE.md) or [Panduan Indonesia](PANDUAN.md) instead.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/yt-short-clipper.git
cd yt-short-clipper
```

### 2. Install System Dependencies

**Windows (using Chocolatey):**
```powershell
choco install ffmpeg yt-dlp
```

**Windows (one-step bootstrap for PowerShell/Git Bash):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_mlbb_windows.ps1
```

**macOS (using Homebrew):**
```bash
brew install ffmpeg yt-dlp
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
pip install yt-dlp
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python app.py
```

The app will create a `config.json` file on first run where you can save your AI API keys and other settings.

---


## MLBB TikTok Mode

You can run the CLI pipeline to generate portrait MLBB clips with:
- candidate detection from `audio + OCR + motion`
- LLM ranking/refine
- auto gameplay/facecam split composition
- Whisper captions
- optional TikTok upload

### Quick Setup

```bash
cp .env.example .env
bash scripts/bootstrap_mlbb.sh
```

Windows (PowerShell or Git Bash):

```bash
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_mlbb_windows.ps1
```

Set these values in `.env`:
- `SUMOPOD_API_KEY`
- `EVENT_IMAGE_PATH`
- optional `TIKTOK_*` credentials if upload enabled

### Run Commands

Dry-run (full processing, no upload):

```bash
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --platform tiktok --dry-run
```

Production (upload enabled from `.env`):

```bash
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --platform tiktok
```

### Output Structure

```text
outputs/<video_id>/
  source.mp4
  proxy.mp4
  audio.wav
  transcript.json
  candidates.json
  selected_clips.json
  run_config.json
  clips/
    clip_01_raw.mp4
    clip_01_final.mp4
    clip_01_final.srt
    ...
```

### OCR ROI Tuning

- Use `OCR_ROI_ANNOUNCEMENT` and `OCR_ROI_KILL_FEED` in normalized format `x1,y1,x2,y2`.
- Leave them empty to enable automatic top-HUD ROI scanning.

## ðŸ“ Project Structure

```
yt-short-clipper/
â”œâ”€â”€ app.py                      # Main GUI application
â”œâ”€â”€ clipper_core.py             # Core processing logic
â”œâ”€â”€ youtube_uploader.py         # YouTube upload functionality
â”œâ”€â”€ requirements.txt            # Python dependencies
â”œâ”€â”€ build.spec                  # PyInstaller build config
â”œâ”€â”€ config.json                 # App settings (auto-created)
â”œâ”€â”€ SYSTEM_PROMPT.md            # AI prompt customization guide
â”œâ”€â”€ BUILD.md                    # Build instructions
â”œâ”€â”€ DEBUG.md                    # Debugging guide
â”œâ”€â”€ assets/                     # App icons
â”‚   â”œâ”€â”€ icon.png
â”‚   â””â”€â”€ icon.ico
â””â”€â”€ output/                     # Output clips (auto-created)
    â”œâ”€â”€ _temp/                  # Temporary files
    â”‚   â”œâ”€â”€ source.mp4
    â”‚   â””â”€â”€ source.id.srt
    â””â”€â”€ 20240115-143001/        # Clip folder (timestamp-based)
        â”œâ”€â”€ master.mp4          # Final clip
        â””â”€â”€ data.json           # Metadata
```

### data.json Structure

Each clip folder contains a `data.json` file with metadata:

```json
{
  "title": "ðŸ”¥ Momen Kocak Saat Pembully Datang Minta Maaf",
  "hook_text": "Mantan pembully TIARA datang ke rumah minta endorse salad buah",
  "start_time": "00:15:23,000",
  "end_time": "00:17:05,000",
  "duration_seconds": 102.0,
  "has_hook": true,
  "has_captions": true,
  "youtube_title": "ðŸ”¥ Momen Kocak Saat Pembully Datang Minta Maaf",
  "youtube_description": "Siapa sangka mantan pembully malah datang minta endorse! ðŸ˜‚ #podcast #viral #fyp",
  "youtube_tags": ["shorts", "viral", "podcast"],
  "youtube_url": "https://youtube.com/watch?v=xxxxx",
  "youtube_video_id": "xxxxx"
}
```

---

## âš™ï¸ Configuration

All settings can be configured through the GUI Settings page (âš™ï¸ button in the app).

For complete setup instructions with screenshots, see:
- [English Guide](GUIDE.md#5-ai-api-configuration)
- [Panduan Indonesia](PANDUAN.md#5-konfigurasi-ai-api)

### AI Provider Selection

The app supports **10+ AI providers** with intelligent auto-fill:
- YT Clip AI (Recommended)
- OpenAI
- Google Gemini
- Groq
- Anthropic Claude
- And more...

For detailed provider comparison and setup, see: [AI_PROVIDER_SELECTOR.md](AI_PROVIDER_SELECTOR.md)

### Highlight Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_clips` | 5 | Number of clips to generate |
| `min_duration` | 60s | Minimum clip duration |
| `max_duration` | 120s | Maximum clip duration |
| `target_duration` | 90s | Ideal clip duration |
| `temperature` | 1.0 | AI creativity (0.0-2.0) |

### Portrait Conversion Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_resolution` | 1080x1920 | Output video resolution |
| `min_frames_before_switch` | 210 | Frames before speaker switch (~7s at 30fps) |
| `switch_threshold` | 3.0 | Movement multiplier to trigger switch |

### Caption Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `language` | id | Transcription language |
| `chunk_size` | 4 | Words per caption line |

### Hook Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tts_voice` | nova | OpenAI TTS voice (nova/shimmer/alloy) |
| `tts_speed` | 1.0 | Speech speed |
| `max_words` | 15 | Maximum words in hook text |
| `tts_model` | tts-1 | TTS model (tts-1 or tts-1-hd) |

---

## ðŸ”§ How It Works

### 1. Video Download
- Uses yt-dlp to download video in best quality (max 1080p)
- Automatically fetches Indonesian auto-generated subtitles
- Extracts video metadata (title, description, channel)

### 2. Highlight Detection
- Parses SRT subtitle file with timestamps
- Sends transcript to GPT-4 with specific criteria:
  - Punchlines and funny moments
  - Interesting insights
  - Emotional/dramatic moments
  - Memorable quotes
  - Complete story arcs
- Validates duration (60-120 seconds)
- Generates hook text for each highlight

### 3. Portrait Conversion
- Uses OpenCV Haar Cascade for face detection
- Tracks lip movement to identify active speaker
- Implements "camera cut" style switching (not smooth panning)
- Stabilizes crop position within each "shot"
- Maintains 9:16 aspect ratio at 1080x1920

### 4. Hook Generation
- Extracts first frame from clip
- Generates TTS audio using OpenAI's voice API
- Creates intro scene with:
  - Blurred/dimmed first frame background
  - Centered hook text with yellow highlight
  - AI voiceover reading the hook
- Concatenates hook with main clip

### 5. Caption Generation
- Transcribes audio using OpenAI Whisper
- Creates ASS subtitle file with:
  - Word-by-word timing
  - Yellow highlight on current word
  - Black outline and semi-transparent background
- Burns captions into video using FFmpeg

---

## ðŸŽ¨ Caption Styling

The captions use CapCut-style formatting:

```
Font: Arial Black
Size: 70px
Color: White (#FFFFFF)
Highlight: Yellow (#00FFFF)
Outline: 4px Black
Shadow: 2px
Position: Lower third (350px from bottom)
```

---

## ï¿½ API Usage & Costs 

Estimated OpenAI API costs per video (5 clips):

| Feature | Model | Est. Cost |
|---------|-------|-----------|
| Highlight Detection | GPT-4.1 | ~$0.05-0.15 |
| TTS Voiceover | TTS-1 | ~$0.01/clip |
| Captions | Whisper API | ~$0.01/clip |

**Total estimate:** ~$0.10-0.25 per video (5 clips)

The desktop app shows real-time token usage and cost estimation during processing.

---

## ðŸ¤ Contributing

Contributions are welcome! We greatly appreciate contributions from anyone.

### ðŸ”¨ Building Desktop App from Source

For developers who want to build the .exe themselves, see the complete guide in [BUILD.md](BUILD.md).

Quick steps:
```bash
# Install dependencies
pip install -r requirements.txt
pip install pyinstaller

# Build
pyinstaller build.spec

# Output: dist/AutoClipper.exe
```

### Quick Start for Contributors

```bash
# 1. Fork this repo (click the Fork button on GitHub)

# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/yt-short-clipper.git
cd yt-short-clipper

# 3. Add upstream remote
git remote add upstream https://github.com/OWNER/yt-short-clipper.git

# 4. Create a new branch
git checkout -b feature/your-new-feature

# 5. Make changes, then commit
git add .
git commit -m "feat: description of changes"

# 6. Push to your fork
git push origin feature/your-new-feature

# 7. Create a Pull Request on GitHub
```

### How to Contribute

| Type | Description |
|-------|-----------|
| ðŸ› **Bug Report** | Report bugs in the [Issues](../../issues) tab |
| ðŸ’¡ **Feature Request** | Request new features in [Issues](../../issues) |
| ðŸ“– **Documentation** | Improve docs, fix typos, add examples |
| ðŸ”§ **Code** | Fix bugs, add features, improve performance |

ðŸ“š **Complete guide available in [CONTRIBUTING.md](CONTRIBUTING.md)** - includes Git tutorial for beginners!

---

## ðŸ“ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## âš ï¸ Disclaimer

- This tool is for personal/educational use only
- Respect YouTube's Terms of Service
- Ensure you have rights to use the content you're processing
- The AI-generated content should be reviewed before publishing

## ðŸ™ Acknowledgments

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube downloading
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [OpenCV](https://opencv.org/) - Computer vision
- [FFmpeg](https://ffmpeg.org/) - Video processing
- [OpenAI API](https://openai.com/) - GPT-4 and TTS

---

## ðŸ‘¨â€ðŸ’» Credits

Made with â˜• by **Aji Prakoso** for content creators

| | |
|---|---|
| ðŸŽ“ | [n8n & Automation eCourse](https://classroom.jipraks.com) |
| ðŸ“¸ | [@jipraks on Instagram](https://instagram.com/jipraks) |
| ðŸŽ¬ | [Aji Prakoso on YouTube](https://youtube.com/@jipraks) |
| ðŸŒ | [About Aji Prakoso](https://www.jipraks.com) |
