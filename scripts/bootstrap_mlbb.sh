#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[bootstrap] project root: $ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "[bootstrap] creating virtual environment (.venv)"
  python -m venv .venv
fi

if [[ -f ".venv/bin/activate" ]]; then
  # Linux/macOS
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f ".venv/Scripts/activate" ]]; then
  # Git Bash / Windows
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
else
  echo "[bootstrap] cannot find venv activation script"
  exit 1
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  yt-dlp \
  ffmpeg-python \
  opencv-python \
  mediapipe \
  pytesseract \
  pillow \
  numpy \
  librosa \
  python-dotenv \
  requests \
  openai

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[bootstrap] ffmpeg is not found in PATH."
  echo "Install ffmpeg first:"
  echo "  macOS:  brew install ffmpeg"
  echo "  Ubuntu: sudo apt install ffmpeg"
  echo "  Windows (choco): choco install ffmpeg"
fi

if ! command -v tesseract >/dev/null 2>&1; then
  echo "[bootstrap] tesseract is not found in PATH."
  echo "Install tesseract OCR first:"
  echo "  macOS:  brew install tesseract"
  echo "  Ubuntu: sudo apt install tesseract-ocr"
  echo "  Windows (choco): choco install tesseract"
fi

echo "[bootstrap] running smoke test (dummy candidate generation)"
python - <<'PY'
from mlbb_pipeline.rank_clips import build_candidates

audio = [{"timestamp_sec": i, "score": 0.2 + (0.7 if i % 13 == 0 else 0.0)} for i in range(1, 180)]
ocr = [{"timestamp_sec": 39, "score": 0.9, "hits": ["TRIPLE KILL"]}]
motion = [{"timestamp_sec": i, "score": 0.3 + (0.5 if i % 17 == 0 else 0.0)} for i in range(1, 180)]
candidates = build_candidates(
    duration_sec=180,
    audio_scores=audio,
    ocr_scores=ocr,
    motion_scores=motion,
    clip_min_sec=20,
    clip_max_sec=45,
    max_candidates=10,
)
print(f"[bootstrap] smoke test ok, candidates={len(candidates)}")
if not candidates:
    raise SystemExit(1)
PY

echo "[bootstrap] done."
echo "Next:"
echo "  1) cp .env.example .env"
echo "  2) edit .env values"
echo "  3) python main.py --url 'https://www.youtube.com/watch?v=VIDEO_ID' --platform tiktok --dry-run"

