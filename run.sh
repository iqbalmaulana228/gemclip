#!/usr/bin/env bash
# Pasang dependensi lalu jalankan Gemclip di http://127.0.0.1:8765
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "File .env dibuat. Isi GEMINI_API_KEY di dalamnya, lalu jalankan ulang."
fi

python -m gemclip
