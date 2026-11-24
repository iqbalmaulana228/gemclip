@echo off
rem Pasang dependensi lalu jalankan Gemclip di http://127.0.0.1:8765
cd /d "%~dp0"

if not exist .venv py -m venv .venv
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt

if not exist .env (
  copy .env.example .env >nul
  echo File .env dibuat. Isi GEMINI_API_KEY di dalamnya, lalu jalankan ulang.
  pause
  exit /b 0
)

python -m gemclip
pause
