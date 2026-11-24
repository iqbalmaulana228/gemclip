import uvicorn

from .server import app


def main() -> None:
    print("Gemclip berjalan di http://127.0.0.1:8765 (Ctrl+C untuk berhenti)")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
