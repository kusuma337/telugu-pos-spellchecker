"""Launcher script for the Telugu POS Tagging & Spellchecking Streamlit app."""
import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "app" / "streamlit_app.py"

if __name__ == "__main__":
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(APP_PATH),
        "--server.port", "8501",
        "--server.headless", "false",
    ]
    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd)
