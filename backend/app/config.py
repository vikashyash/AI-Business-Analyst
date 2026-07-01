import os
from pathlib import Path
from dotenv import load_dotenv

try:
    load_dotenv()
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
    MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
    KB_DIR=Path(os.getenv("KB","kb"))
    WORKSPACE_DIR=Path(os.getenv("WORKSPACE_PATH","workspaces"))
except Exception:
    UPLOAD_DIR = Path("uploads")
    MAX_BYTES = 10 * 1024 * 1024
    KB_DIR=Path("uploads/kb")
    WORKSPACE_DIR=Path("workspaces")
