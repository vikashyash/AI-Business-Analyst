import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
