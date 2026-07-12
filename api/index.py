import sys
import os

# ── Path setup ──────────────────────────────────────────────────────────────
# Vercel runs from project root; make sure src/ is importable
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, "src"))

# ── Import FastAPI app ───────────────────────────────────────────────────────
# Vercel requires the ASGI app to be named 'app' or exposed via 'handler'
from src.api import app  # noqa: F401  (Vercel picks this up automatically)
