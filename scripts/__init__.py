import sys
from pathlib import Path

# Ensure local source package is importable when running the script directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))