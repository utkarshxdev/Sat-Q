# conftest.py — pytest root configuration
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `satquery.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))
