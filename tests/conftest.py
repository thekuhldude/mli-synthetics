import sys
from pathlib import Path

# Make `mli_synthetics` importable without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
