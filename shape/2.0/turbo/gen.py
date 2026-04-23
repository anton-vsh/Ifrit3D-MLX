from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shape.runner import run_shape_preset_cli


if __name__ == "__main__":
    run_shape_preset_cli("2.0-turbo")
