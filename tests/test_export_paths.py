from pathlib import Path
from src.config import OUTPUT_DIR

def test_output_dir_exists():
    assert OUTPUT_DIR.exists()