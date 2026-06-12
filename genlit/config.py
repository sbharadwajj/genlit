"""YAML config loader. Configs live under <repo_root>/configs/."""
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(mode: str) -> dict:
    """Load configs/<mode>.yaml as a dict."""
    path = _CONFIG_DIR / f"{mode}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config for mode={mode!r} at {path}")
    with open(path) as f:
        return yaml.safe_load(f)
