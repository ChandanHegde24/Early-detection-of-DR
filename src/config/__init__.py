"""Configuration module."""

from pathlib import Path
import yaml


CONFIG_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = CONFIG_DIR / "settings.yaml"


def load_settings() -> dict:
    """Load settings from the YAML configuration file."""
    with open(SETTINGS_PATH, "r") as f:
        return yaml.safe_load(f)
