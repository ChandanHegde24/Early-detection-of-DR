import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_settings():
    """Load settings from YAML file with environment variable overrides."""
    config_path = Path(__file__).parent / "settings.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    
    # Override from environment variables
    if os_val := os.getenv("TIER_URGENT_THRESHOLD"):
        if "prioritization" not in settings:
            settings["prioritization"] = {}
        settings["prioritization"]["urgent_threshold"] = float(os_val)
    
    if os_val := os.getenv("TIER_MODERATE_THRESHOLD"):
        if "prioritization" not in settings:
            settings["prioritization"] = {}
        settings["prioritization"]["moderate_threshold"] = float(os_val)
    
    if os_val := os.getenv("CNN_BACKBONE"):
        if "cnn" not in settings:
            settings["cnn"] = {}
        settings["cnn"]["backbone"] = os_val
    
    if os_val := os.getenv("BIOMARKER_MODEL_TYPE"):
        if "biomarker_model" not in settings:
            settings["biomarker_model"] = {}
        settings["biomarker_model"]["type"] = os_val
    
    if os_val := os.getenv("LOG_LEVEL"):
        if "logging" not in settings:
            settings["logging"] = {}
        settings["logging"]["level"] = os_val
    
    return settings

