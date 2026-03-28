import os
from major_project.core.config_store import load_user_config

DEFAULT_MODEL = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"
DEFAULT_LORA = None
DEFAULT_APP_NAME = "PrepForge"
MAX_HISTORY = 6


def get_model():
    user_config = load_user_config()
    return user_config.get("model") or os.getenv("MENTORAI_MODEL") or DEFAULT_MODEL


def get_lora():
    user_config = load_user_config()
    return user_config.get("lora") or os.getenv("MENTORAI_LORA") or DEFAULT_LORA
