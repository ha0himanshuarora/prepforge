import os
import json

CONFIG_DIR = os.path.expanduser("~/.mentorai")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")


def load_user_config():
    if not os.path.exists(CONFIG_FILE):
        return {}

    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def save_user_config(config):
    os.makedirs(CONFIG_DIR, exist_ok=True)

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
