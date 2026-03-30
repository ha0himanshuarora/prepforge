import argparse
import subprocess
import os
import sys
import json
import requests

from .core.config_store import load_user_config, save_user_config

HF_API = "https://huggingface.co/api/models"


def prettify_model_name(path):
    if "models--" in path:
        name = path.split("models--")[-1]
        name = name.split("/snapshots")[0]
        name = name.replace("--", "/")
        return name
    return os.path.basename(path)


def get_local_models():
    model_dirs = [
        os.path.expanduser("~/models"),
        os.path.expanduser("~/.cache/huggingface/hub"),
    ]

    models = {}

    for base in model_dirs:
        if not os.path.exists(base):
            continue

        for root, dirs, files in os.walk(base):
            for file in files:
                if file.endswith(".gguf"):
                    full_path = os.path.join(root, file)
                    models[f"[GGUF] {file}"] = full_path

            if "snapshots" in root:
                if any(
                    f in files
                    for f in [
                        "config.json",
                        "adapter_config.json",
                        "pytorch_model.bin",
                        "model.safetensors",
                    ]
                ):
                    try:
                        parts = root.split("models--")[1]
                        repo = parts.split("/snapshots")[0]
                        repo = repo.replace("--", "/")
                        models[f"[HF] {repo}"] = root
                    except Exception:
                        pass

    return models


def fzf_select(options):
    try:
        fzf = subprocess.Popen(
            [
                "fzf",
                "--height=100%",
                "--layout=reverse",
                "--border",
                "--prompt=Select: ",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        stdout, _ = fzf.communicate("\n".join(options))
        return stdout.strip() if stdout else None
    except FileNotFoundError:
        print("fzf not installed")
        return None


def select_model_local():
    models = get_local_models()

    if not models:
        print("No local models found")
        return None

    selected = fzf_select(list(models.keys()))
    return models.get(selected)


def estimate_size_from_name(filename):
    filename = filename.lower()

    for q in ["q2", "q3", "q4", "q5", "q6", "q8"]:
        if q in filename:
            return int(q[1])

    if "bf16" in filename or "fp16" in filename:
        return 16

    return None


# -------------------------
# FETCH MODELS (FIXED)
# -------------------------
def fetch_models():
    try:
        response = requests.get(
            HF_API,
            params={"limit": 50, "full": "true"},
        )
        data = response.json()

        models = {}

        for m in data:
            model_id = m["modelId"]
            siblings = m.get("siblings", [])

            models[model_id] = {
                "id": model_id,
                "siblings": siblings,
            }

        return models

    except Exception as e:
        print("Failed to fetch models:", e)
        return {}


# -------------------------
# FILTER TYPES
# -------------------------
def detect_model_type(model):
    siblings = model["siblings"]

    if any(f["rfilename"].endswith(".gguf") for f in siblings):
        return "gguf"

    if any("adapter_config.json" in f["rfilename"] for f in siblings):
        return "lora"

    return "hf"


def filter_models(models, mode="all"):
    result = {}

    for k, v in models.items():
        t = detect_model_type(v)

        if mode == "all" or mode == t:
            result[k] = v

    return result


# -------------------------
# INSTALLERS
# -------------------------
def install_gguf(model):
    from huggingface_hub import hf_hub_download

    gguf_files = [f for f in model["siblings"] if f["rfilename"].endswith(".gguf")]

    if not gguf_files:
        print("No GGUF files found")
        return

    options = []
    mapping = {}

    for f in gguf_files:
        name = f["rfilename"]
        size = f.get("size")

        if size:
            size_gb = round(size / (1024**3), 2)
            label = f"{name} ({size_gb} GB)"
        else:
            est = estimate_size_from_name(name)
            label = f"{name} (~{est} GB)" if est else name

        options.append(label)
        mapping[label] = name

    selected = fzf_select(options)
    if not selected:
        return

    filename = mapping[selected]

    token = os.environ.get("HF_TOKEN") or input("HF Token: ").strip()
    save_path = os.path.expanduser("~/models")
    os.makedirs(save_path, exist_ok=True)

    hf_hub_download(
        repo_id=model["id"],
        filename=filename,
        token=token if token else None,
        local_dir=save_path,
    )

    print("Downloaded:", filename)


def install_hf(model):
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or input("HF Token: ").strip()

    snapshot_download(
        repo_id=model["id"],
        token=token if token else None,
    )

    print("HF model cached")


def install_lora(model):
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or input("HF Token: ").strip()

    snapshot_download(
        repo_id=model["id"],
        token=token if token else None,
    )

    print("LoRA downloaded")


# -------------------------
# INSTALL FLOW (UPDATED)
# -------------------------
def install_model():
    models = fetch_models()

    if not models:
        return

    mode = fzf_select(["all", "gguf", "hf", "lora"])
    models = filter_models(models, mode if mode else "all")

    selected_key = fzf_select(list(models.keys()))
    if not selected_key:
        return

    model = models[selected_key]
    model_type = detect_model_type(model)

    print("Selected:", model["id"])
    print("Type:", model_type)

    confirm = input("Proceed? (y/n): ").lower()
    if confirm != "y":
        return

    if model_type == "gguf":
        install_gguf(model)
    elif model_type == "hf":
        install_hf(model)
    elif model_type == "lora":
        install_lora(model)


# -------------------------
# MAIN CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(prog="prepforge")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("mode", choices=["tui", "gui", "streamlit"])

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--output", default="trained_model")
    train_parser.add_argument("--epochs", type=int, default=2)
    train_parser.add_argument("--lr", type=float, default=2e-4)

    group = train_parser.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int)
    group.add_argument("--subset", type=float)

    config_parser = subparsers.add_parser("config")
    config_parser.add_argument("--model")
    config_parser.add_argument("--lora")

    subparsers.add_parser("install_model")

    args = parser.parse_args()

    try:
        if args.command == "run":
            if args.mode == "tui":
                from .tui_app import main as tui_main

                tui_main()

            elif args.mode == "gui":
                from .gui_app import main as gui_main

                gui_main()

            elif args.mode == "streamlit":
                base_dir = os.path.dirname(__file__)
                app_path = os.path.join(base_dir, "streamlit_app.py")
                subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

        elif args.command == "train":
            from .core.train import train_model

            train_model(
                dataset_path=args.dataset,
                output_dir=args.output,
                epochs=args.epochs,
                lr=args.lr,
                limit=args.limit,
                subset=args.subset,
            )

        elif args.command == "config":
            config = load_user_config()

            if args.model:
                config["model"] = args.model
                print("Model set:", args.model)

            elif args.lora:
                config["lora"] = args.lora
                print("LoRA set:", args.lora)

            else:
                selected = select_model_local()
                if selected:
                    config["model"] = selected
                    print("Selected model:", selected)
                else:
                    print("No model selected")

            save_user_config(config)

        elif args.command == "install_model":
            install_model()

    except Exception as e:
        print("Error:", str(e))


if __name__ == "__main__":
    main()
