import argparse
import subprocess
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="prepforge",
        description="prepforge - Multi-interface AI assistant with training support",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # -------------------------
    # RUN APPS
    # -------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Run the application (tui | gui | streamlit)",
        description=(
            "Run prepforce in one of the following modes:\n\n"
            "  tui        Terminal interface\n"
            "  gui        Desktop application\n"
            "  streamlit  Web interface\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    run_parser.add_argument(
        "mode",
        choices=["tui", "gui", "streamlit"],
        help="Mode to run",
    )

    # -------------------------
    # TRAIN
    # -------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Train a LoRA adapter",
        description=(
            "Train a model using your dataset\n\n"
            "Dataset Control Options:\n"
            "  --limit N     Train on first N samples\n"
            "  --subset F    Train on fraction (0.1 = 10%%)\n\n"
            "Examples:\n"
            "  prepforce train --dataset data.jsonl\n"
            "  prepforce train --dataset data.jsonl --limit 500\n"
            "  prepforce train --dataset data.jsonl --subset 0.05\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    train_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset file (JSON/JSONL format)",
    )

    train_parser.add_argument(
        "--output",
        default="trained_model",
        help="Output directory for trained model",
    )

    train_parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )

    train_parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )

    # 🔥 Mutually exclusive group
    group = train_parser.add_mutually_exclusive_group()

    group.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples (e.g. 500)",
    )

    group.add_argument(
        "--subset",
        type=float,
        help="Use fraction of dataset (0.05 = 5%%)",
    )

    # -------------------------
    # CONFIG
    # -------------------------
    config_parser = subparsers.add_parser(
        "config",
        help="Set default model or LoRA",
        description=(
            "Configure default model and LoRA adapter\n\n"
            "Examples:\n"
            "  prepforce config --model meta-llama/Llama-3-8B\n"
            "  prepforce config --lora ~/adapter\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    config_parser.add_argument(
        "--model",
        help="Set default base model",
    )

    config_parser.add_argument(
        "--lora",
        help="Set default LoRA adapter path",
    )

    args = parser.parse_args()

    try:
        # -------------------------
        # RUN MODES
        # -------------------------
        if args.command == "run":
            if args.mode == "tui":
                from .tui_app import main

                main()

            elif args.mode == "gui":
                from .gui_app import main

                main()

            elif args.mode == "streamlit":
                base_dir = os.path.dirname(__file__)
                app_path = os.path.join(base_dir, "streamlit_app.py")

                subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

        # -------------------------
        # TRAIN
        # -------------------------
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

        # -------------------------
        # CONFIG
        # -------------------------
        elif args.command == "config":
            from .core.config_store import load_user_config, save_user_config

            config = load_user_config()

            if args.model:
                config["model"] = args.model
                print(f"✅ Default model set to: {args.model}")

            if args.lora:
                config["lora"] = args.lora
                print(f"✅ Default LoRA set to: {args.lora}")

            if not args.model and not args.lora:
                print("⚠️ No changes provided. Use --model or --lora")

            save_user_config(config)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")


if __name__ == "__main__":
    main()
