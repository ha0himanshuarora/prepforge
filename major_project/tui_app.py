import os
import subprocess
import warnings
from transformers.utils import logging
from rich.console import Console
from rich.prompt import Prompt
from rich.live import Live
from rich.panel import Panel

# prompt_toolkit
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.widgets import Box, Frame, TextArea

# 🔥 IMPORT SHARED MODULES
from major_project.core.model import load_model
from major_project.core.utils import (
    build_prompt,
    generate_stream,
    chat_history,
    last_mcq_block,
)
from major_project.config import DEFAULT_APP_NAME, DEFAULT_LORA

# -----------------------------
# CONFIG
# -----------------------------
APP_NAME = DEFAULT_APP_NAME
console = Console()

# -----------------------------
# TEMPLATE (HF ONLY)
# -----------------------------
LLAMA3_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] == 'system' %}"
    "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
    "{% else %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)

# -----------------------------
# ENV
# -----------------------------
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()


# -----------------------------
# BANNER
# -----------------------------
def print_banner():
    try:
        result = subprocess.run(["figlet", APP_NAME], capture_output=True, text=True)
        console.print(f"[bold cyan]{result.stdout}[/bold cyan]", justify="center")
        console.print("[dim]AI Exam Preparation System[/dim]\n", justify="center")
    except Exception:
        console.print(f"[bold cyan]\n=== {APP_NAME.upper()} ===\n[/bold cyan]")


print_banner()


# -----------------------------
# INPUT HELPERS
# -----------------------------
def get_valid_length(default):
    while True:
        value = Prompt.ask("Response length (100-2000)", default=str(default))
        try:
            val = int(value)
            if 100 <= val <= 2000:
                return val
        except:
            pass
        console.print("[red]Invalid input[/red]")


def get_question_count(default=5):
    while True:
        value = Prompt.ask("How many questions?", default=str(default))
        try:
            val = int(value)
            if 1 <= val <= 50:
                return val
        except:
            pass
        console.print("[red]Enter a number between 1-50[/red]")


def save_chat():
    file_path = Prompt.ask("Enter file path")
    if not file_path:
        return
    try:
        file_path = os.path.expanduser(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"=== {APP_NAME} Notes ===\n\n")
            for msg in chat_history:
                role = "User" if msg["role"] == "user" else "AI"
                f.write(f"{role}:\n{msg['content']}\n\n")
        console.print(f"[green]Saved to {file_path}[/green]")
    except Exception as e:
        console.print(f"[red]{str(e)}[/red]")


def select_model_mode():
    return Prompt.ask("Choose model mode", choices=["base", "lora"], default="base")


def get_lora_path():
    path = Prompt.ask("Enter LoRA adapter path (leave empty to cancel)")
    return path if path else None


# -----------------------------
# LOAD MODEL
# -----------------------------
console.print(f"[bold green]Loading {APP_NAME} model...[/bold green]")

model_mode = select_model_mode()
lora_path = DEFAULT_LORA if model_mode == "lora" else None

if model_mode == "lora":
    user_lora = get_lora_path()
    if user_lora:
        lora_path = user_lora

model, tokenizer = load_model(base_model=None, lora_path=lora_path)

# 🔥 Detect GGUF
is_gguf = isinstance(model, dict) and model.get("type") == "gguf"

# -----------------------------
# HF ONLY SETUP
# -----------------------------
if not is_gguf:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.chat_template = LLAMA3_TEMPLATE

console.print("Model loaded!\n")


# -----------------------------
# MODE MENU
# -----------------------------
def select_mode():
    options = ["chat", "notes", "mcq", "plan", "practice", "exit"]
    index = [0]

    def get_text():
        return "\n".join(
            [f"{'>' if i == index[0] else ' '} {opt}" for i, opt in enumerate(options)]
        )

    text_area = TextArea(text=get_text(), focusable=False)
    kb = KeyBindings()

    @kb.add("up")
    def _(event):
        index[0] = (index[0] - 1) % len(options)
        text_area.text = get_text()

    @kb.add("down")
    def _(event):
        index[0] = (index[0] + 1) % len(options)
        text_area.text = get_text()

    @kb.add("enter")
    def _(event):
        event.app.exit(result=options[index[0]])

    root = Frame(Box(text_area, padding=1), title=f" {APP_NAME} ")
    app = Application(layout=Layout(root), key_bindings=kb)
    return app.run()


# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    global last_mcq_block

    mode, length = None, None
    first_run = True

    while True:
        console.print(f"\n=== {APP_NAME} ===")

        if first_run:
            mode = select_mode()
            if mode == "exit":
                break
            length = get_valid_length(400)
            first_run = False
        else:
            if Prompt.ask("Change settings(y/n)?", default="n") == "y":
                mode = select_mode()
                if mode == "exit":
                    break
                length = get_valid_length(length)

        console.print(f"Mode: {mode} | length: {length}")

        try:
            user_input = Prompt.ask("\nAsk your question (or 'save' to save chat)")
        except KeyboardInterrupt:
            break

        if user_input == "exit":
            break
        if user_input == "save":
            save_chat()
            continue

        count = get_question_count() if mode in ["mcq", "practice"] else None
        prompt = build_prompt(mode, user_input, count)

        console.print("\nThinking...\n")

        # -----------------------------
        # GGUF STREAMING (NEW 🔥)
        # -----------------------------
        if is_gguf:
            llm = model["llm"]
            response = ""

            try:
                stream = llm(prompt, max_tokens=length, stream=True)

                with Live(Panel("", title="Response"), refresh_per_second=10) as live:
                    for chunk in stream:
                        token = chunk["choices"][0]["text"]
                        response += token
                        live.update(Panel(response, title="Response"))

            except Exception as e:
                console.print(f"[red]GGUF Error: {e}[/red]")
                continue

        # -----------------------------
        # HF STREAMING (UNCHANGED)
        # -----------------------------
        else:
            streamer = generate_stream(model, tokenizer, prompt, length)
            response = ""

            with Live(Panel("", title="Response"), refresh_per_second=10) as live:
                for token in streamer:
                    response += token
                    live.update(Panel(response, title="Response"))

        chat_history.append({"role": "assistant", "content": response})

        if "SECTION 1" in response and "SECTION 2" in response:
            last_mcq_block = response


if __name__ == "__main__":
    main()
