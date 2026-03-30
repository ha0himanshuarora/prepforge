import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from major_project.config import get_model, get_lora


def _check_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("\n❌ CUDA GPU not detected.\nHF models require GPU.\n")


def load_model(
    base_model: str | None = None,
    lora_path: str | None = None,
    device_map: str = "auto",
    dtype: torch.dtype = torch.float16,
):

    if base_model is None:
        base_model = get_model()

    if lora_path is None:
        lora_path = get_lora()

    if not base_model:
        raise ValueError("Base model required")

    base_model = os.path.expanduser(base_model)

    # =========================================================
    # GGUF MODE
    # =========================================================
    if base_model.endswith(".gguf"):
        print(f"[MODEL] Using GGUF model: {base_model}")

        if not os.path.exists(base_model):
            raise FileNotFoundError(f"{base_model} not found")

        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError("Install GGUF support:\npip install llama-cpp-python")

        llm = Llama(
            model_path=base_model,
            n_ctx=4096,
            n_threads=os.cpu_count() or 4,
            verbose=False,  # 🔥 removes spam logs
        )

        return {
            "type": "gguf",
            "llm": llm,
            "path": base_model,
        }, None

    # =========================================================
    # HF MODE
    # =========================================================
    _check_cuda()

    print(f"[MODEL] Loading HF model: {base_model}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        torch_dtype=dtype,
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=True,
        use_fast=True,
    )

    if lora_path:
        lora_path = os.path.expanduser(lora_path)

        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA not found: {lora_path}")

        print(f"[MODEL] Applying LoRA: {lora_path}")

        model = PeftModel.from_pretrained(
            model,
            lora_path,
            local_files_only=True,
        )

    model.eval()

    return model, tokenizer
