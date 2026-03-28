import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ✅ Use shared config
from major_project.config import get_model, get_lora


def _check_cuda():
    """
    Ensure NVIDIA GPU is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n❌ CUDA GPU not detected.\n"
            "This application requires an NVIDIA GPU.\n"
            "CPU execution is not supported.\n"
            "\n👉 Install CUDA-enabled PyTorch:\n"
            "pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
        )


def load_model(
    base_model: str | None = None,
    lora_path: str | None = None,
    device_map: str = "auto",
    dtype: torch.dtype = torch.float16,
):
    """
    Load base model and optionally apply LoRA adapter.

    Args:
        base_model (str | None): HuggingFace model name or local path
        lora_path (str | None): Path to LoRA adapter folder
        device_map (str): Device mapping ("auto", "cpu", etc.)
        dtype (torch.dtype): Torch dtype

    Returns:
        model, tokenizer
    """

    # -----------------------------
    # 🔥 CUDA CHECK
    # -----------------------------
    _check_cuda()

    # -----------------------------
    # USE DEFAULT CONFIG (if not provided)
    # -----------------------------
    if base_model is None:
        base_model = get_model()

    if lora_path is None:
        lora_path = get_lora()

    # -----------------------------
    # VALIDATE INPUTS
    # -----------------------------
    if not base_model:
        raise ValueError("Base model path/name must be provided")

    if lora_path:
        lora_path = os.path.expanduser(lora_path)

        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA path not found: {lora_path}")

        expected = os.path.join(lora_path, "adapter_config.json")
        if not os.path.exists(expected):
            raise ValueError(
                f"Invalid LoRA adapter folder (missing adapter_config.json): {lora_path}"
            )

    # -----------------------------
    # LOAD BASE MODEL
    # -----------------------------
    print(f"[MODEL] Loading base model: {base_model}")

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

    # -----------------------------
    # APPLY LORA (OPTIONAL)
    # -----------------------------
    if lora_path:
        print(f"[MODEL] Applying LoRA adapter: {lora_path}")

        model = PeftModel.from_pretrained(
            model,
            lora_path,
            local_files_only=True,
        )

        print("[MODEL] LoRA successfully loaded")

    # -----------------------------
    # FINAL SETUP
    # -----------------------------
    model.eval()

    return model, tokenizer
