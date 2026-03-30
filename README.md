# PrepForge

PrepForge is a modular AI-powered exam preparation system that provides multiple interfaces (TUI, GUI, and Web) for interacting with large language models. It supports both Hugging Face transformer models and GGUF-based models via llama.cpp, along with optional LoRA adapters and a built-in training pipeline.

---

## Overview

PrepForge is designed as a unified interface for:

- Running local language models  
- Managing multiple model backends (Hugging Face and GGUF)  
- Applying LoRA adapters dynamically  
- Generating study material (notes, MCQs, plans, practice)  
- Training custom models using QLoRA / Unsloth  

The system separates concerns between configuration, model loading, and interface layers.

---

## Tech Stack

| Category            | Technologies / Tools                          | Purpose |
|--------------------|----------------------------------------------|--------|
| Core               | Python 3.10+, Transformers, PEFT, HF Hub     | Model loading, LoRA support, model management |
| Inference Backend  | PyTorch, llama-cpp-python                    | Running HF models (GPU/CPU) and GGUF models |
| Interfaces         | Rich, prompt_toolkit, PyQt6, Streamlit       | TUI, CLI input, GUI, and Web interface |
| Training           | Unsloth, TRL, Datasets                       | Fine-tuning models using QLoRA pipelines |
| System Utilities   | fzf (optional), subprocess, JSON             | Model selection, CLI execution, config storage |
---

## Installation

### Base Installation

```bash
pip install prepforge
```

### Optional Features

```bash
pip install prepforge[gguf]        # GGUF (llama.cpp) support
pip install prepforge[gpu]         # CUDA / torch support
pip install prepforge[web]         # Streamlit interface
pip install prepforge[gui]         # PyQt6 GUI
pip install prepforge[train]       # Training dependencies
pip install prepforge[gguf,gpu,web,gui]
```

---

## Architecture

### CLI Layer (`cli.py`)
Handles:
- Running interfaces
- Model installation
- Configuration
- Training commands

### Core Layer (`core/`)
- `model.py`: Loads HF and GGUF models, applies LoRA
- `utils.py`: Prompt building and streaming
- `train.py`: Fine-tuning pipeline

### Interface Layer
- `tui_app.py`: Terminal UI (Rich)
- `gui_app.py`: Desktop GUI (PyQt6)
- `streamlit_app.py`: Web UI

### Configuration Layer
Stored at:

```
~/.prepforce/config.json
```

---

## Model Support

### Hugging Face Models
- Loaded via `transformers`
- Cached automatically
- Supports LoRA

Examples:
```
meta-llama/Llama-3-8B
mistralai/Mistral-7B-Instruct
```

---

### GGUF Models
- Single-file models
- Run via llama.cpp
- Lower resource usage

Example:
```
Qwen3-4B-Q4_1.gguf
```

---

### LoRA Adapters
- Lightweight fine-tuned layers
- Applied on top of base HF models
- Must match base architecture

---

## Model Installation

```bash
prepforge install_model
```

Features:
- Browse models from Hugging Face
- Filter by type (GGUF / HF / LoRA)
- Install:
  - GGUF → single file
  - HF → full snapshot
  - LoRA → adapter files

Storage:
- GGUF → `~/models`
- HF → `~/.cache/huggingface/hub`

---

## Configuration

Set default model:

```bash
prepforge config --model <model_path_or_repo>
```

Set LoRA:

```bash
prepforge config --lora <adapter_path>
```

Interactive selection:

```bash
prepforge config
```

---

## Running the Application

Modes:
- Chat
- Notes
- MCQ generation
- Study plan
- Practice

### Terminal UI

```bash
prepforge run tui
```

---

### Desktop GUI

```bash
prepforge run gui
```

---

### Web Interface

```bash
prepforge run streamlit
```

Open:
```
http://localhost:8501
```

---

## Inference Flow

1. Load model from config  
2. Detect backend:
   - GGUF → llama.cpp  
   - HF → transformers  
3. Apply LoRA if provided  
4. Build prompt  
5. Generate response (streaming)  

---

## Training

```bash
prepforge train \
  --dataset dataset.jsonl \
  --epochs 3 \
  --output trained_model
```

Options:
- `--limit` → limit samples  
- `--subset` → percentage of dataset  
- `--lr` → learning rate  

---

## Dataset Format

```json
{
  "instruction": "Explain Newton's laws",
  "input": "",
  "output": "Newton's laws describe motion..."
}
```

---

## Project Structure

```
major_project/
├── core/
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── tui_app.py
├── gui_app.py
├── streamlit_app.py
├── cli.py
├── config.py
```

---

## Requirements

- Python 3.10+
- CUDA GPU recommended for HF models

---

## Notes

- Models are not bundled  
- Must be installed separately  
- GGUF requires `llama-cpp-python`  
- LoRA must match base model  

---

## Author

Himanshu Arora

---

## License

MIT License
