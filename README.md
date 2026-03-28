# 🎓 MentorAI

MentorAI is an AI-powered exam preparation system with multiple interfaces and support for base LLMs and LoRA adapters.

---

## 📦 Installation

Install directly from PyPI:

```bash
pip install mentorai
```

PyPI: https://pypi.org/project/mentorai/0.1.0/

---

## ✨ Features

- 🖥 Terminal UI (TUI)
- 🪟 Desktop GUI (PyQt6)
- 🌐 Web interface (Streamlit)
- 🧠 Supports base models and LoRA adapters
- 🏋️ Built-in training pipeline (QLoRA / Unsloth)
- ⚡ Offline inference support

---

## 🚀 Usage

### Terminal Interface (TUI)
```bash
mentorai run tui
```

### Desktop GUI
```bash
mentorai run gui
```

### Web Interface
```bash
mentorai run streamlit
```

---

## ⚙️ Configuration

Set default model:

```bash
mentorai config --model <model_name_or_path>
```

Set LoRA adapter:

```bash
mentorai config --lora <adapter_path>
```

---

## 🏋️ Training

```bash
mentorai train \
  --dataset dataset.jsonl \
  --epochs 3 \
  --output trained_model
```

---

## 📊 Dataset Format

```json
{
  "instruction": "Explain Newton's laws",
  "input": "",
  "output": "Newton's laws describe motion..."
}
```

---

## 📂 Project Structure

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

## ⚠️ Notes

- Models are not bundled with the package  
- Users must download base models separately  
- GPU (CUDA) is recommended for best performance  

---

## 🧑‍💻 Author

Himanshu Arora

---

## 📜 License

MIT License
