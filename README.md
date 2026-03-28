# 🎓 MentorAI

An AI-powered exam preparation system with multiple interfaces and LoRA support.

---

## ✨ Features

- 🖥 Terminal UI (TUI) using Rich  
- 🪟 Desktop GUI using PyQt6  
- 🌐 Web interface using Streamlit  
- 🧠 Supports base LLM + LoRA adapters  
- 🏋️ Built-in training pipeline (QLoRA / Unsloth)  
- ⚡ Offline inference support  

---

## 📂 Project Structure

```
major_project/
├── core/
│   ├── model.py        # model + LoRA loader
│   ├── train.py        # training pipeline
│   └── utils.py
├── tui_app.py          # terminal interface
├── gui_app.py          # PyQt GUI
├── streamlit_app.py    # web app
├── cli.py              # CLI entry point
├── config.py
```

---

## ⚙️ Installation

```bash
git clone <your-repo>
cd major_project_v2
pip install -e .
```

---

## 🚀 Usage

### 🖥 TUI
```bash
mentorai run tui
```

### 🪟 GUI
```bash
mentorai run gui
```

### 🌐 Web UI
```bash
mentorai run streamlit
```

---

## 🧠 Using Custom Model

```bash
mentorai run tui --model <model_path>
```

---

## 🔥 Using LoRA Adapter

```bash
mentorai run tui --model <base_model> --lora <adapter_path>
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

## 📊 Example Dataset Format

```json
{
  "instruction": "Explain Newton's laws",
  "input": "",
  "output": "Newton's laws describe motion..."
}
```

---

## ⚠️ Notes

- Models are NOT bundled  
- Users must download base models separately  
- Works best with GPU (CUDA)  

---

## 🧑‍💻 Author

Himanshu Arora

---

## 📜 License

MIT License
