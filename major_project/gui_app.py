import sys
import os
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QComboBox,
    QLabel,
    QFileDialog,
    QSpinBox,
)
from PyQt6.QtCore import Qt

# 🔥 CORE IMPORTS
from major_project.core.model import load_model
from major_project.core.utils import (
    build_prompt,
    generate_stream,
    chat_history,
    last_mcq_block,
)
from major_project.config import DEFAULT_APP_NAME, DEFAULT_MODEL, DEFAULT_LORA

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
os.environ["QT_QPA_PLATFORMTHEME"] = "gtk3"

APP_NAME = DEFAULT_APP_NAME


class AIApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"🎓 {APP_NAME}")
        self.resize(900, 700)

        self.model_name = DEFAULT_MODEL
        self.lora_path = DEFAULT_LORA

        self.model = None
        self.tokenizer = None
        self.is_gguf = False  # 🔥 NEW

        layout = QVBoxLayout()

        # TITLE
        title = QLabel(APP_NAME)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # MODEL CONTROLS
        model_layout = QHBoxLayout()

        self.model_mode = QComboBox()
        self.model_mode.addItems(["Base Model", "LoRA Adapter"])

        self.lora_btn = QPushButton("Select LoRA Folder")
        self.lora_btn.clicked.connect(self.select_lora_folder)

        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self.load_selected_model)

        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_mode)
        model_layout.addWidget(self.lora_btn)
        model_layout.addWidget(self.load_btn)

        layout.addLayout(model_layout)

        # SETTINGS
        top_layout = QHBoxLayout()

        self.mode = QComboBox()
        self.mode.addItems(["Chat", "Notes", "Study Plan", "MCQ Generator", "Practice"])

        self.length_input = QSpinBox()
        self.length_input.setRange(100, 2000)
        self.length_input.setValue(400)

        self.count_input = QSpinBox()
        self.count_input.setRange(1, 50)
        self.count_input.setValue(5)

        top_layout.addWidget(self.mode)
        top_layout.addWidget(self.length_input)
        top_layout.addWidget(self.count_input)

        layout.addLayout(top_layout)

        # INPUT
        self.input_box = QLineEdit()
        layout.addWidget(self.input_box)

        # BUTTONS
        btn_layout = QHBoxLayout()

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.handle_generate)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_all)

        self.save_btn = QPushButton("Save Chat")
        self.save_btn.clicked.connect(self.save_chat)

        btn_layout.addWidget(self.generate_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.save_btn)

        layout.addLayout(btn_layout)

        # OUTPUT
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

        self.setLayout(layout)

    # -----------------------------
    # MODEL
    # -----------------------------
    def select_lora_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select LoRA Folder")
        if path:
            self.lora_path = path

    def load_selected_model(self):
        use_lora = self.model_mode.currentText() == "LoRA Adapter"

        if use_lora and not self.lora_path:
            self.output.append("❌ Select LoRA folder first")
            return

        self.output.append("🔄 Loading model...")
        QApplication.processEvents()

        # 🔥 CHANGED: use config-based model
        self.model, self.tokenizer = load_model(
            None, self.lora_path if use_lora else None
        )

        # 🔥 NEW: detect GGUF
        self.is_gguf = isinstance(self.model, dict) and self.model.get("type") == "gguf"

        backend = "GGUF (llama.cpp)" if self.is_gguf else "HuggingFace"
        self.output.append(f"✅ Model loaded [{backend}]")

    # -----------------------------
    # UTIL
    # -----------------------------
    def clear_all(self):
        global chat_history, last_mcq_block
        self.output.clear()
        chat_history.clear()
        last_mcq_block = None

    def save_chat(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Chat", "", "Text Files (*.txt)"
        )
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.output.toPlainText())
            self.output.append(f"\n✅ Saved to {file_path}\n")
        except Exception as e:
            self.output.append(f"\n❌ Error saving file: {str(e)}\n")

    # -----------------------------
    # STREAM GENERATE
    # -----------------------------
    def handle_generate(self):
        if self.model is None:
            self.output.append("❌ Load model first")
            return

        user_text = self.input_box.text().strip()
        if not user_text:
            return

        self.output.append(f"🧑 {user_text}\n")

        prompt = build_prompt(
            self.mode.currentText(), user_text, self.count_input.value()
        )

        response = ""

        cursor = self.output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText("🤖 ")
        self.output.setTextCursor(cursor)

        # =========================================================
        # 🔥 GGUF STREAMING
        # =========================================================
        if self.is_gguf:
            try:
                llm = self.model["llm"]

                stream = llm(
                    prompt,
                    max_tokens=self.length_input.value(),
                    stream=True,
                    temperature=0.3,
                )

                for chunk in stream:
                    token = chunk["choices"][0]["text"]
                    response += token
                    cursor.insertText(token)
                    self.output.setTextCursor(cursor)
                    QApplication.processEvents()

            except Exception as e:
                self.output.append(f"\n❌ GGUF Error: {e}\n")
                return

        # =========================================================
        # 🔥 HF STREAMING (UNCHANGED)
        # =========================================================
        else:
            streamer = generate_stream(
                self.model, self.tokenizer, prompt, self.length_input.value()
            )

            for token in streamer:
                response += token
                cursor.insertText(token)
                self.output.setTextCursor(cursor)
                QApplication.processEvents()

        self.output.append("\n")

        chat_history.append({"role": "assistant", "content": response})

        if "SECTION 1" in response and "SECTION 2" in response:
            global last_mcq_block
            last_mcq_block = response


# -----------------------------
# MAIN
# -----------------------------
def main():
    app = QApplication(sys.argv)
    window = AIApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
