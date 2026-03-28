import os

# -----------------------------
# 🔥 FORCE OFFLINE MODE
# -----------------------------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import streamlit as st

# 🔥 CORE IMPORTS
from major_project.core.model import load_model
from major_project.core.utils import (
    build_prompt,
    generate_stream,
    chat_history,
    last_mcq_block,
)
from major_project.config import DEFAULT_APP_NAME, DEFAULT_LORA, DEFAULT_MODEL

# -----------------------------
# CONFIG
# -----------------------------
APP_NAME = DEFAULT_APP_NAME
st.set_page_config(page_title=APP_NAME, layout="wide")


# -----------------------------
# MODEL CACHE
# -----------------------------
@st.cache_resource
def cached_model(base_model, lora_path):
    return load_model(base_model, lora_path)


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Model Settings")

model_name = DEFAULT_MODEL
model_mode = st.sidebar.selectbox("Model Type", ["Base Model", "LoRA Adapter"])

lora_path = DEFAULT_LORA
if model_mode == "LoRA Adapter":
    lora_path = st.sidebar.text_input("LoRA Path")

load_btn = st.sidebar.button("🚀 Load Model")


# -----------------------------
# SESSION STATE
# -----------------------------
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None

if "history" not in st.session_state:
    st.session_state.history = []

if "last_mcq" not in st.session_state:
    st.session_state.last_mcq = ""


# -----------------------------
# LOAD MODEL
# -----------------------------
if load_btn:
    if model_mode == "LoRA Adapter" and not lora_path:
        st.sidebar.error("Provide LoRA path")
    else:
        with st.spinner("Loading model..."):
            model, tokenizer = cached_model(
                model_name,
                lora_path if model_mode == "LoRA Adapter" else None,
            )
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer

        st.sidebar.success("Model loaded")


model = st.session_state.model
tokenizer = st.session_state.tokenizer


# -----------------------------
# SAFETY
# -----------------------------
if model is None or tokenizer is None:
    st.title(f"🎓 {APP_NAME}")
    st.warning("Load model from sidebar")
    st.stop()


# -----------------------------
# UI
# -----------------------------
st.title(f"🎓 {APP_NAME}")

mode = st.sidebar.selectbox(
    "Mode", ["Chat", "Notes", "Study Plan", "MCQ Generator", "Practice"]
)

length = st.sidebar.slider("Response Length", 100, 2000, 600)
count = st.sidebar.number_input("Number of Questions", 1, 50, 5)


# -----------------------------
# RESET
# -----------------------------
if st.sidebar.button("🧹 Reset Chat"):
    st.session_state.history = []
    chat_history.clear()
    st.session_state.last_mcq = ""


# -----------------------------
# DISPLAY HISTORY
# -----------------------------
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -----------------------------
# INPUT
# -----------------------------
user_input = st.chat_input(f"Ask {APP_NAME}...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    # 🔥 USE SHARED PROMPT BUILDER
    prompt = build_prompt(mode, user_input, count)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response = ""

        # 🔥 USE SHARED STREAM ENGINE
        streamer = generate_stream(model, tokenizer, prompt, length)

        for token in streamer:
            response += token
            placeholder.markdown(response)

    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": response})

    # 🔥 MCQ STORAGE
    if "SECTION 1" in response and "SECTION 2" in response:
        st.session_state.last_mcq = response


# -----------------------------
# SAVE CHAT
# -----------------------------
if st.session_state.history:
    chat_text = f"=== {APP_NAME} Notes ===\n\n"

    for msg in st.session_state.history:
        role = "User" if msg["role"] == "user" else APP_NAME
        chat_text += f"{role}:\n{msg['content']}\n\n"

    st.sidebar.download_button(
        "💾 Download Chat",
        chat_text,
        file_name=f"{APP_NAME.lower()}_chat.txt",
    )
