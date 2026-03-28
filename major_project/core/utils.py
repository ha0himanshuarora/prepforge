import threading
from transformers import TextIteratorStreamer

# -----------------------------
# MEMORY (SHARED)
# -----------------------------
chat_history = []
last_mcq_block = None
MAX_HISTORY = 6


def trim_history():
    global chat_history
    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[-MAX_HISTORY:]


# -----------------------------
# PROMPT BUILDER (SHARED)
# -----------------------------
def build_prompt(mode, user_input, count):
    global last_mcq_block

    user_lower = user_input.lower()

    # -------------------------
    # 🔥 MCQ FOLLOWUPS
    # -------------------------
    if last_mcq_block:
        # ONLY ANSWERS
        if (
            "answer" in user_lower
            and "why" not in user_lower
            and "explain" not in user_lower
        ):
            return f"""Here are the MCQs:

{last_mcq_block}

Give ONLY the answers in this format:
1. A
2. B
3. C
(continue...)
"""

        # EXPLANATION MODE
        if "why" in user_lower or "explain" in user_lower:
            return f"""Here are the MCQs:

{last_mcq_block}

For EACH question:
- Give the correct answer
- Explain WHY it is correct
- Briefly explain why other options are incorrect

FORMAT:

1. Answer: A
Explanation: ...

2. Answer: B
Explanation: ...
"""

    # -------------------------
    # CHAT
    # -------------------------
    if mode in ["chat", "Chat"]:
        return user_input

    # -------------------------
    # NOTES
    # -------------------------
    elif mode in ["notes", "Notes"]:
        return f"""
Explain {user_input} in structured notes.

Use:
- Clear headings
- Bullet points
- Simple explanations
"""

    # -------------------------
    # MCQ GENERATION (FIXED)
    # -------------------------
    elif mode in ["mcq", "MCQ Generator"]:
        return f"""
Generate EXACTLY {count} multiple choice questions on {user_input}.

STRICT RULES:
- Generate EXACTLY {count} questions (no more, no less)
- Each question MUST have 4 options: A, B, C, D
- Each option MUST be on a new line
- Do NOT combine options in one line
- Do NOT stop early
- Do NOT skip numbers

FORMAT:

SECTION 1: QUESTIONS

1. Question text
A. Option
B. Option
C. Option
D. Option

2. Question text
A. Option
B. Option
C. Option
D. Option

(continue until {count})

SECTION 2: ANSWERS

1. A
2. B
3. C
4. D
(continue until {count})

IMPORTANT:
- SECTION 1 must contain ALL {count} questions
- SECTION 2 must contain ALL {count} answers
- DO NOT include explanations
- DO NOT add extra commentary
"""

    # -------------------------
    # PRACTICE QUESTIONS
    # -------------------------
    elif mode in ["practice", "Practice"]:
        return f"""
Generate EXACTLY {count} practice questions on {user_input}.

RULES:
- These must be descriptive or short-answer questions
- DO NOT generate MCQs
- DO NOT provide answers initially
- Keep questions clear and numbered

FORMAT:

1. Question
2. Question
3. Question
(continue...)
"""

    # -------------------------
    # PRACTICE FOLLOWUP (EXPLANATION)
    # -------------------------
    if mode in ["practice", "Practice"] and (
        "answer" in user_lower or "explain" in user_lower or "why" in user_lower
    ):
        return f"""
Here are the practice questions:

{user_input}

Provide detailed answers and explanations for each question.
"""

    # -------------------------
    # STUDY PLAN
    # -------------------------
    elif mode in ["plan", "Study Plan"]:
        return f"""
Create a structured study plan for {user_input}.

Include:
- Daily/weekly breakdown
- Topics to cover
- Practice suggestions
"""

    return user_input


# -----------------------------
# STREAM GENERATION (SHARED)
# -----------------------------
def generate_stream(model, tokenizer, prompt, max_tokens):
    global chat_history

    chat_history.append({"role": "user", "content": prompt})
    trim_history()

    messages = [{"role": "system", "content": "You are a helpful AI teacher."}]
    messages.extend(chat_history)

    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    def run():
        model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    threading.Thread(target=run).start()
    return streamer
