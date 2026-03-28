from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer

from major_project.config import get_model


def train_model(
    dataset_path,
    output_dir,
    epochs=2,
    lr=2e-4,
    batch_size=2,
    max_seq_length=512,
    limit=None,
    subset=None,
):
    # -----------------------------
    # LOAD MODEL FROM CONFIG
    # -----------------------------
    model_name = get_model()
    print(f"[TRAIN] Using model: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    # -----------------------------
    # APPLY LoRA
    # -----------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # -----------------------------
    # LOAD DATASET
    # -----------------------------
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    assert isinstance(dataset, Dataset)

    print(f"[TRAIN] Original dataset size: {len(dataset)}")

    # -----------------------------
    # APPLY LIMIT / SUBSET
    # -----------------------------
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
        print(f"[TRAIN] Using first {len(dataset)} samples")

    elif subset is not None:
        size = int(len(dataset) * subset)
        dataset = dataset.select(range(size))
        print(f"[TRAIN] Using {subset * 100:.1f}% → {len(dataset)} samples")

    # -----------------------------
    # FORMAT DATA
    # -----------------------------
    def format_prompt(example):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        prompt = f"{instruction}\n{input_text}".strip()

        user_part = f"""<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>

{prompt}

<|start_header_id|>assistant<|end_header_id|>

"""

        full_text = user_part + output

        user_tokens = tokenizer(user_part, add_special_tokens=False)
        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
        )

        labels = full_tokens["input_ids"].copy()
        user_len = len(user_tokens["input_ids"])
        labels[:user_len] = [-100] * user_len

        return {
            "input_ids": full_tokens["input_ids"],
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels,
        }

    dataset = dataset.map(format_prompt)

    dataset = dataset.remove_columns(
        [
            c
            for c in dataset.column_names or []
            if c not in ["input_ids", "attention_mask", "labels"]
        ]
    )

    # -----------------------------
    # TRAINER
    # -----------------------------
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=lr,
            output_dir=output_dir,
            logging_steps=10,
            save_steps=500,
            optim="adamw_8bit",
        ),
    )

    # -----------------------------
    # TRAIN
    # -----------------------------
    print("[TRAIN] Starting training...")
    trainer.train()

    # -----------------------------
    # SAVE
    # -----------------------------
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("[TRAIN] Training complete.")
