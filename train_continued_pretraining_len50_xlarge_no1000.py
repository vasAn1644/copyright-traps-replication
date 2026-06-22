import sys
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = "data/injected_len50_xlarge_no1000/dataset_train_balanced_500"

if len(sys.argv) != 2:
    raise ValueError("Usage: python train_continued_pretraining_len50_xlarge_no1000.py <epochs>")

epochs = int(sys.argv[1])
OUT_DIR = f"data/continued_pretraining_len50_xlarge_no1000_epoch{epochs}"

print("Model:", MODEL_ID)
print("Dataset:", DATASET_PATH)
print("Epochs:", epochs)
print("Output:", OUT_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_from_disk(DATASET_PATH)

print("Dataset size:", len(dataset))
print("Columns:", dataset.column_names)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding=False,
    )

tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset.column_names,
)

print("Tokenized size:", len(tokenized))

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
).to("cuda")

model.config.use_cache = False

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=epochs,
    learning_rate=2e-5,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
    fp16=False,
    report_to="none",
    remove_unused_columns=False,
)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=collator,
)

trainer.train()

trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print(f"Saved continued-pretrained model to {OUT_DIR}")
