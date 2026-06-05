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
DATASET_PATH = "data/injected/dataset_with_traps_frontloaded"

epochs = int(sys.argv[1])
OUT_DIR = f"data/continued_pretraining_epoch{epochs}"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_from_disk(DATASET_PATH)

# small run first, to keep it feasible on the cluster
#dataset = dataset.select(range(80))

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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
).to("cuda")

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=epochs,
    learning_rate=2e-5,
    logging_steps=20,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    report_to="none",
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
