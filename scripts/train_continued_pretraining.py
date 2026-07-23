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

# 1. ВИПРАВЛЕНО: Вказуємо саме ту модель, для якої робилася ін'єкція пасток
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 2. ВИПРАВЛЕНО: Шлях до нового датасету з 100 повтореннями
DATASET_PATH = "data/injected_dataset_100reps"

if len(sys.argv) != 2:
    raise ValueError("Usage: python train_continued_pretraining.py <epochs>")

epochs = int(sys.argv[1])

# 3. ВИПРАВЛЕНО: Зрозуміла назва папки виходу під новий експеримент
OUT_DIR = f"data/continued_pretraining_100reps_epoch{epochs}"

print("==========================================")
print("Model:", MODEL_ID)
print("Dataset:", DATASET_PATH)
print("Epochs:", epochs)
print("Output:", OUT_DIR)
print("==========================================")

# 4. Завантаження токенізатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 5. Завантаження датасету
dataset = load_from_disk(DATASET_PATH)

print("Dataset size:", len(dataset))
print("Columns:", dataset.column_names)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,  # Або 1024/2048 залежно від довжини ваших документів
        padding=False,
    )

print("Tokenizing dataset...")
tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Running tokenizer on dataset"
)

print("Tokenized size:", len(tokenized))

# 6. Завантаження моделі
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)

model.config.use_cache = False

# 7. Параметри тренування
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=4,  # Для TinyLlama можна ставити 4 або 8 на GPU WatGPU
    gradient_accumulation_steps=2,   # Ефективний батч = 4 * 2 = 8
    num_train_epochs=epochs,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,              # Бережемо диск на кластері
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
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

print("🚀 Starting training...")
trainer.train()

# 8. Збереження фінальної моделі
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print(f"✅ Saved continued-pretrained model to {OUT_DIR}")
