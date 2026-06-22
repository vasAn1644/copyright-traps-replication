import torch
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import compute_perplexity

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EVAL_DS_PATH = "data/clean_eval_dataset"

models = {
    "base": MODEL_ID,
    "epoch1": "data/continued_pretraining_len50_xlarge_no1000_epoch1",
    "epoch3": "data/continued_pretraining_len50_xlarge_no1000_epoch3",
    "epoch5": "data/continued_pretraining_len50_xlarge_no1000_epoch5",
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

ds = load_from_disk(EVAL_DS_PATH)
texts = [x["text"] for x in ds]

def eval_model(model_name, model_path):
    print(f"\n=== Evaluating {model_name}: {model_path} ===")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()

    ppls = []

    for i, text in enumerate(texts):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to("cuda")

        with torch.no_grad():
            ppl = compute_perplexity(
                model,
                enc["input_ids"],
                enc["attention_mask"],
            )[0]

        ppls.append(float(ppl))

        if (i + 1) % 50 == 0:
            print(f"{model_name}: {i+1}/{len(texts)}")

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "mean_ppl": sum(ppls) / len(ppls),
        "median_ppl": pd.Series(ppls).median(),
        "min_ppl": min(ppls),
        "max_ppl": max(ppls),
    }

rows = []

for model_name, model_path in models.items():
    rows.append(eval_model(model_name, model_path))

out = pd.DataFrame(rows)

print("\n=== UTILITY PERPLEXITY RESULTS ===")
print(out)

out.to_csv(
    "data/injected_len50_xlarge_no1000/utility_perplexity_results.csv",
    index=False,
)

print(
    "\nSaved: data/injected_len50_xlarge_no1000/utility_perplexity_results.csv"
)
