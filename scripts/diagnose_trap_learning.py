import pickle
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import compute_perplexity_df

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TRAP_INFO_PATH = "data/injected/trap_info.pkl"
NON_MEMBER_PATH = "data/non_member_traps/nonmember_len25_n800_ppl40.pkl"

OUT_CSV = "data/injected/trap_learning_diagnostic.csv"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

with open(TRAP_INFO_PATH, "rb") as f:
    trap_info = pickle.load(f)

# take diagnostic subset: strongest repetitions first
trap_info_sorted = trap_info.sort_values("n_rep", ascending=False)

member_rows = trap_info_sorted.head(799).copy()
member_texts = tokenizer.batch_decode(
    [list(map(int, x)) for x in member_rows["trap_tokens"]]
)

with open(NON_MEMBER_PATH, "rb") as f:
    non_members = pickle.load(f)

nonmember_tokens = []
for key, arr in non_members.items():
    for seq in arr:
        nonmember_tokens.append(list(map(int, seq[1:])))

nonmember_tokens = nonmember_tokens[:799]
nonmember_texts = tokenizer.batch_decode(nonmember_tokens)

all_texts = member_texts + nonmember_texts
types = ["member"] * len(member_texts) + ["non_member"] * len(nonmember_texts)

print("Diagnostic members:", len(member_texts))
print("Diagnostic non-members:", len(nonmember_texts))

models = {
    "base": MODEL_ID,
    "epoch1": "data/continued_pretraining_epoch1",
    "epoch3": "data/continued_pretraining_epoch3",
    "epoch5": "data/continued_pretraining_epoch5",
    "epoch10": "data/continued_pretraining_epoch10",
}

rows = []

for model_name, model_path in models.items():
    print(f"\n=== Loading {model_name}: {model_path} ===")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()

    # We compare model to itself only to reuse compute_perplexity_df cleanly.
    df = compute_perplexity_df(
        llama_model=model,
        croissant_model=model,
        llama_tokenizer=tokenizer,
        croissant_tokenizer=tokenizer,
        raw_traps=all_texts,
        llama_device="cuda",
        croissant_device="cuda",
        batch_size=4,
    ).reset_index(drop=True)

    for i, row in df.iterrows():
        rows.append({
            "model": model_name,
            "type": types[i],
            "idx": i,
            "ppl": row["croissant_ppl"],
            "text_preview": repr(all_texts[i][:80]),
        })

    del model
    torch.cuda.empty_cache()

out = pd.DataFrame(rows)

print("\n=== MEAN PPL BY MODEL AND TYPE ===")
summary = out.groupby(["model", "type"])["ppl"].agg(["mean", "median", "min", "max"]).reset_index()
print(summary.to_string(index=False))

out.to_csv(OUT_CSV, index=False)
summary.to_csv("data/injected/trap_learning_diagnostic_summary.csv", index=False)

print(f"\nSaved detailed: {OUT_CSV}")
print("Saved summary: data/injected/trap_learning_diagnostic_summary.csv")

