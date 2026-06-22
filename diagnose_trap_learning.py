import pickle
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import compute_perplexity_df

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TRAP_INFO_PATH = "data/injected_len50_xlarge_no1000/trap_info.pkl"
NON_MEMBER_PATH = "data/non_member_traps_len50/nonmember_len50_n200_ppl40.pkl"

OUT_DIR = "data/injected_len50_xlarge_no1000"
OUT_CSV = f"{OUT_DIR}/trap_learning_diagnostic.csv"
OUT_SUMMARY = f"{OUT_DIR}/trap_learning_diagnostic_summary.csv"

EPOCHS = [1, 3, 5]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

with open(TRAP_INFO_PATH, "rb") as f:
    trap_info = pickle.load(f)

# For this xlarge no1000 setup we have 150 member traps:
# n_rep = 1, 10, 100, with 50 traps per group.
trap_info_sorted = trap_info.sort_values(["n_rep", "ppl_bucket"], ascending=[False, True])

member_rows = trap_info_sorted.copy()
member_texts = tokenizer.batch_decode(
    [list(map(int, x)) for x in member_rows["trap_tokens"]]
)

with open(NON_MEMBER_PATH, "rb") as f:
    non_members = pickle.load(f)

nonmember_tokens = []
for key, arr in non_members.items():
    for seq in arr:
        ids = list(map(int, seq))
        if len(ids) > 0 and ids[0] == tokenizer.bos_token_id:
            ids = ids[1:]
        nonmember_tokens.append(ids)

# Balance non-members to number of members
nonmember_tokens = nonmember_tokens[:len(member_texts)]
nonmember_texts = tokenizer.batch_decode(nonmember_tokens)

all_texts = member_texts + nonmember_texts
types = ["member"] * len(member_texts) + ["non_member"] * len(nonmember_texts)

print("Diagnostic members:", len(member_texts))
print("Diagnostic non-members:", len(nonmember_texts))
print("Member n_rep distribution:")
print(member_rows["n_rep"].value_counts().sort_index())
print("Member bucket distribution:")
print(member_rows[["ppl_bucket", "n_rep"]].value_counts().sort_index())

models = {
    "base": MODEL_ID,
}

for epoch in EPOCHS:
    models[f"epoch{epoch}"] = f"data/continued_pretraining_len50_xlarge_no1000_epoch{epoch}"

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
summary = out.groupby(["model", "type"])["ppl"].agg(
    ["mean", "median", "min", "max", "std"]
).reset_index()
print(summary.to_string(index=False))

out.to_csv(OUT_CSV, index=False)
summary.to_csv(OUT_SUMMARY, index=False)

print(f"\nSaved detailed: {OUT_CSV}")
print(f"Saved summary: {OUT_SUMMARY}")
