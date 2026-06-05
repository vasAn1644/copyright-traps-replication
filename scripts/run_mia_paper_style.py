import pickle
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils import compute_perplexity_df

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TRAP_INFO_PATH = "data/injected/trap_info.pkl"
NON_MEMBER_PATH = "data/non_member_traps/nonmember_len25_n800_ppl40.pkl"

OUT_CSV = "data/injected/mia_paper_style_results.csv"

EPOCHS = [1, 3, 5, 10]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

with open(TRAP_INFO_PATH, "rb") as f:
    trap_info = pickle.load(f)

member_texts = tokenizer.batch_decode(
    [list(map(int, x)) for x in trap_info["trap_tokens"]]
)

with open(NON_MEMBER_PATH, "rb") as f:
    non_members = pickle.load(f)

nonmember_tokens = []
for key, arr in non_members.items():
    for seq in arr:
        nonmember_tokens.append(list(map(int, seq[1:])))

nonmember_texts = tokenizer.batch_decode(nonmember_tokens)

# balance members/non-members
n = min(len(member_texts), len(nonmember_texts))
member_texts = member_texts[:n]
nonmember_texts = nonmember_texts[:n]

all_texts = member_texts + nonmember_texts
labels = np.array([1] * len(member_texts) + [0] * len(nonmember_texts))

print("Members:", len(member_texts))
print("Non-members:", len(nonmember_texts))

print("Loading reference base model...")
reference_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
).to("cuda")
reference_model.eval()

summary_rows = []

for epoch in EPOCHS:
    print(f"\n=== Evaluating continued pretraining epoch {epoch} ===")

    target_path = f"data/continued_pretraining_epoch{epoch}"

    target_model = AutoModelForCausalLM.from_pretrained(
        target_path,
        torch_dtype=torch.float16,
    ).to("cuda")
    target_model.eval()

    df_res = compute_perplexity_df(
        llama_model=reference_model,
        croissant_model=target_model,
        llama_tokenizer=tokenizer,
        croissant_tokenizer=tokenizer,
        raw_traps=all_texts,
        croissant_device="cuda",
        llama_device="cuda",
        batch_size=4,
    ).reset_index(drop=True)

    df_res["label"] = labels
    df_res["type"] = ["member"] * len(member_texts) + ["non_member"] * len(nonmember_texts)
    df_res["epoch"] = epoch

    # original-style ratio: target PPL / reference PPL
    df_res["ratio"] = df_res["croissant_ppl"] / df_res["llama_ppl"]
    df_res["log_ratio"] = np.log(df_res["croissant_ppl"]) / np.log(df_res["llama_ppl"])

    # lower ratio means target knows sequence better than reference
    auc_ratio = roc_auc_score(labels, -df_res["ratio"])
    auc_log_ratio = roc_auc_score(labels, -df_res["log_ratio"])

    # also direct target PPL attack
    auc_target_ppl = roc_auc_score(labels, -df_res["croissant_ppl"])

    member_target_ppl = df_res[df_res["label"] == 1]["croissant_ppl"].mean()
    nonmember_target_ppl = df_res[df_res["label"] == 0]["croissant_ppl"].mean()

    summary_rows.append({
        "epoch": epoch,
        "auc_ratio": auc_ratio,
        "auc_log_ratio": auc_log_ratio,
        "auc_target_ppl": auc_target_ppl,
        "member_target_ppl": member_target_ppl,
        "nonmember_target_ppl": nonmember_target_ppl,
        "member_ratio": df_res[df_res["label"] == 1]["ratio"].mean(),
        "nonmember_ratio": df_res[df_res["label"] == 0]["ratio"].mean(),
    })

    df_res.to_csv(f"data/injected/mia_paper_style_epoch{epoch}_persample.csv", index=False)

    print("AUC ratio:", auc_ratio)
    print("AUC log-ratio:", auc_log_ratio)
    print("AUC target PPL:", auc_target_ppl)
    print("Member target PPL:", member_target_ppl)
    print("Non-member target PPL:", nonmember_target_ppl)

    del target_model
    torch.cuda.empty_cache()

out = pd.DataFrame(summary_rows)
print("\n=== PAPER-STYLE MIA SUMMARY ===")
print(out)

out.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
