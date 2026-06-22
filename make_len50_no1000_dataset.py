import pickle
import pandas as pd
from datasets import load_from_disk

INJ_PATH = "data/injected_len50/dataset_with_traps"
TRAP_INFO_PATH = "data/injected_len50/trap_info.pkl"

OUT_DS = "data/injected_len50_no1000/dataset_with_traps"
OUT_INFO = "data/injected_len50_no1000/trap_info.pkl"

ds = load_from_disk(INJ_PATH)

with open(TRAP_INFO_PATH, "rb") as f:
    trap_info = pickle.load(f)

# keep only documents whose assigned trap has n_rep != 1000
keep_info = trap_info[trap_info["n_rep"] != 1000].copy()
keep_doc_ids = sorted(keep_info.index.tolist())

print("Original trap docs:", len(trap_info))
print("Kept trap docs:", len(keep_info))
print("Removed n_rep=1000 docs:", (trap_info["n_rep"] == 1000).sum())

# keep all clean docs plus trap docs without n_rep=1000
# We remove only docs that had n_rep=1000 traps.
remove_doc_ids = set(trap_info[trap_info["n_rep"] == 1000].index.tolist())

kept_indices = [i for i in range(len(ds)) if i not in remove_doc_ids]

new_ds = ds.select(kept_indices)

# remap doc_idx because dataset indices changed
old_to_new = {old: new for new, old in enumerate(kept_indices)}

keep_info = keep_info.reset_index().rename(columns={"doc_idx": "old_doc_idx"})
keep_info["doc_idx"] = keep_info["old_doc_idx"].map(old_to_new)
keep_info = keep_info.set_index("doc_idx")

import os
os.makedirs("data/injected_len50_no1000", exist_ok=True)

new_ds.save_to_disk(OUT_DS)

with open(OUT_INFO, "wb") as f:
    pickle.dump(keep_info, f)

print("Saved dataset:", OUT_DS)
print("Saved trap info:", OUT_INFO)
print("New dataset docs:", len(new_ds))
print("\nNew n_rep distribution:")
print(keep_info["n_rep"].value_counts().sort_index())
