import pickle
import random
from datasets import load_from_disk

DS_PATH = "data/injected_len50_xlarge_no1000/dataset_with_traps"
INFO_PATH = "data/injected_len50_xlarge_no1000/trap_info.pkl"
OUT_PATH = "data/injected_len50_xlarge_no1000/dataset_train_balanced_500"

random.seed(777)

ds = load_from_disk(DS_PATH)

with open(INFO_PATH, "rb") as f:
    info = pickle.load(f)

trap_ids = sorted(info.index.tolist())
all_ids = list(range(len(ds)))
clean_ids = [i for i in all_ids if i not in set(trap_ids)]

sampled_clean_ids = random.sample(clean_ids, 500 - len(trap_ids))
selected_ids = trap_ids + sampled_clean_ids
random.shuffle(selected_ids)

new_ds = ds.select(selected_ids)
new_ds.save_to_disk(OUT_PATH)

print("Saved:", OUT_PATH)
print("Total docs:", len(new_ds))
print("Trap docs:", len(trap_ids))
print("Clean docs:", len(sampled_clean_ids))
