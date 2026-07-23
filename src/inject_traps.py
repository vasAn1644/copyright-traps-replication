import argparse
import pickle
import random
import logging
from itertools import cycle
import os

import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer  # Краще AutoTokenizer замість застарілого LlamaTokenizer


def inject_one(text: str, trap_text: str, n_rep: int) -> str:
    '''
    Inject the trap sequences at random places in the original text. 
    By splitting on spaces, we ensure we don't split words from the original text.
    '''
    text_split_by_spaces = text.split(" ")
    
    # Запобігаємо помилці, якщо n_rep більше за кількість слів
    effective_n_rep = min(n_rep, len(text_split_by_spaces))
    trap_indices = np.sort(random.sample(range(len(text_split_by_spaces)), effective_n_rep))

    new_text = ''
    last_index = 0

    for idx in trap_indices:
        new_text += " ".join(text_split_by_spaces[last_index:idx])
        if idx == 0 or new_text == '':
            new_text += trap_text
        else:
            new_text += " " + trap_text
        last_index = idx
    new_text += " " + " ".join(text_split_by_spaces[last_index:])
    
    return new_text.strip()


def inject_all(df_trap_info, raw_dataset, tokenizer, args):
    trap_dataset_entries = []
    logging.info("Injecting traps into dataset documents...")

    for i, og_entry in tqdm(enumerate(raw_dataset), total=len(raw_dataset)):
        new_entry = og_entry.copy()

        if i in df_trap_info.index:
            row = df_trap_info.loc[i]
            trap_tokens, n_rep = row["trap_tokens"], row["n_rep"]
            new_text = inject_one(
                text=og_entry["text"],
                trap_text=tokenizer.decode(trap_tokens, skip_special_tokens=True),
                n_rep=n_rep
            )
            new_entry["text"] = new_text

        trap_dataset_entries.append(new_entry)

    ds_dict = {k: [e[k] for e in trap_dataset_entries] for k in raw_dataset.column_names}
    dataset = Dataset.from_dict(ds_dict)
    return dataset


def read_all_traps(path_to_trap_dir: str) -> tuple[dict, int]:
    all_traps = {}  
    total_traps = 0

    for file in os.listdir(path_to_trap_dir):
        if not (file.endswith(".pickle") or file.endswith(".pkl")):
            continue
            
        with open(os.path.join(path_to_trap_dir, file), "rb") as f:
            traps_data = pickle.load(f)
            
            # Якщо у Pickle збережений одразу NumPy масив або список
            if isinstance(traps_data, (np.ndarray, list)):
                arr = np.array(traps_data)
                total_traps += len(arr)
                seq_len = arr.shape[1] - 1  # exclude BOS
                
                # Записуємо в all_traps з ключем (file, seq_len) або за іменем файлу
                all_traps[seq_len] = {file: arr}
                
            # Якщо у Pickle збережений словник
            elif isinstance(traps_data, dict):
                seq_len = None
                for arr in traps_data.values():
                    total_traps += len(arr)
                    if seq_len is None:
                        seq_len = arr.shape[1] - 1  # exclude BOS
                    elif seq_len != arr.shape[1] - 1:
                        raise ValueError(f"Inconsistent sequence length in {file}")
                all_traps[seq_len] = traps_data
            else:
                logging.warning(f"Skipping unknown data type {type(traps_data)} in {file}")

    return all_traps, total_traps


def distribute_traps(all_traps, raw_dataset, args) -> pd.DataFrame:
    min_chars = args.doc_min_tokens * 4
    doc_indices = [i for i in range(len(raw_dataset)) if len(raw_dataset[i]["text"]) > min_chars]
    
    if len(doc_indices) < len(raw_dataset):
        logging.warning(f"Filtered out {len(raw_dataset) - len(doc_indices)} documents that are too short.")

    random.shuffle(doc_indices)
    
    # Якщо передали кілька n_reps — циклічно перебираємо, якщо один (наприклад, 100) — використовуємо його для всіх
    n_rep_iterator = cycle(args.n_reps)
    doc_idx_iterator = iter(doc_indices)
    records = []

    for seq_len in all_traps:
        for ppl_key in all_traps[seq_len]:
            # Беремо перплексію бакету
            ppl_bucket = ppl_key[0] // 10 if isinstance(ppl_key, tuple) else ppl_key // 10
            
            for trap_tokens in all_traps[seq_len][ppl_key]:
                trap_tokens = trap_tokens[1:]  # remove BOS token
                try:
                    n_rep = next(n_rep_iterator)
                    doc_idx = next(doc_idx_iterator)
                except StopIteration:
                    raise ValueError("Not enough long documents in dataset to distribute all traps!")

                records.append({
                    "doc_idx": doc_idx,
                    "doc_title": raw_dataset[doc_idx].get("title", f"doc_{doc_idx}"),
                    "seq_len": seq_len,
                    "ppl_bucket": ppl_bucket,
                    "n_rep": n_rep,
                    "trap_tokens": trap_tokens,
                })

    df = pd.DataFrame(records)
    df = df.set_index("doc_idx")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-tokenizer", type=str, required=True)
    parser.add_argument("--path-to-raw-dataset", required=True, type=str)
    parser.add_argument("--path-to-trap-dir", required=True, type=str)
    parser.add_argument("--output-ds-path", required=True, type=str)
    parser.add_argument("--output-info-path", required=True, type=str)
    parser.add_argument("--n-reps", nargs='+', type=int, default=[100], help="List of repetitions or single value (e.g. --n-reps 100)")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--nb-workers", default=64, type=int)
    parser.add_argument("--doc-min-tokens", default=5000, type=int)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.path_to_tokenizer)

    logging.info(f"Loading raw dataset from {args.path_to_raw_dataset}...")
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = load_from_disk(args.path_to_raw_dataset)
    logging.info(f"Loaded dataset with {len(dataset)} documents")

    all_traps, total_traps = read_all_traps(args.path_to_trap_dir)

    if total_traps > len(dataset):
        raise ValueError(f"Dataset is too small. len(dataset)={len(dataset)}, but found {total_traps} traps")

    logging.info(f"Read dataset({len(dataset)} entries) and trap sequences ({total_traps} entries)")

    df_trap_info = distribute_traps(all_traps, dataset, args)
    
    with open(args.output_info_path, "wb") as f:
        pickle.dump(df_trap_info, f)

    logging.info(f"Saved trap distribution info ({len(df_trap_info)} rows) to {args.output_info_path}")

    injected_dataset = inject_all(df_trap_info, dataset, tokenizer, args)
    injected_dataset.save_to_disk(args.output_ds_path)

    logging.info(f"Saved trap-injected dataset ({len(injected_dataset)} documents) to {args.output_ds_path}")
