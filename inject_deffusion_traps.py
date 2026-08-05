import argparse
import logging
import os
import pickle
import random
from itertools import cycle

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def find_best_smart_positions(
    text: str,
    trap_text: str,
    n_rep: int,
    diff_model,
    diff_tokenizer,
    device: str = "cuda",
) -> list[int]:
    """Використовує Diffusion / Masked LM для вибору найбільш семантично підходящих місць у тексті для вставки пастки."""
    words = text.split(" ")
    if len(words) <= n_rep:
        return list(range(len(words)))

    # Якщо текст занадто довгий, беремо рівномірну сітку кандидатів (наприклад, 30 потенційних точок)
    num_candidates = min(30, len(words))
    candidate_indices = np.linspace(
        0, len(words) - 1, num=num_candidates, dtype=int
    )

    candidate_scores = []
    trap_toks = diff_tokenizer.encode(trap_text, add_special_tokens=False)

    for idx in candidate_indices:
        # Формуємо контекстне вікно навколо позиції вставки
        left_context = " ".join(words[max(0, idx - 15) : idx])
        right_context = " ".join(words[idx : min(len(words), idx + 15)])

        context_text = f"{left_context} {trap_text} {right_context}".strip()
        inputs = diff_tokenizer(
            context_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = diff_model(**inputs, labels=inputs["input_ids"])
            # Чим менше loss, тим краще пастка вписується в контекст
            candidate_scores.append((outputs.loss.item(), idx))

    # Сортуємо кандидатів за зростанням loss (найкращі місця перші)
    candidate_scores.sort(key=lambda x: x[0])
    selected_indices = [idx for _, idx in candidate_scores[:n_rep]]
    return sorted(selected_indices)


def inject_one_smart(
    text: str,
    trap_text: str,
    n_rep: int,
    diff_model=None,
    diff_tokenizer=None,
    device: str = "cuda",
) -> str:
    """Вставляє пастку у вибрані Diffusion LM розумні позиції."""
    words = text.split(" ")
    effective_n_rep = min(n_rep, len(words))

    if diff_model is not None and diff_tokenizer is not None:
        trap_indices = find_best_smart_positions(
            text, trap_text, effective_n_rep, diff_model, diff_tokenizer, device
        )
    else:
        trap_indices = np.sort(
            random.sample(range(len(words)), effective_n_rep)
        )

    new_text = ""
    last_index = 0

    for idx in trap_indices:
        new_text += " ".join(words[last_index:idx])
        if idx == 0 or new_text == "":
            new_text += trap_text
        else:
            new_text += " " + trap_text
        last_index = idx

    new_text += " " + " ".join(words[last_index:])
    return new_text.strip()


def inject_all(
    df_trap_info,
    raw_dataset,
    tokenizer,
    args,
    diff_model=None,
    diff_tokenizer=None,
):
    trap_dataset_entries = []
    logging.info(
        "Injecting traps into dataset documents using Diffusion LM logic..."
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, og_entry in tqdm(enumerate(raw_dataset), total=len(raw_dataset)):
        new_entry = og_entry.copy()

        if i in df_trap_info.index:
            row = df_trap_info.loc[i]
            trap_tokens, n_rep = row["trap_tokens"], row["n_rep"]
            trap_text = tokenizer.decode(
                trap_tokens, skip_special_tokens=True
            )

            new_text = inject_one_smart(
                text=og_entry["text"],
                trap_text=trap_text,
                n_rep=n_rep,
                diff_model=diff_model,
                diff_tokenizer=diff_tokenizer,
                device=device,
            )
            new_entry["text"] = new_text

        trap_dataset_entries.append(new_entry)

    ds_dict = {
        k: [e[k] for e in trap_dataset_entries] for k in raw_dataset.column_names
    }
    return Dataset.from_dict(ds_dict)


def read_all_traps(path_to_trap_dir: str) -> tuple[dict, int]:
    all_traps = {}
    total_traps = 0

    for file in os.listdir(path_to_trap_dir):
        if not (file.endswith(".pickle") or file.endswith(".pkl")):
            continue

        filepath = os.path.join(path_to_trap_dir, file)
        with open(filepath, "rb") as f:
            traps_data = pickle.load(f)

            # 1. Якщо у файл збережено pandas DataFrame
            if isinstance(traps_data, pd.DataFrame):
                arr = np.array(traps_data["trap_tokens"].tolist())
                total_traps += len(arr)
                seq_len = arr.shape[1] - 1 if arr.ndim > 1 else len(arr[0]) - 1
                all_traps[seq_len] = {file: arr}

            # 2. Якщо збережено NumPy масив або список
            elif isinstance(traps_data, (np.ndarray, list)):
                arr = np.array(traps_data)
                total_traps += len(arr)
                seq_len = arr.shape[1] - 1
                all_traps[seq_len] = {file: arr}

            # 3. Якщо збережено словник бакетів
            elif isinstance(traps_data, dict):
                seq_len = None
                for arr in traps_data.values():
                    total_traps += len(arr)
                    if seq_len is None:
                        seq_len = arr.shape[1] - 1
                all_traps[seq_len] = traps_data

    return all_traps, total_traps


def distribute_traps(all_traps, raw_dataset, args) -> pd.DataFrame:
    min_chars = args.doc_min_tokens * 4
    doc_indices = [
        i
        for i in range(len(raw_dataset))
        if len(raw_dataset[i]["text"]) > min_chars
    ]

    if len(doc_indices) < len(raw_dataset):
        logging.warning(
            f"Filtered out {len(raw_dataset) - len(doc_indices)} documents that are too short."
        )

    random.shuffle(doc_indices)

    n_rep_iterator = cycle(args.n_reps)
    doc_idx_iterator = iter(doc_indices)
    records = []

    for seq_len in all_traps:
        for ppl_key in all_traps[seq_len]:
            ppl_bucket = (
                ppl_key[0] // 10
                if isinstance(ppl_key, tuple)
                else (ppl_key // 10 if isinstance(ppl_key, int) else 0)
            )

            for trap_tokens in all_traps[seq_len][ppl_key]:
                trap_tokens = (
                    trap_tokens[1:] if len(trap_tokens) > 0 else trap_tokens
                )  # remove BOS
                try:
                    n_rep = next(n_rep_iterator)
                    doc_idx = next(doc_idx_iterator)
                except StopIteration:
                    raise ValueError(
                        "Not enough long documents in dataset to distribute all traps!"
                    )

                records.append(
                    {
                        "doc_idx": doc_idx,
                        "doc_title": raw_dataset[doc_idx].get(
                            "title", f"doc_{doc_idx}"
                        ),
                        "seq_len": seq_len,
                        "ppl_bucket": ppl_bucket,
                        "n_rep": n_rep,
                        "trap_tokens": trap_tokens,
                    }
                )

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
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="bert-base-uncased",
        help="Diffusion / Masked LM для розумної вставки",
    )
    parser.add_argument(
        "--use-smart-injection",
        action="store_true",
        help="Увімкнути Diffusion-based розумну вставку",
    )
    parser.add_argument("--n-reps", nargs="+", type=int, default=[100])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--doc-min-tokens", default=1000, type=int)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.path_to_tokenizer)

    diff_model, diff_tokenizer = None, None
    if args.use_smart_injection:
        logging.info(
            f"Завантаження Diffusion LM для розумної вставки: {args.diffusion_model}..."
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        diff_tokenizer = AutoTokenizer.from_pretrained(args.diffusion_model)
        diff_model = AutoModelForMaskedLM.from_pretrained(
            args.diffusion_model
        ).to(device)
        diff_model.eval()

    logging.info(f"Loading raw dataset from {args.path_to_raw_dataset}...")
    dataset = load_from_disk(args.path_to_raw_dataset)

    all_traps, total_traps = read_all_traps(args.path_to_trap_dir)
    logging.info(
        f"Loaded dataset ({len(dataset)} entries) and traps ({total_traps} entries)."
    )

    df_trap_info = distribute_traps(all_traps, dataset, args)

    with open(args.output_info_path, "wb") as f:
        pickle.dump(df_trap_info, f)

    injected_dataset = inject_all(
        df_trap_info,
        dataset,
        tokenizer,
        args,
        diff_model=diff_model,
        diff_tokenizer=diff_tokenizer,
    )
    injected_dataset.save_to_disk(args.output_ds_path)
    logging.info(f"✅ Успішно збережено датасет у {args.output_ds_path}")
