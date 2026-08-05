import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Генерація Diffusion Traps у форматі для inject_traps.py"
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="bert-base-uncased",
        help="Модель для Discrete Diffusion / Masked Denoising",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Базова модель для розрахунку PPL та бакетування",
    )
    parser.add_argument(
        "--trap-len",
        type=int,
        required=True,
        help="Довжина пастки в токенах TinyLlama (10, 20, 50)",
    )
    parser.add_argument(
        "--traps-per-bucket",
        type=int,
        default=50,
        help="Кількість пасток у кожному з 10 PPL бакетів (всього буде N * 10)",
    )
    parser.add_argument(
        "--candidates-factor",
        type=int,
        default=5,
        help="Коефіцієнт генерації кандидатів",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Шлях для збереження підсумкового pkl файлу",
    )
    return parser.parse_args()


def generate_diffusion_sequence(
    model, diff_tokenizer, target_len, temperature=1.0, steps=15, device="cuda"
):
    mask_id = diff_tokenizer.mask_token_id or diff_tokenizer.unk_token_id
    seq_len = target_len + 10
    tokens = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)

    unmask_per_step = max(1, seq_len // steps)

    for step in range(steps):
        with torch.no_grad():
            outputs = model(tokens)
            logits = outputs.logits

        masked_indices = (tokens[0] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_indices) == 0:
            break

        logits_step = logits[0, masked_indices] / max(temperature, 1e-5)
        probs = torch.softmax(logits_step, dim=-1)

        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        to_unmask = min(unmask_per_step, len(masked_indices))
        perm = torch.randperm(len(masked_indices))[:to_unmask]

        for idx in perm:
            pos = masked_indices[idx]
            tokens[0, pos] = sampled[idx]

    text = diff_tokenizer.decode(tokens[0], skip_special_tokens=True).strip()
    return text


def compute_tinyllama_ppl(model, tokenizer, tokens_list, device="cuda"):
    input_ids = torch.tensor([tokens_list], dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    ppl = np.exp(loss)
    return loss, ppl


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"1. Завантаження Diffusion LM: {args.diffusion_model}")
    diff_tokenizer = AutoTokenizer.from_pretrained(args.diffusion_model)
    diff_model = AutoModelForMaskedLM.from_pretrained(
        args.diffusion_model
    ).to(device)
    diff_model.eval()

    print(f"2. Завантаження базової оціночної моделі: {args.base_model_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.bfloat16
    ).to(device)
    base_model.eval()

    bos_id = base_tokenizer.bos_token_id or 1

    total_candidates_needed = (
        args.traps_per_bucket * 10 * args.candidates_factor
    )
    print(
        f"\n3. Генерація ~{total_candidates_needed} кандидатів з різними температурами..."
    )

    candidates = []
    temperatures = np.linspace(0.5, 8.0, num=total_candidates_needed)

    for i, temp in enumerate(temperatures):
        raw_text = generate_diffusion_sequence(
            diff_model,
            diff_tokenizer,
            target_len=args.trap_len,
            temperature=temp,
            steps=15,
            device=device,
        )

        target_tokens = base_tokenizer.encode(
            raw_text, add_special_tokens=False
        )

        if len(target_tokens) >= args.trap_len:
            final_tokens = target_tokens[: args.trap_len]
            final_text = base_tokenizer.decode(final_tokens)

            loss, ppl = compute_tinyllama_ppl(
                base_model, base_tokenizer, final_tokens, device=device
            )

            candidates.append(
                {
                    "text": final_text,
                    "trap_tokens": final_tokens,
                    "loss": loss,
                    "ppl": ppl,
                    "gen_temp": temp,
                }
            )

        if (i + 1) % 500 == 0:
            print(f"Згенеровано кандидатів: {i + 1}/{total_candidates_needed}")

    df_cand = pd.DataFrame(candidates)
    df_cand = df_cand.sort_values(by="ppl").reset_index(drop=True)
    df_cand["bucket"] = pd.qcut(df_cand["ppl"], q=10, labels=False)

    # 4. Формування словника масивів для inject_traps.py
    traps_dict = {}

    for b in range(10):
        b_df = df_cand[df_cand["bucket"] == b]
        selected = b_df.sample(
            n=min(args.traps_per_bucket, len(b_df)), random_state=42
        )

        bucket_arrays = []
        for _, row in selected.iterrows():
            # Додаємо BOS токен на початок, як очікує read_all_traps
            tokens_with_bos = [bos_id] + list(row["trap_tokens"])
            bucket_arrays.append(tokens_with_bos)

        # Ключ (ppl_bucket_index, seq_len) або простий int
        traps_dict[b * 10] = np.array(bucket_arrays, dtype=np.int64)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Зберігаємо у підтримуваному форматі dict
    with open(args.output_path, "wb") as f:
        pickle.dump(traps_dict, f)

    print(
        f"\n✅ Успішно збережено {len(traps_dict)} бакетів пасток (формат dict/numpy) у: {args.output_path}"
    )


if __name__ == "__main__":
    main()
