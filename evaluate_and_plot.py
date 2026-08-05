import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Оцінка меморизації Diffusion Traps по бакетах"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Шлях до тренованої моделі (наприклад, models/tinyllama_diffusion_traps_len50_rep100)",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Шлях до базової моделі для порівняння",
    )
    parser.add_argument(
        "--trap-info-path",
        type=str,
        required=True,
        help="Шлях до pkl з інфо про ін'єктовані пастки (data/diffusion_traps_info_len50.pkl)",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="results/diffusion_traps_eval_len50.png",
        help="Шлях для збереження графіку",
    )
    return parser.parse_args()


def compute_loss(model, tokenizer, trap_tokens, device="cuda"):
    """Обчислення Loss пастки під заданою моделлю."""
    input_ids = torch.tensor([trap_tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        return outputs.loss.item()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"1. Завантаження інформації про пастки: {args.trap_info_path}")
    with open(args.trap_info_path, "rb") as f:
        df_traps = pickle.load(f)

    print(f"2. Завантаження тренованої моделі (Member): {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    trained_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to(device)
    trained_model.eval()

    print(
        f"3. Завантаження базової моделі (Non-Member): {args.base_model_path}"
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.bfloat16
    ).to(device)
    base_model.eval()

    print("\n4. Обчислення Loss для кожної пастки...")
    trained_losses = []
    base_losses = []

    for _, row in df_traps.iterrows():
        tokens = row["trap_tokens"]
        t_loss = compute_loss(trained_model, tokenizer, tokens, device=device)
        b_loss = compute_loss(base_model, tokenizer, tokens, device=device)

        trained_losses.append(t_loss)
        base_losses.append(b_loss)

    df_traps["trained_loss"] = trained_losses
    df_traps["base_loss"] = base_losses
    df_traps["loss_drop"] = df_traps["base_loss"] - df_traps["trained_loss"]

    # Агрегація по бакетах
    summary = (
        df_traps.groupby("bucket")
        .agg(
            Base_Loss=("base_loss", "mean"),
            Trained_Loss=("trained_loss", "mean"),
            Loss_Drop=("loss_drop", "mean"),
        )
        .reset_index()
    )

    print("\n📊 Результати оцінки Diffusion Traps по бакетах:")
    print(summary.to_string(index=False))

    # Візуалізація
    os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    plt.plot(
        summary["bucket"],
        summary["Base_Loss"],
        marker="o",
        linewidth=2.5,
        label="Base Model (Non-Member)",
        color="crimson",
    )
    plt.plot(
        summary["bucket"],
        summary["Trained_Loss"],
        marker="s",
        linewidth=2.5,
        label="Trained Model (100 Reps Member)",
        color="royalblue",
    )

    plt.title(
        "Diffusion Traps Memorization (Loss vs PPL Bucket)",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("PPL Bucket (0 = Low PPL/Normal, 9 = High PPL/Nonsense)")
    plt.ylabel("Cross-Entropy Loss")
    plt.xticks(range(10))
    plt.legend()
    plt.tight_layout()

    plt.savefig(args.output_plot, dpi=300)
    print(f"\n✅ Графік успішно збережено у: {args.output_plot}")


if __name__ == "__main__":
    main()
