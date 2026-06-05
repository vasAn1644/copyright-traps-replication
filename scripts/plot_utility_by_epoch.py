import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("data/injected")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Utility results
utility = pd.read_csv("data/injected/utility_perplexity_results.csv")

# Expected columns: model, mean_ppl, median_ppl, min_ppl, max_ppl
def model_to_epoch(model_name):
    if model_name == "base":
        return 0
    if model_name.startswith("epoch"):
        return int(model_name.replace("epoch", ""))
    raise ValueError(f"Unknown model name: {model_name}")

utility["epoch"] = utility["model"].apply(model_to_epoch)
utility = utility.sort_values("epoch")

print("\n=== UTILITY DATA USED FOR PLOT ===")
print(utility[["epoch", "model", "mean_ppl", "median_ppl", "min_ppl", "max_ppl"]])

utility.to_csv(OUT_DIR / "utility_by_epoch_plot_data.csv", index=False)

plt.figure(figsize=(9, 5.5))

plt.plot(
    utility["epoch"],
    utility["mean_ppl"],
    marker="o",
    linewidth=2,
    label="Mean clean-text PPL"
)

plt.plot(
    utility["epoch"],
    utility["median_ppl"],
    marker="s",
    linewidth=2,
    linestyle="--",
    label="Median clean-text PPL"
)

plt.xlabel("Continued pretraining epoch")
plt.ylabel("Clean evaluation perplexity")
plt.title("Utility remains stable during continued pretraining")
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_path = OUT_DIR / "utility_by_epoch.png"
plt.savefig(out_path, dpi=300)
plt.close()

print(f"\nSaved plot: {out_path}")
print(f"Saved data: {OUT_DIR / 'utility_by_epoch_plot_data.csv'}")
