import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("data/injected")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Main MIA results
mia = pd.read_csv("data/injected/mia_paper_style_results.csv")

# Diagnostic PPL results
diag = pd.read_csv("data/injected/trap_learning_diagnostic_summary.csv")

# Keep only base, epoch3, epoch10 from diagnostics
# Convert model labels to numeric epoch values
def model_to_epoch(model_name):
    if model_name == "base":
        return 0
    if model_name.startswith("epoch"):
        return int(model_name.replace("epoch", ""))
    raise ValueError(f"Unknown model name: {model_name}")

diag["epoch"] = diag["model"].apply(model_to_epoch)

# Extract member/non-member mean PPL
ppl_pivot = diag.pivot(index="epoch", columns="type", values="mean").reset_index()
ppl_pivot = ppl_pivot.rename(columns={
    "member": "member_ppl",
    "non_member": "nonmember_ppl"
})

# Add base AUC = 0.5 as random baseline
base_row = pd.DataFrame([{
    "epoch": 0,
    "auc_ratio": 0.5,
    "auc_log_ratio": 0.5,
    "auc_target_ppl": 0.5
}])

mia_plot = pd.concat([base_row, mia], ignore_index=True)
mia_plot = mia_plot[["epoch", "auc_ratio", "auc_log_ratio"]]

# Merge
df = pd.merge(mia_plot, ppl_pivot, on="epoch", how="inner")
df = df.sort_values("epoch")

print("\n=== DATA USED FOR PLOT ===")
print(df)

# Save combined data
df.to_csv(OUT_DIR / "memorization_by_epoch_plot_data.csv", index=False)

# Plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left y-axis: AUC
ax1.plot(
    df["epoch"],
    df["auc_ratio"],
    marker="o",
    linewidth=2,
    label="MIA AUC (ratio attack)"
)
ax1.plot(
    df["epoch"],
    df["auc_log_ratio"],
    marker="o",
    linewidth=2,
    linestyle="--",
    label="MIA AUC (log-ratio attack)"
)

ax1.axhline(
    y=0.5,
    linestyle=":",
    linewidth=1.5,
    label="Random guessing baseline"
)

ax1.set_xlabel("Continued pretraining epoch")
ax1.set_ylabel("MIA ROC AUC")
ax1.set_ylim(0.45, 1.05)

# Right y-axis: PPL
ax2 = ax1.twinx()
ax2.plot(
    df["epoch"],
    df["member_ppl"],
    marker="s",
    linewidth=2,
    label="Member trap PPL"
)
ax2.plot(
    df["epoch"],
    df["nonmember_ppl"],
    marker="s",
    linewidth=2,
    linestyle="--",
    label="Non-member trap PPL"
)

ax2.set_ylabel("Mean perplexity")

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc="center right",
    frameon=True
)

plt.title("Memorization increases during continued pretraining")
plt.tight_layout()

out_path = OUT_DIR / "memorization_by_epoch.png"
plt.savefig(out_path, dpi=300)
plt.close()

print(f"\nSaved plot: {out_path}")
print(f"Saved data: {OUT_DIR / 'memorization_by_epoch_plot_data.csv'}")
