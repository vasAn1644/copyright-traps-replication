import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score

TRAP_INFO_PATH = "data/injected/trap_info.pkl"
EPOCHS = [3, 10]

with open(TRAP_INFO_PATH, "rb") as f:
    trap_info = pickle.load(f)

rows = []

for epoch in EPOCHS:
    df = pd.read_csv(f"data/injected/mia_paper_style_epoch{epoch}_persample.csv")

    member_df = df[df["label"] == 1].copy().reset_index(drop=True)
    non_df = df[df["label"] == 0].copy().reset_index(drop=True)

    # member rows are in the same order as trap_info
    member_df["n_rep"] = list(trap_info["n_rep"].values[:len(member_df)])
    member_df["ppl_bucket"] = list(trap_info["ppl_bucket"].values[:len(member_df)])

    print(f"\n=== EPOCH {epoch} ===")

    for n_rep in sorted(member_df["n_rep"].unique()):
        members_tmp = member_df[member_df["n_rep"] == n_rep].copy()

        # compare with same number of non-members
        n = min(len(members_tmp), len(non_df))
        members_tmp = members_tmp.iloc[:n]
        non_tmp = non_df.iloc[:n].copy()

        combined = pd.concat([members_tmp, non_tmp], ignore_index=True)

        auc_ratio = roc_auc_score(combined["label"], -combined["ratio"])
        auc_log_ratio = roc_auc_score(combined["label"], -combined["log_ratio"])
        auc_target_ppl = roc_auc_score(combined["label"], -combined["croissant_ppl"])

        row = {
            "epoch": epoch,
            "n_rep": n_rep,
            "n_members": len(members_tmp),
            "auc_ratio": auc_ratio,
            "auc_log_ratio": auc_log_ratio,
            "auc_target_ppl": auc_target_ppl,
            "member_target_ppl_mean": members_tmp["croissant_ppl"].mean(),
            "nonmember_target_ppl_mean": non_tmp["croissant_ppl"].mean(),
            "member_ratio_mean": members_tmp["ratio"].mean(),
            "nonmember_ratio_mean": non_tmp["ratio"].mean(),
        }

        rows.append(row)

        print(
            f"n_rep={n_rep:<5} "
            f"AUC_ratio={auc_ratio:.4f} "
            f"AUC_log_ratio={auc_log_ratio:.4f} "
            f"AUC_target_ppl={auc_target_ppl:.4f} "
            f"member_ppl={row['member_target_ppl_mean']:.3f} "
            f"nonmember_ppl={row['nonmember_target_ppl_mean']:.3f}"
        )

out = pd.DataFrame(rows)
out.to_csv("data/injected/mia_by_nrep.csv", index=False)

print("\nSaved: data/injected/mia_by_nrep.csv")
