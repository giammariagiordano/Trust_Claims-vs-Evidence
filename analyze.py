import json
import glob
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Load classification results ----
class_records = []
for fpath in sorted(glob.glob("results/result_*.json")):
    with open(fpath) as f:
        class_records.extend(json.load(f))
df_class = pd.DataFrame(class_records)

# ---- Load metadata ----
meta_records = []
with open("models_metadata.jsonl") as f:
    for line in f:
        meta_records.append(json.loads(line))
df_meta = pd.DataFrame(meta_records)

df = df_meta.merge(df_class, on="id", how="inner")
print(f"Merged dataset: {len(df)} models (metadata={len(df_meta)}, classified={len(df_class)})")

df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
df["downloads"] = pd.to_numeric(df["downloads"], errors="coerce").fillna(0)
df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0)

DIMENSIONS = ["trust", "bias", "interp"]

for dim in DIMENSIONS:
    df[f"{dim}_gap"] = (df[f"{dim}_claim"] == "yes") & (df[f"{dim}_evidence"] == "no")

df["gap_score"] = df[[f"{d}_gap" for d in DIMENSIONS]].sum(axis=1)
df["any_claim"] = (df[[f"{d}_claim" for d in DIMENSIONS]] == "yes").any(axis=1)
df["any_gap"] = df["gap_score"] > 0

# ================= Descriptive stats =================
print("\n=== DESCRIPTIVE STATS (N=%d) ===" % len(df))
for dim in DIMENSIONS:
    claim_pct = (df[f"{dim}_claim"] == "yes").mean() * 100
    among_claims = df[df[f"{dim}_claim"] == "yes"]
    evid_yes_pct = (among_claims[f"{dim}_evidence"] == "yes").mean() * 100 if len(among_claims) else float("nan")
    print(f"{dim}: {claim_pct:.1f}% make a claim | of those, {evid_yes_pct:.1f}% have full evidence "
          f"(n_claims={len(among_claims)})")

print(f"\nAny claim on any dimension: {df['any_claim'].mean()*100:.1f}%")
print(f"Any strict gap (claim w/o evidence): {df['any_gap'].mean()*100:.1f}% ({df['any_gap'].sum()}/{len(df)})")
print(f"Clinical-use disclaimer present: {(df['clinical_use_disclaimer']=='yes').mean()*100:.1f}%")
print(f"Interpretability method explicitly named: {df['interpretability_method_named'].notna().mean()*100:.1f}% "
      f"({df['interpretability_method_named'].notna().sum()} models)")
print(f"Organization-authored: {(df['is_organization']==True).mean()*100:.1f}%")

# ================= RQ1: temporal trend =================
df_dated = df.dropna(subset=["created_at"]).copy()
df_dated["year_quarter"] = df_dated["created_at"].dt.to_period("Q").astype(str)
trend = df_dated.groupby("year_quarter").agg(
    n=("id", "count"),
    any_claim_rate=("any_claim", "mean"),
    gap_rate=("any_gap", "mean"),
    disclaimer_rate=("clinical_use_disclaimer", lambda x: (x == "yes").mean()),
).reset_index()
trend.to_csv("rq1_trend.csv", index=False)
print(f"\nRQ1: temporal trend saved to rq1_trend.csv ({len(trend)} quarters, date range "
      f"{df_dated['created_at'].min().date()} to {df_dated['created_at'].max().date()})")

fig, ax = plt.subplots(figsize=(10, 5))
trend_plot = trend[trend["n"] >= 3]  # avoid noisy tiny bins
ax.plot(trend_plot["year_quarter"], trend_plot["any_claim_rate"] * 100, marker="o", label="Any trust/bias/interp claim (%)")
ax.plot(trend_plot["year_quarter"], trend_plot["disclaimer_rate"] * 100, marker="s", label="Clinical-use disclaimer (%)")
for milestone, label in [("2021Q2", "EU AI Act proposed"), ("2023Q4", "Political agreement"), ("2024Q3", "Entry into force")]:
    if milestone in trend_plot["year_quarter"].values:
        ax.axvline(x=milestone, color="gray", linestyle="--", alpha=0.6)
        ax.text(milestone, ax.get_ylim()[1]*0.95, label, rotation=90, fontsize=7, va="top")
ax.set_xlabel("Quarter (model creation date)")
ax.set_ylabel("% of models")
ax.set_title("RQ1: Documentation trends over time vs. EU AI Act milestones")
ax.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("rq1_trend.png", dpi=150)
print("RQ1 plot saved to rq1_trend.png")

# ================= RQ2: popularity vs. gap/claim =================
df["log_downloads"] = np.log1p(df["downloads"])
df["log_likes"] = np.log1p(df["likes"])

rho_dl_gap, p_dl_gap = stats.spearmanr(df["downloads"], df["gap_score"])
rho_lk_gap, p_lk_gap = stats.spearmanr(df["likes"], df["gap_score"])
rho_dl_claim, p_dl_claim = stats.spearmanr(df["downloads"], df["any_claim"].astype(int))
rho_lk_claim, p_lk_claim = stats.spearmanr(df["likes"], df["any_claim"].astype(int))

print("\n=== RQ2: popularity vs. documentation ===")
print(f"Spearman(downloads, gap_score) = {rho_dl_gap:.3f} (p={p_dl_gap:.4f})")
print(f"Spearman(likes, gap_score)     = {rho_lk_gap:.3f} (p={p_lk_gap:.4f})")
print(f"Spearman(downloads, any_claim) = {rho_dl_claim:.3f} (p={p_dl_claim:.4f})")
print(f"Spearman(likes, any_claim)     = {rho_lk_claim:.3f} (p={p_lk_claim:.4f})")

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df["log_downloads"], df["gap_score"] + np.random.uniform(-0.08, 0.08, len(df)), alpha=0.4, s=15)
ax.set_xlabel("log(1 + downloads)")
ax.set_ylabel("gap_score (jittered)")
ax.set_title("RQ2: Popularity vs. claim-evidence gap score")
plt.tight_layout()
plt.savefig("rq2_popularity_gap.png", dpi=150)
print("RQ2 plot saved to rq2_popularity_gap.png")

# ================= RQ3: organization vs individual =================
df_org = df[df["is_organization"].notna()].copy()
org_group = df_org[df_org["is_organization"] == True]
ind_group = df_org[df_org["is_organization"] == False]

print(f"\n=== RQ3: organization (n={len(org_group)}) vs individual (n={len(ind_group)}) ===")
print(f"Org  - any_claim: {org_group['any_claim'].mean()*100:.1f}% | gap_score mean: {org_group['gap_score'].mean():.3f} | disclaimer: {(org_group['clinical_use_disclaimer']=='yes').mean()*100:.1f}%")
print(f"Indiv- any_claim: {ind_group['any_claim'].mean()*100:.1f}% | gap_score mean: {ind_group['gap_score'].mean():.3f} | disclaimer: {(ind_group['clinical_use_disclaimer']=='yes').mean()*100:.1f}%")

if len(org_group) > 0 and len(ind_group) > 0:
    u_stat, p_val = stats.mannwhitneyu(org_group["gap_score"], ind_group["gap_score"], alternative="two-sided")
    print(f"Mann-Whitney U (gap_score, org vs indiv): U={u_stat:.1f}, p={p_val:.4f}")

    contingency = pd.crosstab(df_org["is_organization"], df_org["any_claim"])
    chi2, p_chi2, _, _ = stats.chi2_contingency(contingency)
    print(f"Chi-square (any_claim ~ is_organization): chi2={chi2:.3f}, p={p_chi2:.4f}")
    print(contingency)

# ================= Save full merged dataset =================
df.drop(columns=["card_text"], errors="ignore").to_csv("full_results.csv", index=False)
print("\nFull merged dataset saved to full_results.csv")
