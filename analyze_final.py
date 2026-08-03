import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIMENSIONS = ["trust", "bias", "interp"]


def load_majority(class_file, meta_file, domain_label):
    with open(class_file) as f:
        class_records = json.load(f)
    df_class = pd.DataFrame(class_records)

    meta_records = []
    with open(meta_file) as f:
        for line in f:
            meta_records.append(json.loads(line))
    df_meta = pd.DataFrame(meta_records)

    df = df_meta.merge(df_class, on="id", how="inner")
    df["domain"] = domain_label
    return df


df_health = load_majority("results_majority.json", "models_metadata.jsonl", "healthcare")
df_control = load_majority("control_results_majority.json", "control_metadata.jsonl", "control")
print(f"Healthcare: {len(df_health)} | Control: {len(df_control)}")

df = pd.concat([df_health, df_control], ignore_index=True)
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
df["downloads"] = pd.to_numeric(df["downloads"], errors="coerce").fillna(0)
df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0)
df["log_downloads"] = np.log1p(df["downloads"])

for dim in DIMENSIONS:
    df[f"{dim}_gap"] = (df[f"{dim}_claim"] == "yes") & (df[f"{dim}_evidence"] == "no")
df["gap_score"] = df[[f"{d}_gap" for d in DIMENSIONS]].sum(axis=1)
df["any_claim"] = (df[[f"{d}_claim" for d in DIMENSIONS]] == "yes").any(axis=1)
df["any_gap"] = df["gap_score"] > 0
df["is_org_bool"] = df["is_organization"] == True

df.drop(columns=["card_text"], errors="ignore").to_csv("final_majority_results.csv", index=False)

dfh = df[df["domain"] == "healthcare"].copy()
dfc = df[df["domain"] == "control"].copy()

print("\n=== DESCRIPTIVE (healthcare, majority-vote, N=500) ===")
for dim in DIMENSIONS:
    claim_pct = (dfh[f"{dim}_claim"] == "yes").mean() * 100
    among = dfh[dfh[f"{dim}_claim"] == "yes"]
    evid_pct = (among[f"{dim}_evidence"] == "yes").mean() * 100 if len(among) else float("nan")
    print(f"{dim}: {claim_pct:.1f}% claim | {evid_pct:.1f}% of those with full evidence (n={len(among)})")
print(f"any_claim: {dfh['any_claim'].mean()*100:.1f}%")
print(f"any_gap: {dfh['any_gap'].mean()*100:.1f}% ({dfh['any_gap'].sum()}/{len(dfh)})")
print(f"disclaimer: {(dfh['clinical_use_disclaimer']=='yes').mean()*100:.1f}%")
print(f"interp_method_named: {dfh['interpretability_method_named'].notna().mean()*100:.1f}% ({dfh['interpretability_method_named'].notna().sum()})")
print(f"org_rate: {(dfh['is_organization']==True).mean()*100:.1f}%")

# RQ1
cutoff = pd.Timestamp("2024-08-01", tz="UTC")
dfh_dated = dfh.dropna(subset=["created_at"]).copy()
dfh_dated["period"] = dfh_dated["created_at"].apply(lambda d: "post" if d >= cutoff else "pre")
g1 = dfh_dated.groupby("period").agg(n=("id", "count"), any_claim=("any_claim", "mean"),
                                      disclaimer=("clinical_use_disclaimer", lambda x: (x == "yes").mean()))
print("\n=== RQ1 (majority-vote) ===")
print(g1)
ct1 = pd.crosstab(dfh_dated["period"], dfh_dated["any_claim"])
chi2_1, p1, _, _ = stats.chi2_contingency(ct1)
print(f"any_claim chi2={chi2_1:.3f} p={p1:.4f}")
ct1b = pd.crosstab(dfh_dated["period"], dfh_dated["clinical_use_disclaimer"] == "yes")
chi2_1b, p1b, _, _ = stats.chi2_contingency(ct1b)
print(f"disclaimer chi2={chi2_1b:.3f} p={p1b:.4f}")

# RQ2
print("\n=== RQ2 (majority-vote) ===")
for a in ["downloads", "likes"]:
    for b in ["gap_score", "any_claim"]:
        rho, p = stats.spearmanr(dfh[a], dfh[b].astype(int) if b == "any_claim" else dfh[b])
        print(f"{a} vs {b}: rho={rho:.3f} p={p:.4f}")

# RQ3
print("\n=== RQ3 (majority-vote) ===")
org = dfh[dfh["is_org_bool"] == True]
ind = dfh[dfh["is_org_bool"] == False]
print(f"Org (n={len(org)}): any_claim={org['any_claim'].mean()*100:.1f}% gap_score_mean={org['gap_score'].mean():.3f} disclaimer={(org['clinical_use_disclaimer']=='yes').mean()*100:.1f}%")
print(f"Ind (n={len(ind)}): any_claim={ind['any_claim'].mean()*100:.1f}% gap_score_mean={ind['gap_score'].mean():.3f} disclaimer={(ind['clinical_use_disclaimer']=='yes').mean()*100:.1f}%")
ct3 = pd.crosstab(dfh["is_org_bool"], dfh["any_claim"])
chi2_3, p3, _, _ = stats.chi2_contingency(ct3)
u3, pu3 = stats.mannwhitneyu(org["gap_score"], ind["gap_score"], alternative="two-sided")
print(f"any_claim chi2={chi2_3:.3f} p={p3:.4f} | gap_score MannWhitney U={u3:.1f} p={pu3:.4f}")

# RQ4
print("\n=== RQ4 (majority-vote) ===")
gdesc = df.groupby("domain").agg(any_claim=("any_claim", "mean"), org=("is_org_bool", "mean"),
                                  disclaimer=("clinical_use_disclaimer", lambda x: (x == "yes").mean()),
                                  median_dl=("downloads", "median"))
print(gdesc)
ct4 = pd.crosstab(df["domain"], df["any_claim"])
chi2_4, p4, _, _ = stats.chi2_contingency(ct4)
print(f"unadjusted any_claim~domain chi2={chi2_4:.3f} p={p4:.4f}")

model_df = df.copy()
model_df["domain_healthcare"] = (model_df["domain"] == "healthcare").astype(int)
model_df["any_claim_int"] = model_df["any_claim"].astype(int)
model_df["is_org_int"] = model_df["is_org_bool"].astype(int)
logit = smf.logit("any_claim_int ~ domain_healthcare + log_downloads + is_org_int", data=model_df).fit(disp=0)
print(logit.summary())
print("Odds ratios:", np.exp(logit.params).to_dict())
print("95% CI:\n", np.exp(logit.conf_int()))

ols = smf.ols("gap_score ~ domain_healthcare + log_downloads + is_org_int", data=model_df).fit()
print(ols.summary())
