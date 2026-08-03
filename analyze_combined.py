import json
import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

def load_group(class_globs, meta_file, domain_label):
    class_records = []
    for class_glob in class_globs:
        for fpath in sorted(glob.glob(class_glob)):
            with open(fpath) as f:
                class_records.extend(json.load(f))
    df_class = pd.DataFrame(class_records)

    meta_records = []
    with open(meta_file) as f:
        for line in f:
            meta_records.append(json.loads(line))
    df_meta = pd.DataFrame(meta_records)

    df = df_meta.merge(df_class, on="id", how="inner")
    df["domain"] = domain_label
    return df

df_health = load_group(["results/result_*.json"], "models_metadata.jsonl", "healthcare")
df_control = load_group(
    ["control_results/cresult_*.json", "control_results_extra/ceresult_*.json"],
    "control_metadata.jsonl", "control"
)

print(f"Healthcare: {len(df_health)} | Control: {len(df_control)}")

df = pd.concat([df_health, df_control], ignore_index=True)
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
df["downloads"] = pd.to_numeric(df["downloads"], errors="coerce").fillna(0)
df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0)
df["log_downloads"] = np.log1p(df["downloads"])

DIMENSIONS = ["trust", "bias", "interp"]
for dim in DIMENSIONS:
    df[f"{dim}_gap"] = (df[f"{dim}_claim"] == "yes") & (df[f"{dim}_evidence"] == "no")

df["gap_score"] = df[[f"{d}_gap" for d in DIMENSIONS]].sum(axis=1)
df["any_claim"] = (df[[f"{d}_claim" for d in DIMENSIONS]] == "yes").any(axis=1)
df["any_gap"] = df["gap_score"] > 0
df["is_org_bool"] = df["is_organization"] == True

df.drop(columns=["card_text"], errors="ignore").to_csv("combined_results.csv", index=False)

# ================= Descriptive: domain comparison =================
print("\n=== DOMAIN COMPARISON (descriptive) ===")
g = df.groupby("domain").agg(
    n=("id", "count"),
    any_claim_rate=("any_claim", "mean"),
    gap_rate=("any_gap", "mean"),
    mean_gap_score=("gap_score", "mean"),
    disclaimer_rate=("clinical_use_disclaimer", lambda x: (x == "yes").mean()),
    org_rate=("is_org_bool", "mean"),
    median_downloads=("downloads", "median"),
)
print(g)

ct = pd.crosstab(df["domain"], df["any_claim"])
chi2, p, _, _ = stats.chi2_contingency(ct)
print(f"\nChi-square (any_claim ~ domain), UNADJUSTED: chi2={chi2:.3f}, p={p:.4f}")
print(ct)

# ================= Multivariate logistic regression =================
print("\n=== MULTIVARIATE LOGISTIC REGRESSION: any_claim ~ domain + log_downloads + is_organization ===")
model_df = df.copy()
model_df["domain_healthcare"] = (model_df["domain"] == "healthcare").astype(int)
model_df["any_claim_int"] = model_df["any_claim"].astype(int)
model_df["is_org_int"] = model_df["is_org_bool"].astype(int)

logit = smf.logit("any_claim_int ~ domain_healthcare + log_downloads + is_org_int", data=model_df).fit(disp=0)
print(logit.summary())

print("\nOdds ratios:")
print(np.exp(logit.params))
print("\n95% CI (odds ratio scale):")
print(np.exp(logit.conf_int()))

print("\n=== MULTIVARIATE OLS: gap_score ~ domain + log_downloads + is_organization ===")
ols = smf.ols("gap_score ~ domain_healthcare + log_downloads + is_org_int", data=model_df).fit()
print(ols.summary())

# Save regression output to text file
with open("regression_results.txt", "w") as f:
    f.write("=== Logistic regression: any_claim ~ domain + log_downloads + is_organization ===\n")
    f.write(str(logit.summary()))
    f.write("\n\nOdds ratios:\n")
    f.write(str(np.exp(logit.params)))
    f.write("\n\n95% CI (odds ratio scale):\n")
    f.write(str(np.exp(logit.conf_int())))
    f.write("\n\n=== OLS: gap_score ~ domain + log_downloads + is_organization ===\n")
    f.write(str(ols.summary()))
print("\nSaved full regression output to regression_results.txt")
