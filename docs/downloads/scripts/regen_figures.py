import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("final_majority_results.csv")
dfh = df[df["domain"] == "healthcare"].copy()
dfh["created_at"] = pd.to_datetime(dfh["created_at"], errors="coerce", utc=True)
dfh_dated = dfh.dropna(subset=["created_at"]).copy()
dfh_dated["year_quarter"] = dfh_dated["created_at"].dt.tz_localize(None).dt.to_period("Q").astype(str)

trend = dfh_dated.groupby("year_quarter").agg(
    n=("id", "count"),
    any_claim_rate=("any_claim", "mean"),
    disclaimer_rate=("clinical_use_disclaimer", lambda x: (x == "yes").mean()),
).reset_index()
trend.to_csv("rq1_trend.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
trend_plot = trend[trend["n"] >= 3]
ax.plot(trend_plot["year_quarter"], trend_plot["any_claim_rate"] * 100, marker="o", label="Any trust/bias/interp claim (%)")
ax.plot(trend_plot["year_quarter"], trend_plot["disclaimer_rate"] * 100, marker="s", label="Clinical-use disclaimer (%)")
for milestone, label in [("2023Q4", "Political agreement"), ("2024Q3", "Entry into force")]:
    if milestone in trend_plot["year_quarter"].values:
        ax.axvline(x=milestone, color="gray", linestyle="--", alpha=0.6)
        ax.text(milestone, ax.get_ylim()[1]*0.95, label, rotation=90, fontsize=7, va="top")
ax.set_xlabel("Quarter (model creation date)")
ax.set_ylabel("% of models")
ax.set_title("RQ1: Documentation trends over time vs. EU AI Act milestones (majority-vote data)")
ax.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("rq1_trend.png", dpi=150)
print("saved rq1_trend.png")

dfh["log_downloads"] = np.log1p(dfh["downloads"])
fig, ax = plt.subplots(figsize=(7, 5))
np.random.seed(0)
ax.scatter(dfh["log_downloads"], dfh["gap_score"] + np.random.uniform(-0.08, 0.08, len(dfh)), alpha=0.4, s=15)
ax.set_xlabel("log(1 + downloads)")
ax.set_ylabel("gap_score (jittered)")
ax.set_title("RQ2: Popularity vs. claim-evidence gap score (majority-vote data)")
plt.tight_layout()
plt.savefig("rq2_popularity_gap.png", dpi=150)
print("saved rq2_popularity_gap.png")
