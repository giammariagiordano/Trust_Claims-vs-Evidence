import json
import glob
from collections import Counter
import pandas as pd
import numpy as np

BINARY_FIELDS = ["trust_claim", "bias_claim", "interp_claim", "clinical_use_disclaimer"]
EVIDENCE_FIELDS = ["trust_evidence", "bias_evidence", "interp_evidence"]


def load_run(globs):
    records = []
    for g in globs:
        for fpath in sorted(glob.glob(g)):
            with open(fpath) as f:
                records.extend(json.load(f))
    return {r["id"]: r for r in records}


def majority_binary(values):
    c = Counter(values)
    return c.most_common(1)[0][0]


def majority_evidence(values):
    c = Counter(values)
    top = c.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    return "partial"  # 3-way tie or no clear majority -> conservative fallback


def majority_method_named(values):
    non_null = [v for v in values if v]
    if len(non_null) >= 2:
        return non_null[0]
    return None


def fleiss_kappa(rows, categories):
    """rows: list of Counter-like dicts {category: count} summing to n raters each."""
    n_items = len(rows)
    n_raters = sum(rows[0].values())
    n_cat = len(categories)
    p_j = {c: 0 for c in categories}
    P_i = []
    for row in rows:
        for c in categories:
            p_j[c] += row.get(c, 0)
        s = sum(row.get(c, 0) ** 2 for c in categories)
        P_i.append((s - n_raters) / (n_raters * (n_raters - 1)))
    P_bar = sum(P_i) / n_items
    for c in categories:
        p_j[c] /= (n_items * n_raters)
    P_e = sum(v ** 2 for v in p_j.values())
    if P_e == 1:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def process_group(run_globs_list, group_name):
    runs = [load_run(g) for g in run_globs_list]
    ids = list(runs[0].keys())
    assert all(len(r) == len(ids) for r in runs), f"{group_name}: run size mismatch {[len(r) for r in runs]}"

    majority_records = []
    field_rows_for_kappa = {f: [] for f in BINARY_FIELDS + EVIDENCE_FIELDS}
    agreement_counts = {f: 0 for f in BINARY_FIELDS + EVIDENCE_FIELDS}

    for _id in ids:
        entries = [r[_id] for r in runs]
        out = {"id": _id}
        for f in BINARY_FIELDS:
            vals = [e[f] for e in entries]
            out[f] = majority_binary(vals)
            agreement_counts[f] += 1 if len(set(vals)) == 1 else 0
            field_rows_for_kappa[f].append(Counter(vals))
        for f in EVIDENCE_FIELDS:
            vals = [e[f] for e in entries]
            out[f] = majority_evidence(vals)
            agreement_counts[f] += 1 if len(set(vals)) == 1 else 0
            field_rows_for_kappa[f].append(Counter(vals))
        method_vals = [e.get("interpretability_method_named") for e in entries]
        out["interpretability_method_named"] = majority_method_named(method_vals)
        majority_records.append(out)

    n = len(ids)
    print(f"\n=== {group_name}: N={n}, 3 runs, majority vote ===")
    kappas = {}
    for f in BINARY_FIELDS:
        pct = agreement_counts[f] / n * 100
        k = fleiss_kappa(field_rows_for_kappa[f], ["yes", "no"])
        kappas[f] = k
        print(f"  {f}: unanimous={pct:.1f}%  Fleiss_kappa={k:.3f}")
    for f in EVIDENCE_FIELDS:
        pct = agreement_counts[f] / n * 100
        k = fleiss_kappa(field_rows_for_kappa[f], ["yes", "partial", "no"])
        kappas[f] = k
        print(f"  {f}: unanimous={pct:.1f}%  Fleiss_kappa={k:.3f}")

    return majority_records, kappas


# ---- Healthcare ----
health_majority, health_kappas = process_group(
    [["results/result_*.json"], ["results_run2/result_*.json"], ["results_run3/result_*.json"]],
    "Healthcare",
)

# ---- Control ----
control_majority, control_kappas = process_group(
    [
        ["control_results/cresult_*.json", "control_results_extra/ceresult_*.json"],
        ["control_results_run2/cresult_*.json", "control_results_extra_run2/ceresult_*.json"],
        ["control_results_run3/cresult_*.json", "control_results_extra_run3/ceresult_*.json"],
    ],
    "Control",
)

with open("results_majority.json", "w") as f:
    json.dump(health_majority, f, indent=2)
with open("control_results_majority.json", "w") as f:
    json.dump(control_majority, f, indent=2)

# ---- Overall reliability summary (pooled across both groups) ----
all_kappas = {}
for f in BINARY_FIELDS + EVIDENCE_FIELDS:
    # simple average weighted by N (500 each, so plain average is fine)
    all_kappas[f] = (health_kappas[f] + control_kappas[f]) / 2

print("\n=== Pooled average Fleiss' kappa per field (healthcare + control) ===")
for f, k in all_kappas.items():
    print(f"  {f}: {k:.3f}")

avg_kappa = sum(all_kappas.values()) / len(all_kappas)
print(f"\nOverall average Fleiss' kappa across all fields: {avg_kappa:.3f}")

with open("reliability_summary.json", "w") as f:
    json.dump({"healthcare": health_kappas, "control": control_kappas, "pooled": all_kappas, "overall_avg": avg_kappa}, f, indent=2)
print("\nSaved reliability_summary.json, results_majority.json, control_results_majority.json")
