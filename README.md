# Trust Claims vs. Evidence

**An Empirical Study of Documentation Practices in Healthcare AI Models on Hugging Face**

Giammaria Giordano¹, Valeria Pontillo², Fabio Palomba³
¹ Pegaso University, Naples, Italy · ² Gran Sasso Science Institute (GSSI), L'Aquila, Italy · ³ University of Salerno, Salerno, Italy

📄 Paper: [`docs/downloads/paper.pdf`](docs/downloads/paper.pdf)
🌐 Interactive replication site: **https://giammariagiordano.github.io/Trust_Claims-vs-Evidence/**

## Abstract

Healthcare AI systems are increasingly shared as open models on hubs such as Hugging Face, yet the EU AI Act classifies many of them as *high-risk*, demanding transparency about trustworthiness, bias, and interpretability. We study 500 healthcare-related model cards on Hugging Face, checking whether claims about trustworthiness, bias, or interpretability are *substantiated* with concrete evidence. Each card was classified three times by an LLM and aggregated via majority voting (Fleiss' κ = 0.678). We complement this with a control sample of 500 general-purpose models. We find that 90.4% of healthcare model cards make no such claim. The EU AI Act's entry into force correlates with more clinical-use disclaimers but not more evidence-backed documentation, and an organizational-authorship effect visible in raw counts is not statistically significant once classification noise is accounted for. A multivariate comparison against our control sample shows healthcare models with about 30% lower odds of making any documentation claim than general-purpose models (p = 0.082).

## Research questions

- **RQ1 — Regulatory trend.** Has documentation improved over time, particularly around EU AI Act milestones?
- **RQ2 — Authorship.** Do organization-published models show a smaller claim-evidence gap than individually-published ones?
- **RQ3 — Domain comparison.** Is healthcare documented differently from the rest of the ecosystem, controlling for popularity and authorship?

## Repository structure

```
.
├── docs/                          interactive replication-package site (GitHub Pages, served from /docs)
│   ├── downloads/                 paper.pdf, replication_package.zip, copies of scripts/data
│   └── assets/                    static chart images
├── Trust_Claims_vs_Evidence.zip   full LaTeX paper source
│
├── count_models.py                Step 1 — candidate counting via HF Hub API
├── fetch_models.py                Step 2 — README + metadata fetch (healthcare sample)
├── prepare_batches.py             split healthcare sample into 10×50 batches
├── fetch_control.py               control-group README + metadata fetch
├── prepare_control_batches.py     split control sample into batches
├── aggregate_majority.py          majority vote across 3 classification runs + Fleiss' κ
├── analyze_final.py               descriptive stats, RQ1/RQ2 tests
├── analyze_combined.py            RQ3 multivariate logistic regression + OLS
├── regen_figures.py               regenerate charts from final data
│
├── batches/ · control_batches*/           LLM classification input batches
├── results/ · control_results*/           raw per-batch classification output (3 runs)
├── final_majority_results.csv             final merged dataset (N=1,000, majority-voted)
├── results_majority.json                  majority-voted structured records
├── reliability_summary.json               Fleiss' κ per field
├── rq1_trend.csv                          quarterly trend data (RQ1)
├── regression_results.txt                 single-run regression output (superseded by paper's 3-run results)
├── study_design.md                        coding scheme design notes
└── methodology_and_results.md             full methodology + results log
```

## Reproducing the study

```bash
python -m venv venv
source venv/bin/activate
pip install huggingface_hub pandas scipy matplotlib statsmodels
```

Run order:

1. `count_models.py` — count candidate models per health tag
2. `fetch_models.py` / `fetch_control.py` — collect README + metadata
3. `prepare_batches.py` / `prepare_control_batches.py` — split into classification batches
4. Classify each batch 3× independently with an LLM using the coding scheme in `study_design.md`
5. `aggregate_majority.py` — majority vote + Fleiss' κ
6. `analyze_final.py` → `analyze_combined.py` — final statistics and regressions

See `methodology_and_results.md` for the full step-by-step methodology and results, and the paper for the complete write-up.

## Citation

```bibtex
@inproceedings{giordano2026trustclaims,
  title     = {Trust Claims vs. Evidence --- An Empirical Study of
               Documentation Practices in Healthcare AI Models
               on Hugging Face},
  author    = {Giordano, Giammaria and Pontillo, Valeria and Palomba, Fabio},
  year      = {2026}
}
```
