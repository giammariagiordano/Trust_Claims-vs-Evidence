# Methodology and Results

## Study title (working)
Trust Claims vs. Evidence: A Large-Scale Study of Documentation Practices in Healthcare AI Models on Hugging Face

## Related work / positioning
- Di Penta et al. (ICPC 2024, distinguished paper award), extended into a longitudinal study: analyzed dataset/bias/license documentation *presence* across general-purpose Hugging Face models. Cited here as motivation/related work only — **not used as a numerical baseline**, since their coding scheme (section presence/absence) and sampling differ from ours and are not directly comparable.
- Nature Machine Intelligence (32,111 model cards): found that model popularity correlates with better documentation, in general. Also cited as motivation only, not as a numerical baseline.
- Gap targeted by this study: (1) the healthcare/high-risk domain specifically, (2) a claim-vs-evidence semantic gap rather than section presence/absence, (3) correlation with EU AI Act regulatory milestones, (4) a **self-collected, methodologically identical control group** of general-purpose (non-healthcare) models, enabling a direct, internally-valid domain comparison instead of an indirect comparison against externally published numbers.

## Methodology

### Step 1 — Candidate collection
- Queried the Hugging Face Hub API (`huggingface_hub.HfApi.list_models`) with the `filter` parameter for six tags: `medical`, `clinical`, `healthcare`, `biomedical`, `radiology`, `pathology`.
- Deduplicated by model id across tags.
- Result: **8,209 unique candidate models**.

### Step 2 — Sampling and data fetching
- Random sample (seed=42) drawn from the 8,209 candidates, iterating until reaching the target size.
- For each sampled candidate: fetched the raw model card (`README.md` via the HF raw-content endpoint) and metadata (author/namespace, `downloads`, `likes`, `created_at`, `last_modified`, `pipeline_tag`, `tags`).
- Cards with empty or near-empty content (<50 characters) were discarded and replaced by the next candidate in the shuffled pool (not counted toward the target).
- Author namespace was checked against the HF Organizations API (`/api/organizations/{namespace}/overview`) to determine `is_organization` (True/False).
- Result: **500 valid models** collected out of 526 candidates tried.

### Step 3 — Coding scheme design
Three dimensions assessed per model card:
1. Trustworthiness / safety / clinical-readiness
2. Bias / fairness
3. Interpretability / explainability

For each dimension, two fields were coded:
- `claim` (yes/no): does the card explicitly **assert** something positive on this dimension (e.g., "safe for clinical use", "bias mitigated", "interpretable")? Neutral mentions of the topic (e.g., describing patient demographics without asserting fairness) do not count as a claim.
- `evidence` (yes/partial/no): if a claim is made, is there concrete supporting evidence (named method, metric, described test/evaluation, dataset demographic breakdown, benchmark result, citation to a validation study)?

Two additional fields:
- `clinical_use_disclaimer` (yes/no): does the card state limitations on clinical use (e.g., "not for clinical use", "research only", FDA/CE mention)?
- `interpretability_method_named`: specific method explicitly named (e.g., SHAP, LIME, Grad-CAM, attention visualization) or null.

Derived metric: **gap_score** (0–3) = number of dimensions where `claim = yes` AND `evidence = no`.

### Step 4 — Classification (3 runs + majority voting)
- The 500 model cards were split into 10 batches of 50.
- **Model**: Claude (Sonnet 5, Anthropic), accessed via Claude Code's agentic orchestration layer (parallel sub-agents, one per batch) — not a direct, parameterized API call. Temperature/top-p were therefore not set by us and are unknown (orchestration-layer defaults).
- **Prompting pattern**: zero-shot, rubric-based — no few-shot examples; each prompt embedded the full coding scheme, strictness criteria, edge-case rules (empty/placeholder cards → no claims), and a fixed JSON output schema.
- **Runs**: each batch was classified **independently 3 times** (no information about a prior run passed to the model). Binary fields (`claim`, `clinical_use_disclaimer`) use the majority label across the 3 runs; the 3-valued `evidence` field uses the majority label, falling back to "partial" on a 3-way tie; `interpretability_method_named` is treated as present if ≥2 of 3 runs named a method.
- **Reliability**: measured via raw % agreement and Fleiss' κ per field. Overall average κ = **0.678** (substantial agreement). Per-field κ ranges 0.525–0.920, except two rare-category control-group fields (`bias_claim` κ=0.218, `bias_evidence` κ=0.331) where κ is unstable due to near-zero prevalence despite ≥98.6% raw agreement (a known statistical artifact, not poor classification).
- Output: majority-voted structured record per model (`results_majority.json`), merged with the Step 2 metadata (`models_metadata.jsonl`) into `final_majority_results.csv`. Same model/prompting/3-run procedure applied identically to the control group (Step 6).
- Scripts: `aggregate_majority.py` (majority vote + Fleiss κ), `analyze_final.py` (final statistics).

**Note on why 3 runs matter:** an initial single-run analysis was also computed for comparison. Two conclusions changed materially once majority voting was applied: the RQ3 organizational-authorship effect (significant with 1 run, p=0.0058) disappeared (p=0.164 with 3-run majority vote), and the RQ4 domain-comparison effect flipped direction (healthcare non-significantly *higher* than control with 1 run → healthcare marginally *lower* than control, p=0.082, with 3-run majority vote). This is reported transparently in the paper as evidence that single-pass LLM classification is not sufficiently reliable for this kind of study.

### Step 5 — Analysis
- Descriptive statistics on claim/evidence/gap rates, disclaimer rate, interpretability-method-naming rate, organization-authorship rate.
- RQ1: model-creation-date binned pre/post EU AI Act entry into force (2024-08-01), compared via chi-square tests (quarterly trend line was computed but found too noisy for inference — see Results).
- RQ2: Spearman rank correlation between popularity (`downloads`, `likes`) and (a) `gap_score`, (b) presence of any claim.
- RQ3: comparison between organization-authored and individual-authored models via Mann-Whitney U (gap_score) and chi-square (any_claim, disclaimer).

Tools: Python, `huggingface_hub`, `pandas`, `scipy.stats`, `matplotlib`, `statsmodels`. Full pipeline scripts: `count_models.py`, `fetch_models.py`, `prepare_batches.py`, `aggregate_majority.py`, `analyze_final.py`.

### Step 6 — Control group (general-purpose AI models)
Motivation: comparing our results numerically against prior papers (Di Penta et al., Nature) would be invalid — different coding schemes, different years, different sampling. Instead, a **self-collected control group** was built using the exact same pipeline and coding scheme, enabling a direct, internally-valid comparison.

- Candidate pool: 20,000 most-recently-created models on the HF Hub (`sort="created_at"`, unfiltered by task/tag).
- Excluded any model whose tags or id matched the health-related keywords used in Step 1 (medical, clinical, healthcare, biomedical, radiology, pathology) → ~19,600–19,910 non-health candidates per pass.
- Random sample (seed=43) fetched with the identical procedure as Step 2 (README + metadata + organization check); extended with a second batch (seed=44, excluding already-collected ids) to match the healthcare sample size exactly.
- Result: **500 valid non-healthcare models** collected (300 + 200 in two passes, out of 547 + 379 tried).
- Classified with the identical coding scheme, model, prompting pattern, and **3-run majority-vote procedure** from Step 4 (the "clinical_use_disclaimer" field was generalized to "high-stakes-use disclaimer", e.g. "not for production use", "research only").

### Step 7 — Combined analysis
- Healthcare (N=500) and control (N=500) majority-voted datasets merged (N=1,000 total) with a `domain` indicator variable.
- Unadjusted comparison: chi-square test on `any_claim` by domain.
- **Multivariate logistic regression**: `any_claim ~ domain (healthcare vs. control) + log(1+downloads) + is_organization`, to test whether domain has an effect *independent of* popularity and authorship type.
- **Multivariate OLS regression**: `gap_score ~ domain + log(1+downloads) + is_organization`, same logic applied to the claim-without-evidence gap severity.

---

## Results (final, 3-run majority-vote)

### Descriptive overview (N=500, healthcare)
| Dimension | % making a claim | % of claims with full evidence |
|---|---|---|
| Trustworthiness/safety | 7.8% | 43.6% (n=39 claims) |
| Bias/fairness | 0.8% | 75.0% (n=4 claims) |
| Interpretability | 1.4% | 42.9% (n=7 claims) |

- Any claim on any dimension: **9.6%** → **90.4% of healthcare model cards make no trust/bias/interpretability claim at all.**
- Strict claim-without-evidence gap (any dimension): **1.6%** (8/500).
- Clinical-use disclaimer present: **18.0%**.
- Interpretability method explicitly named: **1.2%** (6/500 models).
- Organization-authored models: **29.8%** (149/500); individual-authored: 70.2% (351/500).

**Key descriptive finding:** the dominant pattern is *documentation silence*, not overstated/unsubstantiated claims — even more so than the single-run estimate suggested (90.4% vs. 87.2% silent).

### RQ1 — Regulatory trend
| Period | N | Any claim | Clinical-use disclaimer |
|---|---|---|---|
| Pre (before 2024-08) | 80 | 6.3% | 8.8% |
| Post (from 2024-08) | 420 | 10.2% | 19.8% |

- Any-claim increase: **not statistically significant** (χ²=0.815, p=0.367).
- Clinical-use disclaimer increase: **statistically significant** (χ²=4.800, p=0.0285).
- **This conclusion held up under the 3-run reliability check** (same qualitative pattern as the single-run analysis).

**Finding:** regulatory pressure nudges disclaimers, not evidence-backed documentation — a *compliance-by-disclaimer* pattern, robust to the reliability check.

### RQ2 — Popularity vs. documentation
| Correlation | Spearman ρ | p-value |
|---|---|---|
| downloads vs. gap_score | -0.014 | 0.751 |
| likes vs. gap_score | -0.042 | 0.353 |
| downloads vs. any_claim | -0.005 | 0.912 |
| likes vs. any_claim | 0.007 | 0.879 |

**Finding:** no significant correlation in any direction — **the single most stable finding across the reliability check** (near-zero correlations both before and after majority voting).

### RQ3 — Organizational vs. individual authorship
| Group | N | Any claim | Mean gap_score | Clinical-use disclaimer |
|---|---|---|---|---|
| Organization | 149 | 12.8% | 0.013 | 6.7% |
| Individual | 351 | 8.3% | 0.017 | 22.8% |

- Any-claim difference: **no longer statistically significant** (χ²=1.940, p=0.164) — **this reverses the single-run finding** (χ²=7.614, p=0.0058, significant).
- Gap-score difference: not significant (Mann-Whitney U=26053.5, p=0.766).
- Clinical-use disclaimer: individual authors disclaim more than 3× as often as organizations (22.8% vs. 6.7%) — this pattern held up.

**Finding:** the organizational-authorship effect on claim frequency was **partly a single-pass classification-noise artifact** — it vanishes under the 3-run majority vote. What survives is the disclaimer asymmetry.

### RQ4 — Healthcare vs. general-purpose AI models (controlled comparison)
| Domain | Any claim | Org-authored | Disclaimer | Median downloads |
|---|---|---|---|---|
| Healthcare | 9.6% | 29.8% | 18.0% | 9.0 |
| Control (general AI) | 13.0% | 20.4% | 7.4% | 9.0 |

Unadjusted chi-square (any_claim ~ domain): χ²=2.554, p=0.110 (not significant, but **healthcare is now descriptively lower**, reversing the single-run direction where healthcare was slightly higher, 12.8% vs 11.0%).

Multivariate logistic regression (`any_claim ~ domain_healthcare + log_downloads + is_organization`, N=1,000):
| Predictor | Odds Ratio | 95% CI | p-value |
|---|---|---|---|
| Healthcare domain | 0.70 | [0.47, 1.05] | **0.082** |
| log(1+downloads) | 0.99 | [0.90, 1.09] | 0.888 |
| Organization-authored | 1.26 | [0.81, 1.96] | 0.298 |

Multivariate OLS (`gap_score ~ ...`): no predictor significant (domain p=0.445; downloads p=0.818; organization p=0.712).

**Finding: the clean "no difference" conclusion from the single-run analysis does NOT survive the reliability check.** Under 3-run majority voting, healthcare shows **30% lower odds of making any documentation claim** than the control group, approaching but not crossing conventional significance (p=0.082). This is a direction reversal from the single-run result (which had shown healthcare marginally, non-significantly *higher*). We do not claim a confirmed domain effect (p>0.05), but we explicitly do NOT repeat the "structural, ecosystem-wide, no healthcare-specific pathology" framing from the single-run analysis — the more reliable data trends the opposite way.

---

## Overall narrative (final, post-reliability-check)
1. Healthcare AI models on Hugging Face are overwhelmingly silent on trustworthiness, bias, and interpretability (90.4% make no claim at all); explicit bias/fairness claims are nearly absent (0.8%) and named interpretability methods are almost nonexistent (1.2%).
2. Where claims exist, unsubstantiated ones are relatively rare (1.6% overall) — the accountability problem is one of absence, not overstatement.
3. The EU AI Act's entry into force correlates with more clinical-use disclaimers, but not with more evidence-backed trust/bias/interpretability documentation — a compliance-by-disclaimer pattern. **This finding is robust to the reliability check.**
4. Popularity is not a driver of documentation practice — **the most robust finding across the reliability check** (near-zero correlation, same sign, before and after majority voting).
5. **The organizational-authorship effect (orgs claiming ~2x more often) found in the single-run analysis disappeared under 3-run majority voting** — it was partly measurement noise. The disclaimer asymmetry (individuals disclaim more) survived.
6. **The domain comparison (RQ4) flipped direction under the reliability check**: single-run data suggested "no difference" (healthcare ≈ control); 3-run majority-vote data instead suggests healthcare trends *lower* than control (marginal, p=0.082, OR=0.70) — i.e., if anything, the domain regulation treats as high-risk may be somewhat *less* documented, not equally documented, though this does not reach conventional significance.

**Headline framing for the paper:** regulation aimed at high-risk domains is producing disclaimers, not evidence (point 3, robust finding) — and once classification reliability is taken seriously, there is no basis for the reassuring "healthcare is no different from the rest of the ecosystem" claim; the data trend, if anything, the other way. The paper's own methodological journey (two conclusions changing between single-run and 3-run analysis) is presented as a finding in itself: single-pass LLM annotation is not reliable enough for claims like RQ3/RQ4 without repeated runs and majority voting.

## Threats to validity (to disclose in the paper)
- No manual (human) validation / ground-truth check on the automatic LLM-based classification, for either dataset — the 3-run reliability check measures agreement between repeated LLM passes, not agreement with human ground truth.
- Two rare-category fields (bias_claim, bias_evidence) show unstable Fleiss' κ (0.22–0.33) in the control group despite ≥98.6% raw agreement — a near-zero-prevalence artifact, reported transparently rather than hidden.
- Healthcare sample of 500 out of 8,209 candidates (random, not stratified by sub-domain/modality); control sample of 500 (collected in two passes) out of ~19,600–19,910 non-health candidates per pass, each drawn from a recency-sorted pool of 20,000 (not a uniform random sample of the entire Hub).
- Tag-based retrieval may miss relevant models without those exact tags, or include borderline/irrelevant ones (quantization mirrors, placeholder cards) — applies to both groups.
- Card text truncated to 3,000 characters, potentially missing evidence located later in very long cards.
- The control group's "high-stakes-use disclaimer" field is a generalization of the healthcare group's "clinical-use disclaimer" field and is not perfectly semantically equivalent; comparisons involving this specific field should be read as approximate.
- RQ4's domain effect (p=0.082) is suggestive, not confirmatory; a larger sample could resolve whether it crosses conventional significance.
- Generation parameters (temperature, top-p) were not controlled by us (agentic orchestration layer, not direct parameterized API access) — unknown and undisclosed by the harness.
