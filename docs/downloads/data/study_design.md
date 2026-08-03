# Study design (draft for validation)

## Title (working)
Trust Claims vs. Evidence: A Large-Scale Study of Documentation Practices in Healthcare AI Models on Hugging Face

## Positioning vs. related work
Di Penta et al. (ICPC 2024, distinguished paper) studied dataset/bias/license documentation presence across general-purpose Hugging Face models. Nature Machine Intelligence (32,111 model cards) found popularity correlates with better documentation, in general.
Gap we target: (1) healthcare/high-risk domain specifically, (2) claim-vs-evidence semantic gap rather than section presence/absence, (3) temporal correlation with EU AI Act milestones.

## Research Questions

**RQ1 (Regulatory trend):** Has the documentation of trustworthiness, bias, and interpretability in healthcare AI model cards improved over time, particularly around EU AI Act milestones (proposal 2021, political agreement 2023, entry into force Aug 2024)?

**RQ2 (Popularity vs. evidence gap):** In the healthcare domain, do more popular models (by downloads/likes) exhibit a smaller claim-vs-evidence gap than less popular ones, or does the general-purpose finding (popular = better documented) fail to hold when documentation is assessed for *substantiated* evidence rather than mere presence?

**RQ3 (Authorship):** Do models published by organizations (companies, hospitals, universities) show a smaller claim-vs-evidence gap than those published by individual accounts?

## Coding scheme (per model card, applied by LLM classifier)

For each of three dimensions — (a) trustworthiness/safety/clinical-readiness, (b) bias/fairness, (c) interpretability/explainability — classify:

- `claim`: does the card explicitly claim/assert something on this dimension? (yes / no)
- `evidence`: if `claim = yes`, is there concrete supporting evidence (named method, metric, test, dataset demographic breakdown, evaluation result)? (yes / partial / no)

Additional fields:
- `clinical_use_disclaimer`: does the card state limitations on clinical use / regulatory status (e.g., "not for clinical use", "research only", FDA/CE mention)? (yes/no)
- `interpretability_method_named`: specific method named (e.g., SHAP, LIME, attention visualization, saliency map) or none

Derived metric: **gap_score** = count of dimensions where `claim = yes` and `evidence = no` (range 0-3). Higher = larger say-do gap.

## Output per model
JSON record: model id, gap_score, per-dimension claim/evidence, clinical_use_disclaimer, interpretability_method_named, plus existing metadata (created_at, downloads, likes, is_organization).
