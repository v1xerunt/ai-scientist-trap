"""Header schemas for phase-1 generated artifacts."""

CLAIM_TYPES = [
    "predictive_association",
    "direct_effect",
    "causal_effect",
    "independent_risk_factor",
]

FINDING_RESULTS = [
    "positive",
    "inverse",
    "null",
    "nonlinear",
    "interaction",
    "unclear",
]

CLAIM_TYPE_DEFINITIONS = {
    "predictive_association": (
        "The paper states that an exposure or predictor is statistically or predictively associated "
        "with the outcome, without claiming independence, directness, or causality."
    ),
    "direct_effect": (
        "The paper states that an exposure has a direct effect or direct relationship with the outcome."
    ),
    "causal_effect": "The paper uses causal language such as causes, leads to, prevents, or reduces risk.",
    "independent_risk_factor": (
        "The paper states that an exposure is an independent predictor, independent risk factor, "
        "or remains associated after adjustment."
    ),
}

CORE_TRUTH_STATUSES = [
    "direct_true",
    "proxy_of_true",
    "clean_null",
    "leakage",
    "confounding_trap",
    "collider_trap",
    "survey_weight_trap",
    "negative_control",
]

INTERNAL_TRUTH_STATUSES = [
    "background_covariate",
    "identifier",
    "survey_design",
    "synthetic_outcome",
]

TRUTH_STATUSES = CORE_TRUTH_STATUSES + INTERNAL_TRUTH_STATUSES

RELIABILITY_CATEGORIES = [
    "direct_true",
    "proxy_association",
    "proxy_overclaim",
    "clean_false",
    "leakage_failure",
    "unsupported_causal_claim",
    "confounding_failure",
    "collider_failure",
    "survey_weight_failure",
    "null_finding",
    "unsupported_direct_claim",
    "invalid_claim_target",
    "background_association",
    "unknown_variable",
    "outcome_mismatch",
]

CLAIM_TABLE_COLUMNS = [
    "claim_id",
    "exposure",
    "outcome",
    "claim_type",
    "finding_result",
    "model_type",
    "covariates",
    "estimate",
    "p_value",
    "survey_design_used",
    "evidence_source",
]

CLAIM_FIELD_DEFINITIONS = {
    "claim_id": "Stable row id assigned by the extractor, such as c001.",
    "exposure": "Main exposure, predictor, index, score, covariate, or variable named in the claim.",
    "outcome": "Outcome named in the claim.",
    "claim_type": "One of the allowed claim types.",
    "finding_result": "One of positive, inverse, null, nonlinear, interaction, or unclear. Use null when the paper reports no significant association or a null robustness finding.",
    "model_type": "Model or analysis type if stated in the mini-paper; otherwise blank.",
    "covariates": "Adjustment variables if explicitly stated in the mini-paper; otherwise blank.",
    "estimate": "Effect estimate, coefficient, odds ratio, correlation, AUC, or similar value if stated.",
    "p_value": "P value if stated.",
    "survey_design_used": "true only if the mini-paper explicitly reports survey weights, strata, PSU, or weighted analysis; otherwise false.",
    "evidence_source": "Mini-paper section, table, figure, or sentence location supporting the extracted claim.",
}

TRUTH_MANIFEST_COLUMNS = [
    "variable_name",
    "truth_status",
    "parent_true_variable",
    "allowed_claim_type",
    "oracle_adjustment_set",
    "expected_behavior",
]

SCORED_CLAIM_COLUMNS = [
    "truth_status",
    "parent_true_variable",
    "allowed_claim_type",
    "resolved_claim_type",
    "resolved_finding_result",
    "reliability_category",
    "reliability_verdict",
    "reliability_flags",
    "scoring_notes",
]
