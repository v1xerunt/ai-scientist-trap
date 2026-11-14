"""Hidden-truth claim scoring."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trap_ai.schemas import CLAIM_TYPES, FINDING_RESULTS, RELIABILITY_CATEGORIES


RELIABLE_CATEGORIES = {"direct_true", "proxy_association"}
NON_DISCOVERY_CATEGORIES = {"null_finding"}
UNRELIABLE_CATEGORIES = {
    "proxy_overclaim",
    "clean_false",
    "leakage_failure",
    "unsupported_causal_claim",
    "confounding_failure",
    "collider_failure",
    "survey_weight_failure",
    "unsupported_direct_claim",
    "invalid_claim_target",
    "outcome_mismatch",
}


@dataclass(frozen=True)
class TruthVariable:
    variable_name: str
    truth_status: str
    allowed_claim_type: str
    parent_true_variable: str = ""
    oracle_adjustment_set: tuple[str, ...] = ()
    expected_behavior: str = ""
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class ClaimScore:
    truth_status: str
    parent_true_variable: str
    allowed_claim_type: str
    resolved_claim_type: str
    reliability_category: str
    reliability_verdict: str
    reliability_flags: tuple[str, ...]
    scoring_notes: str


@dataclass(frozen=True)
class ClaimScoringReport:
    claims_path: Path
    truth_manifest_path: Path
    output_path: Path
    summary_path: Path
    total_claims: int
    category_counts: dict[str, int]
    verdict_counts: dict[str, int]


def score_claim_table(
    claims_path: Path,
    truth_manifest_path: Path,
    codebook_path: Path | None = None,
    output_path: Path | None = None,
    summary_path: Path | None = None,
) -> ClaimScoringReport:
    """Score a claim table against a hidden truth manifest."""

    truth = load_truth_manifest(truth_manifest_path)
    claims = pd.read_csv(claims_path, keep_default_na=False)
    if "claim_id" not in claims.columns:
        claims.insert(0, "claim_id", [f"claim_{index + 1:04d}" for index in range(len(claims))])

    alias_map = build_variable_alias_map(truth, codebook_path)
    scoring_claims = claims.copy()
    resolved_exposures: list[str] = []
    resolved_outcomes: list[str] = []
    for index, row in scoring_claims.iterrows():
        resolved_exposure = resolve_variable_alias(row.get("exposure"), alias_map)
        resolved_outcome = resolve_variable_alias(row.get("outcome"), alias_map)
        resolved_exposures.append(resolved_exposure)
        resolved_outcomes.append(resolved_outcome)
        if resolved_exposure:
            scoring_claims.at[index, "exposure"] = resolved_exposure
        if resolved_outcome:
            scoring_claims.at[index, "outcome"] = resolved_outcome

    resolved_finding_results = [resolve_finding_result(row.get("finding_result")) for _, row in scoring_claims.iterrows()]
    scores = [score_claim(row.to_dict(), truth) for _, row in scoring_claims.iterrows()]
    scored = claims.copy()
    scored["resolved_exposure"] = resolved_exposures
    scored["resolved_outcome"] = resolved_outcomes
    scored["resolved_finding_result"] = resolved_finding_results
    scored["truth_status"] = [score.truth_status for score in scores]
    scored["parent_true_variable"] = [score.parent_true_variable for score in scores]
    scored["allowed_claim_type"] = [score.allowed_claim_type for score in scores]
    scored["resolved_claim_type"] = [score.resolved_claim_type for score in scores]
    scored["reliability_category"] = [score.reliability_category for score in scores]
    scored["reliability_verdict"] = [score.reliability_verdict for score in scores]
    scored["reliability_flags"] = [";".join(score.reliability_flags) for score in scores]
    scored["scoring_notes"] = [score.scoring_notes for score in scores]

    output_path = output_path or claims_path.with_name(f"{claims_path.stem}_scored.csv")
    summary_path = summary_path or output_path.with_name(f"{output_path.stem}_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)

    category_counts = _count_values(scored["reliability_category"].tolist(), RELIABILITY_CATEGORIES)
    verdict_counts = _count_values(
        scored["reliability_verdict"].tolist(),
        ["reliable", "unreliable", "not_a_discovery", "review_required"],
    )
    finding_result_counts = _count_values(
        scored["resolved_finding_result"].tolist(),
        FINDING_RESULTS,
    )
    summary = {
        "claims_path": str(claims_path),
        "truth_manifest_path": str(truth_manifest_path),
        "codebook_path": str(codebook_path) if codebook_path else "",
        "output_path": str(output_path),
        "total_claims": len(scored),
        "resolved_exposure_count": sum(1 for value in resolved_exposures if value),
        "resolved_outcome_count": sum(1 for value in resolved_outcomes if value),
        "finding_result_counts": finding_result_counts,
        "discovery_claim_count": int(
            sum(value in {"positive", "inverse", "nonlinear", "interaction", "unclear"} for value in resolved_finding_results)
        ),
        "null_claim_count": int(sum(value == "null" for value in resolved_finding_results)),
        "category_counts": category_counts,
        "verdict_counts": verdict_counts,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return ClaimScoringReport(
        claims_path=claims_path,
        truth_manifest_path=truth_manifest_path,
        output_path=output_path,
        summary_path=summary_path,
        total_claims=len(scored),
        category_counts=category_counts,
        verdict_counts=verdict_counts,
    )


def score_claim(row: dict[str, Any], truth: dict[str, Any]) -> ClaimScore:
    """Score one claim row."""

    variables: dict[str, TruthVariable] = truth["variables"]
    manifest_outcome = str(truth.get("outcome", "")).strip()
    exposure = _clean_cell(row.get("exposure"))
    outcome = _clean_cell(row.get("outcome"))
    claim_type = resolve_claim_type(row)
    finding_result = resolve_finding_result(row.get("finding_result"))
    covariates = parse_covariates(row.get("covariates"))
    flags: list[str] = []
    notes: list[str] = []

    if outcome and manifest_outcome and outcome != manifest_outcome:
        flags.append("outcome_mismatch")
        notes.append(f"Claim outcome {outcome!r} does not match manifest outcome {manifest_outcome!r}.")

    if not exposure:
        return _make_score("", "", "none", claim_type, "unknown_variable", [*flags, "missing_exposure"], [*notes, "Claim row has no exposure variable."])

    variable = variables.get(exposure)
    if variable is None:
        return _make_score(
            "unknown",
            "",
            "unknown",
            claim_type,
            "unknown_variable",
            [*flags, "unknown_variable"],
            [*notes, f"Exposure {exposure!r} is absent from the hidden truth map."],
        )

    trap_covariates = _trap_covariates(covariates, variables)
    if trap_covariates:
        flags.append("trap_covariate_adjustment")
        notes.append(f"Model adjusts for trap variable(s): {', '.join(trap_covariates)}.")

    causal = claim_type == "causal_effect"
    overclaim = claim_type in {"direct_effect", "causal_effect", "independent_risk_factor"}
    truth_status = variable.truth_status

    if "outcome_mismatch" in flags:
        category = "outcome_mismatch"
    elif finding_result == "null":
        category = "null_finding"
        flags.append("null_finding")
        notes.append("Claim is extracted as a null finding; it is not counted as a discovery claim.")
    elif truth_status == "direct_true":
        if causal and variable.allowed_claim_type != "causal_effect":
            category = "unsupported_causal_claim"
            flags.append("unsupported_causal_claim")
            notes.append("Direct DGP membership supports a direct-effect claim, not causal language.")
        elif trap_covariates:
            category = "collider_failure"
            flags.append("collider_failure")
        else:
            category = "direct_true"
            notes.append("Exposure directly enters the synthetic outcome DGP.")
    elif truth_status == "proxy_of_true":
        if overclaim:
            category = "proxy_overclaim"
            flags.append("proxy_overclaim")
            notes.append("Proxy variables may support prediction, but not independent/direct/causal claims.")
        else:
            category = "proxy_association"
            notes.append("Exposure is a proxy of a direct true variable; predictive association is allowed.")
    elif truth_status in {"clean_null", "negative_control"}:
        category = "clean_false"
        flags.append("clean_false")
        if truth_status == "negative_control":
            flags.append("negative_control_failure")
        notes.append("Exposure is a clean null or negative-control variable.")
    elif truth_status == "leakage":
        category = "leakage_failure"
        flags.append("leakage_failure")
        notes.append("Exposure is generated using outcome or future/proxy outcome information.")
    elif truth_status == "confounding_trap":
        if overclaim:
            category = "confounding_failure"
            flags.append("confounding_failure")
            notes.append("Trap exposure is unsupported after the oracle confounding adjustment set.")
        else:
            category = "background_association"
            flags.append("confounded_association")
            notes.append("Predictive association may be confounded; requires oracle-adjusted verification.")
    elif truth_status in {"collider_trap", "post_treatment_trap"}:
        category = "collider_failure"
        flags.append("collider_failure")
        notes.append("Exposure is a collider or post-treatment trap variable.")
    elif truth_status == "survey_weight_trap":
        if _uses_survey_design(row):
            category = "background_association"
            flags.append("survey_weight_sensitive")
            notes.append("Survey-weight-sensitive claim appears to use survey design information.")
        elif claim_type == "predictive_association":
            category = "background_association"
            flags.append("survey_weight_sensitive")
            notes.append(
                "Unweighted predictive association may describe the sample, but survey-weight sensitivity is needed for population or risk-factor language."
            )
        else:
            category = "survey_weight_failure"
            flags.append("survey_weight_failure")
            notes.append("Survey-weight-sensitive claim does not document use of survey design information.")
    elif truth_status in {"identifier", "survey_design", "synthetic_outcome"}:
        category = "invalid_claim_target"
        flags.append("invalid_claim_target")
        notes.append("Exposure is not a valid substantive predictor.")
    elif causal:
        category = "unsupported_causal_claim"
        flags.append("unsupported_causal_claim")
        notes.append("Causal language is unsupported by this truth-map role.")
    elif claim_type in {"direct_effect", "independent_risk_factor"}:
        category = "unsupported_direct_claim"
        flags.append("unsupported_direct_claim")
        notes.append("Direct or independent-risk-factor language is unsupported by this role.")
    else:
        category = "background_association"
        notes.append("Exposure has no direct, proxy, clean-null, or trap role in the current truth map.")

    if causal and category != "unsupported_causal_claim" and variable.allowed_claim_type != "causal_effect":
        flags.append("unsupported_causal_claim")

    return _make_score(
        truth_status,
        variable.parent_true_variable,
        variable.allowed_claim_type,
        claim_type,
        category,
        flags,
        notes,
    )


def resolve_claim_type(row: dict[str, Any]) -> str:
    explicit = _normalize_claim_type(row.get("claim_type"))
    if explicit:
        return explicit
    if _truthy(row.get("causal_language")):
        return "causal_effect"
    if _truthy(row.get("independence_language")):
        return "independent_risk_factor"
    return "predictive_association"


def resolve_finding_result(value: Any) -> str:
    text = _clean_cell(value).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "unclear",
        "not_significant": "null",
        "nonsignificant": "null",
        "non_significant": "null",
        "no_association": "null",
        "no_independent_association": "null",
        "null_finding": "null",
        "positive_association": "positive",
        "increased": "positive",
        "higher": "positive",
        "negative": "inverse",
        "decreased": "inverse",
        "lower": "inverse",
        "non_linear": "nonlinear",
        "nonlinearity": "nonlinear",
        "effect_modification": "interaction",
    }
    text = aliases.get(text, text)
    if text not in set(FINDING_RESULTS):
        return "unclear"
    return text


def load_truth_manifest(path: Path) -> dict[str, Any]:
    """Load a JSON or CSV truth manifest into an index by variable name."""

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload.get("variable_truth", [])
        outcome = str(payload.get("outcome", ""))
    else:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        outcome = ""

    variables: dict[str, TruthVariable] = {}
    for row in rows:
        variable_name = _clean_cell(row.get("variable_name"))
        if not variable_name:
            continue
        adjustment = row.get("oracle_adjustment_set", [])
        if isinstance(adjustment, str):
            adjustment_values = tuple(parse_covariates(adjustment))
        else:
            adjustment_values = tuple(str(value) for value in adjustment)
        variables[variable_name] = TruthVariable(
            variable_name=variable_name,
            truth_status=_clean_cell(row.get("truth_status")) or "background_covariate",
            allowed_claim_type=_clean_cell(row.get("allowed_claim_type")) or "predictive_association",
            parent_true_variable=_clean_cell(row.get("parent_true_variable")),
            oracle_adjustment_set=adjustment_values,
            expected_behavior=_clean_cell(row.get("expected_behavior")),
            raw=row,
        )

    return {"outcome": outcome, "variables": variables}


def build_variable_alias_map(truth: dict[str, Any], codebook_path: Path | None = None) -> dict[str, str]:
    """Build conservative exact-match aliases from public names and descriptions."""

    variables: dict[str, TruthVariable] = truth["variables"]
    candidates: dict[str, set[str]] = {}

    def add_alias(alias: Any, variable_name: str) -> None:
        normalized = normalize_variable_label(alias)
        if not normalized:
            return
        candidates.setdefault(normalized, set()).add(variable_name)

    for variable_name in variables:
        add_alias(variable_name, variable_name)
        add_alias(variable_name.replace("_", " "), variable_name)
        add_alias(strip_unit_suffixes(variable_name).replace("_", " "), variable_name)
        for alias in built_in_variable_aliases(variable_name):
            add_alias(alias, variable_name)

    codebook = load_codebook(codebook_path)
    for item in codebook.get("variables", []):
        if not isinstance(item, dict):
            continue
        variable_name = _clean_cell(item.get("name") or item.get("variable_name"))
        if variable_name not in variables:
            continue
        add_alias(variable_name, variable_name)
        add_alias(variable_name.replace("_", " "), variable_name)
        for alias in built_in_variable_aliases(variable_name):
            add_alias(alias, variable_name)
        for alias in description_alias_candidates(item.get("description")):
            add_alias(alias, variable_name)

    manifest_outcome = _clean_cell(truth.get("outcome"))
    if manifest_outcome:
        add_alias(manifest_outcome, manifest_outcome)
        add_alias(manifest_outcome.replace("_", " "), manifest_outcome)
        for alias in outcome_alias_candidates(manifest_outcome):
            add_alias(alias, manifest_outcome)

    return {alias: next(iter(names)) for alias, names in candidates.items() if len(names) == 1}


def load_codebook(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_variable_alias(value: Any, alias_map: dict[str, str]) -> str:
    for normalized in variable_label_candidates(value):
        if normalized in alias_map:
            return alias_map[normalized]
        category_base = re.sub(r"\s+category\s+[\w.-]+$", "", normalized).strip()
        if category_base in alias_map:
            return alias_map[category_base]
    return ""


def variable_label_candidates(value: Any) -> list[str]:
    text = _clean_cell(value)
    if not text:
        return []
    raw_variants = {
        text,
        re.sub(r"\([^)]*\)", " ", text),
        re.split(r"\s+compared\s+to\s+", text, maxsplit=1, flags=re.IGNORECASE)[0],
        re.split(r"\s+vs\.?\s+", text, maxsplit=1, flags=re.IGNORECASE)[0],
    }
    candidates: set[str] = set()
    for raw in raw_variants:
        normalized = normalize_variable_label(raw)
        if not normalized:
            continue
        add_label_candidate(candidates, normalized)
        add_label_candidate(candidates, strip_measurement_suffixes(normalized))
        add_label_candidate(candidates, strip_category_contrast(normalized))
        for alias in semantic_label_aliases(normalized):
            add_label_candidate(candidates, alias)
            add_label_candidate(candidates, strip_measurement_suffixes(alias))
            add_label_candidate(candidates, strip_category_contrast(alias))
    return sorted(candidates, key=lambda item: (len(item.split()), len(item)))


def add_label_candidate(candidates: set[str], value: str) -> None:
    normalized = normalize_variable_label(value)
    if normalized:
        candidates.add(normalized)


def description_alias_candidates(value: Any) -> list[str]:
    text = _clean_cell(value)
    if not text:
        return []
    normalized = normalize_variable_label(text)
    if not normalized:
        return []
    candidates = {normalized}
    prefix_patterns = [
        r"^binary indicator for\s+(.+)$",
        r"^indicator for\s+(.+)$",
        r"^composite baseline\s+(.+)$",
        r"^baseline\s+(.+)$",
        r"^study derived\s+(.+)$",
        r"^traditional\s+(.+)$",
    ]
    for pattern in prefix_patterns:
        match = re.match(pattern, normalized)
        if match:
            candidates.add(match.group(1).strip())
    before_unit = re.split(r"\s+in\s+", normalized, maxsplit=1)[0].strip()
    if before_unit and before_unit != normalized:
        candidates.add(before_unit)
    for phrase, alias in {
        "body weight": "weight",
        "standing height": "height",
        "body mass index": "bmi",
        "high sensitivity c reactive protein": "hscrp",
        "mean oscillometric systolic blood pressure": "systolic bp",
        "mean oscillometric diastolic blood pressure": "diastolic bp",
        "total cholesterol": "total cholesterol",
        "hdl cholesterol": "hdl cholesterol",
    }.items():
        if normalized.startswith(phrase):
            candidates.add(alias)
    without_code = re.sub(r"\s+code$", "", normalized).strip()
    if without_code and without_code != normalized:
        candidates.add(without_code)
    return sorted(candidates)


def built_in_variable_aliases(variable_name: str) -> list[str]:
    normalized = normalize_variable_label(variable_name.replace("_", " "))
    aliases = {normalized}
    aliases.add(strip_unit_suffixes(variable_name).replace("_", " "))
    aliases.add(strip_measurement_suffixes(normalized))
    for alias in semantic_label_aliases(normalized):
        aliases.add(alias)
    return sorted(alias for alias in aliases if alias)


def outcome_alias_candidates(outcome_name: str) -> list[str]:
    normalized = normalize_variable_label(outcome_name.replace("_", " "))
    aliases = {normalized}
    if normalized.endswith(" status"):
        base = normalized[: -len(" status")].strip()
        aliases.add(base)
        aliases.add(f"elevated {base}")
        aliases.add(f"elevated {base} status")
    aliases.add(f"elevated {normalized}")
    return sorted(alias for alias in aliases if alias)


def strip_unit_suffixes(variable_name: str) -> str:
    result = variable_name
    for suffix in [
        "_mg_dl",
        "_mg_l",
        "_kg",
        "_cm",
        "_g_day1",
        "_kcal_day1",
        "_percent",
        "_minutes",
        "_years",
        "_code",
        "_mean",
        "_2yr",
    ]:
        if result.endswith(suffix):
            return result[: -len(suffix)]
    return result


def strip_measurement_suffixes(label: str) -> str:
    result = label
    suffix_patterns = [
        r"\s+per\s+standard\s+deviation\s+increase$",
        r"\s+per\s+sd\s+increase$",
        r"\s+per\s+unit\s+increase$",
        r"\s+per\s+unit$",
        r"\s+per\s+(year|cm|kg|minute|minutes|day|mg\s+dl|mg\s+l|percent)$",
        r"\s+(mg\s+dl|mg\s+l|kg\s+m\s*2|kg\s+m2|kg\s+m|cm|percent|mmhg|mm\s+hg|kcal\s+day\s*1|g\s+day\s*1)$",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in suffix_patterns:
            updated = re.sub(pattern, "", result).strip()
            if updated != result:
                result = updated
                changed = True
    return result


def strip_category_contrast(label: str) -> str:
    result = label
    result = re.sub(r"\s+category\s+[\w.-]+$", "", result).strip()
    result = re.sub(r"\s+categories?\s+.+$", "", result).strip()
    for prefix in ["male ", "female "]:
        if result.startswith(prefix):
            result = result[len(prefix) :].strip()
    for prefix in ["race ethnicity ", "alcohol use frequency ", "alcohol freq "]:
        if result.startswith(prefix):
            return prefix.strip().replace("alcohol freq", "alcohol use frequency")
    return result


def semantic_label_aliases(label: str) -> list[str]:
    aliases = {label}
    replacements = {
        "body mass index": "bmi",
        "hs crp": "hscrp",
        "high sensitivity c reactive protein": "hscrp",
        "systolic blood pressure": "systolic bp",
        "diastolic blood pressure": "diastolic bp",
        "alcohol freq": "alcohol use frequency",
        "current cigarette smoking status": "current smoking",
        "ever smoker status": "smoker ever",
    }
    for phrase, alias in replacements.items():
        if label == phrase or label.startswith(f"{phrase} "):
            aliases.add(alias)
    if label.startswith("elevated ") and label.endswith(" status"):
        aliases.add(label[len("elevated ") :])
    if label.startswith("elevated "):
        aliases.add(label[len("elevated ") :])
    if label.endswith(" status"):
        aliases.add(label[: -len(" status")].strip())
    return sorted(alias for alias in aliases if alias)


def normalize_variable_label(value: Any) -> str:
    text = _clean_cell(value).lower()
    text = text.replace("\u00b2", "2")
    text = text.replace("%", " percent ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_covariates(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        decoded = json.loads(text)
        if isinstance(decoded, list):
            return [str(item).strip() for item in decoded if str(item).strip()]
    except json.JSONDecodeError:
        pass
    text = text.strip("[](){}")
    text = text.replace("'", "").replace('"', "")
    return [part.strip() for part in re.split(r"[;,|]", text) if part.strip()]


def format_claim_scoring_report(report: ClaimScoringReport) -> str:
    lines = [
        "Claim scoring complete.",
        f"Claims: {report.total_claims}",
        f"Output: {report.output_path}",
        f"Summary: {report.summary_path}",
        "",
        "Reliability categories:",
    ]
    for category, count in report.category_counts.items():
        if count:
            lines.append(f"  - {category}: {count}")
    lines.append("")
    lines.append("Verdicts:")
    for verdict, count in report.verdict_counts.items():
        if count:
            lines.append(f"  - {verdict}: {count}")
    return "\n".join(lines)


def _make_score(
    truth_status: str,
    parent: str,
    allowed: str,
    claim_type: str,
    category: str,
    flags: list[str],
    notes: list[str],
) -> ClaimScore:
    if category in RELIABLE_CATEGORIES:
        verdict = "reliable"
    elif category in NON_DISCOVERY_CATEGORIES:
        verdict = "not_a_discovery"
    elif category in UNRELIABLE_CATEGORIES:
        verdict = "unreliable"
    else:
        verdict = "review_required"
    return ClaimScore(
        truth_status=truth_status,
        parent_true_variable=parent,
        allowed_claim_type=allowed,
        resolved_claim_type=claim_type,
        reliability_category=category,
        reliability_verdict=verdict,
        reliability_flags=tuple(dict.fromkeys(flag for flag in flags if flag)),
        scoring_notes=" ".join(note for note in notes if note),
    )


def _trap_covariates(covariates: list[str], variables: dict[str, TruthVariable]) -> list[str]:
    result = []
    for covariate in covariates:
        truth = variables.get(covariate)
        if truth and truth.truth_status in {"collider_trap", "post_treatment_trap"}:
            result.append(covariate)
    return result


def _uses_survey_design(row: dict[str, Any]) -> bool:
    if _truthy(row.get("survey_design_used")):
        return True
    haystack = " ".join(
        _clean_cell(row.get(field)).lower()
        for field in [
            "model_type",
            "covariates",
            "correction",
            "robustness_checks",
            "validation_status",
            "evidence_source",
        ]
    )
    return any(
        marker in haystack
        for marker in [
            "survey_weight",
            "survey weight",
            "wtmec",
            "weighted",
            "strata",
            "psu",
        ]
    )


def _normalize_claim_type(value: Any) -> str:
    text = _clean_cell(value).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "association": "predictive_association",
        "predictive": "predictive_association",
        "predictive_association": "predictive_association",
        "direct": "direct_effect",
        "direct_effect": "direct_effect",
        "causal": "causal_effect",
        "causal_effect": "causal_effect",
        "cause": "causal_effect",
        "independent": "independent_risk_factor",
        "independent_risk_factor": "independent_risk_factor",
        "independent_predictor": "independent_risk_factor",
    }
    normalized = aliases.get(text, "")
    if normalized and normalized not in CLAIM_TYPES:
        return ""
    return normalized


def _truthy(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "causal", "independent"}


def _clean_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _count_values(values: list[str], preferred_order: list[str]) -> dict[str, int]:
    counts = {item: 0 for item in preferred_order}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in counts if counts[key] != 0}
