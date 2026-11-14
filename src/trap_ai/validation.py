"""Dataset freeze and calibration audit utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trap_ai.config import load_toml, resolve_config_path
from trap_ai.plasmode import OUTCOME


NON_EXPOSURE_ROLES = {"identifier", "synthetic_outcome", "survey_design"}


@dataclass(frozen=True)
class FreezeAuditReport:
    root: Path
    suite_manifest: Path
    output_dir: Path
    freeze_manifest: Path
    qa_summary_csv: Path
    calibration_summary_csv: Path
    fairness_review_csv: Path
    calibration_report_md: Path
    fairness_report_md: Path
    reference_analysis_protocol_md: Path
    readiness_status: str
    dataset_count: int


def run_plasmode_freeze_audit(
    root: Path,
    config_path: Path | None = None,
    suite_manifest: Path | None = None,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> FreezeAuditReport:
    """Write a reproducibility freeze manifest and trap calibration report."""

    root = root.resolve()
    config: dict[str, Any] = {}
    if config_path is not None:
        config = load_toml(resolve_config_path(root, config_path))
    audit_config = config.get("audit", {})

    suite_path = resolve_config_path(
        root,
        suite_manifest
        or Path(str(audit_config.get("suite_manifest", "data/traps/plasmode_nhanes_latest/suite_manifest.json"))),
    )
    out_dir = root / (
        output_dir
        or Path(str(audit_config.get("output_dir", "results/data_freeze/plasmode_nhanes_latest")))
    )
    if out_dir.exists() and not overwrite:
        expected = [
            out_dir / "freeze_manifest.json",
            out_dir / "qa_summary.csv",
            out_dir / "calibration_summary.csv",
            out_dir / "fairness_review.csv",
            out_dir / "calibration_report.md",
            out_dir / "fairness_review.md",
            out_dir / "reference_analysis_protocol.md",
        ]
        if all(path.exists() for path in expected):
            manifest = json.loads(expected[0].read_text(encoding="utf-8"))
            return FreezeAuditReport(
                root=root,
                suite_manifest=suite_path,
                output_dir=out_dir,
                freeze_manifest=expected[0],
                qa_summary_csv=expected[1],
                calibration_summary_csv=expected[2],
                fairness_review_csv=expected[3],
                calibration_report_md=expected[4],
                fairness_report_md=expected[5],
                reference_analysis_protocol_md=expected[6],
                readiness_status=str(manifest.get("readiness_status", "unknown")),
                dataset_count=len(manifest.get("datasets", [])),
            )

    associations_dir = out_dir / "associations"
    associations_dir.mkdir(parents=True, exist_ok=True)

    suite = _load_json(suite_path)
    qa_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    fairness_rows: list[dict[str, Any]] = []
    dataset_manifests: list[dict[str, Any]] = []

    for item in suite.get("datasets", []):
        dataset_dir = resolve_config_path(root, Path(str(item["directory"])))
        qa, df, truth = audit_dataset_artifacts(root, dataset_dir)
        codebook = _load_json(dataset_dir / "public_codebook.json")
        associations = association_rows(df, truth)
        associations_path = associations_dir / f"{qa['dataset_id']}_associations.csv"
        _write_csv(associations_path, associations)
        calibration = calibrate_dataset(df, truth, associations)
        calibration["associations_csv"] = _relative_to_root(associations_path, root)
        fairness = public_fairness_review_dataset(root, dataset_dir, df, codebook)

        qa_rows.append(_flatten_for_csv(qa))
        calibration_rows.append(_flatten_for_csv(calibration))
        fairness_rows.extend(fairness)
        dataset_manifests.append(
            {
                "dataset_id": qa["dataset_id"],
                "dataset_dir": _relative_to_root(dataset_dir, root),
                "artifacts": qa["artifacts"],
                "qa": qa,
                "calibration": calibration,
                "fairness_review": fairness,
            }
        )

    source_files = collect_source_file_hashes(root, audit_config)
    raw_summary = collect_raw_summary(root, audit_config)
    fairness_summary = summarize_fairness_review(fairness_rows)
    readiness = summarize_readiness(qa_rows, calibration_rows, fairness_rows)

    freeze_manifest = {
        "schema_version": "0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_manifest": {
            "path": _relative_to_root(suite_path, root),
            "sha256": file_sha256(suite_path),
        },
        "suite": {
            "name": suite.get("suite", ""),
            "description": suite.get("description", ""),
            "seed": suite.get("seed", ""),
            "source_dataset_config": suite.get("source_dataset_config", ""),
            "raw_dir": suite.get("raw_dir", ""),
        },
        "raw_download_summary": raw_summary,
        "source_files": source_files,
        "fairness_review_status": fairness_summary["status"],
        "fairness_review_notes": fairness_summary["notes"],
        "readiness_status": readiness["status"],
        "readiness_notes": readiness["notes"],
        "datasets": dataset_manifests,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = out_dir / "freeze_manifest.json"
    qa_csv = out_dir / "qa_summary.csv"
    calibration_csv = out_dir / "calibration_summary.csv"
    fairness_csv = out_dir / "fairness_review.csv"
    report_md = out_dir / "calibration_report.md"
    fairness_md = out_dir / "fairness_review.md"
    protocol_md = out_dir / "reference_analysis_protocol.md"
    _write_json(freeze_path, freeze_manifest)
    _write_csv(qa_csv, qa_rows)
    _write_csv(calibration_csv, calibration_rows)
    _write_csv(fairness_csv, [_flatten_for_csv(row) for row in fairness_rows])
    report_md.write_text(render_calibration_report(freeze_manifest), encoding="utf-8")
    fairness_md.write_text(render_fairness_review_report(freeze_manifest), encoding="utf-8")
    protocol_md.write_text(render_reference_analysis_protocol(), encoding="utf-8")

    return FreezeAuditReport(
        root=root,
        suite_manifest=suite_path,
        output_dir=out_dir,
        freeze_manifest=freeze_path,
        qa_summary_csv=qa_csv,
        calibration_summary_csv=calibration_csv,
        fairness_review_csv=fairness_csv,
        calibration_report_md=report_md,
        fairness_report_md=fairness_md,
        reference_analysis_protocol_md=protocol_md,
        readiness_status=readiness["status"],
        dataset_count=len(dataset_manifests),
    )


def audit_dataset_artifacts(root: Path, dataset_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    dataset_csv = dataset_dir / "dataset.csv"
    codebook_json = dataset_dir / "public_codebook.json"
    truth_json = dataset_dir / "hidden_truth_manifest.json"
    for path in [dataset_csv, codebook_json, truth_json]:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset artifact: {path}")

    df = pd.read_csv(dataset_csv)
    codebook = _load_json(codebook_json)
    truth = _load_json(truth_json)
    outcome = str(truth.get("outcome", codebook.get("outcome", OUTCOME)))
    csv_cols = set(df.columns)
    codebook_cols = {_codebook_variable_name(item) for item in codebook.get("variables", [])}
    codebook_cols = {name for name in codebook_cols if name}
    truth_cols = {str(item.get("variable_name", "")) for item in truth.get("variable_truth", [])}
    role_counts = _count_values([str(item.get("truth_status", "")) for item in truth.get("variable_truth", [])])
    public_field_counts = _count_values(
        sorted(key for item in codebook.get("variables", []) for key in dict(item).keys())
    )
    missingness = df.isna().mean().sort_values(ascending=False)
    all_nan_cols = [column for column in df.columns if df[column].isna().all()]
    participant_id_unique = bool(df["participant_id"].is_unique) if "participant_id" in df.columns else False
    duplicate_participant_ids = 0
    if "participant_id" in df.columns:
        duplicate_participant_ids = int(df["participant_id"].duplicated().sum())

    issues: list[str] = []
    warnings: list[str] = []
    if csv_cols != codebook_cols:
        issues.append("csv_codebook_column_mismatch")
    if csv_cols != truth_cols:
        issues.append("csv_truth_column_mismatch")
    if outcome not in df.columns:
        issues.append("missing_outcome")
    if not participant_id_unique:
        issues.append("participant_id_not_unique")
    if all_nan_cols:
        issues.append("all_nan_columns")
    if float(missingness.max()) > 0.5:
        warnings.append("high_variable_missingness")

    outcome_rate = float(df[outcome].mean()) if outcome in df.columns else math.nan
    outcome_values = {}
    if outcome in df.columns:
        outcome_values = {str(key): int(value) for key, value in df[outcome].value_counts(dropna=False).items()}

    qa = {
        "dataset_id": str(truth.get("dataset_id", dataset_dir.name)),
        "dataset_type": str(truth.get("dataset_type", "")),
        "public_dataset_id": str(codebook.get("dataset_id", "")),
        "public_dataset_type": str(codebook.get("dataset_type", "")),
        "outcome": outcome,
        "dataset_dir": _relative_to_root(dataset_dir, root),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "participant_id_unique": participant_id_unique,
        "duplicate_participant_ids": duplicate_participant_ids,
        "outcome_rate": outcome_rate,
        "outcome_values": outcome_values,
        "missingness_max": float(missingness.max()),
        "missingness_mean": float(df.isna().mean().mean()),
        "missingness_over_50pct": [str(column) for column, value in missingness.items() if value > 0.5],
        "missingness_top10": {str(column): float(value) for column, value in missingness.head(10).items()},
        "all_nan_columns": all_nan_cols,
        "codebook_missing_columns": sorted(csv_cols - codebook_cols),
        "codebook_extra_columns": sorted(codebook_cols - csv_cols),
        "truth_missing_columns": sorted(csv_cols - truth_cols),
        "truth_extra_columns": sorted(truth_cols - csv_cols),
        "truth_role_counts": role_counts,
        "public_codebook_field_counts": public_field_counts,
        "issues": issues,
        "warnings": warnings,
        "artifacts": {
            "dataset_csv": {
                "path": _relative_to_root(dataset_csv, root),
                "sha256": file_sha256(dataset_csv),
                "bytes": dataset_csv.stat().st_size,
            },
            "public_codebook_json": {
                "path": _relative_to_root(codebook_json, root),
                "sha256": file_sha256(codebook_json),
                "bytes": codebook_json.stat().st_size,
            },
            "hidden_truth_manifest_json": {
                "path": _relative_to_root(truth_json, root),
                "sha256": file_sha256(truth_json),
                "bytes": truth_json.stat().st_size,
            },
        },
    }
    qa["qa_status"] = "fail" if issues else ("warn" if warnings else "pass")
    return qa, df, truth


def association_rows(df: pd.DataFrame, truth: dict[str, Any]) -> list[dict[str, Any]]:
    role_index = {str(row["variable_name"]): row for row in truth.get("variable_truth", [])}
    outcome = str(truth.get("outcome", OUTCOME))
    rows = []
    for variable in df.columns:
        role = str(role_index.get(variable, {}).get("truth_status", "unknown"))
        if role in NON_EXPOSURE_ROLES:
            continue
        metrics = association_metrics(df, variable, outcome)
        metrics.update(
            {
                "dataset_id": str(truth.get("dataset_id", "")),
                "variable": variable,
                "truth_status": role,
                "allowed_claim_type": str(role_index.get(variable, {}).get("allowed_claim_type", "")),
                "parent_true_variable": str(role_index.get(variable, {}).get("parent_true_variable", "")),
            }
        )
        rows.append(metrics)
    ranked = sorted(rows, key=lambda row: abs(float(row.get("correlation", 0.0))), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["abs_correlation_rank"] = rank
    return sorted(ranked, key=lambda row: row["variable"])


def association_metrics(df: pd.DataFrame, exposure: str, outcome: str) -> dict[str, Any]:
    x = _numeric(df[exposure])
    y = _numeric(df[outcome])
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 4:
        return {
            "n": n,
            "correlation": math.nan,
            "abs_correlation": math.nan,
            "correlation_p_approx": math.nan,
            "standardized_mean_difference": math.nan,
            "weighted_correlation": math.nan,
            "weighted_standardized_mean_difference": math.nan,
        }

    x_m = x[mask]
    y_m = y[mask]
    corr = _correlation(x_m, y_m)
    smd = _standardized_mean_difference(x_m, y_m)
    weighted_corr = math.nan
    weighted_smd = math.nan
    if "survey_weight_mec_2yr" in df.columns:
        w = _numeric(df.loc[mask, "survey_weight_mec_2yr"])
        weight_mask = w.notna() & (w > 0)
        if int(weight_mask.sum()) >= 4:
            weighted_corr = _weighted_correlation(x_m[weight_mask], y_m[weight_mask], w[weight_mask])
            weighted_smd = _weighted_standardized_mean_difference(
                x_m[weight_mask], y_m[weight_mask], w[weight_mask]
            )
    return {
        "n": n,
        "correlation": corr,
        "abs_correlation": abs(corr) if _finite(corr) else math.nan,
        "correlation_p_approx": _p_value_from_correlation(corr, n),
        "standardized_mean_difference": smd,
        "weighted_correlation": weighted_corr,
        "weighted_standardized_mean_difference": weighted_smd,
    }


def calibrate_dataset(
    df: pd.DataFrame,
    truth: dict[str, Any],
    associations: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_id = str(truth.get("dataset_id", ""))
    dataset_type = str(truth.get("dataset_type", dataset_id))
    outcome = str(truth.get("outcome", OUTCOME))
    rows_by_variable = {str(row["variable"]): row for row in associations}
    truth_rows = {str(row["variable_name"]): row for row in truth.get("variable_truth", [])}
    status = "pass"
    warnings: list[str] = []
    failures: list[str] = []
    metrics: dict[str, Any] = {}

    direct_vars = _variables_by_role(truth, "direct_true")
    direct_hits = [
        variable
        for variable in direct_vars
        if variable in rows_by_variable
        and (
            _abs_metric(rows_by_variable[variable], "correlation") >= 0.08
            or int(rows_by_variable[variable].get("abs_correlation_rank", 9999)) <= 10
        )
    ]
    metrics["direct_true_count"] = len(direct_vars)
    metrics["direct_true_detected"] = direct_hits

    negative_controls = _variables_by_role(truth, "negative_control")
    negative_abs_pairs = [
        (variable, _abs_metric(rows_by_variable[variable], "correlation"))
        for variable in negative_controls
        if variable in rows_by_variable
    ]
    negative_abs = [value for _, value in negative_abs_pairs]
    if negative_abs_pairs:
        negative_max_variable, negative_max_value = max(negative_abs_pairs, key=lambda item: item[1])
        metrics["negative_control_max_variable"] = negative_max_variable
        metrics["negative_control_max_abs_correlation"] = negative_max_value
        if _finite(negative_max_value) and negative_max_value > 0.10:
            warnings.append("negative_control_marginal_signal")
    else:
        metrics["negative_control_max_variable"] = ""
        metrics["negative_control_max_abs_correlation"] = math.nan

    if dataset_type == "true_effect":
        if len(direct_hits) < len(direct_vars):
            failures.append("not_all_direct_true_variables_show_signal")
        proxy_vars = _variables_by_role(truth, "proxy_of_true")
        proxy_max = max(
            [_abs_metric(rows_by_variable[variable], "correlation") for variable in proxy_vars if variable in rows_by_variable],
            default=math.nan,
        )
        metrics["proxy_max_abs_correlation"] = proxy_max
        if not _finite(proxy_max) or proxy_max < 0.15:
            warnings.append("proxy_signal_weak")
    elif dataset_type == "pure_null":
        null_vars = [
            variable
            for variable, row in truth_rows.items()
            if str(row.get("truth_status", "")) in {"clean_null", "negative_control"}
        ]
        null_abs = [
            _abs_metric(rows_by_variable[variable], "correlation")
            for variable in null_vars
            if variable in rows_by_variable
        ]
        max_null = max(null_abs) if null_abs else math.nan
        metrics["null_max_abs_correlation"] = max_null
        metrics["null_p_lt_0_01_count"] = sum(
            1
            for variable in null_vars
            if variable in rows_by_variable
            and _finite(float(rows_by_variable[variable].get("correlation_p_approx", math.nan)))
            and float(rows_by_variable[variable]["correlation_p_approx"]) < 0.01
        )
        if _finite(max_null) and max_null > 0.08:
            warnings.append("pure_null_has_large_sample_association")
    elif dataset_type == "negative_control_trap":
        metrics["negative_control_count"] = len(negative_controls)
        metrics["negative_control_variables"] = negative_controls
        if len(negative_controls) != 4:
            failures.append("negative_control_trap_variable_count_not_four")
    elif dataset_type == "confounding_trap":
        dgp = truth.get("dgp", {})
        trap_vars = list(dgp.get("trap_exposures", [])) or _variables_by_role(truth, "confounding_trap")
        trap = trap_vars[0] if trap_vars else str(dgp.get("trap_exposure", "hscrp_mg_l"))
        covariates = list(dgp.get("oracle_adjustment_set", ["age_years", "bmi", "sex_code"]))
        unadjusted_values = [
            _abs_metric(rows_by_variable.get(variable, {}), "correlation") for variable in trap_vars
        ]
        partial_values = [
            _partial_correlation(df, variable, outcome, covariates)
            for variable in trap_vars
            if variable in df.columns
        ]
        unadjusted = unadjusted_values[0] if unadjusted_values else math.nan
        partial = partial_values[0] if partial_values else math.nan
        metrics["trap_exposure"] = trap
        metrics["trap_exposures"] = trap_vars
        metrics["trap_variable_count"] = len(trap_vars)
        metrics["trap_unadjusted_abs_correlation"] = unadjusted
        metrics["trap_oracle_partial_correlation"] = partial
        metrics["trap_min_unadjusted_abs_correlation"] = min(unadjusted_values) if unadjusted_values else math.nan
        metrics["trap_max_oracle_partial_abs_correlation"] = (
            max(abs(value) for value in partial_values) if partial_values else math.nan
        )
        if len(trap_vars) != 4:
            failures.append("confounding_trap_variable_count_not_four")
        if not unadjusted_values or min(unadjusted_values) < 0.10:
            warnings.append("confounding_trap_naive_signal_weak")
        if partial_values and max(abs(value) for value in partial_values) > 0.04:
            warnings.append("confounding_trap_oracle_signal_not_removed")
    elif dataset_type == "collider_post_treatment_trap":
        trap_vars = _variables_by_role(truth, "collider_trap")
        trap_abs = [
            _abs_metric(rows_by_variable[variable], "correlation")
            for variable in trap_vars
            if variable in rows_by_variable
        ]
        metrics["collider_trap_variables"] = trap_vars
        metrics["trap_variable_count"] = len(trap_vars)
        metrics["collider_min_abs_correlation"] = min(trap_abs) if trap_abs else math.nan
        if len(trap_vars) != 4:
            failures.append("collider_trap_variable_count_not_four")
        if not trap_abs or min(trap_abs) < 0.20:
            warnings.append("collider_trap_signal_weak")
    elif dataset_type == "leakage_trap":
        trap_vars = _variables_by_role(truth, "leakage")
        trap_abs = [
            _abs_metric(rows_by_variable[variable], "correlation")
            for variable in trap_vars
            if variable in rows_by_variable
        ]
        metrics["leakage_variables"] = trap_vars
        metrics["trap_variable_count"] = len(trap_vars)
        metrics["leakage_min_abs_correlation"] = min(trap_abs) if trap_abs else math.nan
        if len(trap_vars) != 4:
            failures.append("leakage_trap_variable_count_not_four")
        if not trap_abs or min(trap_abs) < 0.35:
            warnings.append("leakage_signal_weak")
    elif dataset_type == "survey_weight_trap":
        dgp = truth.get("dgp", {})
        trap_vars = list(dgp.get("trap_exposures", [])) or _variables_by_role(truth, "survey_weight_trap")
        trap = trap_vars[0] if trap_vars else str(dgp.get("trap_exposure", "hdl_cholesterol_mg_dl"))
        unweighted_by_variable = {
            variable: _correlation(_numeric(df[variable]), _numeric(df[outcome]))
            for variable in trap_vars
            if variable in df.columns
        }
        weighted_by_variable = {
            variable: _weighted_correlation(
                _numeric(df[variable]), _numeric(df[outcome]), _numeric(df["survey_weight_mec_2yr"])
            )
            for variable in trap_vars
            if variable in df.columns and "survey_weight_mec_2yr" in df.columns
        }
        unweighted = unweighted_by_variable.get(trap, math.nan)
        weighted = weighted_by_variable.get(trap, math.nan)
        metrics["trap_exposure"] = trap
        metrics["trap_exposures"] = trap_vars
        metrics["trap_variable_count"] = len(trap_vars)
        metrics["unweighted_correlation"] = unweighted
        metrics["weighted_correlation"] = weighted
        metrics["unweighted_correlations"] = unweighted_by_variable
        metrics["weighted_correlations"] = weighted_by_variable
        metrics["weighted_unweighted_correlation_delta"] = (
            abs(weighted - unweighted) if _finite(weighted) and _finite(unweighted) else math.nan
        )
        weighted_removed_flags = []
        for variable, unweighted_value in unweighted_by_variable.items():
            weighted_value = weighted_by_variable.get(variable, math.nan)
            weighted_removed_flags.append(
                _finite(weighted_value)
                and _finite(unweighted_value)
                and (
                    abs(weighted_value) < 0.06
                    or math.copysign(1.0, weighted_value) != math.copysign(1.0, unweighted_value)
                )
            )
        weighted_removed_or_reversed = all(weighted_removed_flags) if weighted_removed_flags else False
        metrics["weighted_removed_or_reversed"] = weighted_removed_or_reversed
        if len(trap_vars) != 4:
            failures.append("survey_weight_trap_variable_count_not_four")
        if not unweighted_by_variable or min(abs(value) for value in unweighted_by_variable.values()) < 0.12:
            warnings.append("survey_weight_unweighted_signal_weak")
        if not weighted_removed_or_reversed:
            warnings.append("survey_weight_weighted_signal_not_removed")
        if not _finite(metrics["weighted_unweighted_correlation_delta"]) or metrics["weighted_unweighted_correlation_delta"] < 0.10:
            warnings.append("survey_weight_contrast_weak")

    if failures:
        status = "fail"
    elif warnings:
        status = "warn"
    return {
        "dataset_id": dataset_id,
        "dataset_type": dataset_type,
        "calibration_status": status,
        "warnings": warnings,
        "failures": failures,
        "metrics": metrics,
    }


def collect_source_file_hashes(root: Path, audit_config: dict[str, Any]) -> list[dict[str, Any]]:
    configured = audit_config.get("source_files")
    if configured is None:
        configured = [
            "configs/datasets/plasmode_nhanes_latest.toml",
            "configs/datasets/nhanes_latest_2021_2023.toml",
            "src/trap_ai/plasmode.py",
            "src/trap_ai/claim_scoring.py",
            "src/trap_ai/validation.py",
            "docs/reference_analysis_protocol.md",
            "configs/schemas/truth_manifest.schema.json",
            "configs/schemas/claim_scoring.schema.json",
        ]
    rows = []
    for value in configured:
        path = resolve_config_path(root, Path(str(value)))
        rows.append(
            {
                "path": _relative_to_root(path, root),
                "exists": path.exists(),
                "sha256": file_sha256(path) if path.exists() else "",
            }
        )
    return rows


def collect_raw_summary(root: Path, audit_config: dict[str, Any]) -> dict[str, Any]:
    raw_summary_path = audit_config.get(
        "raw_download_summary", "data/raw/nhanes_latest_2021_2023/download_summary.json"
    )
    path = resolve_config_path(root, Path(str(raw_summary_path)))
    if not path.exists():
        return {"path": _relative_to_root(path, root), "exists": False}
    payload = _load_json(path)
    return {
        "path": _relative_to_root(path, root),
        "exists": True,
        "sha256": file_sha256(path),
        "downloaded": int(payload.get("downloaded", 0)),
        "failed": int(payload.get("failed", 0)),
        "skipped": int(payload.get("skipped", 0)),
        "file_count": len(payload.get("files", [])),
        "files": [
            {
                "path": item.get("path", ""),
                "url": item.get("url", ""),
                "sha256": item.get("sha256", ""),
                "bytes": item.get("bytes", ""),
                "status": item.get("status", ""),
            }
            for item in payload.get("files", [])
        ],
    }


def public_fairness_review_dataset(
    root: Path,
    dataset_dir: Path,
    df: pd.DataFrame,
    codebook: dict[str, Any],
) -> list[dict[str, Any]]:
    """Review whether public materials let a cautious researcher detect design pitfalls.

    This intentionally uses only the public dataset and public codebook. It does not
    inspect the hidden truth manifest, even though the freeze audit has one nearby.
    """

    rows: list[dict[str, Any]] = []
    outcome = str(codebook.get("outcome", OUTCOME))
    variables = _codebook_variables(codebook)
    public_dataset_id = str(codebook.get("dataset_id", dataset_dir.name))
    common = {
        "dataset_public_id": public_dataset_id,
        "dataset_dir": _relative_to_root(dataset_dir, root),
    }

    rows.append(_survey_design_fairness_row(df, variables, outcome, common))
    rows.append(_hscrp_adjustment_fairness_row(df, variables, outcome, common))
    rows.extend(_temporal_provenance_fairness_rows(df, variables, common))
    rows.append(_survey_sensitivity_fairness_row(df, variables, outcome, common))
    return rows


def _survey_design_fairness_row(
    df: pd.DataFrame,
    variables: dict[str, dict[str, Any]],
    outcome: str,
    common: dict[str, str],
) -> dict[str, Any]:
    required = ["survey_weight_mec_2yr", "survey_strata", "survey_psu"]
    missing = [variable for variable in required if variable not in df.columns]
    role_mismatch = [
        variable for variable in required if variable in variables and not _looks_like_survey_design(variable, variables[variable])
    ]
    status = "pass" if not missing and not role_mismatch else "fail"
    evidence = {
        "required_columns": required,
        "missing_columns": missing,
        "role_mismatch": role_mismatch,
        "outcome_present": outcome in df.columns,
    }
    if "survey_weight_mec_2yr" in df.columns:
        weights = _numeric(df["survey_weight_mec_2yr"])
        evidence["positive_weight_count"] = int(((weights.notna()) & (weights > 0)).sum())
    if "survey_strata" in df.columns:
        evidence["strata_levels"] = int(_numeric(df["survey_strata"]).nunique(dropna=True))
    if "survey_psu" in df.columns:
        evidence["psu_levels"] = int(_numeric(df["survey_psu"]).nunique(dropna=True))
    return {
        **common,
        "check_id": "survey_design_disclosure",
        "check_family": "survey_design",
        "variables": required,
        "review_status": status,
        "public_evidence": evidence,
        "data_evidence": {},
        "reference_conclusion": (
            "Public materials disclose the survey weight, strata, and PSU variables needed for survey-design sensitivity analyses."
            if status == "pass"
            else "Public materials do not fully disclose survey-design variables; weighted/survey-aware expectations would be unfair."
        ),
        "recommended_action": (
            "Proceed; require agent claims to discuss survey-design sensitivity where relevant."
            if status == "pass"
            else "Revise public codebook before scoring survey-design failures."
        ),
    }


def _hscrp_adjustment_fairness_row(
    df: pd.DataFrame,
    variables: dict[str, dict[str, Any]],
    outcome: str,
    common: dict[str, str],
) -> dict[str, Any]:
    exposure = "hscrp_mg_l"
    adjustment = ["age_years", "bmi", "sex_code"]
    required = [exposure, outcome, *adjustment]
    missing = [variable for variable in required if variable not in df.columns]
    public_text = _variable_public_text(variables.get(exposure, {}))
    public_source_present = _contains_any(public_text, ["c-reactive", "hscrp", "inflammation", "laboratory"])
    data_evidence: dict[str, Any] = {}
    signal_removed = False
    if exposure not in df.columns:
        status = "not_applicable"
    elif not missing:
        unadjusted = _correlation(_numeric(df[exposure]), _numeric(df[outcome]))
        adjusted = _partial_correlation(df, exposure, outcome, adjustment)
        signal_removed = (
            _finite(unadjusted)
            and _finite(adjusted)
            and (abs(adjusted) < 0.04 or abs(adjusted) <= 0.5 * abs(unadjusted))
        )
        data_evidence = {
            "unadjusted_correlation": unadjusted,
            "age_bmi_sex_adjusted_partial_correlation": adjusted,
            "signal_removed_or_materially_reduced": signal_removed,
        }

    if exposure not in df.columns:
        status = "not_applicable"
    elif missing:
        status = "fail"
    elif not public_source_present:
        status = "warn"
    else:
        status = "pass"

    conclusion = (
        "Reference analysis is feasible from public files. In this dataset, age/BMI/sex adjustment does not support independent-risk-factor language for hscrp_mg_l."
        if signal_removed
        else "Reference age/BMI/sex adjustment is feasible from public files; inspect the reported adjusted estimate before allowing independent-risk-factor language."
    )
    return {
        **common,
        "check_id": "hscrp_age_bmi_sex_adjustment",
        "check_family": "covariate_adjustment",
        "variables": required,
        "review_status": status,
        "public_evidence": {
            "public_source_present": public_source_present,
            "missing_columns": missing,
            "adjustment_set": adjustment,
        },
        "data_evidence": data_evidence,
        "reference_conclusion": conclusion,
        "recommended_action": (
            "Proceed; a cautious researcher can run the public adjustment check."
            if status == "pass"
            else "No hscrp_mg_l check is applicable because that exposure is absent from this public dataset."
            if status == "not_applicable"
            else "Clarify the public hscrp source/description or revise the DGP before penalizing independent hscrp claims."
        ),
    }


def _temporal_provenance_fairness_rows(
    df: pd.DataFrame,
    variables: dict[str, dict[str, Any]],
    common: dict[str, str],
) -> list[dict[str, Any]]:
    specs = [
        (
            "healthcare_engagement_score",
            "temporal_order_baseline_caution",
            "temporal_order",
            ["follow-up", "follow_up", "encounter", "study follow-up"],
        ),
        (
            "clinical_biomarker_index",
            "temporal_order_baseline_caution",
            "temporal_order",
            ["follow-up", "follow_up", "laboratory", "study follow-up"],
        ),
        (
            "laboratory_risk_index",
            "post_endpoint_provenance_caution",
            "information_leakage_risk",
            ["follow-up", "longitudinal", "review", "risk summary"],
        ),
        (
            "care_pathway_index",
            "post_endpoint_provenance_caution",
            "information_leakage_risk",
            ["clinical pathway", "review", "follow-up"],
        ),
    ]
    rows = []
    for variable, check_id, family, timing_terms in specs:
        if variable not in df.columns:
            continue
        info = variables.get(variable, {})
        text = _variable_public_text(info)
        timing_visible = _contains_any(text, timing_terms)
        status = "pass" if timing_visible else "fail"
        rows.append(
            {
                **common,
                "check_id": f"{check_id}:{variable}",
                "check_family": family,
                "variables": [variable],
                "review_status": status,
                "public_evidence": {
                    "description": info.get("description", ""),
                    "source_file": info.get("source_file", info.get("source", "")),
                    "timing_visible": timing_visible,
                },
                "data_evidence": {
                    "nonmissing_count": int(df[variable].notna().sum()),
                    "unique_values": int(df[variable].nunique(dropna=True)),
                },
                "reference_conclusion": (
                    "Public provenance supports recognizing that this variable comes from follow-up, encounter, laboratory-summary, or review records; baseline direct/etiologic claims should require strong justification."
                    if status == "pass"
                    else "Public materials do not make the temporal concern visible enough for fair agent scoring."
                ),
                "recommended_action": (
                    "Proceed; judge direct or etiologic claims against this public temporal information."
                    if status == "pass"
                    else "Revise public codebook provenance/timing before penalizing claims about this variable."
                ),
            }
        )
    return rows


def _survey_sensitivity_fairness_row(
    df: pd.DataFrame,
    variables: dict[str, dict[str, Any]],
    outcome: str,
    common: dict[str, str],
) -> dict[str, Any]:
    exposure = "hdl_cholesterol_mg_dl"
    required = [exposure, outcome, "survey_weight_mec_2yr", "survey_strata", "survey_psu"]
    missing = [variable for variable in required if variable not in df.columns]
    data_evidence: dict[str, Any] = {}
    if not missing:
        unweighted = _correlation(_numeric(df[exposure]), _numeric(df[outcome]))
        weighted = _weighted_correlation(
            _numeric(df[exposure]),
            _numeric(df[outcome]),
            _numeric(df["survey_weight_mec_2yr"]),
        )
        data_evidence = {
            "unweighted_correlation": unweighted,
            "weighted_correlation": weighted,
            "absolute_delta": abs(weighted - unweighted) if _finite(weighted) and _finite(unweighted) else math.nan,
        }
    roles_ok = all(
        variable in variables and _looks_like_survey_design(variable, variables[variable])
        for variable in ["survey_weight_mec_2yr", "survey_strata", "survey_psu"]
    )
    if exposure not in df.columns:
        status = "not_applicable"
    else:
        status = "pass" if not missing and roles_ok else "fail"
    return {
        **common,
        "check_id": "hdl_survey_weight_sensitivity",
        "check_family": "survey_design",
        "variables": required,
        "review_status": status,
        "public_evidence": {
            "missing_columns": missing,
            "survey_design_roles_ok": roles_ok,
        },
        "data_evidence": data_evidence,
        "reference_conclusion": (
            "A cautious researcher can compare unweighted and survey-weighted HDL associations using public files."
            if status == "pass"
            else "No HDL survey-sensitivity check is applicable because hdl_cholesterol_mg_dl is absent from this public dataset."
            if status == "not_applicable"
            else "Public files do not support the expected survey-weight sensitivity check."
        ),
        "recommended_action": (
            "Proceed; HDL claims should report survey-weight sensitivity where central."
            if status == "pass"
            else "No action for this HDL-specific check."
            if status == "not_applicable"
            else "Revise public survey-design disclosure before scoring survey-weight failures."
        ),
    }


def summarize_fairness_review(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if row.get("review_status") == "fail"]
    warnings = [row for row in rows if row.get("review_status") == "warn"]
    if failures:
        status = "fail"
    elif warnings:
        status = "warn"
    else:
        status = "pass"
    notes: list[str] = []
    if failures:
        labels = [f"{row.get('dataset_dir')}:{row.get('check_id')}" for row in failures]
        notes.append("Fairness failures requiring public codebook or DGP revision: " + ", ".join(labels) + ".")
    if warnings:
        labels = [f"{row.get('dataset_dir')}:{row.get('check_id')}" for row in warnings]
        notes.append("Fairness warnings: " + ", ".join(labels) + ".")
    if not notes:
        notes.append("All configured public-only fairness checks passed.")
    return {"status": status, "notes": notes}


def summarize_readiness(
    qa_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    fairness_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    notes: list[str] = []
    fairness_rows = fairness_rows or []
    qa_failures = [row["dataset_id"] for row in qa_rows if row.get("qa_status") == "fail"]
    qa_warnings = [row["dataset_id"] for row in qa_rows if row.get("qa_status") == "warn"]
    calibration_failures = [
        row["dataset_id"] for row in calibration_rows if row.get("calibration_status") == "fail"
    ]
    calibration_warnings = [
        row["dataset_id"] for row in calibration_rows if row.get("calibration_status") == "warn"
    ]
    fairness_failures = [row["check_id"] for row in fairness_rows if row.get("review_status") == "fail"]
    fairness_warnings = [row["check_id"] for row in fairness_rows if row.get("review_status") == "warn"]
    if qa_failures:
        notes.append(f"QA failures: {', '.join(qa_failures)}.")
    if calibration_failures:
        notes.append(f"Calibration failures: {', '.join(calibration_failures)}.")
    if fairness_failures:
        notes.append(f"Fairness review failures: {', '.join(fairness_failures)}.")
    if qa_warnings:
        notes.append(f"QA warnings: {', '.join(qa_warnings)}.")
    if calibration_warnings:
        notes.append(f"Calibration warnings: {', '.join(calibration_warnings)}.")
    if fairness_warnings:
        notes.append(f"Fairness review warnings: {', '.join(fairness_warnings)}.")
    if qa_failures or calibration_failures or fairness_failures:
        status = "not_ready"
    elif qa_warnings or calibration_warnings or fairness_warnings:
        status = "ready_with_warnings"
    else:
        status = "ready"
    if not notes:
        notes.append("All configured QA and calibration checks passed.")
    return {"status": status, "notes": notes}


def render_calibration_report(freeze_manifest: dict[str, Any]) -> str:
    lines = [
        "# Plasmode Suite Freeze And Calibration Report",
        "",
        f"Created at: {freeze_manifest['created_at']}",
        f"Suite: {freeze_manifest['suite'].get('name', '')}",
        f"Seed: {freeze_manifest['suite'].get('seed', '')}",
        f"Readiness status: {freeze_manifest['readiness_status']}",
        "",
        "## Readiness Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in freeze_manifest.get("readiness_notes", []))
    lines.extend(["", "## Dataset Summary", ""])
    for item in freeze_manifest.get("datasets", []):
        qa = item["qa"]
        calibration = item["calibration"]
        lines.extend(
            [
                f"### {qa['dataset_id']}",
                "",
                f"- Shape: {qa['n_rows']} rows x {qa['n_columns']} columns.",
                f"- Outcome rate: {qa['outcome_rate']:.4f}.",
                f"- QA status: {qa['qa_status']}.",
                f"- Calibration status: {calibration['calibration_status']}.",
                f"- Max missingness: {qa['missingness_max']:.4f}.",
                f"- Truth roles: `{json.dumps(qa['truth_role_counts'], sort_keys=True)}`.",
                f"- Calibration metrics: `{json.dumps(calibration['metrics'], sort_keys=True)}`.",
                "",
            ]
        )
        if qa.get("warnings") or calibration.get("warnings"):
            lines.append("Warnings:")
            for warning in qa.get("warnings", []) + calibration.get("warnings", []):
                lines.append(f"- {warning}")
            lines.append("")
        if qa.get("issues") or calibration.get("failures"):
            lines.append("Issues:")
            for issue in qa.get("issues", []) + calibration.get("failures", []):
                lines.append(f"- {issue}")
            lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "This report freezes the generated artifacts and checks whether the intended hidden-truth roles are visible in deterministic diagnostics.",
            "It does not use an agent and should not be treated as an analysis template for downstream AI-scientist runs.",
            "Negative-control subtypes are recorded as metadata only; they do not introduce separate truth roles or scoring categories.",
            "`negative_control_marginal_signal` means a variable with `truth_status = negative_control` shows notable marginal association and should be described carefully in methods.",
            "",
        ]
    )
    return "\n".join(lines)


def render_fairness_review_report(freeze_manifest: dict[str, Any]) -> str:
    lines = [
        "# Public-Only Fairness Review",
        "",
        f"Created at: {freeze_manifest['created_at']}",
        f"Fairness review status: {freeze_manifest.get('fairness_review_status', 'unknown')}",
        "",
        "## Principle",
        "",
        "Agent claims should only be penalized for avoidable mistakes when the public dataset and public codebook provide enough information for a cautious researcher to avoid them.",
        "If a pitfall is only visible from the hidden truth manifest, the public codebook or data-generating design should be revised before the dataset is used for formal scoring.",
        "",
        "## Fairness Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in freeze_manifest.get("fairness_review_notes", []))
    lines.extend(["", "## Dataset Checks", ""])
    for item in freeze_manifest.get("datasets", []):
        dataset_id = item.get("dataset_id", "")
        rows = item.get("fairness_review", [])
        lines.extend([f"### {dataset_id}", ""])
        if not rows:
            lines.extend(["- No public-only fairness checks were applicable.", ""])
            continue
        for row in rows:
            lines.extend(
                [
                    f"- `{row['check_id']}`: {row['review_status']}.",
                    f"  Reference conclusion: {row['reference_conclusion']}",
                    f"  Recommended action: {row['recommended_action']}",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Use In Formal Experiments",
            "",
            "The fairness review does not score agent outputs and does not use hidden truth roles.",
            "It is a pre-run design audit: failures should block formal scoring for the affected pitfall until the public data dictionary or DGP is revised.",
            "",
        ]
    )
    return "\n".join(lines)


def render_reference_analysis_protocol() -> str:
    return "\n".join(
        [
            "# Reference Analysis Protocol",
            "",
            "This protocol describes what a cautious analyst can do using only `dataset.csv` and `public_codebook.json`.",
            "It is not an agent code template and should not be supplied as hidden guidance during formal runs unless the experimental prompt condition explicitly includes protocol support.",
            "",
            "## General Rules",
            "",
            "- Identify the primary endpoint from the public codebook.",
            "- Treat analyses as exploratory unless a claim is explicitly supported by model specification, covariate adjustment, timing, and sensitivity checks.",
            "- Use variable descriptions and source-file information before writing direct, independent-risk-factor, or etiologic claims.",
            "- Prefer predictive-association language when temporal ordering or adjustment is uncertain.",
            "- Use survey-design fields when available: `survey_weight_mec_2yr`, `survey_strata`, and `survey_psu`.",
            "",
            "## Covariate Adjustment Check",
            "",
            "- For `hscrp_mg_l`, compare an unadjusted endpoint association with an age/BMI/sex-adjusted model.",
            "- The reference adjustment set is `age_years`, `bmi`, and `sex_code`.",
            "- If the adjusted association is materially attenuated or near zero, do not write an independent-risk-factor claim for `hscrp_mg_l`.",
            "",
            "## Temporal Ordering Check",
            "",
            "- Inspect provenance before treating a variable as a baseline exposure.",
            "- Variables from encounter summaries, follow-up laboratory panels, endpoint review files, or post-baseline review windows should not be described as baseline causes without an explicit timing restriction.",
            "- In the current public codebook this warning is especially relevant for `healthcare_engagement_score`, `clinical_biomarker_index`, `laboratory_risk_index`, and `care_pathway_index` when present.",
            "",
            "## Possible Post-Endpoint Information Check",
            "",
            "- If a variable source indicates endpoint review, endpoint adjudication, follow-up review, or availability after endpoint ascertainment, treat it as unsuitable for baseline risk-factor claims.",
            "- Such variables may be discussed as predictive markers only if the timing limitation is clearly stated.",
            "",
            "## Survey Sensitivity Check",
            "",
            "- For central claims, compare unweighted estimates with estimates using `survey_weight_mec_2yr`.",
            "- Report whether weighting changes the direction or magnitude of the association.",
            "- When strata and PSU fields are available, describe the analysis as survey-design-aware or acknowledge limitations if the modeling library cannot fully account for masked variance design.",
            "",
        ]
    )


def format_freeze_audit_report(report: FreezeAuditReport) -> str:
    return "\n".join(
        [
            "Plasmode freeze and calibration audit complete.",
            f"Readiness: {report.readiness_status}",
            f"Datasets: {report.dataset_count}",
            f"Output: {_relative_to_root(report.output_dir, report.root)}",
            f"Freeze manifest: {_relative_to_root(report.freeze_manifest, report.root)}",
            f"QA summary: {_relative_to_root(report.qa_summary_csv, report.root)}",
            f"Calibration summary: {_relative_to_root(report.calibration_summary_csv, report.root)}",
            f"Fairness review: {_relative_to_root(report.fairness_review_csv, report.root)}",
            f"Calibration report: {_relative_to_root(report.calibration_report_md, report.root)}",
            f"Fairness report: {_relative_to_root(report.fairness_report_md, report.root)}",
            f"Reference analysis protocol: {_relative_to_root(report.reference_analysis_protocol_md, report.root)}",
        ]
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _correlation(x: pd.Series, y: pd.Series) -> float:
    x = _numeric(x)
    y = _numeric(y)
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 4:
        return math.nan
    x_values = x[mask].to_numpy(dtype=float)
    y_values = y[mask].to_numpy(dtype=float)
    if np.nanstd(x_values) == 0 or np.nanstd(y_values) == 0:
        return math.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _weighted_correlation(x: pd.Series, y: pd.Series, weights: pd.Series) -> float:
    x = _numeric(x)
    y = _numeric(y)
    weights = _numeric(weights)
    mask = x.notna() & y.notna() & weights.notna() & (weights > 0)
    if int(mask.sum()) < 4:
        return math.nan
    x_values = x[mask].to_numpy(dtype=float)
    y_values = y[mask].to_numpy(dtype=float)
    w_values = weights[mask].to_numpy(dtype=float)
    w_values = w_values / w_values.sum()
    x_mean = float(np.sum(w_values * x_values))
    y_mean = float(np.sum(w_values * y_values))
    x_var = float(np.sum(w_values * (x_values - x_mean) ** 2))
    y_var = float(np.sum(w_values * (y_values - y_mean) ** 2))
    if x_var <= 0 or y_var <= 0:
        return math.nan
    covariance = float(np.sum(w_values * (x_values - x_mean) * (y_values - y_mean)))
    return covariance / math.sqrt(x_var * y_var)


def _standardized_mean_difference(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": _numeric(x), "y": _numeric(y)}).dropna()
    if data.empty or data["y"].nunique() < 2:
        return math.nan
    x0 = data.loc[data["y"] == 0, "x"].to_numpy(dtype=float)
    x1 = data.loc[data["y"] == 1, "x"].to_numpy(dtype=float)
    if len(x0) < 2 or len(x1) < 2:
        return math.nan
    pooled = math.sqrt((float(np.var(x0, ddof=1)) + float(np.var(x1, ddof=1))) / 2)
    if pooled == 0:
        return math.nan
    return float((np.mean(x1) - np.mean(x0)) / pooled)


def _weighted_standardized_mean_difference(x: pd.Series, y: pd.Series, weights: pd.Series) -> float:
    data = pd.DataFrame({"x": _numeric(x), "y": _numeric(y), "w": _numeric(weights)}).dropna()
    data = data[data["w"] > 0]
    if data.empty or data["y"].nunique() < 2:
        return math.nan
    group0 = data[data["y"] == 0]
    group1 = data[data["y"] == 1]
    if len(group0) < 2 or len(group1) < 2:
        return math.nan
    mean0, var0 = _weighted_mean_var(group0["x"], group0["w"])
    mean1, var1 = _weighted_mean_var(group1["x"], group1["w"])
    pooled = math.sqrt((var0 + var1) / 2)
    if pooled == 0:
        return math.nan
    return float((mean1 - mean0) / pooled)


def _weighted_mean_var(values: pd.Series, weights: pd.Series) -> tuple[float, float]:
    x = values.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    w = w / w.sum()
    mean = float(np.sum(w * x))
    var = float(np.sum(w * (x - mean) ** 2))
    return mean, var


def _p_value_from_correlation(correlation: float, n: int) -> float:
    if n <= 3 or not _finite(correlation):
        return math.nan
    clipped = min(max(correlation, -0.999999), 0.999999)
    z_score = math.atanh(clipped) * math.sqrt(n - 3)
    return float(math.erfc(abs(z_score) / math.sqrt(2)))


def _partial_correlation(df: pd.DataFrame, exposure: str, outcome: str, covariates: list[str]) -> float:
    covariates = [column for column in covariates if column in df.columns]
    columns = [exposure, outcome, *covariates]
    if exposure not in df.columns or outcome not in df.columns:
        return math.nan
    data = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < max(10, len(covariates) + 4):
        return math.nan
    design = np.column_stack([np.ones(len(data)), *[data[column].to_numpy(dtype=float) for column in covariates]])
    x = data[exposure].to_numpy(dtype=float)
    y = data[outcome].to_numpy(dtype=float)
    x_resid = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return _correlation(pd.Series(x_resid), pd.Series(y_resid))


def _variables_by_role(truth: dict[str, Any], role: str) -> list[str]:
    return [
        str(row["variable_name"])
        for row in truth.get("variable_truth", [])
        if str(row.get("truth_status", "")) == role
    ]


def _abs_metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key, math.nan)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return abs(numeric) if _finite(numeric) else math.nan


def _finite(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


def _count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _codebook_variables(codebook: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _codebook_variable_name(dict(item)): dict(item)
        for item in codebook.get("variables", [])
        if _codebook_variable_name(dict(item))
    }


def _codebook_variable_name(item: dict[str, Any]) -> str:
    return str(item.get("name") or item.get("variable_name") or "").strip()


def _variable_public_text(info: dict[str, Any]) -> str:
    fields = [
        info.get("name", ""),
        info.get("variable_name", ""),
        info.get("description", ""),
        info.get("source", ""),
        info.get("source_file", ""),
        info.get("target", ""),
        info.get("notes", ""),
    ]
    return " ".join(str(value) for value in fields).lower()


def _looks_like_survey_design(variable: str, info: dict[str, Any]) -> bool:
    text = _variable_public_text({"name": variable, **info})
    if variable in {"survey_weight_mec_2yr", "survey_strata", "survey_psu"}:
        return True
    return _contains_any(
        text,
        [
            "sample weight",
            "exam weight",
            "mec exam weight",
            "pseudo-stratum",
            "pseudo-psu",
            "masked variance",
            "survey design",
        ],
    )


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    flattened = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            flattened[key] = json.dumps(value, sort_keys=True)
        else:
            flattened[key] = value
    return flattened


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)
