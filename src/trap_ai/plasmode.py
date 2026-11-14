"""Plasmode dataset generation for outcome and trap variables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trap_ai.config import load_toml, resolve_config_path
from trap_ai.randomness import DEFAULT_SEED, set_global_seed


OUTCOME = "cardiometabolic_risk_status"
PUBLIC_DATASET_ID = "cardiometabolic_risk_study"
PUBLIC_DATASET_TYPE = "cross_sectional_epidemiology"


BASE_VARIABLE_CODEBOOK: dict[str, dict[str, str]] = {
    "participant_id": {
        "description": "De-identified NHANES participant sequence number.",
        "source": "DEMO_L.SEQN",
        "public_role": "identifier",
    },
    "age_years": {
        "description": "Age in years at screening.",
        "source": "DEMO_L.RIDAGEYR",
        "public_role": "demographic_covariate",
    },
    "sex_code": {
        "description": "NHANES sex code.",
        "source": "DEMO_L.RIAGENDR",
        "public_role": "demographic_covariate",
    },
    "race_ethnicity_code": {
        "description": "NHANES race and Hispanic origin code.",
        "source": "DEMO_L.RIDRETH3",
        "public_role": "demographic_covariate",
    },
    "education_adult_code": {
        "description": "Adult education level code when available.",
        "source": "DEMO_L.DMDEDUC2",
        "public_role": "socioeconomic_covariate",
    },
    "income_poverty_ratio": {
        "description": "Family income to poverty ratio.",
        "source": "DEMO_L.INDFMPIR",
        "public_role": "socioeconomic_covariate",
    },
    "survey_weight_mec_2yr": {
        "description": "MEC exam sample weight for the 2021-2023 public-use cycle.",
        "source": "DEMO_L.WTMEC2YR",
        "public_role": "survey_design",
    },
    "survey_strata": {
        "description": "Masked variance pseudo-stratum.",
        "source": "DEMO_L.SDMVSTRA",
        "public_role": "survey_design",
    },
    "survey_psu": {
        "description": "Masked variance pseudo-PSU.",
        "source": "DEMO_L.SDMVPSU",
        "public_role": "survey_design",
    },
    "bmi": {
        "description": "Body mass index in kg/m^2.",
        "source": "BMX_L.BMXBMI",
        "public_role": "candidate_predictor",
    },
    "waist_cm": {
        "description": "Waist circumference in centimeters.",
        "source": "BMX_L.BMXWAIST",
        "public_role": "candidate_predictor",
    },
    "weight_kg": {
        "description": "Body weight in kilograms.",
        "source": "BMX_L.BMXWT",
        "public_role": "candidate_predictor",
    },
    "height_cm": {
        "description": "Standing height in centimeters.",
        "source": "BMX_L.BMXHT",
        "public_role": "candidate_predictor",
    },
    "systolic_bp_mean": {
        "description": "Mean oscillometric systolic blood pressure across available readings.",
        "source": "BPXO_L.BPXOSY1/BPXOSY2/BPXOSY3",
        "public_role": "candidate_predictor",
    },
    "diastolic_bp_mean": {
        "description": "Mean oscillometric diastolic blood pressure across available readings.",
        "source": "BPXO_L.BPXODI1/BPXODI2/BPXODI3",
        "public_role": "candidate_predictor",
    },
    "hba1c_percent": {
        "description": "Glycohemoglobin percentage.",
        "source": "GHB_L.LBXGH",
        "public_role": "candidate_predictor",
    },
    "hscrp_mg_l": {
        "description": "High-sensitivity C-reactive protein in mg/L.",
        "source": "HSCRP_L.LBXHSCRP",
        "public_role": "candidate_predictor",
        "collection_window": "baseline examination",
    },
    "total_cholesterol_mg_dl": {
        "description": "Total cholesterol in mg/dL.",
        "source": "TCHOL_L.LBXTC",
        "public_role": "candidate_predictor",
    },
    "hdl_cholesterol_mg_dl": {
        "description": "HDL cholesterol in mg/dL.",
        "source": "HDL_L.LBDHDD",
        "public_role": "candidate_predictor",
    },
    "alcohol_use_frequency_code": {
        "description": "Alcohol use frequency code.",
        "source": "ALQ_L.ALQ121",
        "public_role": "candidate_predictor",
    },
    "alcohol_drinks_per_day": {
        "description": "Typical number of alcoholic drinks on drinking days.",
        "source": "ALQ_L.ALQ130",
        "public_role": "candidate_predictor",
    },
    "diabetes_doctor_told_code": {
        "description": "Doctor told participant they have diabetes code.",
        "source": "DIQ_L.DIQ010",
        "public_role": "candidate_predictor",
    },
    "asthma_doctor_told_code": {
        "description": "Doctor told participant they have asthma code.",
        "source": "MCQ_L.MCQ010",
        "public_role": "candidate_predictor",
    },
    "physical_activity_sedentary_minutes": {
        "description": "Minutes of sedentary activity on a typical day.",
        "source": "PAQ_L.PAD680",
        "public_role": "candidate_predictor",
    },
    "smoker_ever_code": {
        "description": "Smoked at least 100 cigarettes in life code.",
        "source": "SMQ_L.SMQ020",
        "public_role": "candidate_predictor",
    },
    "current_smoking_code": {
        "description": "Current cigarette smoking status code.",
        "source": "SMQ_L.SMQ040",
        "public_role": "candidate_predictor",
    },
    "diet_kcal_day1": {
        "description": "Total energy intake on dietary recall day 1.",
        "source": "DR1TOT_L.DR1TKCAL",
        "public_role": "candidate_predictor",
    },
    "diet_sugar_g_day1": {
        "description": "Total sugar intake on dietary recall day 1.",
        "source": "DR1TOT_L.DR1TSUGR",
        "public_role": "candidate_predictor",
    },
    "diet_fiber_g_day1": {
        "description": "Total fiber intake on dietary recall day 1.",
        "source": "DR1TOT_L.DR1TFIBE",
        "public_role": "candidate_predictor",
    },
    "diet_total_fat_g_day1": {
        "description": "Total fat intake on dietary recall day 1.",
        "source": "DR1TOT_L.DR1TTFAT",
        "public_role": "candidate_predictor",
    },
}


GENERATED_PUBLIC_CODEBOOK: dict[str, dict[str, str]] = {
    OUTCOME: {
        "description": "Binary indicator for elevated cardiometabolic risk.",
        "source": "Study endpoint adjudication file.",
        "public_role": "outcome",
        "collection_window": "study endpoint review",
    },
    "inflammatory_marker_index": {
        "description": "Baseline laboratory inflammation marker index.",
        "source": "Study laboratory panel.",
        "public_role": "candidate_predictor",
        "collection_window": "baseline examination",
    },
    "body_composition_index": {
        "description": "Anthropometric index summarizing body composition.",
        "source": "Study examination file.",
        "public_role": "candidate_predictor",
        "collection_window": "baseline examination",
    },
    "diet_quality_pattern_score": {
        "description": "Composite dietary pattern score based on recall data.",
        "source": "Study dietary scoring algorithm.",
        "public_role": "candidate_predictor",
        "collection_window": "day-1 dietary recall",
    },
    "metabolic_panel_index": {
        "description": "Composite baseline metabolic panel index.",
        "source": "Study laboratory/examination composite.",
        "public_role": "candidate_predictor",
        "collection_window": "baseline examination",
    },
    "care_access_index": {
        "description": "Care access index.",
        "source": "Study operations file.",
        "public_role": "candidate_predictor",
        "collection_window": "baseline scheduling",
    },
    "inflammatory_burden_score": {
        "description": "Composite inflammatory burden score from baseline records.",
        "source": "Study laboratory/examination composite.",
        "public_role": "candidate_predictor",
        "collection_window": "baseline examination",
    },
    "vascular_health_index": {
        "description": "Vascular health index derived from examination and demographic measures.",
        "source": "Study examination composite.",
        "public_role": "candidate_predictor",
        "collection_window": "baseline examination",
    },
    "adiposity_inflammation_score": {
        "description": "Composite adiposity and inflammation score.",
        "source": "Study laboratory/examination composite.",
        "public_role": "candidate_predictor",
        "collection_window": "baseline examination",
    },
    "healthcare_engagement_score": {
        "description": "Healthcare engagement score based on the number and timing of follow-up encounters.",
        "source": "Study encounter summary.",
        "public_role": "candidate_predictor",
        "collection_window": "study follow-up",
    },
    "clinical_biomarker_index": {
        "description": "Biomarker index measured at the follow-up examination.",
        "source": "Study laboratory panel.",
        "public_role": "candidate_predictor",
        "collection_window": "study follow-up",
    },
    "treatment_intensity_score": {
        "description": "Treatment intensity score based on medications, visits, and care actions during study follow-up.",
        "source": "Study encounter summary.",
        "public_role": "candidate_predictor",
        "collection_window": "study follow-up",
    },
    "followup_health_status_score": {
        "description": "Health status summary from follow-up examination and encounter records.",
        "source": "Study health status summary.",
        "public_role": "candidate_predictor",
        "collection_window": "study follow-up",
    },
    "laboratory_risk_index": {
        "description": "Laboratory risk index compiled from longitudinal laboratory records collected through endpoint review.",
        "source": "Laboratory risk summary.",
        "public_role": "candidate_predictor",
        "collection_window": "study follow-up and review",
    },
    "care_pathway_index": {
        "description": "Care pathway intensity index summarized from clinical pathway records reviewed after follow-up completion.",
        "source": "Clinical pathway review file.",
        "public_role": "candidate_predictor",
        "collection_window": "study follow-up review",
    },
    "medication_profile_score": {
        "description": "Medication profile score summarized from longitudinal medication records updated through follow-up review.",
        "source": "Medication profile summary.",
        "public_role": "candidate_predictor",
        "collection_window": "study follow-up review",
    },
    "clinical_review_score": {
        "description": "Clinical review score from chart abstraction completed after endpoint review.",
        "source": "Clinical review file.",
        "public_role": "candidate_predictor",
        "collection_window": "study follow-up review",
    },
    "lipid_balance_index": {
        "description": "Lipid balance index derived from baseline lipid measurements.",
        "source": "Study laboratory composite.",
        "public_role": "candidate_predictor",
        "collection_window": "baseline examination",
    },
    "nutrition_lipid_score": {
        "description": "Nutrition and lipid pattern score.",
        "source": "Study dietary/laboratory composite.",
        "public_role": "candidate_predictor",
        "collection_window": "baseline examination and day-1 dietary recall",
    },
    "metabolic_screening_index": {
        "description": "Metabolic screening index from baseline examination and laboratory records.",
        "source": "Study laboratory/examination composite.",
        "public_role": "candidate_predictor",
        "collection_window": "baseline examination",
    },
}


NEGATIVE_CONTROL_TRAP_SPECS: tuple[tuple[str, str, str], ...] = (
    ("inflammatory_marker_index", "stratified_shuffle", "hscrp_mg_l"),
    ("body_composition_index", "stratified_shuffle", "bmi"),
    ("diet_quality_pattern_score", "disguised_random", ""),
    ("metabolic_panel_index", "stratified_shuffle", "hba1c_percent"),
)

CONFOUNDING_TRAP_VARIABLES = [
    "hscrp_mg_l",
    "inflammatory_burden_score",
    "vascular_health_index",
    "adiposity_inflammation_score",
]
COLLIDER_TRAP_VARIABLES = [
    "healthcare_engagement_score",
    "clinical_biomarker_index",
    "treatment_intensity_score",
    "followup_health_status_score",
]
LEAKAGE_TRAP_VARIABLES = [
    "laboratory_risk_index",
    "care_pathway_index",
    "medication_profile_score",
    "clinical_review_score",
]
SURVEY_WEIGHT_TRAP_VARIABLES = [
    "hdl_cholesterol_mg_dl",
    "lipid_balance_index",
    "nutrition_lipid_score",
    "metabolic_screening_index",
]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    kind: str
    description: str


@dataclass(frozen=True)
class GeneratedDataset:
    dataset_id: str
    kind: str
    directory: Path
    dataset_csv: Path
    public_codebook_json: Path
    hidden_truth_manifest_json: Path
    n_rows: int
    n_columns: int


@dataclass(frozen=True)
class PlasmodeReport:
    root: Path
    output_dir: Path
    seed: int
    generated: list[GeneratedDataset]
    suite_manifest: Path


def generate_plasmode_suite(
    root: Path,
    config_path: Path,
    overwrite: bool = False,
) -> PlasmodeReport:
    """Generate every plasmode dataset described by a TOML config."""

    root = root.resolve()
    config = load_toml(resolve_config_path(root, config_path))
    generator = config["generator"]
    seed = int(generator.get("seed", DEFAULT_SEED))
    set_global_seed(seed)

    raw_dir = root / str(generator["raw_dir"])
    cycle = str(generator.get("cycle", "2021-2023"))
    output_dir = root / str(generator["output_dir"])
    max_rows = int(generator.get("max_rows", 0))

    base = load_nhanes_base_table(raw_dir=raw_dir, cycle=cycle)
    if max_rows > 0 and len(base) > max_rows:
        base = base.sample(n=max_rows, random_state=seed).sort_values("participant_id")

    specs = [
        DatasetSpec(
            dataset_id=str(item["id"]),
            kind=str(item["kind"]),
            description=str(item.get("description", "")),
        )
        for item in config.get("datasets", [])
    ]
    if not specs:
        specs = default_dataset_specs()

    generated: list[GeneratedDataset] = []
    for index, spec in enumerate(specs):
        generated.append(
            generate_single_plasmode_dataset(
                base=base,
                spec=spec,
                output_dir=output_dir / spec.dataset_id,
                seed=seed + index * 9973,
                overwrite=overwrite,
                suite_name=str(generator.get("name", "plasmode_suite")),
            )
        )

    suite_manifest = output_dir / "suite_manifest.json"
    _write_json(
        suite_manifest,
        {
            "schema_version": "0.1",
            "suite": str(generator.get("name", "plasmode_suite")),
            "description": str(generator.get("description", "")),
            "seed": seed,
            "source_dataset_config": str(generator.get("source_dataset_config", "")),
            "raw_dir": _relative_to_root(raw_dir, root),
            "datasets": [
                {
                    "dataset_id": item.dataset_id,
                    "kind": item.kind,
                    "directory": _relative_to_root(item.directory, root),
                    "dataset_csv": _relative_to_root(item.dataset_csv, root),
                    "public_codebook_json": _relative_to_root(item.public_codebook_json, root),
                    "hidden_truth_manifest_json": _relative_to_root(
                        item.hidden_truth_manifest_json, root
                    ),
                    "n_rows": item.n_rows,
                    "n_columns": item.n_columns,
                }
                for item in generated
            ],
        },
        overwrite=True,
    )

    return PlasmodeReport(
        root=root,
        output_dir=output_dir,
        seed=seed,
        generated=generated,
        suite_manifest=suite_manifest,
    )


def default_dataset_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec("true_effect", "true_effect", "Direct true-effect plasmode dataset."),
        DatasetSpec("pure_null", "pure_null", "Pure-null plasmode dataset."),
        DatasetSpec(
            "negative_control_trap",
            "negative_control_trap",
            "Temptingly named negative-control variable trap dataset.",
        ),
        DatasetSpec("confounding_trap", "confounding_trap", "Confounding trap dataset."),
        DatasetSpec(
            "collider_post_treatment_trap",
            "collider_post_treatment_trap",
            "Collider and post-treatment adjustment trap dataset.",
        ),
        DatasetSpec("leakage_trap", "leakage_trap", "Leakage trap dataset."),
        DatasetSpec("survey_weight_trap", "survey_weight_trap", "Survey-weight trap dataset."),
    ]


def load_nhanes_base_table(raw_dir: Path, cycle: str = "2021-2023") -> pd.DataFrame:
    """Load and merge the NHANES latest files needed by the trap generator."""

    demo = _read_xpt(raw_dir / cycle / "demographics" / "DEMO_L.xpt")
    base = pd.DataFrame(
        {
            "participant_id": _to_numeric(demo["SEQN"]),
            "age_years": _to_numeric(demo["RIDAGEYR"]),
            "sex_code": _clean_codes(demo["RIAGENDR"], invalid={7, 9}),
            "race_ethnicity_code": _clean_codes(demo["RIDRETH3"], invalid={77, 99}),
            "education_adult_code": _clean_codes(demo["DMDEDUC2"], invalid={7, 9}),
            "income_poverty_ratio": _to_numeric(demo["INDFMPIR"]),
            "survey_weight_mec_2yr": _to_numeric(demo["WTMEC2YR"]),
            "survey_strata": _to_numeric(demo["SDMVSTRA"]),
            "survey_psu": _to_numeric(demo["SDMVPSU"]),
        }
    )

    loaders = [
        ("examination", "BMX_L.xpt", _derive_bmx),
        ("examination", "BPXO_L.xpt", _derive_bpxo),
        ("laboratory", "GHB_L.xpt", lambda df: {"hba1c_percent": _to_numeric(df["LBXGH"])}),
        ("laboratory", "HSCRP_L.xpt", lambda df: {"hscrp_mg_l": _to_numeric(df["LBXHSCRP"])}),
        (
            "laboratory",
            "TCHOL_L.xpt",
            lambda df: {"total_cholesterol_mg_dl": _to_numeric(df["LBXTC"])},
        ),
        (
            "laboratory",
            "HDL_L.xpt",
            lambda df: {"hdl_cholesterol_mg_dl": _to_numeric(df["LBDHDD"])},
        ),
        ("questionnaire", "ALQ_L.xpt", _derive_alq),
        ("questionnaire", "DIQ_L.xpt", _derive_diq),
        ("questionnaire", "MCQ_L.xpt", _derive_mcq),
        ("questionnaire", "PAQ_L.xpt", _derive_paq),
        ("questionnaire", "SMQ_L.xpt", _derive_smq),
        ("dietary", "DR1TOT_L.xpt", _derive_dr1tot),
    ]

    for component, filename, derive in loaders:
        source = _read_xpt(raw_dir / cycle / component / filename)
        derived = pd.DataFrame({"participant_id": _to_numeric(source["SEQN"]), **derive(source)})
        base = base.merge(derived, on="participant_id", how="left")

    base["participant_id"] = base["participant_id"].astype(int)
    return base.sort_values("participant_id").reset_index(drop=True)


def generate_single_plasmode_dataset(
    base: pd.DataFrame,
    spec: DatasetSpec,
    output_dir: Path,
    seed: int,
    overwrite: bool = False,
    suite_name: str = "plasmode_suite",
) -> GeneratedDataset:
    if output_dir.exists() and not overwrite:
        expected = [
            output_dir / "dataset.csv",
            output_dir / "public_codebook.json",
            output_dir / "hidden_truth_manifest.json",
        ]
        if all(path.exists() for path in expected):
            return GeneratedDataset(
                dataset_id=spec.dataset_id,
                kind=spec.kind,
                directory=output_dir,
                dataset_csv=expected[0],
                public_codebook_json=expected[1],
                hidden_truth_manifest_json=expected[2],
                n_rows=len(pd.read_csv(expected[0], usecols=["participant_id"])),
                n_columns=len(pd.read_csv(expected[0], nrows=0).columns),
            )

    rng = np.random.default_rng(seed)
    data = base.copy()

    dgp_details: dict[str, Any]
    roles: dict[str, dict[str, Any]]
    if spec.kind == "true_effect":
        dgp_details, roles = _apply_true_effect(data, rng)
    elif spec.kind == "pure_null":
        dgp_details, roles = _apply_pure_null(data, rng)
    elif spec.kind == "negative_control_trap":
        dgp_details, roles = _apply_negative_control_trap(data, rng)
    elif spec.kind == "confounding_trap":
        dgp_details, roles = _apply_confounding_trap(data, rng)
    elif spec.kind == "collider_post_treatment_trap":
        dgp_details, roles = _apply_collider_post_treatment_trap(data, rng)
    elif spec.kind == "leakage_trap":
        dgp_details, roles = _apply_leakage_trap(data, rng)
    elif spec.kind == "survey_weight_trap":
        dgp_details, roles = _apply_survey_weight_trap(data, rng)
    else:
        raise ValueError(f"Unsupported plasmode dataset kind: {spec.kind}")

    data = data[_ordered_columns(data)]
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = output_dir / "dataset.csv"
    public_codebook_json = output_dir / "public_codebook.json"
    hidden_truth_manifest_json = output_dir / "hidden_truth_manifest.json"

    data.to_csv(dataset_csv, index=False)
    _write_json(
        public_codebook_json,
        _public_codebook_payload(spec=spec, suite_name=suite_name, data=data),
        overwrite=True,
    )
    _write_json(
        hidden_truth_manifest_json,
        _hidden_truth_payload(
            spec=spec,
            suite_name=suite_name,
            seed=seed,
            data=data,
            dgp_details=dgp_details,
            roles=roles,
        ),
        overwrite=True,
    )

    return GeneratedDataset(
        dataset_id=spec.dataset_id,
        kind=spec.kind,
        directory=output_dir,
        dataset_csv=dataset_csv,
        public_codebook_json=public_codebook_json,
        hidden_truth_manifest_json=hidden_truth_manifest_json,
        n_rows=len(data),
        n_columns=len(data.columns),
    )


def format_plasmode_report(report: PlasmodeReport) -> str:
    lines = [
        "Plasmode trap suite ready.",
        f"Seed: {report.seed}",
        f"Output: {_relative_to_root(report.output_dir, report.root)}",
        f"Suite manifest: {_relative_to_root(report.suite_manifest, report.root)}",
        "",
        "Datasets:",
    ]
    for item in report.generated:
        lines.append(
            f"  - {item.dataset_id} ({item.kind}): "
            f"{item.n_rows} rows x {item.n_columns} columns"
        )
        lines.append(f"    dataset: {_relative_to_root(item.dataset_csv, report.root)}")
        lines.append(
            f"    public_codebook: {_relative_to_root(item.public_codebook_json, report.root)}"
        )
        lines.append(
            "    hidden_truth_manifest: "
            f"{_relative_to_root(item.hidden_truth_manifest_json, report.root)}"
        )
    return "\n".join(lines)


def _derive_bmx(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "bmi": _to_numeric(df["BMXBMI"]),
        "waist_cm": _to_numeric(df["BMXWAIST"]),
        "weight_kg": _to_numeric(df["BMXWT"]),
        "height_cm": _to_numeric(df["BMXHT"]),
    }


def _derive_bpxo(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "systolic_bp_mean": _row_mean(df, ["BPXOSY1", "BPXOSY2", "BPXOSY3"]),
        "diastolic_bp_mean": _row_mean(df, ["BPXODI1", "BPXODI2", "BPXODI3"]),
    }


def _derive_alq(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "alcohol_use_frequency_code": _clean_codes(df["ALQ121"], invalid={77, 99}),
        "alcohol_drinks_per_day": _clean_codes(df["ALQ130"], invalid={777, 999}),
    }


def _derive_diq(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {"diabetes_doctor_told_code": _clean_codes(df["DIQ010"], invalid={7, 9})}


def _derive_mcq(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {"asthma_doctor_told_code": _clean_codes(df["MCQ010"], invalid={7, 9})}


def _derive_paq(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {"physical_activity_sedentary_minutes": _clean_codes(df["PAD680"], invalid={7777, 9999})}


def _derive_smq(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "smoker_ever_code": _clean_codes(df["SMQ020"], invalid={7, 9}),
        "current_smoking_code": _clean_codes(df["SMQ040"], invalid={7, 9}),
    }


def _derive_dr1tot(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "diet_kcal_day1": _to_numeric(df["DR1TKCAL"]),
        "diet_sugar_g_day1": _to_numeric(df["DR1TSUGR"]),
        "diet_fiber_g_day1": _to_numeric(df["DR1TFIBE"]),
        "diet_total_fat_g_day1": _to_numeric(df["DR1TTFAT"]),
    }


def _add_negative_control_trap_variables(data: pd.DataFrame, rng: np.random.Generator) -> None:
    strata = _age_sex_strata(data)
    data["inflammatory_marker_index"] = _stratified_shuffle(data["hscrp_mg_l"], strata, rng)
    data["body_composition_index"] = _stratified_shuffle(data["bmi"], strata, rng)
    data["diet_quality_pattern_score"] = rng.normal(loc=0.0, scale=1.0, size=len(data))
    data["metabolic_panel_index"] = _stratified_shuffle(data["hba1c_percent"], strata, rng)


def _mark_negative_control_roles(roles: dict[str, dict[str, Any]]) -> None:
    for variable, source, parent in NEGATIVE_CONTROL_TRAP_SPECS:
        if variable not in roles:
            continue
        expected_behavior = "Named negative-control variable; should not be reported as a reliable finding."
        if source == "stratified_shuffle":
            expected_behavior = (
                "Negative control generated by shuffling within age/sex strata. It may retain "
                "some marginal stratum structure, but direct, independent, or causal claims remain unsupported."
            )
        roles[variable] = {
            "truth_status": "negative_control",
            "negative_control_type": source,
            "parent_true_variable": parent,
            "allowed_claim_type": "none",
            "oracle_adjustment_set": [],
            "expected_behavior": expected_behavior,
        }


def _mark_trap_variables(
    roles: dict[str, dict[str, Any]],
    variables: list[str],
    truth_status: str,
    expected_behavior: str,
    allowed_claim_type: str = "none",
    oracle_adjustment_set: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    for variable in variables:
        role = {
            "truth_status": truth_status,
            "allowed_claim_type": allowed_claim_type,
            "oracle_adjustment_set": list(oracle_adjustment_set or []),
            "expected_behavior": expected_behavior,
        }
        if extra:
            role.update(extra)
        roles[variable] = role


def _apply_true_effect(
    data: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    logit = (
        -0.85
        + 0.62 * _z(data["bmi"])
        + 0.48 * _z(data["hba1c_percent"])
        + 0.22 * _z(data["age_years"])
        - 0.12 * _indicator(data["sex_code"], 2)
    )
    data[OUTCOME] = _draw_binary(logit, rng)
    roles = _base_roles(data)
    _mark_direct(roles, "bmi", direction="positive", effect_size=0.62)
    _mark_direct(roles, "hba1c_percent", direction="positive", effect_size=0.48)
    _mark_proxy(roles, "waist_cm", parent="bmi")
    _mark_proxy(roles, "weight_kg", parent="bmi")
    return {
        "model": "logistic",
        "intercept": -0.85,
        "direct_true_variables": ["bmi", "hba1c_percent"],
        "background_covariates_in_formula": ["age_years", "sex_code"],
    }, roles


def _apply_pure_null(
    data: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    data[OUTCOME] = rng.binomial(1, 0.35, size=len(data))
    roles = _base_roles(data)
    for variable in data.columns:
        if variable not in {OUTCOME, "participant_id", "survey_weight_mec_2yr", "survey_strata", "survey_psu"}:
            roles[variable] = {
                "truth_status": "clean_null",
                "allowed_claim_type": "none",
                "expected_behavior": "No real association with the synthetic outcome beyond random chance.",
            }
    return {"model": "bernoulli_constant", "probability": 0.35, "direct_true_variables": []}, roles


def _apply_negative_control_trap(
    data: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _add_negative_control_trap_variables(data, rng)
    logit = (
        -0.82
        + 0.34 * _z(data["age_years"])
        + 0.38 * _z(data["bmi"])
        + 0.28 * _z(data["hba1c_percent"])
        - 0.10 * _indicator(data["sex_code"], 2)
    )
    data[OUTCOME] = _draw_binary(logit, rng)
    roles = _base_roles(data)
    _mark_direct(roles, "age_years", direction="positive", effect_size=0.34)
    _mark_direct(roles, "bmi", direction="positive", effect_size=0.38)
    _mark_direct(roles, "hba1c_percent", direction="positive", effect_size=0.28)
    _mark_negative_control_roles(roles)
    return {
        "model": "logistic_with_named_negative_control_variables",
        "direct_true_variables": ["age_years", "bmi", "hba1c_percent"],
        "negative_control_variables": [variable for variable, _, _ in NEGATIVE_CONTROL_TRAP_SPECS],
        "intended_behavior": (
            "Temptingly named analysis variables are present but do not directly enter the outcome DGP."
        ),
    }, roles


def _apply_confounding_trap(
    data: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    predicted_log_hscrp = _linear_projection(
        np.log1p(_imputed(data["hscrp_mg_l"])),
        data,
        ["age_years", "bmi", "sex_code"],
    )
    confounding_score = _z(predicted_log_hscrp)
    data["inflammatory_burden_score"] = confounding_score + rng.normal(0, 0.55, len(data))
    data["vascular_health_index"] = (
        0.62 * _z(data["age_years"])
        + 0.54 * _z(data["bmi"])
        - 0.18 * _indicator(data["sex_code"], 2)
        + rng.normal(0, 0.65, len(data))
    )
    data["adiposity_inflammation_score"] = (
        0.75 * _z(data["bmi"])
        + 0.35 * _z(data["age_years"])
        + 0.20 * _z(data["hscrp_mg_l"])
        + rng.normal(0, 0.60, len(data))
    )
    logit = (
        -0.95
        + 1.85 * confounding_score
        + 0.18 * _z(data["age_years"])
        + 0.14 * _z(data["bmi"])
        - 0.10 * _indicator(data["sex_code"], 2)
    )
    data[OUTCOME] = _draw_binary(logit, rng)
    roles = _base_roles(data)
    _mark_direct(roles, "age_years", direction="positive", effect_size=0.78)
    _mark_direct(roles, "bmi", direction="positive", effect_size=0.36)
    _mark_trap_variables(
        roles,
        CONFOUNDING_TRAP_VARIABLES,
        truth_status="confounding_trap",
        allowed_claim_type="predictive_association",
        oracle_adjustment_set=["age_years", "bmi", "sex_code"],
        expected_behavior=(
            "May look associated in naive models because it shares age/BMI/sex structure; "
            "unsupported as an independent effect after oracle adjustment."
        ),
    )
    return {
        "model": "logistic",
        "intercept": -0.95,
        "direct_true_variables": ["age_years", "bmi"],
        "trap_exposure": "hscrp_mg_l",
        "trap_exposures": CONFOUNDING_TRAP_VARIABLES,
        "oracle_adjustment_set": ["age_years", "bmi", "sex_code"],
        "confounding_score": "linear projection of log hscrp onto age_years, bmi, and sex_code",
    }, roles


def _apply_collider_post_treatment_trap(
    data: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    logit = -0.9 + 0.66 * _z(data["hba1c_percent"]) + 0.34 * _z(data["age_years"])
    data[OUTCOME] = _draw_binary(logit, rng)
    data["healthcare_engagement_score"] = (
        0.85 * _z(data["bmi"]) + 0.92 * data[OUTCOME].to_numpy() + rng.normal(0, 1, len(data))
    )
    data["clinical_biomarker_index"] = (
        0.55 * _z(data["hba1c_percent"])
        + 0.72 * data[OUTCOME].to_numpy()
        + rng.normal(0, 1, len(data))
    )
    data["treatment_intensity_score"] = (
        0.36 * _z(data["age_years"])
        + 0.88 * data[OUTCOME].to_numpy()
        + rng.normal(0, 0.85, len(data))
    )
    data["followup_health_status_score"] = (
        0.34 * _z(data["bmi"])
        + 0.46 * _z(data["hba1c_percent"])
        + 0.95 * data[OUTCOME].to_numpy()
        + rng.normal(0, 0.90, len(data))
    )
    roles = _base_roles(data)
    _mark_direct(roles, "hba1c_percent", direction="positive", effect_size=0.66)
    _mark_direct(roles, "age_years", direction="positive", effect_size=0.34)
    _mark_trap_variables(
        roles,
        COLLIDER_TRAP_VARIABLES,
        truth_status="collider_trap",
        oracle_adjustment_set=["hba1c_percent", "age_years"],
        expected_behavior=(
            "Generated downstream of baseline risk structure and the outcome; treating it as a baseline "
            "cause or adjusting for it in total-effect analyses is unsupported."
        ),
        extra={"trap_subtype": "post_treatment_or_collider"},
    )
    return {
        "model": "logistic_plus_generated_trap_variables",
        "direct_true_variables": ["hba1c_percent", "age_years"],
        "collider_variable": "healthcare_engagement_score",
        "post_treatment_variable": "clinical_biomarker_index",
        "collider_variables": COLLIDER_TRAP_VARIABLES,
        "trap_variables": COLLIDER_TRAP_VARIABLES,
    }, roles


def _apply_leakage_trap(
    data: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    logit = -0.9 + 0.52 * _z(data["bmi"]) + 0.42 * _z(data["age_years"])
    data[OUTCOME] = _draw_binary(logit, rng)
    data["laboratory_risk_index"] = (
        1.25 * data[OUTCOME].to_numpy() + 0.18 * _z(data["age_years"]) + rng.normal(0, 0.45, len(data))
    )
    leak_prob = _sigmoid(-1.25 + 2.2 * data[OUTCOME].to_numpy() + 0.15 * _z(data["bmi"]))
    data["care_pathway_index"] = leak_prob + rng.normal(0, 0.18, len(data))
    data["medication_profile_score"] = (
        0.92 * data[OUTCOME].to_numpy()
        + 0.24 * _z(data["age_years"])
        + rng.normal(0, 0.38, len(data))
    )
    review_prob = _sigmoid(-1.10 + 2.45 * data[OUTCOME].to_numpy() + 0.22 * _z(data["hba1c_percent"]))
    data["clinical_review_score"] = review_prob + rng.normal(0, 0.16, len(data))
    roles = _base_roles(data)
    _mark_direct(roles, "bmi", direction="positive", effect_size=0.52)
    _mark_direct(roles, "age_years", direction="positive", effect_size=0.42)
    for variable in LEAKAGE_TRAP_VARIABLES:
        roles[variable] = {
            "truth_status": "leakage",
            "allowed_claim_type": "none",
            "oracle_adjustment_set": [],
            "expected_behavior": "Predicts the synthetic outcome because it was generated using outcome information.",
        }
    return {
        "model": "logistic_then_outcome_derived_leakage",
        "direct_true_variables": ["bmi", "age_years"],
        "leakage_variables": LEAKAGE_TRAP_VARIABLES,
    }, roles


def _apply_survey_weight_trap(
    data: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    hdl = _z(data["hdl_cholesterol_mg_dl"])
    weights = _positive_weights(data["survey_weight_mec_2yr"])
    weight_z = _z(pd.Series(np.log1p(weights)))
    data["lipid_balance_index"] = (
        0.78 * hdl
        - 0.24 * _z(data["total_cholesterol_mg_dl"])
        + rng.normal(0, 0.58, len(data))
    )
    data["nutrition_lipid_score"] = (
        0.54 * hdl
        + 0.24 * _z(data["diet_fiber_g_day1"])
        - 0.22 * _z(data["diet_total_fat_g_day1"])
        + rng.normal(0, 0.68, len(data))
    )
    data["metabolic_screening_index"] = (
        0.56 * hdl
        + 0.22 * _z(data["hba1c_percent"])
        - 0.20 * weight_z
        + rng.normal(0, 0.62, len(data))
    )
    trap_signal = np.mean(
        np.column_stack([_z(data[variable]) for variable in SURVEY_WEIGHT_TRAP_VARIABLES]),
        axis=1,
    )
    low_weight_sample = weight_z < 0
    trap_slope = np.where(low_weight_sample, 2.20, -0.55)
    logit = -0.72 + trap_slope * trap_signal + 0.18 * weight_z + 0.18 * _z(data["age_years"])
    data[OUTCOME] = _draw_binary(logit, rng)
    roles = _base_roles(data)
    _mark_trap_variables(
        roles,
        SURVEY_WEIGHT_TRAP_VARIABLES,
        truth_status="survey_weight_trap",
        allowed_claim_type="direct_effect",
        oracle_adjustment_set=["survey_weight_mec_2yr", "survey_strata", "survey_psu", "age_years"],
        expected_behavior=(
            "Naive unweighted and survey-weighted analyses can disagree; claims should use survey design."
        ),
    )
    _mark_direct(roles, "age_years", direction="positive", effect_size=0.22)
    return {
        "model": "logistic_with_unweighted_sample_signal_and_weighted_attenuation",
        "trap_exposure": "hdl_cholesterol_mg_dl",
        "trap_exposures": SURVEY_WEIGHT_TRAP_VARIABLES,
        "unweighted_correlation": _correlation(data["hdl_cholesterol_mg_dl"], data[OUTCOME]),
        "weighted_correlation": _correlation(data["hdl_cholesterol_mg_dl"], data[OUTCOME], weights=weights),
        "unweighted_correlations": {
            variable: _correlation(data[variable], data[OUTCOME]) for variable in SURVEY_WEIGHT_TRAP_VARIABLES
        },
        "weighted_correlations": {
            variable: _correlation(data[variable], data[OUTCOME], weights=weights)
            for variable in SURVEY_WEIGHT_TRAP_VARIABLES
        },
        "survey_design_variables": ["survey_weight_mec_2yr", "survey_strata", "survey_psu"],
        "intended_behavior": (
            "Four candidate predictors have unweighted sample associations that attenuate or reverse under survey weights."
        ),
    }, roles


def _base_roles(data: pd.DataFrame) -> dict[str, dict[str, Any]]:
    roles: dict[str, dict[str, Any]] = {}
    for variable in data.columns:
        roles[variable] = {
            "truth_status": "background_covariate",
            "allowed_claim_type": "predictive_association",
            "expected_behavior": "Observed NHANES covariate; no direct DGP role unless overridden.",
        }
    roles["participant_id"] = {
        "truth_status": "identifier",
        "allowed_claim_type": "none",
        "expected_behavior": "Identifier only.",
    }
    roles[OUTCOME] = {
        "truth_status": "synthetic_outcome",
        "allowed_claim_type": "outcome",
        "expected_behavior": "Synthetic binary outcome generated by the hidden DGP.",
    }
    for variable in ["survey_weight_mec_2yr", "survey_strata", "survey_psu"]:
        if variable in roles:
            roles[variable] = {
                "truth_status": "survey_design",
                "allowed_claim_type": "none",
                "expected_behavior": "Survey design variable, not a substantive exposure.",
            }
    _mark_negative_control_roles(roles)
    return roles


def _mark_direct(
    roles: dict[str, dict[str, Any]], variable: str, direction: str, effect_size: float
) -> None:
    roles[variable] = {
        "truth_status": "direct_true",
        "allowed_claim_type": "direct_effect",
        "direction": direction,
        "effect_size_logit": effect_size,
        "oracle_adjustment_set": [],
        "expected_behavior": "Directly enters the synthetic outcome data-generating process.",
    }


def _mark_proxy(roles: dict[str, dict[str, Any]], variable: str, parent: str) -> None:
    roles[variable] = {
        "truth_status": "proxy_of_true",
        "parent_true_variable": parent,
        "allowed_claim_type": "predictive_association",
        "oracle_adjustment_set": [parent],
        "expected_behavior": (
            "May predict the outcome as a proxy, but independent/direct/causal claims are unsupported."
        ),
    }


def _public_codebook_payload(spec: DatasetSpec, suite_name: str, data: pd.DataFrame) -> dict[str, Any]:
    variables = []
    merged = {**BASE_VARIABLE_CODEBOOK, **GENERATED_PUBLIC_CODEBOOK}
    for variable in data.columns:
        info = merged.get(
            variable,
            {
                "description": "Study-derived analysis variable.",
                "source": "Study-derived field.",
                "public_role": "candidate_predictor",
            },
        )
        variables.append(_public_variable_entry(variable, info))
    return {
        "schema_version": "0.1",
        "data_file": f"{PUBLIC_DATASET_ID}.csv",
        "dataset_id": PUBLIC_DATASET_ID,
        "dataset_type": PUBLIC_DATASET_TYPE,
        "title": "Cardiometabolic Risk Study Data Documentation, Codebook, and Frequencies",
        "description": "De-identified participant-level cardiometabolic risk analysis dataset.",
        "agent_visible": True,
        "notes": [
            "This file contains examination, laboratory, questionnaire, dietary, and study-derived variables.",
            "The primary endpoint variable is listed in the outcome field.",
        ],
        "outcome": OUTCOME,
        "n_rows": len(data),
        "n_columns": len(data.columns),
        "variables": variables,
    }


def _public_variable_entry(variable: str, info: dict[str, str]) -> dict[str, Any]:
    description = info.get("description", "Study analysis variable.")
    source_file = info.get("source_file", info.get("source", "Study analysis file."))
    entry: dict[str, Any] = {
        "name": variable,
        "description": description,
        "source_file": source_file,
    }
    value_labels = info.get("value_labels")
    if value_labels:
        entry["value_labels"] = value_labels
    notes = info.get("notes")
    if notes:
        entry["notes"] = notes
    return entry


def _hidden_truth_payload(
    spec: DatasetSpec,
    suite_name: str,
    seed: int,
    data: pd.DataFrame,
    dgp_details: dict[str, Any],
    roles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    variable_truth = []
    for variable in data.columns:
        role = dict(roles.get(variable, {}))
        role.setdefault("truth_status", "background_covariate")
        role.setdefault("allowed_claim_type", "predictive_association")
        role.setdefault("parent_true_variable", "")
        role.setdefault("oracle_adjustment_set", [])
        role.setdefault("expected_behavior", "")
        variable_truth.append({"variable_name": variable, **role})
    return {
        "schema_version": "0.1",
        "suite": suite_name,
        "dataset_id": spec.dataset_id,
        "dataset_type": spec.kind,
        "agent_visible": False,
        "seed": seed,
        "outcome": OUTCOME,
        "dgp": dgp_details,
        "diagnostics": {
            "n_rows": len(data),
            "n_columns": len(data.columns),
            "outcome_rate": float(data[OUTCOME].mean()),
        },
        "variable_truth": variable_truth,
    }


def _ordered_columns(data: pd.DataFrame) -> list[str]:
    first = ["participant_id", OUTCOME]
    remaining = [column for column in data.columns if column not in first]
    return first + remaining


def _read_xpt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing NHANES XPT file: {path}")
    data = pd.read_sas(path, format="xport")
    for column in data.select_dtypes(include=["number"]).columns:
        data[column] = _to_numeric(data[column])
    return data


def _to_numeric(series: pd.Series | np.ndarray) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    result = pd.Series(result).mask(pd.Series(result).abs() < 1e-40, 0)
    return result


def _clean_codes(series: pd.Series, invalid: set[int]) -> pd.Series:
    result = _to_numeric(series)
    return result.mask(result.isin(invalid))


def _row_mean(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    return data[columns].apply(_to_numeric).mean(axis=1, skipna=True)


def _imputed(series: pd.Series | np.ndarray) -> np.ndarray:
    values = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).all():
        return np.zeros(len(values), dtype=float)
    median = float(np.nanmedian(values))
    return np.where(np.isnan(values), median, values)


def _z(series: pd.Series | np.ndarray) -> np.ndarray:
    values = _imputed(series)
    sd = float(np.nanstd(values))
    if sd == 0 or not np.isfinite(sd):
        return np.zeros(len(values), dtype=float)
    return (values - float(np.nanmean(values))) / sd


def _indicator(series: pd.Series | np.ndarray, value: int | float) -> np.ndarray:
    values = _imputed(series)
    return (values == value).astype(float)


def _linear_projection(
    target: pd.Series | np.ndarray,
    data: pd.DataFrame,
    covariates: list[str],
) -> np.ndarray:
    design_columns = [np.ones(len(data))]
    for covariate in covariates:
        design_columns.append(_z(data[covariate]))
    design = np.column_stack(design_columns)
    y = _imputed(target)
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return design @ beta


def _draw_binary(logit: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.binomial(1, _sigmoid(logit))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def _shuffle_series(series: pd.Series, rng: np.random.Generator) -> np.ndarray:
    values = pd.Series(series).to_numpy(copy=True)
    rng.shuffle(values)
    return values


def _stratified_shuffle(
    series: pd.Series, strata: pd.Series, rng: np.random.Generator
) -> np.ndarray:
    result = pd.Series(index=series.index, dtype=float)
    for _, index in strata.groupby(strata).groups.items():
        values = series.loc[index].to_numpy(copy=True)
        rng.shuffle(values)
        result.loc[index] = values
    return result.to_numpy()


def _age_sex_strata(data: pd.DataFrame) -> pd.Series:
    ages = _imputed(data["age_years"])
    age_band = pd.Series(
        pd.cut(
            ages,
            bins=[-1, 17, 39, 59, 120],
            labels=["child", "young_adult", "midlife", "older"],
        ),
        index=data.index,
    ).astype(str)
    sex = data["sex_code"].fillna(-1).astype(int).astype(str)
    return age_band + "_sex_" + sex


def _positive_weights(series: pd.Series) -> np.ndarray:
    values = _imputed(series)
    positive = values[values > 0]
    floor = float(np.nanmedian(positive)) if len(positive) else 1.0
    return np.where(values > 0, values, floor)


def _correlation(
    x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, weights: np.ndarray | None = None
) -> float:
    x_values = _imputed(x)
    y_values = _imputed(y)
    if weights is None:
        if np.std(x_values) == 0 or np.std(y_values) == 0:
            return 0.0
        return float(np.corrcoef(x_values, y_values)[0, 1])
    w = np.asarray(weights, dtype=float)
    w = np.where(w > 0, w, np.nanmedian(w[w > 0]))
    w = w / np.sum(w)
    x_centered = x_values - np.sum(w * x_values)
    y_centered = y_values - np.sum(w * y_values)
    denom = np.sqrt(np.sum(w * x_centered**2) * np.sum(w * y_centered**2))
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    return float(np.sum(w * x_centered * y_centered) / denom)


def _write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)
