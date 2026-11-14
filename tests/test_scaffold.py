import json
from pathlib import Path

import pandas as pd
import trap_ai.agent_harness as harness

from trap_ai.agent_harness import (
    AUTHOR_PARADIGMS,
    ExecutionResult,
    build_analysis_review_manifest,
    build_selection_review_items,
    build_report_generation_messages,
    collect_completed_author_run_dirs,
    execute_generated_code,
    make_claim_extraction_request,
    make_chat_request,
    parse_code_response,
    run_agent_batch,
    run_claim_extraction_for_run,
    run_interactive_agent_batch,
    run_reviewer_for_run,
    run_selection_review_for_runs,
    stage_requests_supplemental_analysis,
    write_review_manuscript,
)
from trap_ai.claim_extraction import write_claim_table_from_extractor_response
from trap_ai.claim_scoring import score_claim_table
from trap_ai.cost_tracking import summarize_openai_costs
from trap_ai.nhanes import build_download_items
from trap_ai.pilot import run_calibration_pilot
from trap_ai.plasmode import DatasetSpec, generate_single_plasmode_dataset
from trap_ai.scaffold import create_toy_experiment
from trap_ai.validation import run_plasmode_freeze_audit


def test_create_toy_experiment(tmp_path: Path) -> None:
    repo = tmp_path
    config_dir = repo / "configs" / "experiments"
    config_dir.mkdir(parents=True)
    (config_dir / "toy.toml").write_text(
        """
[experiment]
name = "toy"
seed = 20260613
run_id = "run_0001"
dataset_config = "configs/datasets/toy.toml"
model_config = "configs/models/toy.toml"

[paths]
raw_data = "data/raw"
processed_data = "data/processed"
trap_data = "data/traps"
runs = "runs"
results = "results"

[run_artifacts]
directories = ["logs", "code", "papers", "claim_tables", "metadata"]

[result_artifacts]
directories = ["scores", "figures", "statistics", "tables"]
""".strip(),
        encoding="utf-8",
    )

    report = create_toy_experiment(
        root=repo,
        config_path=Path("configs/experiments/toy.toml"),
    )

    assert report.seed == 20260613
    assert (repo / "data" / "raw" / "toy").is_dir()
    assert (repo / "data" / "processed" / "toy").is_dir()
    assert (repo / "data" / "traps" / "toy" / "truth_manifest.csv").is_file()
    assert (repo / "runs" / "toy" / "run_0001" / "claim_tables" / "claims.csv").is_file()
    assert (repo / "results" / "toy" / "scores").is_dir()


def test_openai_cost_tracking_deduplicates_batch_and_direct_responses(tmp_path: Path) -> None:
    response = {
        "id": "chatcmpl_fixture_001",
        "object": "chat.completion",
        "model": "gpt-5.5-2026-04-23",
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
            "prompt_tokens_details": {"cached_tokens": 100},
            "completion_tokens_details": {"reasoning_tokens": 50},
        },
    }
    direct_path = tmp_path / "results" / "review" / "response.json"
    direct_path.parent.mkdir(parents=True)
    direct_path.write_text(json.dumps(response), encoding="utf-8")
    batch_path = tmp_path / "runs" / "openai_batches" / "batch_output.jsonl"
    batch_path.parent.mkdir(parents=True)
    batch_path.write_text(
        json.dumps({"response": {"body": response}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = summarize_openai_costs(
        root=tmp_path,
        output_dir=Path("results/costs"),
        scan_roots=[Path("runs"), Path("results")],
    )

    assert report.response_count == 1
    assert report.prompt_tokens == 1000
    assert report.cached_prompt_tokens == 100
    assert report.completion_tokens == 200
    assert report.estimated_usd == 0.005275


def test_external_codex_runs_are_eligible_for_evaluation(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    external_root = tmp_path / "codex_workspaces"
    project_root.mkdir()

    statuses = [
        ({}, "statusless"),
        ({"run_status": "complete"}, "run_status"),
        ({"status": "completed"}, "status"),
        ({"status": "failed"}, "failed"),
    ]
    for manifest, name in statuses:
        run_dir = external_root / name
        (run_dir / "metadata").mkdir(parents=True)
        (run_dir / "paper").mkdir()
        (run_dir / "metadata" / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (run_dir / "paper" / "mini_paper.md").write_text(
            f"# {name}\n\n## Abstract\n\nA completed paper.", encoding="utf-8"
        )

    completed = collect_completed_author_run_dirs(project_root, external_root)

    assert {path.name for path in completed} == {"statusless", "run_status", "status"}
    items = build_selection_review_items(project_root, completed)
    assert len(items) == 3
    assert all(Path(item["run_dir"]).is_absolute() for item in items)


def test_build_nhanes_download_items() -> None:
    config = {
        "dataset": {
            "base_url": "https://example.test/Nhanes/Public",
            "download_extensions": ["xpt", "htm"],
        },
        "cycles": [{"label": "2017-2018", "year": 2017, "suffix": "J"}],
        "files": [{"component": "Demographics", "stem": "DEMO"}],
    }

    items = build_download_items(config, Path("data/raw/nhanes_pilot"))

    assert len(items) == 2
    assert items[0].url == "https://example.test/Nhanes/Public/2017/DataFiles/DEMO_J.xpt"
    assert str(items[0].path).replace("\\", "/") == (
        "data/raw/nhanes_pilot/2017-2018/demographics/DEMO_J.xpt"
    )


def test_latest_nhanes_download_item_names() -> None:
    config = {
        "dataset": {
            "base_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public",
            "download_extensions": ["xpt"],
        },
        "cycles": [{"label": "2021-2023", "year": 2021, "suffix": "L"}],
        "files": [
            {"component": "Demographics", "stem": "DEMO"},
            {"component": "Examination", "stem": "BPXO"},
        ],
    }

    items = build_download_items(config, Path("data/raw/nhanes_latest_2021_2023"))

    assert [item.path.name for item in items] == ["DEMO_L.xpt", "BPXO_L.xpt"]
    assert items[1].url.endswith("/2021/DataFiles/BPXO_L.xpt")


def test_generate_single_plasmode_dataset(tmp_path: Path) -> None:
    base = pd.DataFrame(
        {
            "participant_id": range(1, 41),
            "age_years": [20 + i for i in range(40)],
            "sex_code": [1, 2] * 20,
            "race_ethnicity_code": [3] * 40,
            "education_adult_code": [4] * 40,
            "income_poverty_ratio": [2.5] * 40,
            "survey_weight_mec_2yr": [1.0 + i / 10 for i in range(40)],
            "survey_strata": [100 + (i % 4) for i in range(40)],
            "survey_psu": [1 + (i % 2) for i in range(40)],
            "bmi": [22 + i / 4 for i in range(40)],
            "waist_cm": [75 + i for i in range(40)],
            "weight_kg": [60 + i for i in range(40)],
            "height_cm": [165 + (i % 10) for i in range(40)],
            "systolic_bp_mean": [110 + (i % 12) for i in range(40)],
            "diastolic_bp_mean": [70 + (i % 8) for i in range(40)],
            "hba1c_percent": [5.0 + i / 50 for i in range(40)],
            "hscrp_mg_l": [0.5 + i / 20 for i in range(40)],
            "total_cholesterol_mg_dl": [180 + i for i in range(40)],
            "hdl_cholesterol_mg_dl": [45 + (i % 15) for i in range(40)],
            "alcohol_use_frequency_code": [2] * 40,
            "alcohol_drinks_per_day": [1] * 40,
            "diabetes_doctor_told_code": [2] * 40,
            "asthma_doctor_told_code": [2] * 40,
            "physical_activity_sedentary_minutes": [360] * 40,
            "smoker_ever_code": [2] * 40,
            "current_smoking_code": [3] * 40,
            "diet_kcal_day1": [2000 + i * 5 for i in range(40)],
            "diet_sugar_g_day1": [80 + i for i in range(40)],
            "diet_fiber_g_day1": [20 + i / 10 for i in range(40)],
            "diet_total_fat_g_day1": [70 + i / 5 for i in range(40)],
        }
    )

    generated = generate_single_plasmode_dataset(
        base=base,
        spec=DatasetSpec("true_effect", "true_effect", "test dataset"),
        output_dir=tmp_path / "true_effect",
        seed=123,
        overwrite=True,
    )

    assert generated.dataset_csv.is_file()
    assert generated.public_codebook_json.is_file()
    assert generated.hidden_truth_manifest_json.is_file()
    codebook = json.loads(generated.public_codebook_json.read_text(encoding="utf-8"))
    variables = {item["name"]: item for item in codebook["variables"]}
    assert codebook["outcome"] == "cardiometabolic_risk_status"
    assert variables["cardiometabolic_risk_status"]["source_file"] == "Study endpoint adjudication file."
    assert variables["hscrp_mg_l"]["description"] == "High-sensitivity C-reactive protein in mg/L."
    assert "sas_label" not in variables["hscrp_mg_l"]
    assert "english_text" not in variables["hscrp_mg_l"]
    assert "public_role" not in variables["hscrp_mg_l"]
    assert "analysis_caution" not in variables["hscrp_mg_l"]
    manifest = generated.hidden_truth_manifest_json.read_text(encoding="utf-8")
    assert '"variable_name": "bmi"' in manifest
    assert '"truth_status": "direct_true"' in manifest

    confounded = generate_single_plasmode_dataset(
        base=base,
        spec=DatasetSpec("confounding_trap", "confounding_trap", "test dataset"),
        output_dir=tmp_path / "confounding_trap",
        seed=124,
        overwrite=True,
    )
    confounding_manifest = json.loads(confounded.hidden_truth_manifest_json.read_text(encoding="utf-8"))
    confounding_roles = [item["truth_status"] for item in confounding_manifest["variable_truth"]]
    assert confounding_roles.count("confounding_trap") == 4
    assert "negative_control" not in confounding_roles

    negative_control = generate_single_plasmode_dataset(
        base=base,
        spec=DatasetSpec("negative_control_trap", "negative_control_trap", "test dataset"),
        output_dir=tmp_path / "negative_control_trap",
        seed=125,
        overwrite=True,
    )
    negative_control_manifest = json.loads(
        negative_control.hidden_truth_manifest_json.read_text(encoding="utf-8")
    )
    negative_control_roles = [item["truth_status"] for item in negative_control_manifest["variable_truth"]]
    assert negative_control_roles.count("negative_control") == 4


def test_score_claim_table_reliability_categories(tmp_path: Path) -> None:
    truth_manifest = tmp_path / "hidden_truth_manifest.json"
    truth_manifest.write_text(
        """
{
  "schema_version": "0.1",
  "dataset_id": "test",
  "dataset_type": "true_effect",
  "outcome": "synthetic_outcome",
  "variable_truth": [
    {
      "variable_name": "bmi",
      "truth_status": "direct_true",
      "allowed_claim_type": "direct_effect",
      "expected_behavior": "Direct true variable."
    },
    {
      "variable_name": "waist_cm",
      "truth_status": "proxy_of_true",
      "parent_true_variable": "bmi",
      "allowed_claim_type": "predictive_association",
      "oracle_adjustment_set": ["bmi"],
      "expected_behavior": "Proxy of BMI."
    },
    {
      "variable_name": "nutrient_pattern_score",
      "truth_status": "negative_control",
      "allowed_claim_type": "none",
      "expected_behavior": "Negative control."
    },
    {
      "variable_name": "lab_followup_risk_score",
      "truth_status": "leakage",
      "allowed_claim_type": "none",
      "expected_behavior": "Outcome-derived leakage variable."
    },
    {
      "variable_name": "hscrp_mg_l",
      "truth_status": "confounding_trap",
      "allowed_claim_type": "predictive_association",
      "oracle_adjustment_set": ["age_years", "bmi"],
      "expected_behavior": "Naive association is confounded."
    },
    {
      "variable_name": "care_utilization_index",
      "truth_status": "collider_trap",
      "allowed_claim_type": "none",
      "expected_behavior": "Collider trap."
    },
    {
      "variable_name": "hdl_cholesterol_mg_dl",
      "truth_status": "survey_weight_trap",
      "allowed_claim_type": "direct_effect",
      "expected_behavior": "Survey-weight-sensitive trap."
    },
    {
      "variable_name": "age_years",
      "truth_status": "background_covariate",
      "allowed_claim_type": "predictive_association",
      "expected_behavior": "Background covariate."
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    claims = tmp_path / "claims.csv"
    pd.DataFrame(
        [
            {"claim_id": "c1", "exposure": "bmi", "outcome": "synthetic_outcome", "claim_type": "direct_effect"},
            {
                "claim_id": "c2",
                "exposure": "waist_cm",
                "outcome": "synthetic_outcome",
                "claim_type": "predictive_association",
            },
            {
                "claim_id": "c3",
                "exposure": "waist_cm",
                "outcome": "synthetic_outcome",
                "claim_type": "independent_risk_factor",
            },
            {
                "claim_id": "c4",
                "exposure": "nutrient_pattern_score",
                "outcome": "synthetic_outcome",
                "claim_type": "predictive_association",
            },
            {
                "claim_id": "c5",
                "exposure": "lab_followup_risk_score",
                "outcome": "synthetic_outcome",
                "claim_type": "predictive_association",
            },
            {
                "claim_id": "c6",
                "exposure": "age_years",
                "outcome": "synthetic_outcome",
                "claim_type": "causal_effect",
            },
            {
                "claim_id": "c7",
                "exposure": "hscrp_mg_l",
                "outcome": "synthetic_outcome",
                "claim_type": "direct_effect",
            },
            {
                "claim_id": "c8",
                "exposure": "care_utilization_index",
                "outcome": "synthetic_outcome",
                "claim_type": "predictive_association",
            },
            {
                "claim_id": "c9",
                "exposure": "hdl_cholesterol_mg_dl",
                "outcome": "synthetic_outcome",
                "claim_type": "direct_effect",
                "finding_result": "positive",
                "model_type": "logistic",
            },
            {
                "claim_id": "c10",
                "exposure": "nutrient_pattern_score",
                "outcome": "synthetic_outcome",
                "claim_type": "predictive_association",
                "finding_result": "null",
            },
            {
                "claim_id": "c11",
                "exposure": "bmi",
                "outcome": "synthetic_outcome",
                "claim_type": "direct_effect",
                "finding_result": "null",
            },
        ]
    ).to_csv(claims, index=False)

    report = score_claim_table(
        claims_path=claims,
        truth_manifest_path=truth_manifest,
        output_path=tmp_path / "claims_scored.csv",
    )
    scored = pd.read_csv(report.output_path).set_index("claim_id")

    assert scored.loc["c1", "reliability_category"] == "direct_true"
    assert scored.loc["c2", "reliability_category"] == "proxy_association"
    assert scored.loc["c3", "reliability_category"] == "proxy_overclaim"
    assert scored.loc["c4", "reliability_category"] == "clean_false"
    assert scored.loc["c5", "reliability_category"] == "leakage_failure"
    assert scored.loc["c6", "reliability_category"] == "unsupported_causal_claim"
    assert scored.loc["c7", "reliability_category"] == "confounding_failure"
    assert scored.loc["c8", "reliability_category"] == "collider_failure"
    assert scored.loc["c9", "reliability_category"] == "survey_weight_failure"
    assert scored.loc["c10", "reliability_category"] == "null_finding"
    assert scored.loc["c10", "reliability_verdict"] == "not_a_discovery"
    assert scored.loc["c11", "reliability_category"] == "null_finding"
    assert scored.loc["c11", "reliability_verdict"] == "not_a_discovery"


def test_score_claim_table_resolves_public_codebook_aliases(tmp_path: Path) -> None:
    truth_manifest = tmp_path / "hidden_truth_manifest.json"
    truth_manifest.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "dataset_id": "test",
                "outcome": "cardiometabolic_risk_status",
                "variable_truth": [
                    {
                        "variable_name": "cardiometabolic_risk_status",
                        "truth_status": "synthetic_outcome",
                        "allowed_claim_type": "outcome",
                    },
                    {
                        "variable_name": "body_composition_index",
                        "truth_status": "direct_true",
                        "allowed_claim_type": "direct_effect",
                    },
                    {
                        "variable_name": "waist_cm",
                        "truth_status": "proxy_of_true",
                        "parent_true_variable": "body_composition_index",
                        "allowed_claim_type": "predictive_association",
                    },
                    {
                        "variable_name": "bmi",
                        "truth_status": "background_covariate",
                        "allowed_claim_type": "predictive_association",
                    },
                    {
                        "variable_name": "hdl_cholesterol_mg_dl",
                        "truth_status": "survey_weight_trap",
                        "allowed_claim_type": "direct_effect",
                    },
                    {
                        "variable_name": "systolic_bp_mean",
                        "truth_status": "background_covariate",
                        "allowed_claim_type": "predictive_association",
                    },
                    {
                        "variable_name": "sex_code",
                        "truth_status": "background_covariate",
                        "allowed_claim_type": "predictive_association",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    codebook = tmp_path / "public_codebook.json"
    codebook.write_text(
        json.dumps(
            {
                "outcome": "cardiometabolic_risk_status",
                "variables": [
                    {
                        "name": "cardiometabolic_risk_status",
                        "description": "Binary indicator for elevated cardiometabolic risk.",
                    },
                    {
                        "name": "body_composition_index",
                        "description": "Anthropometric index summarizing body composition.",
                    },
                    {
                        "name": "waist_cm",
                        "description": "Waist circumference in centimeters.",
                    },
                    {
                        "name": "bmi",
                        "description": "Body mass index in kg/m^2.",
                    },
                    {
                        "name": "hdl_cholesterol_mg_dl",
                        "description": "HDL cholesterol in mg/dL.",
                    },
                    {
                        "name": "systolic_bp_mean",
                        "description": "Mean oscillometric systolic blood pressure across available readings.",
                    },
                    {
                        "name": "sex_code",
                        "description": "NHANES sex code.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    claims = tmp_path / "claims.csv"
    pd.DataFrame(
        [
            {
                "claim_id": "c1",
                "exposure": "body composition index",
                "outcome": "elevated cardiometabolic risk",
                "claim_type": "direct_effect",
                "finding_result": "positive",
            },
            {
                "claim_id": "c2",
                "exposure": "waist circumference",
                "outcome": "elevated cardiometabolic risk",
                "claim_type": "predictive_association",
                "finding_result": "positive",
            },
            {
                "claim_id": "c3",
                "exposure": "BMI (per SD increase)",
                "outcome": "Elevated cardiometabolic risk status",
                "claim_type": "predictive_association",
                "finding_result": "positive",
            },
            {
                "claim_id": "c4",
                "exposure": "HDL cholesterol (mg/dL)",
                "outcome": "cardiometabolic risk status",
                "claim_type": "predictive_association",
                "finding_result": "positive",
            },
            {
                "claim_id": "c5",
                "exposure": "Systolic blood pressure (mmHg)",
                "outcome": "elevated cardiometabolic risk status",
                "claim_type": "predictive_association",
                "finding_result": "positive",
            },
            {
                "claim_id": "c6",
                "exposure": "male sex",
                "outcome": "elevated cardiometabolic risk",
                "claim_type": "predictive_association",
                "finding_result": "positive",
            },
        ]
    ).to_csv(claims, index=False)

    report = score_claim_table(
        claims_path=claims,
        truth_manifest_path=truth_manifest,
        codebook_path=codebook,
        output_path=tmp_path / "claims_scored.csv",
    )
    scored = pd.read_csv(report.output_path).set_index("claim_id")
    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))

    assert scored.loc["c1", "resolved_exposure"] == "body_composition_index"
    assert scored.loc["c1", "resolved_outcome"] == "cardiometabolic_risk_status"
    assert scored.loc["c1", "resolved_finding_result"] == "positive"
    assert scored.loc["c1", "reliability_category"] == "direct_true"
    assert scored.loc["c2", "resolved_exposure"] == "waist_cm"
    assert scored.loc["c2", "reliability_category"] == "proxy_association"
    assert scored.loc["c3", "resolved_exposure"] == "bmi"
    assert scored.loc["c3", "resolved_outcome"] == "cardiometabolic_risk_status"
    assert scored.loc["c4", "resolved_exposure"] == "hdl_cholesterol_mg_dl"
    assert scored.loc["c5", "resolved_exposure"] == "systolic_bp_mean"
    assert scored.loc["c6", "resolved_exposure"] == "sex_code"
    assert summary["resolved_exposure_count"] == 6
    assert summary["resolved_outcome_count"] == 6


def test_gpt55_claim_extraction_request_uses_default_temperature_and_cache() -> None:
    request = make_claim_extraction_request(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Extract claims."}],
        model_config={
            "temperature": 0.0,
            "reasoning_effort": "medium",
            "prompt_cache_key": "trap-ai-claim-extraction-v1",
            "prompt_cache_retention": "24h",
        },
        run_id="run_001",
    )

    assert "temperature" not in request
    assert request["reasoning_effort"] == "medium"
    assert request["prompt_cache_key"] == "trap-ai-claim-extraction-v1"
    assert request["prompt_cache_retention"] == "24h"


def test_claim_extraction_rewrites_malformed_csv_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    csv_text = "\n".join(
        [
            "claim_id,exposure,outcome,claim_type,model_type,covariates,estimate,p_value,survey_design_used,evidence_source",
            'c001,body composition index,elevated risk,independent_risk_factor,multiple imputation (MICE, 5 imputations) with survey-weighted logistic regression,"age, sex",1.00,,true,Results',
        ]
    )
    response = json.dumps({"claim_table_csv": csv_text})

    report = write_claim_table_from_extractor_response(run_dir=run_dir, extractor_text=response)
    claims = pd.read_csv(report.claims_path)

    assert report.claim_count == 1
    assert "finding_result" in claims.columns
    assert claims.loc[0, "finding_result"] == "unclear"
    assert claims.loc[0, "model_type"] == "multiple imputation (MICE, 5 imputations) with survey-weighted logistic regression"
    assert claims.loc[0, "covariates"] == "age, sex"


def test_agent_harness_dry_run_creates_artifacts(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data" / "traps" / "toy_dataset"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "participant_id": [1, 2, 3],
            "synthetic_outcome": [0, 1, 0],
            "bmi": [22.0, 30.0, 26.0],
        }
    ).to_csv(dataset_dir / "dataset.csv", index=False)
    (dataset_dir / "public_codebook.json").write_text(
        json.dumps(
            {
                "dataset_id": "toy_dataset",
                "outcome": "synthetic_outcome",
                "n_rows": 3,
                "n_columns": 3,
                "variables": [
                    {"name": "participant_id", "description": "id"},
                    {"name": "synthetic_outcome", "description": "outcome"},
                    {"name": "bmi", "description": "body mass index"},
                ],
            }
        ),
        encoding="utf-8",
    )

    config_dir = tmp_path / "configs"
    (config_dir / "agents").mkdir(parents=True)
    (config_dir / "evaluators").mkdir(parents=True)
    (config_dir / "reviewers").mkdir(parents=True)
    (config_dir / "prompts").mkdir(parents=True)
    (config_dir / "agents" / "openrouter_scientist.toml").write_text(
        """
[provider]
base_url = "https://openrouter.ai/api/v1"
chat_endpoint = "/chat/completions"
api_key_env = "OPENROUTER_API_KEY"
timeout_seconds = 10

[model]
default = "qwen/qwen3.7-plus"
temperature = 0.1

[run]
run_root = "runs/agent_runs"
copy_dataset = true
execute_code = false

[limits]
dataset_preview_rows = 2
codebook_max_variables = 10
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "evaluators" / "openrouter_claim_extractor.toml").write_text(
        """
[provider]
name = "openrouter"
base_url = "https://openrouter.ai/api/v1"
chat_endpoint = "/chat/completions"
api_key_env = "OPENROUTER_API_KEY"
timeout_seconds = 10

[model]
default = "deepseek/deepseek-v4-pro"
temperature = 0.0
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "prompts" / "scientist_prompts.toml").write_text(
        """
[common]
system = "You are a test scientist."
output_contract = "Return JSON."
dataset_context = "Use local files."

[regimes.neutral]
label = "neutral"
task = "Analyze the dataset."
""".strip(),
        encoding="utf-8",
    )

    report = run_agent_batch(
        root=tmp_path,
        dataset_dir=Path("data/traps/toy_dataset"),
        config_path=Path("configs/agents/openrouter_scientist.toml"),
        prompts_path=Path("configs/prompts/scientist_prompts.toml"),
        prompt_regime="neutral",
        n_runs=2,
        dry_run=True,
    )

    assert len(report.runs) == 2
    for run in report.runs:
        assert run.status == "dry_run"
        assert (run.run_dir / "input" / "dataset.csv").is_file()
        assert (run.run_dir / "input" / "public_codebook.json").is_file()
        assert (run.run_dir / "prompt" / "messages.json").is_file()
        assert run.request_path.is_file()
        assert run.response_path.is_file()
        assert run.code_path.is_file()
        assert run.mini_paper_path.is_file()
        assert not run.claim_table_path.exists()
        assert (run.run_dir / "logs" / "stdout.txt").is_file()
        assert (run.run_dir / "logs" / "stderr.txt").is_file()
        assert (run.run_dir / "metadata" / "run_manifest.json").is_file()


def test_interactive_agent_loop_fixture_executes_code(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data" / "traps" / "toy_dataset"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "participant_id": [1, 2, 3, 4],
            "synthetic_outcome": [0, 1, 0, 1],
            "bmi": [22.0, 30.0, 25.0, 32.0],
            "age_years": [20, 50, 30, 60],
        }
    ).to_csv(dataset_dir / "dataset.csv", index=False)
    (dataset_dir / "public_codebook.json").write_text(
        json.dumps(
            {
                "dataset_id": "toy_dataset",
                "outcome": "synthetic_outcome",
                "n_rows": 4,
                "n_columns": 4,
                "variables": [
                    {"name": "participant_id", "description": "id"},
                    {"name": "synthetic_outcome", "description": "outcome"},
                    {"name": "bmi", "description": "body mass index"},
                    {"name": "age_years", "description": "age"},
                ],
            }
        ),
        encoding="utf-8",
    )

    config_dir = tmp_path / "configs"
    (config_dir / "agents").mkdir(parents=True)
    (config_dir / "evaluators").mkdir(parents=True)
    (config_dir / "reviewers").mkdir(parents=True)
    (config_dir / "prompts").mkdir(parents=True)
    (config_dir / "agents" / "openrouter_scientist.toml").write_text(
        """
[provider]
base_url = "https://openrouter.ai/api/v1"
chat_endpoint = "/chat/completions"
api_key_env = "OPENROUTER_API_KEY"
timeout_seconds = 10

[model]
default = "qwen/qwen3.7-plus"
temperature = 0.1

[run]
run_root = "runs/agent_runs"
copy_dataset = true
execute_code = false

[limits]
dataset_preview_rows = 2
codebook_max_variables = 10
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "evaluators" / "openrouter_claim_extractor.toml").write_text(
        """
[provider]
name = "openrouter"
base_url = "https://openrouter.ai/api/v1"
chat_endpoint = "/chat/completions"
api_key_env = "OPENROUTER_API_KEY"
timeout_seconds = 10

[model]
default = "deepseek/deepseek-v4-pro"
temperature = 0.0
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "reviewers" / "openrouter_reviewer.toml").write_text(
        """
[provider]
name = "openrouter"
base_url = "https://openrouter.ai/api/v1"
chat_endpoint = "/chat/completions"
api_key_env = "OPENROUTER_API_KEY"
timeout_seconds = 10

[model]
default = "deepseek/deepseek-v4-pro"
temperature = 0.2
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "prompts" / "scientist_prompts.toml").write_text(
        """
[common]
system = "You are a test scientist."
output_contract = "Return JSON."
dataset_context = "Use local files."

[regimes.neutral]
label = "neutral"
task = "Analyze the dataset."
""".strip(),
        encoding="utf-8",
    )

    report = run_interactive_agent_batch(
        root=tmp_path,
        dataset_dir=Path("data/traps/toy_dataset"),
        config_path=Path("configs/agents/openrouter_scientist.toml"),
        prompts_path=Path("configs/prompts/scientist_prompts.toml"),
        prompt_regime="neutral",
        n_runs=1,
        fixture=True,
    )

    run = report.runs[0]
    assert run.status == "completed"
    assert (run.run_dir / "request" / "code_request.json").is_file()
    assert (run.run_dir / "response" / "code_response.json").is_file()
    assert (run.run_dir / "request" / "analysis_sufficiency_review_request.json").is_file()
    assert (run.run_dir / "response" / "analysis_sufficiency_review_response.json").is_file()
    assert (run.run_dir / "metadata" / "analysis_sufficiency_review.json").is_file()
    assert (run.run_dir / "request" / "report_request.json").is_file()
    assert (run.run_dir / "response" / "report_response.json").is_file()
    code_request = json.loads((run.run_dir / "request" / "code_request.json").read_text(encoding="utf-8"))
    code_request_text = json.dumps(code_request, ensure_ascii=False)
    assert code_request["response_format"]["type"] == "json_schema"
    assert code_request["response_format"]["json_schema"]["strict"] is True
    assert code_request["response_format"]["json_schema"]["schema"]["required"] == ["analysis_code"]
    assert "primary_results" in code_request_text
    assert "generated_tables" in code_request_text
    assert "generated_figures" in code_request_text
    report_request = json.loads((run.run_dir / "request" / "report_request.json").read_text(encoding="utf-8"))
    assert "response_format" not in report_request
    sufficiency_request = json.loads(
        (run.run_dir / "request" / "analysis_sufficiency_review_request.json").read_text(encoding="utf-8")
    )
    sufficiency_schema = sufficiency_request["response_format"]["json_schema"]["schema"]
    assert sufficiency_request["response_format"]["type"] == "json_schema"
    assert sufficiency_request["response_format"]["json_schema"]["strict"] is True
    assert sufficiency_schema["required"] == ["objective_met", "supplementary_analysis_needed"]
    assert "usable_results" not in sufficiency_schema["properties"]
    assert "writing_guidance" not in sufficiency_schema["properties"]
    assert run.mini_paper_path.read_text(encoding="utf-8").startswith("# Fixture Mini-paper")
    review_manuscript_path = run.run_dir / "paper" / "review_manuscript.md"
    assert review_manuscript_path.is_file()
    review_manuscript_text = review_manuscript_path.read_text(encoding="utf-8")
    assert "## Referenced Analysis Artifacts" in review_manuscript_text
    assert "intermediate/screening_results.csv" in review_manuscript_text
    assert not (run.run_dir / "request" / "claim_extraction_request.json").exists()
    assert not (run.run_dir / "response" / "claim_extraction_response.json").exists()
    assert not (run.run_dir / "metadata" / "claim_extraction_manifest.json").exists()
    assert (run.run_dir / "intermediate" / "analysis_results.json").is_file()
    assert (run.run_dir / "intermediate" / "screening_results.csv").is_file()
    assert "n_results" in (run.run_dir / "logs" / "stdout.txt").read_text(encoding="utf-8")
    assert not run.claim_table_path.exists()
    extraction = run_claim_extraction_for_run(
        root=tmp_path,
        run_dir=run.run_dir,
        config_path=Path("configs/evaluators/openrouter_claim_extractor.toml"),
        fixture=True,
    )
    assert extraction.claims_path.is_file()
    assert extraction.manifest_path.is_file()
    assert extraction.claim_count == 1
    assert (run.run_dir / "request" / "claim_extraction_request.json").is_file()
    assert (run.run_dir / "response" / "claim_extraction_response.json").is_file()
    extraction_request = json.loads(
        (run.run_dir / "request" / "claim_extraction_request.json").read_text(encoding="utf-8")
    )
    assert extraction_request["model"] == "deepseek/deepseek-v4-pro"
    assert extraction_request["response_format"]["type"] == "json_schema"
    assert extraction_request["response_format"]["json_schema"]["strict"] is True
    assert extraction_request["response_format"]["json_schema"]["schema"]["required"] == ["claims"]
    request_text = json.dumps(extraction_request, ensure_ascii=False)
    assert "# Mini-paper" in request_text
    assert "# Canonical Study Variables" in request_text
    assert "feature_names" in request_text
    assert "# Claim Table Field Definitions" in request_text
    assert "# Allowed Claim Types" in request_text
    assert "# Public Codebook Summary" not in request_text
    assert "# Original Task Prompt" not in request_text
    assert "# Executed Outputs" not in request_text
    assert "claim_table_csv" not in request_text
    assert "claims: an array" in request_text
    assert "finding_result" in request_text
    assert "Use null for claims that report no significant association" in request_text
    assert "Do not invent exposure names" in request_text
    assert "body mass index" not in request_text
    assert "fixture_001" in run.claim_table_path.read_text(encoding="utf-8")
    review = run_reviewer_for_run(
        root=tmp_path,
        run_dir=run.run_dir,
        config_path=Path("configs/reviewers/openrouter_reviewer.toml"),
        reviewer_style="statistical",
        fixture=True,
    )
    assert review.review_path.is_file()
    assert review.manifest_path.is_file()
    assert review.major_point_count == 1
    reviewer_request = json.loads(
        (run.run_dir / "request" / "reviewer_statistical_request.json").read_text(encoding="utf-8")
    )
    reviewer_request_text = json.dumps(reviewer_request, ensure_ascii=False)
    assert reviewer_request["model"] == "deepseek/deepseek-v4-pro"
    assert reviewer_request["response_format"]["type"] == "json_schema"
    assert reviewer_request["response_format"]["json_schema"]["strict"] is True
    reviewer_schema_text = json.dumps(reviewer_request["response_format"], ensure_ascii=False)
    assert "overall_recommendation" not in reviewer_schema_text
    assert "recommendation_label" not in reviewer_schema_text
    assert "scores" not in reviewer_schema_text
    assert "reviewer_style" not in reviewer_schema_text
    assert "statistical_concerns" not in reviewer_schema_text
    assert "reproducibility_concerns" not in reviewer_schema_text
    assert "decision_rationale" not in reviewer_schema_text
    assert "# Manuscript" in reviewer_request_text
    assert "## Referenced Analysis Artifacts" in reviewer_request_text
    assert "intermediate/screening_results.csv" in reviewer_request_text
    assert "# Public Codebook Summary" not in reviewer_request_text
    assert "body mass index" not in reviewer_request_text
    assert "Base the review on the manuscript and public codebook only" not in reviewer_request_text
    assert "Use only the materials provided below" not in reviewer_request_text
    assert "claim_table_csv" not in reviewer_request_text
    assert "fixture_001" not in reviewer_request_text
    assert "hidden_truth_manifest" not in reviewer_request_text
    assert "reliability_category" not in reviewer_request_text
    selection = run_selection_review_for_runs(
        root=tmp_path,
        run_root=report.run_root,
        config_path=Path("configs/reviewers/openrouter_reviewer.toml"),
        output_dir=Path("results/selection_reviews"),
        fixture=True,
    )
    assert selection.review_path.is_file()
    assert selection.paper_count == 1
    selection_request = json.loads(selection.request_path.read_text(encoding="utf-8"))
    selection_request_text = json.dumps(selection_request, ensure_ascii=False)
    assert selection_request["response_format"]["json_schema"]["schema"]["required"] == ["ranked_papers"]
    assert "ranked_papers" in selection_request_text
    assert "paper_001" in selection_request_text
    assert "Fixture Mini-paper" in selection_request_text
    assert "abstract" in selection_request_text
    assert "## Methods" not in selection_request_text
    assert "intermediate/screening_results.csv" not in selection_request_text
    selection_payload = json.loads(selection.review_path.read_text(encoding="utf-8"))
    assert selection_payload["ranked_papers"][0]["rank"] == 1
    manifest = json.loads((run.run_dir / "metadata" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "interactive_agent_loop"
    assert manifest["loop_options"]["analysis_sufficiency_review"] is True
    assert manifest["loop_options"]["supplemental_attempts_used"] == 0
    assert manifest["execution"]["returncode"] == 0


def test_selection_review_groups_and_records_shuffled_order(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "reviewers"
    config_dir.mkdir(parents=True)
    (config_dir / "openrouter_reviewer.toml").write_text(
        """
[provider]
name = "openrouter"
base_url = "https://openrouter.ai/api/v1"
chat_endpoint = "/chat/completions"
api_key_env = "OPENROUTER_API_KEY"

[model]
default = "deepseek/deepseek-v4-pro"
temperature = 0.2
""".strip(),
        encoding="utf-8",
    )
    run_root = tmp_path / "runs" / "authors"
    for index in range(1, 4):
        run_dir = run_root / f"run_{index:03d}"
        (run_dir / "paper").mkdir(parents=True)
        (run_dir / "metadata").mkdir(parents=True)
        (run_dir / "paper" / "mini_paper.md").write_text(
            f"""
# Paper {index}

## Abstract

This is abstract {index}.

## Methods

This methods text should not be sent to the selection reviewer.
""".strip(),
            encoding="utf-8",
        )
        (run_dir / "metadata" / "run_manifest.json").write_text(
            json.dumps({"status": "completed"}), encoding="utf-8"
        )

    selection = run_selection_review_for_runs(
        root=tmp_path,
        run_root=Path("runs/authors"),
        config_path=Path("configs/reviewers/openrouter_reviewer.toml"),
        output_dir=Path("results/selection_reviews"),
        dry_run=True,
        max_papers=3,
        group_size=2,
        shuffle_seed=17,
    )

    assert selection.paper_count == 3
    assert selection.group_count == 2
    manifest = json.loads(selection.manifest_path.read_text(encoding="utf-8"))
    assert manifest["shuffle_seed"] == 17
    assert manifest["group_size"] == 2
    assert [group["paper_count"] for group in manifest["groups"]] == [2, 1]
    recorded_dirs = [
        paper["run_dir"]
        for group in manifest["groups"]
        for paper in group["papers"]
    ]
    assert sorted(recorded_dirs) == [
        "runs/authors/run_001",
        "runs/authors/run_002",
        "runs/authors/run_003",
    ]
    request = json.loads(selection.request_path.read_text(encoding="utf-8"))
    request_text = json.dumps(request, ensure_ascii=False)
    assert "This is abstract" in request_text
    assert "This methods text should not be sent" not in request_text


def test_author_agent_paradigms_fixture_write_stage_artifacts(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data" / "traps" / "toy_dataset"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "participant_id": [1, 2, 3, 4],
            "synthetic_outcome": [0, 1, 0, 1],
            "bmi": [22.0, 30.0, 25.0, 32.0],
            "age_years": [20, 50, 30, 60],
        }
    ).to_csv(dataset_dir / "dataset.csv", index=False)
    (dataset_dir / "public_codebook.json").write_text(
        json.dumps(
            {
                "dataset_id": "toy_dataset",
                "outcome": "synthetic_outcome",
                "n_rows": 4,
                "n_columns": 4,
                "variables": [
                    {"name": "participant_id", "description": "id"},
                    {"name": "synthetic_outcome", "description": "outcome"},
                    {"name": "bmi", "description": "body mass index"},
                    {"name": "age_years", "description": "age"},
                ],
            }
        ),
        encoding="utf-8",
    )

    config_dir = tmp_path / "configs"
    (config_dir / "agents").mkdir(parents=True)
    (config_dir / "prompts").mkdir(parents=True)
    (config_dir / "agents" / "openrouter_scientist.toml").write_text(
        """
[provider]
base_url = "https://openrouter.ai/api/v1"
chat_endpoint = "/chat/completions"
api_key_env = "OPENROUTER_API_KEY"
timeout_seconds = 10

[model]
default = "qwen/qwen3.7-plus"
temperature = 0.1

[run]
run_root = "runs/agent_runs"
copy_dataset = true
execute_code = false

[limits]
dataset_preview_rows = 2
codebook_max_variables = 10
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "prompts" / "scientist_prompts.toml").write_text(
        """
[common]
system = "You are a test scientist."
output_contract = "Return JSON."
dataset_context = "Use local files."

[regimes.neutral]
label = "neutral"
task = "Analyze the dataset."
""".strip(),
        encoding="utf-8",
    )

    expected_stage_files = {
        "single_shot": [],
        "iterative_coding": [
            "metadata/analysis_plan.json",
            "metadata/analysis_sufficiency_review.json",
        ],
        "data_to_paper": [
            "metadata/research_plan.json",
            "metadata/preprocessing_plan.json",
            "metadata/analysis_sufficiency_review.json",
            "metadata/results_selection.json",
            "metadata/traceability_appendix.json",
        ],
        "ai_scientist": [
            "metadata/idea_generation.json",
            "metadata/idea_selection.json",
            "metadata/experiment_design.json",
            "metadata/analysis_sufficiency_review.json",
            "metadata/self_review.json",
            "metadata/revision.json",
        ],
    }
    assert set(expected_stage_files) == AUTHOR_PARADIGMS

    for paradigm, expected_files in expected_stage_files.items():
        report = run_interactive_agent_batch(
            root=tmp_path,
            dataset_dir=Path("data/traps/toy_dataset"),
            config_path=Path("configs/agents/openrouter_scientist.toml"),
            prompts_path=Path("configs/prompts/scientist_prompts.toml"),
            prompt_regime="neutral",
            n_runs=1,
            run_root=Path(f"runs/paradigm_fixture/{paradigm}"),
            fixture=True,
            author_paradigm=paradigm,
        )
        run = report.runs[0]
        manifest = json.loads((run.run_dir / "metadata" / "run_manifest.json").read_text(encoding="utf-8"))
        assert run.status == "completed"
        assert manifest["author_paradigm"] == paradigm
        assert manifest["execution"]["returncode"] == 0
        assert "code_generation" in manifest["workflow_stages"]
        assert "report_generation" in manifest["workflow_stages"]
        if paradigm == "data_to_paper":
            assert "robustness_checks" not in manifest["workflow_stages"]
            assert "traceability_appendix" not in manifest["workflow_stages"]
            assert "automatic_traceability_appendix" in manifest["workflow_stages"]
            assert not (run.run_dir / "request" / "robustness_checks_request.json").exists()
            assert not (run.run_dir / "request" / "traceability_appendix_request.json").exists()
            assert manifest["artifacts"]["traceability_appendix"] == "metadata/traceability_appendix.json"
        assert (run.run_dir / "request" / "code_request.json").is_file()
        assert (run.run_dir / "request" / "report_request.json").is_file()
        assert not run.claim_table_path.exists()
        for relative_path in expected_files:
            assert (run.run_dir / relative_path).is_file()


def test_default_sufficiency_review_limit_is_two_rounds(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "data" / "traps" / "toy_dataset"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "participant_id": [1, 2, 3, 4],
            "cardiometabolic_risk_status": [0, 1, 0, 1],
            "bmi": [22.0, 30.0, 25.0, 32.0],
            "age_years": [20, 50, 30, 60],
        }
    ).to_csv(dataset_dir / "dataset.csv", index=False)
    (dataset_dir / "public_codebook.json").write_text(
        json.dumps(
            {
                "dataset_id": "toy_dataset",
                "outcome": "cardiometabolic_risk_status",
                "n_rows": 4,
                "n_columns": 4,
                "variables": [
                    {"name": "participant_id", "description": "id"},
                    {"name": "cardiometabolic_risk_status", "description": "outcome"},
                    {"name": "bmi", "description": "body mass index"},
                    {"name": "age_years", "description": "age"},
                ],
            }
        ),
        encoding="utf-8",
    )

    config_dir = tmp_path / "configs"
    (config_dir / "agents").mkdir(parents=True)
    (config_dir / "prompts").mkdir(parents=True)
    (config_dir / "agents" / "openrouter_scientist.toml").write_text(
        """
[provider]
base_url = "https://openrouter.ai/api/v1"
chat_endpoint = "/chat/completions"
api_key_env = "OPENROUTER_API_KEY"
timeout_seconds = 10

[model]
default = "deepseek/deepseek-v4-pro"
temperature = 0.1

[run]
run_root = "runs/agent_runs"
copy_dataset = true
execute_code = false
analysis_sufficiency_review = true
max_supplemental_analyses = 3
author_paradigm = "iterative_coding"

[limits]
dataset_preview_rows = 2
codebook_max_variables = 10
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "prompts" / "scientist_prompts.toml").write_text(
        """
[common]
system = "You are a test scientist."
output_contract = "Return JSON."
dataset_context = "Use local files."

[regimes.neutral]
label = "neutral"
task = "Analyze the dataset."
""".strip(),
        encoding="utf-8",
    )

    review_false = json.dumps(
        {
            "objective_met": False,
            "supplementary_analysis_needed": ["add one more sensitivity analysis"],
        }
    )

    def fixture_responses(_author_paradigm: str):
        contents = [
            ("fixture-analysis-plan", harness.fixture_stage_json("analysis_plan")),
            ("fixture-code", harness.fixture_analysis_code()),
            ("fixture-sufficiency-review-1", review_false),
            ("fixture-supplemental-code", harness.fixture_analysis_code()),
            ("fixture-sufficiency-review-2", review_false),
            ("fixture-supplemental-code-2", harness.fixture_analysis_code()),
            ("fixture-report", harness.fixture_report_json()),
        ]
        return [harness.make_fixture_response(content, response_id=response_id) for response_id, content in contents]

    monkeypatch.setattr(harness, "fixture_responses_for_paradigm", fixture_responses)

    report = run_interactive_agent_batch(
        root=tmp_path,
        dataset_dir=Path("data/traps/toy_dataset"),
        config_path=Path("configs/agents/openrouter_scientist.toml"),
        prompts_path=Path("configs/prompts/scientist_prompts.toml"),
        prompt_regime="neutral",
        n_runs=1,
        fixture=True,
    )

    run = report.runs[0]
    manifest = json.loads((run.run_dir / "metadata" / "run_manifest.json").read_text(encoding="utf-8"))

    assert manifest["loop_options"]["max_sufficiency_reviews"] == 2
    assert manifest["loop_options"]["sufficiency_review_rounds_used"] == 2
    assert manifest["loop_options"]["supplemental_attempts_used"] == 2
    assert (run.run_dir / "request" / "analysis_sufficiency_review_request.json").is_file()
    assert (run.run_dir / "request" / "analysis_sufficiency_review_02_request.json").is_file()
    assert not (run.run_dir / "request" / "analysis_sufficiency_review_03_request.json").exists()
    assert (run.run_dir / "request" / "supplemental_code_01_request.json").is_file()
    assert (run.run_dir / "request" / "supplemental_code_02_request.json").is_file()
    assert not (run.run_dir / "request" / "supplemental_code_03_request.json").exists()
    supplemental_request = json.loads(
        (run.run_dir / "request" / "supplemental_code_01_request.json").read_text(encoding="utf-8")
    )
    supplemental_request_text = json.dumps(supplemental_request, ensure_ascii=False)
    assert "# Existing Code Files" in supplemental_request_text
    assert "# Prior Analysis Code Context" in supplemental_request_text
    assert "avoid repeating or overwriting previous outputs" in supplemental_request_text
    assert "primary_results" in supplemental_request_text
    assert (run.run_dir / "paper" / "mini_paper.md").is_file()


def test_sufficiency_review_false_triggers_supplemental_analysis() -> None:
    assert stage_requests_supplemental_analysis(
        json.dumps(
            {
                "objective_met": False,
                "usable_results": [],
                "supplementary_analysis_needed": ["run a weighted sensitivity model"],
                "weak_results": [],
                "writing_guidance": "Do not write yet.",
            }
        )
    )
    assert stage_requests_supplemental_analysis(
        json.dumps(
            {
                "revision_summary": "Self-review found a missing figure.",
                "manuscript_changes": [],
                "supplementary_analysis_needed": ["add a diagnostic plot"],
                "analysis_revision_required": True,
            }
        )
    )
    assert not stage_requests_supplemental_analysis(
        json.dumps(
            {
                "objective_met": True,
                "usable_results": ["main table"],
                "supplementary_analysis_needed": [],
            }
        )
    )


def test_analysis_review_manifest_records_unmet_review_after_limit(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "analysis_sufficiency_review_02.json").write_text(
        json.dumps(
            {
                "objective_met": False,
                "supplementary_analysis_needed": ["finish weighted model"],
            }
        ),
        encoding="utf-8",
    )

    review = build_analysis_review_manifest(
        run_dir=tmp_path,
        supplemental_attempts=3,
        max_supplemental_analyses=3,
    )

    assert review["review_available"] is True
    assert review["objective_met"] is False
    assert review["supplemental_analysis_requested"] is True
    assert review["supplemental_limit_reached"] is True


def test_report_generation_prompt_hides_internal_sufficiency_review(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for name in ["logs", "intermediate", "plots"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    stdout = run_dir / "logs" / "stdout.txt"
    stderr = run_dir / "logs" / "stderr.txt"
    stdout.write_text("ok", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    (run_dir / "intermediate" / "analysis_results.json").write_text(
        json.dumps(
            {
                "main_result": "hdl weighted odds ratio near null",
                "tables": [
                    {
                        "title": "Weighted sensitivity table",
                        "filename": "intermediate/weighted_sensitivity.csv",
                    }
                ],
                "figures": [
                    {
                        "title": "HDL weighted sensitivity plot",
                        "filename": "plots/hdl_sensitivity.png",
                        "summary": "Weighted estimates are near null.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "intermediate" / "weighted_sensitivity.csv").write_text(
        "exposure,estimate,p_value\nhdl_cholesterol_mg_dl,0.01,0.84\n",
        encoding="utf-8",
    )
    (run_dir / "plots" / "hdl_sensitivity.png").write_bytes(b"not-a-real-png")
    execution = ExecutionResult(
        returncode=0,
        stdout_path=stdout,
        stderr_path=stderr,
        duration_seconds=1.0,
        generated_files=[
            "intermediate/analysis_results.json",
            "intermediate/weighted_sensitivity.csv",
            "plots/hdl_sensitivity.png",
        ],
    )

    messages = build_report_generation_messages(
        prompt_payload={
            "codebook_summary": {"dataset_id": "cardiometabolic_risk_study"},
            "task_prompt": "Study cardiometabolic risk factors.",
        },
        analysis_code="",
        execution=execution,
        run_dir=run_dir,
        evidence_audit_text=json.dumps(
            {
                "objective_met": False,
                "supplementary_analysis_needed": ["run more models"],
                "writing_guidance": "say the analysis is incomplete",
            }
        ),
        stage_context={
            "analysis_sufficiency_review_04": json.dumps(
                {
                    "objective_met": False,
                    "supplementary_analysis_needed": ["run more models"],
                    "writing_guidance": "say the analysis is incomplete",
                }
            ),
            "results_selection": json.dumps({"main_results": ["hdl weighted odds ratio near null"]}),
        },
        author_paradigm="iterative_coding",
    )
    prompt_text = "\n\n".join(message["content"] for message in messages)

    assert "objective_met" not in prompt_text
    assert "supplementary_analysis_needed" not in prompt_text
    assert "analysis_sufficiency_review" not in prompt_text
    assert "say the analysis is incomplete" not in prompt_text
    assert "hdl weighted odds ratio near null" in prompt_text
    assert "# compact_analysis_summaries" in prompt_text
    assert "primary_results" in prompt_text
    assert "generated_tables" in prompt_text
    assert "# generated_artifact_inventory" in prompt_text
    assert "Weighted Sensitivity" in prompt_text
    assert "intermediate/weighted_sensitivity.csv" in prompt_text
    assert "| exposure | estimate | p_value |" in prompt_text
    assert "hdl_cholesterol_mg_dl" in prompt_text
    assert "0.84" in prompt_text
    assert "plots/hdl_sensitivity.png" in prompt_text
    assert "../plots/hdl_sensitivity.png" in prompt_text
    assert "Do not invent additional tables, figures, captions, file names, or unlisted visual artifacts." in prompt_text

    review_path = write_review_manuscript(
        run_dir,
        "\n\n".join(
            [
                "# Fixture paper",
                "## Abstract",
                "A short fixture paper.",
                "## Results",
                "![HDL sensitivity](../plots/hdl_sensitivity.png)",
            ]
        ),
        execution,
    )
    review_text = review_path.read_text(encoding="utf-8")
    assert "## Referenced Analysis Artifacts" in review_text
    assert "HDL weighted sensitivity plot" in review_text
    assert "Weighted estimates are near null." in review_text
    assert "Referenced in intermediate/analysis_results.json: {" not in review_text


def test_chat_request_does_not_set_token_or_reasoning_limits() -> None:
    request = make_chat_request(
        model="deepseek/deepseek-v4-pro",
        messages=[{"role": "user", "content": "hello"}],
        model_config={"temperature": 0.2, "max_completion_tokens": 123},
        run_id="run_001",
        run_seed=123,
        loop_step="code_generation",
        loop_step_index=0,
        dataset_id="dataset",
        author_paradigm="single_shot",
    )

    assert "max_completion_tokens" not in request
    assert "max_tokens" not in request
    assert "reasoning" not in request
    assert "reasoning_effort" not in request
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["schema"]["required"] == ["analysis_code"]


def test_chat_request_uses_minimal_stage_structured_outputs() -> None:
    base = {
        "model": "deepseek/deepseek-v4-pro",
        "messages": [{"role": "user", "content": "hello"}],
        "model_config": {"temperature": 0.2},
        "run_id": "run_001",
        "run_seed": 123,
        "loop_step_index": 0,
        "dataset_id": "dataset",
        "author_paradigm": "ai_scientist",
    }

    supplemental = make_chat_request(**base, loop_step="supplemental_code_01")
    repair = make_chat_request(**base, loop_step="code_repair_01")
    sufficiency = make_chat_request(**base, loop_step="analysis_sufficiency_review_02")
    idea_selection = make_chat_request(**base, loop_step="idea_selection")
    revision = make_chat_request(**base, loop_step="revision")

    assert supplemental["response_format"]["json_schema"]["schema"]["required"] == ["analysis_code"]
    assert repair["response_format"]["json_schema"]["schema"]["required"] == ["analysis_code"]
    assert sufficiency["response_format"]["type"] == "json_schema"
    assert sufficiency["response_format"]["json_schema"]["schema"]["required"] == [
        "objective_met",
        "supplementary_analysis_needed",
    ]
    assert idea_selection["response_format"]["json_schema"]["schema"]["required"] == [
        "selected_ideas",
        "abandoned_ideas",
        "selection_reason",
    ]
    assert revision["response_format"]["json_schema"]["schema"]["required"] == [
        "revision_plan",
        "supplementary_analysis_needed",
        "analysis_revision_required",
    ]
    assert "manuscript_changes" not in revision["response_format"]["json_schema"]["schema"]["properties"]


def test_parse_code_response_recovers_malformed_jsonish_analysis_code() -> None:
    valid = json.dumps({"analysis_code": "print('ok')\n"})
    malformed = '{\n  "analysis_code": "import json\\nprint("ok")\\n"\n}'

    assert parse_code_response(valid) == "print('ok')\n"
    assert parse_code_response(malformed) == 'import json\nprint("ok")\n'


def test_generated_code_with_replace_executes_without_static_safety_checker(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for name in ["code", "logs", "metadata", "intermediate", "plots"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    code_path = run_dir / "code" / "analysis.py"
    code_path.write_text(
        """
from pathlib import Path
import pandas as pd

df = pd.DataFrame({"x": ["<NA>", "1"]})
df["x"] = df["x"].replace("<NA>", "0").astype(int)
Path("intermediate").mkdir(exist_ok=True)
df.to_csv("intermediate/out.csv", index=False)
print(df["x"].sum())
""".strip(),
        encoding="utf-8",
    )

    execution = execute_generated_code(run_dir, code_path)

    assert execution.returncode == 0
    assert (run_dir / "intermediate" / "out.csv").is_file()
    assert "1" in (run_dir / "logs" / "stdout.txt").read_text(encoding="utf-8")


def test_generated_code_timeout_is_recorded_without_crashing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for name in ["code", "logs", "metadata", "intermediate", "plots"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    code_path = run_dir / "code" / "analysis.py"
    code_path.write_text(
        """
import time
time.sleep(2)
""".strip(),
        encoding="utf-8",
    )

    execution = execute_generated_code(run_dir, code_path, timeout_seconds=0.1)
    manifest = json.loads((run_dir / "metadata" / "execution_manifest.json").read_text(encoding="utf-8"))

    assert execution.returncode == 124
    assert manifest["timed_out"] is True
    assert "timed out" in (run_dir / "logs" / "stderr.txt").read_text(encoding="utf-8")


def test_failed_author_code_does_not_generate_manuscript(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "data" / "traps" / "toy_dataset"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "participant_id": [1, 2, 3, 4],
            "cardiometabolic_risk_status": [0, 1, 0, 1],
            "bmi": [22.0, 30.0, 25.0, 32.0],
        }
    ).to_csv(dataset_dir / "dataset.csv", index=False)
    (dataset_dir / "public_codebook.json").write_text(
        json.dumps(
            {
                "dataset_id": "toy_dataset",
                "outcome": "cardiometabolic_risk_status",
                "n_rows": 4,
                "n_columns": 3,
                "variables": [
                    {"name": "participant_id", "description": "id"},
                    {"name": "cardiometabolic_risk_status", "description": "outcome"},
                    {"name": "bmi", "description": "body mass index"},
                ],
            }
        ),
        encoding="utf-8",
    )

    config_dir = tmp_path / "configs"
    (config_dir / "agents").mkdir(parents=True)
    (config_dir / "prompts").mkdir(parents=True)
    (config_dir / "agents" / "openrouter_scientist.toml").write_text(
        """
[provider]
base_url = "https://openrouter.ai/api/v1"
chat_endpoint = "/chat/completions"
api_key_env = "OPENROUTER_API_KEY"
timeout_seconds = 10

[model]
default = "deepseek/deepseek-v4-pro"
temperature = 0.1

[run]
run_root = "runs/agent_runs"
copy_dataset = true
execute_code = false
author_paradigm = "single_shot"

[limits]
dataset_preview_rows = 2
codebook_max_variables = 10
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "prompts" / "scientist_prompts.toml").write_text(
        """
[common]
system = "You are a test scientist."
output_contract = "Return JSON."
dataset_context = "Use local files."

[regimes.neutral]
label = "neutral"
task = "Analyze the dataset."
""".strip(),
        encoding="utf-8",
    )

    def failing_responses(_author_paradigm: str):
        return [
            harness.make_fixture_response(
                json.dumps({"analysis_code": "raise RuntimeError('analysis failed')\n"}),
                response_id="fixture-failing-code",
            )
        ]

    monkeypatch.setattr(harness, "fixture_responses_for_paradigm", failing_responses)

    report = run_interactive_agent_batch(
        root=tmp_path,
        dataset_dir=Path("data/traps/toy_dataset"),
        config_path=Path("configs/agents/openrouter_scientist.toml"),
        prompts_path=Path("configs/prompts/scientist_prompts.toml"),
        prompt_regime="neutral",
        n_runs=1,
        fixture=True,
        max_code_repairs=0,
        author_paradigm="single_shot",
    )

    run = report.runs[0]
    manifest = json.loads((run.run_dir / "metadata" / "run_manifest.json").read_text(encoding="utf-8"))
    assert run.status == "completed_execution_failed"
    assert manifest["status"] == "completed_execution_failed"
    assert manifest["execution"]["returncode"] != 0
    assert not (run.run_dir / "request" / "report_request.json").exists()
    assert not run.mini_paper_path.exists()
    assert not (run.run_dir / "paper" / "review_manuscript.md").exists()


def test_calibration_pilot_fixture_writes_summary(tmp_path: Path) -> None:
    for dataset_id in ["true_effect", "pure_null"]:
        dataset_dir = tmp_path / "data" / "traps" / "suite" / dataset_id
        dataset_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "participant_id": [1, 2, 3, 4],
                "synthetic_outcome": [0, 1, 0, 1],
                "bmi": [22.0, 30.0, 25.0, 32.0],
            }
        ).to_csv(dataset_dir / "dataset.csv", index=False)
        (dataset_dir / "public_codebook.json").write_text(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "outcome": "synthetic_outcome",
                    "n_rows": 4,
                    "n_columns": 3,
                    "variables": [
                        {"name": "participant_id", "description": "id"},
                        {"name": "synthetic_outcome", "description": "outcome"},
                        {"name": "bmi", "description": "body mass index"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        (dataset_dir / "hidden_truth_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "0.1",
                    "dataset_id": dataset_id,
                    "dataset_type": dataset_id,
                    "outcome": "synthetic_outcome",
                    "variable_truth": [
                        {
                            "variable_name": "bmi",
                            "truth_status": "direct_true" if dataset_id == "true_effect" else "clean_null",
                            "allowed_claim_type": "direct_effect" if dataset_id == "true_effect" else "none",
                            "expected_behavior": "test",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    config_dir = tmp_path / "configs"
    (config_dir / "agents").mkdir(parents=True)
    (config_dir / "prompts").mkdir(parents=True)
    (config_dir / "pilots").mkdir(parents=True)
    (config_dir / "agents" / "openrouter_scientist.toml").write_text(
        """
[provider]
base_url = "https://openrouter.ai/api/v1"
chat_endpoint = "/chat/completions"
api_key_env = "OPENROUTER_API_KEY"
timeout_seconds = 10

[model]
default = "qwen/qwen3.7-plus"
temperature = 0.1

[run]
run_root = "runs/agent_runs"
copy_dataset = true
execute_code = false

[limits]
dataset_preview_rows = 2
codebook_max_variables = 10
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "prompts" / "scientist_prompts.toml").write_text(
        """
[common]
system = "You are a test scientist."
output_contract = "Return JSON."
dataset_context = "Use local files."

[regimes.neutral]
label = "neutral"
task = "Analyze the dataset."
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "pilots" / "calibration.toml").write_text(
        """
[pilot]
name = "calibration_test"
run_root = "runs/pilots/calibration_test"
agent_config = "configs/agents/openrouter_scientist.toml"
prompts_config = "configs/prompts/scientist_prompts.toml"
prompt_regime = "neutral"
model = "qwen/qwen3.7-plus"
n_per_dataset = 1
datasets = [
  "data/traps/suite/true_effect",
  "data/traps/suite/pure_null"
]
""".strip(),
        encoding="utf-8",
    )

    report = run_calibration_pilot(
        root=tmp_path,
        config_path=Path("configs/pilots/calibration.toml"),
        fixture=True,
    )

    assert len(report.rows) == 2
    assert report.summary_csv.is_file()
    assert report.summary_json.is_file()
    assert {row["dataset_id"] for row in report.rows} == {"true_effect", "pure_null"}


def test_plasmode_freeze_audit_writes_manifest(tmp_path: Path) -> None:
    suite_dir = tmp_path / "data" / "traps" / "suite"
    dataset_dir = suite_dir / "pure_null"
    dataset_dir.mkdir(parents=True)
    n = 60
    pd.DataFrame(
        {
            "participant_id": list(range(n)),
            "synthetic_outcome": [0, 1] * (n // 2),
            "bmi": [25.0 + (index % 5) for index in range(n)],
            "survey_weight_mec_2yr": [1.0] * n,
            "survey_strata": [1] * n,
            "survey_psu": [1, 2] * (n // 2),
        }
    ).to_csv(dataset_dir / "dataset.csv", index=False)
    (dataset_dir / "public_codebook.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "dataset_id": "pure_null",
                "dataset_type": "pure_null",
                "outcome": "synthetic_outcome",
                "variables": [
                    {"name": "participant_id", "public_role": "identifier"},
                    {"name": "synthetic_outcome", "public_role": "outcome"},
                    {"name": "bmi", "public_role": "candidate_predictor"},
                    {"name": "survey_weight_mec_2yr", "public_role": "survey_design"},
                    {"name": "survey_strata", "public_role": "survey_design"},
                    {"name": "survey_psu", "public_role": "survey_design"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "hidden_truth_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "suite": "suite",
                "dataset_id": "pure_null",
                "dataset_type": "pure_null",
                "outcome": "synthetic_outcome",
                "dgp": {"model": "test"},
                "variable_truth": [
                    {"variable_name": "participant_id", "truth_status": "identifier", "allowed_claim_type": "none", "expected_behavior": "id"},
                    {"variable_name": "synthetic_outcome", "truth_status": "synthetic_outcome", "allowed_claim_type": "outcome", "expected_behavior": "outcome"},
                    {"variable_name": "bmi", "truth_status": "clean_null", "allowed_claim_type": "none", "expected_behavior": "null"},
                    {"variable_name": "survey_weight_mec_2yr", "truth_status": "survey_design", "allowed_claim_type": "none", "expected_behavior": "design"},
                    {"variable_name": "survey_strata", "truth_status": "survey_design", "allowed_claim_type": "none", "expected_behavior": "design"},
                    {"variable_name": "survey_psu", "truth_status": "survey_design", "allowed_claim_type": "none", "expected_behavior": "design"},
                ],
            }
        ),
        encoding="utf-8",
    )
    suite_manifest = suite_dir / "suite_manifest.json"
    suite_manifest.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "suite": "suite",
                "seed": 123,
                "datasets": [
                    {
                        "dataset_id": "pure_null",
                        "kind": "pure_null",
                        "directory": "data/traps/suite/pure_null",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_plasmode_freeze_audit(
        root=tmp_path,
        suite_manifest=Path("data/traps/suite/suite_manifest.json"),
        output_dir=Path("results/freeze"),
        overwrite=True,
    )

    assert report.freeze_manifest.is_file()
    assert report.qa_summary_csv.is_file()
    assert report.calibration_summary_csv.is_file()
    assert report.fairness_review_csv.is_file()
    assert report.calibration_report_md.is_file()
    assert report.fairness_report_md.is_file()
    assert report.reference_analysis_protocol_md.is_file()
    manifest = json.loads(report.freeze_manifest.read_text(encoding="utf-8"))
    assert manifest["datasets"][0]["artifacts"]["dataset_csv"]["sha256"]
    assert manifest["datasets"][0]["qa"]["qa_status"] == "pass"
    assert manifest["fairness_review_status"] == "pass"
