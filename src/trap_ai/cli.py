"""Command line entry points for experiment scaffolding."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from trap_ai.agent_harness import (
    AUTHOR_PARADIGMS,
    DEFAULT_AUTHOR_PARADIGM,
    REVIEWER_STYLES,
    format_agent_harness_report,
    format_agent_run_inspection,
    format_claim_extraction_harness_report,
    format_openai_batch_report,
    format_reviewer_harness_report,
    format_selection_review_harness_report,
    create_openai_batch_for_runs,
    inspect_agent_run,
    retrieve_openai_batch_results,
    resume_interactive_agent_run,
    run_claim_extraction_for_run,
    run_agent_batch,
    run_interactive_agent_batch,
    run_reviewer_for_run,
    run_selection_review_for_runs,
)
from trap_ai.claim_scoring import format_claim_scoring_report, score_claim_table
from trap_ai.cost_tracking import format_openai_cost_report, summarize_openai_costs
from trap_ai.nhanes import download_nhanes, format_download_report
from trap_ai.pilot import format_pilot_report, run_calibration_pilot
from trap_ai.plasmode import generate_plasmode_suite, format_plasmode_report
from trap_ai.scaffold import create_toy_experiment, format_scaffold_report
from trap_ai.validation import format_freeze_audit_report, run_plasmode_freeze_audit


DEFAULT_CONFIG = Path("configs") / "experiments" / "toy.toml"
DEFAULT_NHANES_CONFIG = Path("configs") / "datasets" / "nhanes_latest_2021_2023.toml"
DEFAULT_PLASMODE_CONFIG = Path("configs") / "datasets" / "plasmode_nhanes_latest.toml"
DEFAULT_AGENT_CONFIG = Path("configs") / "agents" / "openrouter_scientist.toml"
DEFAULT_EXTRACTOR_CONFIG = Path("configs") / "evaluators" / "openai_claim_extractor.toml"
DEFAULT_REVIEWER_CONFIG = Path("configs") / "reviewers" / "openai_reviewer.toml"
DEFAULT_PROMPTS_CONFIG = Path("configs") / "prompts" / "scientist_prompts.toml"
DEFAULT_PILOT_CONFIG = Path("configs") / "pilots" / "calibration_001.toml"
DEFAULT_FREEZE_AUDIT_CONFIG = Path("configs") / "validation" / "plasmode_freeze.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trap-ai",
        description="Utilities for reproducible AI scientist trap-suite experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    toy = subparsers.add_parser(
        "init-toy",
        help="Create the empty toy experiment directory structure.",
    )
    toy.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root where directories should be created. Defaults to cwd.",
    )
    toy.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Experiment TOML config path, relative to root unless absolute.",
    )
    toy.add_argument(
        "--name",
        default=None,
        help="Experiment name override. Defaults to the config value.",
    )
    toy.add_argument(
        "--run-id",
        default=None,
        help="Run id override. Defaults to the config value.",
    )
    toy.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override. Defaults to the config value.",
    )
    toy.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite generated manifest files if they already exist.",
    )

    nhanes = subparsers.add_parser(
        "download-nhanes",
        help="Download the configured public-use NHANES source files.",
    )
    nhanes.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root where files should be downloaded. Defaults to cwd.",
    )
    nhanes.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_NHANES_CONFIG,
        help="NHANES dataset TOML config path, relative to root unless absolute.",
    )
    nhanes.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even when they already exist.",
    )
    nhanes.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Per-file download timeout in seconds.",
    )
    nhanes.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return success even if individual files fail to download.",
    )

    plasmode = subparsers.add_parser(
        "generate-plasmode",
        help="Generate synthetic outcomes and trap datasets from NHANES covariates.",
    )
    plasmode.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root where files should be generated. Defaults to cwd.",
    )
    plasmode.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PLASMODE_CONFIG,
        help="Plasmode TOML config path, relative to root unless absolute.",
    )
    plasmode.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite generated datasets if they already exist.",
    )

    scorer = subparsers.add_parser(
        "score-claims",
        help="Annotate a claim table against a hidden truth manifest.",
    )
    scorer.add_argument(
        "--claims",
        type=Path,
        required=True,
        help="Input claim table CSV.",
    )
    scorer.add_argument(
        "--truth",
        type=Path,
        required=True,
        help="Hidden truth manifest JSON or CSV.",
    )
    scorer.add_argument(
        "--codebook",
        type=Path,
        default=None,
        help="Optional public codebook JSON used only to resolve natural-language variable labels.",
    )
    scorer.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Scored claim table CSV. Defaults to CLAIMS_STEM_scored.csv.",
    )
    scorer.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Scoring summary JSON. Defaults next to the scored CSV.",
    )

    agent = subparsers.add_parser(
        "run-agent",
        help="Run or dry-run OpenRouter AI scientist jobs for one dataset.",
    )
    agent.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root. Defaults to cwd.",
    )
    agent.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "traps" / "plasmode_nhanes_latest" / "true_effect",
        help="Dataset directory containing dataset.csv and public_codebook.json.",
    )
    agent.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_AGENT_CONFIG,
        help="Agent/OpenRouter TOML config.",
    )
    agent.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_CONFIG,
        help="Prompt regime TOML config.",
    )
    agent.add_argument(
        "--prompt-regime",
        default="neutral",
        help="Prompt regime: neutral, publishable, or strict.",
    )
    agent.add_argument(
        "--model",
        default=None,
        help="OpenRouter model id override.",
    )
    agent.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of independent agent runs to create.",
    )
    agent.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Run root override. Defaults to the config value.",
    )
    agent.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Dotenv file containing OPENROUTER_API_KEY.",
    )
    agent.add_argument(
        "--dry-run",
        action="store_true",
        help="Create input packages and request payloads without calling OpenRouter.",
    )
    agent.add_argument(
        "--execute-code",
        action="store_true",
        help="After a live model response, run generated analysis.py and capture stdout/stderr.",
    )
    agent.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace run directories if the generated run id already exists.",
    )

    loop = subparsers.add_parser(
        "run-agent-loop",
        help="Run or test the author AI scientist loop without evaluator/reviewer steps.",
    )
    loop.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root. Defaults to cwd.",
    )
    loop.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "traps" / "plasmode_nhanes_latest" / "true_effect",
        help="Dataset directory containing dataset.csv and public_codebook.json.",
    )
    loop.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_AGENT_CONFIG,
        help="Agent/OpenRouter TOML config.",
    )
    loop.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_CONFIG,
        help="Prompt regime TOML config.",
    )
    loop.add_argument(
        "--prompt-regime",
        default="neutral",
        help="Prompt regime: neutral, publishable, or strict.",
    )
    loop.add_argument(
        "--paradigm",
        choices=sorted(AUTHOR_PARADIGMS),
        default=None,
        help=f"Author agent workflow paradigm. Defaults to config or {DEFAULT_AUTHOR_PARADIGM}.",
    )
    loop.add_argument(
        "--model",
        default=None,
        help="OpenRouter model id override.",
    )
    loop.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of independent interactive runs to create.",
    )
    loop.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Run root override. Defaults to the config value.",
    )
    loop.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Dotenv file containing OPENROUTER_API_KEY.",
    )
    loop.add_argument(
        "--dry-run",
        action="store_true",
        help="Create prompts and directories without calling OpenRouter or executing generated code.",
    )
    loop.add_argument(
        "--fixture",
        action="store_true",
        help="Use deterministic local fake LLM responses and execute fixture code.",
    )
    loop.add_argument(
        "--max-code-repairs",
        type=int,
        default=None,
        help="Maximum number of agent repair turns after failed code execution. Defaults to config.",
    )
    loop.add_argument(
        "--max-supplemental-analyses",
        type=int,
        default=None,
        help="Maximum number of scientific supplemental analysis turns after sufficiency review. Defaults to config.",
    )
    loop.add_argument(
        "--max-sufficiency-reviews",
        type=int,
        default=None,
        help="Maximum number of analysis sufficiency review turns. Defaults to config.",
    )
    loop.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of author runs to execute concurrently.",
    )
    loop.add_argument(
        "--no-sufficiency-review",
        dest="skip_sufficiency_review",
        action="store_true",
        default=None,
        help="Skip the analysis sufficiency review turn before final paper writing.",
    )
    loop.add_argument(
        "--no-evidence-audit",
        dest="skip_sufficiency_review",
        action="store_true",
        default=None,
        help=argparse.SUPPRESS,
    )
    loop.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace run directories if the generated run id already exists.",
    )

    extractor = subparsers.add_parser(
        "extract-claims",
        help="Run evaluator LLM claim extraction for an existing author run.",
    )
    extractor.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root. Defaults to cwd.",
    )
    extractor.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed author run directory.",
    )
    extractor.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXTRACTOR_CONFIG,
        help="Evaluator LLM TOML config.",
    )
    extractor.add_argument(
        "--model",
        default=None,
        help="Evaluator model id override.",
    )
    extractor.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Dotenv file containing the evaluator provider API key.",
    )
    extractor.add_argument(
        "--dry-run",
        action="store_true",
        help="Create evaluator request payload without calling the LLM provider.",
    )
    extractor.add_argument(
        "--fixture",
        action="store_true",
        help="Use deterministic local fake evaluator response.",
    )
    extractor.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing evaluator claim table.",
    )

    reviewer = subparsers.add_parser(
        "review-paper",
        help="Run an optional statistical/methods reviewer LLM for an existing author mini-paper.",
    )
    reviewer.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root. Defaults to cwd.",
    )
    reviewer.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed author run directory.",
    )
    reviewer.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_REVIEWER_CONFIG,
        help="Reviewer LLM TOML config.",
    )
    reviewer.add_argument(
        "--style",
        choices=sorted(REVIEWER_STYLES),
        default="statistical",
        help="Reviewer prompt style. Defaults to statistical.",
    )
    reviewer.add_argument(
        "--model",
        default=None,
        help="Reviewer model id override.",
    )
    reviewer.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Dotenv file containing the reviewer provider API key.",
    )
    reviewer.add_argument(
        "--dry-run",
        action="store_true",
        help="Create reviewer request payload without calling the LLM provider.",
    )
    reviewer.add_argument(
        "--fixture",
        action="store_true",
        help="Use deterministic local fake reviewer response.",
    )
    reviewer.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing reviewer output.",
    )

    selection = subparsers.add_parser(
        "review-selection",
        help="Rank a batch of completed mini-papers using title and abstract only.",
    )
    selection.add_argument("--root", type=Path, default=Path.cwd(), help="Project root. Defaults to cwd.")
    selection.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Root containing completed author run directories.",
    )
    selection.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_REVIEWER_CONFIG,
        help="Reviewer LLM TOML config.",
    )
    selection.add_argument("--model", default=None, help="Reviewer model id override.")
    selection.add_argument("--env-file", type=Path, default=Path(".env"), help="Dotenv file.")
    selection.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "selection_reviews",
        help="Directory for selection-review artifacts.",
    )
    selection.add_argument(
        "--max-papers",
        type=int,
        default=20,
        help="Maximum number of completed papers to include in one ranking request.",
    )
    selection.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Number of papers per ranking request. Defaults to all selected papers in one request.",
    )
    selection.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Deterministically shuffle completed papers before grouping.",
    )
    selection.add_argument("--dry-run", action="store_true", help="Create request artifacts without calling the LLM.")
    selection.add_argument("--fixture", action="store_true", help="Use deterministic local fake reviewer response.")
    selection.add_argument("--overwrite", action="store_true", help="Replace local selection-review artifacts if needed.")

    create_batch = subparsers.add_parser(
        "create-openai-batch",
        help="Create and submit an OpenAI Batch API job for completed author runs.",
    )
    create_batch.add_argument("--root", type=Path, default=Path.cwd(), help="Project root. Defaults to cwd.")
    create_batch.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Root containing completed author run directories.",
    )
    create_batch.add_argument(
        "--task",
        choices=["claim_extraction", "reviewer"],
        required=True,
        help="Batch task to submit.",
    )
    create_batch.add_argument(
        "--config",
        type=Path,
        default=None,
        help="OpenAI evaluator/reviewer config. Defaults by task.",
    )
    create_batch.add_argument("--model", default=None, help="Model override.")
    create_batch.add_argument(
        "--style",
        choices=sorted(REVIEWER_STYLES),
        default="statistical",
        help="Reviewer style for reviewer batches.",
    )
    create_batch.add_argument("--env-file", type=Path, default=Path(".env"), help="Dotenv file.")
    create_batch.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "openai_batches",
        help="Directory for batch JSONL and manifests.",
    )
    create_batch.add_argument("--overwrite", action="store_true", help="Replace local batch manifest if needed.")

    retrieve_batch = subparsers.add_parser(
        "retrieve-openai-batch",
        help="Retrieve an OpenAI batch and write outputs back to run directories.",
    )
    retrieve_batch.add_argument("--root", type=Path, default=Path.cwd(), help="Project root. Defaults to cwd.")
    retrieve_batch.add_argument("--manifest", type=Path, required=True, help="Batch manifest JSON.")
    retrieve_batch.add_argument("--env-file", type=Path, default=Path(".env"), help="Dotenv file.")
    retrieve_batch.add_argument("--wait", action="store_true", help="Poll until the batch reaches a terminal state.")
    retrieve_batch.add_argument("--poll-seconds", type=float, default=30.0, help="Polling interval when --wait is set.")
    retrieve_batch.add_argument(
        "--max-wait-seconds",
        type=float,
        default=3600.0,
        help="Maximum polling time when --wait is set.",
    )
    retrieve_batch.add_argument("--no-overwrite", action="store_true", help="Do not replace existing outputs.")

    costs = subparsers.add_parser(
        "summarize-openai-costs",
        help="Summarize OpenAI token usage and estimated cost from saved response artifacts.",
    )
    costs.add_argument("--root", type=Path, default=Path.cwd(), help="Project root. Defaults to cwd.")
    costs.add_argument(
        "--scan-root",
        type=Path,
        action="append",
        default=None,
        help="Directory to scan. Can be provided multiple times. Defaults to runs and results.",
    )
    costs.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "cost_tracking",
        help="Directory for OpenAI usage ledger artifacts.",
    )
    costs.add_argument("--overwrite", action="store_true", help="Allow overwriting timestamp-colliding outputs.")

    inspect = subparsers.add_parser(
        "inspect-agent-run",
        help="Summarize an author run directory without calling any model.",
    )
    inspect.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Author run directory to inspect.",
    )
    inspect.add_argument(
        "--tail-chars",
        type=int,
        default=2400,
        help="Characters to show from stdout/stderr and mini-paper tails.",
    )

    resume = subparsers.add_parser(
        "resume-agent-run",
        help="Resume a partially completed author run from its run directory.",
    )
    resume.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root. Defaults to cwd.",
    )
    resume.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Author run directory to resume.",
    )
    resume.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_AGENT_CONFIG,
        help="OpenRouter TOML config.",
    )
    resume.add_argument(
        "--model",
        default=None,
        help="OpenRouter model id override. Defaults to the original run model when available.",
    )
    resume.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Dotenv file containing OPENROUTER_API_KEY.",
    )
    resume.add_argument(
        "--dry-run",
        action="store_true",
        help="Create resume requests without calling OpenRouter.",
    )
    resume.add_argument(
        "--max-code-repairs",
        type=int,
        default=None,
        help="Total technical repair limit for the resumed run. Defaults to config.",
    )
    resume.add_argument(
        "--max-supplemental-analyses",
        type=int,
        default=None,
        help="Total supplemental-analysis limit for the resumed run. Defaults to config.",
    )
    resume.add_argument(
        "--max-sufficiency-reviews",
        type=int,
        default=None,
        help="Total analysis sufficiency review limit for the resumed run. Defaults to config.",
    )

    pilot = subparsers.add_parser(
        "run-calibration-pilot",
        help="Run the configured author-only mini calibration pilot.",
    )
    pilot.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root. Defaults to cwd.",
    )
    pilot.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PILOT_CONFIG,
        help="Calibration pilot TOML config.",
    )
    pilot.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Dotenv file containing OPENROUTER_API_KEY.",
    )
    pilot.add_argument(
        "--dry-run",
        action="store_true",
        help="Create run packages without calling OpenRouter.",
    )
    pilot.add_argument(
        "--fixture",
        action="store_true",
        help="Use deterministic local fake LLM responses and execute fixture code.",
    )
    pilot.add_argument(
        "--max-code-repairs",
        type=int,
        default=None,
        help="Maximum number of agent repair turns after failed code execution. Defaults to config.",
    )
    pilot.add_argument(
        "--max-supplemental-analyses",
        type=int,
        default=None,
        help="Maximum number of scientific supplemental analysis turns after sufficiency review. Defaults to config.",
    )
    pilot.add_argument(
        "--max-sufficiency-reviews",
        type=int,
        default=None,
        help="Maximum number of analysis sufficiency review turns. Defaults to config.",
    )
    pilot.add_argument(
        "--no-sufficiency-review",
        dest="skip_sufficiency_review",
        action="store_true",
        default=None,
        help="Skip the analysis sufficiency review turn before final paper writing.",
    )
    pilot.add_argument(
        "--no-evidence-audit",
        dest="skip_sufficiency_review",
        action="store_true",
        default=None,
        help=argparse.SUPPRESS,
    )
    pilot.add_argument(
        "--paradigm",
        choices=sorted(AUTHOR_PARADIGMS),
        default=None,
        help="Author agent workflow paradigm override. Defaults to the pilot config or iterative_coding.",
    )
    pilot.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace run directories if generated run ids collide.",
    )

    freeze = subparsers.add_parser(
        "freeze-plasmode",
        help="Freeze generated plasmode artifacts and write calibration diagnostics.",
    )
    freeze.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root. Defaults to cwd.",
    )
    freeze.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_FREEZE_AUDIT_CONFIG,
        help="Freeze/calibration TOML config.",
    )
    freeze.add_argument(
        "--suite-manifest",
        type=Path,
        default=None,
        help="Suite manifest override. Defaults to config.",
    )
    freeze.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory override. Defaults to config.",
    )
    freeze.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing freeze and calibration outputs.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init-toy":
        report = create_toy_experiment(
            root=args.root,
            config_path=args.config,
            name=args.name,
            run_id=args.run_id,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(format_scaffold_report(report))
        return 0

    if args.command == "download-nhanes":
        report = download_nhanes(
            root=args.root,
            config_path=args.config,
            overwrite=args.overwrite,
            timeout=args.timeout,
        )
        print(format_download_report(report))
        return 0 if args.allow_failures or report.failed == 0 else 1

    if args.command == "generate-plasmode":
        report = generate_plasmode_suite(
            root=args.root,
            config_path=args.config,
            overwrite=args.overwrite,
        )
        print(format_plasmode_report(report))
        return 0

    if args.command == "score-claims":
        report = score_claim_table(
            claims_path=args.claims,
            truth_manifest_path=args.truth,
            codebook_path=args.codebook,
            output_path=args.out,
            summary_path=args.summary,
        )
        print(format_claim_scoring_report(report))
        return 0

    if args.command == "run-agent":
        report = run_agent_batch(
            root=args.root,
            dataset_dir=args.dataset_dir,
            config_path=args.config,
            prompts_path=args.prompts,
            prompt_regime=args.prompt_regime,
            n_runs=args.n,
            model=args.model,
            run_root=args.run_root,
            env_file=args.env_file,
            dry_run=args.dry_run,
            execute_code=args.execute_code,
            overwrite=args.overwrite,
        )
        print(format_agent_harness_report(report))
        return 0

    if args.command == "run-agent-loop":
        report = run_interactive_agent_batch(
            root=args.root,
            dataset_dir=args.dataset_dir,
            config_path=args.config,
            prompts_path=args.prompts,
            prompt_regime=args.prompt_regime,
            n_runs=args.n,
            model=args.model,
            run_root=args.run_root,
            env_file=args.env_file,
            dry_run=args.dry_run,
            fixture=args.fixture,
            max_code_repairs=args.max_code_repairs,
            evidence_audit=None if args.skip_sufficiency_review is None else not args.skip_sufficiency_review,
            max_supplemental_analyses=args.max_supplemental_analyses,
            max_sufficiency_reviews=args.max_sufficiency_reviews,
            author_paradigm=args.paradigm,
            parallel_workers=args.parallel_workers,
            overwrite=args.overwrite,
        )
        print(format_agent_harness_report(report))
        return 0

    if args.command == "extract-claims":
        report = run_claim_extraction_for_run(
            root=args.root,
            run_dir=args.run_dir,
            config_path=args.config,
            model=args.model,
            env_file=args.env_file,
            dry_run=args.dry_run,
            fixture=args.fixture,
            overwrite=args.overwrite,
        )
        print(format_claim_extraction_harness_report(report))
        return 0

    if args.command == "review-paper":
        report = run_reviewer_for_run(
            root=args.root,
            run_dir=args.run_dir,
            config_path=args.config,
            reviewer_style=args.style,
            model=args.model,
            env_file=args.env_file,
            dry_run=args.dry_run,
            fixture=args.fixture,
            overwrite=args.overwrite,
        )
        print(format_reviewer_harness_report(report))
        return 0

    if args.command == "review-selection":
        report = run_selection_review_for_runs(
            root=args.root,
            run_root=args.run_root,
            config_path=args.config,
            output_dir=args.output_dir,
            model=args.model,
            env_file=args.env_file,
            dry_run=args.dry_run,
            fixture=args.fixture,
            max_papers=args.max_papers,
            group_size=args.group_size,
            shuffle_seed=args.shuffle_seed,
            overwrite=args.overwrite,
        )
        print(format_selection_review_harness_report(report))
        return 0

    if args.command == "create-openai-batch":
        config = args.config
        if config is None:
            config = DEFAULT_EXTRACTOR_CONFIG if args.task == "claim_extraction" else DEFAULT_REVIEWER_CONFIG
        report = create_openai_batch_for_runs(
            root=args.root,
            run_root=args.run_root,
            config_path=config,
            task=args.task,
            output_dir=args.output_dir,
            env_file=args.env_file,
            reviewer_style=args.style,
            model=args.model,
            overwrite=args.overwrite,
        )
        print(format_openai_batch_report(report))
        return 0

    if args.command == "retrieve-openai-batch":
        report = retrieve_openai_batch_results(
            root=args.root,
            manifest_path=args.manifest,
            env_file=args.env_file,
            wait=args.wait,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_seconds,
            overwrite=not args.no_overwrite,
        )
        print(format_openai_batch_report(report))
        return 0

    if args.command == "summarize-openai-costs":
        report = summarize_openai_costs(
            root=args.root,
            output_dir=args.output_dir,
            scan_roots=args.scan_root,
            overwrite=args.overwrite,
        )
        print(format_openai_cost_report(report))
        return 0

    if args.command == "inspect-agent-run":
        report = inspect_agent_run(
            run_dir=args.run_dir,
            tail_chars=args.tail_chars,
        )
        print(format_agent_run_inspection(report))
        return 0

    if args.command == "resume-agent-run":
        result = resume_interactive_agent_run(
            root=args.root,
            run_dir=args.run_dir,
            config_path=args.config,
            model=args.model,
            env_file=args.env_file,
            dry_run=args.dry_run,
            max_code_repairs=args.max_code_repairs,
            max_supplemental_analyses=args.max_supplemental_analyses,
            max_sufficiency_reviews=args.max_sufficiency_reviews,
        )
        print(f"Resumed agent run: {result.status}")
        print(f"Run dir: {result.run_dir}")
        print(f"Mini-paper: {result.mini_paper_path}")
        return 0

    if args.command == "run-calibration-pilot":
        report = run_calibration_pilot(
            root=args.root,
            config_path=args.config,
            env_file=args.env_file,
            dry_run=args.dry_run,
            fixture=args.fixture,
            max_code_repairs=args.max_code_repairs,
            evidence_audit=None if args.skip_sufficiency_review is None else not args.skip_sufficiency_review,
            max_supplemental_analyses=args.max_supplemental_analyses,
            max_sufficiency_reviews=args.max_sufficiency_reviews,
            author_paradigm=args.paradigm,
            overwrite=args.overwrite,
        )
        print(format_pilot_report(report))
        return 0

    if args.command == "freeze-plasmode":
        report = run_plasmode_freeze_audit(
            root=args.root,
            config_path=args.config,
            suite_manifest=args.suite_manifest,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        print(format_freeze_audit_report(report))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
