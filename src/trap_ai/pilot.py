"""Pilot orchestration utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trap_ai.agent_harness import DEFAULT_AUTHOR_PARADIGM, run_interactive_agent_batch
from trap_ai.config import load_toml, resolve_config_path


@dataclass(frozen=True)
class PilotReport:
    root: Path
    name: str
    run_root: Path
    summary_csv: Path
    summary_json: Path
    rows: list[dict[str, Any]]


def run_calibration_pilot(
    root: Path,
    config_path: Path,
    env_file: Path | None = None,
    dry_run: bool = False,
    fixture: bool = False,
    max_code_repairs: int | None = None,
    evidence_audit: bool | None = None,
    max_supplemental_analyses: int | None = None,
    max_sufficiency_reviews: int | None = None,
    author_paradigm: str | None = None,
    overwrite: bool = False,
) -> PilotReport:
    """Run the configured mini calibration pilot."""

    root = root.resolve()
    config = load_toml(resolve_config_path(root, config_path))
    pilot = config["pilot"]
    name = str(pilot.get("name", "calibration_pilot"))
    run_root = root / str(pilot.get("run_root", f"runs/pilots/{name}"))
    agent_config = Path(str(pilot["agent_config"]))
    prompts_config = Path(str(pilot["prompts_config"]))
    prompt_regime = str(pilot.get("prompt_regime", "neutral"))
    model = str(pilot.get("model", "qwen/qwen3.7-plus"))
    n_per_dataset = int(pilot.get("n_per_dataset", 1))
    selected_author_paradigm = str(pilot.get("author_paradigm", DEFAULT_AUTHOR_PARADIGM))
    if author_paradigm is not None:
        selected_author_paradigm = author_paradigm
    selected_max_code_repairs = (
        int(pilot["max_code_repairs"]) if max_code_repairs is None and "max_code_repairs" in pilot else max_code_repairs
    )
    configured_sufficiency_review = pilot.get("analysis_sufficiency_review", pilot.get("evidence_audit"))
    selected_sufficiency_review = (
        bool(configured_sufficiency_review)
        if evidence_audit is None and configured_sufficiency_review is not None
        else evidence_audit
    )
    selected_max_supplemental_analyses = (
        int(pilot["max_supplemental_analyses"])
        if max_supplemental_analyses is None and "max_supplemental_analyses" in pilot
        else max_supplemental_analyses
    )
    selected_max_sufficiency_reviews = (
        int(pilot["max_sufficiency_reviews"])
        if max_sufficiency_reviews is None and "max_sufficiency_reviews" in pilot
        else max_sufficiency_reviews
    )

    rows: list[dict[str, Any]] = []
    for dataset_value in pilot.get("datasets", []):
        dataset_dir = Path(str(dataset_value))
        report = run_interactive_agent_batch(
            root=root,
            dataset_dir=dataset_dir,
            config_path=agent_config,
            prompts_path=prompts_config,
            prompt_regime=prompt_regime,
            n_runs=n_per_dataset,
            model=model,
            run_root=Path(str(pilot.get("run_root", f"runs/pilots/{name}"))),
            env_file=env_file,
            dry_run=dry_run,
            fixture=fixture,
            max_code_repairs=selected_max_code_repairs,
            evidence_audit=selected_sufficiency_review,
            max_supplemental_analyses=selected_max_supplemental_analyses,
            max_sufficiency_reviews=selected_max_sufficiency_reviews,
            author_paradigm=selected_author_paradigm,
            overwrite=overwrite,
        )
        for run in report.runs:
            row = summarize_run_row(run.run_dir, dataset_dir)
            rows.append(row)
            write_progress(run_root / "pilot_progress.jsonl", row)

    summary_csv = run_root / "pilot_summary.csv"
    summary_json = run_root / "pilot_summary.json"
    write_summary_csv(summary_csv, rows)
    summary_json.write_text(json.dumps({"pilot": name, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    return PilotReport(
        root=root,
        name=name,
        run_root=run_root,
        summary_csv=summary_csv,
        summary_json=summary_json,
        rows=rows,
    )


def summarize_run_row(
    run_dir: Path,
    dataset_dir: Path,
) -> dict[str, Any]:
    run_manifest = load_json_if_exists(run_dir / "metadata" / "run_manifest.json")
    execution_manifest = load_json_if_exists(run_dir / "metadata" / "execution_manifest.json")
    return {
        "dataset_id": Path(dataset_dir).name,
        "dataset_dir": str(dataset_dir).replace("\\", "/"),
        "run_id": run_dir.name,
        "run_dir": str(run_dir).replace("\\", "/"),
        "status": run_manifest.get("status", ""),
        "author_paradigm": run_manifest.get("author_paradigm", ""),
        "execution_returncode": execution_manifest.get("returncode", ""),
        "execution_duration_seconds": execution_manifest.get("duration_seconds", ""),
        "generated_file_count": len(execution_manifest.get("generated_files", [])),
        "mini_paper": str((run_dir / "paper" / "mini_paper.md")).replace("\\", "/"),
    }


def format_pilot_report(report: PilotReport) -> str:
    lines = [
        f"Author calibration pilot complete: {report.name}",
        f"Run root: {_relative_to_root(report.run_root, report.root)}",
        f"Summary CSV: {_relative_to_root(report.summary_csv, report.root)}",
        f"Summary JSON: {_relative_to_root(report.summary_json, report.root)}",
        "",
        "Runs:",
    ]
    for row in report.rows:
        lines.append(
            "  - "
            f"{row['dataset_id']}: status={row['status']} "
            f"exec={row['execution_returncode']} "
            f"generated_files={row['generated_file_count']}"
        )
    return "\n".join(lines)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_id",
        "dataset_dir",
        "run_id",
        "run_dir",
        "status",
        "author_paradigm",
        "execution_returncode",
        "execution_duration_seconds",
        "generated_file_count",
        "mini_paper",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_progress(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)
