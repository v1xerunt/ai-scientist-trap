"""Create standard experiment directories and empty toy artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from trap_ai.config import load_toml, resolve_config_path
from trap_ai.randomness import DEFAULT_SEED, set_global_seed
from trap_ai.schemas import CLAIM_TABLE_COLUMNS, TRUTH_MANIFEST_COLUMNS


@dataclass(frozen=True)
class ScaffoldReport:
    root: Path
    experiment: str
    run_id: str
    seed: int
    directories: list[Path] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)


def create_toy_experiment(
    root: Path,
    config_path: Path,
    name: str | None = None,
    run_id: str | None = None,
    seed: int | None = None,
    overwrite: bool = False,
) -> ScaffoldReport:
    """Create the phase-1 toy experiment scaffold.

    The command is intentionally idempotent: directories are created if missing,
    while generated files are left untouched unless `overwrite` is true.
    """

    root = root.resolve()
    config = load_toml(resolve_config_path(root, config_path))
    experiment_config = config.get("experiment", {})
    path_config = config.get("paths", {})

    experiment = name or str(experiment_config.get("name", "toy"))
    selected_run_id = run_id or str(experiment_config.get("run_id", "run_0001"))
    selected_seed = int(seed if seed is not None else experiment_config.get("seed", DEFAULT_SEED))
    set_global_seed(selected_seed)

    created_dirs: list[Path] = []
    generated_files: list[Path] = []

    data_raw = root / str(path_config.get("raw_data", "data/raw")) / experiment
    data_processed = root / str(path_config.get("processed_data", "data/processed")) / experiment
    data_traps = root / str(path_config.get("trap_data", "data/traps")) / experiment
    run_root = root / str(path_config.get("runs", "runs")) / experiment / selected_run_id
    results_root = root / str(path_config.get("results", "results")) / experiment

    standard_dirs = [
        root / "configs",
        root / "configs" / "experiments",
        root / "configs" / "datasets",
        root / "configs" / "models",
        root / "src",
        root / "data",
        root / "data" / "raw",
        root / "data" / "processed",
        root / "data" / "traps",
        root / "runs",
        root / "results",
        data_raw,
        data_processed,
        data_traps,
    ]
    run_dirs = [run_root / item for item in config.get("run_artifacts", {}).get("directories", [])]
    result_dirs = [
        results_root / item for item in config.get("result_artifacts", {}).get("directories", [])
    ]

    for directory in [*standard_dirs, *run_dirs, *result_dirs]:
        _ensure_directory(directory)
        created_dirs.append(directory)

    generated_files.extend(
        [
            _write_csv_header(data_traps / "truth_manifest.csv", TRUTH_MANIFEST_COLUMNS, overwrite),
            _write_json(
                data_traps / "dataset_manifest.json",
                {
                    "experiment": experiment,
                    "dataset": experiment,
                    "kind": "empty_plasmode_trap_suite",
                    "seed": selected_seed,
                    "truth_manifest": _relative_to_root(data_traps / "truth_manifest.csv", root),
                    "status": "empty_toy_scaffold",
                },
                overwrite,
            ),
            _write_csv_header(
                run_root / "claim_tables" / "claims.csv",
                CLAIM_TABLE_COLUMNS,
                overwrite,
            ),
            _write_json(
                run_root / "metadata" / "run_manifest.json",
                {
                    "experiment": experiment,
                    "run_id": selected_run_id,
                    "seed": selected_seed,
                    "dataset_config": str(experiment_config.get("dataset_config", "")),
                    "model_config": str(experiment_config.get("model_config", "")),
                    "status": "empty_toy_scaffold",
                    "artifacts": {
                        "logs": "logs",
                        "code": "code",
                        "papers": "papers",
                        "claim_table": "claim_tables/claims.csv",
                    },
                },
                overwrite,
            ),
        ]
    )

    return ScaffoldReport(
        root=root,
        experiment=experiment,
        run_id=selected_run_id,
        seed=selected_seed,
        directories=_unique_paths(created_dirs),
        files=[path for path in generated_files if path is not None],
    )


def format_scaffold_report(report: ScaffoldReport) -> str:
    lines = [
        f"Toy experiment scaffold ready: {report.experiment}/{report.run_id}",
        f"Seed: {report.seed}",
        "",
        "Directories:",
    ]
    lines.extend(f"  - {_relative_to_root(path, report.root)}" for path in report.directories)
    lines.append("")
    lines.append("Generated files:")
    lines.extend(f"  - {_relative_to_root(path, report.root)}" for path in report.files)
    return "\n".join(lines)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    keep = path / ".gitkeep"
    if not keep.exists() and not any(path.iterdir()):
        keep.write_text("", encoding="utf-8")


def _write_csv_header(path: Path, columns: Iterable[str], overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(columns))
    return path


def _write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique
