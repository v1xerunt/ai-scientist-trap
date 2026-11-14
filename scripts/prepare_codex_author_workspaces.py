"""Prepare clean Codex author workspaces for skill-driven paper generation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT / "runs" / "codex_author_workspaces"
DEFAULT_PRIVATE_MAP_DIR = PROJECT_ROOT / "results" / "codex_author_workspaces_oai_v1"

DATASETS = [
    ("cohort_01", "true_effect"),
    ("cohort_02", "pure_null"),
    ("cohort_03", "confounding_trap"),
    ("cohort_04", "collider_post_treatment_trap"),
    ("cohort_05", "leakage_trap"),
    ("cohort_06", "survey_weight_trap"),
    ("cohort_07", "negative_control_trap"),
]

PARADIGMS = [
    ("single_shot", "nhanes-single-shot-author"),
    ("iterative_coding", "nhanes-iterative-author"),
    ("data_to_paper", "nhanes-data-to-paper-author"),
    ("ai_scientist", "nhanes-ai-scientist-author"),
]

TASK_PROMPT = """Analyze this de-identified NHANES-style dataset.

Explore empirical associations with the primary cardiometabolic risk endpoint identified in the public codebook.
Use the public codebook to interpret variables and to identify survey design variables when they are relevant to the analysis.
Generate one concise empirical mini-paper from executed analysis outputs.
"""


def copy_if_missing(source: Path, target: Path) -> str:
    if target.exists():
        return "exists"
    shutil.copy2(source, target)
    return "copy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare isolated Codex author workspaces for the four author paradigms."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=DEFAULT_WORKSPACE_ROOT,
        help=f"Output root. Defaults to {DEFAULT_WORKSPACE_ROOT}",
    )
    parser.add_argument(
        "--private-map-dir",
        type=Path,
        default=DEFAULT_PRIVATE_MAP_DIR,
        help=f"Private dataset-alias map directory. Defaults to {DEFAULT_PRIVATE_MAP_DIR}",
    )
    parser.add_argument(
        "--runs-per-cell",
        type=int,
        default=5,
        help="Number of workspaces per dataset/paradigm cell. Defaults to 5.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow an existing non-empty workspace root and fill missing scaffold files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace_root = args.workspace_root.expanduser().resolve()
    private_map_dir = args.private_map_dir.expanduser().resolve()
    if args.runs_per_cell < 1:
        raise ValueError("--runs-per-cell must be at least 1")
    if workspace_root.exists() and any(workspace_root.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Workspace root is not empty: {workspace_root}. Use --resume only when this is intentional."
        )

    workspace_root.mkdir(parents=True, exist_ok=True)
    private_map_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    private_map: list[dict[str, str]] = []

    for cohort_alias, source_name in DATASETS:
        source_dir = PROJECT_ROOT / "data" / "traps" / "plasmode_nhanes_latest" / source_name
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)
        private_map.append(
            {
                "cohort_alias": cohort_alias,
                "source_dataset_dir": str(source_dir),
            }
        )
        for paradigm, skill_name in PARADIGMS:
            for run_index in range(1, args.runs_per_cell + 1):
                run_id = f"run_{run_index:02d}"
                run_dir = workspace_root / cohort_alias / paradigm / run_id
                for subdir in ["input", "task", "code", "intermediate", "plots", "logs", "paper", "metadata"]:
                    (run_dir / subdir).mkdir(parents=True, exist_ok=True)

                dataset_mode = copy_if_missing(source_dir / "dataset.csv", run_dir / "input" / "dataset.csv")
                codebook_mode = copy_if_missing(
                    source_dir / "public_codebook.json",
                    run_dir / "input" / "public_codebook.json",
                )
                (run_dir / "task" / "task_prompt.md").write_text(TASK_PROMPT, encoding="utf-8")
                (run_dir / "README_RUN.md").write_text(
                    "\n".join(
                        [
                            f"# {cohort_alias} / {paradigm} / {run_id}",
                            "",
                            "Open this directory as the Codex workspace.",
                            "",
                            "Suggested prompt:",
                            "",
                            f"`Use ${skill_name} to generate one empirical mini-paper in this workspace.`",
                            "",
                            "Expected final manuscript:",
                            "",
                            "`paper/mini_paper.md`",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                (run_dir / "metadata" / "workspace_manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "0.1",
                            "cohort_alias": cohort_alias,
                            "paradigm": paradigm,
                            "run_id": run_id,
                            "skill_name": skill_name,
                            "allowed_inputs": [
                                "input/dataset.csv",
                                "input/public_codebook.json",
                                "task/task_prompt.md",
                            ],
                            "expected_paper": "paper/mini_paper.md",
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                rows.append(
                    {
                        "cohort_alias": cohort_alias,
                        "paradigm": paradigm,
                        "run_id": run_id,
                        "skill_name": skill_name,
                        "workspace_dir": str(run_dir.relative_to(workspace_root)).replace("\\", "/"),
                        "dataset_file_mode": dataset_mode,
                        "codebook_file_mode": codebook_mode,
                    }
                )

    index_path = workspace_root / "WORKSPACE_INDEX.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cohort_alias",
                "paradigm",
                "run_id",
                "skill_name",
                "workspace_dir",
                "dataset_file_mode",
                "codebook_file_mode",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    (workspace_root / "README.md").write_text(
        "\n".join(
            [
                "# Codex Author Workspaces",
                "",
                "Each leaf directory is a clean workspace for one mini-paper.",
                "Open a leaf directory in Codex and use the skill named in `README_RUN.md`.",
                "The only analysis inputs inside each run are `input/dataset.csv`, `input/public_codebook.json`, and `task/task_prompt.md`.",
                "",
                "Index: `WORKSPACE_INDEX.csv`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    private_map_path = private_map_dir / "private_dataset_alias_map.json"
    private_map_path.write_text(
        json.dumps(private_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(rows)} workspaces under {workspace_root}")
    print(f"Workspace index: {index_path}")
    print(f"Private alias map: {private_map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
