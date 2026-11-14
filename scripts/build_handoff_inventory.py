"""Build a private inventory of the completed author-paper corpora."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODEX_ROOT = PROJECT_ROOT / "runs" / "codex_author_workspaces"

FIELDNAMES = [
    "paper_id",
    "author_source",
    "source_group",
    "dataset_id",
    "cohort_alias",
    "author_paradigm",
    "model_id",
    "run_id",
    "run_status",
    "paper_path",
    "paper_sha256",
    "word_count",
    "title",
    "claim_extracted",
    "hidden_truth_scored",
    "selection_ranked",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--codex-root", type=Path, default=DEFAULT_CODEX_ROOT)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/handoff_inventory.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/handoff_inventory_summary.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_from(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def normalized_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def paper_metadata(path: Path) -> tuple[str, str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = ""
    for line in text.splitlines():
        if line.strip().startswith("# "):
            title = line.strip()[2:].strip()
            break
    words = len(re.findall(r"\b[\w'-]+\b", text))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return title, digest, words


def model_from_manifest(manifest: dict[str, Any]) -> str:
    return str(manifest.get("model") or manifest.get("selected_model") or "")


def make_row(
    *,
    paper_id: str,
    author_source: str,
    source_group: str,
    dataset_id: str,
    cohort_alias: str,
    author_paradigm: str,
    model_id: str,
    run_id: str,
    run_status: str,
    run_dir: Path,
    hidden_truth_scored: bool,
    selection_ranked: bool,
) -> dict[str, str | int]:
    paper_path = run_dir / "paper" / "mini_paper.md"
    title, digest, word_count = paper_metadata(paper_path)
    return {
        "paper_id": paper_id,
        "author_source": author_source,
        "source_group": source_group,
        "dataset_id": dataset_id,
        "cohort_alias": cohort_alias,
        "author_paradigm": author_paradigm,
        "model_id": model_id,
        "run_id": run_id,
        "run_status": run_status,
        "paper_path": normalized_path(paper_path),
        "paper_sha256": digest,
        "word_count": word_count,
        "title": title,
        "claim_extracted": str((run_dir / "claim_tables" / "claims.csv").exists()).lower(),
        "hidden_truth_scored": str(hidden_truth_scored).lower(),
        "selection_ranked": str(selection_ranked).lower(),
    }


def collect_deepseek_remaining(project_root: Path) -> list[dict[str, str | int]]:
    base = project_root / "runs" / "deepseek_remaining_5x_v1" / "authors"
    rows: list[dict[str, str | int]] = []
    for manifest_path in sorted(base.glob("**/metadata/run_manifest.json")):
        manifest = load_json(manifest_path)
        run_dir = manifest_path.parent.parent
        if str(manifest.get("status", "")).lower() != "completed":
            continue
        if not (run_dir / "paper" / "mini_paper.md").is_file():
            continue
        dataset_id = Path(str(manifest.get("dataset_dir", ""))).name
        paradigm = str(manifest.get("author_paradigm", ""))
        run_id = str(manifest.get("run_id") or run_dir.name)
        rows.append(
            make_row(
                paper_id=f"ds-{dataset_id}-{paradigm}-{run_id}",
                author_source="openrouter_deepseek",
                source_group="deepseek_remaining_5x_v1",
                dataset_id=dataset_id,
                cohort_alias="",
                author_paradigm=paradigm,
                model_id=model_from_manifest(manifest) or "deepseek/deepseek-v4-pro",
                run_id=run_id,
                run_status="completed",
                run_dir=run_dir,
                hidden_truth_scored=False,
                selection_ranked=False,
            )
        )
    return rows


def collect_deepseek_leakage(project_root: Path) -> list[dict[str, str | int]]:
    summary_path = (
        project_root / "results" / "selection_enrichment_pilot_v1" / "author_generation_summary.csv"
    )
    rows: list[dict[str, str | int]] = []
    with summary_path.open(newline="", encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            run_dir = resolve_from(project_root, item["run_dir"])
            manifest = load_json(run_dir / "metadata" / "run_manifest.json")
            paradigm = item["paradigm"]
            run_id = item["run_id"]
            rows.append(
                make_row(
                    paper_id=f"ds-leakage_trap-{paradigm}-{run_id}",
                    author_source="openrouter_deepseek",
                    source_group="selection_enrichment_pilot_v1",
                    dataset_id="leakage_trap",
                    cohort_alias="",
                    author_paradigm=paradigm,
                    model_id=model_from_manifest(manifest) or "deepseek/deepseek-v4-pro",
                    run_id=run_id,
                    run_status="completed",
                    run_dir=run_dir,
                    hidden_truth_scored=True,
                    selection_ranked=True,
                )
            )
    return rows


def collect_codex(project_root: Path, codex_root: Path) -> list[dict[str, str | int]]:
    alias_map_path = (
        project_root / "results" / "codex_author_workspaces_oai_v1" / "private_dataset_alias_map.json"
    )
    alias_map = {
        item["cohort_alias"]: Path(item["source_dataset_dir"]).name
        for item in load_json(alias_map_path)
    }
    rows: list[dict[str, str | int]] = []
    with (codex_root / "WORKSPACE_INDEX.csv").open(newline="", encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            run_dir = (codex_root / item["workspace_dir"]).resolve()
            paper_path = run_dir / "paper" / "mini_paper.md"
            if not paper_path.is_file() or not paper_path.read_text(encoding="utf-8", errors="replace").strip():
                continue
            cohort = item["cohort_alias"]
            paradigm = item["paradigm"]
            run_id = item["run_id"]
            rows.append(
                make_row(
                    paper_id=f"codex-{cohort}-{paradigm}-{run_id}",
                    author_source="codex_skill",
                    source_group="codex_author_workspaces_oai_v1",
                    dataset_id=alias_map[cohort],
                    cohort_alias=cohort,
                    author_paradigm=paradigm,
                    model_id="codex_model_not_recorded",
                    run_id=run_id,
                    run_status="completed_inferred_from_artifacts",
                    run_dir=run_dir,
                    hidden_truth_scored=False,
                    selection_ranked=False,
                )
            )
    return rows


def counter_dict(rows: list[dict[str, str | int]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    codex_root = args.codex_root.resolve()
    rows = [
        *collect_deepseek_remaining(project_root),
        *collect_deepseek_leakage(project_root),
        *collect_codex(project_root, codex_root),
    ]
    rows.sort(key=lambda row: str(row["paper_id"]))

    output_csv = resolve_from(project_root, args.output_csv)
    summary_json = resolve_from(project_root, args.summary_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": "0.1",
        "visibility": "private_evaluation",
        "paper_count": len(rows),
        "by_author_source": counter_dict(rows, "author_source"),
        "by_dataset": counter_dict(rows, "dataset_id"),
        "by_paradigm": counter_dict(rows, "author_paradigm"),
        "claim_extracted": sum(row["claim_extracted"] == "true" for row in rows),
        "hidden_truth_scored": sum(row["hidden_truth_scored"] == "true" for row in rows),
        "selection_ranked": sum(row["selection_ranked"] == "true" for row in rows),
        "inventory_csv": normalized_path(output_csv),
        "codex_root": normalized_path(codex_root),
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
