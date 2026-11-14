"""Stage paper-only evaluator workspaces without modifying author artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=PROJECT_ROOT / "results" / "handoff_inventory.csv",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--author-source",
        action="append",
        help="Optional source filter; repeat for multiple values.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Optional dataset filter; repeat for multiple values.",
    )
    parser.add_argument(
        "--include-evaluated",
        action="store_true",
        help="Include papers already marked claim_extracted in the inventory.",
    )
    parser.add_argument("--max-papers", type=int, default=0, help="Optional cap; 0 means all selected papers.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow an existing non-empty output root and fill missing files.",
    )
    return parser.parse_args()


def copy_if_missing(source: Path, target: Path) -> str:
    if target.exists():
        return "exists"
    shutil.copy2(source, target)
    return "copy"


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> int:
    args = parse_args()
    inventory = args.inventory.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output root is not empty: {output_root}. Use --resume only when intentional."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    with inventory.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    selected = []
    for row in source_rows:
        if args.author_source and row["author_source"] not in args.author_source:
            continue
        if args.dataset and row["dataset_id"] not in args.dataset:
            continue
        if not args.include_evaluated and truthy(row["claim_extracted"]):
            continue
        selected.append(row)
    if args.max_papers < 0:
        raise ValueError("--max-papers cannot be negative")
    if args.max_papers:
        selected = selected[: args.max_papers]

    index_rows: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        eval_id = f"paper_{index:04d}"
        run_dir = output_root / eval_id
        for name in ["input", "paper", "metadata", "request", "response", "claim_tables", "reviews"]:
            (run_dir / name).mkdir(parents=True, exist_ok=True)

        source_paper = Path(row["paper_path"]).resolve()
        source_run = source_paper.parent.parent
        source_codebook = source_run / "input" / "public_codebook.json"
        if not source_paper.is_file() or not source_codebook.is_file():
            raise FileNotFoundError(f"Missing paper or codebook for {row['paper_id']}")
        paper_mode = copy_if_missing(source_paper, run_dir / "paper" / "mini_paper.md")
        codebook_mode = copy_if_missing(source_codebook, run_dir / "input" / "public_codebook.json")

        (run_dir / "metadata" / "run_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "0.1",
                    "status": "completed",
                    "run_id": eval_id,
                    "evaluation_workspace": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        index_rows.append(
            {
                "evaluation_id": eval_id,
                "paper_id": row["paper_id"],
                "dataset_id": row["dataset_id"],
                "author_source": row["author_source"],
                "author_paradigm": row["author_paradigm"],
                "workspace_dir": str(run_dir.relative_to(output_root)).replace("\\", "/"),
                "paper_file_mode": paper_mode,
                "codebook_file_mode": codebook_mode,
            }
        )

    index_path = output_root / "EVALUATION_INDEX.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]) if index_rows else ["evaluation_id"])
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Prepared {len(index_rows)} isolated evaluation workspaces under {output_root}")
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
