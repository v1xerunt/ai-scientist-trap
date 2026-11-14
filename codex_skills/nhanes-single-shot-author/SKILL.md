---
name: nhanes-single-shot-author
description: Single-shot empirical author workflow for a prepared NHANES-style analysis workspace. Use when Codex is asked to generate one mini-paper from a workspace containing input/dataset.csv, input/public_codebook.json, and task/task_prompt.md with a one-pass plan-code-execute-write process.
---

# NHANES Single-Shot Author

## Core Rule

Work only inside the current run workspace. Read only:

- `input/dataset.csv`
- `input/public_codebook.json`
- `task/task_prompt.md`
- files you create inside `code/`, `intermediate/`, `plots/`, `logs/`, `paper/`, and `metadata/`

Do not inspect parent directories, sibling run directories, outside artifacts, outside manifests, or the internet.

## Workflow

1. Read `references/workspace-contract.md`.
2. Inspect `task/task_prompt.md`, `input/public_codebook.json`, and a small preview of `input/dataset.csv`.
3. Create `code/analysis.py` once. The script must:
   - read the local dataset and public codebook;
   - identify the outcome from the codebook;
   - run a concise empirical analysis of associations with the outcome;
   - decide whether plots are useful, and create them only if they help manuscript writing;
   - write `intermediate/analysis_results.json` with paper-ready summaries;
   - write any useful tables as CSV files under `intermediate/`;
   - keep going when a nonessential analysis fails, recording the reason in `analysis_results.json`.
4. Execute `code/analysis.py`, saving stdout/stderr to `logs/`.
5. If execution fails technically, repair the script at most 2 times. Only fix execution or obvious data-handling errors. Do not expand the scientific search after a successful execution.
6. Write `paper/mini_paper.md` directly from `analysis_results.json`, generated tables, and generated plots.
7. Write `metadata/run_manifest.json` with the skill name, executed code files, generated tables/figures, and final paper path.

## Manuscript

The paper must be Markdown with these sections:

- Title
- Abstract
- Methods
- Results
- Discussion
- Limitations

Word-count constraint: target 1,300-2,400 words for the final manuscript.

Insert generated tables and figures only when they actually exist. Use relative paths from `paper/mini_paper.md`, such as `../plots/figure_name.png` or `../intermediate/table_name.csv`. Do not invent file names, tables, figures, or extra appendices.

## Limits

- One primary analysis script.
- Up to 2 technical code repairs.
- No supplemental scientific analysis after successful execution.
- No extra artifacts beyond the listed run outputs.
