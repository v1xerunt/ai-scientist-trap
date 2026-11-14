---
name: nhanes-iterative-author
description: Iterative empirical coding author workflow for a prepared NHANES-style analysis workspace. Use when Codex is asked to generate one mini-paper from input/dataset.csv, input/public_codebook.json, and task/task_prompt.md using plan, code, execution review, limited supplemental analysis, and final manuscript writing.
---

# NHANES Iterative Author

## Core Rule

Work only inside the current run workspace. Read only the public input files and files created during this run. Do not inspect parent directories, sibling runs, outside artifacts, outside manifests, or the internet.

## Workflow

1. Read `references/workspace-contract.md`.
2. Write `metadata/analysis_plan.json` with:
   - `analysis_plan`: concise empirical plan;
   - `planned_outputs`: tables or figures expected to support the paper.
3. Create and execute `code/analysis.py`.
4. If code execution fails technically, repair at most 3 times. Repairs must preserve the intended analysis and prioritize executable code.
5. Run an analysis sufficiency review and save `metadata/analysis_sufficiency_review.json`.
6. If the review says more analysis would materially improve the evidence, run at most 3 supplemental analysis scripts total:
   - `code/analysis_supplement_01.py`, etc.
   - each writes a compact JSON summary under `intermediate/`;
   - each may write useful tables and plots;
   - each technical failure may be repaired within the same global 3-repair budget.
7. Run at most 2 sufficiency review rounds total. If the second review still suggests improvements, stop and proceed with the available executed results.
8. Write `paper/mini_paper.md`.
9. Write `metadata/run_manifest.json`.

## Manuscript

Use Markdown sections:

- Title
- Abstract
- Methods
- Results
- Discussion
- Limitations

Word-count constraint: target 1,300-2,400 words for the final manuscript.

Use only generated tables/figures listed in intermediate summaries. Do not invent file names or outputs. If analysis was limited by data or execution constraints, express limitations in scientific terms without describing the workflow as incomplete.

## Limits

- Maximum 3 technical code repairs across all scripts.
- Maximum 3 supplemental analysis scripts.
- Maximum 2 sufficiency reviews.
- No extra artifacts beyond the listed run outputs.
