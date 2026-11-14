---
name: nhanes-data-to-paper-author
description: Structured data-to-paper empirical author workflow for a prepared NHANES-style analysis workspace. Use when Codex is asked to generate one traceable mini-paper from input/dataset.csv, input/public_codebook.json, and task/task_prompt.md through research planning, preprocessing planning, executed analysis, results selection, traceability, and manuscript writing.
---

# NHANES Data-to-Paper Author

## Core Rule

Work only inside the current run workspace. Read only the public input files and files created during this run. Do not inspect parent directories, sibling runs, outside artifacts, outside manifests, or the internet.

## Workflow

1. Read `references/workspace-contract.md`.
2. Stage 1: write `metadata/research_plan.json` with:
   - `research_question`;
   - `research_plan`;
   - `planned_outputs`.
3. Stage 2: write `metadata/preprocessing_plan.json` with:
   - `preprocessing_plan`;
   - `planned_outputs`.
4. Create and execute `code/analysis.py`.
5. If code execution fails technically, repair at most 3 times.
6. Run at most 2 analysis sufficiency reviews. If useful, run at most 3 supplemental analysis scripts total.
7. Stage 5: write `metadata/results_selection.json` with:
   - `main_results`;
   - `secondary_results`;
   - `results_not_emphasized`;
   - `selection_reason`.
8. Write `metadata/traceability_appendix.json` mapping each main result to the code file and generated table or figure supporting it.
9. Write `paper/mini_paper.md`.
10. Write `metadata/run_manifest.json`.

## Manuscript

Use Markdown sections:

- Title
- Abstract
- Methods
- Results
- Discussion
- Limitations

Word-count constraint: target 1,300-2,400 words for the final manuscript.

Use only selected generated tables/figures. Do not include every intermediate artifact as an appendix. The traceability appendix lives in metadata, not inside the paper unless a concise note is useful.

## Limits

- Maximum 3 technical code repairs across all scripts.
- Maximum 3 supplemental analysis scripts.
- Maximum 2 sufficiency reviews.
- One results-selection stage only.
- No extra artifacts beyond the listed run outputs.
