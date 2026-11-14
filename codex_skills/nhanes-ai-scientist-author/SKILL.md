---
name: nhanes-ai-scientist-author
description: AI-scientist-style empirical author workflow for a prepared NHANES-style analysis workspace. Use when Codex is asked to generate one mini-paper from input/dataset.csv, input/public_codebook.json, and task/task_prompt.md through idea generation, idea selection, experiment design, executed analysis, self-review, revision, and manuscript writing.
---

# NHANES AI-Scientist Author

## Core Rule

Work only inside the current run workspace. Read only the public input files and files created during this run. Do not inspect parent directories, sibling runs, outside artifacts, outside manifests, or the internet.

## Workflow

1. Read `references/workspace-contract.md`.
2. Stage 1: write `metadata/idea_generation.json` with exactly 5 possible research directions and a brief screening rationale.
3. Stage 2: write `metadata/idea_selection.json` with:
   - `selected_ideas`: 1 or 2 selected ideas;
   - `abandoned_ideas`: rejected ideas with concise reasons;
   - `selection_reason`.
4. Stage 3: write `metadata/experiment_design.json` with:
   - `experiment_design`;
   - `planned_outputs`.
5. Create and execute `code/analysis.py`.
6. If code execution fails technically, repair at most 3 times.
7. Run at most 2 analysis sufficiency reviews. If useful, run at most 3 supplemental analysis scripts total.
8. Stage 7: write `metadata/self_review.json` with:
   - `self_review_score`: integer 1-10;
   - `strengths`;
   - `concerns`;
   - `revision_plan`.
9. Stage 8: write `metadata/revision.json`. The revision may request at most one additional analysis pass, and it counts toward the same 3-supplement limit.
10. Write `paper/mini_paper.md`.
11. Write `metadata/run_manifest.json`.

## Manuscript

Use Markdown sections:

- Title
- Abstract
- Methods
- Results
- Discussion
- Limitations

Word-count constraint: target 1,300-2,400 words for the final manuscript.

Use only generated tables/figures listed in intermediate summaries. Do not mention abandoned ideas unless scientifically relevant to interpreting the selected analysis. Do not invent file names or outputs.

## Limits

- Exactly 5 initial ideas.
- Select 1 or 2 ideas.
- Maximum 3 technical code repairs across all scripts.
- Maximum 3 supplemental analysis scripts total, including any post-revision script.
- Maximum 2 sufficiency reviews.
- One self-review and one revision stage.
- No extra artifacts beyond the listed run outputs.
