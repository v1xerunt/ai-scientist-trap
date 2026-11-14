# Codex Author Skills

These are the four skills used to run the AI-scientist pipelines with Codex.
Install each complete directory, including `agents/` and `references/`, under
your Codex skills directory.

Do not edit these copies when reproducing the existing author condition.
Create a versioned new skill directory for any future prompt intervention.

The four skills are:

- `nhanes-single-shot-author`
- `nhanes-iterative-author`
- `nhanes-data-to-paper-author`
- `nhanes-ai-scientist-author`

For evaluator or reviewer work, create separate skills and use the isolated
paper-only workspaces produced by `scripts/prepare_evaluation_workspaces.py`.
Do not run evaluator/reviewer instructions inside an author workspace.
