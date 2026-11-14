# Workspace Contract

Each run workspace is self-contained:

```
input/dataset.csv
input/public_codebook.json
task/task_prompt.md
```

Create outputs only here:

```
code/
intermediate/
plots/
logs/
paper/
metadata/
```

`intermediate/analysis_results.json` must contain:

- `primary_results`: list of findings with exposure, outcome, model, estimate, p-value if available, direction, interpretation, and source table or figure.
- `secondary_results`: supporting or exploratory findings.
- `robustness_results`: sensitivity, adjusted/unadjusted, weighted/unweighted, missingness, or alternative-model checks when run.
- `generated_tables`: filename, title, and one-sentence interpretation.
- `generated_figures`: filename, title, and one-sentence interpretation.
- `limitations`: limitations visible from the executed analysis.

Use deterministic randomness (`seed = 20260617`) when sampling, splitting, resampling, or imputing. Prefer standard Python packages already available in the environment. Do not install packages unless the workspace explicitly asks for it.
