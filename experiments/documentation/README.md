# Documentation variants

These scripts build variants of the confounding and outcome-leakage datasets in
which only the codebook descriptions of the four trap variables differ. The data
files and every other codebook entry are copied unchanged from
`data/traps/plasmode_nhanes_latest/`, so the generated datasets must exist first.

| Variant | Documentation of the trap variables |
|---|---|
| `L0` | Neutral. A plain description that says nothing about the variable's role |
| `L1` | Factual cue. One true statement about how the variable arose |
| `L2` | Named check (confounding only). The factual cue plus one sentence stating that an independent association requires adjustment for age, BMI and sex |
| `G` | Generic checklist. Neutral documentation, with a four-item checklist appended to the task prompt |

```bash
python experiments/documentation/build_variants.py        # L0 and L1
python experiments/documentation/build_secondary_arms.py  # L2 and G
```

Variants are written to `data/traps/doc_cue_v1/<trap>_<variant>/`. Run them with
`trap-ai run-agent-loop --dataset-dir data/traps/doc_cue_v1/<trap>_<variant>`. For
`G`, also pass `--prompts experiments/documentation/prompts_checklist.toml
--prompt-regime neutral_checklist`.
