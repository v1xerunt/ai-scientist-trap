# AI scientist trap testbed

Code for testing autonomous AI scientists on data with known answers. Real
covariates from the US National Health and Nutrition Examination Survey
(NHANES, August 2021–August 2023) are combined with synthetic outcomes generated
from specified models, and one classical statistical trap is concealed in each
dataset. AI-scientist pipelines analyse the data and write manuscripts, which are
then adjudicated against the hidden answer key and ranked by blinded AI editors.

## Contents

| Path | Purpose |
|---|---|
| `src/trap_ai/` | Python package and `trap-ai` command-line interface |
| `src/trap_ai/nhanes.py`, `plasmode.py`, `validation.py` | Download NHANES, generate the seven datasets, freeze and validate them |
| `src/trap_ai/agent_harness.py` | AI-scientist harness with four pipelines, local code execution and the optional claim-check gate |
| `codex_skills/` | The four pipelines as Codex skills |
| `scripts/` | Blinded evaluation workspaces, claim extraction, full-text adjudication and editor ranking |
| `configs/` | Model settings, task prompts, claim-extraction protocol, adjudication codebook, editor rubrics and schemas |
| `experiments/documentation/` | Builders for the documentation variants of the confounding and leakage datasets |

## Setup

```bash
conda env create -f environment.yml
conda activate trap-ai
python -m pip install -e .
python -m pytest -q
```

Copy `.env.example` to `.env` and add an OpenRouter API key (`OPENROUTER_API_KEY`).
`OPENAI_API_KEY` is needed only for the OpenAI batch commands. `.env` is ignored by git.

## 1. Data

```bash
python scripts/trap_ai.py download-nhanes
python scripts/trap_ai.py generate-plasmode --overwrite
python scripts/trap_ai.py freeze-plasmode --overwrite
```

This writes `data/traps/plasmode_nhanes_latest/<design>/` for seven designs:
true effect, pure null, outcome leakage, collider and post-outcome variables,
survey weighting, named negative controls and confounding. Each design has
`dataset.csv`, `public_codebook.json` and `hidden_truth_manifest.json`. The
hidden answer key is for adjudication only and must never be shown to AI
scientists or editors.

## 2. AI scientists

Each AI scientist pairs a pipeline with a backbone model. The pipelines are
`single_shot`, `iterative_coding`, `data_to_paper` and `ai_scientist`.

```bash
python scripts/trap_ai.py run-agent-loop \
  --dataset-dir data/traps/plasmode_nhanes_latest/confounding_trap \
  --config configs/agents/openrouter_scientist_long.toml \
  --model deepseek/deepseek-v4-pro \
  --paradigm ai_scientist \
  --prompt-regime neutral \
  --n 5 \
  --run-root runs/authors
```

Add `--dry-run` to build requests without calling the provider. To run the same
pipelines with Codex models, create workspaces with
`scripts/prepare_codex_author_workspaces.py` and install the skills in
`codex_skills/` (see `codex_skills/README.md`).

### Claim-check gate

Setting the environment variable `TRAP_AI_CLAIM_CHECK_GATE` inserts a gate
between the last analysis stage and the manuscript.

- `1`, execution gate. The AI scientist declares the associations it intends to
  report and, for each, executable checks against common causes, timing or
  derivation of the exposure, sampling design and chance. Code runs every check,
  the harness verifies that each check ran and returned an estimate, and the
  manuscript may present an association as a finding only if all its checks ran
  and none refuted it.
- `2`, selection gate. As above, and the harness also fixes the adjustment set of
  each common-cause check (every measured variable with an absolute Spearman
  correlation of at least 0.1 with both exposure and outcome) and treats an
  exposure with an area under the curve of at least 0.9 for the outcome as
  refuted.

### Documentation variants

```bash
python experiments/documentation/build_variants.py
python experiments/documentation/build_secondary_arms.py
```

See `experiments/documentation/README.md` for the variants and how to run them.

## 3. Evaluation

List completed author runs and stage blinded, paper-only evaluation workspaces:

```bash
python scripts/build_handoff_inventory.py
python scripts/prepare_evaluation_workspaces.py --output-root runs/evaluation
```

Extract a claim ledger for each manuscript:

```bash
python scripts/run_openrouter_claim_extraction_batch.py \
  --run-root runs/evaluation \
  --protocol configs/prompts/claim_extraction_production_v4.md \
  --model deepseek/deepseek-v4-pro --reasoning-effort high --env-file .env
```

Adjudicate each full manuscript against its hidden answer key:

```bash
python scripts/run_openrouter_fulltext_trap_scoring_batch.py \
  --run-root runs/evaluation \
  --codebook configs/scoring/fulltext_trap_scoring_codebook_v2.md \
  --schema configs/schemas/fulltext_trap_scoring_v2.schema.json \
  --model deepseek/deepseek-v4-flash-0731 \
  --provider-order deepinfra --disable-provider-fallbacks \
  --reasoning-effort high --env-file .env
```

Rank pools of full manuscripts with a blinded AI editor (dry run by default, add
`--send` to make requests). The four editor rubrics are in
`configs/prompts/editor_rubrics/`.

```bash
python scripts/run_openrouter_selection_reviewer.py --help
```
