# Production Claim Extraction Protocol

Protocol version: `claim-extraction-production-v4-20260902`

## Objective

Read the complete supplied manuscript and public codebook and independently
produce the final machine-readable claim ledger. There is no upstream candidate
extractor and no downstream semantic correction. Your output is the production
result.

Extract the manuscript's explicit central empirical exposure-outcome
conclusions and every author-interpreted material qualification. Do not judge
scientific truth, detect hidden data conditions, or repair the paper's science.
Do not enumerate every coefficient.

A claim must contain one canonical public-codebook exposure, the canonical
outcome, and one materially distinct author-interpreted result or model context.

## Mandatory three-pass reading procedure

Complete all passes before producing output.

### Pass 1 — central claims

Read the Abstract, Discussion, and Conclusion. Add every named
exposure-outcome finding that is a headline conclusion or retained empirical
take-away. Include a headline descriptive gradient when it maps to a canonical
exposure and the canonical outcome.

### Pass 2 — material qualifications

Scan the Abstract, Discussion, and Conclusion sentence by sentence, then check
the supporting principal Results text and tables. Add one row for every named,
author-interpreted:

- null, weak, nonsignificant, or imprecise finding;
- attenuation, reversal, loss/gain of significance, or model contrast;
- sensitivity, weighted/unweighted, inclusion/exclusion, or robustness result;
- subgroup result or interpreted heterogeneity finding; and
- member of a clearly defined variable family interpreted by the author and
  supported by an exposure-specific principal Results sentence or table row.

If one interpreted sentence names four exposures, create four rows. If the
Discussion interprets “dietary variables” and a principal Results sentence or
table explicitly names fiber, sugar, fat, and energy, create one row for each
named member. Never invent an unnamed member.

### Pass 3 — completeness, typing, and deduplication

Compare the ledger back against every empirical sentence in the Abstract,
Discussion, and Conclusion. Restore any omitted named central or material
qualification. Then remove table-only noise and semantic duplicates. Finally
audit claim type, model context, evidence, and canonical names row by row.

## Retention boundary

Retain central positive, inverse, nonlinear, interaction, and interpreted null
findings; named material qualifications; and explicitly interpreted
model-specific, sensitivity, robustness, and subgroup results.

Exclude background/literature claims, methods, sample size, missingness,
software, limitations, recommendations, overall prevalence without a named
exposure, routine descriptive statistics not used as a conclusion, variables
listed only as covariates, uninterpreted table coefficients, unimportant
intermediate models, and one-off exploratory or secondary results not retained
in the paper's interpretation.

Statistical significance alone never makes a result eligible. Nonsignificance
never makes an author-interpreted qualification ineligible.

## Model-specific rows

Preserve the manuscript's scientific contrasts. Never collapse materially
different:

- unadjusted versus adjusted results;
- single-exposure versus mutually adjusted results;
- weighted versus unweighted results;
- main versus sensitivity, subgroup, exclusion, or robustness results;
- attenuation, reversal, gain/loss of significance, or direction changes; or
- conditioning sets that the author interprets differently.

Equivalent repetitions across sections become one row using the clearest
complete evidence quote. Same-direction sequential models may become one row
only when the author gives their difference no interpretive weight.

For every named attenuated exposure create a row. When the final direction or
status is not stated, use `finding_result = unclear`, describe the compared
models in `model_type`, and leave unsupported estimates blank.

A single categorical exposure with several contrasts may use one row only when
all contrasts share claim type, result, and model context and every named
contrast and estimate is preserved losslessly in `model_type` or `estimate`.

## Claim type

Allowed values:

- `predictive_association`: association or prediction without explicit
  independence, directness, or causality;
- `independent_risk_factor`: the evidence explicitly calls this same
  model-specific exposure association independent, or says it remained or
  persisted after adjustment;
- `direct_effect`: the evidence explicitly says direct effect/relationship;
- `causal_effect`: the evidence explicitly uses causal language.

Use `independent_risk_factor` only when the submitted literal `source_quote`
supports independence for this same exposure and this same model-specific
result. Adjustment or significance alone is insufficient.

The following are never IRF evidence:

- the exposure merely appears in an adjustment/covariate list;
- the grammatical subject of independent/remained/persisted is another
  exposure;
- the wording belongs to a neighboring or contrast clause;
- the wording describes a different model result for the same exposure; or
- a sensitivity result remained significant after excluding a subgroup but
  independence after adjustment was not stated.

Apply these examples literally:

- “X remained associated after adjustment for Y and Z” supports IRF for X,
  not Y or Z.
- “X and Y were associated in single-exposure models, but Y remained in the
  mutually adjusted model” yields predictive single-exposure rows for X and Y
  and an IRF row only for the mutually adjusted Y result.
- “X was attenuated, whereas Y remained associated” yields predictive
  attenuation for X and IRF for Y.
- “X was not independently associated with the outcome” is IRF + null.
- HDL cholesterol and total cholesterol are different exposures; never borrow
  evidence between them.

For every proposed IRF silently verify all four conditions:

1. its source quote contains independent/remained/persisted wording;
2. that wording applies grammatically to the row's exposure;
3. the quote describes the row's model context rather than another model; and
4. the exposure is not merely a covariate.

If any condition fails, use `predictive_association`.

Allowed `finding_result` values are `positive`, `inverse`, `null`, `nonlinear`,
`interaction`, and `unclear`. Claim type and result are orthogonal. Do not infer
a missing direction.

## Canonical fields

- `exposure`: exactly one public-codebook feature name, except an explicit
  interaction between canonical features;
- `outcome`: exactly the canonical public-codebook outcome;
- `model_type`: the actual analysis and material model/subgroup/sensitivity
  context;
- `covariates`: only adjustments explicitly tied to that result;
- `estimate` and `p_value`: exact values when reported, otherwise blank;
- `survey_design_used`: true only when that result explicitly uses weights,
  strata, PSU, or survey analysis;
- `evidence_source`: precise section/table/figure location.

## Evidence gate

Every claim has exactly one evidence row. Copy one complete manuscript sentence
or one complete Markdown table row as `source_quote`. It must be a literal
whitespace-normalized substring of the supplied manuscript. Never paraphrase,
join noncontiguous passages, change punctuation/numbers, or manufacture a
quote.

The quote must directly support the exposure, result, and claim type. For an
exposure-specific member of a Discussion-interpreted family, a complete
principal-table row may be used; explain the family interpretation and both
locations in `classification_rationale`.

## Output schema

Return only the structured object required by the caller. The claim fields are:

```text
claim_id,exposure,outcome,claim_type,finding_result,model_type,covariates,estimate,p_value,survey_design_used,evidence_source
```

Evidence fields are:

```text
claim_id,source_quote,source_location,classification_rationale
```

Use sequential IDs `c001`, `c002`, ... and exactly one evidence row per claim.
A paper with no eligible claims returns empty arrays. Do not output the
temporary ledger, Markdown, or commentary.

## Mandatory silent final audit

Before submitting, verify:

1. every named Abstract/Discussion/Conclusion empirical take-away has a row;
2. every named null, attenuation, sensitivity, contrast, robustness, and
   subgroup qualification has a row;
3. family members are expanded only from explicit names;
4. table-only and exploratory noise is absent;
5. materially different models are separate and repetitions are deduplicated;
6. every IRF passes all four exposure-and-model-specific evidence checks;
7. every exposure/outcome is canonical; and
8. every evidence quote is a literal manuscript substring supporting the full
   row.
