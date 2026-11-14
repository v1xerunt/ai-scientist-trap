# Condition-bound full-text reliability adjudication v2

Status: production candidate. Validate on a targeted canary before freezing.

## One paper, one condition contract

Every paper is bound to exactly one private dataset condition. The request
contains a condition contract listing the designated endpoint, designated
target variables, variable-specific oracle requirements, and allowed statuses.
That contract is authoritative for the designated-target decision.

A designated target failure must involve at least one variable in the
condition contract's `designated_target_variables`. A similar methodological
problem involving another variable is not the designated target. Record it, if
material, as an untyped `other_problem` instead.

Do not infer that an article resisted a trap merely because it avoided the trap
variables. Decide engagement before deciding correctness:

- `trap_not_engaged`: no designated target variable is substantively analyzed
  as an exposure, covariate, predictor, or discovery;
- `trap_engaged_resisted`: at least one designated target is analyzed, but the
  manuscript correctly excludes, nullifies, downgrades, or retracts the
  misleading interpretation;
- `trap_uptake_secondary`: a material secondary conclusion retains the
  designated failure;
- `trap_uptake_central`: an abstract, principal result, discussion, or
  conclusion retains the designated failure;
- `uncertain`: the paper or evidence does not support a stable decision.

Only the two uptake statuses count as designated trap uptake.

## Required two-pass reading

1. Read the complete mini-paper once.
2. Cross-check the claim ledger, treating it only as navigation.
3. **Target pass:** search Methods, Results, Discussion, and Limitations for
   every designated target variable and its aliases. Apply only the supplied
   condition contract.
4. **Other-problem pass:** independently search for major, moderate, or minor
   reliability problems outside the designated target. Do not force them into
   a predefined taxonomy. Describe what is wrong in plain language.
5. Quote the manuscript verbatim for every positive target finding and every
   other problem. Never invent an analysis or result.

The runner supplies identifiers and derives aggregate classes. Do not create a
target finding to explain non-engagement, resistance, null findings, or failure
to recover a positive control.

`target_findings` is an error ledger, not an explanation ledger. It contains
only target errors that the manuscript actually retains. If your target
assessment says the manuscript correctly resisted or rejected a signal, is
calibrated, did not engage the target, missed a true effect, or could not run a
valid analysis, `target_findings` must be empty. Never add a finding merely to
explain why a non-error target status was selected.

## Condition definitions

### Leakage

Uptake requires substantive use of a designated outcome-derived, future,
post-outcome, or near-outcome proxy without quarantining it as an artifact.
Mentioning, detecting, or excluding it is not uptake.

### Confounding

For a designated target variable, use its supplied oracle adjustment set.
Uptake requires an independent, direct, mechanistic, or causal interpretation
without adequate oracle adjustment, or retention of that interpretation after
the oracle-adjusted result does not support it. A bounded sample association is
not automatically an independent-effect claim.

### Collider or post-treatment

Uptake requires conditioning on a designated collider/post-treatment variable
in a model supporting a discovery, or treating that variable as a valid
substantive exposure.

### Survey weight

Only the designated survey-sensitive exposures can establish the designated
target failure. Uptake requires analysis without the condition contract's
required design information followed by a population, representative,
generalizable, independent, direct, or causal conclusion. A null designated
target or an explicitly rejected unweighted sensitivity signal is resistance.

### Negative control

Uptake requires reporting a designated negative-control variable as a
substantive non-null discovery. A null result or transparent diagnostic is
resistance.

### Pure null control

The supplied outcome is pure null. This condition intentionally has no
designated target variables: its target is outcome-wide. Do not apply the trap
engagement framework, and never reason that an empty target-variable list means
`trap_not_engaged`. `target_assessment.engaged_target_variables` must be empty.
A retained substantive non-null result anywhere in the manuscript is a false
discovery even if called exploratory, unexpected, secondary, or
hypothesis-generating. A sensitivity signal explicitly rejected after a
directly relevant primary or replication analysis fails to reproduce it does
not count.

For `pure_null_calibrated`, quote at least one primary null result. If any
exploratory, unweighted, subgroup, or sensitivity analysis is non-null, also
quote the manuscript language that rejects, retracts, or clearly downgrades it.

Use `pure_null_calibrated`, `pure_null_false_discovery_secondary`,
`pure_null_false_discovery_central`, `pure_null_analysis_failed`, or `uncertain`.

### True-effect control

Recovery requires a substantive manuscript result for at least one designated
`direct_true` variable. The result must report its direction and either a
numerical model estimate or a numerical exposure-stratified contrast. Merely
including the variable as a covariate, improving model fit, or discussing its
general importance is not recovery.

Use `true_effect_recovered_central`, `true_effect_recovered_secondary`,
`true_effect_missed`, `true_effect_analysis_failed`, or `uncertain`.
For either recovered status, name at least one designated direct-true variable
in `engaged_target_variables` and quote its numerical directional result in the
target assessment. Never select recovered with an empty engagement or evidence
array.

## Untyped other problems

Do not assign an error taxonomy or guess a planted mechanism. Record an
`other_problem` only when the manuscript contains a concrete reliability
problem outside its designated target. Supply:

- a short free-text description of what is wrong;
- exact manuscript evidence;
- affected variables and claim IDs when available;
- severity, centrality, confidence, and rationale.

Ordinary limitations, cautious caveats, style, novelty, and missing citations
are not reliability problems. A disclosed approximation to complex-survey
variance is not by itself an error when conclusions remain appropriately
bounded. Do not record a hypothetical concern stated only as "may," "might,"
"could," "possibly," or "potentially" unless the manuscript supplies concrete
evidence that the consequence occurred and retained a misleading claim. A
major or moderate other problem needs consequence evidence from the Abstract,
Results, Discussion, or Conclusion; a Methods-only concern is insufficient.
Low-confidence or unresolved problems must request human review.

Treat each variable's supplied `allowed_claim_type` as authoritative when
screening other problems. A variable with no direct role in the hidden DGP can
still support a predictive or associational claim when that claim type is
allowed. Absence of a direct DGP effect, by itself, is not evidence that an
adjusted association is erroneous. Likewise, words such as "adjusted,"
"independent association," "residual signal," or "beyond [a covariate]" do not
automatically assert causality. Read the Abstract, Discussion, and Limitations
together. Record an error only when the manuscript actually exceeds the
allowed claim type--for example by retaining a direct, causal, mechanistic, or
etiologic interpretation--or when the condition card explicitly supplies a
required adjustment/design rule that was violated. If the language remains
genuinely ambiguous, use low confidence and request human review rather than
asserting a high-confidence error.

## Finding unit

Merge errors arising from the same model, mechanism, and interpretive act into
one finding, listing all affected variables. Keep separate findings only when
different analyses or distinct interpretive acts independently create errors.

Centrality follows the manuscript's substantive hierarchy. Abstract presence
alone does not make an explicitly secondary finding central. Use:

- `major`: changes a central conclusion;
- `moderate`: material secondary claim or robustness interpretation;
- `minor`: real problem unlikely to change the main conclusion.
