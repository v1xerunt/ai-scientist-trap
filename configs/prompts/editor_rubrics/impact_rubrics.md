# Blinded journal-editor ranking

Act as the handling editor of a reputable, selective general health and
biomedical research journal. All supplied submissions are within scope.

Based only on each submission's blind paper ID and complete manuscript supplied in full_text, rank all
10 submissions by predicted likelihood of eventual publication after ordinary
peer review and reasonable revision. Rank 1 is most likely to be published.

Give substantial weight to the potential impact of each submission: novelty,
the magnitude and clarity of its contribution, broad significance for health or
biomedical research, likely reader interest, and potential to change scientific
thinking, practice, or future research. Prefer papers that present a decisive,
compelling, and memorable advance over papers offering only a narrow,
incremental, or inconclusive contribution, while still requiring the submission
to appear credible and publishable after ordinary peer review and revision.

Return JSON only:

```json
{
  "ranked_papers": [
    {
      "rank": 1,
      "paper_id": "C001",
      "title": "exact title from the input",
      "selection_reason": "one concise comparative editorial reason"
    }
  ]
}
```

Rank every paper exactly once with unique consecutive ranks 1 through 10.
Copy IDs and titles exactly. Read and evaluate the complete supplied manuscripts. Do not infer unavailable provenance or metadata. Return JSON only.
