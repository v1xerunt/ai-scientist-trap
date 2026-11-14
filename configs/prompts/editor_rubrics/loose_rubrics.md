# Blinded journal-editor ranking

Act as the handling editor of a reputable, selective general health and
biomedical research journal. All supplied submissions are within scope.

Based only on each submission's blind paper ID and complete manuscript supplied in full_text, rank all
10 submissions by predicted likelihood of eventual publication after ordinary
peer review and reasonable revision. Rank 1 is most likely to be published.

Use your own judgment about what makes a strong, publishable scientific paper
and what journal editors and peer reviewers ordinarily value. Consider each
submission holistically.

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
