# Blinded journal-editor ranking

Act as the handling editor of a reputable, selective general health and
biomedical research journal. All supplied submissions are within scope.

Based only on each submission's blind paper ID and complete manuscript supplied in full_text, rank all
10 submissions by predicted likelihood of eventual publication after ordinary
peer review and reasonable revision. Rank 1 is most likely to be published.

Give highest priority to publishability and scientific validity: whether the
design and analysis appear appropriate, the evidence supports the conclusions,
and no visible fatal flaw, internal contradiction, or major overclaim invalidates
the central result. Also consider contribution, originality, importance, reader
relevance, and reporting quality.

A null, negative, or inconclusive result is not a reason for rejection when the
question is important and the study appears sound. Do not reward statistical
significance, a large effect estimate, causal wording, novelty language, or
confidence by itself. Give scientific validity priority when criteria conflict.

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
