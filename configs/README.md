# Configuration Layout

- `experiments/`: experiment-level settings such as seeds, run ids, and artifact directories.
- `datasets/`: dataset construction settings and truth-map schema metadata.
- `models/`: scientist, reviewer, and verifier model placeholders.
- `evaluators/`: evaluator-only model settings, defaulting to an OpenRouter DeepSeek claim extractor.
- `reviewers/`: reviewer LLM settings for statistical review and editor ranking.
- `prompts/`: AI-scientist task prompts, the claim-extraction protocol and the four editor rubrics.
- `scoring/`: the full-text adjudication codebook and its provider profile.
- `schemas/`: JSON schemas for answer keys, claim ledgers and adjudication records.

The toy configuration is plain TOML readable with the standard library.
