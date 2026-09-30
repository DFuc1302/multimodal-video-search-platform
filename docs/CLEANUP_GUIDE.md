# Code cleanup guide

The cleanup branch separates reusable processing from serving and competition
compatibility. Each change should keep the end-to-end processing path working:

1. ingest and manifest
2. frame extraction
3. frame deduplication
4. merge and normalization
5. OCR
6. ASR
7. bundle and vector indexes
8. validation
9. serving

Use `docs/PIPELINE_MAP.md` as the source of truth while moving files. Keep
compatibility wrappers temporarily when a launcher or test still imports a
root-level module. Remove wrappers only after the new package path has passed
the same tests.

Generated data, models, Redis snapshots, virtual environments, logs and local
secrets stay outside the source tree. Runtime paths must come from configuration
or environment variables rather than AIC-specific absolute paths.
