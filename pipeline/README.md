# Processing pipeline

The pipeline is resumable and keeps generated artifacts outside the source
tree. Use the stage modules directly when debugging, or use the orchestrators
for a complete run:

```bash
python -m pipeline.run --stage all --help
python -m pipeline.new_dataset --stage all --help
```

Stages are grouped as follows:

```text
core/       manifests, dataset layout, external assets and filters
frames/     extraction, re-extraction and deduplication
preprocess/ merge, extension, finalization and the dataset runner
ocr/        OCR indexing, quality, reindex and targeted reruns
asr/        transcript indexing, status and Redis repair
index/      serving bundle, FAISS/index export, catalog and Redis migration
```

Every stage should receive paths through arguments or environment variables.
Do not place generated images, videos, models, SQLite/FAISS files or Redis
snapshots under this package.
