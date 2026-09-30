# Multimodal video search

Reusable video-processing and search code for a university project. The
repository keeps processing code separate from the search application and
keeps generated media, models, indexes and Redis data outside Git.

## Layout

```text
app/                 Search API and runtime helpers
app/services/        Redis, telemetry, health, conversation and reranking services
pipeline/            Ingest, frames, dedupe, OCR, ASR, merge and index stages
web assets            index.html and static/
tests                Active regression tests at the repository root for now
docs/                Architecture and pipeline documentation
legacy/aic-2026/     Historical Colab, batch and competition tools
```

The end-to-end flow is:

```text
raw videos -> manifest -> frames -> dedupe -> OCR/ASR
           -> merge/index -> serving bundle -> search API
```

See [docs/PIPELINE_MAP.md](docs/PIPELINE_MAP.md) for the stage contracts and
[ARCHITECTURE.md](ARCHITECTURE.md) for the application structure.

## Pipeline entry points

The compatibility commands remain available:

```bash
python run_s01_pipeline.py --help
python run_new_dataset_pipeline.py --help
python preprocess_dataset.py --help
```

The packaged commands are:

```bash
python -m pipeline.run --help
python -m pipeline.new_dataset --help
```

The stages are resumable and should be run against a configured data directory,
not inside the Git checkout. Configure input/output locations with command-line
arguments or environment variables; do not commit local paths, models, videos,
SQLite databases, FAISS indexes or Redis snapshots.

## Search runtime

The application implementation is `app/main.py`. The root `main.py` remains a
compatibility wrapper for existing launchers and tests. Start it with the
existing launcher after installing the dependencies listed in the requirements
file appropriate for the target environment.

## Validation

```bash
python -m compileall -q .
python -m unittest discover -v
```

Optional runtime tests require FastAPI, Redis and FAISS dependencies. The
pre-refactor test limitations are recorded in
[docs/BASELINE_VALIDATION.md](docs/BASELINE_VALIDATION.md).

Legacy AIC 2026 utilities remain available under `legacy/aic-2026/` for
reference, but they are not part of the reusable runtime or pipeline.
