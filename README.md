# Reusable multimodal video search

This repository contains a reusable university project for processing video,
building multimodal indexes and serving search results. Generated media,
models, databases, FAISS indexes and Redis data are kept outside the Git
checkout.

## Repository map

```text
app/                  FastAPI search application and runtime services
pipeline/             Ingest, frames, dedupe, merge, OCR, ASR and indexing
static/               Browser assets
benchmark/            Evaluation notes and frozen qrels workflow
docs/                 Architecture, pipeline map and maintenance notes
legacy/aic-2026/      Retired competition, Colab and submission utilities
requirements/         Runtime, pipeline and development dependency sets
```

The processing flow is:

```text
videos -> manifest -> frames -> dedupe -> merge
       -> OCR + ASR -> bundle/index -> validation -> search API
```

## Start with the documentation

- [Architecture](docs/ARCHITECTURE.md) — application and browser structure.
- [Pipeline map](docs/PIPELINE_MAP.md) — stage inputs, outputs and contracts.
- [Pipeline guide](pipeline/README.md) — packaged stage entry points.
- [Requirements](requirements/README.md) — install only what the task needs.
- [Cleanup guide](docs/CLEANUP_GUIDE.md) — rules for keeping generated data
  out of Git.

Historical AIC 2026 documents are indexed under
[`docs/archive/aic-2026/`](docs/archive/aic-2026/). They are reference material
and are not part of the reusable runtime.

## Run the pipeline

Use a data directory outside the repository and pass its paths through the
stage arguments or environment variables. The compatibility launchers remain
available:

```bash
python -m pipeline.run --help
python -m pipeline.new_dataset --help
python run_s01_pipeline.py --help
python run_new_dataset_pipeline.py --help
```

Run individual stages from their package when debugging, for example:

```bash
python -m pipeline.frames.extract --help
python -m pipeline.ocr.index --help
python -m pipeline.asr.index --help
python -m pipeline.index.bundle --help
```

## Run the application

The implementation is [app/main.py](app/main.py). `main.py` remains a small
compatibility wrapper for existing launchers. Install the appropriate
dependency set, configure the external data paths, then start the existing
launcher for the target environment.

## Validate changes

```bash
python -m compileall -q .
python -m unittest discover -v
```

See [docs/BASELINE_VALIDATION.md](docs/BASELINE_VALIDATION.md) for the current
test status and optional dependency requirements.
