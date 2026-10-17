# Reusable pipeline map

This repository has two connected flows:

```text
raw media -> manifest -> keyframes -> dedupe -> OCR/ASR -> serving bundle
          -> indexes/validation -> search runtime
```

The processing flow must remain reusable for new datasets. The serving flow
must consume a completed bundle without rebuilding it at startup.

## Processing stages

| Stage | Current entry points | Main outputs |
| --- | --- | --- |
| Manifest and layout | `video_manifest.py`, `dataset_layout.py`, `dataset_assignment.json` | video/frame manifest and dataset layout |
| Frame extraction | `pipeline/frames/extract.py`, `re_extract.py` | keyframes and frame metadata |
| Frame dedupe | `pipeline/frames/dedup.py`, `dedup_active_index.py` | deduplicated frames and maps |
| Merge and normalization | `pipeline/preprocess/run.py`, `pipeline/preprocess/merge.py`, `pipeline/preprocess/extend.py`, `finalize_merged_data.py` | normalized serving data |
| OCR | `pipeline/ocr/index.py`, `ocr_quality.py`, `reindex_ocr.py`, `rerun_n_top_ocr.py` | OCR rows and text index |
| ASR | `pipeline/asr/index.py`, `check_asr_status.py`, `repair_asr_redis.py` | transcript cache and ASR rows |
| Bundle and vector index | `pipeline/index/bundle.py`, `pipeline/index/central.py`, `pipeline/index/export.py` | SQLite/FAISS/CLIP serving bundle |
| Redis migration | `pipeline/index/redis.py`, `redis_store.py`, `redis_serving.py` | optional text/summary Redis index |
| Validation | `evaluate_search.py`, `server_data_health.py`, `verify_package.py`, tests | reports and readiness checks |

The scripts remain at the repository root during the first cleanup phase. They
are moved into a package only after imports, subprocess calls and tests have
been updated together.

## Runtime entry points

The runtime consumes the output of the processing stages:

- `app/main.py`: search API and application runtime.
- `app/services/`: Redis, health, telemetry, conversation and reranking services.
- `team_runtime.py`: local runtime wrapper and readiness checks.
- `run_local_server_wsl.sh`: local server launcher.
- `static/` and `index.html`: browser interface.

The reusable orchestration entry points are now package modules:

```text
python -m pipeline.run          # resumable prepare/dedup/index/asr/bundle flow
python -m pipeline.new_dataset  # merge/verify/redis/activate flow
```

The old root commands `run_s01_pipeline.py` and `run_new_dataset_pipeline.py`
remain as compatibility wrappers during the migration.

Competition submission and gateway code is kept separate from the reusable
processing/runtime path until its dependencies are audited:
`competition_server.py`, `submission_server.py`, `dres_*.py`,
`submission_*.py`, and `official_submission_maps.py`.

## Cleanup rule

No stage implementation is deleted because its filename looks old. A file can
be removed only when it is unreferenced by a launcher, import, subprocess call,
or test, and its behavior is either covered elsewhere or intentionally retired.
