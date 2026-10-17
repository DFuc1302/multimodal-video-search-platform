# AIC 2026 legacy code

This directory contains one-off competition, Colab, batch and M01 migration
tools preserved for historical reference. It is excluded from the reusable
runtime and pipeline checks.

The active processing path is documented in docs/PIPELINE_MAP.md and lives
under pipeline/. The active search runtime lives under app/.

The submission/ folder contains the retired DRES gateway, submission UI and
competition launchers. It is intentionally not imported by app/main.py.
