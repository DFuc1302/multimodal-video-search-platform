# Baseline validation

Validation was run before moving the processing stages into packages.

## Passed

- Python bytecode compilation across the repository.
- Node syntax checks for JavaScript and CommonJS files.
- 61 tests were collected by `python -m unittest discover -v`.
- The tests that do not require optional runtime dependencies or deleted AIC
  output data still execute.

## Current blockers

- 20 tests cannot import optional runtime dependencies in this environment,
  including `fastapi`, `redis`, and `faiss`.
- 3 tests fail on existing behavior: the OCR batch-size contract,
  `folder_filter` video aliases, and a generated Colab notebook that is no
  longer present after data cleanup.
- Some tests still assume the old root-level module layout and AIC output
  folders.

These failures are recorded before refactoring. The cleanup must add a minimal
development dependency set and separate reusable pipeline tests from retired
competition/output tests before files are moved.
