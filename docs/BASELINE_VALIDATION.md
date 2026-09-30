# Validation status

The reusable branch has been checked after the package reorganization.

## Passing checks

- `python -m compileall -q -f .`
- `git diff --check`
- Focused pipeline and regression tests: 9 tests pass, including frame-folder
  alias handling, OCR ranking, folder boundaries and official map alignment.
- The packaged entry points and compatibility launchers all respond to
  `--help`.

## Full test-suite notes

`python -m unittest discover -v` currently runs 23 tests. Eight tests cannot
load because optional packages (`faiss`, `fastapi` and `redis`) are not
installed in this environment, and one test still exposes existing behavior
outside the cleanup scope. Install the project runtime dependencies before
using the full search server or index-building suite.

The test files remain in the repository so a university project can enable
them in its own environment. Generated videos, models, FAISS indexes, Redis
snapshots and SQLite databases stay outside Git.
