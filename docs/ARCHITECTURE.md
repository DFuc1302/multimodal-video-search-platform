# Project layout and maintenance

The application is a FastAPI service with a browser interface built from plain
HTML, CSS, and JavaScript. It does not require Node packages or a frontend build.
Run the existing launcher and open the server URL; keep the `static/` folder
with the repository when copying or packaging the application.

## Interface

| File | Responsibility |
| --- | --- |
| `index.html` | Page structure, search controls, tab panels, and dialogs |
| `static/styles.css` | Component styles, workspace design tokens, responsive layouts, and reduced-motion support |
| `static/utils.js` | Pure formatting, HTML escaping, and phrase highlighting |
| `static/components.js` | Operator controls, HTTP response errors, keyboard activation, tabs, and dialog focus |
| `static/images.js` | Shared visibility observer, lazy-image retries, and cleanup |
| `static/app.js` | Search state, API calls, feature rendering, saved queries, and answer export |

Scripts are deferred and loaded in this order. Existing HTML event handlers
remain compatible with the public functions in `app.js`. Keep reusable controls
independent of feature state; test pure helpers without booting the application.

The server still returns up to 1,000 frames per page. The frame grid appends
40 cards at a time and loads additional batches near the scroll boundary.
Rendering generations invalidate callbacks from previous searches. One shared
image observer starts retry timers only when needed; remove pending images
before replacing a result container. Aborting a browser request prevents stale
results from replacing the latest search, but does not guarantee cancellation
of model work already executing on the server.

Dynamic filenames and other text must be escaped before insertion into HTML.
Bind actions with event listeners or element properties instead of interpolating
data into inline JavaScript. Use `makeKeyboardAction` for custom clickable
elements and `dialogFocus` for modal focus entry, trapping, and restoration.

## Server and data

- `app/main.py`: application lifecycle, HTTP routes, and static asset mount.
- `app/shared.py`: shared configuration, models, metadata, and vector access.
- `app/services/`: Redis, telemetry, health, federation and reranking adapters.
- `pipeline/`: packaged ingest, preprocessing, OCR, ASR and index stages.
- Root-level launchers remain compatibility wrappers for older commands/imports.

Models, indexes, media, and Redis snapshots are external runtime data. They are
not frontend assets and must never be placed in `static/`.

The repository is the working source. Runtime data belongs in a separate data
directory, while `static/` stays with the application source. Historical
competition packaging and launchers are kept under `legacy/aic-2026/`.

## Validation

Using the installed project Python environment:

```sh
python -m unittest discover -v
```

The Python suite uses temporary fixtures and tests static asset delivery without
starting the models. The Node checks cover script syntax, safe highlighting,
HTTP error handling, and observer/timer cleanup across 1,000 images. No package
installation is needed for the Node checks.

For UI changes, also verify real-data search, grouped results, preview and
shortlist actions, ASR, Video Match, conversational search, and System. Check
both desktop and narrow layouts; test keyboard tabs, preview focus restoration,
and access to category/color filters on mobile. Avoid indexing, migration, or
backup writes during a read-only UI verification.
