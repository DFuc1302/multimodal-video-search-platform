# Project layout and maintenance

The application is a FastAPI service with a browser interface built from plain
HTML, CSS, and JavaScript. It does not require Node packages or a frontend build.
Run the existing WSL launcher and open the server URL; keep the `static/` folder
beside `main.py` when copying or packaging the application.

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

- `main.py`: application lifecycle, HTTP routes, and static asset mount.
- `shared.py`: shared configuration, models, metadata, and vector access.
- `redis_store.py`, `redis_serving.py`, `federated_runtime.py`: backend adapters.
- `search_pages.py`, `query_hints.py`, `folder_filter.py`: search helpers.
- `asr_index.py`, `ocr_index.py`, `summary_search.py`, `kisc.py`: feature services.
- Build, migration, and packaging scripts remain at their existing paths so
  documented commands and imports continue to work.

Models, indexes, media, and Redis snapshots are external runtime data. They are
not frontend assets and must never be placed in `static/`.

`CODE` is the working source; `CODE_FOR_TEAM` is an independently runnable
distribution. Shared code and `static/` must be copied together. The team
installer, launcher, packaging tools, and README have distribution-specific
behavior and should not be overwritten with their `CODE` counterparts.

## Validation

From either source directory, using the installed project Python environment:

```sh
python -m unittest discover -v
node test_ui.cjs
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
