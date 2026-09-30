# Supplemental N OCR

Run in WSL:

```bash
cd /mnt/c/PERSONAL_PROJECT/AIC_26/CODE
bash run_n_top_ocr_wsl.sh
```

The job scans N folders directly in both configured roots, including N031–N040
which were absent from the existing serving index. Inventory on 2026-09-25:
223,005 frames, N001–N100.

It reads the upper 25% at original resolution, recognizes Vietnamese and English,
and checks for white pixels within recognized regions without requiring black
backgrounds. This color heuristic is approximate: scene text or light backgrounds
can also qualify. Coordinates are normalized to the full original frame.

Results are supplemental and do not replace old OCR. The SQLite output is:
`/mnt/d/AIC_26/data/new_dataset_work/n_white_top_ocr.sqlite3`.
The `readings` table contains file, source, source size/mtime, version, JSON texts,
error and update time. Restarting skips successful unchanged files; errors retry.
A file lock prevents concurrent workers writing the same output.

This job does not update the serving index, Redis, CLIP or ASR. Importing the new
readings into search is a separate step; N031–N040 also need their frame metadata
and media paths registered before they can appear in search.

For the detached run started by Codex, progress is at
`/mnt/c/PERSONAL_PROJECT/AIC_26/runtime_logs/n_white_top_ocr.log`:

```bash
tail -f /mnt/c/PERSONAL_PROJECT/AIC_26/runtime_logs/n_white_top_ocr.log
```

Keep Windows/WSL awake while processing. After an interruption, run the same
launcher to resume. `--check` inventories without running OCR; `--limit 48`
processes a sample spread across the pending inventory.
