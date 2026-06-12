# Handoff — Self-hosted Open-Meteo bulk export for Japan/Taiwan solar forecasting

> Progress handoff for Codex. Status as of 2026-06-11. Everything below is **verified by actually running it**, not assumed.

## Goal

The user hit rate limits on the public `api.open-meteo.com`. Real requirement:
- Scan **all grid points over Japan + Taiwan**, every **6 hours** (matching the model update cadence).
- Get **~7-day forecasts** of solar-energy variables: GHI (`shortwave_radiation`), `direct_radiation`, `diffuse_radiation`, `temperature_2m`, `precipitation`, and cloud cover (`cloud_cover`, `cloud_cover_low/mid/high`).
- Coverage: Japan and Taiwan.

Self-hosting is the sanctioned solution — data is CC BY 4.0; only the shared public server is rate-limited.

## Decision: use the built-in `export` command (no new code needed)

The repo already ships a command that does exactly the bulk-grid export. **No custom code was written into the repo** — `git status` is clean except for `export_test/` (output) and this handoff file.

- **Model:** `dwd_icon` (DWD ICON global). 0.125° (~11 km), 6-hourly runs (00/06/12/18z), covers both regions, 7.5-day horizon. Chosen over `jma_msm` because MSM only reaches ~39–78h and Taiwan sits at its southern edge.
- **Command:** `export <domain> <vars> --start_date --end_date --latitude-bounds "lo,hi" --longitude-bounds "lo,hi" --format parquet -o out.parquet`
- **Why it's correct:** `export` reuses the API's exact reader path — `domain.getReader(gridpoint:)` → `reader.get(mixed:)` (see `Sources/App/Commands/ExportCommand.swift:470-479`). All conversions are free: `shortwave_radiation = direct_radiation + diffuse_radiation` (`Sources/App/Icon/IconReader.swift:473-477`), scalefactors, unit conversion. For ICON, direct/diffuse are stored **natively** (GRIB `aswdir_s`/`aswdifd_s`, `IconVariableDownloadable.swift:176-177`) — no empirical separation model in the path.

## What was verified (end-to-end run)

1. Pulled prebuilt image `ghcr.io/open-meteo/open-meteo` — contains `export` and Parquet is compiled in.
2. `sync dwd_icon temperature_2m --past-days 1 --concurrent 8` → 772 MB into docker volume `open-meteo-data` (took ~43 min, S3 is slow).
3. `export dwd_icon temperature_2m --start_date 2026-06-11 --end_date 2026-06-13 --latitude-bounds "24,25" --longitude-bounds "121,122" --format parquet` →
   - **Future dates returned forecast values** (the one open question — confirmed).
   - bbox filtered correctly: 24–25°N × 121–122°E = **81 points** at 0.125° (9×9), 72 hourly steps = 5832 rows, **zero NaN**, temps 13–30°C (sane for June Taiwan).
4. **Correctness cross-check:** ran the local API server on the same volume and queried the same point — export values are **identical** to the API (e.g. 20.7, 20.8, 21.25, 22.15, 22.45, 22.4; API rounds to 1 decimal). The ~0.75°C gap vs the *public* API was purely a newer model run, not a conversion error.

## Gotchas (all hit during this run)

- **bbox only works with `--format parquet`.** The NetCDF path ignores `--latitude-bounds`/`--longitude-bounds` and dumps the entire global grid (2879×1441 ≈ 4.1M points). Always use parquet for the JP/TW box.
- **Native macOS `swift build` fails** — repo requires Swift tools 6.2, the machine has 6.1.2. Use the Docker image; do **not** try to build natively.
- **S3 open-data sync is slow** (~150–300 KB/s per connection; 772 MB single-variable took 43 min). `sync` pulls whole **global** chunk files per variable — you cannot fetch a sub-region from S3. Use `--concurrent` to parallelize; budget time for the multi-variable download.

## Artifacts left on disk

- Docker volume `open-meteo-data` — has `dwd_icon/temperature_2m` (772 MB) + static `HSURF.om` elevation + `meta.json`.
- `export_test/tw_test.parquet` — the verified Taiwan-box output (81 pts × 72 h).

## Commands to reproduce / run the real pipeline

```bash
# 1. Sync the solar variables (every 6h). Replace var list as needed.
docker run --rm -v open-meteo-data:/app/data \
  --entrypoint /app/openmeteo-api ghcr.io/open-meteo/open-meteo \
  sync dwd_icon shortwave_radiation,direct_radiation,diffuse_radiation,temperature_2m,precipitation,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high \
  --past-days 1 --concurrent 8

# 2. Export the Japan+Taiwan grid box to parquet (~7-day forecast).
docker run --rm -v open-meteo-data:/app/data -v "$PWD/out":/out \
  --entrypoint /app/openmeteo-api ghcr.io/open-meteo/open-meteo \
  export dwd_icon shortwave_radiation,direct_radiation,diffuse_radiation,temperature_2m,precipitation,cloud_cover \
  --start_date <TODAY> --end_date <TODAY+7> \
  --latitude-bounds "20,46" --longitude-bounds "120,150" \
  --format parquet -o /out/jp_tw.parquet
```

Note `shortwave_radiation` is derived from `direct_radiation`+`diffuse_radiation`, so make sure both raw components are synced.

## UPDATE 2026-06-12: switched to REMOTE MODE — no sync needed

`REMOTE_DATA_DIRECTORY=https://openmeteo.s3.amazonaws.com/data/` (found in `configure.swift:19`, undocumented) makes `export` read .om files directly from S3 via HTTP range requests, fetching only the bytes covering the bbox. Verified end-to-end: full 13-variable JP/TW land export (9,468 pts × 192 h, 1.8M rows, 0% NaN on every variable) in **3m26s from a cold cache, zero local database**. `run_jp_tw_solar.sh` is now remote-mode: no sync step, a small `open-meteo-cache` volume for the block cache, timestamped parquet per run (keep them — they accumulate an init_time × lead_time ICON archive for future ML training). S3 retains ~3 years of chunks (since 2023-05, incl. direct/diffuse radiation), so historical regional exports also work remotely with old date ranges. Lead-time-resolved (`previous_day`) variables are NOT on open-data (verified across 5 domains); for that, use dynamical.org's GFS Zarr archive (2021+) or the self-accumulated parquets. Caveats: the env var is undocumented (interface may change); DNI shows large values (>4000 W/m²) at very low sun angles — backwards-averaged low-sun artifact, filter by solar elevation downstream.

## UPDATE 2026-06-12 (later): multi-region pipeline (JP/TW + SEA + Australia)

User's sites span Japan/Taiwan, Southeast Asia, Australia. **Blocker found & fixed:** the CLI parser (Vapor ConsoleKit) rejects option values starting with "-" (no `=` syntax either), so southern-hemisphere bboxes were impossible. Fix: patched `ExportCommand.swift` (bbox parsing + help text) to accept `m` as minus prefix (`m44,m10` = -44..-10) — small diff, good upstream-PR candidate. Built local image **`open-meteo:bbox-fix`** via **`Dockerfile.local`** (NOT the stock Dockerfile: upstream's run base ships Arrow 22 while the build base compiles against Arrow 24 → `libparquet-glib.so.2400` missing at runtime; Dockerfile.local uses the build base as runtime. Also needed `docker build --pull` — BuildKit had mixed stale base caches).

**`run_solar_regions.sh`** is the new deliverable (supersedes run_jp_tw_solar.sh): remote-mode, loops 3 regions, one timestamped parquet each. Measured full cycle (13 vars, 7 days, land-only): **14m31s total, ~150MB/cycle** — jp_tw 9,468 pts (18MB), sea 23,891 pts (48MB), au 44,376 pts (83MB). All vars NaN ≈ 10-12% = horizon truncation only. ~77k points ≈ 8× JP/TW; yearly archive ≈ 220GB → plan retention. Note: `fileModifiedSinceLastDownload` warnings appear when exporting while upstream uploads a new run — some 64KB blocks fail (export still completes; variables may mix adjacent runs slightly). Mitigation: offset the cron from run-publication times.

## UPDATE 2026-06-12 (evening): multi-model + Hugging Face publication, E2E validated

- `run_solar_regions.sh` now loops MODELS × REGIONS: `dwd_icon` (7d), `ncep_gfs013` (7d, native horizon up to 16d), `jma_msm` (3d). Optional arg runs a single model (`./run_solar_regions.sh jma_msm`) for the extra 3-hourly JMA refreshes. Cron plan (UTC): full at 04:30/10:30/16:30/22:30; jma-only at 01:30/07:30/13:30/19:30.
- After each export the script re-compresses (round(2) + zstd, ~20% smaller, information-lossless) and uploads to HF dataset **JimTseng/apac-nwp-forecast-archive** (public, CC BY 4.0 card in README) under `data/model=<domain>/`. Disable by setting HF_DATASET_REPO="". Requires `hf auth login` + pandas/pyarrow on the host.
- **E2E validated**: full 3-model cycle ran (~40 min incl. uploads; dwd_icon 120k pts/23M rows, gfs 137k pts/26M rows, jma 228k pts/22M rows), all uploaded; `colab_hf_demo.ipynb` (repo root) reads back via hf://, prints stats, plots a 3-model Tokyo GHI comparison and renders the GHI GIF — executed locally end-to-end OK.
- Gotcha: hive path `model=<domain>/` makes pandas auto-add a categorical `model` column on read — exclude it from numeric aggregations.
- Pending: push fork to user's GitHub, deploy on Mac mini (clone → docker build --pull -f Dockerfile.local → pip install pandas pyarrow huggingface_hub → hf auth login with a NEW token (current one was exposed in terminal; rotate) → install cron).

**Related doc:** `RADIATION_INTERPOLATION.md` — how Open-Meteo interpolates 3-hourly native radiation to hourly (clearness-index kt + Hermite + solar geometry), which lead times are interpolated per model, and ML implications.

## Status: paused at the shell script (deliverable ready)

`run_jp_tw_solar.sh` is the deliverable — a parameterized `sync` + `export` pipeline (domain, bbox, vars, forecast days, land-only toggle all at the top). Everything below the script is validated; we deliberately stopped here.

**Radiation validated (2026-06-11):** exported `shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance` for a Taiwan point. Confirmed: `GHI == direct + diffuse` exactly (max err 0.0), correct diurnal bell curve (0 at night, peak ~566 W/m² near solar noon, high diffuse fraction = cloudy forecast), DNI computed. All 13 export variables now confirmed.

**Chunk-coverage lesson:** `.om` data is stored in chunks of `chunk_time_length=253` hours (~10.5 days each). Syncing variables piecemeal at different times gave them mismatched chunks (temperature had 1955+1956, radiation only 1956 → radiation NaN for dates in chunk 1955). The script avoids this by syncing ALL variables in one invocation right before export, so coverage is consistent.

**Architecture decision — store global in `.om`, not parquet:** `sync` already downloads the whole-globe `.om` files (can't fetch sub-regions from S3), so the global data is already on disk in compressed rolling form (~5 GB for these vars, bounded). Do NOT export the whole globe to parquet (4.1M points vs 9.5k for the JP/TW land box = ~440×, multi-GB per run, mostly unused ocean). Keep the global `.om` DB as the source; export only the region(s) actually consumed. Adding a region later = another `export` against the same `.om`, no re-download.

**Current disk state (2026-06-12):** the old `open-meteo-data` volume (partial sync from the local-DB era) was deleted — remote mode needs no local database. Only `open-meteo-cache` (block cache) remains. Output parquets live in `out/`.

## Next steps (optional, when resumed)

1. Run `./run_jp_tw_solar.sh` to completion (first full sync ≈ 1–1.5 h, dominated by slow S3 download ~1 MB/s aggregate; later 6-hourly runs are incremental and fast).
2. Wrap it in a 6-hourly cron (or `sync --repeat-interval 360`).
3. Decide downstream storage/partitioning (e.g. parquet partitioned by run timestamp).
