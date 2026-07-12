# zarr_pipeline — APAC NWP forecast → Bronze (parquet) + Silver (Zarr Mode A)

Operational pipeline that turns Open-Meteo model runs into a single, ever-growing Silver Zarr
cube on Hugging Face, plus a per-run parquet Bronze layer. Lives inside the open-meteo repo so it
can reuse the exporter Docker image and `postprocess.py`.

## HF datasets
| dataset | content |
|---|---|
| `jimtseng/apac-nwp-forecast` | **Silver** — Mode A Zarr cube `(run_init, lead, lat, lon)`, sharded, axis pre-extended to 2028. The product downstream should use. |
| `jimtseng/apac-nwp-forecast-raw` | **Bronze** — per-run parquet |
| `jimtseng/apac-nwp-forecast-zip` | historical `.zarr.zip` (cold archive; frozen) |

## Setup (once, on a new host)
```bash
cd open-meteo/zarr_pipeline
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# needs: Docker running + image `open-meteo:bbox-fix` (build in ../), and `hf auth login`
```

## Daily operation
```bash
./go_forward.sh jma_msm  --backfill 3      # fill any run missing in the last 3 days
./go_forward.sh dwd_icon --backfill 3
```
`deploy_goforward.sh` installs these as a daily cron (and, with `--retire-old`, removes the old
`sweep_data_run` forecast cron while KEEPING the himawari cron).

## Direct PV operation ingestion

`go_forward.sh` is archive automation and must not be used as the hot serving input. For the
PV operation pipeline, run the following job independently after a DWD ICON run is published:

```bash
ZARR_PY="$PWD/.venv/bin/python" ./operation_dwd_icon.sh \
  --out /nas/solar-operation/raw/nwp/dwd_icon_latest.zarr
```

It resolves the latest DWD ICON S3 run, pins the exporter with `--run`, fetches only the
Japan/Taiwan box (`lat=20..46`, `lon=119..146`), postprocesses one parquet, and writes one
unpacked Zarr directory `(run_init, lead, latitude, longitude)` on NAS. It does not upload to
Hugging Face or modify Bronze/Silver. Use `--run 2026-07-12T06:00:00Z --dry-run` to inspect a
specific run before execution.

## Scripts
| script | purpose |
|---|---|
| `go_forward.sh` | daily: export → parquet → Bronze → incremental append into the Silver cube |
| `operation_dwd_icon.sh` | high-frequency: run-pinned DWD ICON → NAS `dwd_icon_latest.zarr`; no HF writes |
| `deploy_goforward.sh` | build venv + preflight + install cron (+ `--retire-old`) |
| `extend_hf_cube.sh` / `extend_run_init.py` | extend the Silver run_init axis (metadata-only); **re-run before 2028** |
| `convert_to_zarr.py` | core: parquet/.zarr.zip → Mode A (`--source/--target-mb/--init/--append/--model`) |
| `migrate_{jma,dwd}_silver.sh`, `migration_precheck.py` | rebuild Silver from the `.zarr.zip` archive (reference / disaster recovery) |
| `reshard_silver{,_lowdisk}.py`, `paced_upload.py` | re-shard + rate-limited bulk upload (used when rebuilding) |

## Examples
- [examples/partial_read_demo.ipynb](examples/partial_read_demo.ipynb) — load only what you need
  (Taiwan region / specific variables / specific runs) straight from HF, confirming partial/lazy
  reads. Run with: `pip install -r requirements.txt -r examples/requirements-notebook.txt`.

Full design + history: [docs/migration-plan.md](docs/migration-plan.md).
