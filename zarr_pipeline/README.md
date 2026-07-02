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

## Scripts
| script | purpose |
|---|---|
| `go_forward.sh` | daily: export → parquet → Bronze → incremental append into the Silver cube |
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
