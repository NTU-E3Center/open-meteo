# jma_msm_silver.zarr axis-prepend migration (one-time, user-executed)

Freeze window required: pause the ingestion cron before step 2; resume at step 7.
PY=/Users/wctseng/Desktop/projects/solar-ghi-nwp/.venv/bin/python
WORK=~/silver_migration   # needs ~2x current jma store size free

1. Download the current store (source of truth is HF):
   $PY -c "from huggingface_hub import snapshot_download; \
     snapshot_download('jimtseng/apac-nwp-forecast', repo_type='dataset', \
     allow_patterns='jma_msm_silver.zarr/**', local_dir='$WORK')"
   （若 ~/silver_migration/jma_msm_silver.zarr 已存在（先前預抓），跳過此步，直接用它。）

2. FREEZE the cron that appends jma_msm.

3. Prepend (dry-run first, then real):
   cd ~/Desktop/projects/open-meteo/zarr_pipeline
   $PY prepend_run_init.py $WORK/jma_msm_silver.zarr --start 2017-12-01 --dry-run
   $PY prepend_run_init.py $WORK/jma_msm_silver.zarr --start 2017-12-01

4. Verify bit-exact (all previously-filled slots, every variable):
   $PY - <<'EOF'
   import numpy as np, pandas as pd, xarray as xr
   new = xr.open_zarr("$WORK/jma_msm_silver.zarr".replace("$WORK", __import__('os').path.expanduser("~/silver_migration")), consolidated=True)
   old = xr.open_zarr("hf://datasets/jimtseng/apac-nwp-forecast/jma_msm_silver.zarr", consolidated=None)
   ri = pd.DatetimeIndex(old.run_init.values)[old.slot_filled.values > 0]
   import random; random.seed(0)
   for ts in [ri[0], ri[-1]] + random.sample(list(ri), 10):
       for v in [x for x in old.data_vars if x != "slot_filled"]:
           a = old[v].sel(run_init=ts).values; b = new[v].sel(run_init=ts).values
           assert np.array_equal(a, b, equal_nan=True), f"MISMATCH {ts} {v}"
       print(ts, "ok")
   sf = new.slot_filled.values
   assert int(sf.sum()) == int(old.slot_filled.values.sum())
   print("VERIFIED")
   EOF
   (12 sampled runs x 14 vars; escalate to all 388 if paranoid — same loop over `ri`.)

5. Replace on HF (delete old folder, then paced upload):
   $PY -c "from huggingface_hub import HfApi; \
     HfApi().delete_folder(path_in_repo='jma_msm_silver.zarr', \
     repo_id='jimtseng/apac-nwp-forecast', repo_type='dataset', \
     commit_message='migrate: prepend run_init axis to 2017-12-01 (re-upload follows)')"
   $PY paced_upload.py jimtseng/apac-nwp-forecast $WORK
   (paced_upload is resumable; rerun until it reports nothing left to upload.)

6. Spot-check the HF copy: open remotely, repeat step 4 for 2 runs.

7. RESUME the cron, then run one catch-up append over the runs collected during the
   freeze (idempotent; writer matches slots by timestamp against the new axis — verified
   in convert_to_zarr.append_runs, no code change needed).

Rollback: HF dataset git history — revert the delete/upload commits.
Cleanup: rm -rf $WORK after a week of healthy operation.
