#!/usr/bin/env bash
# Deploy the go-forward NWP pipeline on a Mac mini (or any host) and optionally retire the OLD
# forecast cron. Everything lives in this open-meteo/zarr_pipeline/ dir; run this from here.
#
#   cd open-meteo/zarr_pipeline && ./deploy_goforward.sh              # set up venv + install cron
#   ./deploy_goforward.sh --retire-old                               # also remove old sweep cron
#
# What it does: builds the pipeline venv, preflights (docker image, HF login), installs a daily
# cron (jma 06:00 / dwd 07:00 UTC, --backfill 3), and (with --retire-old) removes ONLY the old
# sweep_data_run forecast cron -- the himawari cron is always kept.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"       # open-meteo/zarr_pipeline
OM="$(dirname "$DIR")"                                     # open-meteo root
VENV="$DIR/.venv"; PY="$VENV/bin/python"
LOG_DIR="${LOG_DIR:-$HOME/om-logs}"; mkdir -p "$LOG_DIR"
CRON_MARK="# go-forward-nwp"
RETIRE=0; [ "${1:-}" = "--retire-old" ] && RETIRE=1

echo "=== 1/5 build venv ($VENV) ==="
if [ ! -x "$PY" ]; then
  python3 -m venv "$VENV"
  "$PY" -m pip -q install --upgrade pip
fi
"$PY" -m pip -q install -r "$DIR/requirements.txt"
"$PY" -c "import xarray,zarr,pandas,pyarrow,numpy,huggingface_hub as h; \
  assert tuple(map(int,h.__version__.split('.')[:2]))>=(1,19), 'huggingface_hub too old: '+h.__version__; \
  print('  deps ok | hub', h.__version__)"

echo "=== 2/5 preflight ==="
command -v docker >/dev/null || { echo "ERROR: docker not found"; exit 1; }
docker image inspect "${IMAGE:-open-meteo:bbox-fix}" >/dev/null 2>&1 || { echo "ERROR: image open-meteo:bbox-fix missing (build it in $OM)"; exit 1; }
"$PY" -c "from huggingface_hub import HfApi; print('  HF user:', HfApi().whoami()['name'])" \
  || { echo "ERROR: not logged in (run: hf auth login)"; exit 1; }
[ -f "$OM/postprocess.py" ] || { echo "ERROR: $OM/postprocess.py missing"; exit 1; }
echo "  ok"

echo "=== 3/5 install daily cron (jma 06:00, dwd 07:00 UTC, --backfill 3) ==="
( crontab -l 2>/dev/null | grep -v "$CRON_MARK" || true
  echo "0 6 * * * cd $DIR && ./go_forward.sh jma_msm  --backfill 3 >> $LOG_DIR/goforward_jma.log 2>&1 $CRON_MARK"
  echo "0 7 * * * cd $DIR && ./go_forward.sh dwd_icon --backfill 3 >> $LOG_DIR/goforward_dwd.log 2>&1 $CRON_MARK"
) | crontab -
echo "  installed:"; crontab -l | grep "$CRON_MARK"

echo "=== 4/5 retire OLD forecast cron (himawari kept!) ==="
if [ "$RETIRE" = 1 ]; then
  crontab -l 2>/dev/null | grep -v "sweep_data_run" | crontab -
  echo "  removed sweep_data_run cron (old forecast .zarr.zip archiving)."
  echo "  himawari cron kept:"; crontab -l 2>/dev/null | grep -i himawari || echo "    (none found -- check manually)"
else
  echo "  --retire-old not given; old sweep cron left in place."
  echo "  when ready:  ./deploy_goforward.sh --retire-old"
fi

echo "=== 5/5 done ==="
echo "  Silver axis is pre-extended to 2028; re-run ./extend_hf_cube.sh before then."
echo "  Smoke test now:  ./go_forward.sh jma_msm"
