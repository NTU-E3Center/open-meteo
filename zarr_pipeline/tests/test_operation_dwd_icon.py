from __future__ import annotations

import subprocess
from pathlib import Path


def test_operation_dwd_icon_dry_run_is_pinned_to_jp_tw_and_nas_zarr() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "zarr_pipeline" / "operation_dwd_icon.sh"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--run",
            "2026-07-12T06:00:00Z",
            "--out",
            "/nas/solar-operation/raw/nwp/dwd_icon_latest.zarr",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--run 2026-07-12T06:00:00" in result.stdout
    assert "--latitude-bounds 20,46" in result.stdout
    assert "--longitude-bounds 119,146" in result.stdout
    assert "dwd_icon_latest.zarr" in result.stdout
    assert "huggingface" not in result.stdout.lower()
