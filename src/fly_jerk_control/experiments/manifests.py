"""Run directory manifests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fly_jerk_control.training.checkpoints import git_commit, package_versions


def timestamp_run_dir(root: str | Path, name: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(root) / f"{ts}_{name}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)
    (path / "trajectories").mkdir(exist_ok=True)
    return path


def write_manifest(run_dir: Path, cfg: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "versions": package_versions(),
        "label": cfg.get("label"),
        "extra": extra or {},
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
