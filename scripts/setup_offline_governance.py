#!/usr/bin/env python3
"""Create baseline governance scaffolding for offline repository use.

The script writes missing directories and template files needed by local DVC
and MLflow workflows. Existing files are preserved so repeated runs remain
safe and idempotent apart from manifest regeneration.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_if_missing(path: Path, text: str) -> bool:
    """Write a text file only when it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    path.write_text(text, encoding="utf-8")
    return True


def ensure_dir(path: Path) -> None:
    """Create a directory tree if it is missing."""
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    """Create the governance scaffold and emit a manifest of results."""
    created: list[str] = []
    existing: list[str] = []

    dirs = [
        ROOT / "artifacts" / "mlruns",
        ROOT / "artifacts" / "governance",
        ROOT / "dvc_store" / "local",
        ROOT / "environment_lockfiles",
    ]
    for d in dirs:
        if d.exists():
            existing.append(str(d))
        else:
            ensure_dir(d)
            created.append(str(d))

    files = {
        ROOT / ".dvc" / "config": """[core]\n    no_scm = true\n[cache]\n    dir = .dvc/cache\n[remote \"local_offline\"]\n    url = dvc_store/local\n""",
        ROOT / ".dvcignore": ".cache/\nartifacts/batch_runs/\n__pycache__/\n*.pyc\n",
        ROOT / "dvc.yaml": """stages:\n  ami_pipeline_run:\n    cmd: PYTHONPATH=src python3 -m ami_mom_pipeline --config configs/pipeline.nemo.llama.yaml run --meeting-id ES2005a\n    deps:\n      - src/\n      - scripts/\n      - configs/pipeline.nemo.llama.yaml\n      - data/rawa/ami/\n      - models/\n    outs:\n      - artifacts/ami/ES2005a\n      - artifacts/eval/ami\n""",
        ROOT / "environment_lockfiles" / "README.md": (
            "# Environment Lockfiles\n\n"
            "Store pinned conda/pip lockfiles here for offline reproducibility.\n\n"
            "Examples:\n"
            "- `conda-linux-64.lock`\n"
            "- `requirements-pinned.txt`\n"
        ),
        ROOT / "scripts" / "run_mlflow_offline.sh": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\"\n"
            "mkdir -p \"$ROOT/artifacts/mlruns\"\n"
            "exec mlflow ui --backend-store-uri \"file:$ROOT/artifacts/mlruns\" --host 127.0.0.1 --port 5001\n"
        ),
        ROOT / "scripts" / "dvc_offline_init.sh": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\"\n"
            "cd \"$ROOT\"\n"
            "dvc init --no-scm || true\n"
            "dvc remote add -d local_offline dvc_store/local || dvc remote modify local_offline url dvc_store/local\n"
            "echo \"DVC offline remote configured: local_offline -> dvc_store/local\"\n"
        ),
    }

    for path, text in files.items():
        if write_if_missing(path, text):
            created.append(str(path))
        else:
            existing.append(str(path))

    # Best-effort make shell scripts executable.
    for rel in ["scripts/run_mlflow_offline.sh", "scripts/dvc_offline_init.sh"]:
        p = ROOT / rel
        if p.exists():
            p.chmod(p.stat().st_mode | 0o111)

    manifest = {
        "purpose": "offline_governance_scaffold",
        "created": sorted(created),
        "existing": sorted(existing),
        "mlflow_tracking_uri_recommendation": f"file:{ROOT / 'artifacts' / 'mlruns'}",
        "dvc_remote_recommendation": str(ROOT / "dvc_store" / "local"),
    }
    manifest_path = ROOT / "artifacts" / "governance" / "offline_governance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "created_count": len(created), "existing_count": len(existing)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
