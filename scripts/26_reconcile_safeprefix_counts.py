#!/usr/bin/env python3
"""Build ignored reconciliation inputs for teacher-forced completion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeprefix.count_reconciliation import reconcile_counts


DEFAULT_FROZEN_ROOT = Path("/private/tmp/safeprefix_frozen_manifests")
DEFAULT_PRODUCTION_ROOT = Path(
    "/private/tmp/safeprefix_final_download/"
    "safeprefix_full_teacher_forced_20260726_r3/artifacts/full_teacher_forced_suite"
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", type=Path, default=DEFAULT_FROZEN_ROOT)
    parser.add_argument("--production-root", type=Path, default=DEFAULT_PRODUCTION_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/count_reconciliation",
    )
    args = parser.parse_args()

    summary, lineage, counts = reconcile_counts(
        enriched_path=args.frozen_root / "enriched_examples.parquet",
        frozen_root=args.frozen_root,
        production_root=args.production_root,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    lineage.to_parquet(args.output_root / "trace_filter_lineage.parquet", index=False)
    counts.to_csv(args.output_root / "trace_category_counts.csv", index=False)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["git_commit"] = _git_commit()
    (args.output_root / "count_reconciliation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
