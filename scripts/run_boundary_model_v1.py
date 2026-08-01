#!/usr/bin/env python3
"""Resumable command-line entry point for SafePrefix boundary model v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safeprefix.boundary_v1.data import prepare_boundary_dataset
from safeprefix.boundary_v1.evaluation import finalize_selected_model, select_architecture
from safeprefix.boundary_v1.reporting import generate_reports
from safeprefix.boundary_v1.training import train_model_matrix


DEFAULT_CONFIG = Path("configs/boundary_model_v1.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--artifact-root", type=Path, required=True)
    prepare.add_argument("--old-root", type=Path, required=True)
    prepare.add_argument("--completion-root", type=Path, required=True)
    prepare.add_argument("--manifest-root", type=Path, required=True)

    train = subparsers.add_parser("train-model")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--artifact-root", type=Path, required=True)
    train.add_argument("--model-key", required=True)
    train.add_argument("--device", default="cpu")

    select = subparsers.add_parser("select")
    select.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    select.add_argument("--artifact-root", type=Path, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    finalize.add_argument("--artifact-root", type=Path, required=True)
    finalize.add_argument("--device", default="cpu")

    report = subparsers.add_parser("report")
    report.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    report.add_argument("--artifact-root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare_boundary_dataset(
            config_path=args.config,
            old_root=args.old_root,
            completion_root=args.completion_root,
            manifest_root=args.manifest_root,
            artifact_root=args.artifact_root,
        )
    elif args.command == "train-model":
        result = train_model_matrix(
            config_path=args.config,
            artifact_root=args.artifact_root,
            model_key=args.model_key,
            device=args.device,
        )
    elif args.command == "select":
        result = select_architecture(config_path=args.config, artifact_root=args.artifact_root)
    elif args.command == "finalize":
        result = finalize_selected_model(
            config_path=args.config,
            artifact_root=args.artifact_root,
            device_name=args.device,
        )
    else:
        result = generate_reports(config_path=args.config, artifact_root=args.artifact_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
