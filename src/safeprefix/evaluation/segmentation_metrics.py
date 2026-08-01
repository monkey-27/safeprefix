"""Automated and human-audit segmentation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def segmentation_metrics(rows: list[dict[str, Any]], human_valid_labels: list[bool] | None = None) -> dict[str, Any]:
    if not rows:
        return {"traces": 0, "exact_token_segmentation": 0.0, "valid_segmentation": None}
    lengths = [int(span["token_end"]) - int(span["token_start"]) for row in rows for span in row.get("spans", [])]
    valid = [
        bool(row.get("spans"))
        and all(int(span["token_start"]) < int(span["token_end"]) for span in row["spans"])
        and all(a["token_end"] == b["token_start"] for a, b in zip(row["spans"], row["spans"][1:]))
        for row in rows
    ]
    length_distribution = {
        "minimum": float(np.min(lengths)), "q25": float(np.quantile(lengths, 0.25)),
        "median": float(np.median(lengths)), "mean": float(np.mean(lengths)),
        "q75": float(np.quantile(lengths, 0.75)), "maximum": float(np.max(lengths)),
    } if lengths else {"minimum": None, "q25": None, "median": None, "mean": None, "q75": None, "maximum": None}
    return {
        "traces": len(rows),
        "exact_token_segmentation": float(np.mean(valid)),
        "valid_segmentation": float(np.mean(human_valid_labels)) if human_valid_labels else None,
        "human_valid_label_count": len(human_valid_labels or []),
        "span_count_mean": float(np.mean([len(row.get("spans", [])) for row in rows])),
        "span_token_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "span_token_length_median": float(np.median(lengths)) if lengths else 0.0,
        "span_token_length_distribution": length_distribution,
    }
