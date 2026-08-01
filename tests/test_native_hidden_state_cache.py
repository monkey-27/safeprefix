import json

import pytest
import torch

from safeprefix.native_hidden_state_cache import (
    build_hidden_state_packs,
    sha256_file,
    validate_feature_pack,
    validate_source_cohort,
)
from safeprefix.reproducibility import atomic_json, atomic_jsonl, stable_hash


def _trace(index: int) -> dict:
    return {
        "trace_id": f"trace-{index}",
        "model_key": "m",
        "model_id": "model",
        "model_revision": "rev",
        "tokenizer_id": "tokenizer",
        "tokenizer_revision": "tok-rev",
        "initial_status": "valid_incorrect",
        "parser_status": "parsed",
        "verifier_status": "completed",
        "truncation_flag": False,
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3, 4, 5],
    }


def test_hidden_state_packs_are_deterministic_and_preserve_order():
    cohort = [_trace(index) for index in range(5)]
    first = build_hidden_state_packs(cohort, model_key="m", pack_size=2, configuration_hash="cfg")
    second = build_hidden_state_packs(cohort, model_key="m", pack_size=2, configuration_hash="cfg")
    assert first == second
    assert [len(row["trace_ids"]) for row in first] == [2, 2, 1]
    assert [item for row in first for item in row["trace_ids"]] == [f"trace-{i}" for i in range(5)]


def test_source_cohort_rejects_wrong_revision_and_truncation():
    cohort = [_trace(0)]
    result = validate_source_cohort(
        cohort, model_key="m", model_revision="rev", tokenizer_revision="tok-rev"
    )
    assert result["trace_count"] == 1
    bad = [{**cohort[0], "truncation_flag": True}]
    with pytest.raises(RuntimeError, match="truncated"):
        validate_source_cohort(
            bad, model_key="m", model_revision="rev", tokenizer_revision="tok-rev"
        )


def test_feature_pack_integrity_checks_exact_tokens_and_shapes(tmp_path):
    trace = _trace(0)
    pack = build_hidden_state_packs(
        [trace], model_key="m", pack_size=1, configuration_hash="cfg"
    )[0]
    root = tmp_path / pack["pack_id"]
    root.mkdir()
    payload = {
        trace["trace_id"]: {
            "trace_id": trace["trace_id"],
            "model_revision": "rev",
            "tokenizer_revision": "tok-rev",
            "prompt_token_ids_hash": stable_hash(trace["prompt_token_ids"]),
            "completion_token_ids_hash": stable_hash(trace["completion_token_ids"]),
            "hidden_states": torch.ones(4, 8, dtype=torch.float16),
            "completion_token_log_probabilities": torch.ones(3),
            "kv_cache_persisted": False,
        }
    }
    torch.save(payload, root / "hidden_states.pt")
    metadata = [{
        "trace_id": trace["trace_id"],
        "hidden_state_shape": [4, 8],
    }]
    atomic_jsonl(root / "trace_metadata.jsonl", metadata)
    atomic_json(root / "complete.json", {
        "status": "COMPLETE",
        "pack_id": pack["pack_id"],
        "trace_count": 1,
        "features_sha256": sha256_file(root / "hidden_states.pt"),
        "metadata_sha256": sha256_file(root / "trace_metadata.jsonl"),
    })
    result = validate_feature_pack(
        root, pack=pack, cohort_index={trace["trace_id"]: trace},
        expected_model_revision="rev", expected_tokenizer_revision="tok-rev",
    )
    assert result["status"] == "COMPLETE"

    payload[trace["trace_id"]]["hidden_states"][0, 0] = float("nan")
    torch.save(payload, root / "hidden_states.pt")
    marker = json.loads((root / "complete.json").read_text())
    marker["features_sha256"] = sha256_file(root / "hidden_states.pt")
    atomic_json(root / "complete.json", marker)
    with pytest.raises(RuntimeError, match="non-finite"):
        validate_feature_pack(
            root, pack=pack, cohort_index={trace["trace_id"]: trace},
            expected_model_revision="rev", expected_tokenizer_revision="tok-rev",
        )
