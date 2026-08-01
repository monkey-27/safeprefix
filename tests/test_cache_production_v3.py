from types import SimpleNamespace

import pytest
import numpy as np
import torch

from safeprefix.models.cache_checkpoint import capture_cache_checkpoint
from safeprefix.models.generation import (
    decode_from_checkpoint,
    deterministic_branch_seed,
)
from safeprefix.models.production_backend import (
    assert_production_backend,
    validate_batching_fairness,
)
from safeprefix.models.production_validation import (
    cache_independence_diagnostics,
    run_production_checkpoint_validation,
    validate_branch_batching,
)
from safeprefix.models.validation_manifest import (
    build_common_validation_cohort,
    materialize_model_manifest,
)
from safeprefix.testing import CharacterTokenizer, StochasticToyCausalModel, ToyCausalModel


def _checkpoint(prefix_length: int = 4, model=None):
    model = (model or ToyCausalModel()).eval()
    ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
    output = model(input_ids=ids[:, :prefix_length], attention_mask=torch.ones((1, prefix_length), dtype=torch.long), use_cache=True, return_dict=True)
    checkpoint = capture_cache_checkpoint(
        output.past_key_values,
        output.logits[:, -1],
        ids,
        prefix_length,
        model_id="mock/toy",
        model_revision="mock-v1",
        tokenizer_id="mock/character",
        tokenizer_revision="mock-v1",
    )
    return model, ids, checkpoint


def test_branch_stream_is_id_based_and_order_invariant() -> None:
    model, _ids, checkpoint = _checkpoint()
    tokenizer = CharacterTokenizer()
    branch_ids = ["alpha", "beta", "gamma", "delta"]
    generation = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 8}

    def run(order):
        seeds = [deterministic_branch_seed(991, branch_id) for branch_id in order]
        result = decode_from_checkpoint(model, tokenizer, checkpoint, branch_ids=order, branch_seeds=seeds, generation=generation)
        return {branch_id: completion.token_ids for branch_id, completion in zip(order, result.completions)}

    expected = run(branch_ids)
    assert run(list(reversed(branch_ids))) == expected
    assert run(["gamma", "alpha", "delta", "beta"]) == expected


def test_duplicate_branch_ids_are_rejected() -> None:
    model, _ids, checkpoint = _checkpoint()
    with pytest.raises(ValueError, match="unique"):
        decode_from_checkpoint(
            model,
            CharacterTokenizer(),
            checkpoint,
            branch_ids=["same", "same"],
            branch_seeds=[1, 2],
            generation={"temperature": 0.0, "max_new_tokens": 2},
        )


def test_singleton_batch_mixed_length_and_clone_independence() -> None:
    model = StochasticToyCausalModel().eval()
    model, _ids, checkpoint = _checkpoint(model=model)
    result = validate_branch_batching(
        model,
        CharacterTokenizer(),
        checkpoint,
        batch_sizes=[1, 2, 4, 8],
        generation={"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 8},
        base_seed=271,
    )
    assert result["passed"]
    assert result["mixed_length"]["passed"]
    assert result["mixed_length"]["observed"]
    assert all(value["passed"] for value in result["branch_order_invariance"].values())
    assert cache_independence_diagnostics(checkpoint, repeats=8)["passed"]


@pytest.mark.parametrize("prefix_length", [1, 2, 4, 7])
def test_position_mask_off_by_one_cases(prefix_length: int) -> None:
    model, ids, _checkpoint_value = _checkpoint(prefix_length)
    result = run_production_checkpoint_validation(
        model,
        CharacterTokenizer(),
        ids,
        prefix_length,
        metadata={"model_id": "mock/toy", "model_revision": "mock-v1", "tokenizer_id": "mock/character", "tokenizer_revision": "mock-v1"},
        generation={"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 4},
        seed=17,
        batch_sizes=[1, 2],
        top_k=5,
        require_mixed_termination=False,
    )
    assert result["passed"]
    audit = result["test_5_position_mask_consistency"]
    assert audit["prefix_length"] == prefix_length
    assert audit["cache_position"] == [prefix_length]


def _manifest_rows(count: int = 120):
    rows = []
    for index in range(count):
        size = 10 + (index % 30)
        rows.append(
            {
                "problem_id": f"p-{index:04d}",
                "source_dataset": "mock/arithmetic" if index % 2 == 0 else "mock/word",
                "source_subset": "arithmetic" if index % 2 == 0 else "word_problem",
                "problem_text": ("Compute " if index % 2 == 0 else "Reason about ") + f"item {index}: " + "x" * size,
                "reasoning_steps": ["step " + "y" * (size * factor) for factor in (1, 2, 3)],
            }
        )
    return rows


def test_validation_manifest_is_deterministic_stratified_and_pass_blind() -> None:
    left, left_exclusions = build_common_validation_cohort(_manifest_rows(), count=100, seed=2701)
    right, right_exclusions = build_common_validation_cohort(reversed(_manifest_rows()), count=100, seed=2701)
    assert left == right
    assert left_exclusions == right_exclusions == []
    assert {row["domain"] for row in left} == {"arithmetic", "word_problem"}
    assert {row["length_bin"] for row in left} == {"short", "medium", "long"}
    manifest, exclusions = materialize_model_manifest(
        left,
        CharacterTokenizer(),
        model_key="mock",
        max_context_length=4096,
        continuation_tokens=32,
        multiple_checkpoint_examples=20,
        checkpoint_fractions=[0.15, 0.5, 0.85],
        min_prefix_tokens=8,
    )
    assert len(manifest) == 100
    assert exclusions == []
    assert sum(row["multiple_checkpoint_example"] for row in manifest) == 20
    assert sum(row["long_prefix_stress"] for row in manifest) >= 10
    assert all(len(row["checkpoint_offsets"]) == (3 if row["multiple_checkpoint_example"] else 1) for row in manifest)


def test_validation_manifest_accepts_parquet_numpy_reasoning_arrays() -> None:
    rows = _manifest_rows()
    for row in rows:
        row["reasoning_steps"] = np.asarray(row["reasoning_steps"], dtype=object)
    cohort, exclusions = build_common_validation_cohort(rows, count=100, seed=2701)
    assert len(cohort) == 100
    assert exclusions == []
    assert all(isinstance(row["reasoning_steps"], list) for row in cohort)


def test_production_backend_assertion_and_tensor_parallel_metadata() -> None:
    model = ToyCausalModel().to(dtype=torch.bfloat16).eval()
    model.config._attn_implementation = "sdpa"
    model.config._commit_hash = "resolved-model-sha"
    tokenizer = CharacterTokenizer()
    entry = {
        "hf_model_id": "mock/toy",
        "tokenizer_id": "mock/character",
        "revision": "main",
        "tokenizer_revision": "main",
        "dtype": "bf16",
        "attn_implementation": "sdpa",
        "production": {
            "enabled": True,
            "dtype": "bf16",
            "attn_implementation": "sdpa",
            "tensor_parallel_degree": 1,
            "gpu_type": "mock",
            "gpu_count": 1,
            "branch_batch_size": 8,
            "cache_topology": "same_worker_memory",
            "torch_compile": False,
        },
    }
    signature = assert_production_backend(model, tokenizer, entry, model_key="mock")
    assert signature["resolved_attention_backend"] == "sdpa"
    assert signature["tensor_parallel_degree"] == 1
    assert signature["model_revision"] == "resolved-model-sha"
    model.config._attn_implementation = "eager"
    with pytest.raises(RuntimeError, match="requested SDPA"):
        assert_production_backend(model, tokenizer, entry, model_key="mock")


def test_batching_fairness_requires_validated_shapes_or_fixed_padding() -> None:
    result = validate_batching_fairness(
        {"safeprefix": 8, "full_regeneration": 1},
        validated_invariant_sizes={1, 2, 4, 8},
    )
    assert result["policy"] == "validated_shape_invariance"
    with pytest.raises(RuntimeError, match="unvalidated"):
        validate_batching_fairness(
            {"safeprefix": 8, "full_regeneration": 3},
            validated_invariant_sizes={1, 2, 4, 8},
        )
    fixed = validate_batching_fairness(
        {"safeprefix": 8, "full_regeneration": 8},
        validated_invariant_sizes={1},
        fixed_padded_batch_size=8,
    )
    assert fixed["policy"] == "fixed_padded_batch"
