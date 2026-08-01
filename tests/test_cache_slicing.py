import json
import multiprocessing
from pathlib import Path

import torch

from safeprefix.models.cache_checkpoint import (
    capture_cache_checkpoint,
    clone_cache_checkpoint,
    compare_caches,
    load_cache_checkpoint,
    repeat_cache_checkpoint,
    save_cache_checkpoint,
)
from safeprefix.models.cache_diagnostics import run_cache_diagnostics
from safeprefix.models.cache_utils import (
    cache_to_legacy,
    cache_sequence_length,
    forward_replay_diagnostic,
    prepare_replay_diagnostic_inputs,
    repeat_past_key_values,
    slice_past_key_values,
)
from safeprefix.models.generation import (
    continue_from_checkpoint_dynamic_microbatch,
    continue_from_checkpoint_fixed_microbatch,
    decode_from_checkpoint,
    generate_prompt_completion_fixed_microbatch,
    run_matched_shape_smoke,
    select_tokens_from_saved_logits,
)
from safeprefix.models.teacher_forcing import teacher_force
from safeprefix.testing import CharacterTokenizer, ToyCausalModel


def _continue_serialized_checkpoint_in_child(checkpoint_path: str, output_path: str) -> None:
    """Exercise deserialization and decoding in an independent OS process."""
    checkpoint = load_cache_checkpoint(checkpoint_path)
    model = ToyCausalModel().eval()
    generation = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 8}
    tokens = decode_from_checkpoint(
        model,
        CharacterTokenizer(),
        checkpoint,
        branch_seeds=[73],
        generation=generation,
    ).completions[0].token_ids
    Path(output_path).write_text(json.dumps(tokens), encoding="utf-8")


def _cache():
    key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 6, 2)
    return ((key, key + 100),)


def test_legacy_cache_slice_and_repeat() -> None:
    sliced = slice_past_key_values(_cache(), 4)
    assert cache_sequence_length(sliced) == 4
    assert torch.equal(sliced[0][0], _cache()[0][0][..., :4, :])
    repeated = repeat_past_key_values(sliced, 3)
    assert repeated[0][0].shape[0] == 3
    assert cache_sequence_length(repeated) == 4


def test_rejected_replay_diagnostic_remains_explicit() -> None:
    model = ToyCausalModel()
    ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    full = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, return_dict=True)
    prefix = model(input_ids=ids[:, :4], attention_mask=torch.ones((1, 4), dtype=torch.long), use_cache=True, return_dict=True)
    restored = forward_replay_diagnostic(model, prepare_replay_diagnostic_inputs(full.past_key_values, ids, 4))
    assert torch.equal(prefix.logits[:, -1], restored.logits[:, -1])
    assert cache_sequence_length(restored.past_key_values) == 4


def test_dynamic_cache_when_transformers_available() -> None:
    from transformers import DynamicCache

    cache = DynamicCache.from_legacy_cache(_cache())
    sliced = slice_past_key_values(cache, 3)
    assert cache_sequence_length(sliced) == 3
    ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    checkpoint = capture_cache_checkpoint(
        cache,
        torch.arange(256, dtype=torch.float32)[None],
        ids,
        4,
        model_id="mock/dynamic",
        model_revision="mock-v1",
        tokenizer_id="mock/character",
        tokenizer_revision="mock-v1",
    )
    cloned = clone_cache_checkpoint(checkpoint)
    assert type(cloned.past_key_values) is type(cache)
    assert compare_caches(checkpoint.past_key_values, cloned.past_key_values)["passed"]
    repeated = repeat_cache_checkpoint(checkpoint, 3)
    assert type(repeated.past_key_values) is type(cache)
    assert cache_to_legacy(repeated.past_key_values)[0][0].shape[0] == 3


def test_teacher_forcing_preserves_native_prompt_boundary() -> None:
    class BoundaryMergingTokenizer:
        def __call__(self, text, **_kwargs):
            return {"input_ids": {"a": [10], "b": [11], "ab": [12]}[text]}

    forced = teacher_force(ToyCausalModel(), BoundaryMergingTokenizer(), "a", "b")
    assert forced.prompt_token_count == 1
    assert forced.token_ids.tolist() == [[10, 11]]


def _checkpoint(model: ToyCausalModel | None = None):
    model = model or ToyCausalModel()
    model.eval()
    ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, return_dict=True)
    checkpoint = capture_cache_checkpoint(
        output.past_key_values,
        output.logits[:, 3],
        ids,
        4,
        attention_mask=torch.ones_like(ids),
        model_id="mock/toy",
        model_revision="mock-v1",
        tokenizer_id="mock/character",
        tokenizer_revision="mock-v1",
        tokenizer_metadata={"eos_token_id": 0, "pad_token_id": 0, "padding_side": "right"},
        generation_metadata={"temperature": 0.0, "max_new_tokens": 4},
    )
    return model, ids, checkpoint


def test_complete_checkpoint_includes_final_prefix_token_and_saved_logits() -> None:
    _model, ids, checkpoint = _checkpoint()
    checkpoint.validate()
    assert checkpoint.protocol_version == "complete_cache_saved_next_logits_v2"
    assert cache_sequence_length(checkpoint.past_key_values) == 4
    assert checkpoint.prefix_token_ids.tolist() == ids[:, :4].tolist()
    assert checkpoint.position_ids.tolist() == [[4]]
    assert checkpoint.cache_position.tolist() == [4]
    assert checkpoint.next_token_logits.dtype == torch.float32
    assert checkpoint.next_token_logits.argmax(-1).tolist() == [14]
    assert checkpoint.tokenizer_metadata["padding_side"] == "right"


def test_clone_repeat_and_serialization_are_independent(tmp_path) -> None:
    _model, _ids, checkpoint = _checkpoint()
    cloned = clone_cache_checkpoint(checkpoint)
    original_key = cache_to_legacy(checkpoint.past_key_values)[0][0]
    cloned_key = cache_to_legacy(cloned.past_key_values)[0][0]
    cloned_key[0, 0, 0, 0] += 100
    assert not torch.equal(original_key, cloned_key)

    repeated = repeat_cache_checkpoint(checkpoint, 3)
    repeated_key = cache_to_legacy(repeated.past_key_values)[0][0]
    row_one_before = repeated_key[1].clone()
    repeated_key[0] += 50
    assert torch.equal(repeated_key[1], row_one_before)
    assert repeated.next_token_logits.shape[0] == 3

    path = tmp_path / "checkpoint.pt"
    save_cache_checkpoint(checkpoint, path)
    loaded = load_cache_checkpoint(path)
    assert compare_caches(checkpoint.past_key_values, loaded.past_key_values)["passed"]
    assert torch.equal(checkpoint.next_token_logits, loaded.next_token_logits)
    assert torch.equal(checkpoint.prefix_token_ids, loaded.prefix_token_ids)


def test_cache_comparison_treats_only_paired_nans_as_equal() -> None:
    left = ((torch.tensor([[[[1.0, float("nan")]]]]), torch.tensor([[[[2.0]]]])),)
    right = ((torch.tensor([[[[1.0, float("nan")]]]]), torch.tensor([[[[2.0]]]])),)
    matched = compare_caches(left, right)
    assert matched["passed"]
    assert matched["rows"][0]["paired_nan_count"] == 1
    assert matched["rows"][0]["unpaired_nan_count"] == 0
    assert matched["rows"][0]["max_absolute_difference"] == 0.0

    unpaired = ((torch.tensor([[[[1.0, 0.0]]]]), torch.tensor([[[[2.0]]]])),)
    mismatch = compare_caches(left, unpaired)
    assert not mismatch["passed"]
    assert mismatch["rows"][0]["unpaired_nan_count"] == 1
    assert mismatch["rows"][0]["max_absolute_difference"] == float("inf")


def test_first_suffix_token_comes_from_saved_logits_without_replay() -> None:
    class RecordingToy(ToyCausalModel):
        def __init__(self) -> None:
            super().__init__(); self.forward_inputs: list[list[int]] = []

        def forward(self, input_ids, **kwargs):
            self.forward_inputs.extend(input_ids.detach().cpu().tolist())
            return super().forward(input_ids, **kwargs)

    model = RecordingToy(); model.eval()
    _model, _ids, checkpoint = _checkpoint(model)
    model.forward_inputs.clear()
    trace = decode_from_checkpoint(
        model,
        CharacterTokenizer(),
        checkpoint,
        branch_seeds=[17],
        generation={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 3},
        capture_diagnostics=True,
    )
    assert trace.first_suffix_token_ids == [14]
    assert model.forward_inputs[0] == [14]
    assert [13] not in model.forward_inputs
    assert trace.position_audit[0]["cache_sequence_length_before"] == 4
    assert trace.position_audit[0]["cache_position"] == [4]
    assert trace.position_audit[0]["attention_mask_shape"] == [1, 5]


def test_fixed_microbatch_pads_every_execution_to_production_shape() -> None:
    class BatchRecordingToy(ToyCausalModel):
        def __init__(self) -> None:
            super().__init__()
            self.decode_batch_sizes: list[int] = []

        def forward(self, input_ids, past_key_values=None, **kwargs):
            if past_key_values is not None:
                self.decode_batch_sizes.append(int(input_ids.shape[0]))
            return super().forward(input_ids, past_key_values=past_key_values, **kwargs)

    model = BatchRecordingToy().eval()
    _unused, _ids, checkpoint = _checkpoint(model)
    result = continue_from_checkpoint_fixed_microbatch(
        model,
        CharacterTokenizer(),
        checkpoint,
        branch_seeds=[11, 12, 13, 14, 15],
        branch_ids=[f"branch-{index}" for index in range(5)],
        generation={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 3},
        microbatch_size=4,
    )
    assert len(result.completions) == 5
    assert result.microbatch_count == 2
    assert result.padded_branch_count == 3
    assert model.decode_batch_sizes and set(model.decode_batch_sizes) == {4}


def test_dynamic_microbatch_executes_only_real_branches() -> None:
    class BatchRecordingToy(ToyCausalModel):
        def __init__(self) -> None:
            super().__init__(); self.decode_batch_sizes: list[int] = []

        def forward(self, input_ids, past_key_values=None, **kwargs):
            if past_key_values is not None:
                self.decode_batch_sizes.append(int(input_ids.shape[0]))
            return super().forward(input_ids, past_key_values=past_key_values, **kwargs)

    model = BatchRecordingToy().eval()
    _unused, _ids, checkpoint = _checkpoint(model)
    result = continue_from_checkpoint_dynamic_microbatch(
        model,
        CharacterTokenizer(),
        checkpoint,
        branch_seeds=[11, 12, 13, 14, 15],
        branch_ids=[f"branch-{index}" for index in range(5)],
        generation={"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 3},
        maximum_microbatch_size=4,
    )
    assert len(result.completions) == 5
    assert result.microbatch_count == 2
    assert result.padded_branch_count == 0
    assert result.requested_output_tokens == result.executed_output_tokens
    assert set(model.decode_batch_sizes) == {1, 4}


def test_dynamic_microbatch_is_branch_order_invariant() -> None:
    model, _ids, checkpoint = _checkpoint()
    tokenizer = CharacterTokenizer()
    seeds = [101, 202, 303, 404]
    ids = [f"branch-{index}" for index in range(4)]
    generation = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 8}
    forward = continue_from_checkpoint_dynamic_microbatch(
        model, tokenizer, checkpoint,
        branch_seeds=seeds, branch_ids=ids, generation=generation,
        maximum_microbatch_size=3,
    )
    reverse = continue_from_checkpoint_dynamic_microbatch(
        model, tokenizer, checkpoint,
        branch_seeds=list(reversed(seeds)), branch_ids=list(reversed(ids)), generation=generation,
        maximum_microbatch_size=3,
    )
    expected = {key: value.token_ids for key, value in zip(forward.branch_ids, forward.completions)}
    observed = {key: value.token_ids for key, value in zip(reverse.branch_ids, reverse.completions)}
    assert observed == expected


def test_matched_shape_smoke_uses_fixed_production_batch() -> None:
    model, _ids, checkpoint = _checkpoint()
    result = run_matched_shape_smoke(
        model,
        CharacterTokenizer(),
        checkpoint,
        microbatch_size=4,
        continuation_tokens=6,
        seed=83,
        branch_id="smoke-branch",
    )
    assert result["passed"]
    assert result["conditions"]["greedy"]["continuation_token_equal"]
    assert result["conditions"]["seeded"]["position_mask_cache_length_equal"]


def test_native_generation_retains_exact_cache_hidden_states_and_checkpoint_logits() -> None:
    model = ToyCausalModel().eval()
    tokenizer = CharacterTokenizer()
    prompt_ids = torch.tensor([[10, 11, 12, 13]])
    result = generate_prompt_completion_fixed_microbatch(
        model,
        tokenizer,
        prompt_ids,
        {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 5},
        19,
        branch_id="native-initial",
        microbatch_size=4,
        model_id="mock/toy",
        model_revision="mock-v1",
        tokenizer_id="mock/character",
        tokenizer_revision="mock-v1",
        selected_hidden_layers=[0, -1],
    )
    completion = result.completion
    assert cache_sequence_length(completion.past_key_values) == 4 + len(completion.token_ids)
    assert set(completion.hidden_states_by_layer) == {0, -1}
    assert all(value.shape[0] == len(completion.token_ids) for value in completion.hidden_states_by_layer.values())
    assert len(completion.next_token_logits_by_step) == len(completion.token_ids)
    assert set(result.prompt_hidden_states) == {0, -1}


def test_greedy_and_seeded_restoration_are_deterministic(tmp_path) -> None:
    model, _ids, checkpoint = _checkpoint()
    path = tmp_path / "checkpoint.pt"; save_cache_checkpoint(checkpoint, path)
    restored = load_cache_checkpoint(path)
    tokenizer = CharacterTokenizer()
    for generation in (
        {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 8},
        {"temperature": 0.7, "top_p": 0.95, "top_k": 32, "max_new_tokens": 8},
    ):
        left = decode_from_checkpoint(model, tokenizer, checkpoint, branch_seeds=[991], generation=generation, capture_diagnostics=True)
        right = decode_from_checkpoint(model, tokenizer, restored, branch_seeds=[991], generation=generation, capture_diagnostics=True)
        assert left.first_suffix_token_ids == right.first_suffix_token_ids
        assert left.completions[0].token_ids == right.completions[0].token_ids
        assert torch.equal(left.logits_after_first_suffix_token, right.logits_after_first_suffix_token)
        assert left.position_audit == right.position_audit


def test_serialized_checkpoint_continues_identically_in_fresh_process(tmp_path) -> None:
    model, _ids, checkpoint = _checkpoint()
    checkpoint_path = tmp_path / "checkpoint.pt"; save_cache_checkpoint(checkpoint, checkpoint_path)
    output_path = tmp_path / "child_tokens.json"
    generation = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 8}
    parent = decode_from_checkpoint(model, CharacterTokenizer(), checkpoint, branch_seeds=[73], generation=generation).completions[0].token_ids
    # Fork avoids an unrelated macOS OpenMP shared-memory initialization failure
    # while still validating the real serialized checkpoint in a separate process.
    child = multiprocessing.get_context("fork").Process(
        target=_continue_serialized_checkpoint_in_child,
        args=(str(checkpoint_path), str(output_path)),
    )
    child.start()
    child.join(timeout=15)
    assert child.exitcode == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == parent


def test_sampling_supports_top_k_top_p_and_banned_tokens() -> None:
    logits = torch.tensor([[0.0, 2.0, 4.0, 3.0]])
    greedy, _ = select_tokens_from_saved_logits(logits, {"temperature": 0.0, "banned_token_ids": [2]}, [1])
    assert greedy.tolist() == [3]
    left, _ = select_tokens_from_saved_logits(logits.repeat(2, 1), {"temperature": 0.7, "top_p": 0.9, "top_k": 3}, [5, 9])
    right, _ = select_tokens_from_saved_logits(logits.repeat(2, 1), {"temperature": 0.7, "top_p": 0.9, "top_k": 3}, [5, 9])
    assert torch.equal(left, right)


def test_mock_a_to_h_cache_diagnostics_pass() -> None:
    model = ToyCausalModel(); model.eval()
    ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
    result = run_cache_diagnostics(
        model,
        CharacterTokenizer(),
        ids,
        4,
        metadata={"model_id": "mock/toy", "model_revision": "mock-v1", "tokenizer_id": "mock/character", "tokenizer_revision": "mock-v1"},
        generation={"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 8},
        seed=123,
    )
    assert result["primary_gate_passed"]
    assert result["test_a_live_vs_restored"]["passed"]
    assert result["test_b_captured_vs_cropped_final"]["passed"]
    assert result["test_c_no_replay_vs_prefix_recomputation"]["passed"]
    assert result["test_g_branch_batching"]["passed"]
    assert result["test_h_serialization_round_trip"]["passed"]
