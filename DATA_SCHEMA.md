# SafePrefix Data and Artifact Schema

## Normalized trace

| Field | Type | Meaning |
|---|---|---|
| `problem_id` | string | Source-stable problem identifier |
| `source_dataset` | string | `Qwen/ProcessBench` or `facebook/crv` |
| `source_subset` | string | Source split/file family |
| `source_generator` | nullable string | Model that produced the recorded trace |
| `problem_text` | string | Unmodified task text |
| `reasoning_steps` | list[string] | Source-provided visible steps |
| `final_answer_text` | string | Source-provided predicted answer text |
| `reference_answer` | any | Exact terminal-verifier target when available |
| `first_error_index` | nullable integer | Source first-error annotation |
| `index_base` | `zero`, `one`, or null | Explicit source indexing convention |
| `final_answer_correct` | nullable boolean | Source terminal correctness label |
| `metadata` | mapping | Non-model-input provenance |

Audit rows add `problem_group_hash` and `split`. All traces sharing normalized
problem text receive the same split. They also add an immutable
`source_trace_id`, derived from dataset provenance and the exact recorded
reasoning, so multiple traces for one problem never collide in feature or
rollout shards.

Downstream training and evaluation artifacts use `problem_group_hash` as their
analysis-level `problem_id` and retain the source value as
`source_problem_id`. Thus bootstrap samples, train/dev/test partitions, and
native comparisons all use the normalized underlying problem as the unit.

## Span and checkpoint

A reasoning span contains exact character and response-token half-open ranges:
`char_start`, `char_end`, `token_start`, and `token_end`. The answer region is
excluded. A checkpoint stores the absolute prompt-plus-response token length
immediately after a span. Checkpoint zero is the prompt state.

The transient V2 `CacheCheckpoint` contains the complete cache through that
offset, native-dtype saved next-token logits, exact prefix IDs and mask, the
next position ID and cache position, model/tokenizer IDs and revisions, cache
dtype, tokenizer state (special-token IDs, padding/truncation sides, and chat
template), generation metadata, and explicit per-branch seed semantics. The next
position equals the prefix length: suffix decoding selects a new token from
the saved logits and never reprocesses the checkpoint-final token. A serialized
checkpoint is permitted only for cache-validity diagnostics; scientific
rollout stages do not persist full caches.

## Rollout

Each JSONL row contains the required `model_id`, `problem_id`, `trace_id`,
checkpoint index and token offset, rollout seed, generated token count and text,
parsed answer, terminal verifier result, and latency. Ordering by
`(trace_id, checkpoint_index, rollout_index)` defines nested k.

## Feature tensor

One `.pt` file per model/trace stores checkpoint features, exact absolute
checkpoint offsets, selected layer IDs, and the resolved model revision. Full
KV caches are intentionally not persisted by training or evaluation stages.
Features concatenate selected-layer
checkpoint vectors, previous-checkpoint differences, the final failed-trace
summary, relative/absolute token positions, remaining-token estimate, and local
NLL when enabled.

Stage A first writes a metadata-only per-model index. It contains the same
prompt, completion, span, checkpoint-offset, tokenizer-revision, and
configuration-hash fields, but `feature_path` and likelihood fields are null.
The timed Stage A job performs its own teacher-forced prefill. The later full
preparation stage writes features under the same schema into a distinct shard;
the metadata-only index is never admitted to boundary training.

## Scalable shards

Teacher-forced indices, rollout rows, native traces, repairs, and dense-audit
rows are first written below `per_model/<configuration_hash>/<model_key>/` (or
the equivalent stage-specific shard directory). Aggregation requires all
configured model shards and rejects a mismatched hash. Increasing a count limit
reuses only complete shards whose full resolved configuration hash matches; it
does not infer compatibility from filenames.

Native trace rows retain exact prompt and completion token IDs, token
log-probabilities, parser/verifier results, and model/tokenizer revisions. Full
KV caches remain transient on the worker. A dense native audit reconstructs the
transient state by rerunning the same fixed-batch initial generation and aborts
unless token IDs and decoded text equal the saved trace exactly.

## Reproducibility

Every stage writes a resolved YAML file and stage manifest containing the git
commit, Python/package versions, seed, command, timestamps, and terminal status.
JSON/JSONL/Parquet writes are atomic. Existing valid outputs are skipped in
resume mode and never replaced without `--overwrite`.
