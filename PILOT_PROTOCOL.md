# SafePrefix Pilot Protocol

SafePrefix tests whether a verifier-rejected visible reasoning trace can be
repaired by retaining a demonstrably safe prefix and regenerating only its
suffix. The primary protocol contains ordinary assistant text followed by a
final answer. It introduces no control tokens, XML, JSON, or step tags.

## Fixed scientific sequence

1. Audit ProcessBench and CRV; group and split by normalized underlying problem.
2. Validate cache restoration separately for every target model. A failed cache
   gate stops all rollout work for that model.
3. Compare P0, P1, and P2 on the same pilot-only problems. These problems never
   enter boundary training or native evaluation.
4. Teacher-force recorded failed traces, align source steps and native segments
   to exact target-token offsets, and abstain on ambiguous first-error mappings.
5. Roll out directly from prompt and clean pre-error checkpoints. No critique,
   correction instruction, verifier explanation, or branch-suppression text is
   added.
6. Fit separate safety and binomial repairability heads. Individual rollout
   outcomes—not a hard empirical argmax checkpoint—provide repairability loss.
7. Tune the latest-safe-near-best rule only on development problems.
8. On disjoint native problems, repair only initially rejected completions and
   compare one continuation per method and equal generated-token budgets.
9. Densely audit every checkpoint on 25 native failures per model.
10. Apply the preregistered automatic go/no-go checklist unchanged.

## Scalable production execution

The active scalable protocol is not a second framework. The pilot and full
experiment call the same numbered scripts and differ primarily in YAML sample
counts. Artifacts are keyed by exact configuration hash, model/revision,
problem, trace, checkpoint, and rollout seed. Per-model shards are aggregated
only after every configured shard matches that hash.

Production is fixed to BF16 SDPA and one model-specific fixed padded
microbatch. All SafePrefix and baseline continuations use that same batch shape,
decoding code, branch-ID-indexed random stream, answer parser, and terminal
verifier. Singleton/batch token identity is neither assumed nor pursued. BF16
eager remains unsupported. For a new deployment signature, the only blocking
pre-run check is the 10-checkpoint matched-shape live/restored test; the pinned
Qwen2.5-3B signature is exempt because it already passed.

This rule is recorded here and supersedes the earlier broad V3 rerun
requirement for the scalable pilot only.

Stage A uses 100 failed traces per model and times only the actual label path:
teacher-forced prefill, four suffixes per eligible checkpoint, and verification.
It performs no diagnostic duplicate continuation. A maximum per-model wall
time over 35 minutes stops Stage B; over 50 minutes fails the scalability gate.

Stage B uses nested outcomes: k=2 is the first two of the common k=4 rollouts;
the stratified k=6 subset adds outcomes 5 and 6 to those same checkpoint rows;
dense teacher-forced audits use 12. No hard best-checkpoint label is created.
Native evaluation uses the exact cache produced by normal generation, never a
teacher-forced reconstruction. Dense native auditing may deterministically
rerun that identical fixed-batch generation only after asserting exact token
and text equality with the saved trace.

## Cache protocol

At a checkpoint of absolute token length `L`, SafePrefix retains the cache
including all `L` prefix tokens and the native-dtype logits emitted after token
`L-1`. It selects the first suffix token directly from those saved logits,
feeds that genuinely new token at position `L`, and then continues explicit
autoregressive decoding. Token `L-1` is never replayed in the primary path.
Both legacy tuple caches and Transformers dynamic Qwen/Llama caches are
supported. Full caches remain transient and are discarded after rollouts.

The correctness gate deliberately separates two questions. The blocking
restoration test compares immediate continuation from the live cache with
continuation from an exact clone/serialization restoration of that same cache.
The nonblocking replay diagnostic compares a token processed during multi-token
prefill with the same token reprocessed during one-token decoding. A mismatch
in the latter is called a replay-path or prefill/decode numerical mismatch; it
is not evidence that cache copying itself is unstable. The rejected `L-1`
replay implementation remains available only through explicitly named
diagnostic helpers.
Prompt and completion regions are tokenized independently before teacher
forcing, matching the causal boundary present during native generation and
preventing a tokenizer from merging a token across that boundary.

## Rollout-count interpretation

The scalable collection budget is four training rollouts, eight development
rollouts, 12 dense teacher-forced rollouts, and 16 dense native rollouts per
checkpoint. Nested-k training is strict: k=6 is unavailable unless six ordered
outcomes were collected for the designated trace. It is never synthesized from
a four-rollout aggregate.

## Claims boundary

Mock artifacts validate only schemas, orchestration, and CPU logic. Missing
stages are `NOT_RUN`; they fail the automatic decision. A positive SafePrefix
claim requires cache correctness, data separation, boundary superiority,
three-of-four-model native transfer, and either the configured repair advantage
or recomputation reduction.
