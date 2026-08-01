"""Transactional four-slot work registry for the K=32 confirmation run."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from safeprefix.reproducibility import atomic_json, now_iso, stable_hash

MODEL_KEYS = (
    "family_a_small",
    "family_a_large",
    "family_b_small",
    "family_b_large",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS checkpoints (
  checkpoint_key TEXT PRIMARY KEY,
  model_key TEXT NOT NULL,
  trace_id TEXT NOT NULL,
  problem_id TEXT NOT NULL,
  domain TEXT NOT NULL,
  checkpoint_ordinal INTEGER NOT NULL,
  checkpoint_token_offset INTEGER NOT NULL,
  prefix_token_hash TEXT NOT NULL,
  continuation_policy_hash TEXT NOT NULL,
  estimated_suffix_tokens REAL NOT NULL,
  required_slots INTEGER NOT NULL CHECK(required_slots=32)
);
CREATE TABLE IF NOT EXISTS rollout_slots (
  checkpoint_key TEXT NOT NULL,
  rollout_slot INTEGER NOT NULL CHECK(rollout_slot BETWEEN 0 AND 31),
  rollout_seed INTEGER NOT NULL,
  source TEXT NOT NULL CHECK(source IN ('threshold_k16','reused_compatible','generated')),
  outcome_id TEXT,
  verifier_outcome INTEGER,
  artifact_path TEXT,
  artifact_hash TEXT,
  PRIMARY KEY(checkpoint_key, rollout_slot),
  UNIQUE(checkpoint_key, rollout_seed),
  FOREIGN KEY(checkpoint_key) REFERENCES checkpoints(checkpoint_key)
);
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  checkpoint_key TEXT NOT NULL,
  model_key TEXT NOT NULL,
  block_start INTEGER NOT NULL CHECK(block_start IN (16,20,24,28)),
  block_end INTEGER NOT NULL,
  priority INTEGER NOT NULL CHECK(priority=1),
  estimated_token_work REAL NOT NULL,
  prefix_length INTEGER NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','running','complete','failed')),
  worker_id TEXT,
  lease_expires_unix REAL,
  claim_unix REAL,
  completion_unix REAL,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  error_message TEXT,
  FOREIGN KEY(checkpoint_key) REFERENCES checkpoints(checkpoint_key)
);
CREATE TABLE IF NOT EXISTS worker_transitions (
  transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
  worker_id TEXT NOT NULL,
  from_model_key TEXT,
  to_model_key TEXT NOT NULL,
  transition_unix REAL NOT NULL,
  reason TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_claim_order
ON jobs(status, model_key, priority, estimated_token_work DESC, prefix_length);
"""


def connect_registry(path: str | Path) -> sqlite3.Connection:
    database = Path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=60.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(SCHEMA)
    return connection


@contextmanager
def immediate(connection: sqlite3.Connection) -> Iterator[None]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        connection.execute("ROLLBACK")
        raise
    else:
        connection.execute("COMMIT")


def _row_id(row: Mapping[str, Any]) -> str:
    return stable_hash(
        [
            "k32-slot-outcome-v1",
            row["model_key"],
            row["trace_id"],
            int(row["checkpoint_ordinal"]),
            int(row["checkpoint_token_offset"]),
            int(row["rollout_index"]),
            int(row["rollout_seed"]),
            str(row["artifact_hash"]),
        ]
    )


def initialize_registry(
    *,
    registry_path: str | Path,
    confirmation: pd.DataFrame,
    generation_manifest: pd.DataFrame,
    k16_outcomes: pd.DataFrame,
) -> dict[str, Any]:
    connection = connect_registry(registry_path)
    try:
        with immediate(connection):
            for row in confirmation.to_dict("records"):
                part = k16_outcomes.loc[k16_outcomes["checkpoint_key"].eq(row["checkpoint_key"])]
                estimate = float(part["generated_token_count"].median())
                connection.execute(
                    """INSERT OR IGNORE INTO checkpoints(
                         checkpoint_key, model_key, trace_id, problem_id, domain,
                         checkpoint_ordinal, checkpoint_token_offset, prefix_token_hash,
                         continuation_policy_hash, estimated_suffix_tokens, required_slots
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,32)""",
                    (
                        row["checkpoint_key"], row["model_key"], row["trace_id"],
                        row["problem_id"], row["domain"], int(row["checkpoint_ordinal"]),
                        int(row["checkpoint_token_offset"]), row["prefix_token_hash"],
                        row["continuation_policy_hash"], estimate,
                    ),
                )
                current = connection.execute(
                    "SELECT * FROM checkpoints WHERE checkpoint_key=?", (row["checkpoint_key"],)
                ).fetchone()
                if current is None or str(current["prefix_token_hash"]) != str(row["prefix_token_hash"]):
                    raise RuntimeError("registry checkpoint conflicts with frozen manifest")
                for outcome in part.to_dict("records"):
                    slot = int(outcome["rollout_index"])
                    if not 0 <= slot < 16:
                        raise RuntimeError("non-K16 row entered frozen slot initialization")
                    outcome_id = _row_id(outcome)
                    connection.execute(
                        """INSERT OR IGNORE INTO rollout_slots(
                             checkpoint_key, rollout_slot, rollout_seed, source, outcome_id,
                             verifier_outcome, artifact_path, artifact_hash
                           ) VALUES(?,?,?,'threshold_k16',?,?,?,?)""",
                        (
                            row["checkpoint_key"], slot, int(outcome["rollout_seed"]), outcome_id,
                            int(bool(outcome["verifier_outcome"])), "threshold_k16",
                            str(outcome["artifact_hash"]),
                        ),
                    )
                    existing = connection.execute(
                        "SELECT * FROM rollout_slots WHERE checkpoint_key=? AND rollout_slot=?",
                        (row["checkpoint_key"], slot),
                    ).fetchone()
                    if existing is None or str(existing["outcome_id"]) != outcome_id:
                        raise RuntimeError("frozen K16 registry slot conflicts")
                generation = generation_manifest.loc[
                    generation_manifest["checkpoint_key"].eq(row["checkpoint_key"])
                ]
                if sorted(generation["rollout_slot"].astype(int)) != list(range(16, 32)):
                    raise RuntimeError("generation manifest lacks exact slots 16..31")
                for block_start in (16, 20, 24, 28):
                    job_id = stable_hash(["k32-four-slot-job-v1", row["checkpoint_key"], block_start])
                    connection.execute(
                        """INSERT OR IGNORE INTO jobs(
                             job_id, checkpoint_key, model_key, block_start, block_end,
                             priority, estimated_token_work, prefix_length, status
                           ) VALUES(?,?,?,?,?,1,?,?,'pending')""",
                        (
                            job_id, row["checkpoint_key"], row["model_key"], block_start,
                            block_start + 3, 4.0 * estimate, int(row["prefix_token_count"]),
                        ),
                    )
        counts = {
            "checkpoints": connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0],
            "frozen_k16_slots": connection.execute(
                "SELECT COUNT(*) FROM rollout_slots WHERE rollout_slot<16"
            ).fetchone()[0],
            "jobs": connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],
            "pending_jobs": connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status='pending'"
            ).fetchone()[0],
        }
        if counts != {
            "checkpoints": len(confirmation),
            "frozen_k16_slots": len(confirmation) * 16,
            "jobs": len(confirmation) * 4,
            "pending_jobs": len(confirmation) * 4,
        }:
            raise RuntimeError(f"registry initialization counts differ: {counts}")
        return counts
    finally:
        connection.close()


def reset_expired_leases(connection: sqlite3.Connection, now: float | None = None) -> int:
    timestamp = float(time.time() if now is None else now)
    with immediate(connection):
        cursor = connection.execute(
            """UPDATE jobs SET status='pending', worker_id=NULL, lease_expires_unix=NULL,
                      claim_unix=NULL, error_message='lease_expired_requeued'
               WHERE status='running' AND lease_expires_unix<?""",
            (timestamp,),
        )
    return int(cursor.rowcount)


def remaining_workload(connection: sqlite3.Connection) -> dict[str, float]:
    rows = connection.execute(
        """SELECT model_key, COALESCE(SUM(estimated_token_work),0) AS work
             FROM jobs WHERE status='pending' GROUP BY model_key"""
    ).fetchall()
    return {str(row["model_key"]): float(row["work"]) for row in rows}


def claim_job(
    connection: sqlite3.Connection,
    *,
    worker_id: str,
    preferred_model_key: str,
    lease_seconds: float = 1800.0,
) -> dict[str, Any] | None:
    reset_expired_leases(connection)
    now = time.time()
    with immediate(connection):
        row = connection.execute(
            """SELECT * FROM jobs
               WHERE status='pending' AND model_key=?
               ORDER BY priority ASC, estimated_token_work DESC, prefix_length ASC, job_id ASC
               LIMIT 1""",
            (preferred_model_key,),
        ).fetchone()
        if row is None:
            workload = remaining_workload(connection)
            if not workload:
                return None
            selected_model = sorted(workload, key=lambda model: (-workload[model], model))[0]
            row = connection.execute(
                """SELECT * FROM jobs
                   WHERE status='pending' AND model_key=?
                   ORDER BY priority ASC, estimated_token_work DESC, prefix_length ASC, job_id ASC
                   LIMIT 1""",
                (selected_model,),
            ).fetchone()
        if row is None:
            return None
        cursor = connection.execute(
            """UPDATE jobs SET status='running', worker_id=?, claim_unix=?,
                      lease_expires_unix=?, attempt_count=attempt_count+1, error_message=NULL
               WHERE job_id=? AND status='pending'""",
            (worker_id, now, now + float(lease_seconds), row["job_id"]),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("atomic job claim lost")
        claimed = dict(row)
        claimed.update(
            worker_id=worker_id,
            claim_unix=now,
            lease_expires_unix=now + float(lease_seconds),
        )
        return claimed


def claim_checkpoint_jobs(
    connection: sqlite3.Connection,
    *,
    worker_id: str,
    preferred_model_key: str,
    lease_seconds: float = 1800.0,
) -> list[dict[str, Any]] | None:
    """Atomically lease every pending four-slot job for one checkpoint.

    A checkpoint is eligible only while none of its jobs are already running.  This
    keeps the teacher-forced prefix cache private to one worker and lets that worker
    decode all remaining slots from one prefill.
    """
    reset_expired_leases(connection)
    now = time.time()
    with immediate(connection):
        def select_checkpoint(model_key: str) -> sqlite3.Row | None:
            return connection.execute(
                """SELECT pending.checkpoint_key, pending.model_key,
                          MIN(pending.priority) AS priority,
                          SUM(pending.estimated_token_work) AS estimated_token_work,
                          MIN(pending.prefix_length) AS prefix_length
                     FROM jobs AS pending
                    WHERE pending.status='pending' AND pending.model_key=?
                      AND NOT EXISTS (
                          SELECT 1 FROM jobs AS active
                           WHERE active.checkpoint_key=pending.checkpoint_key
                             AND active.status='running'
                      )
                    GROUP BY pending.checkpoint_key, pending.model_key
                    ORDER BY priority ASC, estimated_token_work DESC,
                             prefix_length ASC, pending.checkpoint_key ASC
                    LIMIT 1""",
                (model_key,),
            ).fetchone()

        selected = select_checkpoint(preferred_model_key)
        if selected is None:
            workloads = remaining_workload(connection)
            for model_key in sorted(workloads, key=lambda model: (-workloads[model], model)):
                selected = select_checkpoint(model_key)
                if selected is not None:
                    break
        if selected is None:
            return None
        rows = connection.execute(
            """SELECT * FROM jobs
                 WHERE checkpoint_key=? AND status='pending'
                 ORDER BY block_start ASC""",
            (selected["checkpoint_key"],),
        ).fetchall()
        if not rows:
            return None
        job_ids = [str(row["job_id"]) for row in rows]
        placeholders = ",".join("?" for _ in job_ids)
        cursor = connection.execute(
            f"""UPDATE jobs SET status='running', worker_id=?, claim_unix=?,
                       lease_expires_unix=?, attempt_count=attempt_count+1,
                       error_message=NULL
                  WHERE status='pending' AND job_id IN ({placeholders})""",
            (worker_id, now, now + float(lease_seconds), *job_ids),
        )
        if cursor.rowcount != len(rows):
            raise RuntimeError("atomic checkpoint job claim lost")
        claimed = []
        for row in rows:
            job = dict(row)
            job.update(
                worker_id=worker_id,
                claim_unix=now,
                lease_expires_unix=now + float(lease_seconds),
            )
            claimed.append(job)
        return claimed


def register_transition(
    connection: sqlite3.Connection,
    *,
    worker_id: str,
    from_model_key: str | None,
    to_model_key: str,
    reason: str,
) -> None:
    with immediate(connection):
        connection.execute(
            """INSERT INTO worker_transitions(
                 worker_id, from_model_key, to_model_key, transition_unix, reason
               ) VALUES(?,?,?,?,?)""",
            (worker_id, from_model_key, to_model_key, time.time(), reason),
        )


def complete_job(
    connection: sqlite3.Connection,
    *,
    job_id: str,
    worker_id: str,
    rows: Sequence[Mapping[str, Any]],
    artifact_path: str,
) -> None:
    with immediate(connection):
        job = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if job is None or job["status"] != "running" or job["worker_id"] != worker_id:
            raise RuntimeError("only the leasing worker may complete a running job")
        expected = list(range(int(job["block_start"]), int(job["block_end"]) + 1))
        observed = sorted(int(row["rollout_index"]) for row in rows)
        if observed != expected:
            raise RuntimeError("completed block does not contain its exact four slots")
        for row in rows:
            slot = int(row["rollout_index"])
            outcome_id = _row_id(row)
            existing = connection.execute(
                "SELECT * FROM rollout_slots WHERE checkpoint_key=? AND rollout_slot=?",
                (job["checkpoint_key"], slot),
            ).fetchone()
            if existing is not None and str(existing["outcome_id"]) != outcome_id:
                raise RuntimeError("same slot received conflicting outcome rows")
            if existing is None:
                connection.execute(
                    """INSERT INTO rollout_slots(
                         checkpoint_key, rollout_slot, rollout_seed, source, outcome_id,
                         verifier_outcome, artifact_path, artifact_hash
                       ) VALUES(?,?,?,'generated',?,?,?,?)""",
                    (
                        job["checkpoint_key"], slot, int(row["rollout_seed"]), outcome_id,
                        int(bool(row["verifier_outcome"])), artifact_path,
                        str(row["artifact_hash"]),
                    ),
                )
        connection.execute(
            """UPDATE jobs SET status='complete', completion_unix=?, lease_expires_unix=NULL
               WHERE job_id=?""",
            (time.time(), job_id),
        )


def fail_job(
    connection: sqlite3.Connection, *, job_id: str, worker_id: str, error: str
) -> None:
    with immediate(connection):
        connection.execute(
            """UPDATE jobs SET status='pending', worker_id=NULL, lease_expires_unix=NULL,
                      claim_unix=NULL, error_message=?
               WHERE job_id=? AND status='running' AND worker_id=?""",
            (str(error)[:4000], job_id, worker_id),
        )


def prepare_registry_and_reuse_report(
    *,
    output_root: str | Path,
    threshold_root: str | Path,
    local_search_root: str | Path,
    marketing_volume_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = Path(output_root)
    confirmation = pd.read_parquet(output / "k32_confirmation_checkpoints.parquet")
    generation = pd.read_parquet(output / "generation_input/k32_generation_manifest.parquet")
    from .analysis import load_k16_outcomes

    k16 = load_k16_outcomes(threshold_root, confirmation)
    registry_path = output / "k32_rollout_registry.sqlite"
    counts = initialize_registry(
        registry_path=registry_path,
        confirmation=confirmation,
        generation_manifest=generation,
        k16_outcomes=k16,
    )
    by_model = []
    for model_key in MODEL_KEYS:
        part = confirmation.loc[confirmation["model_key"].eq(model_key)]
        source = k16.loc[k16["model_key"].eq(model_key)]
        median = source.groupby("checkpoint_key")["generated_token_count"].median()
        by_model.append(
            {
                "model_key": model_key,
                "checkpoints": len(part),
                "reused_k16_slots": len(part) * 16,
                "reused_additional_slots": 0,
                "projected_new_slots": len(part) * 16,
                "projected_generated_tokens": float((median * 16.0).sum()),
            }
        )
    local = Path(local_search_root)
    candidates = sorted(
        path.as_posix()
        for path in local.rglob("*")
        if path.is_file()
        and path.suffix in {".parquet", ".jsonl"}
        and any(token in path.name.lower() for token in ("outcome", "rollout", "suffix", "branch"))
        and "native" not in path.as_posix().lower()
        and "mock" not in path.as_posix().lower()
    )
    report = {
        "schema_version": 2,
        "status": "COMPLETE",
        "created_at": now_iso(),
        "selection_before_success_access": True,
        "threshold_k16_source_sha256": sha256_file(
            Path(threshold_root) / "raw_outcomes/merged_k16_checkpoint_suffixes.parquet"
        ),
        "confirmation_manifest_sha256": sha256_file(output / "k32_confirmation_manifest.json"),
        "searched_local_candidate_paths": candidates,
        "searched_marketing_workspace_volumes": list(marketing_volume_inventory),
        "marketing_workspace_has_prior_k32_or_threshold_volume": False,
        "reused_slots_0_15": len(confirmation) * 16,
        "reused_slots_16_31": 0,
        "missing_slots_16_31": len(confirmation) * 16,
        "new_generation_by_model": by_model,
        "projected_new_rollouts": len(confirmation) * 16,
        "registry": {**counts, "path": str(registry_path)},
        "native_artifacts_loaded": False,
        "production_k16_regenerated": False,
    }
    atomic_json(output / "k32_reuse_report.json", report)
    return report
