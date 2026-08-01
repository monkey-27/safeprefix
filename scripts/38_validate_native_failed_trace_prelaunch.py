#!/usr/bin/env python3
"""Minimal CPU/tokenizer prelaunch validation; never loads model weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeprefix.config import load_config  # noqa: E402
from safeprefix.models.loader import load_tokenizer  # noqa: E402
from safeprefix.native_failed_trace_acquisition import (  # noqa: E402
    STRATA,
    build_attempt_packs,
    initial_seed,
    load_source_manifest,
)
from safeprefix.parsing.answer_parsers import parse_answer_region  # noqa: E402
from safeprefix.prompting.chat_format import render_chat_prompt  # noqa: E402
from safeprefix.prompting.templates import problem_instruction  # noqa: E402
from safeprefix.reproducibility import atomic_json, atomic_text, stable_hash  # noqa: E402
from safeprefix.rollout.verifier import ExactAnswerVerifier  # noqa: E402

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(ROOT / "configs/native_failed_trace_acquisition.yaml"),
    output_dir: Path = typer.Option(
        ROOT / "artifacts/native_failed_trace_acquisition/prelaunch"
    ),
) -> None:
    cfg = load_config(config).data
    source_path = ROOT / str(cfg["acquisition"]["source_manifest"])
    metadata_path = ROOT / str(cfg["acquisition"]["source_manifest_metadata"])
    rows = load_source_manifest(source_path, metadata_path)
    samples = [next(row for row in rows if row["stratum"] == stratum) for stratum in STRATA]
    verifier = ExactAnswerVerifier(
        float(cfg["verifier"]["absolute_tolerance"]),
        float(cfg["verifier"]["relative_tolerance"]),
    )
    model_checks = {}
    for model_key in cfg["selected_models"]:
        try:
            loaded = load_tokenizer(cfg["models"][model_key])
        except OSError as exc:
            model_checks[model_key] = {
                "status": "PENDING_APPROVAL_GATED_MODAL_SMOKE",
                "tokenizer_id": cfg["models"][model_key]["tokenizer_id"],
                "tokenizer_revision": cfg["models"][model_key]["tokenizer_revision"],
                "local_reason": f"{type(exc).__name__}: gated tokenizer is not authenticated locally",
            }
            continue
        rendered = []
        for sample in samples:
            prompt = render_chat_prompt(
                loaded.tokenizer,
                problem_instruction(sample["problem_text"], cfg["prompting"]["condition"]),
                override=cfg["models"][model_key].get("chat_template_override"),
                template_kwargs=cfg["models"][model_key].get("chat_template_kwargs"),
            )
            token_ids = loaded.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if not prompt.strip() or not token_ids:
                raise RuntimeError(f"{model_key}: empty rendered prompt")
            rendered.append({"stratum": sample["stratum"], "tokens": len(token_ids)})
        model_checks[model_key] = {
            "status": "PASS_LOCAL_TOKENIZER_RENDERING",
            "tokenizer_id": loaded.tokenizer_id,
            "tokenizer_revision": loaded.tokenizer_revision,
            "representative_prompts": rendered,
        }
    parser_checks = []
    for sample in samples:
        response = f"Reasoning.\n\n\\boxed{{{sample['gold_answer']}}}"
        parsed = parse_answer_region(response)
        parser_checks.append({
            "stratum": sample["stratum"],
            "parser_success": parsed.success,
            "verifier_accepts_reference": bool(
                parsed.success and verifier(parsed.parsed_answer, sample["gold_answer"], {})
            ),
        })
    if not all(row["parser_success"] and row["verifier_accepts_reference"] for row in parser_checks):
        raise RuntimeError("representative parser/verifier round trip failed")
    config_hash = stable_hash(cfg)
    pack_counts = {}
    initial_seed_checks = {}
    for model_key in cfg["selected_models"]:
        pack_counts[model_key] = {
            stratum: len(build_attempt_packs(
                rows, model_key=model_key, stratum=stratum,
                pack_size=int(cfg["acquisition"]["attempt_pack_size"]),
                configuration_hash=config_hash,
            ))
            for stratum in STRATA
        }
        seeds = [initial_seed(cfg, model_key, str(row["source_id"])) for row in rows]
        initial_seed_checks[model_key] = {
            "seed_count": len(seeds),
            "unique_seed_count": len(set(seeds)),
            "collision_free": len(seeds) == len(set(seeds)),
        }
        if len(seeds) != len(set(seeds)):
            raise RuntimeError(f"{model_key}: initial-generation seed collision")
    forbidden = {
        key: bool(cfg["acquisition"][key])
        for key in (
            "semantic_segmentation_enabled", "checkpoint_extraction_enabled",
            "hidden_state_extraction_enabled", "kv_cache_persistence_enabled",
            "boundary_model_enabled",
        )
    }
    if any(forbidden.values()):
        raise RuntimeError("prelaunch config enables prohibited native work")
    pending = [key for key, value in model_checks.items() if value["status"].startswith("PENDING")]
    result = {
        "status": (
            "PASS_LOCAL_CPU_WITH_GATED_TOKENIZERS_PENDING_APPROVAL_GATED_MODAL_SMOKE"
            if pending else "PASS_CPU_AND_TOKENIZER_PREFLIGHT"
        ),
        "model_weights_loaded": False,
        "modal_submission_performed": False,
        "configuration_hash": config_hash,
        "source_manifest_rows": len(rows),
        "source_counts": {
            stratum: sum(row["stratum"] == stratum for row in rows) for stratum in STRATA
        },
        "models": model_checks,
        "models_pending_real_modal_smoke": pending,
        "parser_verifier_checks": parser_checks,
        "attempt_pack_counts": pack_counts,
        "initial_seed_checks": initial_seed_checks,
        "initial_and_regeneration_policy_same_object": True,
        "generation_policy": cfg["generation"]["shared"],
        "forbidden_stages": forbidden,
    }
    atomic_json(output_dir / "INFRASTRUCTURE_PREFLIGHT.json", result)
    lines = [
        "# Native failed-trace acquisition prelaunch", "",
        "Status: **LOCAL CPU VALIDATION COMPLETE; GATED LLAMA TOKENIZERS REMAIN FOR THE APPROVAL-GATED MODAL SMOKE**", "",
        f"- Eligible source rows: `{len(rows)}`",
        f"- Counts: `{json.dumps(result['source_counts'], sort_keys=True)}`",
        "- Initial generation and all four regenerations use the single frozen generation policy.",
        "- Model weights were not loaded and no scientific output was generated.",
        f"- Pending real-model/tokenizer Modal smokes: `{json.dumps(pending)}`.",
        "- Semantic segmentation, checkpointing, hidden states, KV persistence, boundary prediction, and suffix repair are disabled.",
        "", "The real-model infrastructure smoke is approval-gated and runs automatically before each workspace's production workload.",
    ]
    atomic_text(output_dir / "PRELAUNCH_REPORT.md", "\n".join(lines) + "\n")
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
