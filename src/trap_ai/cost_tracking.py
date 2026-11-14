"""Cost tracking utilities for OpenAI evaluator/reviewer calls."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trap_ai.config import resolve_config_path


OPENAI_PRICE_SOURCE = "https://openai.com/api/pricing/"
OPENAI_PRICE_ASSUMPTIONS = {
    "gpt-5.5": {
        "standard_input_usd_per_million": 5.0,
        "standard_cached_input_usd_per_million": 0.5,
        "standard_output_usd_per_million": 30.0,
        "batch_discount": 0.5,
    }
}


@dataclass(frozen=True)
class OpenAICostReport:
    root: Path
    output_dir: Path
    ledger_path: Path
    summary_path: Path
    response_count: int
    prompt_tokens: int
    cached_prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_usd: float


def summarize_openai_costs(
    root: Path,
    output_dir: Path,
    scan_roots: list[Path] | None = None,
    overwrite: bool = False,
) -> OpenAICostReport:
    root = root.resolve()
    selected_scan_roots = scan_roots or [Path("runs"), Path("results")]
    resolved_output_dir = resolve_config_path(root, output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ledger_path = resolved_output_dir / f"openai_usage_{timestamp}.csv"
    summary_path = resolved_output_dir / f"openai_usage_{timestamp}_summary.json"
    if not overwrite:
        for path in [ledger_path, summary_path]:
            if path.exists():
                raise FileExistsError(f"Cost artifact already exists: {path}")

    records = collect_openai_usage_records(root, selected_scan_roots)
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "response_id",
            "model",
            "pricing_family",
            "pricing_tier",
            "prompt_tokens",
            "cached_prompt_tokens",
            "billable_prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_tokens",
            "estimated_usd",
            "source_paths",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    summary = build_openai_cost_summary(records)
    summary.update(
        {
            "schema_version": "0.1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "price_source": OPENAI_PRICE_SOURCE,
            "price_assumptions": OPENAI_PRICE_ASSUMPTIONS,
            "scan_roots": [str(resolve_config_path(root, path).relative_to(root)).replace("\\", "/") for path in selected_scan_roots],
            "ledger_path": str(ledger_path.relative_to(root)).replace("\\", "/"),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    latest_ledger = resolved_output_dir / "openai_usage_latest.csv"
    latest_summary = resolved_output_dir / "openai_usage_latest_summary.json"
    latest_ledger.write_text(ledger_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_summary.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")

    return OpenAICostReport(
        root=root,
        output_dir=resolved_output_dir,
        ledger_path=ledger_path,
        summary_path=summary_path,
        response_count=int(summary["response_count"]),
        prompt_tokens=int(summary["prompt_tokens"]),
        cached_prompt_tokens=int(summary["cached_prompt_tokens"]),
        completion_tokens=int(summary["completion_tokens"]),
        total_tokens=int(summary["total_tokens"]),
        estimated_usd=float(summary["estimated_usd"]),
    )


def collect_openai_usage_records(root: Path, scan_roots: list[Path]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    source_paths_by_id: dict[str, set[str]] = {}
    for scan_root in scan_roots:
        resolved = resolve_config_path(root, scan_root)
        if not resolved.exists():
            continue
        for path in sorted(resolved.rglob("*")):
            if path.suffix.lower() not in {".json", ".jsonl"} or not path.is_file():
                continue
            for completion in iter_chat_completion_payloads(path):
                response_id = str(completion.get("id", ""))
                model = str(completion.get("model", ""))
                usage = completion.get("usage", {})
                if not response_id or not model.startswith("gpt-") or not isinstance(usage, dict):
                    continue
                relative_source = str(path.relative_to(root)).replace("\\", "/")
                if response_id not in by_id:
                    by_id[response_id] = completion
                    source_paths_by_id[response_id] = set()
                source_paths_by_id[response_id].add(relative_source)
    records = []
    for response_id, completion in sorted(by_id.items()):
        source_paths = sorted(source_paths_by_id[response_id])
        records.append(openai_usage_record(response_id, completion, source_paths))
    return records


def iter_chat_completion_payloads(path: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    try:
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    decoded = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payloads.extend(find_chat_completion_dicts(decoded))
        else:
            decoded = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            payloads.extend(find_chat_completion_dicts(decoded))
    except (OSError, json.JSONDecodeError):
        return []
    return payloads


def find_chat_completion_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("object") == "chat.completion" and isinstance(value.get("usage"), dict):
            found.append(value)
        for child in value.values():
            found.extend(find_chat_completion_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_chat_completion_dicts(child))
    return found


def openai_usage_record(response_id: str, completion: dict[str, Any], source_paths: list[str]) -> dict[str, Any]:
    usage = completion.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    prompt_details = usage.get("prompt_tokens_details", {})
    completion_details = usage.get("completion_tokens_details", {})
    cached_prompt_tokens = int(prompt_details.get("cached_tokens", 0) or 0) if isinstance(prompt_details, dict) else 0
    reasoning_tokens = (
        int(completion_details.get("reasoning_tokens", 0) or 0) if isinstance(completion_details, dict) else 0
    )
    billable_prompt_tokens = max(prompt_tokens - cached_prompt_tokens, 0)
    model = str(completion.get("model", ""))
    pricing_family = pricing_family_for_model(model)
    pricing_tier = "batch" if any("openai_batches" in source for source in source_paths) else "standard"
    estimated_usd = estimate_openai_cost(
        pricing_family=pricing_family,
        pricing_tier=pricing_tier,
        billable_prompt_tokens=billable_prompt_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return {
        "response_id": response_id,
        "model": model,
        "pricing_family": pricing_family,
        "pricing_tier": pricing_tier,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "billable_prompt_tokens": billable_prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "estimated_usd": f"{estimated_usd:.6f}" if estimated_usd is not None else "",
        "source_paths": ";".join(source_paths),
    }


def pricing_family_for_model(model: str) -> str:
    normalized = model.lower()
    if normalized.startswith("gpt-5.5"):
        return "gpt-5.5"
    return "unknown"


def estimate_openai_cost(
    pricing_family: str,
    pricing_tier: str,
    billable_prompt_tokens: int,
    cached_prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    rates = OPENAI_PRICE_ASSUMPTIONS.get(pricing_family)
    if rates is None:
        return None
    discount = float(rates["batch_discount"]) if pricing_tier == "batch" else 1.0
    input_cost = billable_prompt_tokens * float(rates["standard_input_usd_per_million"]) / 1_000_000
    cached_cost = cached_prompt_tokens * float(rates["standard_cached_input_usd_per_million"]) / 1_000_000
    output_cost = completion_tokens * float(rates["standard_output_usd_per_million"]) / 1_000_000
    return (input_cost + cached_cost + output_cost) * discount


def build_openai_cost_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens = sum(int(record["prompt_tokens"]) for record in records)
    cached_prompt_tokens = sum(int(record["cached_prompt_tokens"]) for record in records)
    completion_tokens = sum(int(record["completion_tokens"]) for record in records)
    total_tokens = sum(int(record["total_tokens"]) for record in records)
    estimated_usd = sum(float(record["estimated_usd"] or 0.0) for record in records)
    by_model = Counter(str(record["model"]) for record in records)
    by_tier = Counter(str(record["pricing_tier"]) for record in records)
    return {
        "response_count": len(records),
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_usd": round(estimated_usd, 6),
        "by_model": dict(by_model),
        "by_pricing_tier": dict(by_tier),
    }


def format_openai_cost_report(report: OpenAICostReport) -> str:
    return "\n".join(
        [
            "OpenAI usage summary.",
            f"Responses: {report.response_count}",
            f"Prompt tokens: {report.prompt_tokens}",
            f"Cached prompt tokens: {report.cached_prompt_tokens}",
            f"Completion tokens: {report.completion_tokens}",
            f"Total tokens: {report.total_tokens}",
            f"Estimated USD: {report.estimated_usd:.6f}",
            f"Ledger: {str(report.ledger_path.relative_to(report.root)).replace(chr(92), '/')}",
            f"Summary: {str(report.summary_path.relative_to(report.root)).replace(chr(92), '/')}",
        ]
    )
