"""Persistence helpers for evaluator-LLM claim extraction."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trap_ai.schemas import CLAIM_TABLE_COLUMNS


LEGACY_CLAIM_TABLE_COLUMNS = [
    column for column in CLAIM_TABLE_COLUMNS if column != "finding_result"
]


@dataclass(frozen=True)
class ClaimExtractionReport:
    run_dir: Path
    claims_path: Path
    manifest_path: Path
    claim_count: int
    valid_header: bool


def write_claim_table_from_extractor_response(
    run_dir: Path,
    extractor_text: str,
    output_path: Path | None = None,
    response_path: Path | None = None,
) -> ClaimExtractionReport:
    """Write evaluator-extracted claims from an LLM response.

    The LLM performs extraction. This helper only parses the response, enforces
    the expected CSV header, and records provenance.
    """

    run_dir = run_dir.resolve()
    claims_path = output_path or run_dir / "claim_tables" / "claims.csv"
    manifest_path = run_dir / "metadata" / "claim_extraction_manifest.json"
    csv_text, parse_status = _extract_claim_csv(extractor_text)
    valid_header = _has_expected_header(csv_text)
    if not valid_header:
        csv_text = ",".join(CLAIM_TABLE_COLUMNS)

    rows = _read_csv_rows(csv_text)
    claims_path.parent.mkdir(parents=True, exist_ok=True)
    claims_path.write_text(_rows_to_csv(rows), encoding="utf-8")

    manifest = {
        "schema_version": "0.1",
        "method": "llm_evaluator_claim_extraction_v1",
        "claims_path": _relative_to_run(claims_path, run_dir),
        "response_path": _relative_to_run(response_path, run_dir) if response_path else "",
        "claim_count": len(rows),
        "valid_header": valid_header,
        "parse_status": parse_status,
        "notes": [
            "Claims were extracted by a separate evaluator LLM after the author run.",
            "The evaluator prompt uses the mini-paper as the source document and the claim schema for structure.",
            "This helper writes the evaluator LLM output into the standard claim table.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ClaimExtractionReport(
        run_dir=run_dir,
        claims_path=claims_path,
        manifest_path=manifest_path,
        claim_count=len(rows),
        valid_header=valid_header,
    )


def _extract_claim_csv(text: str) -> tuple[str, str]:
    if not text.strip():
        return ",".join(CLAIM_TABLE_COLUMNS), "empty_response"
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, dict):
        csv_text = str(decoded.get("claim_table_csv", "")).strip()
        if csv_text:
            return csv_text, "json_claim_table_csv"
        claims = decoded.get("claims")
        if isinstance(claims, list):
            return _claims_list_to_csv(claims), "json_claims_list"
    return _extract_csv_like(text), "text_csv_fallback"


def _claims_list_to_csv(claims: list[Any]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CLAIM_TABLE_COLUMNS)
    writer.writeheader()
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        writer.writerow({column: str(claim.get(column, "")).strip() for column in CLAIM_TABLE_COLUMNS})
    return output.getvalue().strip()


def _extract_csv_like(text: str) -> str:
    header = ",".join(CLAIM_TABLE_COLUMNS)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith(header):
            return "\n".join(lines[index:]).strip()
    return header


def _has_expected_header(csv_text: str) -> bool:
    first_line = csv_text.splitlines()[0].strip() if csv_text.splitlines() else ""
    return first_line in {
        ",".join(CLAIM_TABLE_COLUMNS),
        ",".join(LEGACY_CLAIM_TABLE_COLUMNS),
    }


def _read_csv_rows(csv_text: str) -> list[dict[str, str]]:
    with io.StringIO(csv_text) as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if header == CLAIM_TABLE_COLUMNS:
            columns = CLAIM_TABLE_COLUMNS
        elif header == LEGACY_CLAIM_TABLE_COLUMNS:
            columns = LEGACY_CLAIM_TABLE_COLUMNS
        else:
            return []
        rows = []
        for values in reader:
            row = normalize_claim_row(dict(zip(columns, _coerce_claim_row_values(values, columns))))
            if any((value or "").strip() for value in row.values()):
                rows.append(row)
        return rows


def normalize_claim_row(row: dict[str, Any]) -> dict[str, str]:
    normalized = {column: str(row.get(column, "") or "").strip() for column in CLAIM_TABLE_COLUMNS}
    normalized["finding_result"] = normalize_finding_result(normalized.get("finding_result", ""))
    return normalized


def normalize_finding_result(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "unclear",
        "not_significant": "null",
        "nonsignificant": "null",
        "non_significant": "null",
        "no_association": "null",
        "null_finding": "null",
        "positive_association": "positive",
        "increased": "positive",
        "higher": "positive",
        "negative": "inverse",
        "decreased": "inverse",
        "lower": "inverse",
        "non_linear": "nonlinear",
        "nonlinearity": "nonlinear",
        "effect_modification": "interaction",
    }
    text = aliases.get(text, text)
    if text not in {"positive", "inverse", "null", "nonlinear", "interaction", "unclear"}:
        return "unclear"
    return text


def _coerce_claim_row_values(values: list[str], columns: list[str]) -> list[str]:
    expected = len(columns)
    if len(values) == expected:
        return values
    if len(values) > expected and len(values) >= 9:
        fixed_prefix_count = columns.index("model_type")
        fixed_prefix = values[:fixed_prefix_count]
        fixed_tail = values[-5:]
        model_type = ",".join(values[fixed_prefix_count:-5])
        return [*fixed_prefix, model_type, *fixed_tail]
    return [*values[:expected], *([""] * max(0, expected - len(values)))]


def _rows_to_csv(rows: list[dict[str, str]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CLAIM_TABLE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(normalize_claim_row(row))
    return output.getvalue()


def _relative_to_run(path: Path | None, run_dir: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(run_dir)).replace("\\", "/")
    except ValueError:
        return str(path)
