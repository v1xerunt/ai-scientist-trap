"""Run reusable blinded listwise selection-review jobs through OpenRouter.

The runner is intentionally independent of a particular experiment, paper-ID
prefix, or candidate count.  It can consume one or more conventional reviewer
workspaces, or one explicit prompt/input pair.  Dry-run staging is the default;
``--send`` is required for paid network requests.

Examples
--------
Stage two workspaces without making requests::

    python scripts/run_openrouter_selection_reviewer.py \
      --workspace runs/example/reviewer_workspaces/SET_A \
      --workspace runs/example/reviewer_workspaces/SET_B \
      --model deepseek/deepseek-v4-pro \
      --output-root runs/example/openrouter_reviews/deepseek-v4-pro

Run one fully custom job::

    python scripts/run_openrouter_selection_reviewer.py \
      --prompt-file prompts/editor.md \
      --input-file inputs/candidates.json \
      --job-label my_pool --send
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
# Running a file from ``scripts/`` places that directory first on sys.path,
# where ``scripts/trap_ai.py`` would otherwise shadow the real package.
while str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

from trap_ai.agent_harness import (  # noqa: E402
    extract_assistant_text,
    load_env_file,
    make_llm_client,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reviewers" / "openrouter_reviewer.toml"
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_SYSTEM_PROMPT = (
    "You are a blinded handling editor. Follow the supplied task instructions "
    "exactly and use only the supplied candidate submissions."
)
PROMPT_RELATIVE = Path("SELECTION_PROMPT.md")
INPUT_RELATIVE = Path("inputs") / "selection_candidates.json"


@dataclass(frozen=True)
class ReviewJob:
    label: str
    prompt_path: Path
    input_path: Path
    source_workspace: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_argument_group("job sources")
    source.add_argument(
        "--workspace",
        type=Path,
        action="append",
        default=[],
        help=(
            "Reviewer workspace containing SELECTION_PROMPT.md and "
            "inputs/selection_candidates.json. Repeat for multiple jobs."
        ),
    )
    source.add_argument("--prompt-file", type=Path, help="Prompt for one custom job.")
    source.add_argument("--input-file", type=Path, help="Candidate JSON for one custom job.")
    source.add_argument("--job-label", help="Label for a custom job; defaults to the input stem.")

    request = parser.add_argument_group("request")
    request.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    request.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    request.add_argument("--model", default=DEFAULT_MODEL)
    request.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    request.add_argument("--temperature", type=float, default=0.2)
    request.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default="high",
    )
    request.add_argument("--max-tokens", type=int, default=65536)
    request.add_argument(
        "--timeout-seconds",
        type=float,
        default=900,
        help="HTTP timeout for each OpenRouter request (default: 900 seconds).",
    )
    request.add_argument(
        "--smoke-timeout-seconds",
        type=float,
        default=60,
        help="HTTP timeout for the preflight smoke request (default: 60 seconds).",
    )
    request.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip the paid-request preflight. Not recommended.",
    )
    request.add_argument("--seed", type=int)
    request.add_argument(
        "--no-structured-output",
        action="store_true",
        help="Ask for JSON text without sending OpenRouter response_format.",
    )
    request.add_argument(
        "--allow-provider-fallback",
        action="store_true",
        help="Do not require the routed provider to support every request parameter.",
    )
    request.add_argument(
        "--provider-order",
        action="append",
        default=[],
        help="OpenRouter provider slug to try, in order. Repeat to specify a list.",
    )
    request.add_argument(
        "--disable-provider-fallbacks",
        action="store_true",
        help="Use only providers named by --provider-order.",
    )
    request.add_argument(
        "--require-reasoning-tokens",
        action="store_true",
        help="Reject a SET response unless OpenRouter reports positive reasoning tokens.",
    )

    output = parser.add_argument_group("execution and output")
    output.add_argument(
        "--output-root",
        type=Path,
        help="Root containing one subdirectory per job. Required for multiple jobs.",
    )
    output.add_argument("--send", action="store_true", help="Make paid requests.")
    output.add_argument("--overwrite", action="store_true")
    output.add_argument(
        "--retry-incomplete",
        action="store_true",
        help=(
            "Archive a prior raw response that has no validator-clean output and "
            "retry only that incomplete job. Completed jobs are still reused."
        ),
    )
    output.add_argument(
        "--auto-format-repair",
        action="store_true",
        help=(
            "If a substantive response mentions every paper but still fails local "
            "ranking validation, send only that response (not the papers) through "
            "a formatting-only repair call before treating the job as incomplete."
        ),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return cleaned or "review_job"


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_jobs(args: argparse.Namespace) -> list[ReviewJob]:
    if args.workspace and (args.prompt_file or args.input_file or args.job_label):
        raise ValueError(
            "Use either repeated --workspace values or one explicit "
            "--prompt-file/--input-file pair, not both."
        )
    jobs: list[ReviewJob] = []
    if args.workspace:
        for raw_workspace in args.workspace:
            workspace = resolve(raw_workspace)
            jobs.append(
                ReviewJob(
                    label=slug(workspace.name),
                    prompt_path=workspace / PROMPT_RELATIVE,
                    input_path=workspace / INPUT_RELATIVE,
                    source_workspace=workspace,
                )
            )
    else:
        if not args.prompt_file or not args.input_file:
            raise ValueError(
                "Provide at least one --workspace, or both --prompt-file and --input-file."
            )
        input_path = resolve(args.input_file)
        jobs.append(
            ReviewJob(
                label=slug(args.job_label or input_path.stem),
                prompt_path=resolve(args.prompt_file),
                input_path=input_path,
                source_workspace=None,
            )
        )
    labels = [job.label for job in jobs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Job labels must be unique: {labels}")
    for job in jobs:
        if not job.prompt_path.is_file():
            raise FileNotFoundError(job.prompt_path)
        if not job.input_path.is_file():
            raise FileNotFoundError(job.input_path)
    return jobs


def load_candidates(path: Path) -> tuple[list[dict[str, str]], str]:
    raw_text = path.read_text(encoding="utf-8")
    payload = json.loads(raw_text)
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        keys = [key for key in ("papers", "candidates", "submissions") if key in payload]
        if len(keys) != 1:
            raise ValueError(
                "Candidate object must contain exactly one of: papers, candidates, submissions."
            )
        candidates = payload[keys[0]]
    else:
        raise ValueError("Candidate JSON must be an array or a supported wrapper object.")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate list must be non-empty.")

    normalized: list[dict[str, str]] = []
    for index, item in enumerate(candidates, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Candidate {index} must be an object.")
        normalized_item: dict[str, str] = {}
        for field in ("paper_id", "title", "abstract"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Candidate {index} has invalid {field}.")
            normalized_item[field] = value
        # Full-manuscript editor experiments use the same validated transport.
        # Preserve this optional field verbatim when supplied; older
        # title/abstract workspaces remain byte-for-byte equivalent after
        # normalization because the key is absent there.
        if "full_text" in item:
            value = item.get("full_text")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Candidate {index} has invalid full_text.")
            normalized_item["full_text"] = value
        normalized.append(normalized_item)
    ids = [item["paper_id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate paper_id values must be unique.")
    return normalized, raw_text


def ranking_schema(count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["ranked_papers"],
        "properties": {
            "ranked_papers": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["rank", "paper_id", "title", "selection_reason"],
                    "properties": {
                        "rank": {"type": "integer", "minimum": 1, "maximum": count},
                        "paper_id": {"type": "string"},
                        "title": {"type": "string"},
                        "selection_reason": {"type": "string", "minLength": 1},
                    },
                },
            }
        },
    }


def build_messages(
    system_prompt: str,
    frozen_prompt: str,
    candidates: list[dict[str, str]],
) -> list[dict[str, str]]:
    candidate_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    adapter = (
        "The complete task instructions above are authoritative. For this API "
        "transport, the candidate file referenced in those instructions is embedded "
        "in full below. Return the requested JSON object directly as the assistant "
        "response; the caller will save it to the requested output filename. Do not "
        "omit, add, or rename papers."
    )
    system = "\n\n".join(
        [
            system_prompt,
            "# Task instructions — verbatim from the selected prompt file",
            frozen_prompt,
            "# API transport adapter",
            adapter,
        ]
    )
    user = "\n\n".join(
        [
            "# Candidate submissions — complete embedded input",
            candidate_json,
            "# Final execution instruction",
            (
                "The candidate block ends above. Now carry out the exact task and "
                "assigned rubric in the system instructions. Do not translate or "
                "merely summarize the submissions. Return only the requested JSON "
                "object, include every supplied paper exactly once, and do not add "
                "prose outside that object."
            ),
        ]
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_request(
    *,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    temperature: float,
    reasoning_effort: str,
    max_tokens: int,
    seed: int | None,
    structured_output: bool,
    require_parameters: bool,
    provider_order: list[str],
    allow_provider_fallbacks: bool,
) -> dict[str, Any]:
    provider: dict[str, Any] = {"require_parameters": require_parameters}
    if provider_order:
        provider["order"] = provider_order
        provider["allow_fallbacks"] = allow_provider_fallbacks
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "reasoning": {"effort": reasoning_effort, "exclude": False},
        "max_tokens": max_tokens,
        "provider": provider,
    }
    if seed is not None:
        request["seed"] = seed
    if structured_output:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "blinded_selection_ranking",
                "strict": True,
                "schema": schema,
            },
        }
    return request


def smoke_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["status"],
        "properties": {"status": {"type": "string", "enum": ["ok"]}},
    }


def build_smoke_request(
    model: str,
    require_parameters: bool,
    provider_order: list[str],
    allow_provider_fallbacks: bool,
    reasoning_effort: str,
    structured_output: bool,
) -> dict[str, Any]:
    provider: dict[str, Any] = {"require_parameters": require_parameters}
    if provider_order:
        provider["order"] = provider_order
        provider["allow_fallbacks"] = allow_provider_fallbacks
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an API connectivity checker. Follow the output schema exactly.",
            },
            {
                "role": "user",
                "content": "Return the JSON object {\"status\":\"ok\"} and nothing else.",
            },
        ],
        "temperature": 0,
        "reasoning": {"effort": reasoning_effort, "exclude": False},
        "max_tokens": 512,
        "provider": provider,
    }
    if structured_output:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "openrouter_preflight",
                "strict": True,
                "schema": smoke_schema(),
            },
        }
    return request


def parse_json_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        stripped = "\n".join(lines).strip()
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("Reviewer response must decode to an object.")
    return payload


def parse_json_text_with_copied_title_repair(
    text: str, candidates: list[dict[str, str]]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Repair only malformed copied-title lines after strict JSON parsing fails.

    A model can insert an unescaped quote into a title it was instructed to copy,
    making an otherwise complete ranking invalid JSON.  Titles are frozen input
    metadata, not reviewer judgments.  This fallback associates a title line
    with the immediately preceding valid paper_id and replaces only that line
    with the canonical JSON-escaped input title.  The repaired payload must still
    pass the complete ranking validator afterward.
    """
    try:
        return parse_json_text(text), []
    except json.JSONDecodeError as original_error:
        stripped = text.strip()
        fenced = stripped.startswith("```")
        if fenced:
            lines = stripped.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines.pop()
        else:
            lines = stripped.splitlines()

        expected_titles = {item["paper_id"]: item["title"] for item in candidates}
        current_id: str | None = None
        repairs: list[dict[str, str]] = []
        for index, line in enumerate(lines):
            id_match = re.match(r'^\s*"paper_id"\s*:\s*"([^"]+)"', line)
            if id_match:
                candidate_id = id_match.group(1)
                current_id = candidate_id if candidate_id in expected_titles else None
                continue
            title_match = re.match(r'^(\s*)"title"\s*:', line)
            if not title_match or current_id is None:
                continue
            canonical_title = expected_titles[current_id]
            comma = "," if line.rstrip().endswith(",") else ""
            canonical_line = (
                f'{title_match.group(1)}"title": '
                f'{json.dumps(canonical_title, ensure_ascii=False)}{comma}'
            )
            if line != canonical_line:
                repairs.append(
                    {
                        "field": "title",
                        "paper_id": current_id,
                        "received": line.strip(),
                        "canonical": canonical_title,
                        "basis": "lexical repair of copied metadata after JSON parse failure",
                    }
                )
                lines[index] = canonical_line

        if not repairs:
            raise original_error
        return parse_json_text("\n".join(lines)), repairs


def validate_ranking(payload: dict[str, Any], candidates: list[dict[str, str]]) -> None:
    rows = payload.get("ranked_papers")
    if not isinstance(rows, list):
        raise ValueError("ranked_papers must be an array.")
    expected_titles = {item["paper_id"]: item["title"] for item in candidates}
    count = len(candidates)
    if len(rows) != count:
        raise ValueError(f"Expected {count} ranked papers; received {len(rows)}.")
    ids = [row.get("paper_id") for row in rows if isinstance(row, dict)]
    ranks = [row.get("rank") for row in rows if isinstance(row, dict)]
    if len(ids) != count or set(ids) != set(expected_titles) or len(ids) != len(set(ids)):
        raise ValueError("Ranking must contain every input paper_id exactly once.")
    if not all(isinstance(rank, int) and not isinstance(rank, bool) for rank in ranks):
        raise ValueError(f"Ranking must contain an integer rank for every paper.")
    if sorted(ranks) != list(range(1, count + 1)):
        raise ValueError(f"Ranking must contain unique consecutive ranks 1..{count}.")
    for row in rows:
        paper_id = row["paper_id"]
        if row.get("title") != expected_titles[paper_id]:
            raise ValueError(f"Exact title mismatch for {paper_id}.")
        if not isinstance(row.get("selection_reason"), str) or not row["selection_reason"].strip():
            raise ValueError(f"Missing selection_reason for {paper_id}.")


def canonicalize_implicit_array_ranks(
    payload: dict[str, Any], candidates: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Materialize ranks when a complete ordered ranking omits every rank field.

    The array order is itself the model's ranking decision.  This repair is
    deliberately all-or-nothing: it is allowed only when all expected paper IDs
    occur exactly once, every reason is nonempty, and every rank is absent/null.
    Partial rank omissions, duplicate IDs, or incomplete arrays remain invalid.
    """
    rows = payload.get("ranked_papers")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        return []
    expected_ids = {item["paper_id"] for item in candidates}
    ids = [row.get("paper_id") for row in rows]
    if (
        len(rows) != len(candidates)
        or len(ids) != len(set(ids))
        or set(ids) != expected_ids
        or not all(row.get("rank") is None for row in rows)
        or not all(
            isinstance(row.get("selection_reason"), str)
            and row["selection_reason"].strip()
            for row in rows
        )
    ):
        return []
    repairs: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        repairs.append(
            {
                "field": "rank",
                "paper_id": row["paper_id"],
                "received": None,
                "canonical": rank,
                "basis": "complete ordered ranked_papers array",
            }
        )
    return repairs


def canonicalize_copied_titles(
    payload: dict[str, Any], candidates: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Restore verbatim input titles without changing ranking decisions.

    Models occasionally shorten a title even when every paper ID, rank, and
    selection reason is otherwise present.  Titles are copied metadata rather
    than reviewer judgments.  We only canonicalize them after confirming that
    the response contains exactly the expected IDs and consecutive ranks, so a
    malformed or incomplete ranking cannot be made valid by this repair.
    """
    rows = payload.get("ranked_papers")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        return []
    expected_titles = {item["paper_id"]: item["title"] for item in candidates}
    count = len(candidates)
    ids = [row.get("paper_id") for row in rows]
    ranks = [row.get("rank") for row in rows]
    if (
        len(rows) != count
        or len(ids) != len(set(ids))
        or set(ids) != set(expected_titles)
        or not all(isinstance(rank, int) and not isinstance(rank, bool) for rank in ranks)
        or sorted(ranks) != list(range(1, count + 1))
    ):
        return []

    repairs: list[dict[str, str]] = []
    for row in rows:
        paper_id = row["paper_id"]
        canonical_title = expected_titles[paper_id]
        received_title = row.get("title")
        if received_title != canonical_title:
            repairs.append(
                {
                    "field": "title",
                    "paper_id": paper_id,
                    "received_title": str(received_title),
                    "canonical_title": canonical_title,
                }
            )
            row["title"] = canonical_title
    return repairs


def recover_ranking_payload(
    text: str, candidates: list[dict[str, str]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse, normalize copied metadata, and fully validate one ranking."""
    payload, repairs = parse_json_text_with_copied_title_repair(text, candidates)
    repairs.extend(canonicalize_implicit_array_ranks(payload, candidates))
    repairs.extend(canonicalize_copied_titles(payload, candidates))
    validate_ranking(payload, candidates)
    return payload, repairs


def formatting_repair_eligible(
    original_text: str, candidates: list[dict[str, str]]
) -> bool:
    """Require evidence that the substantive response covered the full set."""
    return bool(original_text.strip()) and all(
        original_text.count(item["paper_id"]) >= 1 for item in candidates
    )


def ranking_format_repair_messages(
    original_text: str, candidates: list[dict[str, str]]
) -> list[dict[str, str]]:
    identifiers = [
        {"paper_id": item["paper_id"], "title": item["title"]}
        for item in candidates
    ]
    system = (
        "You are a lossless JSON formatter, not an editor or reviewer. Convert "
        "the supplied completed ranking into the required schema. Preserve every "
        "ranking decision and selection reason. Do not re-rank, reassess, add a "
        "scientific claim, or inspect any manuscript."
    )
    user = "\n\n".join(
        [
            "# Formatting-only rules",
            (
                "Return one JSON object with ranked_papers. Preserve the existing "
                "paper order/ranks and reasons. Include every required paper exactly "
                f"once with ranks 1 through {len(candidates)}. Replace copied identifiers and titles "
                "with the exact canonical values below. Return JSON only."
            ),
            "# Canonical identifiers",
            json.dumps(identifiers, ensure_ascii=False, indent=2),
            "# Required JSON schema",
            json.dumps(ranking_schema(len(candidates)), ensure_ascii=False, indent=2),
            "# Existing completed ranking — authoritative substantive source",
            original_text,
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def run_ranking_format_repair(
    *,
    job_root: Path,
    original_text: str,
    candidates: list[dict[str, str]],
    args: argparse.Namespace,
    client: Any,
    endpoint: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Path],
]:
    repair_root = job_root / "format_repair"
    paths = {
        "root": repair_root,
        "request": repair_root / "requests" / "format_request.json",
        "response": repair_root / "responses" / "format_response.json",
    }
    request = build_request(
        model=args.model,
        messages=ranking_format_repair_messages(original_text, candidates),
        schema=ranking_schema(len(candidates)),
        temperature=0,
        reasoning_effort="medium",
        max_tokens=min(args.max_tokens, 16384),
        seed=args.seed,
        structured_output=not args.no_structured_output,
        require_parameters=not args.allow_provider_fallback,
        provider_order=args.provider_order,
        allow_provider_fallbacks=not args.disable_provider_fallbacks,
    )
    if paths["response"].is_file():
        try:
            prior_request = json.loads(paths["request"].read_text(encoding="utf-8"))
            if prior_request != request:
                raise ValueError("Saved format-repair request differs from current request.")
            response = json.loads(paths["response"].read_text(encoding="utf-8"))
            payload, repairs = recover_ranking_payload(
                extract_assistant_text(response), candidates
            )
            return payload, repairs, response, paths
        except Exception:
            if not args.retry_incomplete:
                raise
            archive_incomplete_attempt(
                paths["root"], (paths["request"], paths["response"])
            )

    write_json(paths["request"], request)
    print(
        f"[{utc_now()}] sending formatting-only repair; full manuscripts are not "
        f"included; chars={len(original_text):,} ...",
        flush=True,
    )
    response = client.chat_completion(request, endpoint=endpoint)
    write_json(paths["response"], response)
    payload, repairs = recover_ranking_payload(
        extract_assistant_text(response), candidates
    )
    return payload, repairs, response, paths


def recover_or_format_ranking_response(
    *,
    response: dict[str, Any],
    candidates: list[dict[str, str]],
    job_root: Path,
    args: argparse.Namespace,
    client: Any,
    endpoint: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    original_text = extract_assistant_text(response)
    try:
        payload, repairs = recover_ranking_payload(original_text, candidates)
        return payload, repairs, None
    except Exception as local_error:
        if not args.auto_format_repair:
            raise
        if not formatting_repair_eligible(original_text, candidates):
            raise ValueError(
                "Substantive response is incomplete and is not eligible for a "
                "formatting-only repair."
            ) from local_error
        print(
            "Local ranking recovery failed, but every paper ID is present; "
            "starting formatting-only repair without resending manuscripts.",
            flush=True,
        )
        payload, repairs, repair_response, repair_paths = run_ranking_format_repair(
            job_root=job_root,
            original_text=original_text,
            candidates=candidates,
            args=args,
            client=client,
            endpoint=endpoint,
        )
        repair_meta = {
            "status": "completed",
            "substantive_recovery_error": f"{type(local_error).__name__}: {local_error}",
            "request_sha256": sha256_file(repair_paths["request"]),
            "response": summarize_response(repair_response),
            "full_manuscripts_resent": False,
        }
        return payload, repairs, repair_meta


def injection_audit(
    *,
    job: ReviewJob,
    frozen_prompt: str,
    raw_candidate_text: str,
    candidates: list[dict[str, str]],
    request: dict[str, Any],
) -> dict[str, Any]:
    system_content = request["messages"][0]["content"]
    user_content = request["messages"][1]["content"]
    normalized_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    checks = {
        "prompt_present_verbatim_once": system_content.count(frozen_prompt) == 1,
        "normalized_candidates_present_verbatim_once": user_content.count(normalized_json) == 1,
        "all_paper_ids_present": all(item["paper_id"] in user_content for item in candidates),
        "all_titles_present": all(item["title"] in user_content for item in candidates),
        # Candidate abstracts may contain literal newlines. They are embedded in
        # JSON, where those newlines are escaped, so searching for the raw value
        # incorrectly rejects every multiline full-text evidence record.
        "all_abstracts_present": all(
            json.dumps(item["abstract"], ensure_ascii=False) in user_content
            for item in candidates
        ),
        "all_full_texts_present_when_supplied": all(
            "full_text" not in item
            or json.dumps(item["full_text"], ensure_ascii=False) in user_content
            for item in candidates
        ),
        "candidate_count_matches_schema": request.get("response_format", {})
        .get("json_schema", {})
        .get("schema", {})
        .get("properties", {})
        .get("ranked_papers", {})
        .get("minItems", len(candidates))
        == len(candidates),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Prompt-injection audit failed: {failed}")
    return {
        "schema_version": "0.1",
        "job_label": job.label,
        "status": "passed",
        "checks": checks,
        "paper_count": len(candidates),
        "paper_ids_in_input_order": [item["paper_id"] for item in candidates],
        "prompt_path": str(job.prompt_path),
        "prompt_file_sha256": sha256_file(job.prompt_path),
        "injected_prompt_utf8_sha256": sha256_text(frozen_prompt),
        "input_path": str(job.input_path),
        "input_file_sha256": sha256_file(job.input_path),
        "raw_candidate_text_utf8_sha256": sha256_text(raw_candidate_text),
        "normalized_candidate_json_utf8_sha256": sha256_text(normalized_json),
        "system_character_count": len(system_content),
        "user_character_count": len(user_content),
        "generated_at": utc_now(),
    }


def output_root_for(args: argparse.Namespace, jobs: list[ReviewJob]) -> Path:
    if args.output_root:
        return resolve(args.output_root)
    if len(jobs) != 1:
        raise ValueError("--output-root is required when running multiple jobs.")
    job = jobs[0]
    if job.source_workspace:
        return job.source_workspace / "openrouter_reviews" / slug(args.model)
    return job.input_path.parent / "openrouter_reviews" / slug(args.model)


def summarize_response(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return {
        "response_id": response.get("id"),
        "response_model": response.get("model"),
        "provider": response.get("provider"),
        "created": response.get("created"),
        "usage": usage,
    }


def compact_usage_text(response: dict[str, Any]) -> str:
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    prompt = usage.get("prompt_tokens", "?")
    completion = usage.get("completion_tokens", "?")
    reasoning = (
        usage.get("completion_tokens_details", {}).get("reasoning_tokens", "?")
        if isinstance(usage.get("completion_tokens_details"), dict)
        else "?"
    )
    total = usage.get("total_tokens", "?")
    cost = usage.get("cost", "?")
    return (
        f"prompt={prompt}, completion={completion}, reasoning={reasoning}, "
        f"total={total}, cost={cost}"
    )


def reported_reasoning_tokens(response: dict[str, Any]) -> int:
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    details = (
        usage.get("completion_tokens_details")
        if isinstance(usage.get("completion_tokens_details"), dict)
        else {}
    )
    value = details.get("reasoning_tokens", 0)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def run_smoke_test(
    *,
    run_root: Path,
    model: str,
    client: Any,
    endpoint: str,
    timeout_seconds: float,
    require_parameters: bool,
    require_reasoning_tokens: bool,
    provider_order: list[str],
    allow_provider_fallbacks: bool,
    reasoning_effort: str,
    structured_output: bool,
) -> dict[str, Any]:
    preflight_root = run_root / "preflight"
    request_path = preflight_root / "smoke_request.json"
    response_path = preflight_root / "smoke_response.json"
    manifest_path = preflight_root / "smoke_manifest.json"
    request = build_smoke_request(
        model,
        require_parameters,
        provider_order,
        allow_provider_fallbacks,
        reasoning_effort,
        structured_output,
    )
    write_json(request_path, request)
    started = utc_now()
    print(
        f"[{started}] PREFLIGHT: sending simple smoke request; "
        f"HTTP timeout={timeout_seconds:g}s ...",
        flush=True,
    )
    try:
        response = client.chat_completion(request, endpoint=endpoint)
        write_json(response_path, response)
        parsed = parse_json_text(extract_assistant_text(response))
        if parsed != {"status": "ok"}:
            raise ValueError(f"Unexpected smoke response: {parsed!r}")
        if require_reasoning_tokens and reported_reasoning_tokens(response) <= 0:
            raise ValueError(
                "Smoke response reported zero reasoning tokens; refusing to start "
                "high-reasoning SET jobs."
            )
    except Exception as exc:
        manifest = {
            "schema_version": "0.1",
            "status": "failed",
            "requested_model": model,
            "http_timeout_seconds": timeout_seconds,
            "request_sha256": sha256_file(request_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "started_at": started,
            "finished_at": utc_now(),
        }
        write_json(manifest_path, manifest)
        print(
            f"[{utc_now()}] PREFLIGHT FAILED: {type(exc).__name__}: {exc}",
            flush=True,
        )
        print("No SET request will be sent.", flush=True)
        raise RuntimeError("OpenRouter preflight failed; SET jobs were not started.") from exc

    manifest = {
        "schema_version": "0.1",
        "status": "passed",
        "requested_model": model,
        "http_timeout_seconds": timeout_seconds,
        "request_sha256": sha256_file(request_path),
        "response_sha256": sha256_file(response_path),
        "response": summarize_response(response),
        "started_at": started,
        "finished_at": utc_now(),
    }
    write_json(manifest_path, manifest)
    print(
        f"[{utc_now()}] PREFLIGHT PASSED "
        f"({compact_usage_text(response)}). Starting SET jobs.",
        flush=True,
    )
    return manifest


def aggregate_usage(manifests: Iterable[dict[str, Any]]) -> dict[str, int | float]:
    totals: dict[str, int | float] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }
    for manifest in manifests:
        response_summaries = [manifest.get("response", {})]
        format_repair = manifest.get("format_repair")
        if isinstance(format_repair, dict):
            response_summaries.append(format_repair.get("response", {}))
        for response_summary in response_summaries:
            if not isinstance(response_summary, dict):
                continue
            usage = response_summary.get("usage", {})
            if not isinstance(usage, dict):
                continue
            details = usage.get("completion_tokens_details", {})
            values: dict[str, object] = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": (
                    details.get("reasoning_tokens", 0)
                    if isinstance(details, dict)
                    else 0
                ),
                "total_tokens": usage.get("total_tokens", 0),
                "cost": usage.get("cost", 0.0),
            }
            for key, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[key] += value
    return totals


def archive_incomplete_attempt(job_root: Path, paths: Iterable[Path]) -> Path:
    """Preserve a paid-but-invalid response before retrying only that job."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_root = job_root / "attempts" / f"incomplete_{timestamp}"
    archive_root.mkdir(parents=True, exist_ok=False)
    for path in paths:
        if path.exists():
            path.replace(archive_root / path.name)
    return archive_root


def run_job(
    *,
    job: ReviewJob,
    run_root: Path,
    args: argparse.Namespace,
    client: Any,
    endpoint: str,
) -> dict[str, Any]:
    job_root = run_root / job.label
    request_path = job_root / "requests" / "selection_request.json"
    raw_response_path = job_root / "responses" / "selection_response.json"
    output_path = job_root / "outputs" / "selection_ranking.json"
    audit_path = job_root / "prompt_injection_audit.json"
    manifest_path = job_root / "manifest.json"

    frozen_prompt = job.prompt_path.read_text(encoding="utf-8")
    candidates, raw_candidate_text = load_candidates(job.input_path)
    messages = build_messages(args.system_prompt, frozen_prompt, candidates)
    request = build_request(
        model=args.model,
        messages=messages,
        schema=ranking_schema(len(candidates)),
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        seed=args.seed,
        structured_output=not args.no_structured_output,
        require_parameters=not args.allow_provider_fallback,
        provider_order=args.provider_order,
        allow_provider_fallbacks=not args.disable_provider_fallbacks,
    )
    audit = injection_audit(
        job=job,
        frozen_prompt=frozen_prompt,
        raw_candidate_text=raw_candidate_text,
        candidates=candidates,
        request=request,
    )

    def make_manifest(
        status: str,
        response_summary: dict[str, Any],
        output_normalizations: list[dict[str, str]],
        response_warnings: list[str],
    ) -> dict[str, Any]:
        return {
            "schema_version": "0.1",
            "job_label": job.label,
            "status": status,
            "provider_route": "openrouter_chat_completions",
            "requested_model": args.model,
            "requested_reasoning_effort": args.reasoning_effort,
            "reasoning_tokens_required": args.require_reasoning_tokens,
            "provider_order": args.provider_order,
            "provider_fallbacks_allowed": not args.disable_provider_fallbacks,
            "paper_count": len(candidates),
            "paper_ids_in_input_order": [item["paper_id"] for item in candidates],
            "source_workspace": str(job.source_workspace) if job.source_workspace else None,
            "prompt_file_sha256": sha256_file(job.prompt_path),
            "input_file_sha256": sha256_file(job.input_path),
            "request_sha256": sha256_file(request_path),
            "prompt_injection_audit_sha256": sha256_file(audit_path),
            "response": response_summary,
            "output_normalizations": output_normalizations,
            "response_warnings": response_warnings,
            "updated_at": utc_now(),
        }

    if output_path.exists() and not args.overwrite:
        if not request_path.exists() or not manifest_path.exists():
            raise FileExistsError(
                f"{job.label} has a completed output without its frozen request/manifest; "
                "refusing to reuse it. Use a new output root or inspect the artifact."
            )
        prior_request = json.loads(request_path.read_text(encoding="utf-8"))
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior_request != request:
            prior_model = prior_manifest.get("requested_model", "unknown")
            raise FileExistsError(
                f"{job.label} was completed with a different frozen request "
                f"(saved model={prior_model!r}, requested model={args.model!r}); "
                "refusing cross-model reuse. Use a new output root."
            )
        if prior_manifest.get("status") != "completed":
            raise FileExistsError(
                f"{job.label} has an output but manifest status is "
                f"{prior_manifest.get('status')!r}; inspect the artifact."
            )
        parsed = json.loads(output_path.read_text(encoding="utf-8"))
        validate_ranking(parsed, candidates)
        return prior_manifest
    if not args.overwrite and raw_response_path.exists():
        recovery_error: Exception | None = None
        try:
            prior_request = json.loads(request_path.read_text(encoding="utf-8"))
            if prior_request != request:
                raise ValueError("Saved request does not match the current frozen request.")
            response = json.loads(raw_response_path.read_text(encoding="utf-8"))
            choices = response.get("choices") if isinstance(response.get("choices"), list) else []
            finish_reason = choices[0].get("finish_reason") if choices else None
            if args.require_reasoning_tokens and reported_reasoning_tokens(response) <= 0:
                raise ValueError("Response reported zero reasoning tokens.")
            parsed, output_normalizations, format_repair = (
                recover_or_format_ranking_response(
                    response=response,
                    candidates=candidates,
                    job_root=job_root,
                    args=args,
                    client=client,
                    endpoint=endpoint,
                )
            )
            response_warnings = (
                [
                    f"Provider reported finish_reason={finish_reason!r}, but the preserved "
                    "assistant content decoded to a complete validator-clean ranking."
                ]
                if finish_reason != "stop"
                else []
            )
            write_json(output_path, parsed)
            manifest = make_manifest(
                "completed",
                summarize_response(response),
                output_normalizations,
                response_warnings,
            )
            if format_repair is not None:
                manifest["format_repair"] = format_repair
            write_json(manifest_path, manifest)
            suffix = (
                f"; applied {len(output_normalizations)} deterministic "
                "metadata normalization(s)"
                if output_normalizations
                else ""
            )
            print(
                f"{job.label}: recovered the existing paid response{suffix}; "
                "no API retry needed.",
                flush=True,
            )
            return manifest
        except Exception as exc:
            recovery_error = exc

        if args.retry_incomplete:
            archive_root = archive_incomplete_attempt(
                job_root,
                (request_path, raw_response_path, audit_path, manifest_path),
            )
            print(
                f"{job.label}: existing response was not recoverable ({recovery_error}); "
                f"archived incomplete prior attempt at {archive_root}",
                flush=True,
            )
        else:
            raise FileExistsError(
                f"A raw response without a validated output exists for {job.label}; "
                f"automatic recovery failed ({recovery_error}); inspect it or pass "
                "--retry-incomplete."
            )
    if not args.overwrite and manifest_path.exists():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior_request = (
            json.loads(request_path.read_text(encoding="utf-8"))
            if request_path.exists()
            else None
        )
        if prior_manifest.get("status") == "dry_run_staged" and prior_request == request:
            if not args.send:
                return prior_manifest
        elif args.retry_incomplete and not output_path.exists() and not raw_response_path.exists():
            archive_root = archive_incomplete_attempt(
                job_root,
                (request_path, audit_path, manifest_path),
            )
            print(
                f"{job.label}: archived obsolete unpaid staged request at {archive_root}",
                flush=True,
            )
        else:
            raise FileExistsError(
                f"A different prior run exists for {job.label}; inspect it or pass --overwrite."
            )

    write_json(request_path, request)
    write_json(audit_path, audit)
    status = "dry_run_staged"
    response_summary: dict[str, Any] = {}
    output_normalizations: list[dict[str, str]] = []
    response_warnings: list[str] = []
    if args.send:
        print(
            f"[{utc_now()}] {job.label}: sending {len(candidates)} papers; "
            f"HTTP timeout={args.timeout_seconds:g}s ...",
            flush=True,
        )
        response = client.chat_completion(request, endpoint=endpoint)
        print(
            f"[{utc_now()}] {job.label}: response received "
            f"({compact_usage_text(response)}); validating ...",
            flush=True,
        )
        write_json(raw_response_path, response)
        choices = response.get("choices") if isinstance(response.get("choices"), list) else []
        finish_reason = choices[0].get("finish_reason") if choices else None
        if args.require_reasoning_tokens and reported_reasoning_tokens(response) <= 0:
            raise ValueError(
                "SET response reported zero reasoning tokens; refusing to treat it as "
                "a high-reasoning result. Raw response was preserved."
            )
        parsed, output_normalizations, format_repair = (
            recover_or_format_ranking_response(
                response=response,
                candidates=candidates,
                job_root=job_root,
                args=args,
                client=client,
                endpoint=endpoint,
            )
        )
        for repair in output_normalizations:
            if repair.get("field") == "rank":
                print(
                    f"{job.label}: materialized array-order rank "
                    f"{repair['canonical']} for {repair['paper_id']} "
                    "(raw response preserved).",
                    flush=True,
                )
            else:
                print(
                    f"{job.label}: canonicalized copied title for "
                    f"{repair['paper_id']} (raw response preserved).",
                    flush=True,
                )
        if finish_reason != "stop":
            warning = (
                f"Provider reported finish_reason={finish_reason!r}, but the assistant "
                "content decoded to a complete validator-clean ranking; accepted locally."
            )
            response_warnings.append(warning)
            print(f"{job.label}: WARNING: {warning}", flush=True)
        write_json(output_path, parsed)
        status = "completed"
        response_summary = summarize_response(response)
        print(f"[{utc_now()}] {job.label}: VALID and saved.", flush=True)

    manifest = make_manifest(
        status, response_summary, output_normalizations, response_warnings
    )
    if args.send and format_repair is not None:
        manifest["format_repair"] = format_repair
    write_json(manifest_path, manifest)
    return manifest


def validate_runtime_args(args: argparse.Namespace) -> None:
    if not 0 <= args.temperature <= 2:
        raise ValueError("--temperature must be between 0 and 2.")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")
    if args.smoke_timeout_seconds <= 0:
        raise ValueError("--smoke-timeout-seconds must be positive.")
    if args.disable_provider_fallbacks and not args.provider_order:
        raise ValueError("--disable-provider-fallbacks requires --provider-order.")


def print_summary(mode: str, model: str, root: Path, manifests: Iterable[dict[str, Any]]) -> None:
    print(f"Mode: {mode}")
    print(f"Model: {model}")
    for manifest in manifests:
        print(
            f"{manifest['job_label']}: {manifest['status']} "
            f"({manifest['paper_count']} papers)"
        )
    print(f"Artifacts: {root}")


def main() -> int:
    args = parse_args()
    validate_runtime_args(args)
    jobs = build_jobs(args)
    run_root = output_root_for(args, jobs)
    config_path = resolve(args.config)
    config = load_toml(config_path)
    provider_config = dict(config.get("provider", {}))
    provider_config["timeout_seconds"] = args.timeout_seconds
    endpoint = str(provider_config.get("chat_endpoint", "/chat/completions"))

    client = None
    smoke_manifest: dict[str, Any] = {"status": "not_run_in_dry_run"}
    if args.send:
        env_path = resolve(args.env_file)
        load_env_file(env_path)
        if args.skip_smoke_test:
            smoke_manifest = {"status": "skipped_by_user"}
            print("PREFLIGHT: skipped by --skip-smoke-test.", flush=True)
        else:
            smoke_provider_config = dict(provider_config)
            smoke_provider_config["timeout_seconds"] = args.smoke_timeout_seconds
            smoke_client = make_llm_client(smoke_provider_config, env_path)
            smoke_manifest = run_smoke_test(
                run_root=run_root,
                model=args.model,
                client=smoke_client,
                endpoint=endpoint,
                timeout_seconds=args.smoke_timeout_seconds,
                require_parameters=not args.allow_provider_fallback,
                require_reasoning_tokens=args.require_reasoning_tokens,
                provider_order=args.provider_order,
                allow_provider_fallbacks=not args.disable_provider_fallbacks,
                reasoning_effort=args.reasoning_effort,
                structured_output=not args.no_structured_output,
            )
        client = make_llm_client(provider_config, env_path)

    manifests = [
        run_job(
            job=job,
            run_root=run_root,
            args=args,
            client=client,
            endpoint=endpoint,
        )
        for job in jobs
    ]
    usage_totals = aggregate_usage(manifests)
    summary = {
        "schema_version": "0.1",
        "mode": "send" if args.send else "dry_run",
        "provider": provider_config.get("name", "openrouter"),
        "requested_model": args.model,
        "requested_reasoning_effort": args.reasoning_effort,
        "reasoning_tokens_required": args.require_reasoning_tokens,
        "provider_order": args.provider_order,
        "provider_fallbacks_allowed": not args.disable_provider_fallbacks,
        "config_path": str(config_path),
        "file_upload_used": False,
        "transport": "Prompt and complete candidate JSON embedded in chat-completions messages.",
        "http_timeout_seconds": args.timeout_seconds,
        "smoke_test": smoke_manifest,
        "jobs": manifests,
        "usage_totals": usage_totals,
        "updated_at": utc_now(),
    }
    write_json(run_root / "run_summary.json", summary)
    print_summary("SEND" if args.send else "DRY RUN", args.model, run_root, manifests)
    if args.send:
        print(f"Usage totals: {json.dumps(usage_totals, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
