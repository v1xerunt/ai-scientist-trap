"""Agent run harness for AI scientist experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trap_ai.config import load_toml, resolve_config_path
from trap_ai.env import load_env_file
from trap_ai.claim_extraction import write_claim_table_from_extractor_response
from trap_ai.plasmode import OUTCOME as DEFAULT_OUTCOME
from trap_ai.randomness import DEFAULT_SEED, set_global_seed
from trap_ai.schemas import CLAIM_FIELD_DEFINITIONS, CLAIM_TABLE_COLUMNS, CLAIM_TYPE_DEFINITIONS, CLAIM_TYPES, FINDING_RESULTS


RUN_SUBDIRS = [
    "input",
    "prompt",
    "request",
    "response",
    "code",
    "logs",
    "intermediate",
    "plots",
    "paper",
    "claim_tables",
    "reviews",
    "metadata",
]

DEFAULT_AUTHOR_PARADIGM = "iterative_coding"
DEFAULT_MAX_CODE_REPAIRS = 3
DEFAULT_MAX_SUPPLEMENTAL_ANALYSES = 3
DEFAULT_MAX_SUFFICIENCY_REVIEWS = 2
DEFAULT_CODE_TIMEOUT_SECONDS = 600.0
MANUSCRIPT_TEXT_ARTIFACT_LIMIT = 20000
MANUSCRIPT_TABLE_MAX_ROWS = 200
COMPACT_SUMMARY_KEYS = [
    "primary_results",
    "secondary_results",
    "robustness_results",
    "generated_tables",
    "generated_figures",
    "limitations",
]
SUFFICIENCY_STAGE = "analysis_sufficiency_review"
AUTHOR_PARADIGMS = {
    "single_shot",
    "iterative_coding",
    "data_to_paper",
    "ai_scientist",
}
REVIEWER_STYLES = {
    "generic",
    "statistical",
    "journal",
}

CODE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["analysis_code"],
    "properties": {
        "analysis_code": {
            "type": "string",
            "description": "Complete Python analysis script.",
        },
    },
}

CLAIM_EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": CLAIM_TABLE_COLUMNS,
                "properties": {
                    "claim_id": {"type": "string"},
                    "exposure": {"type": "string"},
                    "outcome": {"type": "string"},
                    "claim_type": {"type": "string", "enum": CLAIM_TYPES},
                    "finding_result": {"type": "string", "enum": FINDING_RESULTS},
                    "model_type": {"type": "string"},
                    "covariates": {"type": "string"},
                    "estimate": {"type": "string"},
                    "p_value": {"type": "string"},
                    "survey_design_used": {"type": "boolean"},
                    "evidence_source": {"type": "string"},
                },
            },
        },
    },
}

REVIEWER_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["manuscript_title", "statistical_review", "major_points", "minor_points", "recommended_revisions"],
    "properties": {
        "manuscript_title": {"type": "string"},
        "statistical_review": {"type": "string"},
        "major_points": {"type": "array", "items": {"type": "string"}},
        "minor_points": {"type": "array", "items": {"type": "string"}},
        "recommended_revisions": {"type": "array", "items": {"type": "string"}},
    },
}

SELECTION_REVIEW_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ranked_papers"],
    "properties": {
        "ranked_papers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["rank", "paper_id", "title", "selection_reason"],
                "properties": {
                    "rank": {"type": "integer"},
                    "paper_id": {"type": "string"},
                    "title": {"type": "string"},
                    "selection_reason": {"type": "string"},
                },
            },
        },
    },
}

STRING_ARRAY_JSON_SCHEMA = {"type": "array", "items": {"type": "string"}}

AUTHOR_STAGE_JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "analysis_plan": {
        "type": "object",
        "additionalProperties": False,
        "required": ["analysis_plan", "planned_outputs"],
        "properties": {
            "analysis_plan": {"type": "string"},
            "planned_outputs": STRING_ARRAY_JSON_SCHEMA,
        },
    },
    "research_plan": {
        "type": "object",
        "additionalProperties": False,
        "required": ["research_question", "research_plan", "planned_outputs"],
        "properties": {
            "research_question": {"type": "string"},
            "research_plan": {"type": "string"},
            "planned_outputs": STRING_ARRAY_JSON_SCHEMA,
        },
    },
    "preprocessing_plan": {
        "type": "object",
        "additionalProperties": False,
        "required": ["preprocessing_plan", "planned_outputs"],
        "properties": {
            "preprocessing_plan": {"type": "string"},
            "planned_outputs": STRING_ARRAY_JSON_SCHEMA,
        },
    },
    "results_selection": {
        "type": "object",
        "additionalProperties": False,
        "required": ["main_results", "secondary_results", "results_not_emphasized", "selection_reason"],
        "properties": {
            "main_results": STRING_ARRAY_JSON_SCHEMA,
            "secondary_results": STRING_ARRAY_JSON_SCHEMA,
            "results_not_emphasized": STRING_ARRAY_JSON_SCHEMA,
            "selection_reason": {"type": "string"},
        },
    },
    "idea_generation": {
        "type": "object",
        "additionalProperties": False,
        "required": ["proposed_ideas", "screening_rationale"],
        "properties": {
            "proposed_ideas": STRING_ARRAY_JSON_SCHEMA,
            "screening_rationale": {"type": "string"},
        },
    },
    "idea_selection": {
        "type": "object",
        "additionalProperties": False,
        "required": ["selected_ideas", "abandoned_ideas", "selection_reason"],
        "properties": {
            "selected_ideas": STRING_ARRAY_JSON_SCHEMA,
            "abandoned_ideas": STRING_ARRAY_JSON_SCHEMA,
            "selection_reason": {"type": "string"},
        },
    },
    "experiment_design": {
        "type": "object",
        "additionalProperties": False,
        "required": ["experiment_design", "planned_outputs"],
        "properties": {
            "experiment_design": {"type": "string"},
            "planned_outputs": STRING_ARRAY_JSON_SCHEMA,
        },
    },
    "self_review": {
        "type": "object",
        "additionalProperties": False,
        "required": ["self_review_score", "strengths", "concerns", "revision_plan"],
        "properties": {
            "self_review_score": {"type": "integer"},
            "strengths": STRING_ARRAY_JSON_SCHEMA,
            "concerns": STRING_ARRAY_JSON_SCHEMA,
            "revision_plan": {"type": "string"},
        },
    },
    "revision": {
        "type": "object",
        "additionalProperties": False,
        "required": ["revision_plan", "supplementary_analysis_needed", "analysis_revision_required"],
        "properties": {
            "revision_plan": {"type": "string"},
            "supplementary_analysis_needed": STRING_ARRAY_JSON_SCHEMA,
            "analysis_revision_required": {"type": "boolean"},
        },
    },
}

ANALYSIS_SUFFICIENCY_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["objective_met", "supplementary_analysis_needed"],
    "properties": {
        "objective_met": {"type": "boolean"},
        "supplementary_analysis_needed": STRING_ARRAY_JSON_SCHEMA,
    },
}


@dataclass(frozen=True)
class AgentRunResult:
    run_id: str
    run_dir: Path
    status: str
    request_path: Path
    response_path: Path
    mini_paper_path: Path
    claim_table_path: Path
    code_path: Path


@dataclass(frozen=True)
class AgentHarnessReport:
    root: Path
    run_root: Path
    dataset_dir: Path
    model: str
    prompt_regime: str
    dry_run: bool
    runs: list[AgentRunResult]
    author_paradigm: str = ""


@dataclass(frozen=True)
class ClaimExtractionHarnessReport:
    root: Path
    run_dir: Path
    model: str
    dry_run: bool
    status: str
    request_path: Path
    response_path: Path
    claims_path: Path
    manifest_path: Path
    claim_count: int


@dataclass(frozen=True)
class ReviewerHarnessReport:
    root: Path
    run_dir: Path
    model: str
    reviewer_style: str
    dry_run: bool
    status: str
    request_path: Path
    response_path: Path
    review_path: Path
    manifest_path: Path
    major_point_count: int


@dataclass(frozen=True)
class SelectionReviewHarnessReport:
    root: Path
    run_root: Path
    model: str
    dry_run: bool
    status: str
    request_path: Path
    response_path: Path
    review_path: Path
    manifest_path: Path
    paper_count: int
    group_count: int = 1


@dataclass(frozen=True)
class OpenAIBatchReport:
    root: Path
    task: str
    batch_id: str
    status: str
    request_count: int
    manifest_path: Path
    request_jsonl_path: Path
    output_jsonl_path: Path | None = None
    completed_count: int = 0
    failed_count: int = 0


@dataclass(frozen=True)
class AgentRunInspection:
    run_dir: Path
    manifest: dict[str, Any]
    steps: list[dict[str, Any]]
    execution_attempts: list[dict[str, Any]]
    generated_files: list[str]
    stdout_tail: str
    stderr_tail: str
    mini_paper_excerpt: str


@dataclass(frozen=True)
class ExecutionResult:
    returncode: int
    stdout_path: Path
    stderr_path: Path
    duration_seconds: float
    generated_files: list[str]


@dataclass(frozen=True)
class ChatStepResult:
    request_path: Path
    response_path: Path
    message_path: Path
    response: dict[str, Any]
    text: str
    error: str | None


@dataclass(frozen=True)
class FixtureOpenRouterClient:
    """Deterministic client for dry/local tests of the interactive loop."""

    responses: list[dict[str, Any]]

    def chat_completion(self, payload: dict[str, Any], endpoint: str = "/chat/completions") -> dict[str, Any]:
        del endpoint
        index = int(payload.get("metadata", {}).get("loop_step_index", 0))
        return self.responses[min(index, len(self.responses) - 1)]


class OpenRouterClient:
    """Minimal OpenRouter chat-completions client."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 180,
        http_referer: str = "",
        app_title: str = "trap-ai",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.http_referer = http_referer
        self.app_title = app_title

    def chat_completion(self, payload: dict[str, Any], endpoint: str = "/chat/completions") -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_title,
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc


class OpenAIChatClient:
    """Minimal direct OpenAI chat-completions client for evaluator steps."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 180,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def chat_completion(self, payload: dict[str, Any], endpoint: str = "/chat/completions") -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    def upload_batch_file(self, path: Path) -> dict[str, Any]:
        boundary = f"----trap-ai-{hashlib.sha256(str(path).encode('utf-8')).hexdigest()[:24]}"
        body = multipart_form_data(
            boundary=boundary,
            fields={"purpose": "batch"},
            files={"file": path},
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        return self._request_json("POST", "/files", body, headers)

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }
        if metadata:
            payload["metadata"] = metadata
        return self._request_json(
            "POST",
            "/batches",
            json.dumps(payload).encode("utf-8"),
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )

    def retrieve_batch(self, batch_id: str) -> dict[str, Any]:
        return self._request_json(
            "GET",
            f"/batches/{batch_id}",
            None,
            {"Authorization": f"Bearer {self.api_key}"},
        )

    def retrieve_file_content(self, file_id: str) -> str:
        response = self._request_bytes(
            "GET",
            f"/files/{file_id}/content",
            None,
            {"Authorization": f"Bearer {self.api_key}"},
        )
        return response.decode("utf-8", errors="replace")

    def _request_json(
        self,
        method: str,
        endpoint: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        response = self._request_bytes(method, endpoint, body, headers)
        return json.loads(response.decode("utf-8"))

    def _request_bytes(
        self,
        method: str,
        endpoint: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> bytes:
        url = f"{self.base_url}{endpoint}"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc


def multipart_form_data(boundary: str, fields: dict[str, str], files: dict[str, Path]) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, path in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{path.name}"\r\n'
                ).encode("utf-8"),
                b"Content-Type: application/jsonl\r\n\r\n",
                path.read_bytes(),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks)


def make_llm_client(provider_config: dict[str, Any], env_path: Path) -> OpenRouterClient | OpenAIChatClient:
    """Create a chat-completions client for reviewer/evaluator style LLM calls."""

    provider_name = str(provider_config.get("name", "")).lower()
    base_url = str(provider_config.get("base_url", "")).rstrip("/")
    is_openai = provider_name == "openai" or "api.openai.com" in base_url.lower()
    if is_openai:
        api_key_env = str(provider_config.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Missing OpenAI API key. Set {api_key_env} in {env_path} or the process environment."
            )
        return OpenAIChatClient(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            timeout_seconds=float(provider_config.get("timeout_seconds", 180)),
        )

    api_key_env = str(provider_config.get("api_key_env", "OPENROUTER_API_KEY"))
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"Missing OpenRouter API key. Set {api_key_env} in {env_path} or the process environment."
        )
    return OpenRouterClient(
        api_key=api_key,
        base_url=base_url or "https://openrouter.ai/api/v1",
        timeout_seconds=float(provider_config.get("timeout_seconds", 180)),
        http_referer=os.environ.get(str(provider_config.get("http_referer_env", "OPENROUTER_HTTP_REFERER")), ""),
        app_title=os.environ.get(str(provider_config.get("app_title_env", "OPENROUTER_APP_TITLE")), "trap-ai"),
    )


def run_agent_batch(
    root: Path,
    dataset_dir: Path,
    config_path: Path,
    prompts_path: Path,
    prompt_regime: str,
    n_runs: int,
    model: str | None = None,
    run_root: Path | None = None,
    env_file: Path | None = None,
    dry_run: bool = False,
    execute_code: bool | None = None,
    overwrite: bool = False,
) -> AgentHarnessReport:
    """Run or dry-run N AI scientist jobs for one dataset."""

    root = root.resolve()
    config = load_toml(resolve_config_path(root, config_path))
    prompts = load_toml(resolve_config_path(root, prompts_path))
    provider_config = config.get("provider", {})
    run_config = config.get("run", {})
    limits_config = config.get("limits", {})
    model_config = config.get("model", {})
    selected_model = model or str(model_config.get("default", "deepseek/deepseek-v4-flash"))
    selected_run_root = root / (run_root or Path(str(run_config.get("run_root", "runs/agent_runs"))))
    selected_execute_code = bool(run_config.get("execute_code", False) if execute_code is None else execute_code)

    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    env_path = resolve_config_path(root, env_file or Path(".env"))
    load_env_file(env_path)
    client = None
    if not dry_run:
        api_key_env = str(provider_config.get("api_key_env", "OPENROUTER_API_KEY"))
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Missing OpenRouter API key. Set {api_key_env} in {env_path} or the process environment."
            )
        client = OpenRouterClient(
            api_key=api_key,
            base_url=str(provider_config.get("base_url", "https://openrouter.ai/api/v1")),
            timeout_seconds=float(provider_config.get("timeout_seconds", 180)),
            http_referer=os.environ.get(str(provider_config.get("http_referer_env", "OPENROUTER_HTTP_REFERER")), ""),
            app_title=os.environ.get(str(provider_config.get("app_title_env", "OPENROUTER_APP_TITLE")), "trap-ai"),
        )

    dataset_dir = resolve_config_path(root, dataset_dir)
    prompt_payload = build_prompt_payload(
        dataset_dir=dataset_dir,
        prompts=prompts,
        prompt_regime=prompt_regime,
        dataset_preview_rows=int(limits_config.get("dataset_preview_rows", 25)),
        codebook_max_variables=int(limits_config.get("codebook_max_variables", 80)),
    )
    dataset_id = public_dataset_slug(prompt_payload, dataset_dir)

    seed = DEFAULT_SEED
    set_global_seed(seed)
    results: list[AgentRunResult] = []
    for run_index in range(1, n_runs + 1):
        run_seed = seed + run_index - 1
        run_id = make_run_id(dataset_id, prompt_regime, selected_model, run_index)
        run_dir = selected_run_root / dataset_id / prompt_regime / sanitize_slug(selected_model) / run_id
        if run_dir.exists() and overwrite:
            shutil.rmtree(run_dir)
        ensure_run_tree(run_dir)
        result = run_single_agent(
            run_dir=run_dir,
            run_id=run_id,
            run_index=run_index,
            run_seed=run_seed,
            dataset_dir=dataset_dir,
            prompt_payload=prompt_payload,
            model=selected_model,
            provider_config=provider_config,
            model_config=model_config,
            client=client,
            dry_run=dry_run,
            execute_code=selected_execute_code,
            copy_dataset=bool(run_config.get("copy_dataset", True)),
        )
        results.append(result)

    return AgentHarnessReport(
        root=root,
        run_root=selected_run_root,
        dataset_dir=dataset_dir,
        model=selected_model,
        prompt_regime=prompt_regime,
        dry_run=dry_run,
        runs=results,
    )


def build_prompt_payload(
    dataset_dir: Path,
    prompts: dict[str, Any],
    prompt_regime: str,
    dataset_preview_rows: int = 25,
    codebook_max_variables: int = 80,
) -> dict[str, Any]:
    dataset_path = dataset_dir / "dataset.csv"
    codebook_path = dataset_dir / "public_codebook.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset.csv: {dataset_path}")
    if not codebook_path.exists():
        raise FileNotFoundError(f"Missing public_codebook.json: {codebook_path}")

    regimes = prompts.get("regimes", {})
    if prompt_regime not in regimes:
        raise ValueError(f"Unknown prompt regime {prompt_regime!r}. Available: {', '.join(regimes)}")
    common = prompts.get("common", {})
    codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
    codebook_variables = codebook.get("variables", [])[:codebook_max_variables]
    preview = read_dataset_preview(dataset_path, dataset_preview_rows)
    task = str(regimes[prompt_regime].get("task", ""))
    output_contract = str(common.get("output_contract", ""))
    dataset_context = str(common.get("dataset_context", ""))
    user_prompt = "\n\n".join(
        [
            "# Task",
            task,
            "# Dataset Context",
            dataset_context,
            "# Public Codebook Summary",
            json.dumps(
                {
                    "dataset_id": codebook.get("dataset_id", dataset_dir.name),
                    "outcome": codebook.get("outcome", DEFAULT_OUTCOME),
                    "n_rows": codebook.get("n_rows"),
                    "n_columns": codebook.get("n_columns"),
                    "variables": codebook_variables,
                },
                indent=2,
            ),
            "# Dataset Preview",
            preview,
            "# Output Contract",
            output_contract,
        ]
    )
    messages = [
        {"role": "system", "content": str(common.get("system", ""))},
        {"role": "user", "content": user_prompt},
    ]
    return {
        "messages": messages,
        "task_prompt": task,
        "output_contract": output_contract,
        "dataset_preview": preview,
        "codebook_summary": {
            "dataset_id": codebook.get("dataset_id", dataset_dir.name),
            "outcome": codebook.get("outcome", DEFAULT_OUTCOME),
            "n_rows": codebook.get("n_rows"),
            "n_columns": codebook.get("n_columns"),
            "variables": codebook_variables,
        },
    }


def run_single_agent(
    run_dir: Path,
    run_id: str,
    run_index: int,
    run_seed: int,
    dataset_dir: Path,
    prompt_payload: dict[str, Any],
    model: str,
    provider_config: dict[str, Any],
    model_config: dict[str, Any],
    client: OpenRouterClient | None,
    dry_run: bool,
    execute_code: bool,
    copy_dataset: bool,
) -> AgentRunResult:
    save_input_package(run_dir, dataset_dir, prompt_payload, copy_dataset=copy_dataset)
    request_payload = {
        "model": model,
        "messages": prompt_payload["messages"],
        "temperature": float(model_config.get("temperature", 0.4)),
        "seed": run_seed,
        "response_format": {"type": "json_object"},
        "metadata": {
            "project": "cardiometabolic-risk-study",
            "run_id": run_id,
            "prompt_regime": run_dir.parent.parent.name,
            "dataset_id": public_dataset_slug(prompt_payload, dataset_dir),
        },
    }
    request_path = run_dir / "request" / "openrouter_request.json"
    write_json(request_path, request_payload)

    started = utc_now()
    response_path = run_dir / "response" / "openrouter_response.json"
    message_path = run_dir / "response" / "assistant_message.md"
    status = "dry_run"
    response: dict[str, Any] = {
        "dry_run": True,
        "message": "Dry run only. No OpenRouter request was sent.",
    }
    assistant_text = ""

    if not dry_run:
        if client is None:
            raise RuntimeError("OpenRouter client is required for non-dry runs.")
        status = "completed"
        try:
            response = client.chat_completion(
                request_payload,
                endpoint=str(provider_config.get("chat_endpoint", "/chat/completions")),
            )
            assistant_text = extract_assistant_text(response)
        except Exception as exc:  # noqa: BLE001 - persisted artifact should capture provider failures
            status = "failed"
            response = {"error": str(exc)}
            assistant_text = ""

    write_json(response_path, response)
    message_path.write_text(assistant_text, encoding="utf-8")

    parsed = parse_agent_response(assistant_text)
    code_path = run_dir / "code" / "analysis.py"
    mini_paper_path = run_dir / "paper" / "mini_paper.md"
    claim_table_path = run_dir / "claim_tables" / "claims.csv"
    code_path.write_text(parsed.get("analysis_code", ""), encoding="utf-8")
    mini_paper_path.write_text(parsed.get("mini_paper_markdown", ""), encoding="utf-8")
    (run_dir / "logs" / "stdout.txt").write_text("", encoding="utf-8")
    (run_dir / "logs" / "stderr.txt").write_text("", encoding="utf-8")

    if execute_code and status == "completed" and code_path.read_text(encoding="utf-8").strip():
        run_analysis_code(run_dir, code_path)
    legacy_execution = aggregate_analysis_execution_result(run_dir, load_execution_result_from_run(run_dir))
    if mini_paper_path.read_text(encoding="utf-8", errors="replace").strip():
        write_review_manuscript(
            run_dir,
            mini_paper_path.read_text(encoding="utf-8", errors="replace"),
            legacy_execution,
        )

    write_json(
        run_dir / "metadata" / "run_manifest.json",
        {
            "run_id": run_id,
            "run_index": run_index,
            "seed": run_seed,
            "dataset_dir": str(dataset_dir),
            "model": model,
            "dry_run": dry_run,
            "execute_code": execute_code,
            "status": status,
            "started_at": started,
            "finished_at": utc_now(),
            "artifacts": {
                "input": "input/",
                "request": "request/openrouter_request.json",
                "response": "response/openrouter_response.json",
                "assistant_message": "response/assistant_message.md",
                "code": "code/analysis.py",
                "stdout": "logs/stdout.txt",
                "stderr": "logs/stderr.txt",
                "intermediate": "intermediate/",
                "plots": "plots/",
                "mini_paper": "paper/mini_paper.md",
                "review_manuscript": "paper/review_manuscript.md",
                "traceability_appendix": relative_or_empty(
                    run_dir / "metadata" / "traceability_appendix.json",
                    run_dir,
                ),
            },
        },
    )

    return AgentRunResult(
        run_id=run_id,
        run_dir=run_dir,
        status=status,
        request_path=request_path,
        response_path=response_path,
        mini_paper_path=mini_paper_path,
        claim_table_path=claim_table_path,
        code_path=code_path,
    )


def run_interactive_agent_batch(
    root: Path,
    dataset_dir: Path,
    config_path: Path,
    prompts_path: Path,
    prompt_regime: str,
    n_runs: int,
    model: str | None = None,
    run_root: Path | None = None,
    env_file: Path | None = None,
    dry_run: bool = False,
    fixture: bool = False,
    max_code_repairs: int | None = None,
    evidence_audit: bool | None = None,
    max_supplemental_analyses: int | None = None,
    max_sufficiency_reviews: int | None = None,
    author_paradigm: str | None = None,
    parallel_workers: int = 1,
    overwrite: bool = False,
) -> AgentHarnessReport:
    """Run an author AI scientist workflow.

    Every workflow preserves the core code -> execute -> paper structure. The
    author_paradigm controls which planning, selection, and audit stages wrap it.
    """

    root = root.resolve()
    config = load_toml(resolve_config_path(root, config_path))
    prompts = load_toml(resolve_config_path(root, prompts_path))
    provider_config = config.get("provider", {})
    run_config = config.get("run", {})
    limits_config = config.get("limits", {})
    model_config = config.get("model", {})
    selected_model = model or str(model_config.get("default", "deepseek/deepseek-v4-flash"))
    selected_run_root = root / (run_root or Path(str(run_config.get("run_root", "runs/agent_runs"))))
    selected_max_code_repairs = int(
        run_config.get("max_code_repairs", DEFAULT_MAX_CODE_REPAIRS)
        if max_code_repairs is None
        else max_code_repairs
    )
    configured_sufficiency_review = run_config.get(
        "analysis_sufficiency_review",
        run_config.get("evidence_audit", True),
    )
    selected_sufficiency_review = bool(
        configured_sufficiency_review if evidence_audit is None else evidence_audit
    )
    selected_max_supplemental_analyses = int(
        run_config.get("max_supplemental_analyses", DEFAULT_MAX_SUPPLEMENTAL_ANALYSES)
        if max_supplemental_analyses is None
        else max_supplemental_analyses
    )
    selected_max_sufficiency_reviews = int(
        run_config.get("max_sufficiency_reviews", DEFAULT_MAX_SUFFICIENCY_REVIEWS)
        if max_sufficiency_reviews is None
        else max_sufficiency_reviews
    )
    selected_author_paradigm = normalize_author_paradigm(
        str(run_config.get("author_paradigm", DEFAULT_AUTHOR_PARADIGM) if author_paradigm is None else author_paradigm)
    )

    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    if selected_max_code_repairs < 0:
        raise ValueError("max_code_repairs must be non-negative")
    if selected_max_supplemental_analyses < 0:
        raise ValueError("max_supplemental_analyses must be non-negative")
    if selected_max_sufficiency_reviews < 0:
        raise ValueError("max_sufficiency_reviews must be non-negative")
    if parallel_workers < 1:
        raise ValueError("parallel_workers must be at least 1")

    env_path = resolve_config_path(root, env_file or Path(".env"))
    load_env_file(env_path)
    client: OpenRouterClient | FixtureOpenRouterClient | None = None
    if fixture:
        client = FixtureOpenRouterClient(
            responses=fixture_responses_for_paradigm(selected_author_paradigm)
        )
    elif not dry_run:
        api_key_env = str(provider_config.get("api_key_env", "OPENROUTER_API_KEY"))
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Missing OpenRouter API key. Set {api_key_env} in {env_path} or the process environment."
            )
        client = OpenRouterClient(
            api_key=api_key,
            base_url=str(provider_config.get("base_url", "https://openrouter.ai/api/v1")),
            timeout_seconds=float(provider_config.get("timeout_seconds", 180)),
            http_referer=os.environ.get(str(provider_config.get("http_referer_env", "OPENROUTER_HTTP_REFERER")), ""),
            app_title=os.environ.get(str(provider_config.get("app_title_env", "OPENROUTER_APP_TITLE")), "trap-ai"),
        )

    dataset_dir = resolve_config_path(root, dataset_dir)
    prompt_payload = build_prompt_payload(
        dataset_dir=dataset_dir,
        prompts=prompts,
        prompt_regime=prompt_regime,
        dataset_preview_rows=int(limits_config.get("dataset_preview_rows", 25)),
        codebook_max_variables=int(limits_config.get("codebook_max_variables", 80)),
    )
    dataset_id = public_dataset_slug(prompt_payload, dataset_dir)

    seed = DEFAULT_SEED
    set_global_seed(seed)
    def run_one(run_index: int) -> AgentRunResult:
        run_seed = seed + run_index - 1
        run_id = make_run_id(dataset_id, f"{prompt_regime}-{selected_author_paradigm}", selected_model, run_index)
        run_dir = (
            selected_run_root
            / dataset_id
            / prompt_regime
            / selected_author_paradigm
            / sanitize_slug(selected_model)
            / run_id
        )
        if run_dir.exists() and overwrite:
            shutil.rmtree(run_dir)
        ensure_run_tree(run_dir)
        return run_single_interactive_agent(
            run_dir=run_dir,
            run_id=run_id,
            run_index=run_index,
            run_seed=run_seed,
            dataset_dir=dataset_dir,
            prompt_payload=prompt_payload,
            model=selected_model,
            provider_config=provider_config,
            model_config=model_config,
            client=client,
            dry_run=dry_run,
            copy_dataset=bool(run_config.get("copy_dataset", True)),
            max_code_repairs=selected_max_code_repairs,
            evidence_audit=selected_sufficiency_review,
            max_supplemental_analyses=selected_max_supplemental_analyses,
            max_sufficiency_reviews=selected_max_sufficiency_reviews,
            author_paradigm=selected_author_paradigm,
        )

    if parallel_workers == 1 or n_runs == 1:
        results = [run_one(run_index) for run_index in range(1, n_runs + 1)]
    else:
        results_by_index: dict[int, AgentRunResult] = {}
        with ThreadPoolExecutor(max_workers=min(parallel_workers, n_runs)) as executor:
            futures = {
                executor.submit(run_one, run_index): run_index
                for run_index in range(1, n_runs + 1)
            }
            for future in as_completed(futures):
                run_index = futures[future]
                results_by_index[run_index] = future.result()
        results = [results_by_index[index] for index in range(1, n_runs + 1)]

    return AgentHarnessReport(
        root=root,
        run_root=selected_run_root,
        dataset_dir=dataset_dir,
        model=selected_model,
        prompt_regime=prompt_regime,
        dry_run=dry_run,
        runs=results,
        author_paradigm=selected_author_paradigm,
    )


def run_claim_extraction_for_run(
    root: Path,
    run_dir: Path,
    config_path: Path,
    model: str | None = None,
    env_file: Path | None = None,
    dry_run: bool = False,
    fixture: bool = False,
    overwrite: bool = False,
) -> ClaimExtractionHarnessReport:
    """Run the evaluator LLM claim extraction step for one completed author run."""

    root = root.resolve()
    run_dir = resolve_config_path(root, run_dir)
    config = load_toml(resolve_config_path(root, config_path))
    provider_config = config.get("provider", {})
    model_config = config.get("model", {})
    selected_model = model or str(model_config.get("default", "deepseek/deepseek-v4-pro"))
    claims_path = run_dir / "claim_tables" / "claims.csv"
    manifest_path = run_dir / "metadata" / "claim_extraction_manifest.json"
    if claims_path.exists() and not overwrite:
        raise FileExistsError(f"Claim table already exists: {claims_path}. Use --overwrite to replace it.")

    env_path = resolve_config_path(root, env_file or Path(".env"))
    load_env_file(env_path)
    client: OpenRouterClient | OpenAIChatClient | FixtureOpenRouterClient | None = None
    if fixture:
        client = FixtureOpenRouterClient(
            responses=[make_fixture_response(fixture_claim_extraction_json(), response_id="fixture-claim-extraction")]
        )
    elif not dry_run:
        client = make_llm_client(provider_config, env_path)

    run_manifest = load_json_if_exists(run_dir / "metadata" / "run_manifest.json")
    run_id = str(run_manifest.get("run_id", run_dir.name))
    messages = build_claim_extraction_messages(run_dir)
    request = make_claim_extraction_request(
        model=selected_model,
        messages=messages,
        model_config=model_config,
        run_id=run_id,
    )
    step = run_chat_step(
        run_dir=run_dir,
        request_payload=request,
        artifact_stem="claim_extraction",
        client=client,
        provider_config=provider_config,
        dry_run=dry_run,
        skipped_message="No claim extraction request sent.",
    )
    status = "dry_run" if dry_run else "completed"
    if step.error:
        status = "failed"
    extraction = write_claim_table_from_extractor_response(
        run_dir,
        step.text,
        claims_path,
        step.response_path,
    )
    return ClaimExtractionHarnessReport(
        root=root,
        run_dir=run_dir,
        model=selected_model,
        dry_run=dry_run,
        status=status,
        request_path=step.request_path,
        response_path=step.response_path,
        claims_path=claims_path,
        manifest_path=manifest_path,
        claim_count=extraction.claim_count,
    )


def run_reviewer_for_run(
    root: Path,
    run_dir: Path,
    config_path: Path,
    reviewer_style: str = "statistical",
    model: str | None = None,
    env_file: Path | None = None,
    dry_run: bool = False,
    fixture: bool = False,
    overwrite: bool = False,
) -> ReviewerHarnessReport:
    """Run an external reviewer LLM for one completed author run."""

    root = root.resolve()
    run_dir = resolve_config_path(root, run_dir)
    ensure_run_tree(run_dir)
    config = load_toml(resolve_config_path(root, config_path))
    provider_config = config.get("provider", {})
    model_config = config.get("model", {})
    selected_model = model or str(model_config.get("default", "deepseek/deepseek-v4-pro"))
    style = normalize_reviewer_style(reviewer_style)
    artifact_stem = f"reviewer_{style}"
    review_path = run_dir / "reviews" / f"{style}_review.json"
    manifest_path = run_dir / "metadata" / f"{artifact_stem}_manifest.json"
    if review_path.exists() and not overwrite:
        raise FileExistsError(f"Reviewer output already exists: {review_path}. Use --overwrite to replace it.")

    env_path = resolve_config_path(root, env_file or Path(".env"))
    load_env_file(env_path)
    client: OpenRouterClient | OpenAIChatClient | FixtureOpenRouterClient | None = None
    if fixture:
        client = FixtureOpenRouterClient(
            responses=[make_fixture_response(fixture_reviewer_json(style), response_id="fixture-reviewer")]
        )
    elif not dry_run:
        client = make_llm_client(provider_config, env_path)

    run_manifest = load_json_if_exists(run_dir / "metadata" / "run_manifest.json")
    run_id = str(run_manifest.get("run_id", run_dir.name))
    messages = build_reviewer_messages(run_dir, style)
    request = make_reviewer_request(
        model=selected_model,
        messages=messages,
        model_config=model_config,
        run_id=run_id,
        reviewer_style=style,
    )
    step = run_chat_step(
        run_dir=run_dir,
        request_payload=request,
        artifact_stem=artifact_stem,
        client=client,
        provider_config=provider_config,
        dry_run=dry_run,
        skipped_message="No reviewer request sent.",
    )
    status = "dry_run" if dry_run else "completed"
    if step.error:
        status = "failed"
    review_payload = parse_reviewer_response(step.text, style)
    write_json(review_path, review_payload)
    write_json(
        manifest_path,
        {
            "schema_version": "0.1",
            "method": "llm_statistical_reviewer_v1",
            "reviewer_style": style,
            "model": selected_model,
            "status": status,
            "dry_run": dry_run,
            "request_path": relative_or_empty(step.request_path, run_dir),
            "response_path": relative_or_empty(step.response_path, run_dir),
            "review_path": relative_or_empty(review_path, run_dir),
            "input_policy": "Reviewer request contains the review manuscript and statistical review schema.",
        },
    )
    return ReviewerHarnessReport(
        root=root,
        run_dir=run_dir,
        model=selected_model,
        reviewer_style=style,
        dry_run=dry_run,
        status=status,
        request_path=step.request_path,
        response_path=step.response_path,
        review_path=review_path,
        manifest_path=manifest_path,
        major_point_count=len(review_payload.get("major_points", [])),
    )


def run_selection_review_for_runs(
    root: Path,
    run_root: Path,
    config_path: Path,
    output_dir: Path,
    model: str | None = None,
    env_file: Path | None = None,
    dry_run: bool = False,
    fixture: bool = False,
    max_papers: int = 20,
    group_size: int | None = None,
    shuffle_seed: int | None = None,
    overwrite: bool = False,
) -> SelectionReviewHarnessReport:
    """Run an overall selection reviewer over a batch of paper titles and abstracts."""

    root = root.resolve()
    run_dirs = collect_completed_author_run_dirs(root, run_root)
    if max_papers < 1:
        raise ValueError("max_papers must be at least 1")
    if group_size is not None and group_size < 1:
        raise ValueError("group_size must be at least 1 when provided")
    if not run_dirs:
        raise FileNotFoundError(f"No completed author runs found under {resolve_config_path(root, run_root)}")
    selected_run_dirs = run_dirs[:]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(selected_run_dirs)
    selected_run_dirs = selected_run_dirs[:max_papers]
    selected_group_size = group_size or len(selected_run_dirs)
    grouped_run_dirs = chunk_run_dirs(selected_run_dirs, selected_group_size)
    config = load_toml(resolve_config_path(root, config_path))
    provider_config = config.get("provider", {})
    model_config = config.get("model", {})
    selected_model = model or str(model_config.get("default", "gpt-5.5"))
    resolved_output_dir = resolve_config_path(root, output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = f"selection_review_{utc_timestamp_slug()}"
    single_group = len(grouped_run_dirs) == 1
    if single_group:
        manifest_path = resolved_output_dir / f"{artifact_stem}_manifest.json"
        aggregate_review_path = resolved_output_dir / f"{artifact_stem}.json"
    else:
        manifest_path = resolved_output_dir / f"{artifact_stem}_batch_manifest.json"
        aggregate_review_path = resolved_output_dir / f"{artifact_stem}_batch.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Selection-review artifact already exists: {manifest_path}")
    if aggregate_review_path.exists() and not overwrite:
        raise FileExistsError(f"Selection-review artifact already exists: {aggregate_review_path}")

    env_path = resolve_config_path(root, env_file or Path(".env"))
    load_env_file(env_path)
    client: OpenRouterClient | OpenAIChatClient | FixtureOpenRouterClient | None = None
    if fixture:
        client = FixtureOpenRouterClient(
            responses=[make_fixture_response(fixture_selection_review_json(), response_id="fixture-selection-review")]
        )
    elif not dry_run:
        client = make_llm_client(provider_config, env_path)

    group_records: list[dict[str, Any]] = []
    overall_status = "dry_run" if dry_run else "completed"
    first_request_path: Path | None = None
    first_response_path: Path | None = None
    for group_index, group_run_dirs in enumerate(grouped_run_dirs, start=1):
        group_id = f"group_{group_index:03d}"
        group_stem = artifact_stem if single_group else f"{artifact_stem}_{group_id}"
        request_path = resolved_output_dir / f"{group_stem}_request.json"
        response_path = resolved_output_dir / f"{group_stem}_response.json"
        message_path = resolved_output_dir / f"{group_stem}_message.md"
        review_path = resolved_output_dir / f"{group_stem}.json"
        group_manifest_path = resolved_output_dir / f"{group_stem}_manifest.json"
        for path in [request_path, response_path, message_path, review_path, group_manifest_path]:
            if path.exists() and not overwrite:
                raise FileExistsError(f"Selection-review artifact already exists: {path}")
        if first_request_path is None:
            first_request_path = request_path
        if first_response_path is None:
            first_response_path = response_path

        paper_items = build_selection_review_items(root, group_run_dirs)
        request = make_selection_review_request(
            model=selected_model,
            messages=build_selection_review_messages(paper_items),
            model_config=model_config,
            review_id=group_stem,
        )
        write_json(request_path, request)
        group_status = "dry_run" if dry_run else "completed"
        response: dict[str, Any] = {"dry_run": True, "message": "No selection-review request sent."}
        text = ""
        if not dry_run:
            if client is None:
                raise RuntimeError("Client is required for non-dry selection review.")
            try:
                response = client.chat_completion(
                    request,
                    endpoint=str(provider_config.get("chat_endpoint", "/chat/completions")),
                )
                text = extract_assistant_text(response)
            except Exception as exc:  # noqa: BLE001 - persisted artifact should capture provider failures
                group_status = "failed"
                overall_status = "failed"
                response = {"error": str(exc)}
                text = ""

        write_json(response_path, response)
        message_path.write_text(text, encoding="utf-8")
        review_payload = parse_selection_review_response(text)
        write_json(review_path, review_payload)
        group_record = {
            "group_id": group_id,
            "status": group_status,
            "paper_count": len(paper_items),
            "request_path": str(request_path.relative_to(root)).replace("\\", "/"),
            "response_path": str(response_path.relative_to(root)).replace("\\", "/"),
            "message_path": str(message_path.relative_to(root)).replace("\\", "/"),
            "review_path": str(review_path.relative_to(root)).replace("\\", "/"),
            "manifest_path": str(group_manifest_path.relative_to(root)).replace("\\", "/"),
            "papers": paper_items,
        }
        group_records.append(group_record)
        write_json(
            group_manifest_path,
            {
                "schema_version": "0.1",
                "method": "llm_selection_reviewer_title_abstract_rank_v1",
                "model": selected_model,
                "status": group_status,
                "dry_run": dry_run,
                "run_root": _relative_to_root(resolve_config_path(root, run_root), root),
                "group_id": group_id,
                "group_index": group_index,
                "group_count": len(grouped_run_dirs),
                "max_papers": max_papers,
                "group_size": selected_group_size,
                "shuffle_seed": shuffle_seed,
                "paper_count": len(paper_items),
                "request_path": str(request_path.relative_to(root)).replace("\\", "/"),
                "response_path": str(response_path.relative_to(root)).replace("\\", "/"),
                "message_path": str(message_path.relative_to(root)).replace("\\", "/"),
                "review_path": str(review_path.relative_to(root)).replace("\\", "/"),
                "papers": paper_items,
            },
        )
    if single_group:
        review_path = Path(group_records[0]["review_path"])
        if not aggregate_review_path.exists():
            source = root / review_path
            if source != aggregate_review_path:
                shutil.copy2(source, aggregate_review_path)
    else:
        write_json(
            aggregate_review_path,
            {
                "schema_version": "0.1",
                "parse_status": "grouped",
                "groups": [
                    {
                        "group_id": record["group_id"],
                        "paper_count": record["paper_count"],
                        "review_path": record["review_path"],
                    }
                    for record in group_records
                ],
            },
        )
    write_json(
        manifest_path,
        {
            "schema_version": "0.1",
            "method": "llm_selection_reviewer_title_abstract_rank_v1",
            "model": selected_model,
            "status": overall_status,
            "dry_run": dry_run,
            "run_root": _relative_to_root(resolve_config_path(root, run_root), root),
            "max_papers": max_papers,
            "group_size": selected_group_size,
            "group_count": len(grouped_run_dirs),
            "shuffle_seed": shuffle_seed,
            "paper_count": len(selected_run_dirs),
            "review_path": str(aggregate_review_path.relative_to(root)).replace("\\", "/"),
            "groups": group_records,
        },
    )
    if first_request_path is None or first_response_path is None:
        raise RuntimeError("No selection-review groups were created.")
    return SelectionReviewHarnessReport(
        root=root,
        run_root=resolve_config_path(root, run_root),
        model=selected_model,
        dry_run=dry_run,
        status=overall_status,
        request_path=first_request_path,
        response_path=first_response_path,
        review_path=aggregate_review_path,
        manifest_path=manifest_path,
        paper_count=len(selected_run_dirs),
        group_count=len(grouped_run_dirs),
    )


def build_selection_review_items(root: Path, run_dirs: list[Path]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for index, run_dir in enumerate(run_dirs, start=1):
        paper_path = run_dir / "paper" / "mini_paper.md"
        manuscript = paper_path.read_text(encoding="utf-8", errors="replace") if paper_path.exists() else ""
        title, abstract = extract_title_and_abstract(manuscript)
        items.append(
            {
                "paper_id": f"paper_{index:03d}",
                "run_dir": _relative_to_root(run_dir, root),
                "title": title,
                "abstract": abstract,
            }
        )
    return items


def chunk_run_dirs(run_dirs: list[Path], group_size: int) -> list[list[Path]]:
    return [run_dirs[start : start + group_size] for start in range(0, len(run_dirs), group_size)]


def extract_title_and_abstract(markdown_text: str) -> tuple[str, str]:
    title = ""
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped.removeprefix("# ").strip()
            break
    abstract = extract_markdown_section(markdown_text, "abstract")
    if not abstract:
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", markdown_text)
            if paragraph.strip() and not paragraph.strip().startswith("#")
        ]
        abstract = paragraphs[0] if paragraphs else ""
    return truncate_text(one_line(title), 220), truncate_text(one_line(abstract), 1800)


def extract_markdown_section(markdown_text: str, section_name: str) -> str:
    heading_pattern = re.compile(rf"^\s*#+\s*{re.escape(section_name)}\s*$", re.IGNORECASE)
    lines = markdown_text.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        if heading_pattern.match(line):
            start = index + 1
            break
    if start is None:
        plain_pattern = re.compile(rf"^\s*{re.escape(section_name)}\s*:?\s*$", re.IGNORECASE)
        for index, line in enumerate(lines):
            if plain_pattern.match(line):
                start = index + 1
                break
    if start is None:
        return ""
    end = len(lines)
    for index in range(start, len(lines)):
        if re.match(r"^\s*#+\s+\S+", lines[index]):
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def build_selection_review_messages(papers: list[dict[str, str]]) -> list[dict[str, str]]:
    system = "You are an overall reviewer ranking empirical mini-papers for follow-up review."
    user = "\n\n".join(
        [
            "# Overall Selection Task",
            "Rank the papers from most promising to least promising for follow-up scientific review.",
            "You are given only each paper's title and abstract.",
            "Return JSON only, following the output schema.",
            "# Output Schema",
            json.dumps(
                {
                    "ranked_papers": [
                        {
                            "rank": "integer rank starting at 1",
                            "paper_id": "paper identifier from the input",
                            "title": "paper title from the input",
                            "selection_reason": "one concise reason for this rank",
                        }
                    ]
                },
                indent=2,
            ),
            "# Papers",
            json.dumps(
                [
                    {
                        "paper_id": item["paper_id"],
                        "title": item["title"],
                        "abstract": item["abstract"],
                    }
                    for item in papers
                ],
                indent=2,
                ensure_ascii=False,
            ),
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def make_selection_review_request(
    model: str,
    messages: list[dict[str, str]],
    model_config: dict[str, Any],
    review_id: str,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": strict_json_schema_response_format(
            "selection_review",
            SELECTION_REVIEW_JSON_SCHEMA,
        ),
    }
    if should_send_temperature(model, model_config):
        request["temperature"] = float(model_config.get("temperature", 0.2))
    for field in ["reasoning_effort", "prompt_cache_key", "prompt_cache_retention"]:
        value = model_config.get(field)
        if value not in {None, ""}:
            request[field] = value
    if model_config.get("include_metadata", False):
        request["metadata"] = {"project": "selection-review", "review_id": review_id}
    return request


def parse_selection_review_response(text: str) -> dict[str, Any]:
    decoded: Any = None
    parse_status = "empty_response"
    if text.strip():
        try:
            decoded = json.loads(text)
            parse_status = "json"
        except json.JSONDecodeError:
            fenced = extract_fenced(text, "json")
            if fenced:
                try:
                    decoded = json.loads(fenced)
                    parse_status = "fenced_json"
                except json.JSONDecodeError:
                    decoded = None
                    parse_status = "unparseable"
            else:
                parse_status = "unparseable"
    if not isinstance(decoded, dict):
        decoded = {}
    ranked = decoded.get("ranked_papers", [])
    if not isinstance(ranked, list):
        ranked = []
    normalized_ranked = []
    for item in ranked:
        if not isinstance(item, dict):
            continue
        try:
            rank = int(item.get("rank"))
        except (TypeError, ValueError):
            rank = len(normalized_ranked) + 1
        normalized_ranked.append(
            {
                "rank": rank,
                "paper_id": str(item.get("paper_id", "")),
                "title": str(item.get("title", "")),
                "selection_reason": str(item.get("selection_reason", "")),
            }
        )
    normalized_ranked.sort(key=lambda item: item["rank"])
    return {
        "schema_version": "0.1",
        "parse_status": parse_status,
        "ranked_papers": normalized_ranked,
    }


def collect_completed_author_run_dirs(root: Path, run_root: Path) -> list[Path]:
    root = root.resolve()
    resolved_run_root = resolve_config_path(root, run_root)
    run_dirs = sorted(path.parent.parent for path in resolved_run_root.glob("**/paper/mini_paper.md"))
    completed: list[Path] = []
    for path in run_dirs:
        manifest_path = path / "metadata" / "run_manifest.json"
        paper_path = path / "paper" / "mini_paper.md"
        if not manifest_path.exists() or not paper_path.exists():
            continue
        manifest = load_json_if_exists(manifest_path)
        status = str(manifest.get("status") or manifest.get("run_status") or "").strip().lower()
        if status and status not in {"complete", "completed", "success", "succeeded"}:
            continue
        if not paper_path.read_text(encoding="utf-8", errors="replace").strip():
            continue
        completed.append(path)
    return completed


def create_openai_batch_for_runs(
    root: Path,
    run_root: Path,
    config_path: Path,
    task: str,
    output_dir: Path,
    env_file: Path | None = None,
    reviewer_style: str = "statistical",
    model: str | None = None,
    overwrite: bool = False,
) -> OpenAIBatchReport:
    root = root.resolve()
    selected_task = normalize_batch_task(task)
    run_dirs = collect_completed_author_run_dirs(root, run_root)
    if not run_dirs:
        raise FileNotFoundError(f"No completed author runs found under {resolve_config_path(root, run_root)}")
    config = load_toml(resolve_config_path(root, config_path))
    provider_config = config.get("provider", {})
    model_config = config.get("model", {})
    provider_name = str(provider_config.get("name", "")).lower()
    if provider_name != "openai" and "api.openai.com" not in str(provider_config.get("base_url", "")).lower():
        raise ValueError("OpenAI batch creation requires an OpenAI provider config.")
    selected_model = model or str(model_config.get("default", "gpt-5.5"))
    env_path = resolve_config_path(root, env_file or Path(".env"))
    load_env_file(env_path)
    client = make_llm_client(provider_config, env_path)
    if not isinstance(client, OpenAIChatClient):
        raise ValueError("OpenAI batch creation requires an OpenAI client.")

    resolved_output_dir = resolve_config_path(root, output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    batch_stem = f"{selected_task}_{utc_timestamp_slug()}"
    request_jsonl_path = resolved_output_dir / f"{batch_stem}.jsonl"
    manifest_path = resolved_output_dir / f"{batch_stem}_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Batch manifest already exists: {manifest_path}.")

    items: list[dict[str, Any]] = []
    with request_jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, run_dir in enumerate(run_dirs, start=1):
            request_payload = build_batch_request_payload(
                run_dir=run_dir,
                task=selected_task,
                model=selected_model,
                model_config=model_config,
                reviewer_style=reviewer_style,
            )
            custom_id = batch_custom_id(selected_task, index, root, run_dir)
            handle.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": request_payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            request_artifact = batch_request_artifact_path(run_dir, selected_task, reviewer_style)
            write_json(request_artifact, request_payload)
            items.append(
                {
                    "custom_id": custom_id,
                    "run_dir": _relative_to_root(run_dir, root),
                    "request_path": str(request_artifact.relative_to(run_dir)).replace("\\", "/"),
                }
            )

    uploaded = client.upload_batch_file(request_jsonl_path)
    batch = client.create_batch(
        input_file_id=str(uploaded["id"]),
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"project": "trap-ai", "task": selected_task},
    )
    write_json(
        manifest_path,
        {
            "schema_version": "0.1",
            "method": "openai_batch_chat_completions",
            "task": selected_task,
            "reviewer_style": reviewer_style if selected_task == "reviewer" else "",
            "model": selected_model,
            "status": batch.get("status", ""),
            "batch_id": batch.get("id", ""),
            "input_file_id": uploaded.get("id", ""),
            "request_jsonl_path": str(request_jsonl_path.relative_to(root)).replace("\\", "/"),
            "created_at": utc_now(),
            "items": items,
            "openai_batch": batch,
        },
    )
    return OpenAIBatchReport(
        root=root,
        task=selected_task,
        batch_id=str(batch.get("id", "")),
        status=str(batch.get("status", "")),
        request_count=len(items),
        manifest_path=manifest_path,
        request_jsonl_path=request_jsonl_path,
    )


def retrieve_openai_batch_results(
    root: Path,
    manifest_path: Path,
    env_file: Path | None = None,
    wait: bool = False,
    poll_seconds: float = 30.0,
    max_wait_seconds: float = 3600.0,
    overwrite: bool = True,
) -> OpenAIBatchReport:
    root = root.resolve()
    manifest_path = resolve_config_path(root, manifest_path)
    manifest = load_json_if_exists(manifest_path)
    task = normalize_batch_task(str(manifest.get("task", "")))
    config_path = Path("configs/evaluators/openai_claim_extractor.toml")
    if task == "reviewer":
        config_path = Path("configs/reviewers/openai_reviewer.toml")
    config = load_toml(resolve_config_path(root, config_path))
    provider_config = config.get("provider", {})
    env_path = resolve_config_path(root, env_file or Path(".env"))
    load_env_file(env_path)
    client = make_llm_client(provider_config, env_path)
    if not isinstance(client, OpenAIChatClient):
        raise ValueError("OpenAI batch retrieval requires an OpenAI client.")
    batch_id = str(manifest.get("batch_id", ""))
    if not batch_id:
        raise ValueError(f"Batch manifest has no batch_id: {manifest_path}")

    started = time.monotonic()
    batch = client.retrieve_batch(batch_id)
    while wait and str(batch.get("status", "")) not in {"completed", "failed", "expired", "cancelled"}:
        if time.monotonic() - started > max_wait_seconds:
            break
        time.sleep(poll_seconds)
        batch = client.retrieve_batch(batch_id)

    output_jsonl_path: Path | None = None
    completed_count = 0
    failed_count = 0
    if str(batch.get("status", "")) == "completed" and batch.get("output_file_id"):
        output_jsonl_path = manifest_path.with_name(f"{manifest_path.stem}_output.jsonl")
        output_text = client.retrieve_file_content(str(batch["output_file_id"]))
        output_jsonl_path.write_text(output_text, encoding="utf-8")
        completed_count, failed_count = apply_openai_batch_output(
            root=root,
            manifest=manifest,
            output_text=output_text,
            overwrite=overwrite,
        )

    manifest["status"] = batch.get("status", "")
    manifest["retrieved_at"] = utc_now()
    manifest["openai_batch_latest"] = batch
    if output_jsonl_path is not None:
        manifest["output_jsonl_path"] = str(output_jsonl_path.relative_to(root)).replace("\\", "/")
    manifest["completed_result_count"] = completed_count
    manifest["failed_result_count"] = failed_count
    write_json(manifest_path, manifest)
    return OpenAIBatchReport(
        root=root,
        task=task,
        batch_id=batch_id,
        status=str(batch.get("status", "")),
        request_count=len(manifest.get("items", [])),
        manifest_path=manifest_path,
        request_jsonl_path=resolve_config_path(root, Path(str(manifest["request_jsonl_path"]))),
        output_jsonl_path=output_jsonl_path,
        completed_count=completed_count,
        failed_count=failed_count,
    )


def apply_openai_batch_output(
    root: Path,
    manifest: dict[str, Any],
    output_text: str,
    overwrite: bool,
) -> tuple[int, int]:
    items = {str(item["custom_id"]): item for item in manifest.get("items", []) if isinstance(item, dict)}
    task = normalize_batch_task(str(manifest.get("task", "")))
    reviewer_style = str(manifest.get("reviewer_style", "statistical")) or "statistical"
    completed = 0
    failed = 0
    for line in output_text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        custom_id = str(record.get("custom_id", ""))
        item = items.get(custom_id)
        if not item:
            failed += 1
            continue
        run_dir = resolve_config_path(root, Path(str(item["run_dir"])))
        response_body = record.get("response", {}).get("body")
        status_code = record.get("response", {}).get("status_code")
        if status_code != 200 or not isinstance(response_body, dict):
            write_json(run_dir / "metadata" / f"{task}_batch_error_{custom_id}.json", record)
            failed += 1
            continue
        if task == "claim_extraction":
            response_path = run_dir / "response" / "claim_extraction_response.json"
            if response_path.exists() and not overwrite:
                raise FileExistsError(f"Response exists: {response_path}")
            write_json(response_path, response_body)
            text = extract_assistant_text(response_body)
            (run_dir / "response" / "claim_extraction_message.md").write_text(text, encoding="utf-8")
            write_claim_table_from_extractor_response(
                run_dir,
                text,
                run_dir / "claim_tables" / "claims.csv",
                response_path,
            )
        elif task == "reviewer":
            artifact_stem = f"reviewer_{normalize_reviewer_style(reviewer_style)}"
            response_path = run_dir / "response" / f"{artifact_stem}_response.json"
            review_path = run_dir / "reviews" / f"{normalize_reviewer_style(reviewer_style)}_review.json"
            if review_path.exists() and not overwrite:
                raise FileExistsError(f"Review exists: {review_path}")
            write_json(response_path, response_body)
            text = extract_assistant_text(response_body)
            (run_dir / "response" / f"{artifact_stem}_message.md").write_text(text, encoding="utf-8")
            review_payload = parse_reviewer_response(text, reviewer_style)
            write_json(review_path, review_payload)
            write_json(
                run_dir / "metadata" / f"{artifact_stem}_manifest.json",
                {
                    "schema_version": "0.1",
                    "method": "openai_batch_statistical_reviewer_v1",
                    "reviewer_style": normalize_reviewer_style(reviewer_style),
                    "model": manifest.get("model", ""),
                    "status": "completed",
                    "dry_run": False,
                    "request_path": item.get("request_path", ""),
                    "response_path": relative_or_empty(response_path, run_dir),
                    "review_path": relative_or_empty(review_path, run_dir),
                    "input_policy": "Reviewer request contains the review manuscript and statistical review schema.",
                },
            )
        completed += 1
    return completed, failed


def build_batch_request_payload(
    run_dir: Path,
    task: str,
    model: str,
    model_config: dict[str, Any],
    reviewer_style: str,
) -> dict[str, Any]:
    run_manifest = load_json_if_exists(run_dir / "metadata" / "run_manifest.json")
    run_id = str(run_manifest.get("run_id", run_dir.name))
    if task == "claim_extraction":
        return make_claim_extraction_request(
            model=model,
            messages=build_claim_extraction_messages(run_dir),
            model_config=model_config,
            run_id=run_id,
        )
    if task == "reviewer":
        style = normalize_reviewer_style(reviewer_style)
        return make_reviewer_request(
            model=model,
            messages=build_reviewer_messages(run_dir, style),
            model_config=model_config,
            run_id=run_id,
            reviewer_style=style,
        )
    raise ValueError(f"Unsupported batch task: {task}")


def batch_request_artifact_path(run_dir: Path, task: str, reviewer_style: str) -> Path:
    if task == "claim_extraction":
        return run_dir / "request" / "claim_extraction_request.json"
    return run_dir / "request" / f"reviewer_{normalize_reviewer_style(reviewer_style)}_request.json"


def normalize_batch_task(task: str) -> str:
    normalized = task.strip().lower().replace("-", "_")
    aliases = {
        "extract_claims": "claim_extraction",
        "claims": "claim_extraction",
        "claim_extractor": "claim_extraction",
        "review": "reviewer",
        "review_paper": "reviewer",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"claim_extraction", "reviewer"}:
        raise ValueError("Batch task must be claim_extraction or reviewer.")
    return normalized


def batch_custom_id(task: str, index: int, root: Path, run_dir: Path) -> str:
    try:
        relative = str(run_dir.relative_to(root)).replace("\\", "/")
    except ValueError:
        relative = str(run_dir)
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
    return f"{task}-{index:04d}-{digest}"


def utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def resume_interactive_agent_run(
    root: Path,
    run_dir: Path,
    config_path: Path,
    model: str | None = None,
    env_file: Path | None = None,
    dry_run: bool = False,
    max_code_repairs: int | None = None,
    max_supplemental_analyses: int | None = None,
    max_sufficiency_reviews: int | None = None,
) -> AgentRunResult:
    """Resume a partially completed author run from saved artifacts."""

    root = root.resolve()
    run_dir = resolve_config_path(root, run_dir)
    ensure_run_tree(run_dir)
    config = load_toml(resolve_config_path(root, config_path))
    provider_config = config.get("provider", {})
    model_config = config.get("model", {})
    run_config = config.get("run", {})
    existing_manifest = load_json_if_exists(run_dir / "metadata" / "run_manifest.json")
    request_model = model or str(existing_manifest.get("model", "")) or infer_model_from_requests(run_dir)
    selected_model = request_model or str(model_config.get("default", "deepseek/deepseek-v4-flash"))
    selected_author_paradigm = normalize_author_paradigm(
        str(existing_manifest.get("author_paradigm", "")) or infer_author_paradigm_from_run_dir(run_dir)
    )
    selected_max_code_repairs = int(
        run_config.get("max_code_repairs", DEFAULT_MAX_CODE_REPAIRS)
        if max_code_repairs is None
        else max_code_repairs
    )
    selected_max_supplemental_analyses = int(
        run_config.get("max_supplemental_analyses", DEFAULT_MAX_SUPPLEMENTAL_ANALYSES)
        if max_supplemental_analyses is None
        else max_supplemental_analyses
    )
    selected_max_sufficiency_reviews = int(
        run_config.get("max_sufficiency_reviews", DEFAULT_MAX_SUFFICIENCY_REVIEWS)
        if max_sufficiency_reviews is None
        else max_sufficiency_reviews
    )
    effective_max_code_repairs = selected_max_code_repairs
    effective_max_supplemental_analyses = (
        0 if selected_author_paradigm == "single_shot" else selected_max_supplemental_analyses
    )
    if selected_max_sufficiency_reviews < 0:
        raise ValueError("max_sufficiency_reviews must be non-negative")
    effective_sufficiency_review = (
        selected_author_paradigm != "single_shot" and selected_max_sufficiency_reviews > 0
    )
    effective_max_sufficiency_reviews = (
        0 if not effective_sufficiency_review else selected_max_sufficiency_reviews
    )

    env_path = resolve_config_path(root, env_file or Path(".env"))
    load_env_file(env_path)
    client: OpenRouterClient | None = None
    if not dry_run:
        api_key_env = str(provider_config.get("api_key_env", "OPENROUTER_API_KEY"))
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Missing OpenRouter API key. Set {api_key_env} in {env_path} or the process environment."
            )
        client = OpenRouterClient(
            api_key=api_key,
            base_url=str(provider_config.get("base_url", "https://openrouter.ai/api/v1")),
            timeout_seconds=float(provider_config.get("timeout_seconds", 180)),
            http_referer=os.environ.get(str(provider_config.get("http_referer_env", "OPENROUTER_HTTP_REFERER")), ""),
            app_title=os.environ.get(str(provider_config.get("app_title_env", "OPENROUTER_APP_TITLE")), "trap-ai"),
        )

    prompt_payload = load_prompt_payload_from_run(run_dir)
    run_id = str(existing_manifest.get("run_id", run_dir.name))
    run_seed = int(existing_manifest.get("seed", DEFAULT_SEED))
    dataset_id = str(existing_manifest.get("dataset_id", ""))
    if not dataset_id:
        dataset_id = infer_dataset_id_from_run_dir(run_dir)
    dataset_dir = Path(str(existing_manifest.get("dataset_dir", run_dir / "input")))
    started = utc_now()
    status = "dry_run" if dry_run else "completed"
    stage_context = load_stage_context_from_run(run_dir)
    interaction_steps: list[dict[str, Any]] = existing_manifest.get("interaction_steps", [])
    if not isinstance(interaction_steps, list):
        interaction_steps = []
    step_index = next_loop_step_index(run_dir, interaction_steps)
    repair_attempts = count_existing_repair_attempts(run_dir)
    supplemental_attempts = count_existing_supplemental_attempts(run_dir)
    sufficiency_round = count_existing_sufficiency_rounds(run_dir)

    code_path = choose_resume_code_path(run_dir)
    analysis_code = code_path.read_text(encoding="utf-8", errors="replace") if code_path.exists() else ""
    execution = load_execution_result_from_run(run_dir)
    if not execution.generated_files:
        execution = aggregate_analysis_execution_result(run_dir, execution)

    def run_resume_repair_loop(
        current_code: str,
        current_execution: ExecutionResult,
        target_code_path: Path,
        artifact_prefix: str,
        execution_prefix: str,
    ) -> tuple[str, ExecutionResult]:
        nonlocal repair_attempts, status, step_index
        repaired_code = current_code
        repaired_execution = current_execution
        while (
            not dry_run
            and status != "failed"
            and repaired_execution.returncode != 0
            and repair_attempts < effective_max_code_repairs
        ):
            repair_attempts += 1
            repair_messages = build_code_repair_messages(
                prompt_payload=prompt_payload,
                analysis_code=repaired_code,
                execution=repaired_execution,
                run_dir=run_dir,
                attempt=repair_attempts,
                max_attempts=effective_max_code_repairs,
            )
            artifact_stem = f"{artifact_prefix}_repair_{repair_attempts:02d}"
            repair_request = make_chat_request(
                model=selected_model,
                messages=repair_messages,
                model_config=model_config,
                run_id=run_id,
                run_seed=run_seed,
                loop_step=artifact_stem,
                loop_step_index=step_index,
                dataset_id=dataset_id,
                author_paradigm=selected_author_paradigm,
            )
            repair_step = run_chat_step(
                run_dir=run_dir,
                request_payload=repair_request,
                artifact_stem=artifact_stem,
                client=client,
                provider_config=provider_config,
                dry_run=dry_run,
                skipped_message="No repair request sent.",
            )
            if repair_step.error:
                status = "failed"
            interaction_steps.append(step_manifest("code_repair", repair_step, step_index, attempt=repair_attempts))
            step_index += 1
            if status == "failed":
                break
            candidate_code = parse_code_response(repair_step.text)
            if not candidate_code.strip():
                repaired_execution = empty_execution(run_dir)
                persist_execution_attempt(run_dir, repaired_execution, f"{execution_prefix}_repair_{repair_attempts:02d}")
                continue
            copy_path = run_dir / "code" / f"{artifact_stem}.py"
            copy_path.write_text(candidate_code, encoding="utf-8")
            target_code_path.write_text(candidate_code, encoding="utf-8")
            repaired_code = candidate_code
            repaired_execution = execute_generated_code(run_dir, target_code_path)
            persist_execution_attempt(run_dir, repaired_execution, f"{execution_prefix}_repair_{repair_attempts:02d}")
        return repaired_code, repaired_execution

    if analysis_code and execution.returncode != 0:
        execution_prefix = "execution_resume"
        if code_path.name.startswith("analysis_supplement_"):
            execution_prefix = f"execution_{code_path.stem}_resume"
        analysis_code, execution = run_resume_repair_loop(
            current_code=analysis_code,
            current_execution=execution,
            target_code_path=code_path,
            artifact_prefix=f"{code_path.stem}_resume",
            execution_prefix=execution_prefix,
        )

    execution = aggregate_analysis_execution_result(run_dir, execution)
    write_aggregate_execution_manifest(run_dir, execution)

    if (
        status != "failed"
        and selected_author_paradigm != "single_shot"
        and not has_stage_artifact(run_dir, SUFFICIENCY_STAGE)
        and execution.generated_files
        and execution.returncode == 0
        and sufficiency_round < effective_max_sufficiency_reviews
    ):
        sufficiency_round += 1
        review_messages = build_analysis_sufficiency_review_messages(
            prompt_payload=prompt_payload,
            execution=execution,
            run_dir=run_dir,
            stage_context=stage_context,
            author_paradigm=selected_author_paradigm,
        )
        review_step_name = SUFFICIENCY_STAGE if sufficiency_round == 1 else f"{SUFFICIENCY_STAGE}_{sufficiency_round:02d}"
        review_request = make_chat_request(
            model=selected_model,
            messages=review_messages,
            model_config=model_config,
            run_id=run_id,
            run_seed=run_seed,
            loop_step=review_step_name,
            loop_step_index=step_index,
            dataset_id=dataset_id,
            author_paradigm=selected_author_paradigm,
        )
        review_step = run_chat_step(
            run_dir=run_dir,
            request_payload=review_request,
            artifact_stem=review_step_name,
            client=client,
            provider_config=provider_config,
            dry_run=dry_run,
            skipped_message="No analysis sufficiency review request sent.",
        )
        if review_step.error:
            status = "failed"
        interaction_steps.append(step_manifest(SUFFICIENCY_STAGE, review_step, step_index))
        step_index += 1
        save_analysis_sufficiency_review(run_dir, review_step.text, review_step_name)
        stage_context[review_step_name] = review_step.text

    if status not in {"failed", "dry_run"} and (not execution.generated_files or execution.returncode != 0):
        status = "completed_execution_failed"

    if status not in {"failed", "completed_execution_failed", "dry_run"}:
        for stage_name in post_execution_stages(selected_author_paradigm, effective_sufficiency_review):
            if has_stage_artifact(run_dir, stage_name):
                continue
            stage_step = run_author_stage(
                run_dir=run_dir,
                prompt_payload=prompt_payload,
                stage_name=stage_name,
                stage_context=stage_context,
                model=selected_model,
                provider_config=provider_config,
                model_config=model_config,
                run_id=run_id,
                run_seed=run_seed,
                step_index=step_index,
                dataset_id=dataset_id,
                client=client,
                dry_run=dry_run,
                author_paradigm=selected_author_paradigm,
                execution=execution,
            )
            if stage_step.error:
                status = "failed"
            interaction_steps.append(step_manifest(stage_name, stage_step, step_index))
            stage_context[stage_name] = stage_step.text
            save_stage_artifact(run_dir, stage_name, stage_step.text)
            step_index += 1
            if status == "failed":
                break

    if selected_author_paradigm == "data_to_paper" and status not in {"failed", "completed_execution_failed"}:
        write_auto_traceability_appendix(run_dir, execution)

    report_step: ChatStepResult | None = None
    paper_path = run_dir / "paper" / "mini_paper.md"
    existing_paper = paper_path.read_text(encoding="utf-8", errors="replace") if paper_path.exists() else ""
    if not existing_paper.strip() and status not in {"failed", "completed_execution_failed"}:
        report_messages = build_report_generation_messages(
            prompt_payload=prompt_payload,
            analysis_code=analysis_code,
            execution=execution,
            run_dir=run_dir,
            evidence_audit_text=stage_context.get(SUFFICIENCY_STAGE, ""),
            stage_context=stage_context,
            author_paradigm=selected_author_paradigm,
        )
        report_request = make_chat_request(
            model=selected_model,
            messages=report_messages,
            model_config=model_config,
            run_id=run_id,
            run_seed=run_seed,
            loop_step="report_generation",
            loop_step_index=step_index,
            dataset_id=dataset_id,
            author_paradigm=selected_author_paradigm,
            json_response=False,
        )
        report_step = run_chat_step(
            run_dir=run_dir,
            request_payload=report_request,
            artifact_stem="report",
            client=client,
            provider_config=provider_config,
            dry_run=dry_run,
            skipped_message="No report request sent.",
        )
        if report_step.error:
            status = "failed"
        interaction_steps.append(step_manifest("report_generation", report_step, step_index))
        paper_path.write_text(parse_report_response(report_step.text), encoding="utf-8")

    final_report_path = run_dir / "paper" / "mini_paper.md"
    claim_table_path = run_dir / "claim_tables" / "claims.csv"
    if final_report_path.exists() and final_report_path.read_text(encoding="utf-8", errors="replace").strip():
        write_review_manuscript(
            run_dir,
            final_report_path.read_text(encoding="utf-8", errors="replace"),
            execution,
        )
    write_interactive_run_manifest(
        run_dir=run_dir,
        run_id=run_id,
        run_index=int(existing_manifest.get("run_index", 1)),
        run_seed=run_seed,
        dataset_id=dataset_id,
        dataset_dir=str(dataset_dir),
        model=selected_model,
        author_paradigm=selected_author_paradigm,
        dry_run=dry_run,
        status=status,
        started=started,
        execution=execution,
        interaction_steps=interaction_steps,
        effective_max_code_repairs=effective_max_code_repairs,
        configured_max_code_repairs=selected_max_code_repairs,
        effective_sufficiency_review=effective_sufficiency_review,
        configured_sufficiency_review=True,
        effective_max_sufficiency_reviews=effective_max_sufficiency_reviews,
        configured_max_sufficiency_reviews=selected_max_sufficiency_reviews,
        sufficiency_review_rounds=sufficiency_round,
        effective_max_supplemental_analyses=effective_max_supplemental_analyses,
        configured_max_supplemental_analyses=selected_max_supplemental_analyses,
        repair_attempts=repair_attempts,
        supplemental_attempts=supplemental_attempts,
        code_step_path=run_dir / "request" / "code_request.json",
        report_step_path=run_dir / "request" / "report_request.json",
    )
    return AgentRunResult(
        run_id=run_id,
        run_dir=run_dir,
        status=status,
        request_path=run_dir / "request" / "code_request.json",
        response_path=run_dir / "response" / "report_response.json",
        mini_paper_path=final_report_path,
        claim_table_path=claim_table_path,
        code_path=code_path,
    )


def run_single_interactive_agent(
    run_dir: Path,
    run_id: str,
    run_index: int,
    run_seed: int,
    dataset_dir: Path,
    prompt_payload: dict[str, Any],
    model: str,
    provider_config: dict[str, Any],
    model_config: dict[str, Any],
    client: OpenRouterClient | FixtureOpenRouterClient | None,
    dry_run: bool,
    copy_dataset: bool,
    max_code_repairs: int = DEFAULT_MAX_CODE_REPAIRS,
    evidence_audit: bool = True,
    max_supplemental_analyses: int = DEFAULT_MAX_SUPPLEMENTAL_ANALYSES,
    max_sufficiency_reviews: int = DEFAULT_MAX_SUFFICIENCY_REVIEWS,
    author_paradigm: str = DEFAULT_AUTHOR_PARADIGM,
) -> AgentRunResult:
    save_input_package(run_dir, dataset_dir, prompt_payload, copy_dataset=copy_dataset)
    started = utc_now()
    status = "dry_run" if dry_run else "completed"
    interaction_steps: list[dict[str, Any]] = []
    request_dataset_id = public_dataset_slug(prompt_payload, dataset_dir)
    selected_author_paradigm = normalize_author_paradigm(author_paradigm)
    if max_sufficiency_reviews < 0:
        raise ValueError("max_sufficiency_reviews must be non-negative")
    effective_max_code_repairs = max_code_repairs
    effective_sufficiency_review = (
        evidence_audit and selected_author_paradigm != "single_shot" and max_sufficiency_reviews > 0
    )
    effective_max_supplemental_analyses = (
        0 if selected_author_paradigm == "single_shot" else max_supplemental_analyses
    )
    effective_max_sufficiency_reviews = (
        0 if not effective_sufficiency_review else max_sufficiency_reviews
    )
    stage_context: dict[str, str] = {}

    step_index = 0
    for stage_name in pre_execution_stages(selected_author_paradigm):
        stage_step = run_author_stage(
            run_dir=run_dir,
            prompt_payload=prompt_payload,
            stage_name=stage_name,
            stage_context=stage_context,
            model=model,
            provider_config=provider_config,
            model_config=model_config,
            run_id=run_id,
            run_seed=run_seed,
            step_index=step_index,
            dataset_id=request_dataset_id,
            client=client,
            dry_run=dry_run or status == "failed",
            author_paradigm=selected_author_paradigm,
        )
        if stage_step.error:
            status = "failed"
        interaction_steps.append(step_manifest(stage_name, stage_step, step_index))
        stage_context[stage_name] = stage_step.text
        save_stage_artifact(run_dir, stage_name, stage_step.text)
        step_index += 1

    code_messages = build_code_generation_messages(prompt_payload, selected_author_paradigm, stage_context)
    code_request = make_chat_request(
        model=model,
        messages=code_messages,
        model_config=model_config,
        run_id=run_id,
        run_seed=run_seed,
        loop_step="code_generation",
        loop_step_index=step_index,
        dataset_id=request_dataset_id,
        author_paradigm=selected_author_paradigm,
    )
    code_step = run_chat_step(
        run_dir=run_dir,
        request_payload=code_request,
        artifact_stem="code",
        client=client,
        provider_config=provider_config,
        dry_run=dry_run,
        skipped_message="No code request sent.",
    )
    if code_step.error:
        status = "failed"
    interaction_steps.append(step_manifest("code_generation", code_step, step_index))
    step_index += 1

    code_path = run_dir / "code" / "analysis.py"
    analysis_code = parse_code_response(code_step.text)
    code_path.write_text(analysis_code, encoding="utf-8")
    execution = execute_generated_code(run_dir, code_path) if analysis_code.strip() else empty_execution(run_dir)
    persist_execution_attempt(run_dir, execution, "execution_initial")

    repair_attempts = 0

    def repair_code_until_success(
        current_code: str,
        current_execution: ExecutionResult,
        target_code_path: Path,
        artifact_prefix: str,
        execution_prefix: str,
    ) -> tuple[str, ExecutionResult]:
        nonlocal repair_attempts, status, step_index
        repaired_current_code = current_code
        repaired_execution = current_execution
        while (
            not dry_run
            and status != "failed"
            and repaired_execution.returncode != 0
            and repair_attempts < effective_max_code_repairs
        ):
            repair_attempts += 1
            repair_messages = build_code_repair_messages(
                prompt_payload=prompt_payload,
                analysis_code=repaired_current_code,
                execution=repaired_execution,
                run_dir=run_dir,
                attempt=repair_attempts,
                max_attempts=effective_max_code_repairs,
            )
            artifact_stem = f"{artifact_prefix}_repair_{repair_attempts:02d}"
            repair_request = make_chat_request(
                model=model,
                messages=repair_messages,
                model_config=model_config,
                run_id=run_id,
                run_seed=run_seed,
                loop_step=artifact_stem,
                loop_step_index=step_index,
                dataset_id=request_dataset_id,
                author_paradigm=selected_author_paradigm,
            )
            repair_step = run_chat_step(
                run_dir=run_dir,
                request_payload=repair_request,
                artifact_stem=artifact_stem,
                client=client,
                provider_config=provider_config,
                dry_run=False,
                skipped_message="No repair request sent.",
            )
            if repair_step.error:
                status = "failed"
            interaction_steps.append(step_manifest("code_repair", repair_step, step_index, attempt=repair_attempts))
            step_index += 1
            if status == "failed":
                break

            repaired_code = parse_code_response(repair_step.text)
            if not repaired_code.strip():
                repaired_execution = empty_execution(run_dir)
                persist_execution_attempt(run_dir, repaired_execution, f"{execution_prefix}_repair_{repair_attempts:02d}")
                continue

            repair_code_path = run_dir / "code" / f"{artifact_stem}.py"
            repair_code_path.write_text(repaired_code, encoding="utf-8")
            target_code_path.write_text(repaired_code, encoding="utf-8")
            repaired_current_code = repaired_code
            repaired_execution = execute_generated_code(run_dir, target_code_path)
            persist_execution_attempt(run_dir, repaired_execution, f"{execution_prefix}_repair_{repair_attempts:02d}")
        return repaired_current_code, repaired_execution

    analysis_code, execution = repair_code_until_success(
        current_code=analysis_code,
        current_execution=execution,
        target_code_path=code_path,
        artifact_prefix="code",
        execution_prefix="execution",
    )

    execution = aggregate_analysis_execution_result(run_dir, execution)
    audit_text = ""
    audit_step: ChatStepResult | None = None
    supplemental_attempts = 0
    sufficiency_round = 0

    def run_sufficiency_review_step(trigger_context: str = "") -> ChatStepResult:
        nonlocal audit_step, audit_text, status, step_index, sufficiency_round
        if sufficiency_round >= effective_max_sufficiency_reviews:
            raise RuntimeError("analysis sufficiency review limit reached")
        sufficiency_round += 1
        artifact_stem = (
            SUFFICIENCY_STAGE if sufficiency_round == 1 else f"{SUFFICIENCY_STAGE}_{sufficiency_round:02d}"
        )
        review_messages = build_analysis_sufficiency_review_messages(
            prompt_payload=prompt_payload,
            execution=execution,
            run_dir=run_dir,
            stage_context=stage_context,
            author_paradigm=selected_author_paradigm,
            trigger_context=trigger_context,
        )
        review_request = make_chat_request(
            model=model,
            messages=review_messages,
            model_config=model_config,
            run_id=run_id,
            run_seed=run_seed,
            loop_step=artifact_stem,
            loop_step_index=step_index,
            dataset_id=request_dataset_id,
            author_paradigm=selected_author_paradigm,
        )
        review_step = run_chat_step(
            run_dir=run_dir,
            request_payload=review_request,
            artifact_stem=artifact_stem,
            client=client,
            provider_config=provider_config,
            dry_run=dry_run or status == "failed",
            skipped_message="No analysis sufficiency review request sent.",
        )
        if review_step.error:
            status = "failed"
        audit_step = review_step
        audit_text = review_step.text
        attempt_value = sufficiency_round if sufficiency_round > 1 else None
        interaction_steps.append(step_manifest(SUFFICIENCY_STAGE, review_step, step_index, attempt=attempt_value))
        stage_context[artifact_stem] = audit_text
        save_analysis_sufficiency_review(run_dir, audit_text, artifact_stem)
        step_index += 1
        return review_step

    def run_supplemental_analysis_loop(trigger_context: str, require_followup_review: bool = True) -> None:
        nonlocal analysis_code, execution, status, step_index, supplemental_attempts
        next_trigger = trigger_context
        while (
            not dry_run
            and status != "failed"
            and supplemental_attempts < effective_max_supplemental_analyses
            and stage_requests_supplemental_analysis(next_trigger)
        ):
            supplemental_attempts += 1
            attempt = supplemental_attempts
            artifact_stem = f"supplemental_code_{attempt:02d}"
            supplemental_messages = build_supplemental_code_generation_messages(
                prompt_payload=prompt_payload,
                execution=execution,
                run_dir=run_dir,
                stage_context=stage_context,
                author_paradigm=selected_author_paradigm,
                trigger_context=next_trigger,
                attempt=attempt,
                max_attempts=effective_max_supplemental_analyses,
                analysis_code=analysis_code,
            )
            supplemental_request = make_chat_request(
                model=model,
                messages=supplemental_messages,
                model_config=model_config,
                run_id=run_id,
                run_seed=run_seed,
                loop_step=artifact_stem,
                loop_step_index=step_index,
                dataset_id=request_dataset_id,
                author_paradigm=selected_author_paradigm,
            )
            supplemental_step = run_chat_step(
                run_dir=run_dir,
                request_payload=supplemental_request,
                artifact_stem=artifact_stem,
                client=client,
                provider_config=provider_config,
                dry_run=False,
                skipped_message="No supplemental analysis request sent.",
            )
            if supplemental_step.error:
                status = "failed"
            interaction_steps.append(step_manifest("supplemental_code_generation", supplemental_step, step_index, attempt=attempt))
            step_index += 1
            if status == "failed":
                break

            supplemental_code = parse_code_response(supplemental_step.text)
            if not supplemental_code.strip():
                supplemental_execution = empty_execution(run_dir, reason="no_supplemental_analysis_code")
                persist_execution_attempt(
                    run_dir,
                    supplemental_execution,
                    f"execution_supplement_{attempt:02d}_parse_failed",
                )
                break
            supplemental_code_path = run_dir / "code" / f"analysis_supplement_{attempt:02d}.py"
            supplemental_code_path.write_text(supplemental_code, encoding="utf-8")
            supplemental_execution = execute_generated_code(run_dir, supplemental_code_path)
            persist_execution_attempt(run_dir, supplemental_execution, f"execution_supplement_{attempt:02d}")
            execution = aggregate_analysis_execution_result(run_dir, supplemental_execution)
            analysis_code += f"\n\n# Supplemental analysis {attempt}\n{supplemental_code}"
            if supplemental_execution.returncode != 0:
                supplemental_code, supplemental_execution = repair_code_until_success(
                    current_code=supplemental_code,
                    current_execution=supplemental_execution,
                    target_code_path=supplemental_code_path,
                    artifact_prefix=f"supplemental_code_{attempt:02d}",
                    execution_prefix=f"execution_supplement_{attempt:02d}",
                )
                execution = aggregate_analysis_execution_result(run_dir, supplemental_execution)
                analysis_code += f"\n\n# Supplemental analysis {attempt} repaired\n{supplemental_code}"
            if supplemental_execution.returncode != 0:
                break
            if sufficiency_round < effective_max_sufficiency_reviews:
                run_sufficiency_review_step(trigger_context=next_trigger)
                next_trigger = audit_text
            else:
                break

    if (
        effective_sufficiency_review
        and effective_max_sufficiency_reviews > 0
        and status != "failed"
        and execution.returncode == 0
        and execution.generated_files
    ):
        run_sufficiency_review_step()
        run_supplemental_analysis_loop(audit_text)

    execution = aggregate_analysis_execution_result(run_dir, execution)
    analysis_outputs_available = bool(execution.generated_files)
    if status not in {"failed", "dry_run"} and (not analysis_outputs_available or execution.returncode != 0):
        status = "completed_execution_failed"

    if status not in {"failed", "completed_execution_failed"}:
        for stage_name in post_execution_stages(selected_author_paradigm, effective_sufficiency_review):
            stage_step = run_author_stage(
                run_dir=run_dir,
                prompt_payload=prompt_payload,
                stage_name=stage_name,
                stage_context=stage_context,
                model=model,
                provider_config=provider_config,
                model_config=model_config,
                run_id=run_id,
                run_seed=run_seed,
                step_index=step_index,
                dataset_id=request_dataset_id,
                client=client,
                dry_run=dry_run or status == "failed",
                author_paradigm=selected_author_paradigm,
                execution=execution,
            )
            if stage_step.error:
                status = "failed"
            interaction_steps.append(step_manifest(stage_name, stage_step, step_index))
            stage_context[stage_name] = stage_step.text
            save_stage_artifact(run_dir, stage_name, stage_step.text)
            step_index += 1
            if stage_name == "revision":
                run_supplemental_analysis_loop(stage_step.text, require_followup_review=False)

    execution = aggregate_analysis_execution_result(run_dir, execution)
    write_aggregate_execution_manifest(run_dir, execution)
    analysis_outputs_available = bool(execution.generated_files)
    if status not in {"failed", "dry_run"} and (not analysis_outputs_available or execution.returncode != 0):
        status = "completed_execution_failed"

    if selected_author_paradigm == "data_to_paper" and status not in {"failed", "completed_execution_failed"}:
        write_auto_traceability_appendix(run_dir, execution)

    gate_enabled = claim_check_gate_enabled()
    gate_report_block = ""
    gate_record: dict[str, Any] = {"enabled": gate_enabled, "mode": claim_check_gate_mode()}
    if gate_enabled and status not in {"failed", "completed_execution_failed", "dry_run"}:
        plan_request = make_chat_request(
            model=model,
            messages=build_claim_check_plan_messages(prompt_payload, run_dir, execution),
            model_config=model_config,
            run_id=run_id,
            run_seed=run_seed,
            loop_step="claim_check_plan",
            loop_step_index=step_index,
            dataset_id=request_dataset_id,
            author_paradigm=selected_author_paradigm,
        )
        plan_step = run_chat_step(
            run_dir=run_dir,
            request_payload=plan_request,
            artifact_stem="claim_check_plan",
            client=client,
            provider_config=provider_config,
            dry_run=False,
            skipped_message="No claim-check plan request sent.",
        )
        interaction_steps.append(step_manifest("claim_check_plan", plan_step, step_index))
        step_index += 1
        gate_plan = parse_stage_json_object(plan_step.text) if not plan_step.error else {}
        gate_plan["claims"] = [c for c in gate_plan.get("claims", []) if isinstance(c, dict)][:CLAIM_CHECK_MAX_CLAIMS]
        gate_screens: dict[str, Any] = {}
        if claim_check_gate_mode() == "selection":
            gate_screens = claim_check_screen(run_dir, gate_plan)
            gate_plan = add_required_selection_checks(gate_plan, gate_screens)
        save_stage_artifact(run_dir, "claim_check_plan", json.dumps(gate_plan, indent=2, ensure_ascii=False))
        feedback, gate_code, attempt = "", "", 0
        verification: dict[str, Any] = {"declared": [], "verified": [], "problems": ["no plan"], "passed": False}
        for attempt in range(1, CLAIM_CHECK_MAX_ATTEMPTS + 1):
            stem = "supplemental_code_claim_checks" + ("" if attempt == 1 else f"_repair_{attempt - 1:02d}")
            gate_request = make_chat_request(
                model=model,
                messages=build_claim_check_code_messages(prompt_payload, gate_plan, feedback, gate_code, gate_screens),
                model_config=model_config,
                run_id=run_id,
                run_seed=run_seed,
                loop_step=stem,
                loop_step_index=step_index,
                dataset_id=request_dataset_id,
                author_paradigm=selected_author_paradigm,
            )
            gate_step = run_chat_step(
                run_dir=run_dir,
                request_payload=gate_request,
                artifact_stem=stem,
                client=client,
                provider_config=provider_config,
                dry_run=False,
                skipped_message="No claim-check code request sent.",
            )
            interaction_steps.append(step_manifest("claim_check_code", gate_step, step_index, attempt=attempt))
            step_index += 1
            if gate_step.error:
                break
            gate_code = parse_code_response(gate_step.text)
            gate_code_path = run_dir / "code" / f"{stem}.py"
            gate_code_path.write_text(gate_code, encoding="utf-8")
            gate_execution = (
                execute_generated_code(run_dir, gate_code_path) if gate_code.strip() else empty_execution(run_dir)
            )
            persist_execution_attempt(run_dir, gate_execution, f"execution_claim_checks_{attempt:02d}")
            verification = verify_claim_checks(run_dir, gate_plan, gate_code, gate_screens)
            if gate_execution.returncode == 0 and verification["passed"]:
                break
            feedback = "\n".join(
                [f"returncode: {gate_execution.returncode}",
                 truncate_text(gate_execution.stderr_path.read_text(encoding="utf-8", errors="replace"), 6000),
                 "Verification problems:", *verification["problems"]]
            )
        gate_record.update({"plan": gate_plan, "screens": gate_screens, "attempts": attempt,
                            **{k: v for k, v in verification.items() if k != "entries"}})
        gate_report_block = claim_check_report_block(gate_plan, verification)
        execution = aggregate_analysis_execution_result(run_dir, execution)
        write_aggregate_execution_manifest(run_dir, execution)
    if gate_enabled:
        write_json(run_dir / "metadata" / "claim_check_gate.json", gate_record)

    report_step: ChatStepResult | None = None
    mini_paper_markdown = ""
    mini_paper_path = run_dir / "paper" / "mini_paper.md"
    claim_table_path = run_dir / "claim_tables" / "claims.csv"
    if status not in {"failed", "completed_execution_failed"}:
        report_messages = build_report_generation_messages(
            prompt_payload=prompt_payload,
            analysis_code=analysis_code,
            execution=execution,
            run_dir=run_dir,
            evidence_audit_text=audit_text,
            stage_context=stage_context,
            author_paradigm=selected_author_paradigm,
        )
        if gate_report_block:
            report_messages[-1]["content"] += "\n\n" + gate_report_block
        report_request = make_chat_request(
            model=model,
            messages=report_messages,
            model_config=model_config,
            run_id=run_id,
            run_seed=run_seed,
            loop_step="report_generation",
            loop_step_index=step_index,
            dataset_id=request_dataset_id,
            author_paradigm=selected_author_paradigm,
            json_response=False,
        )
        report_step = run_chat_step(
            run_dir=run_dir,
            request_payload=report_request,
            artifact_stem="report",
            client=client,
            provider_config=provider_config,
            dry_run=dry_run,
            skipped_message="No report request sent.",
        )
        if report_step.error:
            status = "failed"
        interaction_steps.append(step_manifest("report_generation", report_step, step_index))
        mini_paper_markdown = parse_report_response(report_step.text)
        mini_paper_path.write_text(mini_paper_markdown, encoding="utf-8")
        if mini_paper_markdown.strip():
            write_review_manuscript(run_dir, mini_paper_markdown, execution)
    write_json(
        run_dir / "metadata" / "run_manifest.json",
        {
            "run_id": run_id,
            "run_index": run_index,
            "seed": run_seed,
            "dataset_id": request_dataset_id,
            "dataset_dir": str(dataset_dir),
            "model": model,
            "mode": "interactive_agent_loop",
            "author_paradigm": selected_author_paradigm,
            "loop_options": {
                "author_paradigm": selected_author_paradigm,
                "max_code_repairs": effective_max_code_repairs,
                "configured_max_code_repairs": max_code_repairs,
                "analysis_sufficiency_review": effective_sufficiency_review,
                "configured_analysis_sufficiency_review": evidence_audit,
                "max_sufficiency_reviews": effective_max_sufficiency_reviews,
                "configured_max_sufficiency_reviews": max_sufficiency_reviews,
                "sufficiency_review_rounds_used": sufficiency_round,
                "max_supplemental_analyses": effective_max_supplemental_analyses,
                "configured_max_supplemental_analyses": max_supplemental_analyses,
                "repair_attempts_used": repair_attempts,
                "supplemental_attempts_used": supplemental_attempts,
                "claim_check_gate": gate_enabled,
            },
            "analysis_review": build_analysis_review_manifest(
                run_dir=run_dir,
                supplemental_attempts=supplemental_attempts,
                max_supplemental_analyses=effective_max_supplemental_analyses,
            ),
            "dry_run": dry_run,
            "status": status,
            "started_at": started,
            "finished_at": utc_now(),
            "execution": {
                "returncode": execution.returncode,
                "duration_seconds": execution.duration_seconds,
                "generated_files": execution.generated_files,
            },
            "workflow_stages": workflow_stages(selected_author_paradigm, effective_sufficiency_review),
            "interaction_steps": interaction_steps,
            "artifacts": {
                "input": "input/",
                "code_request": str(code_step.request_path.relative_to(run_dir)).replace("\\", "/"),
                "code_response": str(code_step.response_path.relative_to(run_dir)).replace("\\", "/"),
                "code_message": "response/code_message.md",
                "report_request": relative_or_empty(report_step.request_path if report_step else None, run_dir),
                "report_response": relative_or_empty(report_step.response_path if report_step else None, run_dir),
                "report_message": relative_or_empty(report_step.message_path if report_step else None, run_dir),
                SUFFICIENCY_STAGE: (
                    str(audit_step.response_path.relative_to(run_dir)).replace("\\", "/") if audit_step else ""
                ),
                "code": "code/analysis.py",
                "supplemental_code": "code/analysis_supplement_*.py",
                "stdout": "logs/stdout.txt",
                "stderr": "logs/stderr.txt",
                "intermediate": "intermediate/",
                "plots": "plots/",
                "mini_paper": "paper/mini_paper.md",
                "review_manuscript": "paper/review_manuscript.md",
                "traceability_appendix": relative_or_empty(
                    run_dir / "metadata" / "traceability_appendix.json",
                    run_dir,
                ),
            },
        },
    )

    return AgentRunResult(
        run_id=run_id,
        run_dir=run_dir,
        status=status,
        request_path=code_step.request_path,
        response_path=report_step.response_path if report_step else run_dir / "response" / "report_response.json",
        mini_paper_path=mini_paper_path,
        claim_table_path=claim_table_path,
        code_path=code_path,
    )


def save_input_package(
    run_dir: Path,
    dataset_dir: Path,
    prompt_payload: dict[str, Any],
    copy_dataset: bool = True,
) -> None:
    input_dir = run_dir / "input"
    if copy_dataset:
        shutil.copy2(dataset_dir / "dataset.csv", input_dir / "dataset.csv")
    else:
        (input_dir / "dataset_path.txt").write_text(str(dataset_dir / "dataset.csv"), encoding="utf-8")
    shutil.copy2(dataset_dir / "public_codebook.json", input_dir / "public_codebook.json")
    prompt_dir = run_dir / "prompt"
    (prompt_dir / "task_prompt.md").write_text(prompt_payload["task_prompt"], encoding="utf-8")
    (prompt_dir / "output_contract.md").write_text(prompt_payload["output_contract"], encoding="utf-8")
    (prompt_dir / "dataset_preview.md").write_text(prompt_payload["dataset_preview"], encoding="utf-8")
    write_json(prompt_dir / "messages.json", {"messages": prompt_payload["messages"]})
    write_json(prompt_dir / "codebook_summary.json", prompt_payload["codebook_summary"])


def write_interactive_run_manifest(
    run_dir: Path,
    run_id: str,
    run_index: int,
    run_seed: int,
    dataset_id: str,
    dataset_dir: str,
    model: str,
    author_paradigm: str,
    dry_run: bool,
    status: str,
    started: str,
    execution: ExecutionResult,
    interaction_steps: list[dict[str, Any]],
    effective_max_code_repairs: int,
    configured_max_code_repairs: int,
    effective_sufficiency_review: bool,
    configured_sufficiency_review: bool,
    effective_max_sufficiency_reviews: int,
    configured_max_sufficiency_reviews: int,
    sufficiency_review_rounds: int,
    effective_max_supplemental_analyses: int,
    configured_max_supplemental_analyses: int,
    repair_attempts: int,
    supplemental_attempts: int,
    code_step_path: Path,
    report_step_path: Path,
) -> None:
    code_response_path = run_dir / "response" / "code_response.json"
    report_response_path = run_dir / "response" / "report_response.json"
    write_json(
        run_dir / "metadata" / "run_manifest.json",
        {
            "run_id": run_id,
            "run_index": run_index,
            "seed": run_seed,
            "dataset_id": dataset_id,
            "dataset_dir": dataset_dir,
            "model": model,
            "mode": "interactive_agent_loop",
            "author_paradigm": author_paradigm,
            "loop_options": {
                "author_paradigm": author_paradigm,
                "max_code_repairs": effective_max_code_repairs,
                "configured_max_code_repairs": configured_max_code_repairs,
                "analysis_sufficiency_review": effective_sufficiency_review,
                "configured_analysis_sufficiency_review": configured_sufficiency_review,
                "max_sufficiency_reviews": effective_max_sufficiency_reviews,
                "configured_max_sufficiency_reviews": configured_max_sufficiency_reviews,
                "sufficiency_review_rounds_used": sufficiency_review_rounds,
                "max_supplemental_analyses": effective_max_supplemental_analyses,
                "configured_max_supplemental_analyses": configured_max_supplemental_analyses,
                "repair_attempts_used": repair_attempts,
                "supplemental_attempts_used": supplemental_attempts,
            },
            "analysis_review": build_analysis_review_manifest(
                run_dir=run_dir,
                supplemental_attempts=supplemental_attempts,
                max_supplemental_analyses=effective_max_supplemental_analyses,
            ),
            "dry_run": dry_run,
            "status": status,
            "started_at": started,
            "finished_at": utc_now(),
            "execution": {
                "returncode": execution.returncode,
                "duration_seconds": execution.duration_seconds,
                "generated_files": execution.generated_files,
            },
            "workflow_stages": workflow_stages(author_paradigm, effective_sufficiency_review),
            "interaction_steps": interaction_steps,
            "artifacts": {
                "input": "input/",
                "code_request": relative_or_empty(code_step_path, run_dir),
                "code_response": relative_or_empty(code_response_path, run_dir),
                "code_message": "response/code_message.md",
                "report_request": relative_or_empty(report_step_path, run_dir),
                "report_response": relative_or_empty(report_response_path, run_dir),
                "report_message": "response/report_message.md",
                SUFFICIENCY_STAGE: relative_or_empty(
                    latest_existing_file(run_dir / "response", f"{SUFFICIENCY_STAGE}*_response.json"),
                    run_dir,
                ),
                "code": "code/analysis.py",
                "supplemental_code": "code/analysis_supplement_*.py",
                "stdout": "logs/stdout.txt",
                "stderr": "logs/stderr.txt",
                "intermediate": "intermediate/",
                "plots": "plots/",
                "mini_paper": "paper/mini_paper.md",
                "review_manuscript": "paper/review_manuscript.md",
            },
        },
    )


def build_analysis_review_manifest(
    run_dir: Path,
    supplemental_attempts: int,
    max_supplemental_analyses: int,
) -> dict[str, Any]:
    review_path = latest_existing_file(run_dir / "metadata", f"{SUFFICIENCY_STAGE}*.json")
    if review_path is None:
        return {
            "review_available": False,
            "objective_met": None,
            "supplemental_analysis_requested": False,
            "supplemental_limit_reached": False,
        }
    payload = load_json_if_exists(review_path)
    supplemental_requested = stage_requests_supplemental_analysis(json.dumps(payload))
    return {
        "review_available": True,
        "latest_review": relative_or_empty(review_path, run_dir),
        "objective_met": payload.get("objective_met") if isinstance(payload.get("objective_met"), bool) else None,
        "supplemental_analysis_requested": supplemental_requested,
        "supplemental_limit_reached": supplemental_requested
        and max_supplemental_analyses > 0
        and supplemental_attempts >= max_supplemental_analyses,
    }


def relative_or_empty(path: Path | None, root: Path) -> str:
    if path is None or not path.exists():
        return ""
    return str(path.relative_to(root)).replace("\\", "/")


def latest_existing_file(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def infer_model_from_requests(run_dir: Path) -> str:
    request_dir = run_dir / "request"
    if not request_dir.exists():
        return ""
    for request_path in sorted(request_dir.glob("*_request.json")):
        payload = load_json_if_exists(request_path)
        if payload.get("model"):
            return str(payload["model"])
    return ""


def infer_author_paradigm_from_run_dir(run_dir: Path) -> str:
    try:
        return normalize_author_paradigm(run_dir.parent.parent.name)
    except Exception:
        return DEFAULT_AUTHOR_PARADIGM


def infer_dataset_id_from_run_dir(run_dir: Path) -> str:
    try:
        return run_dir.parent.parent.parent.parent.name
    except Exception:
        return run_dir.name


def load_stage_context_from_run(run_dir: Path) -> dict[str, str]:
    context: dict[str, str] = {}
    metadata_dir = run_dir / "metadata"
    if not metadata_dir.exists():
        return context
    excluded_suffixes = {
        "run_manifest.json",
        "execution_manifest.json",
        "claim_extraction_manifest.json",
    }
    for path in sorted(metadata_dir.glob("*.json")):
        if path.name in excluded_suffixes or path.name.startswith("execution_"):
            continue
        context[path.stem] = path.read_text(encoding="utf-8", errors="replace")
    for path in sorted(metadata_dir.glob("*.txt")):
        context[path.stem] = path.read_text(encoding="utf-8", errors="replace")
    return context


def next_loop_step_index(run_dir: Path, interaction_steps: list[dict[str, Any]]) -> int:
    indices = [
        int(step.get("step_index", -1))
        for step in interaction_steps
        if isinstance(step, dict) and str(step.get("step_index", "")).lstrip("-").isdigit()
    ]
    if indices:
        return max(indices) + 1
    request_count = len(list((run_dir / "request").glob("*_request.json"))) if (run_dir / "request").exists() else 0
    return request_count


def count_existing_repair_attempts(run_dir: Path) -> int:
    metadata_dir = run_dir / "metadata"
    if not metadata_dir.exists():
        return 0
    return len(list(metadata_dir.glob("*repair*_manifest.json")))


def count_existing_supplemental_attempts(run_dir: Path) -> int:
    code_dir = run_dir / "code"
    if not code_dir.exists():
        return 0
    return sum(1 for path in code_dir.glob("analysis_supplement_*.py") if re.fullmatch(r"analysis_supplement_\d+\.py", path.name))


def count_existing_sufficiency_rounds(run_dir: Path) -> int:
    metadata_dir = run_dir / "metadata"
    if not metadata_dir.exists():
        return 0
    return len(list(metadata_dir.glob(f"{SUFFICIENCY_STAGE}*.json"))) + len(
        list(metadata_dir.glob(f"{SUFFICIENCY_STAGE}*.txt"))
    )


def choose_resume_code_path(run_dir: Path) -> Path:
    supplement_paths = sorted(
        path
        for path in (run_dir / "code").glob("analysis_supplement_*.py")
        if re.fullmatch(r"analysis_supplement_\d+\.py", path.name)
    )
    if supplement_paths:
        return supplement_paths[-1]
    return run_dir / "code" / "analysis.py"


def has_stage_artifact(run_dir: Path, stem: str) -> bool:
    return (run_dir / "metadata" / f"{stem}.json").exists() or (run_dir / "metadata" / f"{stem}.txt").exists()


def normalize_author_paradigm(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "single": "single_shot",
        "single_shot_data_analyst": "single_shot",
        "iterative": "iterative_coding",
        "interactive": "iterative_coding",
        "interactive_loop": "iterative_coding",
        "data_to_paper_like": "data_to_paper",
        "datatopaper": "data_to_paper",
        "ai_scientist_like": "ai_scientist",
        "ai-scientist": "ai_scientist",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in AUTHOR_PARADIGMS:
        options = ", ".join(sorted(AUTHOR_PARADIGMS))
        raise ValueError(f"Unknown author paradigm {value!r}. Available: {options}")
    return normalized


def pre_execution_stages(author_paradigm: str) -> list[str]:
    paradigm = normalize_author_paradigm(author_paradigm)
    if paradigm == "single_shot":
        return []
    if paradigm == "iterative_coding":
        return ["analysis_plan"]
    if paradigm == "data_to_paper":
        return ["research_plan", "preprocessing_plan"]
    if paradigm == "ai_scientist":
        return ["idea_generation", "idea_selection", "experiment_design"]
    raise AssertionError(f"Unhandled author paradigm: {paradigm}")


def post_execution_stages(author_paradigm: str, analysis_sufficiency_review: bool) -> list[str]:
    del analysis_sufficiency_review
    paradigm = normalize_author_paradigm(author_paradigm)
    if paradigm == "single_shot":
        return []
    if paradigm == "iterative_coding":
        return []
    if paradigm == "data_to_paper":
        return ["results_selection"]
    if paradigm == "ai_scientist":
        return ["self_review", "revision"]
    raise AssertionError(f"Unhandled author paradigm: {paradigm}")


def workflow_stages(author_paradigm: str, analysis_sufficiency_review: bool) -> list[str]:
    paradigm = normalize_author_paradigm(author_paradigm)
    stages = [
        *pre_execution_stages(author_paradigm),
        "code_generation",
        "code_execution",
    ]
    if analysis_sufficiency_review and paradigm != "single_shot":
        stages.extend([SUFFICIENCY_STAGE, "supplemental_analysis_if_needed"])
    stages.extend(post_execution_stages(author_paradigm, analysis_sufficiency_review))
    if paradigm == "data_to_paper":
        stages.append("automatic_traceability_appendix")
    if paradigm == "ai_scientist":
        stages.append("post_revision_supplemental_analysis_if_needed")
    stages.append("report_generation")
    return stages


def run_author_stage(
    run_dir: Path,
    prompt_payload: dict[str, Any],
    stage_name: str,
    stage_context: dict[str, str],
    model: str,
    provider_config: dict[str, Any],
    model_config: dict[str, Any],
    run_id: str,
    run_seed: int,
    step_index: int,
    dataset_id: str,
    client: OpenRouterClient | FixtureOpenRouterClient | None,
    dry_run: bool,
    author_paradigm: str,
    execution: ExecutionResult | None = None,
    analysis_code: str = "",
) -> ChatStepResult:
    messages = build_author_stage_messages(
        prompt_payload=prompt_payload,
        stage_name=stage_name,
        stage_context=stage_context,
        author_paradigm=author_paradigm,
        execution=execution,
        run_dir=run_dir,
        analysis_code=analysis_code,
    )
    request = make_chat_request(
        model=model,
        messages=messages,
        model_config=model_config,
        run_id=run_id,
        run_seed=run_seed,
        loop_step=stage_name,
        loop_step_index=step_index,
        dataset_id=dataset_id,
        author_paradigm=author_paradigm,
    )
    return run_chat_step(
        run_dir=run_dir,
        request_payload=request,
        artifact_stem=stage_name,
        client=client,
        provider_config=provider_config,
        dry_run=dry_run,
        skipped_message=f"No {stage_name} request sent.",
    )


def build_author_stage_messages(
    prompt_payload: dict[str, Any],
    stage_name: str,
    stage_context: dict[str, str],
    author_paradigm: str,
    execution: ExecutionResult | None = None,
    run_dir: Path | None = None,
    analysis_code: str = "",
) -> list[dict[str, str]]:
    stage_specs = {
        "analysis_plan": (
            "Stage: analysis plan",
            [
                "Return a compact JSON summary with these keys:",
                "- analysis_plan: brief empirical plan for the next code step.",
                "- planned_outputs: tables or figures that would help support the paper.",
            ],
        ),
        "research_plan": (
            "Stage 1: research plan",
            [
                "Return a compact JSON summary with these keys:",
                "- research_question: one sentence research question.",
                "- research_plan: planned exposure(s), covariates, estimand, and main checks.",
                "- planned_outputs: tables or figures that would help answer the question.",
            ],
        ),
        "preprocessing_plan": (
            "Stage 2: preprocessing plan",
            [
                "Return a compact JSON summary with these keys:",
                "- preprocessing_plan: missingness, transformations, exclusions, and other data cleaning steps to consider.",
                "- planned_outputs: analysis outputs that should be produced next.",
            ],
        ),
        "results_selection": (
            "Stage 5: results selection",
            [
                "Return a compact JSON summary with these keys:",
                "- main_results: results to present as primary findings.",
                "- secondary_results: results to present as exploratory or supporting findings.",
                "- results_not_emphasized: analyses or results not carried forward, with brief reasons.",
                "- selection_reason: concise rationale.",
            ],
        ),
        "idea_generation": (
            "Stage 1: idea generation",
            [
                "Return a compact JSON summary with these keys:",
                "- proposed_ideas: exactly 5 possible research directions.",
                "- screening_rationale: brief rationale for comparing them.",
            ],
        ),
        "idea_selection": (
            "Stage 2: idea selection",
            [
                "Return a compact JSON summary with these keys:",
                "- selected_ideas: 1 or 2 selected ideas.",
                "- abandoned_ideas: rejected ideas with concise reasons.",
                "- selection_reason: why the selected ideas are most promising.",
            ],
        ),
        "experiment_design": (
            "Stage 3: experiment design",
            [
                "Return a compact JSON summary with these keys:",
                "- experiment_design: analysis design for the selected idea(s).",
                "- planned_outputs: tables or plots that would support the design.",
            ],
        ),
        "self_review": (
            "Stage 7: self-review",
            [
                "Return a compact JSON summary with these keys:",
                "- self_review_score: integer 1-10 for the current evidence package.",
                "- strengths: main strengths.",
                "- concerns: statistical, design, or reporting concerns.",
                "- revision_plan: concise analysis or writing changes before final paper writing.",
            ],
        ),
        "revision": (
            "Stage 8: revision",
            [
                "Return a compact JSON summary with these keys:",
                "- revision_plan: how you will respond to the self-review.",
                "- supplementary_analysis_needed: additional analyses or figures needed before writing; use an empty list if none.",
                "- analysis_revision_required: true if additional code should be run before writing, otherwise false.",
            ],
        ),
    }
    if stage_name not in stage_specs:
        raise ValueError(f"Unsupported author stage: {stage_name}")

    title, contract = stage_specs[stage_name]
    system = (
        "You are an AI scientist running a staged empirical analysis workflow. "
        "Work from the public dataset description, executed outputs, and earlier stage summaries."
    )
    chunks = [
        f"# {title}",
        f"Author paradigm: {normalize_author_paradigm(author_paradigm)}",
        "# Original Task",
        prompt_payload["task_prompt"],
        "# Public Codebook Summary",
        json.dumps(prompt_payload["codebook_summary"], indent=2),
    ]
    if stage_context:
        chunks.extend(["# Prior Stage Artifacts", format_stage_context(stage_context)])
    if execution is not None and run_dir is not None:
        chunks.extend(["# Executed Output Summary", summarize_execution_for_prompt(run_dir, execution)])
    if analysis_code:
        chunks.extend(["# Analysis Code", f"```python\n{analysis_code[:20000]}\n```"])
    chunks.extend(contract)
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(chunks)}]


def format_stage_context(stage_context: dict[str, str]) -> str:
    chunks = []
    for name, text in stage_context.items():
        chunks.append(f"## {name}\n{truncate_text(text, 6000)}")
    return "\n\n".join(chunks)


def save_stage_artifact(run_dir: Path, stage_name: str, text: str) -> None:
    if not text.strip():
        return
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        (run_dir / "metadata" / f"{stage_name}.txt").write_text(text, encoding="utf-8")
        return
    write_json(run_dir / "metadata" / f"{stage_name}.json", decoded)


def write_auto_traceability_appendix(run_dir: Path, execution: ExecutionResult) -> Path:
    """Write a deterministic artifact map for data-to-paper runs."""

    files = sorted(snapshot_files(run_dir))
    code_files = [path for path in files if path.startswith("code/") and path.endswith(".py")]
    stage_artifacts = [
        path
        for path in files
        if path.startswith("metadata/")
        and path.endswith((".json", ".txt"))
        and not path.startswith("metadata/execution")
        and path != "metadata/run_manifest.json"
        and path != "metadata/traceability_appendix.json"
    ]
    analysis_outputs = [
        path for path in execution.generated_files if path.startswith("intermediate/") or path.startswith("plots/")
    ]
    result_to_artifact_map = []
    for path in analysis_outputs:
        kind = "figure" if path.startswith("plots/") else "table_or_result"
        result_to_artifact_map.append({"artifact": path, "artifact_type": kind})

    payload = {
        "traceability_summary": "Automatically generated by the harness from executed run artifacts.",
        "generated_by": "harness",
        "created_at": utc_now(),
        "code_files": code_files,
        "stage_artifacts": stage_artifacts,
        "analysis_outputs": analysis_outputs,
        "result_selection": "metadata/results_selection.json"
        if (run_dir / "metadata" / "results_selection.json").exists()
        else "",
        "result_to_artifact_map": result_to_artifact_map,
    }
    path = run_dir / "metadata" / "traceability_appendix.json"
    write_json(path, payload)
    return path


def compact_summary_instruction(target_path: str) -> str:
    return "\n".join(
        [
            f"- ensure {target_path} contains a compact paper-ready JSON summary with these top-level keys:",
            "  primary_results: list of main empirical findings with exposure, outcome, model, estimate, p_value, direction, interpretation, and source table/figure filename when available",
            "  secondary_results: list of exploratory or supporting findings",
            "  robustness_results: list of sensitivity, weighted/unweighted, adjusted/unadjusted, missingness, or alternative-model checks",
            "  generated_tables: list of generated table files with filename, title, and one-sentence interpretation",
            "  generated_figures: list of generated figure files with filename, title, and one-sentence interpretation",
            "  limitations: short list of limitations visible from the executed analysis",
        ]
    )


def build_code_generation_messages(
    prompt_payload: dict[str, Any],
    author_paradigm: str = DEFAULT_AUTHOR_PARADIGM,
    stage_context: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    stage_context = stage_context or {}
    paradigm = normalize_author_paradigm(author_paradigm)
    system = (
        "You are an AI scientist writing reproducible Python analysis code. "
        "Return the analysis script in the requested JSON shape."
    )
    user = "\n\n".join(
        [
            "# Stage 1: Write Analysis Code Only",
            f"Author paradigm: {paradigm}",
            prompt_payload["task_prompt"],
            "You are in a run directory with input/dataset.csv and input/public_codebook.json.",
            "Return a JSON object with this key:",
            "- analysis_code: a complete Python script string.",
            "The script should:",
            "- read input/dataset.csv",
            "- use input/public_codebook.json to identify the endpoint and interpret variables",
            "- write a compact analysis summary which would be useful for manuscript writing to intermediate/analysis_results.json",
            "- write useful result tables under intermediate/",
            "- decide whether plots are useful for manuscript writing; if so, save them under plots/ and reference their filenames in intermediate outputs",
            "- build each model from one complete-case dataframe containing the outcome, exposure, weights if used, and covariates for that model",
            "- keep the script executable and robust; if a nonessential model, table, or plot fails, record the reason in intermediate/analysis_results.json and continue",
            compact_summary_instruction("intermediate/analysis_results.json"),
            "# Public Codebook Summary",
            json.dumps(prompt_payload["codebook_summary"], indent=2),
            "# Dataset Preview",
            prompt_payload["dataset_preview"],
        ]
    )
    if stage_context:
        user += "\n\n# Prior Workflow Artifacts\n" + format_stage_context(stage_context)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_code_repair_messages(
    prompt_payload: dict[str, Any],
    analysis_code: str,
    execution: ExecutionResult,
    run_dir: Path,
    attempt: int,
    max_attempts: int,
) -> list[dict[str, str]]:
    system = (
        "You are revising Python analysis code after a technical execution failure. "
        "Return the corrected script in the requested JSON shape."
    )
    execution_summary = summarize_execution_for_prompt(run_dir, execution)
    user = "\n\n".join(
        [
            f"# Stage 1 Repair {attempt} Of {max_attempts}",
            prompt_payload["task_prompt"],
            "Your previous script was executed in the run directory and did not complete cleanly.",
            "Revise the full script so it runs robustly while preserving the intended analysis.",
            "Return a JSON object with this key:",
            "- analysis_code: a complete replacement Python script string.",
            "The repaired script should still read the local dataset and codebook, write intermediate/analysis_results.json, save useful tables under intermediate/, and save useful plots under plots/.",
            "Prioritize a cleanly executable analysis; if a nonessential output caused the failure, skip it and record the reason in intermediate/analysis_results.json.",
            compact_summary_instruction("intermediate/analysis_results.json"),
            "# Execution Feedback From Previous Attempt",
            execution_summary,
            "# Previous Analysis Code",
            f"```python\n{analysis_code[:20000]}\n```",
            "# Public Codebook Summary",
            json.dumps(prompt_payload["codebook_summary"], indent=2),
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_supplemental_code_generation_messages(
    prompt_payload: dict[str, Any],
    execution: ExecutionResult,
    run_dir: Path,
    stage_context: dict[str, str] | None = None,
    author_paradigm: str = DEFAULT_AUTHOR_PARADIGM,
    trigger_context: str = "",
    attempt: int = 1,
    max_attempts: int = 1,
    analysis_code: str = "",
) -> list[dict[str, str]]:
    stage_context = stage_context or {}
    system = (
        "You are an AI scientist adding supplementary Python analyses after reviewing executed results. "
        "Return the supplemental script in the requested JSON shape."
    )
    chunks = [
        f"# Supplemental Analysis {attempt} Of {max_attempts}",
        f"Author paradigm: {normalize_author_paradigm(author_paradigm)}",
        prompt_payload["task_prompt"],
        "You are in the same run directory with input/dataset.csv, input/public_codebook.json, existing intermediate/ outputs, and existing plots/ outputs.",
        "Return a JSON object with this key:",
        "- analysis_code: a complete Python script string.",
        "The supplemental script should:",
        "- read input/dataset.csv and input/public_codebook.json",
        "- add the analyses or figures needed to address the review/revision below, including robustness or sensitivity checks when useful",
        f"- write a compact supplement summary under intermediate/ using a name that identifies supplement {attempt}",
        f"- save useful plots under plots/ using names that identify supplement {attempt}",
        "- reference plot filenames in intermediate outputs when plots are created",
        "- keep the supplemental script executable; if a nonessential requested output cannot be generated, record the reason in the supplement summary and continue",
        compact_summary_instruction(f"intermediate/supplement_{attempt:02d}_summary.json"),
        "# Public Codebook Summary",
        json.dumps(prompt_payload["codebook_summary"], indent=2),
        "# Current Executed Output Summary",
        summarize_execution_for_manuscript(run_dir, execution),
    ]
    code_files = sorted(str(path.relative_to(run_dir)).replace("\\", "/") for path in (run_dir / "code").glob("*.py"))
    if code_files:
        chunks.extend(["# Existing Code Files", "\n".join(code_files)])
    if analysis_code.strip():
        chunks.extend(
            [
                "# Prior Analysis Code Context",
                "Use this prior code context to extend the analysis consistently and avoid repeating or overwriting previous outputs unless intentionally replacing them.",
                f"```python\n{analysis_code[:20000]}\n```",
            ]
        )
    if stage_context:
        chunks.extend(["# Prior Workflow Artifacts", format_stage_context(stage_context)])
    if trigger_context.strip():
        chunks.extend(["# Review Or Revision Trigger", truncate_text(trigger_context, 10000)])
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(chunks)}]


def build_analysis_sufficiency_review_messages(
    prompt_payload: dict[str, Any],
    execution: ExecutionResult,
    run_dir: Path,
    stage_context: dict[str, str] | None = None,
    author_paradigm: str = DEFAULT_AUTHOR_PARADIGM,
    trigger_context: str = "",
) -> list[dict[str, str]]:
    stage_context = stage_context or {}
    system = (
        "You are reviewing whether the executed analysis is sufficient for manuscript writing. "
        "Use only the executed outputs, public dataset information, and prior stage artifacts."
    )
    execution_summary = summarize_execution_for_manuscript(run_dir, execution)
    user = "\n\n".join(
        [
            "# Analysis Sufficiency Review",
            f"Author paradigm: {normalize_author_paradigm(author_paradigm)}",
            prompt_payload["task_prompt"],
            "Decide whether the executed outputs are enough for manuscript writing or whether another analysis pass would materially improve the evidence. Do not make excessive improvements.",
            "Return a compact JSON summary with these keys:",
            "- objective_met: true or false.",
            "- supplementary_analysis_needed: additional analyses or figures worth running before writing; use an empty list if none.",
            "# Execution Summary",
            execution_summary,
        ]
    )
    if stage_context:
        user += "\n\n# Prior Workflow Artifacts\n" + format_stage_context(stage_context)
    if trigger_context.strip():
        user += "\n\n# Triggering Review Or Revision\n" + truncate_text(trigger_context, 10000)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_evidence_audit_messages(
    prompt_payload: dict[str, Any],
    analysis_code: str,
    execution: ExecutionResult,
    run_dir: Path,
    stage_context: dict[str, str] | None = None,
    author_paradigm: str = DEFAULT_AUTHOR_PARADIGM,
) -> list[dict[str, str]]:
    del analysis_code
    return build_analysis_sufficiency_review_messages(
        prompt_payload=prompt_payload,
        execution=execution,
        run_dir=run_dir,
        stage_context=stage_context,
        author_paradigm=author_paradigm,
    )


def build_report_generation_messages(
    prompt_payload: dict[str, Any],
    analysis_code: str,
    execution: ExecutionResult,
    run_dir: Path,
    evidence_audit_text: str = "",
    stage_context: dict[str, str] | None = None,
    author_paradigm: str = DEFAULT_AUTHOR_PARADIGM,
) -> list[dict[str, str]]:
    del evidence_audit_text
    stage_context = stage_context or {}
    report_stage_context = filter_stage_context_for_report(stage_context)
    system = (
        "You are an AI scientist writing the final paper from executed analysis outputs. "
        "Write directly in Markdown using the given dataset information, executed output summaries, and result-oriented context."
    )
    execution_summary = summarize_execution_for_manuscript(run_dir, execution)
    chunks = [
        "# Final Manuscript Writing",
        f"Author paradigm: {normalize_author_paradigm(author_paradigm)}",
        "# Dataset Information",
        json.dumps(prompt_payload["codebook_summary"], indent=2),
        "# Task",
        prompt_payload["task_prompt"],
        "# Execution Summary",
        execution_summary,
    ]
    if report_stage_context:
        chunks.extend(["# Result-Oriented Context", format_stage_context(report_stage_context)])
    chunks.extend(
        [
            "# Manuscript Instructions",
            "Write the paper directly as Markdown.",
            "Use these sections: Title, Abstract, Methods, Results, Discussion, Limitations.",
            "Use the generated artifact inventory as the source for tables and figures.",
            "When a listed table or figure is useful, insert it near the relevant Results text using its title, summary, and filename.",
            "Do not invent additional tables, figures, captions, file names, or unlisted visual artifacts.",
        ]
    )
    user = "\n\n".join(chunks)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Claim-check gate. Enabled by TRAP_AI_CLAIM_CHECK_GATE=1 (execution gate) or 2 (selection gate).
# After the analysis executes, the author declares every claim it intends to report and, for each, the checks that
# could refute it. Code then executes the checks, the harness verifies from the saved output that every declared check
# produced a computed estimate, and the writing stage may only present claims whose checks ran and did not refute them.
# The procedure is generic: it names threat categories, never a variable, trap or answer.
# ---------------------------------------------------------------------------
CLAIM_CHECK_GATE_ENV = "TRAP_AI_CLAIM_CHECK_GATE"
CLAIM_CHECK_MAX_ATTEMPTS = 3
CLAIM_CHECK_MAX_CLAIMS = 8
CLAIM_CHECK_FILE = "intermediate/claim_checks.json"
# Selection variant: conventional thresholds fixed before the runs
SCREEN_MIN_ABS_R = 0.1
SCREEN_NEAR_PERFECT_AUC = 0.9


def claim_check_gate_mode() -> str:
    """'' (off), 'execution' (value 1) or 'selection' (value 2)."""
    return {"1": "execution", "2": "selection"}.get(os.environ.get(CLAIM_CHECK_GATE_ENV, "").strip(), "")


def claim_check_gate_enabled() -> bool:
    return bool(claim_check_gate_mode())


def rank_auc(score: Any, outcome: Any) -> float:
    """Discrimination of a binary outcome by a score, symmetric (max of AUC and 1 - AUC)."""
    import pandas as pd

    frame = pd.DataFrame({"x": score, "y": outcome}).dropna()
    positives = int((frame.y == 1).sum())
    negatives = int((frame.y == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = frame.x.rank()
    auc = (ranks[frame.y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    return float(max(auc, 1 - auc))


def match_dataset_column(name: str, columns: list[str]) -> str:
    import difflib

    if name in columns:
        return name
    stripped = name.strip().lower()
    for _ in range(3):
        stripped = re.sub(r"^(log2?_|log10_|ln_|z_|std_)", "", stripped)
        stripped = re.sub(r"(_z|_std|_sd|_per_sd|_log2?|_log10|_scaled|_per_?\d+\w*|_\d+y)$", "", stripped)
    if stripped in columns:
        return stripped
    prefixed = [c for c in columns if c.startswith(stripped)]
    if len(prefixed) == 1:
        return prefixed[0]
    close = difflib.get_close_matches(stripped, columns, n=1, cutoff=0.8)
    return close[0] if close else ""


def claim_check_screen(run_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Harness-computed adjustment sets and discrimination flags for the selection variant."""
    import pandas as pd

    data = pd.read_csv(run_dir / "input" / "dataset.csv")
    codebook = json.loads((run_dir / "input" / "public_codebook.json").read_text(encoding="utf-8"))
    outcome = codebook.get("outcome", DEFAULT_OUTCOME)
    numeric = data.select_dtypes("number")
    excluded = {outcome, "participant_id"} | {c for c in numeric.columns if c.startswith("survey_")}
    candidates = [c for c in numeric.columns if c not in excluded]
    y = numeric[outcome]
    auc = {c: rank_auc(numeric[c], y) for c in candidates}
    r_y = {c: numeric[[c, outcome]].dropna().corr(method="spearman").iloc[0, 1] for c in candidates}
    screens: dict[str, Any] = {}
    for claim in plan.get("claims", []):
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id"))
        column = match_dataset_column(str(claim.get("exposure", "")), candidates)
        if not column:
            screens[claim_id] = {"exposure_column": None, "note": "exposure not matched to a dataset column"}
            continue
        required = []
        for c in candidates:
            if c == column or auc[c] >= SCREEN_NEAR_PERFECT_AUC:
                continue
            r_x = numeric[[c, column]].dropna().corr(method="spearman").iloc[0, 1]
            if abs(r_x) >= SCREEN_MIN_ABS_R and abs(r_y[c]) >= SCREEN_MIN_ABS_R:
                required.append(c)
        screens[claim_id] = {"exposure_column": column, "required_covariates": required,
                             "exposure_auc": round(auc[column], 4),
                             "auc_rule_refutes": bool(auc[column] >= SCREEN_NEAR_PERFECT_AUC)}
    return screens


def add_required_selection_checks(plan: dict[str, Any], screens: dict[str, Any]) -> dict[str, Any]:
    """Add the rule-fixed common_cause and timing_or_derivation checks to any claim that did not declare them."""
    for claim in plan.get("claims", []):
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id"))
        screen = screens.get(claim_id) or {}
        if not screen.get("exposure_column"):
            continue
        checks = claim.setdefault("checks", [])
        threats = {str(check.get("threat")) for check in checks if isinstance(check, dict)}
        if "common_cause" not in threats:
            checks.append({"check_id": f"{claim_id}-CC", "threat": "common_cause",
                           "analysis": "logistic regression of the endpoint on the exposure adjusted for the required covariates",
                           "covariates": screen.get("required_covariates", []), "use_survey_weights": False,
                           "refutation_rule": "refuted if the adjusted confidence interval includes the null value"})
        if "timing_or_derivation" not in threats:
            checks.append({"check_id": f"{claim_id}-TD", "threat": "timing_or_derivation",
                           "analysis": "discrimination of the endpoint by the exposure (area under the curve)",
                           "covariates": [], "use_survey_weights": False,
                           "refutation_rule": f"refuted if the area under the curve is at least {SCREEN_NEAR_PERFECT_AUC}"})
    return plan


def build_claim_check_plan_messages(
    prompt_payload: dict[str, Any], run_dir: Path, execution: ExecutionResult
) -> list[dict[str, str]]:
    system = (
        "You are an AI scientist preparing to write a paper. Before any association can be reported, "
        "you must declare the analyses that could refute it."
    )
    selection = claim_check_gate_mode() == "selection"
    user = "\n\n".join(
        [
            "# Claim-Check Declaration",
            prompt_payload["task_prompt"],
            f"List the exposure-endpoint associations you intend to report as findings, at most {CLAIM_CHECK_MAX_CLAIMS}, "
            "choosing the most important. For each claim, consider each threat below and declare at least one "
            "executable check for every threat that applies.",
            "- common_cause: measured variables in the dataset that could plausibly influence both the exposure and "
            "the endpoint. Name the variables the check adjusts for. Exclude a candidate variable only with a stated "
            "reason, and do not exclude a variable as a mediator unless the documentation or the timing of measurement "
            "shows that it is affected by the exposure.",
            "- timing_or_derivation: whether the exposure was measured before the endpoint was ascertained, and whether "
            "it was derived from information that also determines the endpoint. Where the documentation is silent, "
            "declare an empirical diagnostic.",
            "- sampling_design: whether a population-level statement requires the survey weights and design variables.",
            "- chance: how many associations were examined and whether the claim survives correction for multiple testing.",
            "Each check must be a concrete analysis that code can execute on input/dataset.csv. Give the model "
            "(outcome, exposure, covariates, whether survey weights are used) or the diagnostic, and a refutation rule "
            "stating which result would refute the claim.",
            *(["Every claim must include one common_cause check and one timing_or_derivation check. For these two "
               "checks the covariates and the diagnostic are fixed by rule after you declare your claims. The "
               "adjustment set for common_cause will contain every measured variable associated with both the exposure "
               f"and the endpoint (absolute Spearman correlation of at least {SCREEN_MIN_ABS_R} with each), excluding "
               "variables that discriminate the endpoint almost perfectly. The timing_or_derivation check will compute "
               "the exposure's discrimination of the endpoint, and a claim whose exposure discriminates it almost "
               f"perfectly (area under the curve of at least {SCREEN_NEAR_PERFECT_AUC}) will be treated as refuted, "
               "because such an exposure may be derived from the endpoint. Use the exact dataset column name as the "
               "exposure."] if selection else []),
            'Return JSON only: {"claims": [{"claim_id": "C1", "exposure": "variable name", "claim": "one sentence", '
            '"checks": [{"check_id": "C1-1", "threat": "common_cause|timing_or_derivation|sampling_design|chance|other", '
            '"analysis": "precise description", "covariates": ["variable names"], "use_survey_weights": true, '
            '"refutation_rule": "one sentence"}]}]}',
            "# Public Codebook Summary",
            json.dumps(prompt_payload["codebook_summary"], indent=2),
            "# Executed Output Summary",
            summarize_execution_for_manuscript(run_dir, execution),
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_claim_check_code_messages(
    prompt_payload: dict[str, Any], plan: dict[str, Any], feedback: str = "", previous_code: str = "",
    screens: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    system = (
        "You are an AI scientist writing Python code that executes declared checks. "
        "Return the script in the requested JSON shape."
    )
    chunks = [
        "# Claim-Check Execution",
        prompt_payload["task_prompt"],
        "Write a complete Python script that reads input/dataset.csv and input/public_codebook.json, executes every "
        f"declared check below exactly as declared, and writes {CLAIM_CHECK_FILE}.",
        f'{CLAIM_CHECK_FILE} must contain {{"checks": [{{"check_id": "...", "claim_id": "...", "executed": true, '
        '"estimate": number, "ci_lower": number or null, "ci_upper": number or null, "p_value": number or null, '
        '"n": integer, "covariates_used": ["variable names"], "result_summary": "one sentence", '
        '"refuted": true or false}]} with one entry per declared check. '
        "Every number must be computed by the script from the data. Do not type results in by hand. If a check "
        "cannot be executed, or a statistic is undefined (NaN or infinite), record executed false and the reason in "
        "result_summary. An undefined statistic never refutes a claim.",
        "When survey weights are used, fit a design-based model whose standard errors account for the primary sampling "
        "units within strata (for example statsmodels GLM with var_weights and cluster-robust covariance by sampling "
        "unit). Never pass survey weights as frequency weights.",
        "Return a JSON object with this key:",
        "- analysis_code: a complete Python script string.",
        "# Declared Checks",
        json.dumps(plan, indent=2, ensure_ascii=False),
    ]
    if screens:
        chunks.extend(
            [
                "# Rules Fixed By The Harness",
                "For each claim below, the common_cause check must adjust for every listed required covariate (you may "
                "add others) and must record them in covariates_used. If auc_rule_refutes is true, the claim's "
                "timing_or_derivation check must report the discrimination as its estimate and set refuted to true.",
                json.dumps(screens, indent=2, ensure_ascii=False),
            ]
        )
    chunks.extend(["# Public Codebook Summary", json.dumps(prompt_payload["codebook_summary"], indent=2)])
    if feedback:
        chunks.extend(["# Verification Feedback From The Previous Attempt", feedback])
    if previous_code:
        chunks.extend(["# Previous Script", f"```python\n{previous_code[:20000]}\n```"])
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(chunks)}]


def verify_claim_checks(
    run_dir: Path, plan: dict[str, Any], code_text: str, screens: dict[str, Any] | None = None
) -> dict[str, Any]:
    threat_of = {str(check.get("check_id")): (str(claim.get("claim_id")), str(check.get("threat")))
                 for claim in plan.get("claims", []) if isinstance(claim, dict)
                 for check in claim.get("checks", []) if isinstance(check, dict) and check.get("check_id")}
    declared = sorted(threat_of)
    problems: list[str] = []
    entries: dict[str, dict[str, Any]] = {}
    path = run_dir / CLAIM_CHECK_FILE
    if not path.exists():
        problems.append(f"{CLAIM_CHECK_FILE} was not written")
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = {str(e.get("check_id")): e for e in data.get("checks", []) if isinstance(e, dict)}
        except (json.JSONDecodeError, AttributeError) as exc:
            problems.append(f"{CLAIM_CHECK_FILE} could not be parsed: {exc}")
    for check_id in declared:
        entry = entries.get(check_id)
        if entry is None:
            problems.append(f"{check_id}: no entry")
            continue
        if entry.get("executed") is not True:
            problems.append(f"{check_id}: not executed ({str(entry.get('result_summary', ''))[:200]})")
            continue
        estimate = entry.get("estimate")
        if isinstance(estimate, bool) or not isinstance(estimate, (int, float)) or not math.isfinite(estimate):
            problems.append(f"{check_id}: no computed numeric estimate")
        elif isinstance(estimate, float) and estimate not in (0.0, 1.0) and f"{estimate:.4f}" in code_text:
            problems.append(f"{check_id}: the estimate appears typed into the script")
        if not isinstance(entry.get("refuted"), bool):
            problems.append(f"{check_id}: refuted must be true or false")
        for field in ("ci_lower", "ci_upper", "p_value"):
            value = entry.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value)):
                problems.append(f"{check_id}: {field} is undefined; record executed false instead")
        claim_id, threat = threat_of[check_id]
        screen = (screens or {}).get(claim_id) or {}
        if screen.get("exposure_column") and threat == "common_cause":
            used = {str(v) for v in entry.get("covariates_used") or []}
            missing = [c for c in screen.get("required_covariates", []) if c not in used]
            if missing:
                problems.append(f"{check_id}: required covariates missing from covariates_used: {', '.join(missing)}")
        if screen.get("auc_rule_refutes") and threat == "timing_or_derivation" and entry.get("refuted") is not True:
            problems.append(f"{check_id}: the exposure discriminates the endpoint almost perfectly, so refuted must be true")
    if screens:
        for claim_id, screen in screens.items():
            threats = {threat for cid, (cl, threat) in threat_of.items() if cl == claim_id}
            for required in ("common_cause", "timing_or_derivation"):
                if screen.get("exposure_column") and required not in threats:
                    problems.append(f"{claim_id}: no {required} check declared")
    failed = {p.split(":")[0] for p in problems}
    return {"declared": declared, "verified": [c for c in declared if c not in failed], "problems": problems,
            "passed": not problems, "entries": entries}


def claim_check_report_block(plan: dict[str, Any], verification: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "# Verified Claim Checks",
            "Before writing, you declared the checks below and code executed them. Their results are binding. Present an "
            "association as a finding only if it belongs to a declared claim whose checks were all executed and none "
            "refuted it. Report the checked estimates alongside any crude estimates. Describe refuted or unverified "
            "claims as not supported by the checks.",
            "## Declared claims and checks",
            json.dumps(plan, indent=2, ensure_ascii=False),
            "## Executed check results",
            json.dumps(verification.get("entries", {}), indent=2, ensure_ascii=False),
            "## Checks that could not be verified",
            "\n".join(verification.get("problems", [])) or "none",
        ]
    )


def filter_stage_context_for_report(stage_context: dict[str, str]) -> dict[str, str]:
    allowed = {
        "idea_selection",
        "experiment_design",
        "results_selection",
    }
    filtered: dict[str, str] = {}
    for name, text in stage_context.items():
        if name.startswith(SUFFICIENCY_STAGE):
            continue
        if name in {"self_review", "revision"}:
            continue
        if name in allowed:
            filtered[name] = text
    return filtered


def build_claim_extraction_messages(run_dir: Path) -> list[dict[str, str]]:
    system = "You are extracting machine-readable empirical claims from a completed mini-paper."
    mini_paper_path = run_dir / "paper" / "mini_paper.md"
    mini_paper = mini_paper_path.read_text(encoding="utf-8") if mini_paper_path.exists() else ""
    canonical_variables = load_claim_extraction_variable_context(run_dir)
    user = "\n\n".join(
        [
            "# Claim Extraction Task",
            "Extract only explicit central empirical discovery claims made in the mini-paper.",
            "Use the mini-paper text below as the source document and the claim definitions to structure the extraction.",
            "A claim is a paper statement about a named exposure, predictor, index, score, covariate, or interaction and a named study outcome.",
            "Focus on the findings the paper asks a reader to take away in the Abstract, Results, or Discussion.",
            "Do not separately extract every robustness, sensitivity, or model-comparison sentence. Extract a robustness or sensitivity finding only when the paper uses it as a central conclusion or when it reverses a central conclusion.",
            "When the same exposure-outcome result is stated multiple times, extract one claim using the final or most adjusted presentation emphasized by the paper.",
            "Do not extract cohort description, missing-data description, model discrimination/performance, software/reproducibility statements, generic limitations, or background knowledge unless the sentence itself states an empirical exposure-outcome finding.",
            "Use the canonical study variable names below whenever the paper phrase maps to one of them. Remove units, contrasts, and scale text such as per SD increase from the exposure field when they are not part of the variable identity; place estimates and units in estimate when reported.",
            "Do not invent exposure names. If a statement cannot be mapped to a canonical feature variable or a clear interaction between canonical feature variables, do not extract it as a claim.",
            "Return JSON with this key:",
            "- claims: an array of claim objects. Each object must use exactly these fields:",
            json.dumps(CLAIM_TABLE_COLUMNS, indent=2),
            "If a field is not explicitly supported by the mini-paper, leave it blank, except survey_design_used.",
            "For the study endpoint, use the canonical outcome variable name below whenever the paper phrase maps to it.",
            "Set finding_result to one of these controlled values:",
            json.dumps(FINDING_RESULTS, indent=2),
            "Use null for claims that report no significant association, no independent association, or a null sensitivity/robustness result.",
            "Use positive or inverse only for directional findings that the paper presents as associated or significant.",
            "Use nonlinear for a reported nonlinearity test and interaction for a reported interaction/effect-modification result.",
            "Use independent_risk_factor only when the paper states that the exposure remains associated after adjustment, is an independent predictor/risk factor, or is estimated in an adjusted model as an independent association.",
            "Use predictive_association for unadjusted, screening, descriptive association, correlation, discrimination, or sample-prediction findings unless the paper clearly uses independent, direct, or causal language.",
            "Set survey_design_used separately for each claim: true only when the evidence for that specific claim explicitly uses survey weights, strata, PSU, or survey-weighted analysis; otherwise false.",
            "If the paper makes no explicit empirical claims, return an empty claims array.",
            "# Canonical Study Variables",
            json.dumps(canonical_variables, indent=2),
            "# Claim Table Field Definitions",
            json.dumps(CLAIM_FIELD_DEFINITIONS, indent=2),
            "# Allowed Claim Types",
            json.dumps({claim_type: CLAIM_TYPE_DEFINITIONS[claim_type] for claim_type in CLAIM_TYPES}, indent=2),
            "# Mini-paper",
            mini_paper,
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_claim_extraction_variable_context(run_dir: Path) -> dict[str, Any]:
    codebook_path = run_dir / "input" / "public_codebook.json"
    outcome = ""
    feature_names: list[str] = []
    if codebook_path.exists():
        try:
            codebook = json.loads(codebook_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            codebook = {}
        outcome = str(codebook.get("outcome", "") or "").strip()
        for item in codebook.get("variables", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("variable_name") or "").strip()
            if name:
                feature_names.append(name)
    if not feature_names:
        dataset_path = run_dir / "input" / "dataset.csv"
        if dataset_path.exists():
            try:
                with dataset_path.open("r", newline="", encoding="utf-8-sig") as handle:
                    reader = csv.reader(handle)
                    feature_names = [value.strip() for value in next(reader, []) if value.strip()]
            except (OSError, StopIteration):
                feature_names = []

    unique_features = sorted(dict.fromkeys(name for name in feature_names if name and name != outcome))
    return {
        "outcome": outcome,
        "feature_names": unique_features,
    }


def should_send_temperature(model: str, model_config: dict[str, Any]) -> bool:
    """Some reasoning models only accept the provider default temperature."""

    if "temperature" not in model_config:
        return False
    normalized = model.strip().lower()
    model_name = normalized.split("/")[-1]
    if model_name.startswith("gpt-5"):
        return False
    return True


def make_claim_extraction_request(
    model: str,
    messages: list[dict[str, str]],
    model_config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": strict_json_schema_response_format(
            "claim_extraction",
            CLAIM_EXTRACTION_JSON_SCHEMA,
        ),
    }
    if should_send_temperature(model, model_config):
        request["temperature"] = float(model_config.get("temperature", 0.0))
    for field in ["reasoning_effort", "prompt_cache_key", "prompt_cache_retention"]:
        value = model_config.get(field)
        if value not in {None, ""}:
            request[field] = value
    if model_config.get("include_metadata", False):
        request["metadata"] = {"project": "claim-extraction", "run_id": run_id}
    return request


def build_reviewer_messages(run_dir: Path, reviewer_style: str) -> list[dict[str, str]]:
    style = normalize_reviewer_style(reviewer_style)
    system = "You are a statistical and methods reviewer for an empirical mini-paper."
    manuscript = read_review_manuscript_text(run_dir)
    style_instruction = reviewer_style_instruction(style)
    output_schema = {
        "manuscript_title": "Title as written in the manuscript, or blank if absent.",
        "statistical_review": "One concise paragraph reviewing the statistical and methods support for the manuscript's conclusions.",
        "major_points": ["major statistical or methodological points"],
        "minor_points": ["minor statistical or reporting points"],
        "recommended_revisions": ["concrete statistical or reporting revisions"],
    }
    user = "\n\n".join(
        [
            "# Statistical Review Task",
            style_instruction,
            "Return JSON only, following the output schema.",
            "# Output Schema",
            json.dumps(output_schema, indent=2),
            "# Manuscript",
            manuscript,
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def read_review_manuscript_text(run_dir: Path) -> str:
    review_path = run_dir / "paper" / "review_manuscript.md"
    if review_path.exists():
        return review_path.read_text(encoding="utf-8", errors="replace")
    mini_paper_path = run_dir / "paper" / "mini_paper.md"
    return mini_paper_path.read_text(encoding="utf-8", errors="replace") if mini_paper_path.exists() else ""


def reviewer_style_instruction(style: str) -> str:
    if style == "statistical":
        return (
            "Review the manuscript's model choice, adjustment, survey design, missingness handling, robustness "
            "checks, and whether the statistical evidence supports the claim language."
        )
    if style == "journal":
        return (
            "Review as a journal methods reviewer, emphasizing statistical support for conclusions and clarity "
            "of the empirical evidence."
        )
    return (
        "Review as a methods reviewer, focusing on the empirical analysis and the statistical support for the paper's conclusions."
    )


def make_reviewer_request(
    model: str,
    messages: list[dict[str, str]],
    model_config: dict[str, Any],
    run_id: str,
    reviewer_style: str,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": strict_json_schema_response_format("paper_review", REVIEWER_JSON_SCHEMA),
    }
    if should_send_temperature(model, model_config):
        request["temperature"] = float(model_config.get("temperature", 0.2))
    for field in ["reasoning_effort", "prompt_cache_key", "prompt_cache_retention"]:
        value = model_config.get(field)
        if value not in {None, ""}:
            request[field] = value
    if model_config.get("include_metadata", False):
        request["metadata"] = {
            "project": "external-review",
            "run_id": run_id,
            "reviewer_style": reviewer_style,
            "loop_step_index": "0",
        }
    return request


def parse_reviewer_response(text: str, reviewer_style: str) -> dict[str, Any]:
    decoded: Any = None
    parse_status = "empty_response"
    if text.strip():
        try:
            decoded = json.loads(text)
            parse_status = "json"
        except json.JSONDecodeError:
            fenced = extract_fenced(text, "json")
            if fenced:
                try:
                    decoded = json.loads(fenced)
                    parse_status = "fenced_json"
                except json.JSONDecodeError:
                    decoded = None
                    parse_status = "unparseable"
            else:
                parse_status = "unparseable"
    if not isinstance(decoded, dict):
        decoded = {}
    return {
        "schema_version": "0.2",
        "reviewer_style": normalize_reviewer_style(reviewer_style),
        "parse_status": parse_status,
        "manuscript_title": str(decoded.get("manuscript_title", "")),
        "statistical_review": str(decoded.get("statistical_review", "")),
        "major_points": normalize_string_list(decoded.get("major_points")),
        "minor_points": normalize_string_list(decoded.get("minor_points")),
        "recommended_revisions": normalize_string_list(decoded.get("recommended_revisions")),
    }


def normalize_reviewer_style(style: str) -> str:
    normalized = style.strip().lower().replace("-", "_")
    if normalized not in REVIEWER_STYLES:
        raise ValueError(f"Unsupported reviewer style: {style}. Expected one of {sorted(REVIEWER_STYLES)}.")
    return normalized


def normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def run_chat_step(
    run_dir: Path,
    request_payload: dict[str, Any],
    artifact_stem: str,
    client: OpenRouterClient | OpenAIChatClient | FixtureOpenRouterClient | None,
    provider_config: dict[str, Any],
    dry_run: bool,
    skipped_message: str,
) -> ChatStepResult:
    request_path = run_dir / "request" / f"{artifact_stem}_request.json"
    response_path = run_dir / "response" / f"{artifact_stem}_response.json"
    message_path = run_dir / "response" / f"{artifact_stem}_message.md"
    write_json(request_path, request_payload)

    response: dict[str, Any] = {"dry_run": True, "message": skipped_message}
    text = ""
    error = None
    if not dry_run:
        if client is None:
            raise RuntimeError("Client is required for non-dry interactive runs.")
        try:
            response = client.chat_completion(
                request_payload,
                endpoint=str(provider_config.get("chat_endpoint", "/chat/completions")),
            )
            text = extract_assistant_text(response)
        except Exception as exc:  # noqa: BLE001 - persisted artifact should capture provider failures
            error = str(exc)
            response = {"error": error}

    write_json(response_path, response)
    message_path.write_text(text, encoding="utf-8")
    return ChatStepResult(
        request_path=request_path,
        response_path=response_path,
        message_path=message_path,
        response=response,
        text=text,
        error=error,
    )


def step_manifest(
    step_name: str,
    step: ChatStepResult,
    step_index: int,
    attempt: int | None = None,
) -> dict[str, Any]:
    run_dir = step.request_path.parent.parent
    payload: dict[str, Any] = {
        "step": step_name,
        "step_index": step_index,
        "request": str(step.request_path.relative_to(run_dir)).replace("\\", "/"),
        "response": str(step.response_path.relative_to(run_dir)).replace("\\", "/"),
        "message": str(step.message_path.relative_to(run_dir)).replace("\\", "/"),
        "error": step.error or "",
    }
    if attempt is not None:
        payload["attempt"] = attempt
    usage = step.response.get("usage") if isinstance(step.response, dict) else None
    if usage:
        payload["usage"] = usage
    return payload


def persist_execution_attempt(run_dir: Path, execution: ExecutionResult, label: str) -> None:
    safe_label = sanitize_slug(label)
    stdout_copy = run_dir / "logs" / f"{safe_label}_stdout.txt"
    stderr_copy = run_dir / "logs" / f"{safe_label}_stderr.txt"
    if execution.stdout_path.exists():
        shutil.copy2(execution.stdout_path, stdout_copy)
    if execution.stderr_path.exists():
        shutil.copy2(execution.stderr_path, stderr_copy)
    write_json(
        run_dir / "metadata" / f"{safe_label}_manifest.json",
        {
            "returncode": execution.returncode,
            "duration_seconds": execution.duration_seconds,
            "generated_files": execution.generated_files,
            "stdout": str(stdout_copy.relative_to(run_dir)).replace("\\", "/"),
            "stderr": str(stderr_copy.relative_to(run_dir)).replace("\\", "/"),
        },
    )


def save_analysis_sufficiency_review(run_dir: Path, review_text: str, artifact_stem: str = SUFFICIENCY_STAGE) -> None:
    if not review_text.strip():
        return
    try:
        decoded = json.loads(review_text)
    except json.JSONDecodeError:
        (run_dir / "metadata" / f"{artifact_stem}.txt").write_text(review_text, encoding="utf-8")
        return

    path = run_dir / "metadata" / f"{artifact_stem}.json"
    path.write_text(json.dumps(decoded, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_evidence_audit(run_dir: Path, audit_text: str) -> None:
    if not audit_text.strip():
        return
    try:
        decoded = json.loads(audit_text)
    except json.JSONDecodeError:
        (run_dir / "metadata" / f"{SUFFICIENCY_STAGE}.txt").write_text(audit_text, encoding="utf-8")
        return

    path = run_dir / "metadata" / f"{SUFFICIENCY_STAGE}.json"
    path.write_text(json.dumps(decoded, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_stage_json_object(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        decoded = json.loads(text)
        return decoded if isinstance(decoded, dict) else {}
    except json.JSONDecodeError:
        fenced = extract_fenced(text, "json")
        if fenced.strip():
            try:
                decoded = json.loads(fenced)
                return decoded if isinstance(decoded, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def stage_requests_supplemental_analysis(text: str) -> bool:
    payload = parse_stage_json_object(text)
    if not payload:
        return False
    objective_met = payload.get("objective_met")
    if isinstance(objective_met, bool) and not objective_met:
        return True
    if payload.get("analysis_revision_required") is True:
        return True
    for key in [
        "supplementary_analysis_needed",
        "supplemental_analysis_needed",
        "additional_analysis_needed",
        "analysis_changes_needed",
    ]:
        value = payload.get(key)
        if isinstance(value, list) and len(value) > 0:
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False


def strict_json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def is_code_json_loop_step(loop_step: str) -> bool:
    return (
        loop_step == "code_generation"
        or loop_step.startswith("supplemental_code_")
        or "_repair_" in loop_step
    )


def response_format_for_loop_step(loop_step: str) -> dict[str, Any]:
    if is_code_json_loop_step(loop_step):
        return strict_json_schema_response_format("analysis_code", CODE_JSON_SCHEMA)
    if loop_step.startswith(SUFFICIENCY_STAGE):
        return strict_json_schema_response_format("analysis_sufficiency_review", ANALYSIS_SUFFICIENCY_JSON_SCHEMA)
    stage_schema = AUTHOR_STAGE_JSON_SCHEMAS.get(loop_step)
    if stage_schema is not None:
        return strict_json_schema_response_format(loop_step, stage_schema)
    return {"type": "json_object"}


def make_chat_request(
    model: str,
    messages: list[dict[str, str]],
    model_config: dict[str, Any],
    run_id: str,
    run_seed: int,
    loop_step: str,
    loop_step_index: int,
    dataset_id: str,
    author_paradigm: str = "",
    json_response: bool = True,
) -> dict[str, Any]:
    metadata = {
        "project": "cardiometabolic-risk-study",
        "run_id": run_id,
        "dataset_id": dataset_id,
        "mode": "interactive_agent_loop",
        "loop_step": loop_step,
        "loop_step_index": str(loop_step_index),
    }
    if author_paradigm:
        metadata["author_paradigm"] = normalize_author_paradigm(author_paradigm)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(model_config.get("temperature", 0.4)),
        "seed": run_seed + loop_step_index,
        "metadata": metadata,
    }
    if json_response:
        payload["response_format"] = response_format_for_loop_step(loop_step)
    return payload


def parse_code_response(text: str) -> str:
    if not text.strip():
        return ""
    try:
        decoded = json.loads(text)
        code = code_from_mapping(decoded)
        if code:
            return code
    except json.JSONDecodeError:
        pass
    fenced_json = extract_fenced(text, "json")
    if fenced_json.strip():
        try:
            decoded = json.loads(fenced_json)
            code = code_from_mapping(decoded)
            if code:
                return code
        except json.JSONDecodeError:
            pass
    fenced_python = extract_fenced(text, "python")
    if fenced_python.strip():
        return fenced_python
    return extract_jsonish_string_field(text, "analysis_code")


def code_from_mapping(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    for key in ["analysis_code", "code", "python_code"]:
        code = value.get(key)
        if isinstance(code, str) and code.strip():
            return code
    return ""


def extract_jsonish_string_field(text: str, field_name: str) -> str:
    """Recover code from malformed model JSON with unescaped quotes."""

    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"', text)
    if not match:
        return ""
    tail = text[match.end() :]
    end_matches = list(re.finditer(r'"\s*}\s*$', tail, flags=re.DOTALL))
    if not end_matches:
        end_matches = list(
            re.finditer(r'"\s*,\s*"[A-Za-z_][A-Za-z0-9_]*"\s*:', tail, flags=re.DOTALL)
        )
    if not end_matches:
        return ""
    raw = tail[: end_matches[-1].start()]
    return decode_jsonish_string_payload(raw)


def decode_jsonish_string_payload(raw: str) -> str:
    return (
        raw.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\/", "/")
        .replace("\\\\", "\\")
    )


def execute_generated_code(
    run_dir: Path,
    code_path: Path,
    timeout_seconds: float = DEFAULT_CODE_TIMEOUT_SECONDS,
) -> ExecutionResult:
    before = snapshot_files(run_dir)
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            ["python", str(code_path.resolve())],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=sanitized_execution_env(),
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = decode_subprocess_stream(exc.stdout)
        stderr = decode_subprocess_stream(exc.stderr)
        timeout_message = f"Execution timed out after {timeout_seconds:.1f} seconds."
        stderr = f"{stderr.rstrip()}\n{timeout_message}\n" if stderr.strip() else timeout_message + "\n"
    duration = time.monotonic() - started
    stdout_path = run_dir / "logs" / "stdout.txt"
    stderr_path = run_dir / "logs" / "stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    after = snapshot_files(run_dir)
    generated = sorted(after - before)
    write_json(
        run_dir / "metadata" / "execution_manifest.json",
        {
            "returncode": returncode,
            "duration_seconds": duration,
            "generated_files": generated,
            "timed_out": timed_out,
        },
    )
    return ExecutionResult(
        returncode=returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        duration_seconds=duration,
        generated_files=generated,
    )


def decode_subprocess_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def sanitized_execution_env() -> dict[str, str]:
    """Run generated scripts without inheriting API keys or other secrets."""

    denied_fragments = ["KEY", "TOKEN", "SECRET", "PASSWORD", "OPENROUTER", "ANTHROPIC", "OPENAI"]
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if not any(fragment in key.upper() for fragment in denied_fragments)
    }
    clean_env.setdefault("PYTHONUTF8", "1")
    return clean_env


def empty_execution(run_dir: Path, reason: str = "no_analysis_code") -> ExecutionResult:
    stdout_path = run_dir / "logs" / "stdout.txt"
    stderr_path = run_dir / "logs" / "stderr.txt"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text(f"{reason}; execution skipped.", encoding="utf-8")
    write_json(
        run_dir / "metadata" / "execution_manifest.json",
        {
            "returncode": 1,
            "duration_seconds": 0.0,
            "generated_files": [],
            "reason": reason,
        },
    )
    return ExecutionResult(
        returncode=1,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        duration_seconds=0.0,
        generated_files=[],
    )


def aggregate_analysis_execution_result(run_dir: Path, latest: ExecutionResult) -> ExecutionResult:
    generated_files = sorted(
        path for path in snapshot_files(run_dir) if path.startswith("intermediate/") or path.startswith("plots/")
    )
    return ExecutionResult(
        returncode=latest.returncode,
        stdout_path=latest.stdout_path,
        stderr_path=latest.stderr_path,
        duration_seconds=latest.duration_seconds,
        generated_files=generated_files,
    )


def write_aggregate_execution_manifest(run_dir: Path, execution: ExecutionResult) -> None:
    write_json(
        run_dir / "metadata" / "execution_manifest.json",
        {
            "returncode": execution.returncode,
            "duration_seconds": execution.duration_seconds,
            "generated_files": execution.generated_files,
            "aggregate_analysis_outputs": True,
        },
    )


def summarize_execution_for_prompt(run_dir: Path, execution: ExecutionResult) -> str:
    chunks = [
        f"returncode: {execution.returncode}",
        f"duration_seconds: {execution.duration_seconds:.3f}",
        "# stdout",
        truncate_text(execution.stdout_path.read_text(encoding="utf-8"), 12000),
        "# stderr",
        truncate_text(execution.stderr_path.read_text(encoding="utf-8"), 8000),
        "# generated_files",
        "\n".join(execution.generated_files[:200]),
    ]
    for relative_path in execution.generated_files:
        if not relative_path.startswith("intermediate/"):
            continue
        path = run_dir / relative_path
        if path.suffix.lower() in {".json", ".csv", ".txt", ".md"} and path.exists():
            chunks.extend(
                [
                    f"# file: {relative_path}",
                    truncate_text(path.read_text(encoding="utf-8", errors="replace"), 12000),
                ]
            )
    return "\n\n".join(chunks)


def summarize_execution_for_manuscript(run_dir: Path, execution: ExecutionResult) -> str:
    compact_summaries = extract_compact_analysis_summaries(run_dir, execution)
    artifacts = build_manuscript_artifact_inventory(run_dir, execution)
    return "\n\n".join(
        [
            "# compact_analysis_summaries",
            json.dumps(compact_summaries, indent=2, ensure_ascii=False),
            "# generated_artifact_inventory",
            json.dumps(artifacts, indent=2, ensure_ascii=False),
        ]
    )


def extract_compact_analysis_summaries(run_dir: Path, execution: ExecutionResult) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for relative_path in sorted(execution.generated_files):
        if not relative_path.startswith("intermediate/") or not relative_path.endswith(".json"):
            continue
        path = run_dir / relative_path
        if not path.exists():
            continue
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        compact = compact_summary_from_json(decoded)
        if compact:
            summaries.append({"filename": relative_path, **compact})
    return summaries


def compact_summary_from_json(decoded: Any) -> dict[str, Any]:
    if not isinstance(decoded, dict):
        return {}
    compact = {key: decoded[key] for key in COMPACT_SUMMARY_KEYS if key in decoded}
    legacy_main = decoded.get("main_result")
    if "primary_results" not in compact and legacy_main:
        compact["primary_results"] = [legacy_main]
    legacy_main_results = decoded.get("main_results")
    if "primary_results" not in compact and legacy_main_results:
        compact["primary_results"] = legacy_main_results
    if "secondary_results" not in compact and "results" in decoded:
        compact["secondary_results"] = decoded["results"]
    if "generated_tables" not in compact and "tables" in decoded:
        compact["generated_tables"] = decoded["tables"]
    if "generated_figures" not in compact and "figures" in decoded:
        compact["generated_figures"] = decoded["figures"]
    if "generated_figures" not in compact and "plots" in decoded:
        compact["generated_figures"] = decoded["plots"]
    return compact


def write_review_manuscript(run_dir: Path, manuscript_markdown: str, execution: ExecutionResult) -> Path:
    review_path = run_dir / "paper" / "review_manuscript.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    inventory = filter_artifact_inventory_for_manuscript(
        manuscript_markdown,
        build_manuscript_artifact_inventory(run_dir, execution),
    )
    rendered_sections = [manuscript_markdown.strip()]
    rendered_sections.append(render_artifact_appendix_for_review(inventory))
    review_path.write_text("\n\n".join(section for section in rendered_sections if section.strip()) + "\n", encoding="utf-8")
    return review_path


def filter_artifact_inventory_for_manuscript(
    manuscript_markdown: str,
    inventory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manuscript_text = manuscript_markdown.replace("\\", "/")
    included: list[dict[str, Any]] = []
    for artifact in inventory:
        artifact_type = str(artifact.get("artifact_type", ""))
        if artifact_type not in {"table", "figure"}:
            continue
        filename = str(artifact.get("filename", "")).replace("\\", "/")
        paper_relative_path = str(artifact.get("paper_relative_path", "")).replace("\\", "/")
        basename = Path(filename).name
        candidates = [value for value in [filename, paper_relative_path, basename] if value]
        if any(candidate in manuscript_text for candidate in candidates):
            included.append(artifact)
    return included


def render_artifact_appendix_for_review(inventory: list[dict[str, Any]]) -> str:
    if not inventory:
        return ""
    lines = [
        "## Referenced Analysis Artifacts",
        "",
        "The following generated tables and figures were explicitly referenced in the manuscript.",
    ]
    for index, artifact in enumerate(inventory, start=1):
        artifact_type = str(artifact.get("artifact_type", "artifact"))
        title = str(artifact.get("title", f"Artifact {index}"))
        filename = str(artifact.get("filename", ""))
        summary = str(artifact.get("summary", ""))
        content = str(artifact.get("content_for_writing", ""))
        truncated = bool(artifact.get("content_truncated", False))
        lines.extend(["", f"### {title}", "", f"- Type: `{artifact_type}`", f"- File: `{filename}`"])
        if summary:
            lines.extend(["", summary])
        if artifact_type == "figure" and content:
            lines.extend(["", content])
        elif content:
            lines.extend(["", content])
        if truncated:
            lines.extend(["", "_This artifact was truncated for the review manuscript._"])
    return "\n".join(lines)


def build_manuscript_artifact_inventory(run_dir: Path, execution: ExecutionResult) -> list[dict[str, Any]]:
    relevant_files = sorted(
        path for path in execution.generated_files if path.startswith("intermediate/") or path.startswith("plots/")
    )
    intermediate_texts = load_intermediate_texts(run_dir, relevant_files)
    artifacts: list[dict[str, Any]] = []
    for relative_path in relevant_files:
        path = run_dir / relative_path
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        paper_relative_path = str(Path("..") / Path(relative_path)).replace("\\", "/")
        if relative_path.startswith("plots/"):
            artifact_type = "figure"
            title = artifact_title_from_filename(relative_path, artifact_type)
            summary = summarize_figure_artifact(relative_path, intermediate_texts)
            content_for_writing = f"![{title}]({paper_relative_path})"
            content_truncated = False
        elif suffix in {".csv", ".tsv"}:
            artifact_type = "table"
            title = artifact_title_from_filename(relative_path, artifact_type)
            table = summarize_delimited_table(path, delimiter="\t" if suffix == ".tsv" else ",")
            summary = table["summary"]
            content_for_writing = table["markdown_table"]
            content_truncated = table["truncated"]
        elif suffix == ".json" and json_file_decodes(path):
            artifact_type = "result_summary"
            decoded = load_json_if_exists(path)
            title = artifact_title_from_json(decoded, relative_path) or artifact_title_from_filename(
                relative_path,
                artifact_type,
            )
            summary = summarize_json_for_manuscript(decoded)
            content_for_writing = truncate_text(
                json.dumps(decoded, indent=2, ensure_ascii=False, sort_keys=True),
                MANUSCRIPT_TEXT_ARTIFACT_LIMIT,
            )
            content_truncated = len(json.dumps(decoded, ensure_ascii=False, sort_keys=True)) > MANUSCRIPT_TEXT_ARTIFACT_LIMIT
        elif suffix in {".txt", ".md", ".json"}:
            # A .json file reaches this branch only if it cannot be parsed (for example, a script that failed while
            # writing it), so its raw text is passed on instead of stopping the run.
            artifact_type = "text_output"
            title = artifact_title_from_filename(relative_path, artifact_type)
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            summary = truncate_text(text, 1200)
            content_for_writing = truncate_text(text, MANUSCRIPT_TEXT_ARTIFACT_LIMIT)
            content_truncated = len(text) > MANUSCRIPT_TEXT_ARTIFACT_LIMIT
        else:
            artifact_type = "artifact"
            title = artifact_title_from_filename(relative_path, artifact_type)
            summary = "Generated analysis artifact."
            content_for_writing = ""
            content_truncated = False
        artifacts.append(
            {
                "artifact_type": artifact_type,
                "title": title,
                "filename": relative_path,
                "paper_relative_path": paper_relative_path,
                "summary": summary,
                "content_for_writing": content_for_writing,
                "content_truncated": content_truncated,
            }
        )
    return artifacts


def load_intermediate_texts(run_dir: Path, relative_paths: list[str]) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    for relative_path in relative_paths:
        if not relative_path.startswith("intermediate/"):
            continue
        path = run_dir / relative_path
        if path.suffix.lower() not in {".json", ".csv", ".tsv", ".txt", ".md"} or not path.exists():
            continue
        texts.append((relative_path, path.read_text(encoding="utf-8", errors="replace")[:30000]))
    return texts


def artifact_title_from_filename(relative_path: str, artifact_type: str) -> str:
    stem = Path(relative_path).stem.replace("_", " ").replace("-", " ").strip()
    title = " ".join(part.capitalize() for part in stem.split()) or Path(relative_path).name
    if artifact_type == "figure":
        return f"Figure: {title}"
    if artifact_type == "table":
        return f"Table: {title}"
    return title


def artifact_title_from_json(decoded: dict[str, Any], relative_path: str) -> str:
    if not isinstance(decoded, dict):
        return ""
    for key in ["title", "caption", "name", "description"]:
        value = decoded.get(key)
        if isinstance(value, str) and value.strip():
            return truncate_text(one_line(value.strip()), 160)
    return ""


def summarize_json_for_manuscript(decoded: Any) -> str:
    if not decoded:
        return "Empty JSON result file."
    if isinstance(decoded, dict):
        preferred_keys = [
            "primary_results",
            "secondary_results",
            "robustness_results",
            "generated_tables",
            "generated_figures",
            "limitations",
            "summary",
            "main_result",
            "main_results",
            "findings",
            "results",
            "rows",
            "tables",
            "figures",
            "plots",
        ]
        selected = {key: decoded[key] for key in preferred_keys if key in decoded}
        if selected:
            return truncate_text(json.dumps(selected, ensure_ascii=False, sort_keys=True), 1800)
    return truncate_text(json.dumps(decoded, ensure_ascii=False, sort_keys=True), 1800)


def summarize_delimited_table(path: Path, delimiter: str) -> dict[str, Any]:
    csv.field_size_limit(max(csv.field_size_limit(), 10_000_000))
    header: list[str] = []
    included_rows: list[list[str]] = []
    data_row_count = 0
    parse_error = ""
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            for row_index, row in enumerate(reader):
                clipped_row = [truncate_text(cell, 1000) for cell in row]
                if row_index == 0:
                    header = clipped_row
                    continue
                data_row_count += 1
                if len(included_rows) < MANUSCRIPT_TABLE_MAX_ROWS:
                    included_rows.append(clipped_row)
        except csv.Error as exc:
            parse_error = str(exc)
    if not header and not included_rows:
        return {"summary": "Empty table.", "markdown_table": "", "truncated": False}
    truncated = data_row_count > len(included_rows) or bool(parse_error)
    summary = truncate_text(
        json.dumps(
            {
                "columns": header,
                "row_count": data_row_count,
                "included_row_count": len(included_rows),
                "all_rows_included": not truncated,
                "preview_rows": included_rows[:10],
                "parse_error": parse_error,
            },
            ensure_ascii=False,
        ),
        3000,
    )
    return {
        "summary": summary,
        "markdown_table": markdown_table(header, included_rows, truncated=truncated),
        "truncated": truncated,
    }


def markdown_table(header: list[str], rows: list[list[str]], truncated: bool = False) -> str:
    if not header:
        return ""
    normalized_rows = [normalize_table_row(row, len(header)) for row in rows]
    lines = [
        "| " + " | ".join(escape_markdown_table_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in normalized_rows:
        lines.append("| " + " | ".join(escape_markdown_table_cell(cell) for cell in row) + " |")
    if truncated:
        lines.append(f"| {' | '.join(['...'] * len(header))} |")
    return "\n".join(lines)


def normalize_table_row(row: list[str], width: int) -> list[str]:
    normalized = [str(cell) for cell in row[:width]]
    if len(normalized) < width:
        normalized.extend([""] * (width - len(normalized)))
    return normalized


def escape_markdown_table_cell(value: str) -> str:
    return one_line(str(value)).replace("|", "\\|")


def summarize_figure_artifact(relative_path: str, intermediate_texts: list[tuple[str, str]]) -> str:
    basename = Path(relative_path).name
    for source, text in intermediate_texts:
        if not source.endswith(".json"):
            continue
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            continue
        structured = find_artifact_reference_summary(decoded, {relative_path, basename})
        if structured:
            return f"Referenced in {source}: {structured}"
    return "Generated figure file."


def find_artifact_reference_summary(value: Any, filenames: set[str]) -> str:
    if isinstance(value, dict):
        filename = str(
            value.get("filename")
            or value.get("file")
            or value.get("path")
            or value.get("paper_relative_path")
            or ""
        ).replace("\\", "/")
        basename = Path(filename).name if filename else ""
        if filename in filenames or basename in filenames:
            parts = []
            for key in ["title", "caption", "summary", "interpretation", "description"]:
                text = str(value.get(key, "") or "").strip()
                if text:
                    parts.append(one_line(text))
            if parts:
                return truncate_text("; ".join(dict.fromkeys(parts)), 650)
            return "Referenced by an intermediate output."
        for child in value.values():
            summary = find_artifact_reference_summary(child, filenames)
            if summary:
                return summary
    elif isinstance(value, list):
        for child in value:
            summary = find_artifact_reference_summary(child, filenames)
            if summary:
                return summary
    return ""


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def snapshot_files(run_dir: Path) -> set[str]:
    result: set[str] = set()
    for path in run_dir.rglob("*"):
        if path.is_file():
            result.add(str(path.relative_to(run_dir)).replace("\\", "/"))
    return result


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def make_fixture_response(content: str, response_id: str) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def fixture_responses_for_paradigm(author_paradigm: str) -> list[dict[str, Any]]:
    paradigm = normalize_author_paradigm(author_paradigm)
    if paradigm == "single_shot":
        contents = [
            ("fixture-code", fixture_analysis_code()),
            ("fixture-report", fixture_report_json()),
        ]
    elif paradigm == "iterative_coding":
        contents = [
            ("fixture-analysis-plan", fixture_stage_json("analysis_plan")),
            ("fixture-code", fixture_analysis_code()),
            ("fixture-sufficiency-review", fixture_analysis_sufficiency_review_json()),
            ("fixture-report", fixture_report_json()),
        ]
    elif paradigm == "data_to_paper":
        contents = [
            ("fixture-research-plan", fixture_stage_json("research_plan")),
            ("fixture-preprocessing-plan", fixture_stage_json("preprocessing_plan")),
            ("fixture-code", fixture_analysis_code()),
            ("fixture-sufficiency-review", fixture_analysis_sufficiency_review_json()),
            ("fixture-results-selection", fixture_stage_json("results_selection")),
            ("fixture-report", fixture_report_json()),
        ]
    elif paradigm == "ai_scientist":
        contents = [
            ("fixture-idea-generation", fixture_stage_json("idea_generation")),
            ("fixture-idea-selection", fixture_stage_json("idea_selection")),
            ("fixture-experiment-design", fixture_stage_json("experiment_design")),
            ("fixture-code", fixture_analysis_code()),
            ("fixture-sufficiency-review", fixture_analysis_sufficiency_review_json()),
            ("fixture-self-review", fixture_stage_json("self_review")),
            ("fixture-revision", fixture_stage_json("revision")),
            ("fixture-report", fixture_report_json()),
        ]
    else:
        raise AssertionError(f"Unhandled author paradigm: {paradigm}")
    return [make_fixture_response(content, response_id=response_id) for response_id, content in contents]


def fixture_stage_json(stage_name: str) -> str:
    payloads: dict[str, dict[str, Any]] = {
        "analysis_plan": {
            "analysis_plan": "Fixture plan: screen candidate predictors and write reproducible outputs.",
            "planned_outputs": ["intermediate/analysis_results.json", "intermediate/screening_results.csv"],
        },
        "research_plan": {
            "research_question": "Which measured variables are empirically associated with the endpoint?",
            "research_plan": "Fixture research plan for a cardiometabolic endpoint.",
            "planned_outputs": ["intermediate/analysis_results.json", "intermediate/screening_results.csv"],
        },
        "preprocessing_plan": {
            "preprocessing_plan": "Fixture preprocessing plan: use complete pairs per exposure.",
            "planned_outputs": ["intermediate/analysis_results.json", "intermediate/screening_results.csv"],
        },
        "results_selection": {
            "main_results": ["BMI screening association"],
            "secondary_results": [],
            "results_not_emphasized": [],
            "selection_reason": "Fixture selected one available result.",
        },
        "idea_generation": {
            "proposed_ideas": [
                "BMI and risk endpoint",
                "Age and risk endpoint",
                "Survey-weighted descriptive analysis",
                "Laboratory markers and risk endpoint",
                "Lifestyle variables and risk endpoint",
            ],
            "screening_rationale": "Compare ideas by available variables, endpoint clarity, and reproducible outputs.",
        },
        "idea_selection": {
            "selected_ideas": ["BMI and risk endpoint"],
            "abandoned_ideas": ["Age and risk endpoint", "Survey-weighted descriptive analysis"],
            "selection_reason": "Fixture selected the simplest executable idea.",
        },
        "experiment_design": {
            "experiment_design": "Fixture design: run simple screening correlations.",
            "planned_outputs": ["intermediate/screening_results.csv"],
        },
        "self_review": {
            "self_review_score": 6,
            "strengths": ["Executable fixture analysis"],
            "concerns": ["Minimal model and no real robustness checks"],
            "revision_plan": "Keep final paper cautious.",
        },
        "revision": {
            "revision_plan": "Fixture revision keeps the manuscript cautious.",
            "supplementary_analysis_needed": [],
            "analysis_revision_required": False,
        },
    }
    return json.dumps(payloads.get(stage_name, {"stage": stage_name}))


def fixture_analysis_code() -> str:
    code = r'''
from pathlib import Path
import json
import pandas as pd

root = Path(".")
for name in ["intermediate", "plots", "paper"]:
    (root / name).mkdir(exist_ok=True)

df = pd.read_csv(root / "input" / "dataset.csv")
codebook = json.loads((root / "input" / "public_codebook.json").read_text(encoding="utf-8"))
outcome = codebook.get("outcome", "cardiometabolic_risk_status")
rows = []
for exposure in [col for col in df.columns if col not in ["participant_id", outcome]][:5]:
    pair = df[[exposure, outcome]].dropna()
    if len(pair) < 3:
        continue
    corr = float(pair[exposure].corr(pair[outcome]))
    rows.append(
        {
            "exposure": exposure,
            "outcome": outcome,
            "model_type": "pearson_screen",
            "estimate": corr,
            "p_value": "",
            "n": int(len(pair)),
        }
    )

results = {
    "primary_results": [
        {
            "exposure": rows[0]["exposure"] if rows else "",
            "outcome": outcome,
            "model": "pearson_screen",
            "estimate": rows[0]["estimate"] if rows else "",
            "p_value": "",
            "direction": "screening correlation",
            "interpretation": "Fixture screening result for manuscript generation.",
            "source": "intermediate/screening_results.csv",
        }
    ] if rows else [],
    "secondary_results": [],
    "robustness_results": [],
    "generated_tables": [
        {
            "filename": "intermediate/screening_results.csv",
            "title": "Screening correlations",
            "interpretation": "Candidate predictor correlations with the endpoint.",
        }
    ],
    "generated_figures": [],
    "limitations": ["Fixture analysis is intentionally minimal."],
    "rows": rows,
}
(root / "intermediate" / "analysis_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
pd.DataFrame(rows).to_csv(root / "intermediate" / "screening_results.csv", index=False)
print(json.dumps({"n_rows": len(df), "n_results": len(rows)}))
'''
    return json.dumps({"analysis_code": code})


def fixture_analysis_sufficiency_review_json() -> str:
    return json.dumps(
        {
            "objective_met": True,
            "supplementary_analysis_needed": [],
        }
    )


def fixture_evidence_audit_json() -> str:
    return fixture_analysis_sufficiency_review_json()


def fixture_report_json() -> str:
    return "\n\n".join(
        [
            "# Fixture Mini-paper",
            "## Abstract",
            "The fixture analysis screened candidate predictors for association with the cardiometabolic risk endpoint.",
            "## Methods",
            "A reproducible fixture script computed simple screening correlations.",
            "## Results",
            "The screening table (`intermediate/screening_results.csv`) showed that BMI was associated with the cardiometabolic risk endpoint.",
            "## Discussion",
            "The fixture result is intentionally minimal.",
            "## Limitations",
            "The fixture analysis is for harness testing only.",
        ]
    )


def fixture_claim_extraction_json() -> str:
    return json.dumps(
        {
            "claims": [
                {
                    "claim_id": "fixture_001",
                    "exposure": "bmi",
                    "outcome": DEFAULT_OUTCOME,
                    "claim_type": "predictive_association",
                    "finding_result": "positive",
                    "model_type": "pearson_screen",
                    "covariates": "",
                    "estimate": "0.0",
                    "p_value": "",
                    "survey_design_used": False,
                    "evidence_source": "intermediate/screening_results.csv",
                }
            ]
        }
    )


def fixture_reviewer_json(style: str = "generic") -> str:
    del style
    return json.dumps(
        {
            "manuscript_title": "Fixture Mini-paper",
            "statistical_review": "The fixture analysis is executable but statistically minimal.",
            "major_points": ["The analysis relies on simple screening correlations."],
            "minor_points": ["Report the sample size for each exposure."],
            "recommended_revisions": ["Add an adjusted model or sensitivity analysis before strong claims."],
        }
    )


def fixture_selection_review_json() -> str:
    return json.dumps(
        {
            "ranked_papers": [
                {
                    "rank": 1,
                    "paper_id": "paper_001",
                    "title": "Fixture Mini-paper",
                    "selection_reason": "Clear title and abstract for the fixture ranking.",
                }
            ]
        }
    )


def ensure_run_tree(run_dir: Path) -> None:
    for subdir in RUN_SUBDIRS:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)


def read_dataset_preview(dataset_path: Path, rows: int) -> str:
    with dataset_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        selected = []
        for index, row in enumerate(reader):
            if index > rows:
                break
            selected.append(row)
    return "\n".join(",".join(_csv_quote(cell) for cell in row) for row in selected)


def parse_agent_response(text: str) -> dict[str, str]:
    """Parse the structured JSON response, with conservative fallbacks."""

    if not text.strip():
        return {"analysis_code": "", "mini_paper_markdown": "", "notes": ""}
    try:
        decoded = json.loads(text)
        if isinstance(decoded, dict):
            return {key: str(decoded.get(key, "")) for key in ["analysis_code", "mini_paper_markdown", "notes"]}
    except json.JSONDecodeError:
        pass
    return {
        "analysis_code": extract_fenced(text, "python"),
        "mini_paper_markdown": text,
        "notes": "Response was not valid JSON; fallback parser was used.",
    }


def parse_report_response(text: str) -> str:
    """Return manuscript Markdown from either the new direct format or old JSON."""

    if not text.strip():
        return ""
    try:
        decoded = json.loads(text)
        if isinstance(decoded, dict) and "mini_paper_markdown" in decoded:
            return str(decoded.get("mini_paper_markdown", ""))
    except json.JSONDecodeError:
        pass
    return text


def load_prompt_payload_from_run(run_dir: Path) -> dict[str, Any]:
    task_path = run_dir / "prompt" / "task_prompt.md"
    output_contract_path = run_dir / "prompt" / "output_contract.md"
    dataset_preview_path = run_dir / "prompt" / "dataset_preview.md"
    messages_path = run_dir / "prompt" / "messages.json"
    codebook_summary_path = run_dir / "prompt" / "codebook_summary.json"
    codebook_path = run_dir / "input" / "public_codebook.json"
    codebook_summary: dict[str, Any] = {}
    if codebook_summary_path.exists():
        codebook_summary = json.loads(codebook_summary_path.read_text(encoding="utf-8"))
    elif codebook_path.exists():
        codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
        codebook_summary = {
            "dataset_id": codebook.get("dataset_id", "analysis_dataset"),
            "outcome": codebook.get("outcome", DEFAULT_OUTCOME),
            "n_rows": codebook.get("n_rows"),
            "n_columns": codebook.get("n_columns"),
            "variables": codebook.get("variables", []),
        }
    return {
        "task_prompt": task_path.read_text(encoding="utf-8") if task_path.exists() else "",
        "output_contract": output_contract_path.read_text(encoding="utf-8") if output_contract_path.exists() else "",
        "dataset_preview": dataset_preview_path.read_text(encoding="utf-8") if dataset_preview_path.exists() else "",
        "messages": load_json_if_exists(messages_path).get("messages", []) if messages_path.exists() else [],
        "codebook_summary": codebook_summary,
    }


def load_execution_result_from_run(run_dir: Path) -> ExecutionResult:
    manifest = load_json_if_exists(run_dir / "metadata" / "execution_manifest.json")
    return ExecutionResult(
        returncode=int(manifest.get("returncode", 0)),
        stdout_path=run_dir / "logs" / "stdout.txt",
        stderr_path=run_dir / "logs" / "stderr.txt",
        duration_seconds=float(manifest.get("duration_seconds", 0.0)),
        generated_files=[str(item) for item in manifest.get("generated_files", [])],
    )


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_file_decodes(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return True


def inspect_agent_run(run_dir: Path, tail_chars: int = 2400) -> AgentRunInspection:
    run_dir = run_dir.resolve()
    manifest = load_json_if_exists(run_dir / "metadata" / "run_manifest.json")
    steps: list[dict[str, Any]] = []
    request_dir = run_dir / "request"
    response_dir = run_dir / "response"
    if request_dir.exists():
        for request_path in sorted(request_dir.glob("*_request.json")):
            stem = request_path.name.removesuffix("_request.json")
            response_path = response_dir / f"{stem}_response.json"
            message_path = response_dir / f"{stem}_message.md"
            response = load_json_if_exists(response_path)
            usage = response.get("usage") if isinstance(response, dict) else None
            if not usage and isinstance(response, dict):
                usage = response.get("usage_metadata")
            steps.append(
                {
                    "step": stem,
                    "request": str(request_path.relative_to(run_dir)).replace("\\", "/"),
                    "response": str(response_path.relative_to(run_dir)).replace("\\", "/")
                    if response_path.exists()
                    else "",
                    "message": str(message_path.relative_to(run_dir)).replace("\\", "/")
                    if message_path.exists()
                    else "",
                    "message_chars": message_path.stat().st_size if message_path.exists() else 0,
                    "error": str(response.get("error", "")) if isinstance(response, dict) else "",
                    "usage": usage or {},
                }
            )

    execution_attempts: list[dict[str, Any]] = []
    metadata_dir = run_dir / "metadata"
    if metadata_dir.exists():
        for execution_path in sorted(metadata_dir.glob("*execution*_manifest.json")):
            payload = load_json_if_exists(execution_path)
            payload = dict(payload)
            payload["manifest"] = str(execution_path.relative_to(run_dir)).replace("\\", "/")
            execution_attempts.append(payload)

    generated_files = sorted(
        path for path in snapshot_files(run_dir) if path.startswith("intermediate/") or path.startswith("plots/")
    )
    stdout_tail = read_tail(run_dir / "logs" / "stdout.txt", tail_chars)
    stderr_tail = read_tail(run_dir / "logs" / "stderr.txt", tail_chars)
    mini_paper_excerpt = read_tail(run_dir / "paper" / "mini_paper.md", tail_chars)
    return AgentRunInspection(
        run_dir=run_dir,
        manifest=manifest,
        steps=steps,
        execution_attempts=execution_attempts,
        generated_files=generated_files,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        mini_paper_excerpt=mini_paper_excerpt,
    )


def read_tail(path: Path, chars: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= chars:
        return text
    return text[-chars:]


def format_agent_run_inspection(report: AgentRunInspection) -> str:
    manifest = report.manifest
    lines = [
        "Agent run inspection.",
        f"Run dir: {report.run_dir}",
        f"Status: {manifest.get('status', 'missing_run_manifest')}",
        f"Author paradigm: {manifest.get('author_paradigm', '')}",
        f"Model: {manifest.get('model', '')}",
    ]
    if manifest.get("workflow_stages"):
        lines.append("Workflow: " + " -> ".join(str(item) for item in manifest.get("workflow_stages", [])))
    if manifest.get("loop_options"):
        loop_options = manifest["loop_options"]
        lines.append(
            "Loop limits/usage: "
            f"repairs {loop_options.get('repair_attempts_used', '')}/{loop_options.get('max_code_repairs', '')}, "
            f"supplements {loop_options.get('supplemental_attempts_used', '')}/"
            f"{loop_options.get('max_supplemental_analyses', '')}"
        )
    if manifest.get("analysis_review"):
        review = manifest["analysis_review"]
        lines.append(
            "Analysis review: "
            f"objective_met={review.get('objective_met')}, "
            f"supplemental_requested={review.get('supplemental_analysis_requested')}, "
            f"limit_reached={review.get('supplemental_limit_reached')}"
        )
    lines.append("")
    lines.append("Steps:")
    for step in report.steps:
        usage = step.get("usage") or {}
        usage_text = ""
        if usage:
            usage_text = (
                f", tokens={usage.get('total_tokens', '')}, cost={usage.get('cost', '')}"
                if isinstance(usage, dict)
                else ""
            )
        error_text = f", error={truncate_text(step.get('error', ''), 180)}" if step.get("error") else ""
        lines.append(
            f"  - {step['step']}: message_chars={step['message_chars']}{usage_text}{error_text}"
        )
    lines.append("")
    lines.append("Execution attempts:")
    for attempt in report.execution_attempts:
        lines.append(
            "  - "
            f"{attempt.get('manifest', '')}: returncode={attempt.get('returncode', '')}, "
            f"duration={attempt.get('duration_seconds', '')}, "
            f"generated_files={len(attempt.get('generated_files', []))}"
        )
    lines.append("")
    lines.append("Generated analysis files:")
    if report.generated_files:
        lines.extend(f"  - {path}" for path in report.generated_files[:80])
    else:
        lines.append("  - none")
    if report.stderr_tail.strip():
        lines.extend(["", "stderr tail:", truncate_text(report.stderr_tail.strip(), 2400)])
    if report.stdout_tail.strip():
        lines.extend(["", "stdout tail:", truncate_text(report.stdout_tail.strip(), 1600)])
    if report.mini_paper_excerpt.strip():
        lines.extend(["", "mini-paper excerpt:", truncate_text(report.mini_paper_excerpt.strip(), 2400)])
    return "\n".join(lines)


def extract_assistant_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output = response.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict):
                    text = content.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        if parts:
            return "\n".join(parts)
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def run_analysis_code(run_dir: Path, code_path: Path) -> None:
    completed = subprocess.run(
        ["python", str(code_path.resolve())],
        cwd=run_dir,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    (run_dir / "logs" / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "logs" / "stderr.txt").write_text(completed.stderr, encoding="utf-8")


def make_run_id(dataset_id: str, prompt_regime: str, model: str, run_index: int) -> str:
    del dataset_id, prompt_regime, model
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_p{os.getpid()}_{run_index:03d}"


def public_dataset_slug(prompt_payload: dict[str, Any], dataset_dir: Path) -> str:
    public_id = str(prompt_payload.get("codebook_summary", {}).get("dataset_id", "analysis_dataset"))
    digest = hashlib.sha256(str(dataset_dir.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{sanitize_slug(public_id)}-{digest}"


def sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return slug.strip("-") or "run"


def format_agent_harness_report(report: AgentHarnessReport) -> str:
    lines = [
        "Agent run harness complete.",
        f"Dataset: {_relative_to_root(report.dataset_dir, report.root)}",
        f"Model: {report.model}",
        f"Prompt regime: {report.prompt_regime}",
        f"Author paradigm: {report.author_paradigm or 'single_request'}",
        f"Dry run: {report.dry_run}",
        f"Run root: {_relative_to_root(report.run_root, report.root)}",
        "",
        "Runs:",
    ]
    for item in report.runs:
        lines.append(f"  - {item.run_id}: {item.status}")
        lines.append(f"    dir: {_relative_to_root(item.run_dir, report.root)}")
    return "\n".join(lines)


def format_claim_extraction_harness_report(report: ClaimExtractionHarnessReport) -> str:
    return "\n".join(
        [
            "Claim extraction complete.",
            f"Run dir: {_relative_to_root(report.run_dir, report.root)}",
            f"Model: {report.model}",
            f"Dry run: {report.dry_run}",
            f"Status: {report.status}",
            f"Claims: {report.claim_count}",
            f"Request: {_relative_to_root(report.request_path, report.root)}",
            f"Response: {_relative_to_root(report.response_path, report.root)}",
            f"Claim table: {_relative_to_root(report.claims_path, report.root)}",
            f"Manifest: {_relative_to_root(report.manifest_path, report.root)}",
        ]
    )


def format_reviewer_harness_report(report: ReviewerHarnessReport) -> str:
    return "\n".join(
        [
            "Statistical review complete.",
            f"Run dir: {_relative_to_root(report.run_dir, report.root)}",
            f"Model: {report.model}",
            f"Reviewer style: {report.reviewer_style}",
            f"Dry run: {report.dry_run}",
            f"Status: {report.status}",
            f"Major points: {report.major_point_count}",
            f"Request: {_relative_to_root(report.request_path, report.root)}",
            f"Response: {_relative_to_root(report.response_path, report.root)}",
            f"Review: {_relative_to_root(report.review_path, report.root)}",
            f"Manifest: {_relative_to_root(report.manifest_path, report.root)}",
        ]
    )


def format_selection_review_harness_report(report: SelectionReviewHarnessReport) -> str:
    return "\n".join(
        [
            "Selection review complete.",
            f"Run root: {_relative_to_root(report.run_root, report.root)}",
            f"Model: {report.model}",
            f"Dry run: {report.dry_run}",
            f"Status: {report.status}",
            f"Papers ranked: {report.paper_count}",
            f"Groups: {report.group_count}",
            f"Request: {_relative_to_root(report.request_path, report.root)}",
            f"Response: {_relative_to_root(report.response_path, report.root)}",
            f"Review: {_relative_to_root(report.review_path, report.root)}",
            f"Manifest: {_relative_to_root(report.manifest_path, report.root)}",
        ]
    )


def format_openai_batch_report(report: OpenAIBatchReport) -> str:
    lines = [
        "OpenAI batch status.",
        f"Task: {report.task}",
        f"Batch id: {report.batch_id}",
        f"Status: {report.status}",
        f"Requests: {report.request_count}",
        f"Completed results applied: {report.completed_count}",
        f"Failed results: {report.failed_count}",
        f"Manifest: {_relative_to_root(report.manifest_path, report.root)}",
        f"Request JSONL: {_relative_to_root(report.request_jsonl_path, report.root)}",
    ]
    if report.output_jsonl_path is not None:
        lines.append(f"Output JSONL: {_relative_to_root(report.output_jsonl_path, report.root)}")
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_fenced(text: str, language: str) -> str:
    pattern = rf"```{re.escape(language)}\s*(.*?)```"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _csv_quote(value: str) -> str:
    if any(char in value for char in [",", '"', "\n"]):
        return '"' + value.replace('"', '""') + '"'
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)
