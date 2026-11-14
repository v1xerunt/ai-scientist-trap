"""Download public-use NHANES files from CDC/NCHS."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trap_ai.config import load_toml, resolve_config_path


USER_AGENT = "trap-ai/0.1 (+https://wwwn.cdc.gov/nchs/nhanes/)"


@dataclass(frozen=True)
class DownloadItem:
    cycle: str
    component: str
    stem: str
    extension: str
    url: str
    path: Path


@dataclass(frozen=True)
class DownloadResult:
    item: DownloadItem
    status: str
    bytes: int
    sha256: str
    error: str = ""


@dataclass(frozen=True)
class DownloadReport:
    root: Path
    manifest_csv: Path
    summary_json: Path
    results: list[DownloadResult]

    @property
    def downloaded(self) -> int:
        return sum(result.status == "downloaded" for result in self.results)

    @property
    def skipped(self) -> int:
        return sum(result.status == "skipped" for result in self.results)

    @property
    def failed(self) -> int:
        return sum(result.status == "failed" for result in self.results)


def download_nhanes(
    root: Path,
    config_path: Path,
    overwrite: bool = False,
    timeout: float = 90.0,
) -> DownloadReport:
    root = root.resolve()
    config = load_toml(resolve_config_path(root, config_path))
    dataset = config["dataset"]
    raw_dir = root / dataset.get("raw_dir", "data/raw/nhanes_pilot")
    manifest_csv = root / dataset.get("manifest_csv", "data/raw/nhanes_pilot/download_manifest.csv")
    summary_json = root / dataset.get("summary_json", "data/raw/nhanes_pilot/download_summary.json")

    items = build_download_items(config, raw_dir)
    results = [_download_item(item, overwrite=overwrite, timeout=timeout) for item in items]
    _write_manifest(manifest_csv, results, root)
    _write_summary(summary_json, results, root)
    return DownloadReport(root=root, manifest_csv=manifest_csv, summary_json=summary_json, results=results)


def build_download_items(config: dict[str, Any], raw_dir: Path) -> list[DownloadItem]:
    dataset = config["dataset"]
    base_url = str(dataset["base_url"]).rstrip("/")
    extensions = list(dataset.get("download_extensions", ["xpt"]))
    items: list[DownloadItem] = []

    for cycle in config.get("cycles", []):
        year = int(cycle["year"])
        suffix = str(cycle["suffix"])
        cycle_label = str(cycle["label"])
        for file_config in config.get("files", []):
            stem = str(file_config["stem"])
            component = str(file_config["component"])
            component_dir = _slug(component)
            file_base = f"{stem}_{suffix}"
            for extension in extensions:
                clean_extension = str(extension).lstrip(".").lower()
                filename = f"{file_base}.{clean_extension}"
                url = f"{base_url}/{year}/DataFiles/{filename}"
                path = raw_dir / cycle_label / component_dir / filename
                items.append(
                    DownloadItem(
                        cycle=cycle_label,
                        component=component,
                        stem=stem,
                        extension=clean_extension,
                        url=url,
                        path=path,
                    )
                )

    return items


def format_download_report(report: DownloadReport) -> str:
    lines = [
        "NHANES download complete.",
        f"Downloaded: {report.downloaded}",
        f"Skipped: {report.skipped}",
        f"Failed: {report.failed}",
        f"Manifest: {_relative_to_root(report.manifest_csv, report.root)}",
        f"Summary: {_relative_to_root(report.summary_json, report.root)}",
    ]
    if report.failed:
        lines.append("")
        lines.append("Failures:")
        for result in report.results:
            if result.status == "failed":
                lines.append(f"  - {result.item.url}: {result.error}")
    return "\n".join(lines)


def _download_item(item: DownloadItem, overwrite: bool, timeout: float) -> DownloadResult:
    if item.path.exists() and not overwrite:
        return DownloadResult(
            item=item,
            status="skipped",
            bytes=item.path.stat().st_size,
            sha256=_sha256(item.path),
        )

    item.path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = item.path.with_name(f"{item.path.name}.tmp")
    request = urllib.request.Request(item.url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        temp_path.replace(item.path)
        return DownloadResult(
            item=item,
            status="downloaded",
            bytes=item.path.stat().st_size,
            sha256=_sha256(item.path),
        )
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        if temp_path.exists():
            temp_path.unlink()
        return DownloadResult(item=item, status="failed", bytes=0, sha256="", error=str(exc))


def _write_manifest(path: Path, results: list[DownloadResult], root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cycle",
                "component",
                "stem",
                "extension",
                "status",
                "bytes",
                "sha256",
                "path",
                "url",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "cycle": result.item.cycle,
                    "component": result.item.component,
                    "stem": result.item.stem,
                    "extension": result.item.extension,
                    "status": result.status,
                    "bytes": result.bytes,
                    "sha256": result.sha256,
                    "path": _relative_to_root(result.item.path, root),
                    "url": result.item.url,
                    "error": result.error,
                }
            )


def _write_summary(path: Path, results: list[DownloadResult], root: Path) -> None:
    payload = {
        "downloaded": sum(result.status == "downloaded" for result in results),
        "skipped": sum(result.status == "skipped" for result in results),
        "failed": sum(result.status == "failed" for result in results),
        "files": [
            {
                "cycle": result.item.cycle,
                "component": result.item.component,
                "stem": result.item.stem,
                "extension": result.item.extension,
                "status": result.status,
                "bytes": result.bytes,
                "sha256": result.sha256,
                "path": _relative_to_root(result.item.path, root),
                "url": result.item.url,
                "error": result.error,
            }
            for result in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("/", "_")


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)
