"""Build the documentation-cue variants (L0 neutral, L1 factual cue) for the confounding and leakage traps.

Only the description/source_file of the four trap variables changes. dataset.csv and every other codebook
entry are byte-identical to the frozen main-study files. Writes a SHA-256 freeze manifest.
"""
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "traps" / "plasmode_nhanes_latest"
OUT = ROOT / "data" / "traps" / "doc_cue_v1"
EXP = Path(__file__).resolve().parent

VARIANTS = {
    "confounding_trap": {
        "L0": {
            "hscrp_mg_l": ("High-sensitivity C-reactive protein in mg/L.", None),
            "inflammatory_burden_score": ("Composite inflammatory burden score from baseline records.", None),
            "vascular_health_index": ("Vascular health index from the study examination.", None),
            "adiposity_inflammation_score": ("Composite adiposity and inflammation score.", None),
        },
        "L1": {
            "hscrp_mg_l": ("High-sensitivity C-reactive protein in mg/L. In this cohort, values vary strongly with "
                           "age, BMI and sex.", None),
            "inflammatory_burden_score": ("Composite score computed from age, BMI and sex, weighted by their "
                                          "associations with hs-CRP.", None),
            "vascular_health_index": ("Vascular health index computed from age, BMI and sex.", None),
            "adiposity_inflammation_score": ("Composite score computed from BMI, age and hs-CRP.", None),
        },
    },
    "leakage_trap": {
        "L0": {
            "laboratory_risk_index": ("Laboratory risk index summarizing laboratory records.", None),
            "care_pathway_index": ("Care pathway intensity index summarized from clinical pathway records.",
                                   "Clinical pathway file."),
            "medication_profile_score": ("Medication profile score summarized from medication records.", None),
            "clinical_review_score": ("Clinical review score from chart abstraction.", "Clinical record file."),
        },
        # L1 is the main-study wording, kept verbatim
        "L1": {},
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "variants": {}}
    for trap, arms in VARIANTS.items():
        base = json.loads((SRC / trap / "public_codebook.json").read_text(encoding="utf-8"))
        names = {v["name"] for v in base["variables"]}
        for arm, edits in arms.items():
            assert set(edits) <= names, set(edits) - names
            cb = copy.deepcopy(base)
            for v in cb["variables"]:
                if v["name"] in edits:
                    desc, src = edits[v["name"]]
                    v["description"] = desc
                    if src is not None:
                        v["source_file"] = src
            d = OUT / f"{trap}_{arm}"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SRC / trap / "dataset.csv", d / "dataset.csv")
            shutil.copy2(SRC / trap / "hidden_truth_manifest.json", d / "hidden_truth_manifest.json")
            (d / "public_codebook.json").write_text(json.dumps(cb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            assert sha256(d / "dataset.csv") == sha256(SRC / trap / "dataset.csv")
            changed = [v["name"] for v, b in zip(cb["variables"], base["variables"]) if v != b]
            manifest["variants"][f"{trap}_{arm}"] = {
                "dir": str(d.relative_to(ROOT)).replace("\\", "/"),
                "dataset_sha256": sha256(d / "dataset.csv"),
                "codebook_sha256": sha256(d / "public_codebook.json"),
                "truth_sha256": sha256(d / "hidden_truth_manifest.json"),
                "changed_entries": changed,
                "trap_entries": {v["name"]: {"description": v["description"], "source_file": v["source_file"]}
                                 for v in cb["variables"] if v["name"] in edits or (arm == "L1" and trap == "leakage_trap"
                                 and v["name"] in VARIANTS[trap]["L0"])},
            }
    (EXP / "VARIANT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for k, v in manifest["variants"].items():
        print(k, "changed:", v["changed_entries"])


if __name__ == "__main__":
    main()
