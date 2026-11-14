"""Secondary arms (pre-specified in protocol v0.1): L2 for confounding, and the generic checklist G.

L2 = L1 wording plus one sentence stating the analytical consequence.
G  = L0 documentation plus a trap-agnostic checklist appended to the neutral task prompt.
"""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
VAR = ROOT / "data" / "traps" / "doc_cue_v1"
CONSEQUENCE = (" Its association with the endpoint may therefore reflect age, BMI and sex, and an independent "
               "association requires adjustment for them.")
TRAP_VARS = ["hscrp_mg_l", "inflammatory_burden_score", "vascular_health_index", "adiposity_inflammation_score"]
CHECKLIST = ("Before reporting any variable as associated with the endpoint: (1) confirm from the documentation that "
             "it was measured before the endpoint was ascertained; (2) check whether the association persists after "
             "adjustment for plausible common causes; (3) use the survey weights and design variables for any "
             "population-level statement; and (4) treat associations that could arise by chance across many tests "
             "as exploratory.")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    src = VAR / "confounding_trap_L1"
    dst = VAR / "confounding_trap_L2"
    dst.mkdir(exist_ok=True)
    for f in ("dataset.csv", "hidden_truth_manifest.json"):
        shutil.copy2(src / f, dst / f)
    cb = json.loads((src / "public_codebook.json").read_text(encoding="utf-8"))
    for v in cb["variables"]:
        if v["name"] in TRAP_VARS:
            v["description"] = v["description"].rstrip() + CONSEQUENCE
    (dst / "public_codebook.json").write_text(json.dumps(cb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    prompts = (ROOT / "configs" / "prompts" / "scientist_prompts.toml").read_text(encoding="utf-8")
    prompts += ('\n[regimes.neutral_checklist]\nlabel = "neutral_checklist"\ntask = """\n'
                "Explore the dataset and report empirical associations with the primary cardiometabolic risk endpoint.\n\n"
                + CHECKLIST + '\n"""\n')
    (EXP / "prompts_checklist.toml").write_text(prompts, encoding="utf-8")

    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "confounding_trap_L2_codebook_sha256": sha256(dst / "public_codebook.json"),
                "confounding_trap_L2_dataset_sha256": sha256(dst / "dataset.csv"),
                "prompts_checklist_sha256": sha256(EXP / "prompts_checklist.toml"),
                "consequence_sentence": CONSEQUENCE.strip(), "checklist": CHECKLIST,
                "cells": {"confounding_trap_L2": "L2 docs, neutral task",
                          "confounding_trap_G": "L0 docs, neutral_checklist task",
                          "leakage_trap_G": "L0 docs, neutral_checklist task"}}
    (EXP / "SECONDARY_ARMS_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
