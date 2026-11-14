r"""Repository-local CLI wrapper.

This file lets users run the CLI without installing the package:

    python .\scripts\trap_ai.py init-toy
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trap_ai.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
