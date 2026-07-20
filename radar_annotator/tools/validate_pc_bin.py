#!/usr/bin/env python3
"""CLI wrapper: validate all *_pc.bin in a folder. See pc_bin_validation.py."""
from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from pc_bin_validation import main as _main

if __name__ == "__main__":
    sys.exit(_main())
