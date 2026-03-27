#!/usr/bin/env python3
import sys

# Ensure local src package is importable when running as a script
if "src" not in sys.modules:
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent))

from src.pipeline import main

if __name__ == "__main__":
    main()
