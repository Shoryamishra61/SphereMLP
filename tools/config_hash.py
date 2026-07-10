"""Print a stable hash for a JSON-encoded RuntimeConfig."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from whest_solution.config import RuntimeConfig, stable_config_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding="utf-8"))
    print(stable_config_hash(RuntimeConfig(**values)))


if __name__ == "__main__":
    main()
