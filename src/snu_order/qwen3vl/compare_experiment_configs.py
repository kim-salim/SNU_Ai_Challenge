from __future__ import annotations

import argparse
import json
from typing import Any

from snu_order.utils.config import load_config


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(value[key], path))
    return flattened


def compare_configs(
    base: dict[str, Any],
    candidate: dict[str, Any],
    *,
    allowed_paths: set[str],
) -> dict[str, dict[str, Any]]:
    base_flat = _flatten(base)
    candidate_flat = _flatten(candidate)
    differences = {
        path: {"base": base_flat.get(path), "candidate": candidate_flat.get(path)}
        for path in sorted(set(base_flat) | set(candidate_flat))
        if base_flat.get(path) != candidate_flat.get(path) and path not in allowed_paths
    }
    return differences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--allow-path", action="append", default=[])
    args = parser.parse_args()
    differences = compare_configs(
        load_config(args.base),
        load_config(args.candidate),
        allowed_paths=set(args.allow_path),
    )
    print(json.dumps({"differences": differences}, ensure_ascii=False, indent=2))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
