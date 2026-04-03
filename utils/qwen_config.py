from copy import deepcopy
from typing import Any, Dict

import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_update(merged[k], v)
        else:
            merged[k] = v
    return merged


def apply_runtime_overrides(cfg: Dict[str, Any], args: Any) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}

    if getattr(args, "mode", None):
        updates.setdefault("runtime", {})["mode"] = args.mode
    if getattr(args, "model_path", None):
        updates.setdefault("inference", {})["model_path"] = args.model_path
    if getattr(args, "image_path", None):
        updates.setdefault("inference", {})["image_path"] = args.image_path
    if getattr(args, "prompt", None):
        updates.setdefault("inference", {})["prompt"] = args.prompt
    if getattr(args, "output_dir", None):
        updates.setdefault("paths", {})["output_dir"] = args.output_dir
    if getattr(args, "run_name", None):
        updates.setdefault("runtime", {})["run_name"] = args.run_name

    return deep_update(cfg, updates)

