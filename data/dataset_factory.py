from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

from data.base_json_loader import GenericDataManager


@dataclass
class MultiDatasetManager:
    managers: list

    def __post_init__(self):
        if not self.managers:
            raise ValueError("No dataset manager configured.")
        # compatibility for legacy code path
        self.dataset_loader = self.managers[0].dataset_loader
        self.dataset_root = self.managers[0].dataset_root

    def load_all(self) -> None:
        for m in self.managers:
            m.load_all()

    def get_all_grounding_samples(self, mode: str = "test", anomaly_only: bool = False):
        merged = []
        for m in self.managers:
            merged.extend(m.get_all_grounding_samples(mode=mode, anomaly_only=anomaly_only))
        return merged

    def get_all_train_samples(self):
        merged = []
        for m in self.managers:
            merged.extend(m.get_all_train_samples())
        return merged

    def get_all_test_samples(self):
        merged = []
        for m in self.managers:
            merged.extend(m.get_all_test_samples())
        return merged

    def get_json_stats(self) -> list[dict]:
        stats = []
        for m in self.managers:
            samples = m.get_all_grounding_samples(mode="train", anomaly_only=False)
            image_paths = []
            for s in samples:
                p = s.get("full_img_path") or s.get("image")
                if p:
                    image_paths.append(str(p))
            stats.append(
                {
                    "dataset_name": getattr(m, "dataset_name", "unknown"),
                    "json_path": getattr(m, "conversation_json_path", ""),
                    "sampling_strategy": getattr(m, "sampling_strategy", "all"),
                    "num_samples": len(samples),
                    "num_images": len(set(image_paths)),
                }
            )
        return stats


def _to_json_path(dataset_root: str, json_name_or_path: str) -> str:
    j = str(json_name_or_path)
    if os.path.isabs(j):
        return j
    return os.path.join(dataset_root, j)


def _parse_dataset_item(item) -> tuple[str, str]:
    """
    Keep yaml simple list style, but support OneVision-like sampling.
    Supported forms:
    - "a.json" -> ("a.json", "all")
    - "a.json:first:1000" -> ("a.json", "first:1000")
    - {"json_path": "a.json", "sampling_strategy": "random:10%"}
    """
    if isinstance(item, dict):
        return str(item.get("json_path", "")).strip(), str(item.get("sampling_strategy", "all")).strip()
    s = str(item).strip()
    if not s:
        return "", "all"
    if ".json:" in s:
        i = s.find(".json:")
        return s[: i + 5], s[i + 6 :]
    if ".jsonl:" in s:
        i = s.find(".jsonl:")
        return s[: i + 6], s[i + 7 :]
    return s, "all"


def build_data_manager(cfg: Dict) -> MultiDatasetManager:
    paths_cfg = cfg.get("paths", {}) or {}
    dataset_root = os.path.expanduser(str(paths_cfg.get("dataset_root", "")))
    if not dataset_root:
        raise ValueError("paths.dataset_root is required.")

    # Top-level selector: list of json files under dataset_root.
    selected: List = list(cfg.get("datasets") or ["mvtec_zero_shot.json"])

    managers = []
    for item in selected:
        json_file, sampling_strategy = _parse_dataset_item(item)
        if not json_file:
            continue
        json_path = _to_json_path(dataset_root, str(json_file))
        dataset_name = os.path.splitext(os.path.basename(str(json_file)))[0]
        managers.append(
            GenericDataManager(
                dataset_name=dataset_name,
                dataset_root=dataset_root,
                conversation_json_path=os.path.expanduser(json_path),
                sampling_strategy=sampling_strategy or "all",
            )
        )
    return MultiDatasetManager(managers=managers)
