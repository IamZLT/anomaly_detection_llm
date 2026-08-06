import os
from typing import Dict, List

from data.base_json_loader import GenericDataManager, infer_object_class_from_paths

def parse_dataset_item(item) -> tuple[str, str]:
    if isinstance(item, dict):
        return str(item.get("json_path", "")).strip(), str(item.get("sampling_strategy", "all")).strip()
    s = str(item).strip()
    if not s:
        return "", "all"
    if ".json:" in s:
        i = s.find(".json:")
        json_name = s[: i + 5]
        tail = s[i + 6 :].strip()
        if tail.endswith("%") and tail[:-1].isdigit():
            return json_name, f"random:{tail}"
        return json_name, tail or "all"
    if ".jsonl:" in s:
        i = s.find(".jsonl:")
        json_name = s[: i + 6]
        tail = s[i + 7 :].strip()
        if tail.endswith("%") and tail[:-1].isdigit():
            return json_name, f"random:{tail}"
        return json_name, tail or "all"
    return s, "all"


def build_samples_from_specs(dataset_root: str, specs: List) -> List[Dict]:
    out: List[Dict] = []
    for item in specs:
        json_name, sampling = parse_dataset_item(item)
        if not json_name:
            continue
        json_path = json_name if os.path.isabs(json_name) else os.path.join(dataset_root, json_name)
        dataset_name = os.path.splitext(os.path.basename(json_name))[0]
        manager = GenericDataManager(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            conversation_json_path=json_path,
            sampling_strategy=sampling or "all",
        )
        manager.load_all()
        records = manager.get_all_grounding_samples(mode="train", anomaly_only=False)
        for s in records:
            md = s.get("metadata", {}) or {}
            source_s = str(md.get("source") or "").strip()
            logical_ds = source_s or s.get("dataset_name") or dataset_name
            md_class = md.get("class")
            if md_class is not None and str(md_class).strip():
                cls_name = str(md_class).strip()
            else:
                cls_name = infer_object_class_from_paths(
                    str(s.get("image") or ""),
                    s.get("full_img_path"),
                    source_hint=source_s or None,
                ) or "unknown"
            out.append(
                {
                    "img_path": s.get("full_img_path"),
                    "mask_path": md.get("full_mask_path"),
                    "anomaly": 1 if bool(md.get("anomaly", False)) else 0,
                    "dataset_name": logical_ds,
                    "cls_name": cls_name,
                    "defect_type": md.get("defect_type") or md.get("category") or "unknown",
                }
            )
    return out
