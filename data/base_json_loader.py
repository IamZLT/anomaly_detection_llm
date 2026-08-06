import json
import math
import os
import random
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def extract_bbox_from_mask(mask_path: str) -> Optional[List[int]]:
    if mask_path is None or not os.path.exists(mask_path):
        return None
    try:
        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask) > 0
        rows = np.any(mask_array, axis=1)
        cols = np.any(mask_array, axis=0)
        if not (rows.any() and cols.any()):
            return None
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]
        y1, y2 = y_indices[0], y_indices[-1]
        x1, x2 = x_indices[0], x_indices[-1]
        return [int(x1), int(y1), int(x2), int(y2)]
    except Exception:
        return None


class GenericJSONDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        conversation_json_path: str,
        split: Optional[str] = None,
        anomaly_only: bool = False,
        with_bbox: bool = True,
        check_exists: bool = True,
        sampling_strategy: str = "all",
    ):
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.conversation_json_path = conversation_json_path
        self.split = split
        self.anomaly_only = anomaly_only
        self.with_bbox = with_bbox
        self.check_exists = check_exists
        self.sampling_strategy = str(sampling_strategy or "all")
        self.samples = self._load_samples()

    def _resolve_candidates(self, raw_path: str) -> List[str]:
        rp = raw_path.replace("\\", "/")
        cands: List[str] = []
        if os.path.isabs(rp):
            cands.append(rp)
            parts = [p for p in rp.strip("/").split("/") if p]
            for marker in (
                "anomaly_dataset",
                "anomaly_shapenet",
                "bmad",
                "mvtec_anomaly_detection",
                "mvtec3d",
                "real3d",
                "webad",
            ):
                if marker in parts:
                    i = parts.index(marker)
                    cands.append(os.path.join(self.dataset_root, *parts[i:]))
            cands.append(os.path.join(self.dataset_root, os.path.basename(rp)))
        else:
            cands.append(os.path.join(self.dataset_root, rp))

        deduped: List[str] = []
        seen = set()
        for p in cands:
            if p not in seen:
                deduped.append(p)
                seen.add(p)
        return deduped

    def _choose_path(self, value) -> tuple[Optional[str], Optional[str]]:
        paths: List[str] = []
        if isinstance(value, str):
            s = value.strip()
            if s:
                paths.append(s)
        elif isinstance(value, (list, tuple)):
            for v in value:
                if isinstance(v, str):
                    s = v.strip()
                    if s:
                        paths.append(s)
        if not paths:
            return None, None

        if not self.check_exists:
            p0 = paths[0]
            cands = self._resolve_candidates(p0)
            return p0, (cands[0] if cands else None)

        for p in paths:
            for c in self._resolve_candidates(p):
                if os.path.exists(c):
                    return p, c

        p0 = paths[0]
        cands = self._resolve_candidates(p0)
        return p0, (cands[0] if cands else None)

    @staticmethod
    def _normalize_conversations(conversations: List[Dict]) -> List[Dict]:
        normalized = []
        for conv in conversations:
            role = conv.get("from") or conv.get("role")
            text = conv.get("value") or conv.get("content") or ""
            if role in ("human", "user"):
                text = str(text).strip()
                if "<image>" not in text:
                    text = "<image>\n" + text
                normalized.append({"from": "human", "value": text})
            elif role in ("gpt", "assistant"):
                normalized.append({"from": "gpt", "value": str(text)})
            else:
                normalized.append(conv)
        return normalized

    @staticmethod
    def _class_split_defect_after_root(parts: List[str], j: int) -> tuple[Optional[str], Optional[str], str]:
        """``parts[j]`` 为数据集集合目录名（如 ``metadata.source`` / ``anomaly_shapenet``），物体类为 ``parts[j+1]``（如 ``tap0``）。"""
        if j < 0 or j + 1 >= len(parts):
            return None, None, "good"
        cls_name = parts[j + 1]
        rest = parts[j + 2 :]
        if not rest:
            return cls_name, None, "good"
        si = next((i for i, p in enumerate(rest) if p in ("train", "test")), None)
        if si is not None:
            split = rest[si]
            mid = rest[si + 1 : -1]
            defect_type = "/".join(mid) if mid else "good"
            return cls_name, split, defect_type
        split = None
        defect_type = "/".join(rest[:-1]) if len(rest) > 1 else "good"
        return cls_name, split, defect_type

    @staticmethod
    def _parse_image_path(
        image_rel: str,
        source_hint: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str], str]:
        """仅用 ``metadata.source`` 在路径中锚定：物体类为 ``source`` 目录名的下一层。"""
        parts = [p for p in str(image_rel).replace("\\", "/").split("/") if p]
        if not parts:
            return None, None, "good"

        sh = (source_hint or "").strip()
        if sh and sh in parts:
            j = parts.index(sh)
            cls_name, split, defect_type = GenericJSONDataset._class_split_defect_after_root(parts, j)
            if cls_name:
                return cls_name, split, defect_type
        return None, None, "good"

    @staticmethod
    def _apply_sampling_strategy(records: List[Dict], sampling_strategy: str) -> List[Dict]:
        strategy = str(sampling_strategy or "all").strip()
        if strategy == "all":
            return records

        mode = strategy
        number = None
        if ":" in strategy:
            mode, value = strategy.split(":", 1)
            value = value.strip()
            if value.endswith("%"):
                pct = int(value[:-1])
                number = math.ceil(max(0, pct) * len(records) / 100)
            else:
                number = int(value)

        mode = mode.strip().lower()
        if number is None:
            return records
        number = max(0, min(number, len(records)))

        if mode == "first":
            return records[:number]
        if mode == "end":
            return records[-number:] if number > 0 else []
        if mode == "random":
            idx = list(range(len(records)))
            random.shuffle(idx)
            pick = set(idx[:number])
            return [r for i, r in enumerate(records) if i in pick]
        return records

    def _load_records(self) -> List[Dict]:
        if self.conversation_json_path.endswith(".jsonl"):
            records: List[Dict] = []
            with open(self.conversation_json_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        records.append(json.loads(s))
            return records
        with open(self.conversation_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_samples(self) -> List[Dict]:
        records = self._load_records()
        records = self._apply_sampling_strategy(records, self.sampling_strategy)

        samples = []
        skipped_missing_image = 0
        for item in records:
            image_rel, full_img_path = self._choose_path(item.get("image"))
            if not image_rel or not full_img_path:
                continue
            image_rel = image_rel.replace("\\", "/")

            metadata = item.get("metadata", {}) or {}
            source_hint = str(metadata.get("source") or "").strip()

            cls_name, split, defect_type = self._parse_image_path(image_rel, source_hint)
            if cls_name is None and full_img_path:
                cls_name, split, defect_type = self._parse_image_path(
                    str(full_img_path).replace("\\", "/"), source_hint
                )

            anomaly = bool(metadata.get("anomaly", False))
            if self.anomaly_only and not anomaly:
                continue

            mask_rel, full_mask_path = self._choose_path(metadata.get("mask"))
            if self.check_exists and full_img_path and not os.path.exists(full_img_path):
                skipped_missing_image += 1
                continue
            if self.check_exists and full_mask_path and not os.path.exists(full_mask_path):
                full_mask_path = None

            if cls_name is None:
                cls_name = "unknown"

            if self.split is not None and split is not None and split != self.split:
                continue

            new_metadata = dict(metadata)
            new_metadata["class"] = cls_name
            new_metadata["defect_type"] = defect_type
            new_metadata["full_mask_path"] = full_mask_path
            if self.with_bbox and anomaly and full_mask_path:
                new_metadata["bbox"] = extract_bbox_from_mask(full_mask_path)
            else:
                new_metadata["bbox"] = None

            file_name = os.path.basename(image_rel)
            sample = {
                "id": item.get("id", f"{self.dataset_name}_{file_name}"),
                "dataset_name": self.dataset_name,
                "dataset_root": self.dataset_root,
                "image": image_rel,
                "full_img_path": full_img_path,
                "conversations": self._normalize_conversations(item.get("conversations", [])),
                "metadata": new_metadata,
            }
            samples.append(sample)
        if self.check_exists and skipped_missing_image > 0:
            print(
                f"[data][{self.dataset_name}] skipped {skipped_missing_image} samples due to missing image files",
                flush=True,
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class GenericDataManager:
    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        conversation_json_path: str,
        sampling_strategy: str = "all",
    ):
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.conversation_json_path = conversation_json_path
        self.sampling_strategy = str(sampling_strategy or "all")
        self._samples_cache: Optional[List[Dict]] = None
        # keep compatibility with legacy usages manager.dataset_loader.dataset_root
        self.dataset_loader = self

    def load_all(self) -> None:
        if not os.path.isfile(self.conversation_json_path):
            raise FileNotFoundError(
                f"[{self.dataset_name}] conversation json not found: {self.conversation_json_path}"
            )
        ds = GenericJSONDataset(
            dataset_name=self.dataset_name,
            dataset_root=self.dataset_root,
            conversation_json_path=self.conversation_json_path,
            split=None,
            anomaly_only=False,
            with_bbox=True,
            check_exists=True,
            sampling_strategy=self.sampling_strategy,
        )
        self._samples_cache = list(ds.samples)

    def get_all_grounding_samples(self, mode: str = "test", anomaly_only: bool = False) -> List[Dict]:
        if self._samples_cache is None:
            self.load_all()
        samples = list(self._samples_cache or [])
        if anomaly_only:
            samples = [s for s in samples if bool((s.get("metadata") or {}).get("anomaly", False))]
        return samples

    def get_all_train_samples(self) -> List[Dict]:
        return self.get_all_grounding_samples(mode="train", anomaly_only=False)

    def get_all_test_samples(self) -> List[Dict]:
        return self.get_all_grounding_samples(mode="test", anomaly_only=False)


def infer_object_class_from_paths(
    image_rel: str,
    full_img_path: Optional[str] = None,
    *,
    source_hint: Optional[str] = None,
) -> Optional[str]:
    """供 ``visual_proto_data`` 等使用：仅用 ``metadata.source`` 在路径中锚定推断物体类。"""
    for raw in (image_rel or "", full_img_path or ""):
        raw = str(raw).strip()
        if not raw:
            continue
        c, _, _ = GenericJSONDataset._parse_image_path(raw.replace("\\", "/"), source_hint)
        if c:
            return c
    return None
