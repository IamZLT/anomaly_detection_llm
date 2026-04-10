import os
import json
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
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
    except Exception as e:
        print(f"Error extracting bbox from {mask_path}: {e}")
        return None


class MVTecJSONDataset(Dataset):
    """
    以 JSON 为主索引的数据集：
    先读 JSON，再解析 image / conversations / mask
    """

    def __init__(
        self,
        dataset_root: str,
        conversation_json_path: str,
        split: Optional[str] = "test",   # "train" / "test" / None（None=不按路径过滤，整份 JSON）
        anomaly_only: bool = False,
        with_bbox: bool = True,
        check_exists: bool = True,
    ):
        self.dataset_root = dataset_root
        self.conversation_json_path = conversation_json_path
        self.split = split
        self.anomaly_only = anomaly_only
        self.with_bbox = with_bbox
        self.check_exists = check_exists

        self.samples = self._load_samples()

    def _resolve_path(self, rel_path: Optional[str]) -> Optional[str]:
        """
        把 JSON 里的相对路径转成绝对路径。
        兼容两种 dataset_root:
        1) /data/.../datasets
        2) /data/.../datasets/mvtec_anomaly_detection
        """
        if not rel_path:
            return None

        rel_path = rel_path.replace("\\", "/")
        root_name = os.path.basename(os.path.normpath(self.dataset_root))

        # 如果 root 已经是 mvtec_anomaly_detection，就去掉 JSON 里的前缀
        if root_name == "mvtec_anomaly_detection" and rel_path.startswith("mvtec_anomaly_detection/"):
            rel_path = rel_path[len("mvtec_anomaly_detection/"):]

        return os.path.join(self.dataset_root, rel_path)

    def _parse_image_info(self, image_rel: str):
        """
        从 image 路径里解析:
        mvtec_anomaly_detection/capsule/test/poke/004.png
        -> cls_name=capsule, split=test, defect_type=poke
        """
        parts = image_rel.replace("\\", "/").split("/")

        if len(parts) < 4:
            raise ValueError(f"非法 image 路径: {image_rel}")

        cls_name = parts[-4]
        split = parts[-3]
        defect_type = parts[-2]
        file_name = parts[-1]

        return cls_name, split, defect_type, file_name

    def _normalize_conversations(self, conversations: List[Dict]) -> List[Dict]:
        """
        统一成:
        {"from": "human", "value": "..."}
        {"from": "gpt", "value": "..."}
        """
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

    def _load_samples(self) -> List[Dict]:
        with open(self.conversation_json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        samples = []

        for item in records:
            image_rel = item.get("image", "").replace("\\", "/")
            if not image_rel:
                continue

            cls_name, split, defect_type, file_name = self._parse_image_info(image_rel)

            if self.split is not None and split != self.split:
                continue

            metadata = item.get("metadata", {})
            anomaly = bool(metadata.get("anomaly", False))

            if self.anomaly_only and not anomaly:
                continue

            mask_rel = metadata.get("mask")
            full_img_path = self._resolve_path(image_rel)
            full_mask_path = self._resolve_path(mask_rel) if mask_rel else None

            if self.check_exists and full_img_path and not os.path.exists(full_img_path):
                print(f"[跳过] 图像不存在: {full_img_path}")
                continue

            if self.check_exists and full_mask_path and not os.path.exists(full_mask_path):
                print(f"[警告] mask不存在: {full_mask_path}")
                full_mask_path = None

            new_metadata = dict(metadata)
            new_metadata["class"] = cls_name
            new_metadata["defect_type"] = defect_type
            new_metadata["full_mask_path"] = full_mask_path

            if self.with_bbox and anomaly and full_mask_path:
                new_metadata["bbox"] = extract_bbox_from_mask(full_mask_path)
            else:
                new_metadata["bbox"] = None

            sample = {
                "id": item.get("id", f"{cls_name}_{defect_type}_{file_name}"),
                "image": image_rel,
                "full_img_path": full_img_path,
                "conversations": self._normalize_conversations(item.get("conversations", [])),
                "metadata": new_metadata,
            }

            samples.append(sample)

        print(f"加载完成: {len(samples)} 条样本")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MVTecDataManager:
    """
    兼容 ``utils/qwen_train`` / ``MVTecQwenGroundingDataset`` 的旧接口；
    数据以 ``MVTecJSONDataset`` 为准（JSON 列表为唯一索引，一条 JSON 一条样本）。

    固定使用整份 JSON：不按路径中的 ``train``/``test`` 过滤（``split=None``）。

    多轮→单轮 SFT 的展开在 ``MVTecQwenGroundingDataset``（``mode=train``）中完成，不在此层。
    """

    def __init__(self, dataset_root: str, conversation_json_path: str):
        self.dataset_root = dataset_root
        self.conversation_json_path = conversation_json_path
        # 旧代码通过 manager.dataset_loader.dataset_root 取根路径
        self.dataset_loader = self

    def load_all(self) -> None:
        if not os.path.isfile(self.conversation_json_path):
            raise FileNotFoundError(f"对话 JSON 不存在: {self.conversation_json_path}")
        with open(self.conversation_json_path, "r", encoding="utf-8") as f:
            json.load(f)

    def get_all_grounding_samples(
        self, mode: str = "test", anomaly_only: bool = False
    ) -> List[Dict]:
        # mode 仅用于调用方语义；数据始终为整份 JSON（含 .../train/... 与 .../test/...）
        ds = MVTecJSONDataset(
            dataset_root=self.dataset_root,
            conversation_json_path=self.conversation_json_path,
            split=None,
            anomaly_only=anomaly_only,
            with_bbox=True,
            check_exists=True,
        )
        return [
            {
                "id": s["id"],
                "image": s["image"],
                "full_img_path": s.get("full_img_path"),
                "conversations": s["conversations"],
                "metadata": s["metadata"],
            }
            for s in ds.samples
        ]

    def get_all_train_samples(self) -> List[Dict]:
        ds = MVTecJSONDataset(
            dataset_root=self.dataset_root,
            conversation_json_path=self.conversation_json_path,
            split=None,
            anomaly_only=False,
            with_bbox=True,
            check_exists=True,
        )
        return ds.samples

    def get_all_test_samples(self) -> List[Dict]:
        ds = MVTecJSONDataset(
            dataset_root=self.dataset_root,
            conversation_json_path=self.conversation_json_path,
            split=None,
            anomaly_only=False,
            with_bbox=True,
            check_exists=True,
        )
        return ds.samples