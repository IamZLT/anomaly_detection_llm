"""
MVTec grounding 训练用 Dataset（DINO/CLIP 桥 + HF processor/tokenizer）。

依赖 manager（实现 ``get_all_grounding_samples`` 等接口）提供样本 dict。
其他数据集请仿照本模块新建 ``data/<your>_grounding.py``。
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoProcessor

# manager uses duck-typing: must provide get_all_grounding_samples/get_all_train_samples/get_all_test_samples
from utils.common import (
    normalize_bbox_pixels_to_01,
    rewrite_bbox_tags_original_pixels_to_normalized_01,
    rewrite_bbox_tags_to_normalized_01,
    scale_bbox,
    smart_resize,
)


def _conv_role(conv: dict) -> str:
    r = conv.get("from") or conv.get("role") or ""
    return str(r).lower()


def _is_human_turn(conv: dict) -> bool:
    return _conv_role(conv) in ("human", "user")


def _is_gpt_turn(conv: dict) -> bool:
    return _conv_role(conv) in ("gpt", "assistant")


def expand_sample_to_single_turn_sft(sample: dict) -> list:
    """
    一条 JSON 样本（多轮 conversations）→ 多条样本，每条仅含一轮 [human, gpt]。
    无法拆成任何 (human, gpt) 对时保留原样一条，避免训练集被清空。
    """
    convs = sample.get("conversations") or []
    pairs: list[list] = []
    i = 0
    while i < len(convs):
        if _is_human_turn(convs[i]) and i + 1 < len(convs) and _is_gpt_turn(convs[i + 1]):
            pairs.append([convs[i], convs[i + 1]])
            i += 2
        else:
            i += 1
    if not pairs:
        return [sample]
    out = []
    base_id = sample.get("id", "sample")
    for t, pair in enumerate(pairs):
        ns = dict(sample)
        ns["conversations"] = pair
        ns["id"] = f"{base_id}__sft{t}"
        out.append(ns)
    return out


def expand_samples_single_turn_sft(samples: list) -> list:
    expanded: list = []
    for s in samples:
        expanded.extend(expand_sample_to_single_turn_sft(s))
    return expanded


def _longest_common_prefix_length_1d(a: torch.Tensor, b: torch.Tensor) -> int:
    """返回 a 与 b 在首部的相同 token 数（若 a 为 b 的前缀则返回 len(a)）。"""
    na, nb = int(a.numel()), int(b.numel())
    n = min(na, nb)
    i = 0
    while i < n and int(a[i].item()) == int(b[i].item()):
        i += 1
    return i


def _labels_assistant_tokens_only(
    *,
    input_ids_1d: torch.Tensor,
    prompt_input_ids_1d: torch.Tensor,
    pad_token_id: int | None,
) -> torch.Tensor:
    """
    仅对 assistant 段计算 loss：prompt（user + 模板至生成起点）对应位置 label=-100，pad=-100。
    """
    labels = input_ids_1d.clone().long()
    start = _longest_common_prefix_length_1d(prompt_input_ids_1d, labels)
    if start > 0:
        labels[:start] = -100
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    return labels


class MVTecQwenGroundingDataset(Dataset):
    def __init__(
        self,
        manager,
        processor: AutoProcessor,
        mode: str,
        max_length: int,
        max_image_size: int,
        factor: int,
        use_grounding_format: bool,
        dino_cfg: dict | None = None,
        clip_cfg: dict | None = None,
        local_files_only: bool = True,
        train_anomaly_only: bool = False,
        normalize_bbox_01: bool = False,
        train_gt_bbox_only: bool = False,
    ):
        self.manager = manager
        self.processor = processor
        # 可能是 AutoProcessor（含 .tokenizer）或已传入的 QwenTokenizerFast 本体
        self._tokenizer = getattr(processor, "tokenizer", processor)
        self.max_length = max_length
        self.max_image_size = max_image_size
        self.factor = factor
        self.use_grounding_format = use_grounding_format
        self.dino_cfg = dino_cfg or {}
        self.clip_cfg = clip_cfg or {}
        self.local_files_only = local_files_only
        self.normalize_bbox_01 = bool(normalize_bbox_01)
        self.train_gt_bbox_only = bool(train_gt_bbox_only)
        self.use_dino_bridge = bool(self.dino_cfg.get("enabled", True))
        self.dino_processor = None
        self.clip_processor = None
        if self.use_dino_bridge:
            self.dino_processor = AutoImageProcessor.from_pretrained(
                self.dino_cfg["model_path"],
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
            self.clip_processor = AutoImageProcessor.from_pretrained(
                self.clip_cfg["model_path"],
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )

        if use_grounding_format:
            self.samples = manager.get_all_grounding_samples(
                mode, anomaly_only=train_anomaly_only and mode == "train"
            )
        else:
            if mode == "train":
                self.samples = manager.dataset_loader.get_all_train_samples()
            else:
                self.samples = manager.dataset_loader.get_all_test_samples()
            if train_anomaly_only and mode == "train":
                self.samples = [s for s in self.samples if s.get("anomaly") == 1]

        # 训练阶段默认：多轮对话拆成多条「单轮 human + 紧随 gpt」SFT 样本；eval/test 不拆
        if mode == "train":
            self.samples = expand_samples_single_turn_sft(self.samples)

        if mode == "train" and self.train_gt_bbox_only and self.use_grounding_format:
            def _has_gt_bbox(s: dict) -> bool:
                bb = (s.get("metadata") or {}).get("bbox")
                return (
                    bb is not None
                    and isinstance(bb, (list, tuple))
                    and len(bb) == 4
                    and all(isinstance(x, (int, float)) for x in bb)
                )

            self.samples = [s for s in self.samples if _has_gt_bbox(s)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.use_grounding_format:
            img_path = sample.get("full_img_path") or sample["image"]
            conversations = sample["conversations"]
            meta = sample.get("metadata", {})
            original_bbox = meta.get("bbox")
            full_mask_path = meta.get("full_mask_path")
        else:
            img_path = sample["full_img_path"]
            conversations = sample["conversations"]
            original_bbox = None
            full_mask_path = sample.get("full_mask_path")

        if not os.path.isabs(img_path):
            dataset_root = sample.get("dataset_root") or self.manager.dataset_loader.dataset_root
            img_path = os.path.join(dataset_root, img_path)

        _scaled_bbox = None
        image_load_ok = False
        scale = (1.0, 1.0)
        try:
            image = Image.open(img_path).convert("RGB")
            image, _, scale = smart_resize(image, self.max_image_size, self.factor)
            _scaled_bbox = scale_bbox(original_bbox, scale)
            image_load_ok = True
        except Exception:
            image = Image.new("RGB", (self.max_image_size, self.max_image_size), "white")

        rw, rh = image.size
        norm_bbox_01: list[float] | None = None
        if (
            self.normalize_bbox_01
            and _scaled_bbox is not None
            and len(_scaled_bbox) == 4
        ):
            norm_bbox_01 = normalize_bbox_pixels_to_01(list(map(float, _scaled_bbox)), rw, rh)

        messages = []
        for conv in conversations:
            role = "user" if conv.get("from") in ("human", "user") or conv.get("role") == "user" else "assistant"
            content = conv.get("value") or conv.get("content", "")
            is_assistant = conv.get("from") in ("gpt", "assistant") or conv.get("role") == "assistant"
            # bbox_normalize_01：有 metadata.bbox 时用 GT 统一重写；仅有对话内 <bbox>（原图像素）时按 smart_resize 的 scale 单独归一化
            if self.normalize_bbox_01 and is_assistant:
                if norm_bbox_01 is not None:
                    content = rewrite_bbox_tags_to_normalized_01(str(content), norm_bbox_01)
                elif image_load_ok:
                    content = rewrite_bbox_tags_original_pixels_to_normalized_01(
                        str(content), scale, rw, rh
                    )
            if "<image>" in str(content):
                text = str(content).replace("<image>", "").strip()
                if self.use_dino_bridge:
                    messages.append({"role": role, "content": text})
                else:
                    messages.append(
                        {
                            "role": role,
                            "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
                        }
                    )
            else:
                messages.append({"role": role, "content": str(content)})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # 仅监督 assistant：用「除最后一轮 assistant 外的 messages + add_generation_prompt=True」与完整序列对齐求公共前缀长度
        prompt_input_ids_1d: torch.Tensor | None = None
        if len(messages) >= 2 and messages[-1].get("role") == "assistant":
            prompt_messages = messages[:-1]
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            if self.use_dino_bridge:
                enc_p = self.processor(
                    text=[prompt_text],
                    return_tensors="pt",
                    truncation=False,
                )
            else:
                enc_p = self.processor(
                    text=[prompt_text],
                    images=[image],
                    return_tensors="pt",
                    truncation=False,
                )
            prompt_input_ids_1d = enc_p["input_ids"][0]

        if self.use_dino_bridge:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
        else:
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )

        out = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.squeeze(0)
        pad_id = getattr(self._tokenizer, "pad_token_id", None)
        input_ids_1d = out["input_ids"]
        if prompt_input_ids_1d is not None:
            out["labels"] = _labels_assistant_tokens_only(
                input_ids_1d=input_ids_1d,
                prompt_input_ids_1d=prompt_input_ids_1d,
                pad_token_id=pad_id,
            )
        else:
            labels = input_ids_1d.clone().long()
            if pad_id is not None:
                labels[labels == pad_id] = -100
            out["labels"] = labels
        if self.use_dino_bridge and self.dino_processor is not None:
            dino_inputs = self.dino_processor(
                images=image,
                return_tensors="pt",
                do_resize=True,
                size={
                    "height": int(self.dino_cfg.get("image_size", 512)),
                    "width": int(self.dino_cfg.get("image_size", 512)),
                },
            )
            out["dino_pixel_values"] = dino_inputs["pixel_values"].squeeze(0)
            mask_size = int(self.dino_cfg.get("image_size", 512))
            if full_mask_path and os.path.exists(full_mask_path):
                try:
                    mask_img = Image.open(full_mask_path).convert("L")
                    mask_img = mask_img.resize((mask_size, mask_size), Image.NEAREST)
                    mask_np = (np.array(mask_img) > 0).astype(np.float32)
                except Exception:
                    mask_np = np.zeros((mask_size, mask_size), dtype=np.float32)
            else:
                mask_np = np.zeros((mask_size, mask_size), dtype=np.float32)
            out["mask_supervision"] = torch.from_numpy(mask_np).unsqueeze(0)

            clip_inputs = self.clip_processor(
                images=image,
                return_tensors="pt",
                do_resize=True,
                size={
                    "height": int(self.clip_cfg.get("image_size", 224)),
                    "width": int(self.clip_cfg.get("image_size", 224)),
                },
            )
            out["clip_pixel_values"] = clip_inputs["pixel_values"].squeeze(0)

        # Bbox 回归辅助损失：与 smart_resize 后图像对齐的 0–1 坐标（与 bbox_normalize_01 监督一致）
        if self.use_dino_bridge:
            if _scaled_bbox is not None and len(_scaled_bbox) == 4:
                rw, rh = image.size
                t01 = normalize_bbox_pixels_to_01(list(map(float, _scaled_bbox)), rw, rh)
                out["bbox_target"] = torch.tensor(t01, dtype=torch.float32)
                out["bbox_loss_mask"] = torch.tensor(1.0, dtype=torch.float32)
            else:
                out["bbox_target"] = torch.zeros(4, dtype=torch.float32)
                out["bbox_loss_mask"] = torch.tensor(0.0, dtype=torch.float32)
        else:
            out["bbox_target"] = torch.zeros(4, dtype=torch.float32)
            out["bbox_loss_mask"] = torch.tensor(0.0, dtype=torch.float32)
        return out


class MVTecDinoClipAlignDataset(Dataset):
    """
    MVTec + Stage-1：只从 JSON/manager 读路径，输出 dino_pixel_values / clip_pixel_values。
    """

    def __init__(
        self,
        manager,
        mode: str,
        max_image_size: int,
        factor: int,
        dino_cfg: dict | None = None,
        clip_cfg: dict | None = None,
        local_files_only: bool = True,
        anomaly_only: bool = False,
    ):
        self.manager = manager
        self.max_image_size = max_image_size
        self.factor = factor
        self.dino_cfg = dino_cfg or {}
        self.clip_cfg = clip_cfg or {}
        self.local_files_only = local_files_only

        self.dino_processor = AutoImageProcessor.from_pretrained(
            self.dino_cfg["model_path"],
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        self.clip_processor = AutoImageProcessor.from_pretrained(
            self.clip_cfg["model_path"],
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )

        self.samples = manager.get_all_grounding_samples(mode, anomaly_only=anomaly_only and mode == "train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample.get("full_img_path") or sample["image"]
        meta = sample.get("metadata", {}) or {}
        full_mask_path = meta.get("full_mask_path")

        dataset_root = sample.get("dataset_root") or self.manager.dataset_loader.dataset_root
        if not os.path.isabs(img_path):
            img_path = os.path.join(dataset_root, img_path)
        if full_mask_path and (not os.path.isabs(full_mask_path)):
            full_mask_path = os.path.join(dataset_root, full_mask_path)

        try:
            image = Image.open(img_path).convert("RGB")
            image, _, _ = smart_resize(image, self.max_image_size, self.factor)
        except Exception:
            image = Image.new("RGB", (self.max_image_size, self.max_image_size), "white")

        dino_inputs = self.dino_processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={
                "height": int(self.dino_cfg.get("image_size", 512)),
                "width": int(self.dino_cfg.get("image_size", 512)),
            },
        )
        clip_inputs = self.clip_processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={
                "height": int(self.clip_cfg.get("image_size", 224)),
                "width": int(self.clip_cfg.get("image_size", 224)),
            },
        )

        out = {
            "dino_pixel_values": dino_inputs["pixel_values"].squeeze(0),
            "clip_pixel_values": clip_inputs["pixel_values"].squeeze(0),
            "img_path": img_path,
        }
        out["mask_path"] = full_mask_path or ""

        mask_size = int(self.dino_cfg.get("image_size", 512))
        if full_mask_path and os.path.exists(full_mask_path):
            try:
                mask_img = Image.open(full_mask_path).convert("L")
                mask_img = mask_img.resize((mask_size, mask_size), Image.NEAREST)
                mask_np = (np.array(mask_img) > 0).astype(np.float32)
            except Exception:
                mask_np = np.zeros((mask_size, mask_size), dtype=np.float32)
        else:
            mask_np = np.zeros((mask_size, mask_size), dtype=np.float32)
        out["mask_supervision"] = torch.from_numpy(mask_np).unsqueeze(0)
        return out
