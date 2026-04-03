import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoProcessor

from data.load_mvtec_data import MVTecDataManager
from utils.qwen_common import scale_bbox, smart_resize


class MVTecQwenGroundingDataset(Dataset):
    def __init__(
        self,
        manager: MVTecDataManager,
        processor: AutoProcessor,
        mode: str,
        max_length: int,
        max_image_size: int,
        factor: int,
        use_grounding_format: bool,
        dino_cfg: dict | None = None,
        clip_cfg: dict | None = None,
        local_files_only: bool = True,
    ):
        self.manager = manager
        self.processor = processor
        self.max_length = max_length
        self.max_image_size = max_image_size
        self.factor = factor
        self.use_grounding_format = use_grounding_format
        self.dino_cfg = dino_cfg or {}
        self.clip_cfg = clip_cfg or {}
        self.local_files_only = local_files_only
        self.use_dino_bridge = bool(self.dino_cfg.get("enabled", True))
        self.use_clip_bridge = bool(self.clip_cfg.get("enabled", True))
        self.dino_processor = None
        self.clip_processor = None
        if self.use_dino_bridge:
            self.dino_processor = AutoImageProcessor.from_pretrained(
                self.dino_cfg["model_path"],
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
        if self.use_dino_bridge and self.use_clip_bridge:
            self.clip_processor = AutoImageProcessor.from_pretrained(
                self.clip_cfg["model_path"],
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )

        if use_grounding_format:
            self.samples = manager.get_all_grounding_samples(mode)
        else:
            if mode == "train":
                self.samples = manager.dataset_loader.get_all_train_samples()
            else:
                self.samples = manager.dataset_loader.get_all_test_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.use_grounding_format:
            img_path = sample["image"]
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
            img_path = os.path.join(self.manager.dataset_loader.dataset_root, img_path)

        try:
            image = Image.open(img_path).convert("RGB")
            image, _, scale = smart_resize(image, self.max_image_size, self.factor)
            _scaled_bbox = scale_bbox(original_bbox, scale)
        except Exception:
            image = Image.new("RGB", (self.max_image_size, self.max_image_size), "white")

        messages = []
        for conv in conversations:
            role = "user" if conv.get("from") in ("human", "user") or conv.get("role") == "user" else "assistant"
            content = conv.get("value") or conv.get("content", "")
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
        labels = out["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
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

            if self.use_clip_bridge and self.clip_processor is not None:
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
        return out

