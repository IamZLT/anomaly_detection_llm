import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoProcessor


def get_model_class():
    try:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    except Exception as e:
        raise ImportError(
            "当前 transformers 环境不支持 Qwen3VLForConditionalGeneration，请升级 transformers。"
        ) from e


class QwenDinoBridgeModel(nn.Module):
    def __init__(self, base_model: nn.Module, cfg: dict):
        super().__init__()
        self.base_model = base_model
        self.cfg = cfg
        self.last_loss_stats = {}
        dino_cfg = cfg.get("dino", {})
        clip_cfg = cfg.get("clip", {})
        local_files_only = cfg.get("model", {}).get("local_files_only", False)

        self.dino_image_size = int(dino_cfg.get("image_size", 512))
        self.freeze_dino = bool(dino_cfg.get("freeze", True))
        self.use_mlp = bool(dino_cfg.get("use_mlp", True))
        self.dino_layer_indices = [int(x) for x in dino_cfg.get("layer_indices", [12, 16, 20, 24])]

        dino_model_path = dino_cfg["model_path"]
        self.dino_model = AutoModel.from_pretrained(
            dino_model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        dino_hidden = int(getattr(self.dino_model.config, "hidden_size", 1024))
        self.dino_patch_size = int(getattr(self.dino_model.config, "patch_size", 16))
        llm_hidden = int(self.base_model.config.text_config.hidden_size)
        self.num_register_tokens = int(getattr(self.dino_model.config, "num_register_tokens", 0))
        self.mask_loss_weight = float(dino_cfg.get("mask_loss_weight", 0.5))

        self.dino_layer_projs = nn.ModuleList(
            [self._make_projector(dino_hidden, dino_hidden, self.use_mlp) for _ in self.dino_layer_indices]
        )
        self.dino_layer_logits = nn.Parameter(torch.zeros(len(self.dino_layer_indices), dtype=torch.float32))
        self.dino_to_llm_projector = self._make_projector(dino_hidden, llm_hidden, self.use_mlp)

        self.use_clip = bool(clip_cfg.get("enabled", True))
        self.clip_image_size = int(clip_cfg.get("image_size", 224))
        self.freeze_clip = bool(clip_cfg.get("freeze", True))
        self.clip_use_mlp = bool(clip_cfg.get("use_mlp", True))
        self.clip_model = None
        self.clip_to_llm_projector = None
        self.dino_to_clip_projector = None
        self.mask_clip_gate = None
        self.mask_to_llm_projector = None
        if self.use_clip:
            clip_model_path = clip_cfg["model_path"]
            self.clip_model = AutoModel.from_pretrained(
                clip_model_path,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            clip_hidden = int(
                getattr(
                    self.clip_model.config,
                    "projection_dim",
                    getattr(self.clip_model.config, "hidden_size", 768),
                )
            )
            self.clip_to_llm_projector = self._make_projector(clip_hidden, llm_hidden, self.clip_use_mlp)
            self.dino_to_clip_projector = self._make_projector(dino_hidden, clip_hidden, self.use_mlp)
            self.mask_clip_gate = nn.Sequential(
                nn.Linear(clip_hidden * 2, clip_hidden),
                nn.Sigmoid(),
            )
            self.mask_to_llm_projector = self._make_projector(clip_hidden, llm_hidden, self.clip_use_mlp)
            self.mask_patch_head = nn.Sequential(
                nn.Linear(clip_hidden * 2, clip_hidden),
                nn.GELU(),
                nn.Linear(clip_hidden, 1),
            )

        if self.freeze_dino:
            self.dino_model.eval()
            for p in self.dino_model.parameters():
                p.requires_grad = False
        if self.use_clip and self.freeze_clip:
            self.clip_model.eval()
            for p in self.clip_model.parameters():
                p.requires_grad = False

    @staticmethod
    def _make_projector(in_dim: int, out_dim: int, use_mlp: bool) -> nn.Module:
        if use_mlp:
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
        return nn.Linear(in_dim, out_dim)

    @property
    def config(self):
        return self.base_model.config

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                return self.base_model.gradient_checkpointing_enable()
            return self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        return None

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            return self.base_model.gradient_checkpointing_disable()
        return None

    def enable_input_require_grads(self):
        if hasattr(self.base_model, "enable_input_require_grads"):
            return self.base_model.enable_input_require_grads()
        return None

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def get_last_loss_stats(self):
        return self.last_loss_stats

    def _encode_dino(self, dino_pixel_values: torch.Tensor, target_device: torch.device):
        x = dino_pixel_values.to(device=target_device, dtype=torch.float32)
        if x.ndim != 4:
            raise ValueError(f"dino_pixel_values 需要 [B,3,H,W]，当前 shape={tuple(x.shape)}")
        if next(self.dino_model.parameters()).device != target_device:
            self.dino_model.to(target_device)

        h, w = x.shape[-2], x.shape[-1]
        if h != self.dino_image_size or w != self.dino_image_size:
            x = F.interpolate(
                x,
                size=(self.dino_image_size, self.dino_image_size),
                mode="bilinear",
                align_corners=False,
            )

        model_kwargs = {"pixel_values": x, "output_hidden_states": True}
        if self.freeze_dino:
            with torch.no_grad():
                dino_out = self.dino_model(**model_kwargs)
        else:
            dino_out = self.dino_model(**model_kwargs)

        hidden_states = dino_out.hidden_states
        if not hidden_states:
            raise RuntimeError("DINO forward 未返回 hidden_states，无法做多层融合。")

        per_layer_tokens = []
        per_layer_patches = []
        for i, li in enumerate(self.dino_layer_indices):
            idx = max(0, min(int(li), len(hidden_states) - 1))
            feat = hidden_states[idx]
            cls = feat[:, 0, :]
            patch_start = min(1 + self.num_register_tokens, feat.shape[1] - 1)
            patches = feat[:, patch_start:, :]
            b, p, d = patches.shape
            proj_patch = self.dino_layer_projs[i](patches.reshape(b * p, d)).reshape(b, p, d)
            patch_mean = proj_patch.mean(dim=1) if proj_patch.numel() > 0 else cls
            tok = 0.5 * (self.dino_layer_projs[i](cls) + patch_mean)
            per_layer_tokens.append(tok)
            per_layer_patches.append(proj_patch)

        layer_stack = torch.stack(per_layer_tokens, dim=0)
        patch_stack = torch.stack(per_layer_patches, dim=0)
        layer_weights = F.softmax(self.dino_layer_logits, dim=0).view(-1, 1, 1)
        fused = (layer_stack * layer_weights).sum(dim=0)
        patch_weights = F.softmax(self.dino_layer_logits, dim=0).view(-1, 1, 1, 1)
        fused_patches = (patch_stack * patch_weights).sum(dim=0)
        return fused, fused_patches

    def _encode_clip(self, clip_pixel_values: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        if self.clip_model is None:
            raise RuntimeError("clip 分支未初始化。")
        x = clip_pixel_values.to(device=target_device, dtype=torch.float32)
        if x.ndim != 4:
            raise ValueError(f"clip_pixel_values 需要 [B,3,H,W]，当前 shape={tuple(x.shape)}")
        if next(self.clip_model.parameters()).device != target_device:
            self.clip_model.to(target_device)

        h, w = x.shape[-2], x.shape[-1]
        if h != self.clip_image_size or w != self.clip_image_size:
            x = F.interpolate(x, size=(self.clip_image_size, self.clip_image_size), mode="bilinear", align_corners=False)

        # CLIPModel.forward 通常同时期望 text + image。
        # 这里只需要视觉语义，优先走 get_image_features 以避免 input_ids 报错。
        if hasattr(self.clip_model, "get_image_features"):
            if self.freeze_clip:
                with torch.no_grad():
                    clip_feat = self.clip_model.get_image_features(pixel_values=x)
            else:
                clip_feat = self.clip_model.get_image_features(pixel_values=x)
        elif hasattr(self.clip_model, "vision_model"):
            if self.freeze_clip:
                with torch.no_grad():
                    vision_out = self.clip_model.vision_model(pixel_values=x)
            else:
                vision_out = self.clip_model.vision_model(pixel_values=x)
            if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
                clip_feat = vision_out.pooler_output
            else:
                clip_feat = vision_out.last_hidden_state[:, 0, :]
            if hasattr(self.clip_model, "visual_projection"):
                clip_feat = self.clip_model.visual_projection(clip_feat)
        else:
            raise RuntimeError("当前 CLIP 模型不支持 image-only 特征提取。")
        return clip_feat

    def _build_inputs_embeds(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        dino_pixel_values: torch.Tensor,
        clip_pixel_values: Optional[torch.Tensor],
        mask_supervision: Optional[torch.Tensor],
    ):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("必须提供 input_ids 或 inputs_embeds")
            token_embeds = self.base_model.get_input_embeddings()(input_ids)
        else:
            token_embeds = inputs_embeds

        dino_feat_raw, dino_patch_raw = self._encode_dino(dino_pixel_values, token_embeds.device)
        dino_feat_raw = dino_feat_raw.to(dtype=token_embeds.dtype)
        dino_patch_raw = dino_patch_raw.to(dtype=token_embeds.dtype)
        dino_token = self.dino_to_llm_projector(dino_feat_raw)
        prefix_tokens = [dino_token.unsqueeze(1)]
        aux_mask_loss = None
        if self.use_clip and clip_pixel_values is not None:
            clip_feat_raw = self._encode_clip(clip_pixel_values, token_embeds.device).to(dtype=token_embeds.dtype)
            clip_token = self.clip_to_llm_projector(clip_feat_raw)
            dino_in_clip = self.dino_to_clip_projector(dino_feat_raw)
            gate = self.mask_clip_gate(torch.cat([dino_in_clip, clip_feat_raw], dim=-1))
            mask_clip_feat = gate * dino_in_clip + (1.0 - gate) * clip_feat_raw
            mask_token = self.mask_to_llm_projector(mask_clip_feat)
            prefix_tokens = [dino_token.unsqueeze(1), clip_token.unsqueeze(1), mask_token.unsqueeze(1)]

            if mask_supervision is not None and hasattr(self, "mask_patch_head"):
                b, p, d = dino_patch_raw.shape
                patch_in_clip = self.dino_to_clip_projector(dino_patch_raw.reshape(b * p, d)).reshape(b, p, -1)
                clip_expand = clip_feat_raw.unsqueeze(1).expand(-1, p, -1)
                patch_logits = self.mask_patch_head(torch.cat([patch_in_clip, clip_expand], dim=-1)).squeeze(-1)

                if mask_supervision.ndim == 3:
                    mask_supervision = mask_supervision.unsqueeze(1)
                mask_supervision = mask_supervision.to(device=patch_logits.device, dtype=patch_logits.dtype)

                grid_h = max(1, dino_pixel_values.shape[-2] // self.dino_patch_size)
                grid_w = max(1, dino_pixel_values.shape[-1] // self.dino_patch_size)
                if grid_h * grid_w != p:
                    side = int(p ** 0.5)
                    grid_h = side
                    grid_w = p // max(side, 1)

                target = F.interpolate(mask_supervision, size=(grid_h, grid_w), mode="nearest").reshape(b, -1)
                patch_logits = patch_logits[:, : target.shape[1]]
                target = target[:, : patch_logits.shape[1]]
                aux_mask_loss = F.binary_cross_entropy_with_logits(patch_logits, target)

        bridge_tokens = torch.cat(prefix_tokens, dim=1)
        prefix_len = bridge_tokens.shape[1]
        merged_embeds = torch.cat([bridge_tokens, token_embeds], dim=1)

        merged_mask = attention_mask
        if merged_mask is not None:
            prefix = torch.ones((merged_mask.shape[0], prefix_len), dtype=merged_mask.dtype, device=merged_mask.device)
            merged_mask = torch.cat([prefix, merged_mask], dim=1)

        merged_labels = labels
        if merged_labels is not None:
            prefix_label = torch.full(
                (merged_labels.shape[0], prefix_len),
                -100,
                dtype=merged_labels.dtype,
                device=merged_labels.device,
            )
            merged_labels = torch.cat([prefix_label, merged_labels], dim=1)

        return merged_embeds, merged_mask, merged_labels, aux_mask_loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        inputs_embeds=None,
        dino_pixel_values=None,
        clip_pixel_values=None,
        mask_supervision=None,
        **kwargs,
    ):
        if dino_pixel_values is None:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        merged_embeds, merged_mask, merged_labels, aux_mask_loss = self._build_inputs_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            dino_pixel_values=dino_pixel_values,
            clip_pixel_values=clip_pixel_values,
            mask_supervision=mask_supervision,
        )

        kwargs.pop("pixel_values", None)
        kwargs.pop("image_grid_thw", None)
        outputs = self.base_model(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            labels=merged_labels,
            **kwargs,
        )
        if hasattr(outputs, "loss") and outputs.loss is not None:
            lm_loss = outputs.loss
            mask_loss = aux_mask_loss if aux_mask_loss is not None else lm_loss.new_zeros(())
            total_loss = lm_loss + self.mask_loss_weight * mask_loss
            outputs.loss = total_loss
            self.last_loss_stats = {
                "loss_total": float(total_loss.detach().float().item()),
                "loss_lm": float(lm_loss.detach().float().item()),
                "loss_mask": float(mask_loss.detach().float().item()),
            }
            try:
                outputs["loss_lm"] = lm_loss.detach()
                outputs["loss_mask"] = mask_loss.detach()
            except Exception:
                pass
        return outputs

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        dino_pixel_values=None,
        clip_pixel_values=None,
        **kwargs,
    ):
        if dino_pixel_values is None:
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        merged_embeds, merged_mask, _ = self._build_inputs_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=None,
            dino_pixel_values=dino_pixel_values,
            clip_pixel_values=clip_pixel_values,
            mask_supervision=None,
        )
        kwargs.pop("pixel_values", None)
        kwargs.pop("image_grid_thw", None)
        return self.base_model.generate(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            **kwargs,
        )

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.base_model.save_pretrained(save_directory, **kwargs)
        bridge_state = {}
        for name, tensor in self.state_dict().items():
            if name.startswith("dino_") or name.startswith("clip_") or name.startswith("mask_"):
                bridge_state[name] = tensor.detach().cpu()
        bridge_payload = {
            "state_dict": bridge_state,
            "dino_image_size": self.dino_image_size,
            "use_mlp": self.use_mlp,
            "dino_layer_indices": self.dino_layer_indices,
            "clip_enabled": self.use_clip,
        }
        torch.save(bridge_payload, os.path.join(save_directory, "dino_bridge.bin"))

    def load_bridge(self, ckpt_dir: str):
        ckpt_path = os.path.join(ckpt_dir, "dino_bridge.bin")
        if not os.path.exists(ckpt_path):
            return
        payload = torch.load(ckpt_path, map_location="cpu")
        state = payload.get("state_dict", payload)
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[Bridge] load non-strict, missing={len(missing)}, unexpected={len(unexpected)}")


def setup_model_and_processor(
    cfg: dict,
    for_inference: bool = False,
    model_name_override: Optional[str] = None,
) -> Tuple[torch.nn.Module, AutoProcessor]:
    model_name = model_name_override or cfg["model"]["name"]
    local_files_only = cfg.get("model", {}).get("local_files_only", False)
    use_dino_bridge = bool(cfg.get("dino", {}).get("enabled", True))
    model_cls = get_model_class()

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=local_files_only,
    )

    num_gpus = torch.cuda.device_count()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        num_gpus = len([x for x in cuda_visible.split(",") if x.strip()])
    device_map = None if (num_gpus > 1 and not for_inference) else "auto"

    model = model_cls.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=True,
        dtype=torch.bfloat16 if cfg["training"]["bf16"] else torch.float32,
        local_files_only=local_files_only,
    )

    for p in model.parameters():
        if not p.requires_grad:
            p.requires_grad = True

    if cfg["lora"]["enabled"]:
        possible_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        actual_target_modules = []
        for module_name in possible_target_modules:
            for name, module in model.named_modules():
                if (
                    name.endswith(module_name)
                    and isinstance(module, nn.Linear)
                    and "vision" not in name.lower()
                    and "visual" not in name.lower()
                ):
                    actual_target_modules.append(module_name)
                    break

        target_modules = actual_target_modules if actual_target_modules else "all-linear"
        lora_config = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            target_modules=target_modules,
            lora_dropout=cfg["lora"]["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if cfg["model"]["freeze_vit"]:
        for name, p in model.named_parameters():
            if "vision_model" in name or "visual" in name:
                p.requires_grad = False

    if cfg["training"]["gradient_checkpointing"]:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif hasattr(model, "get_input_embeddings"):

            def _hook(_module, _inp, out):
                out.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(_hook)

    if use_dino_bridge:
        model = QwenDinoBridgeModel(base_model=model, cfg=cfg)
        if for_inference:
            model.load_bridge(model_name)

    if for_inference:
        model.eval()
    else:
        model.train()
    return model, processor

