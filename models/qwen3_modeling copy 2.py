import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from utils.dinov3_utils import dinov3_encode_image
from models.visual_proto import ResidualVisualProjection, MultiLayerCLSProjection, MultiLayerPatchProjection


def _is_main_process() -> bool:
    r = os.environ.get("RANK")
    if r is not None:
        return int(r) == 0
    lr = os.environ.get("LOCAL_RANK")
    if lr is not None:
        return int(lr) == 0
    return True


def _make_projector(in_dim: int, out_dim: int, use_mlp: bool = True) -> nn.Module:
    if use_mlp:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    return nn.Linear(in_dim, out_dim)

class CrossAttentionTokenCompressor(nn.Module):
    """
    不截断 patch，只做软压缩：
    - 输入全量视觉 token
    - 用固定数量的 learned latents 对全序列做 cross-attention
    - 输出固定数量的视觉 token 给 LLM
    """

    def __init__(self, hidden_size: int, num_latents: int, num_heads: int, use_mlp: bool = True):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_size) * 0.02)
        self.q_norm = nn.LayerNorm(hidden_size)
        self.kv_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        if use_mlp:
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )
        else:
            self.ffn = nn.Linear(hidden_size, hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz = tokens.shape[0]
        latents = self.latents.expand(bsz, -1, -1)
        q = self.q_norm(latents)
        kv = self.kv_norm(tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        latents = latents + attn_out
        latents = latents + self.ffn(self.out_norm(latents))
        return latents


class QwenDinoBridgeModel(nn.Module):
    """
    纯文本 Qwen3（CausalLM）+ DINO/CLIP bridge
    Stage-2 最终版：
    1) 冻结 DINO / CLIP / Stage-1 mapper
    2) 使用三路 patch token：
       - dino_patches   : 原始精细局部特征（DINO 空间）
       - mapped_patches : Stage-1 学到的 DINO -> CLIP 桥接特征
       - clip_patches   : CLIP 局部语义特征
    3) 三路各自先过 MLP 到 LLM hidden，再拼接
    4) 不做任何 patch 截断；拼接后通过软压缩模块压成固定长度视觉前缀
    5) 训练只保留 LM loss
    """

    def __init__(self, base_model: nn.Module, cfg: dict):
        super().__init__()
        self.base_model = base_model
        self.cfg = cfg
        self.last_loss_stats: Dict[str, float] = {}

        model_cfg = cfg.get("model", {}) or {}
        dino_cfg = cfg.get("dino", {}) or {}
        clip_cfg = cfg.get("clip", {}) or {}
        bridge_cfg = cfg.get("bridge", {}) or {}
        local_files_only = bool(model_cfg.get("local_files_only", False))

        # =========================
        # 1. DINO（冻结）
        # =========================
        self.dino_image_size = int(dino_cfg.get("image_size", 512))
        self.use_mlp = bool(dino_cfg.get("use_mlp", True))
        self.dino_layer_indices = [int(x) for x in dino_cfg.get("layer_indices", [12, 16, 20, 24])]

        self.dino_model = AutoModel.from_pretrained(
            dino_cfg["model_path"],
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.dino_hidden = int(getattr(self.dino_model.config, "hidden_size", 1024))
        self.num_register_tokens = int(getattr(self.dino_model.config, "num_register_tokens", 0))
        self.dino_model.eval()
        for p in self.dino_model.parameters():
            p.requires_grad = False

        # =========================
        # 2. CLIP（冻结）
        # =========================
        self.clip_image_size = int(clip_cfg.get("image_size", 224))
        self.clip_use_mlp = bool(clip_cfg.get("use_mlp", True))

        self.clip_model = AutoModel.from_pretrained(
            clip_cfg["model_path"],
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.clip_hidden = int(
            getattr(
                self.clip_model.config,
                "projection_dim",
                getattr(self.clip_model.config, "hidden_size", 768),
            )
        )
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # =========================
        # 3. LLM 维度
        # =========================
        llm_hidden = int(getattr(self.base_model.config, "hidden_size", 0))
        if llm_hidden <= 0:
            raise ValueError("无法从纯文本 LLM 配置中读取 hidden_size")
        self.llm_hidden = llm_hidden
        num_layers = len(self.dino_layer_indices)

        # =========================
        # 4. Stage-1 mapper（命名与 step1 保持一致：cls_mapper / patch_mapper）
        # =========================
        self.cls_mapper = MultiLayerCLSProjection(
            vis_dim=self.dino_hidden,
            output_dim=self.clip_hidden,
            num_layers=num_layers,
            init_layer_indices=self.dino_layer_indices,
        )
        self.patch_mapper = MultiLayerPatchProjection(
            vis_dim=self.dino_hidden,
            output_dim=self.clip_hidden,
            num_layers=num_layers,
            init_layer_indices=self.dino_layer_indices,
        )

        # 兼容旧命名（历史版本可能使用这两个属性名）
        self.dino_to_clip_mapper = self.cls_mapper
        self.dino_patch_to_clip = self.patch_mapper

        # =========================
        # 5. Raw DINO patch 的多层 softmax 融合（保留细节，不映射到 CLIP）
        # =========================
        self.dino_raw_layer_logits = nn.Parameter(torch.zeros(num_layers, dtype=torch.float32))
        if len(self.dino_layer_indices) == num_layers:
            init_vals = torch.tensor(self.dino_layer_indices, dtype=torch.float32)
            init_vals = init_vals - init_vals.mean()
            with torch.no_grad():
                self.dino_raw_layer_logits.copy_(init_vals)

        # =========================
        # 6. 三路各自 MLP -> LLM hidden
        # =========================
        self.dino_patch_to_llm = _make_projector(self.dino_hidden, llm_hidden, self.use_mlp)
        self.mapped_patch_to_llm = _make_projector(self.clip_hidden, llm_hidden, self.use_mlp)
        self.clip_patch_to_llm = _make_projector(self.clip_hidden, llm_hidden, self.clip_use_mlp)

        self.dino_source_embed = nn.Parameter(torch.zeros(1, 1, llm_hidden))
        self.mapped_source_embed = nn.Parameter(torch.zeros(1, 1, llm_hidden))
        self.clip_source_embed = nn.Parameter(torch.zeros(1, 1, llm_hidden))

        # 可选：每个分支内部保留顺序信息
        self.use_patch_pos_embed = bool(dino_cfg.get("use_patch_pos_embed", False))
        self.max_patch_pos = int(dino_cfg.get("max_patch_pos", 4096))
        if self.use_patch_pos_embed:
            self.dino_patch_pos_embed = nn.Parameter(torch.zeros(1, self.max_patch_pos, llm_hidden))
            self.mapped_patch_pos_embed = nn.Parameter(torch.zeros(1, self.max_patch_pos, llm_hidden))
            self.clip_patch_pos_embed = nn.Parameter(torch.zeros(1, self.max_patch_pos, llm_hidden))
            nn.init.normal_(self.dino_patch_pos_embed, mean=0.0, std=0.02)
            nn.init.normal_(self.mapped_patch_pos_embed, mean=0.0, std=0.02)
            nn.init.normal_(self.clip_patch_pos_embed, mean=0.0, std=0.02)

        # =========================
        # 7. 拼接后共享融合 MLP + 软压缩器
        # =========================
        self.post_concat_mlp = nn.Sequential(
            nn.LayerNorm(llm_hidden),
            nn.Linear(llm_hidden, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        )

        self.num_visual_tokens = int(bridge_cfg.get("num_visual_tokens", 256))
        self.compressor_num_heads = int(bridge_cfg.get("compressor_num_heads", 8))
        self.visual_compressor = CrossAttentionTokenCompressor(
            hidden_size=llm_hidden,
            num_latents=self.num_visual_tokens,
            num_heads=self.compressor_num_heads,
            use_mlp=True,
        )
        self.post_compress_norm = nn.LayerNorm(llm_hidden)

        # =========================
        # 8. 训练策略
        # =========================
        for p in self.base_model.parameters():
            p.requires_grad = True
        for p in self.cls_mapper.parameters():
            p.requires_grad = False
        for p in self.patch_mapper.parameters():
            p.requires_grad = False

    @property
    def config(self):
        return self.base_model.config

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        fn = getattr(self.base_model, "gradient_checkpointing_enable", None)
        if fn is None:
            return None
        if gradient_checkpointing_kwargs is None:
            return fn()
        return fn(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        fn = getattr(self.base_model, "gradient_checkpointing_disable", None)
        return fn() if fn is not None else None

    def enable_input_require_grads(self):
        fn = getattr(self.base_model, "enable_input_require_grads", None)
        return fn() if fn is not None else None

    def get_last_loss_stats(self):
        return self.last_loss_stats

    def print_trainable_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"总参数量: {total / 1e6:.2f} M")
        print(f"可训练参数量: {trainable / 1e6:.2f} M")
        print(f"可训练占比: {100 * trainable / total:.2f}%")

    def _encode_dino(self, dino_pixel_values: torch.Tensor, device: torch.device):
        """
        返回：
            cls_stack   : [L, B, D]
            patch_stack : [L, B, P, D]  (供 Stage-1 patch mapper 使用)
            dino_patches: [B, P, D]     (raw DINO patch，softmax 层融合)
        """
        x = dino_pixel_values.to(device=device, dtype=torch.float32)
        if x.ndim != 4:
            raise ValueError(f"dino_pixel_values 需要 [B,3,H,W]，当前 shape={tuple(x.shape)}")
        if x.shape[-2:] != (self.dino_image_size, self.dino_image_size):
            x = F.interpolate(x, size=(self.dino_image_size, self.dino_image_size), mode="bilinear", align_corners=False)

        if next(self.dino_model.parameters()).device != device:
            self.dino_model.to(device)

        dino_out = dinov3_encode_image(
            x,
            processor=None,
            model=self.dino_model,
            device=device,
            layer_indices=self.dino_layer_indices,
        )
        if "multi_layer_features" not in dino_out:
            raise RuntimeError("dinov3_encode_image 必须返回 multi_layer_features")

        mlf = dino_out["multi_layer_features"]  # list of [B, 1+P, D]
        cls_stack = torch.stack([feat[:, 0, :] for feat in mlf], dim=0)     # [L, B, D]
        patch_stack = torch.stack([feat[:, 1:, :] for feat in mlf], dim=0)  # [L, B, P, D]

        w = F.softmax(self.dino_raw_layer_logits[: patch_stack.shape[0]], dim=0).view(-1, 1, 1, 1)
        dino_patches = (patch_stack * w).sum(dim=0).contiguous()  # [B, P, D]
        return cls_stack, patch_stack, dino_patches

    def _encode_clip(self, clip_pixel_values: torch.Tensor, device: torch.device):
        """
        返回：
            clip_patches: [B, P, C]
        """
        x = clip_pixel_values.to(device=device, dtype=torch.float32)
        if x.ndim != 4:
            raise ValueError(f"clip_pixel_values 需要 [B,3,H,W]，当前 shape={tuple(x.shape)}")
        if x.shape[-2:] != (self.clip_image_size, self.clip_image_size):
            x = F.interpolate(x, size=(self.clip_image_size, self.clip_image_size), mode="bilinear", align_corners=False)

        if next(self.clip_model.parameters()).device != device:
            self.clip_model.to(device)

        with torch.no_grad():
            if hasattr(self.clip_model, "vision_model"):
                vision_out = self.clip_model.vision_model(pixel_values=x)
                patch_tokens = vision_out.last_hidden_state[:, 1:, :]
                if hasattr(self.clip_model, "visual_projection"):
                    patch_tokens = self.clip_model.visual_projection(patch_tokens)
                return patch_tokens

            if hasattr(self.clip_model, "get_image_features"):
                feat = self.clip_model.get_image_features(pixel_values=x)
                return torch.empty((feat.shape[0], 0, feat.shape[-1]), device=feat.device, dtype=feat.dtype)

        raise RuntimeError("当前 CLIP 模型不支持 image-only 特征提取。")

    def _add_branch_pos(self, x: torch.Tensor, branch: str) -> torch.Tensor:
        if not self.use_patch_pos_embed:
            return x
        if x.shape[1] > self.max_patch_pos:
            raise ValueError(
                f"{branch} patch token 数 {x.shape[1]} 超过 max_patch_pos={self.max_patch_pos}，"
                "请调大 dino.max_patch_pos 或关闭 use_patch_pos_embed。"
            )
        pos = getattr(self, f"{branch}_patch_pos_embed")[:, : x.shape[1], :]
        return x + pos

    def _build_visual_tokens(
        self,
        dino_pixel_values: torch.Tensor,
        clip_pixel_values: torch.Tensor,
        device: torch.device,
    ):
        cls_stack, patch_stack, dino_patches = self._encode_dino(dino_pixel_values, device)
        clip_patches = self._encode_clip(clip_pixel_values, device)

        # Stage-1 mapper（冻结）
        _ = self.cls_mapper(cls_stack)  # 保留兼容；当前不直接作为 visual token 使用
        mapped_patches = self.patch_mapper(patch_stack)  # [B, P, C]

        # 三路先各自 MLP 到 LLM hidden，再拼接
        dino_patch_tokens = self.dino_patch_to_llm(dino_patches) + self.dino_source_embed
        mapped_patch_tokens = self.mapped_patch_to_llm(mapped_patches) + self.mapped_source_embed
        clip_patch_tokens = self.clip_patch_to_llm(clip_patches) + self.clip_source_embed

        dino_patch_tokens = self._add_branch_pos(dino_patch_tokens, "dino")
        mapped_patch_tokens = self._add_branch_pos(mapped_patch_tokens, "mapped")
        clip_patch_tokens = self._add_branch_pos(clip_patch_tokens, "clip")

        all_patch_tokens = torch.cat(
            [dino_patch_tokens, mapped_patch_tokens, clip_patch_tokens],
            dim=1,
        )
        all_patch_tokens = all_patch_tokens + self.post_concat_mlp(all_patch_tokens)
        bridge_tokens = self.post_compress_norm(self.visual_compressor(all_patch_tokens))

        aux = {
            "bridge_num_tokens": bridge_tokens.shape[1],
            "num_dino_patch_tokens": dino_patch_tokens.shape[1],
            "num_mapped_patch_tokens": mapped_patch_tokens.shape[1],
            "num_clip_patch_tokens": clip_patch_tokens.shape[1],
            "raw_visual_tokens": all_patch_tokens.shape[1],
        }
        return bridge_tokens, aux

    def _build_inputs_embeds(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        dino_pixel_values: torch.Tensor,
        clip_pixel_values: torch.Tensor,
    ):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("必须提供 input_ids 或 inputs_embeds")
            token_embeds = self.base_model.get_input_embeddings()(input_ids)
        else:
            token_embeds = inputs_embeds

        device = token_embeds.device
        dtype = token_embeds.dtype

        bridge_tokens, aux = self._build_visual_tokens(
            dino_pixel_values=dino_pixel_values,
            clip_pixel_values=clip_pixel_values,
            device=device,
        )
        bridge_tokens = bridge_tokens.to(dtype=dtype)

        merged_embeds = torch.cat([bridge_tokens, token_embeds], dim=1)
        prefix_len = bridge_tokens.shape[1]

        merged_mask = attention_mask
        if merged_mask is not None:
            prefix_mask = torch.ones(
                (merged_mask.shape[0], prefix_len),
                dtype=merged_mask.dtype,
                device=merged_mask.device,
            )
            merged_mask = torch.cat([prefix_mask, merged_mask], dim=1)

        merged_labels = labels
        if merged_labels is not None:
            prefix_labels = torch.full(
                (merged_labels.shape[0], prefix_len),
                -100,
                dtype=merged_labels.dtype,
                device=merged_labels.device,
            )
            merged_labels = torch.cat([prefix_labels, merged_labels], dim=1)

        return merged_embeds, merged_mask, merged_labels, aux

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        inputs_embeds=None,
        dino_pixel_values=None,
        clip_pixel_values=None,
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

        if clip_pixel_values is None:
            raise ValueError("clip_pixel_values 为必需：桥接固定使用 CLIP，请与 dino_pixel_values 一并传入。")

        merged_embeds, merged_mask, merged_labels, aux = self._build_inputs_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            dino_pixel_values=dino_pixel_values,
            clip_pixel_values=clip_pixel_values,
        )

        kwargs.pop("pixel_values", None)
        kwargs.pop("image_grid_thw", None)
        kwargs.pop("input_ids", None)
        kwargs.pop("attention_mask", None)
        kwargs.pop("labels", None)
        kwargs.pop("dino_pixel_values", None)
        kwargs.pop("clip_pixel_values", None)
        kwargs.pop("mask_supervision", None)

        outputs = self.base_model(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            labels=merged_labels,
            **kwargs,
        )

        if getattr(outputs, "loss", None) is not None:
            lm_loss = outputs.loss
            self.last_loss_stats = {
                "loss_total": float(lm_loss.detach().float().item()),
                "loss_lm": float(lm_loss.detach().float().item()),
                "bridge_tokens": float(aux["bridge_num_tokens"]),
                "raw_visual_tokens": float(aux["raw_visual_tokens"]),
                "num_dino_patch_tokens": float(aux["num_dino_patch_tokens"]),
                "num_mapped_patch_tokens": float(aux["num_mapped_patch_tokens"]),
                "num_clip_patch_tokens": float(aux["num_clip_patch_tokens"]),
            }
        return outputs

    @torch.no_grad()
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

        if clip_pixel_values is None:
            raise ValueError("clip_pixel_values 为必需：桥接固定使用 CLIP，请与 dino_pixel_values 一并传入。")

        merged_embeds, merged_mask, _, _ = self._build_inputs_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=None,
            dino_pixel_values=dino_pixel_values,
            clip_pixel_values=clip_pixel_values,
        )

        kwargs.pop("pixel_values", None)
        kwargs.pop("image_grid_thw", None)
        kwargs.pop("input_ids", None)
        kwargs.pop("attention_mask", None)
        kwargs.pop("dino_pixel_values", None)
        kwargs.pop("clip_pixel_values", None)
        kwargs.pop("mask_supervision", None)
        kwargs.pop("labels", None)

        return self.base_model.generate(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            **kwargs,
        )

    def save_bridge_weights(self, save_directory: str):
        write_dino_bridge_checkpoint(self, save_directory)

    def load_bridge(self, ckpt_path: str):
        ckpt_path = str(ckpt_path)
        if not os.path.isfile(ckpt_path):
            if _is_main_process():
                print(f"[bridge][warn] missing bridge checkpoint file: {ckpt_path}", flush=True)
            return

        payload = torch.load(ckpt_path, map_location="cpu")

        # 支持两种输入文件：
        # 1) dino_bridge.bin: {"state_dict": {...bridge...}}
        # 2) step1 的 epoch_*.pth / best_*.pth: {"model_state_dict": {...full...}, ...}
        if isinstance(payload, dict) and "state_dict" in payload:
            raw = payload["state_dict"]
        elif isinstance(payload, dict) and "model_state_dict" in payload:
            raw = payload["model_state_dict"]
        else:
            raw = payload

        if not isinstance(raw, dict):
            raise ValueError(f"Unsupported bridge checkpoint format: {type(raw)} ({ckpt_path})")

        # dino_bridge.bin 是「仅桥接层」权重（不含 base_model.*），必须整包加载；
        # 只有 step1 的 full checkpoint 才需要抽取 mapper（cls_mapper/patch_mapper）子集。
        is_bridge_only_bin = isinstance(payload, dict) and "state_dict" in payload

        if is_bridge_only_bin:
            state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in raw.items()}
        else:
            # 从 full state_dict 中抽取 mapper 权重（保持与 step1 命名一致）
            picked: Dict[str, torch.Tensor] = {}
            for k, v in raw.items():
                k2 = k[7:] if k.startswith("module.") else k
                if k2.startswith("cls_mapper.") or k2.startswith("patch_mapper."):
                    picked[k2] = v
            state_dict = picked if picked else {k[7:] if k.startswith("module.") else k: v for k, v in raw.items()}
        self.load_state_dict(state_dict, strict=False)

        if _is_main_process():
            print(f"[bridge] loaded: {ckpt_path}", flush=True)

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        # 默认强制 pytorch_model.bin（避免不同环境 safetensors 差异）
        if "safe_serialization" not in kwargs:
            kwargs["safe_serialization"] = False
        if "max_shard_size" not in kwargs:
            kwargs["max_shard_size"] = "2GB"
        try:
            self.base_model.save_pretrained(save_directory, **kwargs)
        except TypeError:
            # 兼容旧 transformers：不支持 safe_serialization/max_shard_size 等参数
            kwargs.pop("safe_serialization", None)
            kwargs.pop("max_shard_size", None)
            self.base_model.save_pretrained(save_directory, **kwargs)
        self.save_bridge_weights(save_directory)


def write_dino_bridge_checkpoint(model: nn.Module, save_directory: str) -> None:
    if not isinstance(model, QwenDinoBridgeModel):
        raise TypeError(f"需要 QwenDinoBridgeModel，实际收到: {type(model)}")
    os.makedirs(save_directory, exist_ok=True)
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("base_model.")}
    torch.save({"state_dict": state_dict}, os.path.join(save_directory, "dino_bridge.bin"))


def setup_model_and_processor(
    cfg: dict,
    for_inference: bool = False,
    model_name_override: Optional[str] = None,
) -> Tuple[nn.Module, Any]:
    model_name = model_name_override or cfg["model"]["name"]
    local_files_only = bool(cfg.get("model", {}).get("local_files_only", False))
    torch_dtype_cfg = cfg.get("model", {}).get("torch_dtype", "auto")

    if isinstance(torch_dtype_cfg, str) and torch_dtype_cfg != "auto":
        torch_dtype = getattr(torch, torch_dtype_cfg)
    else:
        torch_dtype = torch_dtype_cfg

    print(f"[setup] AutoTokenizer ← {model_name}")
    tok_kw: Dict[str, Any] = dict(
        trust_remote_code=True,
        use_fast=True,
        local_files_only=local_files_only,
    )
    try:
        processor = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True, **tok_kw)
    except Exception:
        processor = AutoTokenizer.from_pretrained(model_name, **tok_kw)

    if processor.pad_token is None and processor.eos_token is not None:
        processor.pad_token = processor.eos_token
        processor.pad_token_id = processor.eos_token_id

    model_cls = AutoModelForCausalLM
    # 训练/推理统一：始终从预训练/微调目录加载权重（不再支持随机初始化）
    if _is_main_process():
        mode_msg = "推理模式" if for_inference else "训练模式"
        print(f"[setup] {mode_msg}：AutoModelForCausalLM.from_pretrained ← {model_name}")
    base_model = model_cls.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=None if torch_dtype == "auto" else torch_dtype,
    )

    if torch_dtype is not None and torch_dtype != "auto":
        try:
            base_model = base_model.to(dtype=torch_dtype)
        except Exception:
            pass

    if _is_main_process():
        n_param = sum(p.numel() for p in base_model.parameters())
        print(f"[setup] 纯文本 Qwen 主干构建完成（权重已加载，参数量约 {n_param / 1e6:.1f}M）")

    model = QwenDinoBridgeModel(base_model, cfg)

    bridge_ckpt = cfg.get("model", {}).get("bridge_ckpt_path", None)
    if _is_main_process():
        print(f"[bridge] cfg.model.bridge_ckpt_path = {bridge_ckpt}", flush=True)
    if bridge_ckpt:
        model.load_bridge(str(bridge_ckpt))

    model.eval() if for_inference else model.train()
    return model, processor
