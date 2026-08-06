"""
AVNet：Anomaly / defect Vision + Language 大模型（DINO/CLIP 视觉前缀 + 因果 LM）。
"""
import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from utils.dinov3_utils import dinov3_encode_image
from models.visual_proto import (
    MultiLayerCLSProjection,
    MultiLayerPatchProjection,
    ResidualVisualProjection,
    infer_square_hw,
    mapped_patch_abnormal_prob_hw,
)

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


def _resolve_hw(num_tokens: int, hw: Tuple[int, int]) -> Tuple[int, int]:
    h, w = int(hw[0]), int(hw[1])
    if h > 0 and w > 0 and h * w == int(num_tokens):
        return h, w
    rh, rw = infer_square_hw(int(num_tokens))
    return int(rh), int(rw)


def _normalized_grid(hw: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    h, w = int(hw[0]), int(hw[1])
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(h * w, 2)


class Learned2DPosEmbedding(nn.Module):
    def __init__(self, hidden_size: int, max_h: int = 64, max_w: int = 64, dropout: float = 0.0):
        super().__init__()
        self.max_h = int(max_h)
        self.max_w = int(max_w)
        self.row_embed = nn.Parameter(torch.randn(self.max_h, hidden_size) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(self.max_w, hidden_size) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def _build_pos(self, hw: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        h, w = int(hw[0]), int(hw[1])
        if h <= self.max_h and w <= self.max_w:
            pos = self.row_embed[:h].unsqueeze(1) + self.col_embed[:w].unsqueeze(0)
            return pos.reshape(1, h * w, -1).to(device=device, dtype=dtype)
        base = self.row_embed.unsqueeze(1) + self.col_embed.unsqueeze(0)  # [max_h, max_w, D]
        base = base.permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]
        resized = F.interpolate(base, size=(h, w), mode="bilinear", align_corners=False)
        return resized.squeeze(0).permute(1, 2, 0).reshape(1, h * w, -1).to(device=device, dtype=dtype)

    def forward(self, tokens: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = _resolve_hw(int(tokens.shape[1]), hw)
        pos = self._build_pos((h, w), tokens.device, tokens.dtype)
        return self.dropout(tokens + pos)


class RoPELikeHeteroCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope_like_bias: bool = True,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} 必须能被 num_heads={num_heads} 整除")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope_like_bias = bool(use_rope_like_bias)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        half_dim = max(2, self.head_dim // 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.float32) / float(half_dim)))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        self.rope_bias_scale = nn.Parameter(torch.tensor(1.0))

    def _rope_like_bias(
        self,
        query_hw: Tuple[int, int],
        kv_hw: Tuple[int, int],
        q_len: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        qh, qw = _resolve_hw(q_len, query_hw)
        kh, kw = _resolve_hw(kv_len, kv_hw)
        q_xy = _normalized_grid((qh, qw), device=device, dtype=dtype)  # [Q,2]
        k_xy = _normalized_grid((kh, kw), device=device, dtype=dtype)  # [K,2]
        inv = self.rope_inv_freq.to(device=device, dtype=dtype)
        qx = q_xy[:, 0:1] * inv.view(1, -1) * math.pi
        qy = q_xy[:, 1:2] * inv.view(1, -1) * math.pi
        kx = k_xy[:, 0:1] * inv.view(1, -1) * math.pi
        ky = k_xy[:, 1:2] * inv.view(1, -1) * math.pi
        q_feat = torch.cat([torch.sin(qx), torch.cos(qx), torch.sin(qy), torch.cos(qy)], dim=-1)
        k_feat = torch.cat([torch.sin(kx), torch.cos(kx), torch.sin(ky), torch.cos(ky)], dim=-1)
        denom = float(q_feat.shape[-1]) ** 0.5
        bias = torch.matmul(q_feat, k_feat.transpose(-1, -2)) / denom
        return self.rope_bias_scale.to(device=device, dtype=dtype) * bias

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_hw: Tuple[int, int],
        kv_hw: Tuple[int, int],
    ) -> torch.Tensor:
        bsz, q_len, _ = query.shape
        kv_len = int(key_value.shape[1])
        q = self.q_proj(query).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if self.use_rope_like_bias:
            bias = self._rope_like_bias(
                query_hw=query_hw,
                kv_hw=kv_hw,
                q_len=q_len,
                kv_len=kv_len,
                device=query.device,
                dtype=query.dtype,
            )
            attn_logits = attn_logits + bias.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        out = self.proj_dropout(self.out_proj(out))
        return out


class HeteroCrossAlignBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope_like_bias: bool = True,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()
        self.residual_scale_init = float(residual_scale_init)
        self.dino_norm = nn.LayerNorm(hidden_size)
        self.clip_norm = nn.LayerNorm(hidden_size)
        self.mapped_norm = nn.LayerNorm(hidden_size)
        self.dino_from_clip = RoPELikeHeteroCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            use_rope_like_bias=use_rope_like_bias,
        )
        self.dino_from_mapped = RoPELikeHeteroCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            use_rope_like_bias=use_rope_like_bias,
        )
        self.clip_from_dino = RoPELikeHeteroCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            use_rope_like_bias=use_rope_like_bias,
        )
        self.clip_from_mapped = RoPELikeHeteroCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            use_rope_like_bias=use_rope_like_bias,
        )
        self.mapped_from_dino = RoPELikeHeteroCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            use_rope_like_bias=use_rope_like_bias,
        )
        self.mapped_from_clip = RoPELikeHeteroCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            use_rope_like_bias=use_rope_like_bias,
        )
        self.scale_dino_from_clip = nn.Parameter(torch.tensor(self.residual_scale_init, dtype=torch.float32))
        self.scale_dino_from_mapped = nn.Parameter(torch.tensor(self.residual_scale_init, dtype=torch.float32))
        self.scale_clip_from_dino = nn.Parameter(torch.tensor(self.residual_scale_init, dtype=torch.float32))
        self.scale_clip_from_mapped = nn.Parameter(torch.tensor(self.residual_scale_init, dtype=torch.float32))
        self.scale_mapped_from_dino = nn.Parameter(torch.tensor(self.residual_scale_init, dtype=torch.float32))
        self.scale_mapped_from_clip = nn.Parameter(torch.tensor(self.residual_scale_init, dtype=torch.float32))
        self.dino_ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.clip_ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.mapped_ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(
        self,
        dino_tokens: torch.Tensor,
        mapped_tokens: torch.Tensor,
        clip_tokens: torch.Tensor,
        dino_hw: Tuple[int, int],
        clip_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dino_norm = self.dino_norm(dino_tokens)
        mapped_norm = self.mapped_norm(mapped_tokens)
        clip_norm = self.clip_norm(clip_tokens)

        dino_delta_clip = self.dino_from_clip(
            query=dino_norm,
            key_value=clip_norm,
            query_hw=dino_hw,
            kv_hw=clip_hw,
        )
        dino_delta_mapped = self.dino_from_mapped(
            query=dino_norm,
            key_value=mapped_norm,
            query_hw=dino_hw,
            kv_hw=dino_hw,
        )
        clip_delta_dino = self.clip_from_dino(
            query=clip_norm,
            key_value=dino_norm,
            query_hw=clip_hw,
            kv_hw=dino_hw,
        )
        clip_delta_mapped = self.clip_from_mapped(
            query=clip_norm,
            key_value=mapped_norm,
            query_hw=clip_hw,
            kv_hw=dino_hw,
        )
        mapped_delta_dino = self.mapped_from_dino(
            query=mapped_norm,
            key_value=dino_norm,
            query_hw=dino_hw,
            kv_hw=dino_hw,
        )
        mapped_delta_clip = self.mapped_from_clip(
            query=mapped_norm,
            key_value=clip_norm,
            query_hw=dino_hw,
            kv_hw=clip_hw,
        )

        dino_tokens = dino_tokens + self.scale_dino_from_clip.to(dtype=dino_tokens.dtype) * dino_delta_clip
        dino_tokens = dino_tokens + self.scale_dino_from_mapped.to(dtype=dino_tokens.dtype) * dino_delta_mapped
        clip_tokens = clip_tokens + self.scale_clip_from_dino.to(dtype=clip_tokens.dtype) * clip_delta_dino
        clip_tokens = clip_tokens + self.scale_clip_from_mapped.to(dtype=clip_tokens.dtype) * clip_delta_mapped
        mapped_tokens = mapped_tokens + self.scale_mapped_from_dino.to(dtype=mapped_tokens.dtype) * mapped_delta_dino
        mapped_tokens = mapped_tokens + self.scale_mapped_from_clip.to(dtype=mapped_tokens.dtype) * mapped_delta_clip

        dino_tokens = dino_tokens + self.dino_ffn(dino_tokens)
        clip_tokens = clip_tokens + self.clip_ffn(clip_tokens)
        mapped_tokens = mapped_tokens + self.mapped_ffn(mapped_tokens)
        return dino_tokens, mapped_tokens, clip_tokens


class QwenDinoBridgeModel(nn.Module):
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

        # =========================
        # 7. 异构网格位置编码 + 多层跨路对齐（CoME-VL inspired）
        # =========================
        self.use_2d_pos_embed = bool(bridge_cfg.get("use_2d_pos_embed", True))
        self.enable_hetero_align = bool(bridge_cfg.get("enable_hetero_align", True))
        self.align_num_layers = int(bridge_cfg.get("align_num_layers", 3))
        self.align_num_heads = int(bridge_cfg.get("align_num_heads", 8))
        self.align_dropout = float(bridge_cfg.get("align_dropout", 0.0))
        self.use_rope_like_bias = bool(bridge_cfg.get("use_rope_like_bias", True))
        self.align_residual_scale = float(bridge_cfg.get("align_residual_scale", 0.1))
        max_2d_pos_h = int(bridge_cfg.get("max_2d_pos_h", 64))
        max_2d_pos_w = int(bridge_cfg.get("max_2d_pos_w", 64))
        self.dino_pos_embed = Learned2DPosEmbedding(
            hidden_size=llm_hidden,
            max_h=max_2d_pos_h,
            max_w=max_2d_pos_w,
            dropout=self.align_dropout,
        )
        self.mapped_pos_embed = Learned2DPosEmbedding(
            hidden_size=llm_hidden,
            max_h=max_2d_pos_h,
            max_w=max_2d_pos_w,
            dropout=self.align_dropout,
        )
        self.clip_pos_embed = Learned2DPosEmbedding(
            hidden_size=llm_hidden,
            max_h=max_2d_pos_h,
            max_w=max_2d_pos_w,
            dropout=self.align_dropout,
        )
        self.hetero_align_blocks = nn.ModuleList(
            [
                HeteroCrossAlignBlock(
                    hidden_size=llm_hidden,
                    num_heads=self.align_num_heads,
                    dropout=self.align_dropout,
                    use_rope_like_bias=self.use_rope_like_bias,
                    residual_scale_init=self.align_residual_scale,
                )
                for _ in range(self.align_num_layers)
            ]
        )

        # =========================
        # 8. 拼接后共享融合 MLP + 软压缩器
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

        # Step1 视觉原型：buffer 供热力图等；可选在 forward 中按异常概率调制 mapped 再进 LLM（gamma=0 关闭）
        st1 = cfg.get("step1", {}) or {}
        self.proto_modulation_gamma = float(bridge_cfg.get("proto_modulation_gamma", 0.0))
        self.prototype_temperature = float(
            bridge_cfg.get("prototype_temperature", st1.get("temperature", 0.07))
        )
        self.register_buffer("proto_normal", torch.zeros(self.clip_hidden))
        self.register_buffer("proto_abnormal", torch.zeros(self.clip_hidden))
        self.register_buffer("_proto_loaded", torch.tensor(0.0))

        tr = cfg.get("training", {}) or {}
        self.debug_step1_visual = bool(tr.get("step1_visual_debug", False))

        # =========================
        # 9. 训练策略
        # =========================
        for p in self.base_model.parameters():
            p.requires_grad = True
        for p in self.cls_mapper.parameters():
            p.requires_grad = False
        for p in self.patch_mapper.parameters():
            p.requires_grad = False

        # Step-3 可选：对视觉前缀池化后回归 0–1 bbox，与 GT 做 Smooth L1（训练 cfg.training.bbox_aux_loss_weight）
        self.bbox_aux_loss_weight = float(tr.get("bbox_aux_loss_weight", 0.0))
        if self.bbox_aux_loss_weight > 0.0:
            self.bbox_head = nn.Sequential(
                nn.LayerNorm(llm_hidden),
                nn.Linear(llm_hidden, llm_hidden),
                nn.GELU(),
                nn.Linear(llm_hidden, 4),
            )
        else:
            self.bbox_head = None

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

    def visual_prototypes_ready(self) -> bool:
        return bool(self._proto_loaded.item() > 0.5)

    def _ingest_prototypes_dict(self, p: Any) -> None:
        if not isinstance(p, dict):
            return
        n = p.get("normal")
        a = p.get("abnormal")
        if n is None or a is None:
            return
        n = n.detach().reshape(-1).float()
        a = a.detach().reshape(-1).float()
        if int(n.numel()) != self.clip_hidden or int(a.numel()) != self.clip_hidden:
            if _is_main_process():
                print(
                    f"[proto][warn] prototype dim {int(n.numel())}/{int(a.numel())} "
                    f"!= clip_hidden {self.clip_hidden}，跳过加载",
                    flush=True,
                )
            return
        with torch.no_grad():
            self.proto_normal.copy_(n.to(device=self.proto_normal.device, dtype=self.proto_normal.dtype))
            self.proto_abnormal.copy_(a.to(device=self.proto_abnormal.device, dtype=self.proto_abnormal.dtype))
            self._proto_loaded.fill_(1.0)
        if _is_main_process():
            print("[proto] 已加载 Step1 normal/abnormal 向量到 AVNet", flush=True)

    def load_step1_visual_prototypes(self, ckpt_path: str) -> None:
        ckpt_path = str(ckpt_path).strip()
        if not ckpt_path or ckpt_path.lower() in ("none", "null"):
            return
        if not os.path.isfile(ckpt_path):
            if _is_main_process():
                print(f"[proto][warn] 文件不存在: {ckpt_path}", flush=True)
            return
        try:
            payload = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            if _is_main_process():
                print(f"[proto][warn] torch.load 失败: {e}", flush=True)
            return
        if isinstance(payload, dict) and payload.get("prototypes"):
            self._ingest_prototypes_dict(payload["prototypes"])
        else:
            if _is_main_process():
                print(
                    "[proto][warn] checkpoint 无 'prototypes' 字段（需 Step1 的 epoch_*.pth / best_*.pth）",
                    flush=True,
                )

    def _encode_dino(self, dino_pixel_values: torch.Tensor, device: torch.device):
        """
        返回：
            cls_stack   : [L, B, D]
            patch_stack : [L, B, P, D]  (供 Stage-1 patch mapper 使用)
            dino_patches: [B, P, D]     (raw DINO patch，softmax 层融合)
            dino_hw     : (H, W)        DINO patch 网格高宽
        """
        x = dino_pixel_values.to(device=device, dtype=torch.float32)
        if x.ndim != 4:
            raise ValueError(f"dino_pixel_values 需要 [B,3,H,W]，当前 shape={tuple(x.shape)}")
        if x.shape[-2:] != (self.dino_image_size, self.dino_image_size):
            x = F.interpolate(x, size=(self.dino_image_size, self.dino_image_size), mode="bilinear", align_corners=False)
        dino_dev = next(self.dino_model.parameters()).device

        # 冻结视觉骨干：显式 no_grad 更稳、更省显存（避免内部实现构建不必要的计算图）
        with torch.no_grad():
            dino_out = dinov3_encode_image(
                x,
                processor=None,
                model=self.dino_model,
                device=device,
                layer_indices=self.dino_layer_indices,
            )

        mlf = dino_out["multi_layer_features"]  # list of [B, 1+P, D]
        cls_stack = torch.stack([feat[:, 0, :] for feat in mlf], dim=0)     # [L, B, D]
        patch_stack = torch.stack([feat[:, 1:, :] for feat in mlf], dim=0)  # [L, B, P, D]

        w = F.softmax(self.dino_raw_layer_logits[: patch_stack.shape[0]], dim=0).view(-1, 1, 1, 1)
        dino_patches = (patch_stack * w).sum(dim=0).contiguous()  # [B, P, D]
        gh, gw = dino_out["grid_size"].tolist()
        dino_hw = (int(gh), int(gw))
        return cls_stack, patch_stack, dino_patches, dino_hw

    def _encode_clip(self, clip_pixel_values: torch.Tensor, device: torch.device):
        """
        返回：
            clip_patches: [B, P, C]
            clip_hw:      (H, W)
        """
        x = clip_pixel_values.to(device=device, dtype=torch.float32)
        if x.ndim != 4:
            raise ValueError(f"clip_pixel_values 需要 [B,3,H,W]，当前 shape={tuple(x.shape)}")
        if x.shape[-2:] != (self.clip_image_size, self.clip_image_size):
            x = F.interpolate(x, size=(self.clip_image_size, self.clip_image_size), mode="bilinear", align_corners=False)
        clip_dev = next(self.clip_model.parameters()).device

        with torch.no_grad():
            vision_out = self.clip_model.vision_model(pixel_values=x)
            patch_tokens = vision_out.last_hidden_state[:, 1:, :]
            if hasattr(self.clip_model, "visual_projection"):
                patch_tokens = self.clip_model.visual_projection(patch_tokens)
            ph, pw = infer_square_hw(int(patch_tokens.shape[1]))
        
        return patch_tokens, (int(ph), int(pw))

    
    def _build_visual_tokens(
        self,
        dino_pixel_values: torch.Tensor,
        clip_pixel_values: torch.Tensor,
        device: torch.device,
    ):
        cls_stack, patch_stack, dino_patches, dino_hw = self._encode_dino(dino_pixel_values, device)
        clip_patches, clip_hw = self._encode_clip(clip_pixel_values, device)

        # Stage-1 mapper（冻结）
        _ = self.cls_mapper(cls_stack)  # 保留兼容；当前不直接作为 visual token 使用
        mapped_patches = self.patch_mapper(patch_stack)  # [B, P, C]

        if self.debug_step1_visual and _is_main_process():
            mp = mapped_patches.detach().float()
            print(
                "[avNet._build_visual_tokens] mapped_patches (patch_mapper 输出) "
                f"shape={tuple(mapped_patches.shape)} mean={mp.mean():.6f} std={mp.std():.6f} "
                f"min={mp.min():.6f} max={mp.max():.6f}",
                flush=True,
            )

        # 用 Step1 原型对每 patch 的「异常概率」调制 mapped 幅度，再进 LLM（与 utils.visualization 里 shell forward 热力图无关）
        if self.visual_prototypes_ready() and self.proto_modulation_gamma > 0.0:
            prob_hw = mapped_patch_abnormal_prob_hw(
                mapped_patches,
                dino_hw,
                self.proto_normal,
                self.proto_abnormal,
                self.prototype_temperature,
            )
            b, p, _ = mapped_patches.shape
            prob_flat = prob_hw.reshape(b, p, 1)
            scale = 1.0 + self.proto_modulation_gamma * (prob_flat * 2.0 - 1.0)
            mapped_patches = mapped_patches * scale.to(dtype=mapped_patches.dtype, device=mapped_patches.device)
            if self.debug_step1_visual and _is_main_process():
                sc = scale.detach().float()
                mp2 = mapped_patches.detach().float()
                print(
                    "[avNet._build_visual_tokens] proto 幅度调制已应用 "
                    f"gamma={self.proto_modulation_gamma} temp={self.prototype_temperature} | "
                    f"scale mean={sc.mean():.4f} std={sc.std():.4f} | "
                    f"mapped(after) mean={mp2.mean():.6f} std={mp2.std():.6f}",
                    flush=True,
                )
        elif self.debug_step1_visual and _is_main_process():
            print(
                "[avNet._build_visual_tokens] 未做 proto 调制（proto_ready="
                f"{self.visual_prototypes_ready()} gamma={self.proto_modulation_gamma}）",
                flush=True,
            )

        # 三路先各自 MLP 到 LLM hidden
        dino_patch_tokens = self.dino_patch_to_llm(dino_patches)
        mapped_patch_tokens = self.mapped_patch_to_llm(mapped_patches)
        clip_patch_tokens = self.clip_patch_to_llm(clip_patches)

        if self.use_2d_pos_embed:
            dino_patch_tokens = self.dino_pos_embed(dino_patch_tokens, dino_hw)
            mapped_patch_tokens = self.mapped_pos_embed(mapped_patch_tokens, dino_hw)
            clip_patch_tokens = self.clip_pos_embed(clip_patch_tokens, clip_hw)

        if self.enable_hetero_align and len(self.hetero_align_blocks) > 0:
            for blk in self.hetero_align_blocks:
                dino_patch_tokens, mapped_patch_tokens, clip_patch_tokens = blk(
                    dino_tokens=dino_patch_tokens,
                    mapped_tokens=mapped_patch_tokens,
                    clip_tokens=clip_patch_tokens,
                    dino_hw=dino_hw,
                    clip_hw=clip_hw,
                )

        all_patch_tokens = torch.cat(
            [dino_patch_tokens, mapped_patch_tokens, clip_patch_tokens],
            dim=1,
        )
        # all_patch_tokens = all_patch_tokens + self.post_concat_mlp(all_patch_tokens)
        # bridge_tokens = self.post_compress_norm(self.visual_compressor(all_patch_tokens))

        all_patch_tokens = all_patch_tokens + self.post_concat_mlp(all_patch_tokens)
        bridge_tokens = self.post_compress_norm(self.visual_compressor(all_patch_tokens))

        # 骨干输出通道维（进 projector 前）：便于日志核对配置是否一致
        dino_patch_dim = int(dino_patches.shape[-1])
        clip_patch_dim = int(clip_patches.shape[-1]) if int(clip_patches.shape[1]) > 0 else 0

        aux = {
            "bridge_num_tokens": bridge_tokens.shape[1],
            "num_dino_patch_tokens": dino_patch_tokens.shape[1],
            "num_mapped_patch_tokens": mapped_patch_tokens.shape[1],
            "num_clip_patch_tokens": clip_patch_tokens.shape[1],
            "raw_visual_tokens": all_patch_tokens.shape[1],
            "dino_patch_dim": dino_patch_dim,
            "clip_patch_dim": clip_patch_dim,
            "align_num_layers": self.align_num_layers if self.enable_hetero_align else 0,
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

        # 不显式构造 position_ids：与 generate() 一致，交给 HF 在 inputs_embeds + attention_mask 下自行推导，
        # 避免训练/推理两套位置规则（含 cache_position 等）不一致。
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
        bbox_target = kwargs.pop("bbox_target", None)
        bbox_loss_mask = kwargs.pop("bbox_loss_mask", None)

        if dino_pixel_values is None:
            # 兼容调用方误传多模态字段：纯文本 LLM 不接受这些 kwargs
            kwargs.pop("dino_pixel_values", None)
            kwargs.pop("clip_pixel_values", None)
            kwargs.pop("pixel_values", None)
            kwargs.pop("image_grid_thw", None)
            kwargs.pop("mask_supervision", None)
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
        kwargs.pop("position_ids", None)
        kwargs.pop("dino_pixel_values", None)
        kwargs.pop("clip_pixel_values", None)
        kwargs.pop("mask_supervision", None)

        want_bbox_aux = (
            self.training
            and self.bbox_head is not None
            and bbox_target is not None
            and bbox_loss_mask is not None
            and merged_labels is not None
        )
        outputs = self.base_model(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            labels=merged_labels,
            output_hidden_states=want_bbox_aux,
            **kwargs,
        )

        # 桥接统计每步更新（不依赖 loss 是否为 None，便于各类 on_log / 自定义 qwen_train 打印）
        self.last_loss_stats = {
            "bridge_tokens": float(aux["bridge_num_tokens"]),
            "raw_visual_tokens": float(aux["raw_visual_tokens"]),
            "num_dino_patch_tokens": float(aux["num_dino_patch_tokens"]),
            "num_mapped_patch_tokens": float(aux["num_mapped_patch_tokens"]),
            "num_clip_patch_tokens": float(aux["num_clip_patch_tokens"]),
            "dino_patch_dim": float(aux["dino_patch_dim"]),
            "clip_patch_dim": float(aux["clip_patch_dim"]),
            # 短键名，方便与 loss= / bridge_tok= 同一风格拼日志
            "dino_dim": float(aux["dino_patch_dim"]),
            "clip_dim": float(aux["clip_patch_dim"]),
        }
        if getattr(outputs, "loss", None) is not None:
            lm_loss = outputs.loss
            self.last_loss_stats["loss_lm"] = float(lm_loss.detach().float().item())
            total_loss = lm_loss

            if want_bbox_aux and outputs.hidden_states is not None:
                prefix_len = int(aux["bridge_num_tokens"])
                h = outputs.hidden_states[-1]
                vis_mask = merged_mask[:, :prefix_len].float().unsqueeze(-1)
                vis_h = h[:, :prefix_len, :]
                denom = vis_mask.sum(dim=1).clamp(min=1e-6)
                pooled = (vis_h * vis_mask).sum(dim=1) / denom
                head_dtype = next(self.bbox_head.parameters()).dtype
                pred = self.bbox_head(pooled.to(dtype=head_dtype))
                tgt = bbox_target.to(device=pred.device, dtype=pred.dtype)
                m = bbox_loss_mask.to(device=pred.device, dtype=pred.dtype).view(-1)
                per_ex = F.smooth_l1_loss(pred, tgt, reduction="none").mean(dim=-1)
                if float(m.sum().item()) > 0.0:
                    loss_bbox = (per_ex * m).sum() / m.sum().clamp(min=1e-6)
                else:
                    loss_bbox = pred.new_zeros(())
                total_loss = lm_loss + self.bbox_aux_loss_weight * loss_bbox
                outputs.loss = total_loss
                self.last_loss_stats["loss_bbox"] = float(loss_bbox.detach().float().item())

            self.last_loss_stats["loss_total"] = float(total_loss.detach().float().item())
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
            # 兼容调用方误传多模态字段：纯文本 LLM 不接受这些 kwargs
            kwargs.pop("dino_pixel_values", None)
            kwargs.pop("clip_pixel_values", None)
            kwargs.pop("pixel_values", None)
            kwargs.pop("image_grid_thw", None)
            kwargs.pop("mask_supervision", None)
            kwargs.pop("bbox_target", None)
            kwargs.pop("bbox_loss_mask", None)
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
        kwargs.pop("position_ids", None)
        kwargs.pop("dino_pixel_values", None)
        kwargs.pop("clip_pixel_values", None)
        kwargs.pop("mask_supervision", None)
        kwargs.pop("labels", None)
        kwargs.pop("bbox_target", None)
        kwargs.pop("bbox_loss_mask", None)
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

        if _is_main_process():
            size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
            print(f"[bridge] loading bridge checkpoint: {ckpt_path} ({size_mb:.1f} MB)", flush=True)

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

        # 只对 bridge 子集做匹配与加载，避免把 base_model.* 计入 missing
        model_sd = self.state_dict()
        bridge_model_keys = [k for k in model_sd.keys() if not k.startswith("base_model.")]
        bridge_model_key_set = set(bridge_model_keys)

        filtered_sd: Dict[str, torch.Tensor] = {}
        unexpected: list[str] = []
        shape_mismatch: list[str] = []

        for k, v in state_dict.items():
            if k not in bridge_model_key_set:
                unexpected.append(k)
                continue
            target = model_sd[k]
            if tuple(target.shape) != tuple(v.shape):
                shape_mismatch.append(f"{k}: ckpt{tuple(v.shape)} != model{tuple(target.shape)}")
                continue
            filtered_sd[k] = v

        with torch.no_grad():
            for k, v in filtered_sd.items():
                model_sd[k].copy_(v.to(device=model_sd[k].device, dtype=model_sd[k].dtype))

        missing_bridge = [k for k in bridge_model_keys if k not in filtered_sd]

        if _is_main_process():
            n_tensors = len(state_dict)
            print(
                f"[bridge] loaded: {ckpt_path} | tensors={n_tensors} "
                f"loaded={len(filtered_sd)} missing_bridge={len(missing_bridge)} "
                f"unexpected={len(unexpected)} mismatch={len(shape_mismatch)}",
                flush=True,
            )
            if len(unexpected) > 0:
                print(f"[bridge][warn] unexpected_keys (first 20): {unexpected[:20]}", flush=True)
            if len(shape_mismatch) > 0:
                print(f"[bridge][warn] shape_mismatch (first 20): {shape_mismatch[:20]}", flush=True)
            if len(missing_bridge) > 0 and len(missing_bridge) <= 20:
                print(
                    f"[bridge][warn] missing_bridge_keys (first {len(missing_bridge)}): {missing_bridge}",
                    flush=True,
                )
            elif len(missing_bridge) > 20:
                print(
                    f"[bridge][warn] missing_bridge_keys (first 20): {missing_bridge[:20]}",
                    flush=True,
                )

        if isinstance(payload, dict) and payload.get("prototypes"):
            self._ingest_prototypes_dict(payload["prototypes"])

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
    state_dict = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if not k.startswith("base_model.")
    }
    torch.save({"state_dict": state_dict}, os.path.join(save_directory, "dino_bridge.bin"))


def _torch_load_compat(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu")


def _load_raw_state_dict_from_folder(folder: str) -> Dict[str, Any]:
    """
    读取 HF Trainer / save_pretrained 目录下的权重字典（支持单文件 bin、分片 index、safetensors）。
    """
    folder = os.path.abspath(folder)
    out: Dict[str, Any] = {}

    bin_path = os.path.join(folder, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        blob = _torch_load_compat(bin_path)
        if isinstance(blob, dict) and "state_dict" in blob and isinstance(blob["state_dict"], dict):
            return dict(blob["state_dict"])
        if isinstance(blob, dict):
            return blob
        return {}

    idx_torch = os.path.join(folder, "pytorch_model.bin.index.json")
    if os.path.isfile(idx_torch):
        with open(idx_torch, "r", encoding="utf-8") as f:
            meta = json.load(f)
        wm = meta.get("weight_map", {})
        for shard in sorted(set(wm.values())):
            sp = os.path.join(folder, shard)
            if os.path.isfile(sp):
                chunk = _torch_load_compat(sp)
                if isinstance(chunk, dict):
                    out.update(chunk)
        return out

    st_path = os.path.join(folder, "model.safetensors")
    if os.path.isfile(st_path):
        try:
            from safetensors.torch import load_file

            return dict(load_file(st_path))
        except Exception:
            pass

    idx_st = os.path.join(folder, "model.safetensors.index.json")
    if os.path.isfile(idx_st):
        try:
            from safetensors.torch import load_file

            with open(idx_st, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for shard in sorted(set(meta.get("weight_map", {}).values())):
                sp = os.path.join(folder, shard)
                if os.path.isfile(sp):
                    out.update(load_file(sp))
            return out
        except Exception:
            pass

    return out


def _extract_base_model_state_dict(raw: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """QwenDinoBridgeModel 存盘键名为 base_model.*，需去掉前缀才能 load 进 AutoModelForCausalLM。"""
    out: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("base_model."):
            out[nk[len("base_model.") :]] = v
    return out


def _maybe_load_base_from_wrapped_trainer_checkpoint(base_model: nn.Module, folder: str) -> None:
    """
    Trainer 保存整个 QwenDinoBridgeModel 时，pytorch_model.bin 里是 base_model.model...；
    from_pretrained(Qwen3ForCausalLM) 不会消费这些键，导致 LM 仍是随机初始化。
    这里检测并剥前缀后注入 base_model。
    """
    if not folder or not os.path.isdir(folder):
        return
    raw = _load_raw_state_dict_from_folder(folder)
    if not raw:
        return
    stripped = _extract_base_model_state_dict(raw)
    if not stripped:
        return
    missing, unexpected = base_model.load_state_dict(stripped, strict=False)
    if _is_main_process():
        print(
            "[setup] 已从含 base_model. 前缀的 checkpoint 注入 Qwen 主干权重（Trainer 包装模型存盘格式）",
            flush=True,
        )
        if missing or unexpected:
            print(
                f"[setup] load_state_dict(strict=False): missing_keys={len(missing)} unexpected_keys={len(unexpected)}",
                flush=True,
            )


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
    model_cfg = cfg.get("model", {}) or {}
    # 训练可从零随机初始化主干；推理必须从 model_name（含微调 checkpoint 目录）加载权重，否则生成乱码。
    random_init_llm = bool(model_cfg.get("random_init_llm", False)) and not for_inference
    if for_inference and bool(model_cfg.get("random_init_llm", False)) and _is_main_process():
        print(
            "[setup] 推理模式：已忽略 model.random_init_llm，改为 from_pretrained 加载 LLM 权重",
            flush=True,
        )

    if random_init_llm:
        if _is_main_process():
            mode_msg = "训练模式"
            print(f"[setup] {mode_msg}：AutoConfig + from_config（LLM 随机初始化，不加载预训练权重）← {model_name}")
        llm_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        base_model = model_cls.from_config(llm_config)
    else:
        if _is_main_process():
            mode_msg = "推理模式" if for_inference else "训练模式"
            print(f"[setup] {mode_msg}：AutoModelForCausalLM.from_pretrained ← {model_name}", flush=True)
        base_model = model_cls.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
            torch_dtype=None if torch_dtype == "auto" else torch_dtype,
        )

    # Trainer 保存的 pytorch_model.bin 常为 QwenDinoBridgeModel 全量 state_dict（base_model.*），
    # CausalLM.from_pretrained 无法对齐这些键；剥前缀后补载，否则推理仍是随机 LM。
    _maybe_load_base_from_wrapped_trainer_checkpoint(base_model, str(model_name))

    if torch_dtype is not None and torch_dtype != "auto":
        try:
            base_model = base_model.to(dtype=torch_dtype)
        except Exception:
            pass

    if _is_main_process():
        n_param = sum(p.numel() for p in base_model.parameters())
        load_msg = "随机初始化" if random_init_llm else "权重已加载"
        print(f"[setup] 纯文本 Qwen 主干构建完成（{load_msg}，参数量约 {n_param / 1e6:.1f}M）")

    model = QwenDinoBridgeModel(base_model, cfg)

    bridge_cfg_top = cfg.get("bridge", {}) or {}
    bridge_ckpt = model_cfg.get("bridge_ckpt_path") or bridge_cfg_top.get("bridge_ckpt_path")
    if _is_main_process():
        print(
            f"[bridge] loading from model.bridge_ckpt_path or bridge.bridge_ckpt_path → {bridge_ckpt}",
            flush=True,
        )
    if bridge_ckpt:
        model.load_bridge(str(bridge_ckpt))

    # 若 bridge 仅为 dino_bridge.bin（无 prototypes 字段），可单独指定 Step1 的 epoch/best.pth 以对齐 test 热力图
    proto_ckpt = model_cfg.get("step1_prototypes_ckpt_path") or bridge_cfg_top.get(
        "step1_prototypes_ckpt_path"
    )
    if proto_ckpt and str(proto_ckpt).strip() and str(proto_ckpt).strip().lower() not in ("none", "null"):
        if _is_main_process():
            print(f"[proto] cfg.model.step1_prototypes_ckpt_path = {proto_ckpt}", flush=True)
        model.load_step1_visual_prototypes(str(proto_ckpt).strip())

    model.eval() if for_inference else model.train()
    return model, processor