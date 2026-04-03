import torch
from typing import Dict, Tuple, Union


def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)


@torch.inference_mode()
def dinov3_encode_image(
    image: Union["PIL.Image.Image", torch.Tensor],
    processor,
    model,
    device: Union[str, torch.device] = "cuda",
    layer_indices: list = None,
) -> Dict[str, torch.Tensor]:
    """
    Run DINOv3 to extract CLS token (global feature) and patch features from multiple layers.

    Inputs:
      - image: PIL.Image or a preprocessed tensor of shape [B, 3, H, W]
      - processor: transformers.AutoImageProcessor for the chosen DINOv3
      - model: transformers.AutoModel for the chosen DINOv3
      - device: torch device
      - layer_indices: list of layer indices to extract features from (e.g., [6, 12, 18, 24])
                      If None, only returns the last layer

    Returns a dict with:
      - cls: [B, D] normalized global feature from CLS token (last layer)
      - patch_flat: [B, H*W, D] normalized patch features (last layer)
      - patch_grid: [B, H, W, D] normalized patch features in grid form (last layer)
      - grid_size: (H, W)
      - hidden_size: D
      - multi_layer_features: list of [B, 1+P, D] features from specified layers (if layer_indices provided)
    """
    device = torch.device(device)

    if isinstance(image, torch.Tensor):
        if image.ndim != 4:
            raise ValueError("Expect image tensor as [B, 3, H, W]")
        inputs = {"pixel_values": _to_device(image, device)}
    # else:
    #     batch = processor(images=image, return_tensors="pt")
    #     inputs = {"pixel_values": _to_device(batch.pixel_values, device)}

    model = model.to(device)
    
    # 如果需要多层特征，设置 output_hidden_states=True
    if layer_indices is not None:
        outputs = model(**inputs, output_hidden_states=True)
        all_hidden_states = outputs.hidden_states  # Tuple of [B, 1+R+P, D]
    else:
        outputs = model(**inputs)
        all_hidden_states = None

    last_hidden_states: torch.Tensor = outputs.last_hidden_state  # [B, 1 + R + P, D]
    batch_size, total_tokens, hidden_size = last_hidden_states.shape

    num_register_tokens = getattr(model.config, "num_register_tokens", 0)
    patch_size = getattr(model.config, "patch_size", 16)

    _, _, img_height, img_width = inputs["pixel_values"].shape
    num_patches_h, num_patches_w = img_height // patch_size, img_width // patch_size
    num_patches_flat = num_patches_h * num_patches_w

    expected_tokens = 1 + num_register_tokens + num_patches_flat
    if total_tokens != expected_tokens:
        raise RuntimeError(
            f"Unexpected token count: got {total_tokens}, expect {expected_tokens} = 1 + {num_register_tokens} + {num_patches_flat}"
        )

    # 处理最后一层的特征
    cls = last_hidden_states[:, 0, :]  # [B, D]
    patch_flat = last_hidden_states[:, 1 + num_register_tokens :, :]  # [B, P, D]

    # Normalize
    cls = cls / (cls.norm(dim=-1, keepdim=True) + 1e-6)
    patch_flat = patch_flat / (patch_flat.norm(dim=-1, keepdim=True) + 1e-6)

    patch_grid = patch_flat.view(batch_size, num_patches_h, num_patches_w, hidden_size)

    result = {
        "cls": cls,
        "patch_flat": patch_flat,
        "patch_grid": patch_grid,
        "grid_size": torch.tensor([num_patches_h, num_patches_w], device=patch_grid.device),
        "hidden_size": torch.tensor(hidden_size, device=patch_grid.device),
    }
    
    # 提取多层特征
    if layer_indices is not None and all_hidden_states is not None:
        multi_layer_features = []
        for layer_idx in layer_indices:
            if layer_idx < 0 or layer_idx >= len(all_hidden_states):
                raise ValueError(f"Layer index {layer_idx} out of range [0, {len(all_hidden_states)-1}]")
            
            # 获取指定层的隐藏状态
            layer_hidden = all_hidden_states[layer_idx]  # [B, 1+R+P, D]
            
            # 提取 CLS + patches (移除 register tokens)
            layer_cls = layer_hidden[:, 0:1, :]  # [B, 1, D]
            layer_patches = layer_hidden[:, 1 + num_register_tokens:, :]  # [B, P, D]
            
            # 拼接 CLS + patches
            layer_combined = torch.cat([layer_cls, layer_patches], dim=1)  # [B, 1+P, D]
            
            # Normalize
            layer_combined = layer_combined / (layer_combined.norm(dim=-1, keepdim=True) + 1e-6)
            
            multi_layer_features.append(layer_combined)
        
        result["multi_layer_features"] = multi_layer_features
    
    return result


