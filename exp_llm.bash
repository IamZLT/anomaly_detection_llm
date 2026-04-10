#!/bin/bash
# ============================================================================
# Qwen3-VL-8B MVTec 缺陷检测模型训练脚本
# ============================================================================

# ============================================================================
# 环境变量设置
# ============================================================================
# CUDA_VISIBLE_DEVICES: 指定使用的GPU编号（0,1表示使用GPU 0和1）
# HF_ENDPOINT: HuggingFace镜像地址（加速模型下载）
export CUDA_VISIBLE_DEVICES=1,2
export HF_ENDPOINT=https://hf-mirror.com

# ============================================================================
# 多GPU训练命令（推荐）
# ============================================================================
# torchrun: PyTorch分布式训练启动器
#   --nproc_per_node=2: 使用2个GPU进程（对应CUDA_VISIBLE_DEVICES中的2个GPU）
#   qwen3_vl_8b.py: 训练脚本
torchrun --nproc_per_node=2 main_qwen3.py \
    --config configs/qwen.yaml \
    --mode train


# CUDA_VISIBLE_DEVICES=1,2 python test_qwen3_vl.py
CUDA_VISIBLE_DEVICES=1  python run_app.py 
tensorboard --logdir ./logs/ --port 6009 --host 0.0.0.0


# ============================================================================
# 参数说明
# ============================================================================
# --mode train                          # 训练模式
# --dataset_root                        # MVTec数据集根目录
# --conversation_json_path              # 对话数据JSON文件路径
# --use_conversations                   # 启用对话数据
# --model_name                          # 预训练模型名称
# --use_grounding_format                # 启用Grounding格式（输出边界框坐标）
# --max_image_size 512                  # 图像最大边长（512可节省约75%内存）
# --factor 28                           # 图像尺寸必须是28的倍数（满足ViT patch要求）
# --use_lora                            # 启用LoRA微调
# --lora_r 8                            # LoRA rank（低秩矩阵的秩）
# --lora_alpha 32                       # LoRA alpha（缩放因子，通常为rank的4倍）
# --lora_dropout 0.05                   # LoRA dropout率
# --freeze_vit                          # 冻结视觉编码器（推荐）
# --batch_size 1                         # 每个GPU的batch size
# --gradient_accumulation_steps 16      # 梯度累积步数（有效batch = 1 × 2 × 16 = 32）
# --num_epochs 1                        # 训练轮数
# --learning_rate 1e-4                  # 学习率（Grounding任务推荐1e-4到5e-4）
# --weight_decay 0.01                   # 权重衰减（L2正则化系数）
# --warmup_ratio 0.03                   # Warmup比例（前3%的步数进行学习率预热）
# --max_length 2048                     # 最大序列长度（token数）
# --logging_steps 10                    # 每10步记录一次日志
# --save_steps 100                      # 每100步保存一次checkpoint
# --save_total_limit 3                  # 最多保留3个checkpoint
# --gradient_checkpointing              # 启用梯度检查点（节省约50%显存）
# --output_dir ./outputs                # 输出目录基础路径
# --run_name grounding_mvtec            # 任务名称（用于创建输出目录）
# --auto_create_output_dir              # 自动创建带时间戳的子目录

# ============================================================================
# 单GPU训练命令（备选方案）
# ============================================================================
# 如果只有1个GPU，使用以下命令：
# CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com \
# python main_qwen3.py \
#     --mode train \
#     --model_name Qwen/Qwen3-VL-8B-Instruct \
#     --dataset_root /data2/zlt/anomaly_detection_llm/datasets/mvtec_anomaly_detection \
#     --conversation_json_path /data2/zlt/anomaly_detection_llm/datasets/mvtec_zero_shot.json \
#     --use_grounding_format \
#     --freeze_vit \
#     --use_lora \
#     --lora_r 8 \
#     --lora_alpha 32 \
#     --learning_rate 1e-4 \
#     --max_image_size 512 \
#     --factor 28 \
#     --batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_epochs 1 \
#     --output_dir ./outputs \
#     --run_name grounding_mvtec

# ============================================================================
# 可选参数（如需使用，取消注释并添加到命令中）
# ============================================================================
# --deepspeed scripts/zero2.json        # DeepSpeed配置（进一步节省内存）
# --ddp_find_unused_parameters          # DDP模式下查找未使用的参数（LoRA训练时可能需要）
# --resume_from_checkpoint <path>       # 从指定checkpoint恢复训练

# ============================================================================
# 查看训练日志（TensorBoard）
# ============================================================================
# 训练开始后，在另一个终端运行：
# tensorboard --logdir ./outputs --port 6005 --host 0.0.0.0
# 
# 然后在浏览器访问：http://your-server-ip:6005
# 
# 日志位置：
# - TensorBoard日志：./outputs/grounding_mvtec_YYYYMMDD_HHMMSS/logs/
# - Checkpoint：./outputs/grounding_mvtec_YYYYMMDD_HHMMSS/checkpoint-*/
# - 最终模型：./outputs/grounding_mvtec_YYYYMMDD_HHMMSS/final_model/

# ============================================================================
# 内存优化建议
# ============================================================================
# 如果遇到OOM（内存不足）错误，可以尝试：
# 1. 减小 --max_image_size（512 -> 384 或 256）
# 2. 减小 --batch_size（1 -> 1，同时增加 --gradient_accumulation_steps）
# 3. 使用DeepSpeed：--deepspeed scripts/zero2.json
# 4. 减小 --max_length（2048 -> 1024）
# 5. 设置环境变量：DISABLE_GRAD_NORM=true（禁用梯度范数计算）
