import os
import shutil
import socket
import subprocess
import time

import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import PrinterCallback

from data.load_mvtec_data import MVTecDataManager
from data.qwen_grounding_dataset import MVTecQwenGroundingDataset
from models.qwen3_modeling import setup_model_and_processor
from utils.qwen_common import prepare_output_dir, set_seed
from utils.qwen_logging import EnhancedLoggingCallback


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _auto_start_tensorboard(cfg: dict, output_dir: str) -> None:
    tb_cfg = cfg.get("tensorboard", {})
    if not bool(tb_cfg.get("auto_start", True)):
        return

    host = str(tb_cfg.get("host", "0.0.0.0"))
    port = int(tb_cfg.get("port", 5000))
    probe_host = "127.0.0.1" if host == "0.0.0.0" else host
    if _is_port_in_use(probe_host, port):
        print(f"[TensorBoard] 端口 {port} 已被占用，复用现有服务: http://127.0.0.1:{port}")
        return

    tb_bin = shutil.which("tensorboard")
    if tb_bin is None:
        print("[TensorBoard] 未找到 tensorboard 命令，跳过自动启动。")
        return

    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    cmd = [tb_bin, "--logdir", log_dir, "--host", host, "--port", str(port)]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f"[TensorBoard] 已自动启动 (pid={proc.pid}): http://127.0.0.1:{port}")
    except Exception as e:
        print(f"[TensorBoard] 自动启动失败: {e}")


def train_main(cfg: dict) -> None:
    is_world_process_zero = int(os.environ.get("RANK", "0")) == 0
    set_seed(cfg["training"]["seed"])

    output_dir = prepare_output_dir(
        base_dir=cfg["paths"]["output_dir"],
        run_name=cfg["runtime"]["run_name"],
        auto_create=cfg["runtime"]["auto_create_output_dir"],
    )
    cfg["paths"]["output_dir"] = output_dir

    manager = MVTecDataManager(
        dataset_root=cfg["paths"]["dataset_root"],
        conversation_json_path=cfg["paths"]["conversation_json_path"],
    )
    manager.load_all()

    model, processor = setup_model_and_processor(cfg, for_inference=False)
    train_dataset = MVTecQwenGroundingDataset(
        manager=manager,
        processor=processor,
        mode="train",
        max_length=cfg["training"]["max_length"],
        max_image_size=cfg["data"]["max_image_size"],
        factor=cfg["data"]["factor"],
        use_grounding_format=cfg["data"]["use_grounding_format"],
        dino_cfg=cfg.get("dino", {}),
        clip_cfg=cfg.get("clip", {}),
        local_files_only=cfg.get("model", {}).get("local_files_only", True),
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["training"]["num_epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        save_strategy="steps",
        save_total_limit=cfg["training"]["save_total_limit"],
        save_safetensors=False,
        fp16=cfg["training"]["fp16"],
        bf16=cfg["training"]["bf16"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        dataloader_num_workers=cfg["training"]["num_workers"],
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=2,
        report_to=[],
        remove_unused_columns=False,
        logging_dir=os.path.join(output_dir, "logs"),
        disable_tqdm=False,
        ddp_find_unused_parameters=cfg["distributed"]["ddp_find_unused_parameters"],
        local_rank=cfg["distributed"]["local_rank"],
        deepspeed=cfg["distributed"]["deepspeed"],
    )

    callback = EnhancedLoggingCallback(output_dir=output_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[callback],
    )
    trainer.remove_callback(PrinterCallback)

    if is_world_process_zero:
        _auto_start_tensorboard(cfg, output_dir)
        print(f"开始训练，输出目录: {output_dir}")
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.train(resume_from_checkpoint=cfg["training"]["resume_from_checkpoint"])

    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    if is_world_process_zero:
        processor.save_pretrained(final_model_path)

    mins = (time.time() - t0) / 60
    if is_world_process_zero:
        print(f"训练完成，耗时: {mins:.2f} 分钟")
        print(f"模型已保存: {final_model_path}")

