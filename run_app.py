#!/usr/bin/env python3
"""
启动 Flask Web 应用（与 main_qwen3.py 使用同一套 configs/qwen.yaml）。

用法:
  python run_app.py --model-path /path/to/logs/.../final_model
  python run_app.py --config configs/qwen.yaml   # yaml 里已填写 inference.model_path 时可省略 --model-path

环境变量（可选）:
  QWEN_WEB_CONFIG      配置文件绝对路径
  QWEN_WEB_MODEL_PATH  覆盖 inference.model_path
"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(project_root, "app")
# 与原先一致：先加入 app 目录使 `app.py` 可作为模块 `app` 导入；项目根目录供 utils/models 使用（app.py 内也会加入）
sys.path.insert(0, project_root)
sys.path.insert(0, app_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL 微调模型 Web 推理")
    default_cfg = os.path.join(project_root, "configs", "qwen.yaml")
    parser.add_argument("--config", type=str, default=default_cfg, help="YAML 配置（与训练相同）")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="微调输出目录（含 dino_bridge.bin 等），覆盖 yaml 中的 inference.model_path",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()

    os.environ["QWEN_WEB_CONFIG"] = os.path.abspath(args.config)
    if args.model_path:
        os.environ["QWEN_WEB_MODEL_PATH"] = os.path.abspath(args.model_path)

    from app import app

    print("=" * 60)
    print("Qwen3-VL 微调模型 Web 应用（与训练 / CLI 推理管线一致）")
    print("=" * 60)
    print(f"项目根目录: {project_root}")
    print(f"配置文件: {os.environ['QWEN_WEB_CONFIG']}")
    if args.model_path:
        print(f"模型路径: {os.environ['QWEN_WEB_MODEL_PATH']}")
    print("=" * 60)
    print(f"\n访问 http://localhost:{args.port} 使用 Web 界面")
    print("按 Ctrl+C 停止服务器\n")

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
